//! Experimental SQL Server staging sink. All native handles stay on one worker.
use crate::{
    dialect_for,
    native_diagnostics::{sql_server_connection_string, sql_server_manager},
};
use arrow_array::{
    Array, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
    UInt8Array,
};
use arrow_schema::{DataType, Schema, TimeUnit};
use async_trait::async_trait;
use chrono::{Datelike, Timelike};
use elm_core::{
    BatchCheckpoint, ConnectorCapabilities, ConsistencyMode, DataSink, DatabaseKind, ElmError,
    Environment, Identifier, JobId, PreflightReport, Relation, Result, WriteMode,
};
use odbc_api::{
    ColumnDescription, Connection, Cursor, ResultSetMetadata,
    buffers::{AnySlice, AnySliceMut, BufferDesc, ColumnarAnyBuffer},
};
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

enum Action {
    Preflight(Arc<Schema>, WriteMode, ConsistencyMode),
    Begin,
    Write(RecordBatch),
    Checkpoint(BatchCheckpoint),
    Publish,
    Abort,
}
enum Reply {
    Unit,
    Preflight(PreflightReport),
}
struct Request {
    action: Action,
    reply: oneshot::Sender<Result<Reply>>,
}
pub struct SqlServerSink {
    requests: mpsc::Sender<Request>,
}

impl SqlServerSink {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        target: Relation,
        job_id: JobId,
        checkpoint: Option<&BatchCheckpoint>,
    ) -> Result<Self> {
        let connection = sql_server_connection_string(environment, password)?;
        let plan = StagingPlan::new(target, job_id)?;
        let checkpoint = checkpoint.cloned();
        let (requests, mut receiver) = mpsc::channel::<Request>(1);
        let (ready, result) = oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let manager = match sql_server_manager() {
                Ok(manager) => manager,
                Err(error) => {
                    let _sent = ready.send(Err(error));
                    return;
                }
            };
            let connection = match manager.connect_with_connection_string(
                connection.expose_secret(),
                odbc_api::ConnectionOptions {
                    login_timeout_sec: Some(15),
                    ..Default::default()
                },
            ) {
                Ok(connection) => connection,
                Err(_) => {
                    let _sent = ready.send(Err(native_error()));
                    return;
                }
            };
            let mut state = SinkState {
                connection: &connection,
                plan,
                schema: None,
                mode: None,
                target_existed: false,
                rows: checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.rows_committed),
                sequence: checkpoint
                    .as_ref()
                    .map_or(Some(0), |checkpoint| checkpoint.sequence.checked_add(1)),
                checkpoint,
                pending: false,
                begun: false,
            };
            for setting in [
                "SET NOCOUNT ON",
                "SET ANSI_WARNINGS ON",
                "SET ARITHABORT ON",
                "SET XACT_ABORT ON",
            ] {
                if let Err(error) = execute(&connection, setting) {
                    let _sent = ready.send(Err(error));
                    return;
                }
            }
            if ready.send(Ok(())).is_err() {
                return;
            }
            while let Some(request) = receiver.blocking_recv() {
                let result = state.handle(request.action);
                if result.is_err() {
                    let _rollback = connection.rollback();
                    let _auto = connection.set_autocommit(true);
                }
                if request.reply.send(result).is_err() {
                    break;
                }
            }
            let _rollback = connection.rollback();
        });
        result.await.map_err(|_| native_error())??;
        Ok(Self { requests })
    }
    async fn request(&mut self, action: Action) -> Result<Reply> {
        let (reply, result) = oneshot::channel();
        self.requests
            .send(Request { action, reply })
            .await
            .map_err(|_| native_error())?;
        result.await.map_err(|_| native_error())?
    }
    async fn unit(&mut self, action: Action) -> Result<()> {
        match self.request(action).await? {
            Reply::Unit => Ok(()),
            _ => Err(ElmError::Internal(
                "unexpected SQL Server worker reply".into(),
            )),
        }
    }
}

#[async_trait]
impl DataSink for SqlServerSink {
    async fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        consistency: ConsistencyMode,
    ) -> Result<PreflightReport> {
        match self
            .request(Action::Preflight(schema, mode, consistency))
            .await?
        {
            Reply::Preflight(report) => Ok(report),
            _ => Err(ElmError::Internal(
                "missing SQL Server preflight report".into(),
            )),
        }
    }
    async fn begin(&mut self) -> Result<()> {
        self.unit(Action::Begin).await
    }
    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.unit(Action::Write(batch.clone())).await
    }
    async fn commit_checkpoint(&mut self, checkpoint: &BatchCheckpoint) -> Result<()> {
        self.unit(Action::Checkpoint(checkpoint.clone())).await
    }
    async fn publish(&mut self) -> Result<()> {
        self.unit(Action::Publish).await
    }
    async fn abort(&mut self) -> Result<()> {
        self.unit(Action::Abort).await
    }
}

#[derive(Clone)]
struct StagingPlan {
    target: Relation,
    stage: Relation,
    receipt: Relation,
    batch_column: Identifier,
    job_id: JobId,
}
impl StagingPlan {
    fn new(mut target: Relation, job_id: JobId) -> Result<Self> {
        if target.catalog.is_some() {
            return Err(ElmError::Unsupported(
                "SQL Server sinks must be in the connection's database; omit catalog".into(),
            ));
        }
        if target.schema.is_none() {
            target.schema = Some(Identifier::new("dbo")?);
        }
        let stage = Relation {
            catalog: None,
            schema: target.schema.clone(),
            name: Identifier::new(format!("elm_stage_{}", job_id.0.simple()))?,
        };
        let receipt = Relation {
            name: Identifier::new(format!("elm_receipt_{}", job_id.0.simple()))?,
            ..stage.clone()
        };
        // A target aliasing a private resource could expose the load before publish,
        // or cause recovery cleanup to delete the published target.
        if target
            .name
            .as_str()
            .eq_ignore_ascii_case(stage.name.as_str())
            || target
                .name
                .as_str()
                .eq_ignore_ascii_case(receipt.name.as_str())
        {
            return Err(ElmError::Conflict(
                "SQL Server destination collides with a private staging resource".into(),
            ));
        }
        Ok(Self {
            target,
            stage,
            receipt,
            batch_column: Identifier::new(format!("elm_batch_{}", job_id.0.simple()))?,
            job_id,
        })
    }
    fn mark_sql(&self, relation: &Relation) -> String {
        format!(
            "EXEC sys.sp_addextendedproperty @name=N'elm_job', @value={}, @level0type=N'SCHEMA', @level0name={}, @level1type=N'TABLE', @level1name={}",
            literal(&self.job_id.to_string()),
            literal(relation.schema.as_ref().map_or("dbo", Identifier::as_str)),
            literal(relation.name.as_str())
        )
    }
    fn is_owned(&self, connection: &Connection<'_>, relation: &Relation) -> Result<bool> {
        Ok(scalar(
            connection,
            &format!(
                "SELECT COUNT_BIG(*) FROM sys.extended_properties WHERE major_id=OBJECT_ID({}) AND minor_id=0 AND name=N'elm_job' AND CONVERT(nvarchar(36),value)={}",
                literal(&quoted(relation)),
                literal(&self.job_id.to_string())
            ),
        )? == 1)
    }
}

struct SinkState<'c, 'e> {
    connection: &'c Connection<'e>,
    plan: StagingPlan,
    schema: Option<Arc<Schema>>,
    mode: Option<WriteMode>,
    target_existed: bool,
    checkpoint: Option<BatchCheckpoint>,
    rows: u64,
    sequence: Option<u64>,
    pending: bool,
    begun: bool,
}
impl SinkState<'_, '_> {
    fn handle(&mut self, action: Action) -> Result<Reply> {
        match action {
            Action::Preflight(schema, mode, consistency) => {
                return self
                    .preflight(schema, mode, consistency)
                    .map(Reply::Preflight);
            }
            Action::Begin => self.begin()?,
            Action::Write(batch) => self.write(&batch)?,
            Action::Checkpoint(checkpoint) => {
                if !self.pending
                    || Some(checkpoint.sequence) != self.sequence
                    || checkpoint.rows_committed != self.rows
                {
                    return Err(ElmError::State(
                        "SQL Server checkpoint does not match the committed staging batch".into(),
                    ));
                }
                self.sequence = checkpoint.sequence.checked_add(1);
                self.pending = false;
            }
            Action::Publish => self.publish()?,
            Action::Abort => {
                self.connection.rollback().map_err(|_| native_error())?;
            }
        }
        Ok(Reply::Unit)
    }
    fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        consistency: ConsistencyMode,
    ) -> Result<PreflightReport> {
        if self.schema.is_some() {
            return Err(ElmError::Conflict(
                "SQL Server sink was already preflighted".into(),
            ));
        }
        if consistency != ConsistencyMode::Atomic {
            return Err(ElmError::Unsupported(
                "SQL Server checkpointed publication is not implemented".into(),
            ));
        }
        let _ddl = create_stage_sql(&self.plan, &schema)?;
        // SQL Server identifier equality follows the database collation, which can
        // treat more than ASCII case as equivalent (for example character width).
        if scalar(
            self.connection,
            &format!(
                "SELECT CAST(CASE WHEN {}={} OR {}={} THEN 1 ELSE 0 END AS BIGINT)",
                literal(self.plan.target.name.as_str()),
                literal(self.plan.stage.name.as_str()),
                literal(self.plan.target.name.as_str()),
                literal(self.plan.receipt.name.as_str()),
            ),
        )? != 0
        {
            return Err(ElmError::Conflict(
                "SQL Server destination collides with a private staging resource under the database collation".into(),
            ));
        }
        let schema_name = self
            .plan
            .target
            .schema
            .as_ref()
            .map_or("dbo", Identifier::as_str);
        if scalar(
            self.connection,
            &format!(
                "SELECT CAST(COALESCE(HAS_PERMS_BY_NAME(DB_NAME(), N'DATABASE', N'CREATE TABLE'),0) * COALESCE(HAS_PERMS_BY_NAME({}, N'SCHEMA', N'ALTER'),0) AS BIGINT)",
                literal(schema_name)
            ),
        )? != 1
        {
            return Err(ElmError::PermissionDenied("SQL Server atomic staging requires CREATE TABLE and ALTER on the destination schema".into()));
        }
        self.target_existed = exists(self.connection, &self.plan.target)?;
        if mode == WriteMode::Fail && self.target_existed {
            return Err(ElmError::Conflict(
                "SQL Server destination already exists".into(),
            ));
        }
        if mode == WriteMode::Append && self.target_existed {
            self.validate_target(&schema)?;
        }
        self.schema = Some(schema);
        self.mode = Some(mode);
        let warnings = vec![
            "Alpha SQL Server destination: limited types; full privilege, platform, and performance acceptance is pending".into(),
        ];
        Ok(PreflightReport {
            capabilities: ConnectorCapabilities {
                database: Some(DatabaseKind::SqlServer),
                native_bulk_read: true,
                native_bulk_write: true,
                atomic_append: true,
                atomic_replace: true,
                non_atomic_replace: false,
                checkpointed_write: false,
                resumable_keyset_read: false,
                supports_lob_spill: false,
            },
            staging_relation: Some(self.plan.stage.clone()),
            warnings,
        })
    }
    fn validate_target(&self, expected: &Schema) -> Result<()> {
        if scalar(
            self.connection,
            &format!(
                "SELECT COUNT_BIG(*) FROM sys.indexes WHERE object_id=OBJECT_ID({}) AND ignore_dup_key=1",
                literal(&quoted(&self.plan.target))
            ),
        )? != 0
        {
            return Err(ElmError::Unsupported(
                "SQL Server APPEND does not support indexes that silently ignore duplicate keys"
                    .into(),
            ));
        }
        if scalar(
            self.connection,
            &format!(
                "SELECT COUNT_BIG(*) FROM sys.triggers WHERE parent_id=OBJECT_ID({}) AND is_disabled=0",
                literal(&quoted(&self.plan.target))
            ),
        )? != 0
        {
            return Err(ElmError::Unsupported(
                "SQL Server APPEND into a target with enabled triggers is not supported".into(),
            ));
        }
        self.validate_relation(&self.plan.target, expected)
    }
    fn validate_relation(&self, relation: &Relation, expected: &Schema) -> Result<()> {
        let mut cursor = self
            .connection
            .execute(
                &format!("SELECT TOP (0) * FROM {}", quoted(relation)),
                (),
                Some(15),
            )
            .map_err(|_| native_error())?
            .ok_or_else(native_error)?;
        if cursor.num_result_cols().map_err(|_| native_error())? as usize != expected.fields().len()
        {
            return Err(ElmError::TypeMapping(
                "SQL Server destination column count differs from input".into(),
            ));
        }
        for (index, field) in expected.fields().iter().enumerate() {
            let mut description = ColumnDescription::default();
            cursor
                .describe_col((index + 1) as u16, &mut description)
                .map_err(|_| native_error())?;
            validate_physical_type(field.data_type(), &description.data_type)?;
            let (actual, _) = crate::sql_server::map_column(&description)?;
            if actual.name() != field.name()
                || actual.data_type() != field.data_type()
                || actual.is_nullable() != field.is_nullable()
            {
                return Err(ElmError::TypeMapping(
                    "SQL Server destination schema does not exactly match input".into(),
                ));
            }
        }
        Ok(())
    }
    fn begin(&mut self) -> Result<()> {
        if self.begun {
            return Err(ElmError::Conflict("SQL Server sink already began".into()));
        }
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| ElmError::State("SQL Server sink was not preflighted".into()))?;
        if exists(self.connection, &self.plan.receipt)? {
            return Err(ElmError::Conflict("SQL Server publication receipt already exists; reconcile publication before replay".into()));
        }
        if exists(self.connection, &self.plan.target)? != self.target_existed {
            return Err(ElmError::Conflict(
                "SQL Server destination changed after preflight".into(),
            ));
        }
        self.connection
            .set_autocommit(false)
            .map_err(|_| native_error())?;
        if exists(self.connection, &self.plan.stage)? {
            if !self.plan.is_owned(self.connection, &self.plan.stage)? {
                return Err(ElmError::Conflict(
                    "Refusing to reuse unrecognized SQL Server staging table".into(),
                ));
            }
            if let Some(checkpoint) = &self.checkpoint {
                let mut fields = schema
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone())
                    .collect::<Vec<_>>();
                fields.push(arrow_schema::Field::new(
                    self.plan.batch_column.as_str(),
                    DataType::Int64,
                    false,
                ));
                self.validate_relation(&self.plan.stage, &Schema::new(fields))?;
                let sequence = i64::try_from(checkpoint.sequence).map_err(|_| {
                    ElmError::State("SQL Server batch sequence exceeds BIGINT".into())
                })?;
                self.connection
                    .execute(
                        &format!(
                            "DELETE FROM {} WHERE {} > ?",
                            quoted(&self.plan.stage),
                            quote_id(&self.plan.batch_column)
                        ),
                        (&sequence,),
                        Some(15),
                    )
                    .map_err(|_| native_error())?;
                if row_count(self.connection, &self.plan.stage)? != self.rows {
                    return Err(ElmError::Conflict(
                        "SQL Server staging row count does not match checkpoint".into(),
                    ));
                }
                self.finish_transaction()?;
                self.begun = true;
                return Ok(());
            }
            execute(
                self.connection,
                &format!("DROP TABLE {}", quoted(&self.plan.stage)),
            )?;
        } else if self.checkpoint.is_some() {
            return Err(ElmError::Conflict(
                "SQL Server checkpoint staging table is missing".into(),
            ));
        }
        execute(self.connection, &create_stage_sql(&self.plan, schema)?)?;
        execute(self.connection, &self.plan.mark_sql(&self.plan.stage))?;
        execute(
            self.connection,
            &format!(
                "CREATE INDEX {} ON {} ({})",
                quote_id(&self.plan.batch_column),
                quoted(&self.plan.stage),
                quote_id(&self.plan.batch_column)
            ),
        )?;
        self.finish_transaction()?;
        self.begun = true;
        Ok(())
    }
    fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if !self.begun || self.pending {
            return Err(ElmError::State(
                "SQL Server sink is not ready for a batch".into(),
            ));
        }
        let schema = self.schema.as_ref().ok_or_else(native_error)?;
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(ElmError::TypeMapping(
                "SQL Server batch schema changed".into(),
            ));
        }
        validate_batch(batch)?;
        let sequence = self
            .sequence
            .and_then(|sequence| i64::try_from(sequence).ok())
            .ok_or_else(|| ElmError::State("SQL Server batch sequence exceeds BIGINT".into()))?;
        let mut descriptions = schema
            .fields()
            .iter()
            .map(|field| sink_type(field.data_type()).map(|(_, desc)| desc))
            .collect::<Result<Vec<_>>>()?;
        descriptions.push(BufferDesc::I64 { nullable: false });
        let row_bytes = descriptions
            .iter()
            .try_fold(0usize, |sum, desc| sum.checked_add(desc.bytes_per_row()))
            .ok_or_else(native_error)?;
        if row_bytes > 2 * 1024 * 1024 {
            return Err(ElmError::TypeMapping(
                "SQL Server row exceeds the native insertion buffer budget".into(),
            ));
        }
        let capacity = (2 * 1024 * 1024 / row_bytes).min(4096);
        let columns = schema
            .fields()
            .iter()
            .map(|field| Identifier::new(field.name()).map(|id| quote_id(&id)))
            .collect::<Result<Vec<_>>>()?;
        let sql = format!(
            "INSERT INTO {} ({}, {}) VALUES ({})",
            quoted(&self.plan.stage),
            columns.join(", "),
            quote_id(&self.plan.batch_column),
            vec!["?"; descriptions.len()].join(", ")
        );
        let mut prepared = self.connection.prepare(&sql).map_err(|_| native_error())?;
        prepared
            .set_query_timeout_sec(15)
            .map_err(|_| native_error())?;
        let mut inserter = prepared
            .into_column_inserter(capacity, descriptions)
            .map_err(|_| native_error())?;
        self.connection
            .set_autocommit(false)
            .map_err(|_| native_error())?;
        for offset in (0..batch.num_rows()).step_by(capacity) {
            let rows = capacity.min(batch.num_rows() - offset);
            for column in 0..batch.num_columns() {
                fill_column(
                    inserter.column_mut(column),
                    batch.column(column).as_ref(),
                    offset,
                    rows,
                )?;
            }
            let AnySliceMut::I64(values) = inserter.column_mut(batch.num_columns()) else {
                return Err(native_error());
            };
            values[..rows].fill(sequence);
            inserter.set_num_rows(rows);
            inserter.execute().map_err(|_| native_error())?;
        }
        drop(inserter);
        let expected = self
            .rows
            .checked_add(batch.num_rows() as u64)
            .ok_or_else(native_error)?;
        let inserted = scalar_with_params(
            self.connection,
            &format!(
                "SELECT COUNT_BIG(*) FROM {} WHERE {} = ?",
                quoted(&self.plan.stage),
                quote_id(&self.plan.batch_column)
            ),
            (&sequence,),
        )?;
        if inserted != batch.num_rows() as i64 {
            return Err(ElmError::TypeMapping(
                "SQL Server bulk insert row count differs from input; staging batch rolled back"
                    .into(),
            ));
        }
        self.finish_transaction()?;
        self.rows = expected;
        self.pending = true;
        Ok(())
    }
    fn publish(&mut self) -> Result<()> {
        if !self.begun || self.pending {
            return Err(ElmError::State(
                "SQL Server staging is not ready for publication".into(),
            ));
        }
        if row_count(self.connection, &self.plan.stage)? != self.rows {
            return Err(ElmError::Conflict(
                "SQL Server staging row count changed before publication".into(),
            ));
        }
        if exists(self.connection, &self.plan.target)? != self.target_existed {
            return Err(ElmError::Conflict(
                "SQL Server destination changed before publication".into(),
            ));
        }
        let mode = self.mode.ok_or_else(native_error)?;
        let schema = self.schema.as_ref().ok_or_else(native_error)?;
        if mode == WriteMode::Append && self.target_existed {
            self.validate_target(schema)?;
        }
        self.connection
            .set_autocommit(false)
            .map_err(|_| native_error())?;
        execute(
            self.connection,
            &format!(
                "CREATE TABLE {} (receipt_id TINYINT PRIMARY KEY CHECK (receipt_id=1), published_rows BIGINT NOT NULL, target_name NVARCHAR(128) NOT NULL)",
                quoted(&self.plan.receipt)
            ),
        )?;
        execute(self.connection, &self.plan.mark_sql(&self.plan.receipt))?;
        let rows = i64::try_from(self.rows).map_err(|_| native_error())?;
        self.connection
            .execute(
                &format!(
                    "INSERT INTO {} VALUES (1, ?, {})",
                    quoted(&self.plan.receipt),
                    literal(self.plan.target.name.as_str())
                ),
                (&rows,),
                Some(15),
            )
            .map_err(|_| native_error())?;
        if mode == WriteMode::Append && self.target_existed {
            let previous_rows = scalar(
                self.connection,
                &format!(
                    "SELECT COUNT_BIG(*) FROM {} WITH (TABLOCKX, HOLDLOCK)",
                    quoted(&self.plan.target)
                ),
            )?;
            let columns = schema
                .fields()
                .iter()
                .map(|field| Identifier::new(field.name()).map(|id| quote_id(&id)))
                .collect::<Result<Vec<_>>>()?
                .join(", ");
            execute(
                self.connection,
                &format!(
                    "INSERT INTO {} ({columns}) SELECT {columns} FROM {}",
                    quoted(&self.plan.target),
                    quoted(&self.plan.stage)
                ),
            )?;
            let expected_rows = previous_rows.checked_add(rows).ok_or_else(native_error)?;
            if scalar(
                self.connection,
                &format!("SELECT COUNT_BIG(*) FROM {}", quoted(&self.plan.target)),
            )? != expected_rows
            {
                return Err(ElmError::TypeMapping("SQL Server publication row count differs from staging; publication rolled back".into()));
            }
            execute(
                self.connection,
                &format!("DROP TABLE {}", quoted(&self.plan.stage)),
            )?;
        } else {
            if self.target_existed {
                execute(
                    self.connection,
                    &format!("DROP TABLE {}", quoted(&self.plan.target)),
                )?;
            }
            execute(
                self.connection,
                &format!(
                    "DROP INDEX {} ON {}",
                    quote_id(&self.plan.batch_column),
                    quoted(&self.plan.stage)
                ),
            )?;
            execute(
                self.connection,
                &format!(
                    "ALTER TABLE {} DROP COLUMN {}",
                    quoted(&self.plan.stage),
                    quote_id(&self.plan.batch_column)
                ),
            )?;
            execute(
                self.connection,
                &format!(
                    "EXEC sys.sp_rename @objname={}, @newname={}, @objtype=N'OBJECT'",
                    literal(&quoted(&self.plan.stage)),
                    literal(self.plan.target.name.as_str())
                ),
            )?;
        }
        self.finish_transaction()?;
        self.begun = false;
        Ok(())
    }
    fn finish_transaction(&self) -> Result<()> {
        self.connection.commit().map_err(|_| native_error())?;
        self.connection
            .set_autocommit(true)
            .map_err(|_| native_error())
    }
}

fn native_error() -> ElmError {
    ElmError::Connection { message: "SQL Server staging operation failed or its commit could not be confirmed; inspect the job receipt before retrying".into(), retryable: false }
}
fn literal(value: &str) -> String {
    format!("N'{}'", value.replace('\'', "''"))
}
fn quote_id(id: &Identifier) -> String {
    dialect_for(DatabaseKind::SqlServer).quote_identifier(id)
}
fn quoted(relation: &Relation) -> String {
    dialect_for(DatabaseKind::SqlServer).quote_relation(relation)
}
fn execute(connection: &Connection<'_>, sql: &str) -> Result<()> {
    connection
        .execute(sql, (), Some(15))
        .map_err(|_| native_error())?;
    Ok(())
}
fn scalar(connection: &Connection<'_>, sql: &str) -> Result<i64> {
    scalar_with_params(connection, sql, ())
}
fn scalar_with_params(
    connection: &Connection<'_>,
    sql: &str,
    params: impl odbc_api::ParameterCollectionRef,
) -> Result<i64> {
    let cursor = connection
        .execute(sql, params, Some(15))
        .map_err(|_| native_error())?
        .ok_or_else(native_error)?;
    let mut bound = cursor
        .bind_buffer(ColumnarAnyBuffer::from_descs(
            1,
            [BufferDesc::I64 { nullable: true }],
        ))
        .map_err(|_| native_error())?;
    let batch = bound
        .fetch_with_truncation_check(true)
        .map_err(|_| native_error())?
        .ok_or_else(native_error)?;
    let AnySlice::NullableI64(mut values) = batch.column(0) else {
        return Err(native_error());
    };
    values.next().flatten().copied().ok_or_else(native_error)
}
fn exists(connection: &Connection<'_>, relation: &Relation) -> Result<bool> {
    Ok(scalar(
        connection,
        &format!(
            "SELECT CAST(CASE WHEN OBJECT_ID({}) IS NULL THEN 0 ELSE 1 END AS BIGINT)",
            literal(&quoted(relation))
        ),
    )? == 1)
}
fn row_count(connection: &Connection<'_>, relation: &Relation) -> Result<u64> {
    u64::try_from(scalar(
        connection,
        &format!("SELECT COUNT_BIG(*) FROM {}", quoted(relation)),
    )?)
    .map_err(|_| native_error())
}
fn sink_type(data_type: &DataType) -> Result<(String, BufferDesc)> {
    if let DataType::Decimal128(precision, scale) = data_type {
        validate_decimal_type(*precision, *scale)?;
        return Ok((
            format!("DECIMAL({precision},{scale})"),
            BufferDesc::Text { max_str_len: 41 },
        ));
    }
    let (sql, buffer) = match data_type {
        DataType::Boolean => ("BIT", BufferDesc::Bit { nullable: true }),
        DataType::UInt8 => ("TINYINT", BufferDesc::U8 { nullable: true }),
        DataType::Int16 => ("SMALLINT", BufferDesc::I16 { nullable: true }),
        DataType::Int32 => ("INT", BufferDesc::I32 { nullable: true }),
        DataType::Int64 => ("BIGINT", BufferDesc::I64 { nullable: true }),
        DataType::Float32 => ("REAL", BufferDesc::F32 { nullable: true }),
        DataType::Float64 => ("FLOAT(53)", BufferDesc::F64 { nullable: true }),
        DataType::Utf8 => ("NVARCHAR(4000)", BufferDesc::WText { max_str_len: 4000 }),
        DataType::Binary => ("VARBINARY(8000)", BufferDesc::Binary { length: 8000 }),
        DataType::Date32 => ("DATE", BufferDesc::Date { nullable: true }),
        DataType::Timestamp(TimeUnit::Microsecond, None) => ("DATETIME2(6)", BufferDesc::Timestamp { nullable: true }),
        _ => return Err(ElmError::TypeMapping("SQL Server sink currently supports boolean, UInt8, Int16/32/64, floats, Decimal128, Date32, timezone-free microsecond timestamps, bounded UTF-8 text, and binary; supply an explicit conversion for other types".into())),
    };
    Ok((sql.into(), buffer))
}
fn temporal_range_error() -> ElmError {
    ElmError::TypeMapping("SQL Server temporal values must be within years 0001..9999".into())
}
fn validate_physical_type(expected: &DataType, actual: &odbc_api::DataType) -> Result<()> {
    // Canonical Arrow types omit SQL width/encoding/precision details. Matching
    // only the canonical schema would permit lossy INSERT ... SELECT publication.
    let compatible = match expected {
        DataType::Timestamp(..) => *actual == (odbc_api::DataType::Timestamp { precision: 6 }),
        DataType::Utf8 => {
            matches!(actual, odbc_api::DataType::WVarchar { length: Some(length) } if length.get() == 4000)
        }
        DataType::Binary => {
            matches!(actual, odbc_api::DataType::Varbinary { length: Some(length) } if length.get() == 8000)
        }
        _ => true,
    };
    if !compatible {
        return Err(ElmError::TypeMapping("SQL Server destination physical type differs from staging: text requires NVARCHAR(4000), binary VARBINARY(8000), and timestamps DATETIME2(6); narrower, fixed-width, or non-Unicode targets could lose data".into()));
    }
    Ok(())
}
fn date_value(days: i32) -> Result<odbc_api::sys::Date> {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).ok_or_else(temporal_range_error)?;
    let value = epoch
        .checked_add_signed(chrono::TimeDelta::days(i64::from(days)))
        .filter(|value| (1..=9999).contains(&value.year()))
        .ok_or_else(temporal_range_error)?;
    Ok(odbc_api::sys::Date {
        year: value.year() as i16,
        month: value.month() as u16,
        day: value.day() as u16,
    })
}
fn timestamp_value(micros: i64) -> Result<odbc_api::sys::Timestamp> {
    // UTC is only used for epoch arithmetic; the Arrow type and SQL value have no timezone.
    let value = chrono::DateTime::from_timestamp_micros(micros)
        .filter(|value| (1..=9999).contains(&value.year()))
        .ok_or_else(temporal_range_error)?;
    Ok(odbc_api::sys::Timestamp {
        year: value.year() as i16,
        month: value.month() as u16,
        day: value.day() as u16,
        hour: value.hour() as u16,
        minute: value.minute() as u16,
        second: value.second() as u16,
        fraction: value.nanosecond(),
    })
}
fn validate_decimal_type(precision: u8, scale: i8) -> Result<()> {
    if !(1..=38).contains(&precision) || scale < 0 || scale as u8 > precision {
        return Err(ElmError::TypeMapping(
            "SQL Server decimals require precision 1..38 and scale 0..precision; supply an explicit conversion".into(),
        ));
    }
    Ok(())
}
fn validate_decimal_coefficient(coefficient: i128, precision: u8, scale: i8) -> Result<()> {
    validate_decimal_type(precision, scale)?;
    if coefficient.unsigned_abs() >= 10u128.pow(u32::from(precision)) {
        return Err(ElmError::TypeMapping(
            "SQL Server decimal coefficient exceeds its declared precision".into(),
        ));
    }
    Ok(())
}
fn decimal_text(coefficient: i128, precision: u8, scale: i8) -> Result<String> {
    validate_decimal_coefficient(coefficient, precision, scale)?;
    let scale = scale as usize;
    let mut text = format!("{:0width$}", coefficient.unsigned_abs(), width = scale + 1);
    if scale > 0 {
        text.insert(text.len() - scale, '.');
    }
    if coefficient < 0 {
        text.insert(0, '-');
    }
    Ok(text)
}
fn create_stage_sql(plan: &StagingPlan, schema: &Schema) -> Result<String> {
    if schema.fields().is_empty() {
        return Err(ElmError::TypeMapping(
            "SQL Server sink requires columns".into(),
        ));
    }
    let mut names = std::collections::BTreeSet::new();
    let mut columns = Vec::new();
    for field in schema.fields() {
        if !names.insert(field.name().to_lowercase()) {
            return Err(ElmError::TypeMapping(
                "SQL Server sink column names must be unique ignoring case".into(),
            ));
        }
        let id = Identifier::new(field.name())?;
        if id.as_str().eq_ignore_ascii_case(plan.batch_column.as_str()) {
            return Err(ElmError::TypeMapping(
                "SQL Server input collides with the private batch column".into(),
            ));
        }
        let (sql_type, _) = sink_type(field.data_type())?;
        columns.push(format!(
            "{} {sql_type} {}",
            quote_id(&id),
            if field.is_nullable() {
                "NULL"
            } else {
                "NOT NULL"
            }
        ));
    }
    columns.push(format!("{} BIGINT NOT NULL", quote_id(&plan.batch_column)));
    Ok(format!(
        "CREATE TABLE {} ({})",
        quoted(&plan.stage),
        columns.join(", ")
    ))
}
fn validate_batch(batch: &RecordBatch) -> Result<()> {
    for column in batch.columns() {
        if let Some(values) = column.as_any().downcast_ref::<Date32Array>() {
            for value in values.iter().flatten() {
                let _validated = date_value(value)?;
            }
        }
        if let Some(values) = column.as_any().downcast_ref::<TimestampMicrosecondArray>() {
            for value in values.iter().flatten() {
                let _validated = timestamp_value(value)?;
            }
        }
        if let Some(values) = column.as_any().downcast_ref::<Decimal128Array>() {
            for coefficient in values.iter().flatten() {
                validate_decimal_coefficient(coefficient, values.precision(), values.scale())?;
            }
        }
        if let Some(values) = column.as_any().downcast_ref::<Float32Array>()
            && values.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(ElmError::TypeMapping(
                "SQL Server cannot store NaN or infinite floats".into(),
            ));
        }
        if let Some(values) = column.as_any().downcast_ref::<Float64Array>()
            && values.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(ElmError::TypeMapping(
                "SQL Server cannot store NaN or infinite floats".into(),
            ));
        }
        if let Some(text) = column.as_any().downcast_ref::<StringArray>() {
            if text
                .iter()
                .flatten()
                .any(|value| value.encode_utf16().count() > 4000)
            {
                return Err(ElmError::TypeMapping(
                    "SQL Server text exceeds 4000 UTF-16 units; LOB sink support is pending".into(),
                ));
            }
        } else if let Some(binary) = column.as_any().downcast_ref::<BinaryArray>()
            && binary.iter().flatten().any(|value| value.len() > 8000)
        {
            return Err(ElmError::TypeMapping(
                "SQL Server binary value exceeds 8000 bytes; LOB sink support is pending".into(),
            ));
        }
    }
    Ok(())
}
fn fill_column(
    buffer: AnySliceMut<'_>,
    array: &dyn Array,
    offset: usize,
    rows: usize,
) -> Result<()> {
    macro_rules! fill {
        ($buffer:ident, $array:ty) => {{
            let array = array
                .as_any()
                .downcast_ref::<$array>()
                .ok_or_else(native_error)?;
            $buffer.write(array.iter().skip(offset).take(rows));
        }};
    }
    match buffer {
        AnySliceMut::NullableI16(mut buffer) => fill!(buffer, Int16Array),
        AnySliceMut::NullableI32(mut buffer) => fill!(buffer, Int32Array),
        AnySliceMut::NullableI64(mut buffer) => fill!(buffer, Int64Array),
        AnySliceMut::NullableU8(mut buffer) => fill!(buffer, UInt8Array),
        AnySliceMut::NullableF32(mut buffer) => fill!(buffer, Float32Array),
        AnySliceMut::NullableF64(mut buffer) => fill!(buffer, Float64Array),
        AnySliceMut::NullableDate(mut buffer) => {
            let array = array
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(native_error)?;
            for (index, value) in array.iter().skip(offset).take(rows).enumerate() {
                buffer.set_cell(index, value.map(date_value).transpose()?);
            }
        }
        AnySliceMut::NullableTimestamp(mut buffer) => {
            let array = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .ok_or_else(native_error)?;
            for (index, value) in array.iter().skip(offset).take(rows).enumerate() {
                buffer.set_cell(index, value.map(timestamp_value).transpose()?);
            }
        }
        AnySliceMut::Text(mut buffer) => {
            let array = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(native_error)?;
            for (index, value) in array.iter().skip(offset).take(rows).enumerate() {
                let encoded = value
                    .map(|value| decimal_text(value, array.precision(), array.scale()))
                    .transpose()?;
                buffer.set_cell(index, encoded.as_deref().map(str::as_bytes));
            }
        }
        AnySliceMut::NullableBit(mut buffer) => {
            let array = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(native_error)?;
            buffer.write(
                array
                    .iter()
                    .skip(offset)
                    .take(rows)
                    .map(|value| value.map(|value| odbc_api::Bit(u8::from(value)))),
            );
        }
        AnySliceMut::WText(mut buffer) => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(native_error)?;
            for (index, value) in array.iter().skip(offset).take(rows).enumerate() {
                let encoded = value.map(|value| value.encode_utf16().collect::<Vec<_>>());
                buffer.set_cell(index, encoded.as_deref());
            }
        }
        AnySliceMut::Binary(mut buffer) => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(native_error)?;
            for (index, value) in array.iter().skip(offset).take(rows).enumerate() {
                buffer.set_cell(index, value);
            }
        }
        _ => return Err(native_error()),
    }
    Ok(())
}

/// A publication receipt is committed in the same transaction as the target changes.
pub async fn sql_server_published_rows(
    environment: &Environment,
    password: &str,
    target: Relation,
    job_id: JobId,
) -> Result<Option<u64>> {
    let connection = sql_server_connection_string(environment, password)?;
    let plan = StagingPlan::new(target, job_id)?;
    tokio::task::spawn_blocking(move || {
        let manager = sql_server_manager()?;
        let connection = manager
            .connect_with_connection_string(
                connection.expose_secret(),
                odbc_api::ConnectionOptions {
                    login_timeout_sec: Some(15),
                    ..Default::default()
                },
            )
            .map_err(|_| native_error())?;
        if !exists(&connection, &plan.receipt)? {
            return Ok(None);
        }
        if !plan.is_owned(&connection, &plan.receipt)? {
            return Err(ElmError::Conflict(
                "Unrecognized SQL Server publication receipt".into(),
            ));
        }
        let rows = scalar(
            &connection,
            &format!("SELECT published_rows FROM {} WHERE receipt_id=1 AND target_name COLLATE Latin1_General_100_BIN2={}", quoted(&plan.receipt), literal(plan.target.name.as_str())),
        )?;
        Ok(Some(u64::try_from(rows).map_err(|_| native_error())?))
    })
    .await
    .map_err(|_| native_error())?
}

pub async fn cleanup_sql_server_staging(
    environment: &Environment,
    password: &str,
    target: Relation,
    job_id: JobId,
) -> Result<()> {
    let connection: SecretString = sql_server_connection_string(environment, password)?;
    let plan = StagingPlan::new(target, job_id)?;
    tokio::task::spawn_blocking(move || {
        let manager = sql_server_manager()?;
        let connection = manager
            .connect_with_connection_string(
                connection.expose_secret(),
                odbc_api::ConnectionOptions {
                    login_timeout_sec: Some(15),
                    ..Default::default()
                },
            )
            .map_err(|_| native_error())?;
        for resource in [&plan.stage, &plan.receipt] {
            if exists(&connection, resource)? && !plan.is_owned(&connection, resource)? {
                return Err(ElmError::Conflict(
                    "Refusing to remove unrecognized SQL Server staging resources".into(),
                ));
            }
        }
        connection
            .set_autocommit(false)
            .map_err(|_| native_error())?;
        let result = (|| {
            for resource in [&plan.stage, &plan.receipt] {
                if exists(&connection, resource)? {
                    execute(&connection, &format!("DROP TABLE {}", quoted(resource)))?;
                }
            }
            Ok(())
        })();
        if result.is_ok() {
            connection.commit().map_err(|_| native_error())?;
        } else {
            let _rollback = connection.rollback();
        }
        result
    })
    .await
    .map_err(|_| native_error())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;

    #[test]
    fn physical_schema_checks_reject_text_encoding_padding_and_precision_loss() {
        use odbc_api::DataType as Sql;
        use std::num::NonZeroUsize;
        let text_length = NonZeroUsize::new(4000);
        let binary_length = NonZeroUsize::new(8000);
        assert!(
            validate_physical_type(
                &DataType::Utf8,
                &Sql::WVarchar {
                    length: text_length
                }
            )
            .is_ok()
        );
        assert!(
            validate_physical_type(
                &DataType::Binary,
                &Sql::Varbinary {
                    length: binary_length
                }
            )
            .is_ok()
        );
        for actual in [
            Sql::Varchar {
                length: text_length,
            },
            Sql::WChar {
                length: text_length,
            },
            Sql::WVarchar {
                length: NonZeroUsize::new(20),
            },
        ] {
            assert!(validate_physical_type(&DataType::Utf8, &actual).is_err());
        }
        for actual in [
            Sql::Binary {
                length: binary_length,
            },
            Sql::Varbinary {
                length: NonZeroUsize::new(20),
            },
        ] {
            assert!(validate_physical_type(&DataType::Binary, &actual).is_err());
        }
        for precision in [0, 3, 7] {
            assert!(
                validate_physical_type(
                    &DataType::Timestamp(TimeUnit::Microsecond, None),
                    &Sql::Timestamp { precision }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn temporal_values_preserve_boundaries_and_reject_out_of_range_and_lossy_types() {
        for (days, expected) in [
            (-719_162, (1, 1, 1)),
            (0, (1970, 1, 1)),
            (2_932_896, (9999, 12, 31)),
        ] {
            let date = date_value(days).unwrap_or_else(|error| panic!("{error}"));
            assert_eq!((date.year, date.month, date.day), expected);
        }
        for days in [i32::MIN, -719_163, 2_932_897, i32::MAX] {
            assert!(date_value(days).is_err());
            assert!(
                validate_batch(&one_column(Arc::new(Date32Array::from(vec![
                    Some(days),
                    None
                ]))))
                .is_err()
            );
        }
        for micros in [
            -62_135_596_800_000_000,
            -1,
            0,
            1,
            1_709_210_096_123_456,
            253_402_300_799_999_999,
        ] {
            let timestamp = timestamp_value(micros).unwrap_or_else(|error| panic!("{error}"));
            assert!(timestamp.fraction.is_multiple_of(1000));
            assert_eq!(
                crate::sql_server::native_timestamp(&timestamp)
                    .unwrap_or_else(|error| panic!("{error}")),
                micros
            );
        }
        let before_epoch = timestamp_value(-1).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            (
                before_epoch.year,
                before_epoch.month,
                before_epoch.day,
                before_epoch.hour,
                before_epoch.minute,
                before_epoch.second,
                before_epoch.fraction
            ),
            (1969, 12, 31, 23, 59, 59, 999_999_000)
        );
        for micros in [
            i64::MIN,
            -62_135_596_800_000_001,
            253_402_300_800_000_000,
            i64::MAX,
        ] {
            assert!(timestamp_value(micros).is_err());
            assert!(
                validate_batch(&one_column(Arc::new(TimestampMicrosecondArray::from(
                    vec![Some(micros), None]
                ))))
                .is_err()
            );
        }
        for data_type in [
            DataType::Date64,
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Second, None),
        ] {
            assert!(sink_type(&data_type).is_err());
        }
        assert_eq!(
            sink_type(&DataType::Date32)
                .unwrap_or_else(|error| panic!("{error}"))
                .0,
            "DATE"
        );
        assert_eq!(
            sink_type(&DataType::Timestamp(TimeUnit::Microsecond, None))
                .unwrap_or_else(|error| panic!("{error}"))
                .0,
            "DATETIME2(6)"
        );
    }

    #[test]
    fn decimal_encoding_is_exact_and_rejects_invalid_coefficients() {
        let maximum = 10i128.pow(38) - 1;
        for (value, precision, scale, expected) in [
            (0, 1, 0, "0".to_owned()),
            (0, 38, 38, format!("0.{}", "0".repeat(38))),
            (-1, 38, 38, format!("-0.{}1", "0".repeat(37))),
            (-123_456_700, 38, 4, "-12345.6700".to_owned()),
            (maximum, 38, 0, "9".repeat(38)),
            (-maximum, 38, 1, format!("-{}.9", "9".repeat(37))),
            (maximum, 38, 38, format!("0.{}", "9".repeat(38))),
        ] {
            let actual =
                decimal_text(value, precision, scale).unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(actual, expected);
            assert!(actual.len() <= 41);
        }
        for (precision, scale) in [(0, 0), (39, 0), (38, -1), (2, 3)] {
            assert!(sink_type(&DataType::Decimal128(precision, scale)).is_err());
            assert!(decimal_text(0, precision, scale).is_err());
        }
        for coefficient in [1000, -1000, i128::MIN, i128::MAX] {
            assert!(decimal_text(coefficient, 3, 2).is_err());
        }
        // Arrow type metadata does not by itself validate every coefficient.
        let invalid = Decimal128Array::from(vec![Some(1000), None])
            .with_precision_and_scale(3, 2)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(validate_batch(&one_column(Arc::new(invalid))).is_err());
    }

    #[test]
    fn decimal_codec_round_trips_every_supported_precision_and_scale() {
        for precision in 1..=38 {
            let maximum = 10i128.pow(u32::from(precision)) - 1;
            for scale in 0..=precision as i8 {
                for value in [0, 1, -1, maximum, -maximum] {
                    let text = decimal_text(value, precision, scale)
                        .unwrap_or_else(|error| panic!("{error}"));
                    let decoded =
                        crate::sql_server::decimal_coefficient(text.as_bytes(), precision, scale)
                            .unwrap_or_else(|error| panic!("{error}"));
                    assert_eq!(decoded, value);
                    assert!(text.len() <= 41);
                }
            }
        }
    }

    #[test]
    fn target_cannot_alias_private_staging_or_receipt() {
        let job_id = JobId::new();
        for prefix in ["elm_stage_", "elm_receipt_", "ELM_STAGE_", "ELM_RECEIPT_"] {
            let target = Relation {
                catalog: None,
                schema: None,
                name: Identifier::new(format!("{prefix}{}", job_id.0.simple()))
                    .unwrap_or_else(|error| panic!("{error}")),
            };
            assert!(matches!(
                StagingPlan::new(target, job_id),
                Err(ElmError::Conflict(_))
            ));
        }
    }

    #[test]
    fn staging_ddl_quotes_names_and_rejects_ambiguous_or_unsupported_columns() {
        let plan = StagingPlan::new(
            Relation {
                catalog: None,
                schema: Some(Identifier::new("a'b]").unwrap_or_else(|error| panic!("{error}"))),
                name: Identifier::new("target").unwrap_or_else(|error| panic!("{error}")),
            },
            JobId::new(),
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let schema = Schema::new(vec![Field::new(
            "x]; DROP TABLE target--",
            DataType::Int64,
            false,
        )]);
        let ddl = create_stage_sql(&plan, &schema).unwrap_or_else(|error| panic!("{error}"));
        assert!(ddl.contains("[x]]; DROP TABLE target--] BIGINT NOT NULL"));
        assert!(ddl.contains("[a'b]]]."));
        assert!(plan.mark_sql(&plan.stage).contains("@level0name=N'a''b]'"));
        let duplicate = Schema::new(vec![
            Field::new("Name", DataType::Int32, true),
            Field::new("name", DataType::Int32, true),
        ]);
        assert!(create_stage_sql(&plan, &duplicate).is_err());
        assert!(sink_type(&DataType::UInt64).is_err());
        assert_eq!(
            sink_type(&DataType::Decimal128(38, 4))
                .unwrap_or_else(|error| panic!("{error}"))
                .0,
            "DECIMAL(38,4)"
        );
    }

    fn one_column(array: Arc<dyn Array>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                array.data_type().clone(),
                true,
            )])),
            vec![array],
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn insertion_limits_count_utf16_units_and_reject_nonfinite_values() {
        let fits = "🌍".repeat(2000);
        let oversized = "🌍".repeat(2001);
        assert!(
            validate_batch(&one_column(Arc::new(StringArray::from(vec![
                Some(fits.as_str()),
                None
            ]))))
            .is_ok()
        );
        assert!(
            validate_batch(&one_column(Arc::new(StringArray::from(vec![
                oversized.as_str()
            ]))))
            .is_err()
        );
        let bytes = vec![0u8; 8001];
        assert!(
            validate_batch(&one_column(Arc::new(BinaryArray::from(vec![
                bytes.as_slice()
            ]))))
            .is_err()
        );
        assert!(validate_batch(&one_column(Arc::new(Float64Array::from(vec![f64::NAN])))).is_err());
        assert!(
            validate_batch(&one_column(Arc::new(Float32Array::from(vec![
                f32::INFINITY
            ]))))
            .is_err()
        );
        assert!(
            validate_batch(&one_column(Arc::new(Float64Array::from(vec![
                Some(f64::MAX),
                None
            ]))))
            .is_ok()
        );
    }
}
