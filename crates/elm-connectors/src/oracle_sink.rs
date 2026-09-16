//! Oracle batch DML and explicit, recoverable non-atomic table replacement.
use crate::{
    dialect_for,
    native_diagnostics::oracle_connect_descriptor,
    oracle_publication::{SwapAction, SwapIdentity, SwapObjects},
};
use arrow_array::{
    Array, BinaryArray, Decimal128Array, Float32Array, Float64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Schema, TimeUnit};
use async_trait::async_trait;
use chrono::{Datelike, Timelike};
use elm_core::{
    BatchCheckpoint, ConnectorCapabilities, ConsistencyMode, DataSink, DatabaseKind, ElmError,
    Environment, Identifier, JobId, PreflightReport, Relation, Result, WriteMode,
};
use oracle::{
    Connection,
    sql_type::{OracleType, Timestamp, ToSql},
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
    Report(PreflightReport),
}
struct Request {
    action: Action,
    reply: oneshot::Sender<Result<Reply>>,
}
pub struct OracleSink {
    requests: mpsc::Sender<Request>,
}

impl OracleSink {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        target: Relation,
        job: JobId,
        checkpoint: Option<&BatchCheckpoint>,
    ) -> Result<Self> {
        let descriptor = oracle_connect_descriptor(environment)?;
        let timeout = std::time::Duration::from_millis(environment.oracle_call_timeout_ms()?);
        let username = environment.username.clone();
        let password = SecretString::from(password.to_owned());
        let plan = Plan::new(target, job)?;
        let checkpoint = checkpoint.cloned();
        let (requests, mut receiver) = mpsc::channel::<Request>(1);
        let (ready, result) = oneshot::channel();
        tokio::task::spawn_blocking(move || {
            let connection = match connect(&username, &password, &descriptor, timeout) {
                Ok(connection) => connection,
                Err(error) => {
                    let _sent = ready.send(Err(error));
                    return;
                }
            };
            let mut state = State {
                connection: &connection,
                plan,
                checkpoint,
                schema: None,
                mode: WriteMode::Fail,
                original: None,
                rows: 0,
                sequence: 0,
                pending: false,
                begun: false,
            };
            if ready.send(Ok(())).is_err() {
                return;
            }
            while let Some(request) = receiver.blocking_recv() {
                let result = state.handle(request.action);
                if result.is_err() {
                    let _rollback = connection.rollback();
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
            _ => Err(native_error()),
        }
    }
}
#[async_trait]
impl DataSink for OracleSink {
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
            Reply::Report(report) => Ok(report),
            _ => Err(native_error()),
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

fn native_error() -> ElmError {
    ElmError::Connection { message: "Oracle staging operation failed; check privileges, constraints, native client and recovery records. Existing targets and recovery backups have not been automatically removed".into(), retryable: false }
}
fn connect(
    username: &str,
    password: &SecretString,
    descriptor: &str,
    timeout: std::time::Duration,
) -> Result<Connection> {
    let connection = Connection::connect(username, password.expose_secret(), descriptor)
        .map_err(|_| native_error())?;
    connection
        .set_call_timeout(Some(timeout))
        .map_err(|_| native_error())?;
    execute(
        &connection,
        "ALTER SESSION SET NLS_NUMERIC_CHARACTERS = '.,'",
    )?;
    Ok(connection)
}
fn execute(connection: &Connection, sql: &str) -> Result<()> {
    connection
        .execute(sql, &[])
        .map_err(native_statement_error)?;
    Ok(())
}
fn native_statement_error(error: oracle::Error) -> ElmError {
    // ORA-01031 is Oracle's specific "insufficient privileges" code; report it distinctly so a
    // caller can tell a privilege failure from a transient connection problem.
    if error.oci_code() == Some(1031) {
        return ElmError::PermissionDenied(
            "Oracle account lacks the privilege required for this staging operation".into(),
        );
    }
    // Never expose native message text: it may contain SQL or row values.
    let category = if let Some(code) = error.oci_code() {
        format!("ORA-{code:05}")
    } else if let Some(code) = error.dpi_code() {
        format!("DPI-{code:04}")
    } else {
        format!("{:?}", error.kind())
    };
    ElmError::Connection {
        message: format!(
            "Oracle staging operation failed ({category}); check privileges, timeout settings, and recovery records"
        ),
        retryable: false,
    }
}
fn quote(value: &str) -> Result<String> {
    Ok(dialect_for(DatabaseKind::Oracle).quote_identifier(&Identifier::new(value)?))
}
struct Plan {
    target: Relation,
    staging: String,
    backup: String,
    receipt: String,
    marker: String,
}
impl Plan {
    fn new(target: Relation, job: JobId) -> Result<Self> {
        if target.catalog.is_some() || target.name.as_str().len() > 128 {
            return Err(ElmError::Validation("Oracle targets require an owner and a table name of at most 128 bytes, without a catalog".into()));
        }
        let suffix = job.0.simple();
        Ok(Self {
            marker: format!("elm:v2:oracle:{job}:{}", target.name),
            target,
            staging: format!("elm_stage_{suffix}"),
            backup: format!("elm_backup_{suffix}"),
            receipt: format!("elm_receipt_{suffix}"),
        })
    }
    fn relation(&self, name: &str) -> Result<String> {
        Ok(dialect_for(DatabaseKind::Oracle).quote_relation(&Relation {
            catalog: None,
            schema: self.target.schema.clone(),
            name: Identifier::new(name)?,
        }))
    }
    fn check_owner(&self, connection: &Connection) -> Result<()> {
        let owner: String = connection
            .query_row_as("SELECT USER FROM dual", &[])
            .map_err(|_| native_error())?;
        if self
            .target
            .schema
            .as_ref()
            .is_some_and(|schema| schema.as_str() != owner)
        {
            return Err(ElmError::Unsupported(
                "Oracle writes currently require the connected user's own schema".into(),
            ));
        }
        Ok(())
    }
    fn id(&self, connection: &Connection, name: &str) -> Result<Option<u64>> {
        connection.query_row_as("SELECT MAX(object_id) FROM user_objects WHERE object_name = :1 AND object_type = 'TABLE' AND subobject_name IS NULL", &[&name]).map_err(|_| native_error())
    }
    fn mark(&self, connection: &Connection, name: &str) -> Result<()> {
        execute(
            connection,
            &format!(
                "COMMENT ON TABLE {} IS '{}'",
                self.relation(name)?,
                self.marker.replace('\'', "''")
            ),
        )
    }
    fn journal(&self, connection: &Connection) -> Result<Option<Journal>> {
        if self.id(connection, &self.receipt)?.is_none() {
            return Ok(None);
        }
        let marker: Option<String> = connection
            .query_row_as(
                "SELECT comments FROM user_tab_comments WHERE table_name = :1",
                &[&self.receipt],
            )
            .map_err(|_| native_error())?;
        if marker.as_deref() != Some(&self.marker) {
            return Err(ElmError::Conflict(
                "Oracle recovery journal is not owned by this job".into(),
            ));
        }
        let sql = format!(
            "SELECT staged_id, original_id, committed_rows, phase, schema_hash FROM {}",
            self.relation(&self.receipt)?
        );
        let (staged, original, rows, phase, schema_hash): (u64, Option<u64>, u64, String, String) =
            connection
                .query_row_as(&sql, &[])
                .map_err(|_| native_error())?;
        Ok(Some(Journal {
            staged,
            original,
            rows,
            phase,
            schema_hash,
        }))
    }
    fn phase(&self, connection: &Connection, phase: &str, rows: u64) -> Result<()> {
        connection
            .execute(
                &format!(
                    "UPDATE {} SET phase = :1, committed_rows = :2",
                    self.relation(&self.receipt)?
                ),
                &[&phase, &rows],
            )
            .map_err(|_| native_error())?;
        connection.commit().map_err(|_| native_error())
    }
    fn rename(&self, connection: &Connection, from: &str, to: &str) -> Result<()> {
        execute(
            connection,
            &format!(
                "ALTER TABLE {} RENAME TO {}",
                self.relation(from)?,
                quote(to)?
            ),
        )
    }
    fn finish(&self, connection: &Connection, journal: &Journal) -> Result<u64> {
        if journal.phase == "published" {
            return Ok(journal.rows);
        }
        if journal.phase != "ready" {
            return Err(ElmError::State(
                "Oracle staging has not been validated for publication".into(),
            ));
        }
        if let Some(original) = journal.original {
            let identity = SwapIdentity {
                staged: journal.staged,
                original,
            };
            for _ in 0..3 {
                let action = identity.next_action(SwapObjects {
                    target: self.id(connection, self.target.name.as_str())?,
                    staging: self.id(connection, &self.staging)?,
                    backup: self.id(connection, &self.backup)?,
                })?;
                match action {
                    SwapAction::RenameOriginalToBackup => {
                        self.rename(connection, self.target.name.as_str(), &self.backup)?
                    }
                    SwapAction::RenameStagingToTarget => {
                        self.rename(connection, &self.staging, self.target.name.as_str())?
                    }
                    SwapAction::Published => {
                        self.phase(connection, "published", journal.rows)?;
                        return Ok(journal.rows);
                    }
                }
            }
            return Err(native_error());
        }
        let target_id = self.id(connection, self.target.name.as_str())?;
        let stage_id = self.id(connection, &self.staging)?;
        if target_id.is_none() && stage_id == Some(journal.staged) {
            self.rename(connection, &self.staging, self.target.name.as_str())?;
        } else if target_id != Some(journal.staged) || stage_id.is_some() {
            return Err(ElmError::Conflict(
                "Oracle publication objects changed; inspect recovery journal".into(),
            ));
        }
        self.phase(connection, "published", journal.rows)?;
        Ok(journal.rows)
    }
}
struct Journal {
    staged: u64,
    original: Option<u64>,
    rows: u64,
    phase: String,
    schema_hash: String,
}
struct State<'a> {
    connection: &'a Connection,
    plan: Plan,
    checkpoint: Option<BatchCheckpoint>,
    schema: Option<Arc<Schema>>,
    mode: WriteMode,
    original: Option<u64>,
    rows: u64,
    sequence: u64,
    pending: bool,
    begun: bool,
}
impl State<'_> {
    fn handle(&mut self, action: Action) -> Result<Reply> {
        match action {
            Action::Preflight(schema, mode, consistency) => {
                return self.preflight(schema, mode, consistency).map(Reply::Report);
            }
            Action::Begin => self.begin()?,
            Action::Write(batch) => self.write(&batch)?,
            Action::Checkpoint(checkpoint) => {
                if !self.pending
                    || checkpoint.sequence != self.sequence
                    || checkpoint.rows_committed != self.rows
                {
                    return Err(ElmError::State(
                        "Oracle checkpoint does not match pending batch".into(),
                    ));
                }
                self.plan.phase(self.connection, "loading", self.rows)?;
                self.sequence = self.sequence.checked_add(1).ok_or_else(native_error)?;
                self.pending = false;
            }
            Action::Publish => self.publish()?,
            Action::Abort => {
                self.connection.rollback().map_err(|_| native_error())?;
                self.begun = false;
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
        self.plan.check_owner(self.connection)?;
        if consistency == ConsistencyMode::Checkpointed
            || (consistency == ConsistencyMode::TableSwap && mode != WriteMode::Replace)
        {
            return Err(ElmError::Unsupported(
                "Oracle supports atomic new-table publication or explicit table-swap REPLACE"
                    .into(),
            ));
        }
        if schema.fields().is_empty() || schema.fields().len() > 999 {
            return Err(ElmError::TypeMapping(
                "Oracle staging requires 1..999 columns".into(),
            ));
        }
        let mut names = std::collections::HashSet::new();
        let mut row_bytes = 128usize;
        for field in schema.fields() {
            if field.name() == "__elm_batch"
                || field.name().len() > 128
                || !names.insert(field.name())
            {
                return Err(ElmError::TypeMapping("Oracle column names must be unique, at most 128 bytes, and may not use __elm_batch".into()));
            }
            row_bytes += column_type(field.data_type())?.1;
            Identifier::new(field.name())?;
        }
        if row_bytes > 2 * 1024 * 1024 {
            return Err(ElmError::TypeMapping(
                "Oracle declared sink row exceeds the native batch budget".into(),
            ));
        }
        self.original = self
            .plan
            .id(self.connection, self.plan.target.name.as_str())?;
        if self.original.is_none() {
            let occupied: u64 = self.connection.query_row_as(
                "SELECT COUNT(*) FROM user_objects WHERE object_name = :1 AND object_type IN ('VIEW', 'MATERIALIZED VIEW', 'SYNONYM', 'SEQUENCE', 'PROCEDURE', 'FUNCTION', 'PACKAGE', 'TYPE')",
                &[&self.plan.target.name.as_str()],
            ).map_err(|_| native_error())?;
            if occupied != 0 {
                return Err(ElmError::Conflict("Oracle destination name belongs to a non-table object; no staging load was started".into()));
            }
        }
        if self.original.is_some() {
            if mode == WriteMode::Fail {
                return Err(ElmError::Conflict(
                    "Oracle destination already exists".into(),
                ));
            }
            if mode == WriteMode::Replace && consistency != ConsistencyMode::TableSwap {
                return Err(ElmError::Unsupported("Existing Oracle targets currently require REPLACE with explicit table-swap consistency; atomic replacement is unavailable".into()));
            }
            // Swapping object identity cannot preserve incoming foreign keys,
            // dependent objects, triggers, grants, or the original table's indexes.
            if mode == WriteMode::Replace {
                let dependencies: u64 = self.connection.query_row_as("SELECT (SELECT COUNT(*) FROM user_dependencies WHERE referenced_name = :1 AND referenced_type = 'TABLE') + (SELECT COUNT(*) FROM all_constraints WHERE constraint_type = 'R' AND r_owner = USER AND r_constraint_name IN (SELECT constraint_name FROM user_constraints WHERE table_name = :1)) FROM dual", &[&self.plan.target.name.as_str()]).map_err(|_| native_error())?;
                if dependencies != 0 {
                    return Err(ElmError::Unsupported("Oracle table swap refuses targets with dependent objects or incoming foreign keys".into()));
                }
            } else {
                self.validate_append(&schema)?;
            }
        }
        self.schema = Some(schema);
        self.mode = mode;
        Ok(PreflightReport {
            capabilities: ConnectorCapabilities {
                database: Some(DatabaseKind::Oracle),
                native_bulk_read: true,
                native_bulk_write: true,
                atomic_append: true,
                atomic_replace: self.original.is_none(),
                non_atomic_replace: true,
                checkpointed_write: false,
                resumable_keyset_read: false,
                supports_lob_spill: false,
            },
            staging_relation: Some(Relation {
                catalog: None,
                schema: self.plan.target.schema.clone(),
                name: Identifier::new(&self.plan.staging)?,
            }),
            warnings: if consistency == ConsistencyMode::TableSwap {
                vec!["Oracle table swap is NON-ATOMIC: the target name may be unavailable between renames. Resume completes a validated swap. Keep exclusive control of target DDL during publication. Original indexes, grants, and constraints stay on the retained backup, not on the replacement".into()]
            } else {
                Vec::new()
            },
        })
    }
    fn validate_append(&self, schema: &Schema) -> Result<()> {
        let name = self.plan.target.name.as_str();
        let special: u64 = self.connection.query_row_as("SELECT (SELECT COUNT(*) FROM user_triggers WHERE table_name = :1) + (SELECT COUNT(*) FROM user_tab_cols WHERE table_name = :1 AND (virtual_column = 'YES' OR identity_column = 'YES')) FROM dual", &[&name]).map_err(|_| native_error())?;
        if special != 0 {
            return Err(ElmError::Unsupported(
                "Oracle atomic append refuses triggers, virtual columns, and identity columns"
                    .into(),
            ));
        }
        let sql = format!("SELECT * FROM {} WHERE 1=0", self.plan.relation(name)?);
        let mut statement = self
            .connection
            .statement(&sql)
            .fetch_array_size(1)
            .prefetch_rows(0)
            .lob_locator()
            .build()
            .map_err(|_| native_error())?;
        let result = statement.query(&[]).map_err(|_| native_error())?;
        if result.column_info().len() != schema.fields().len() {
            return Err(ElmError::TypeMapping(
                "Oracle append column count differs from source".into(),
            ));
        }
        for (column, field) in result.column_info().iter().zip(schema.fields()) {
            let (target, _) = crate::oracle_source::map_column(
                column.name(),
                column.oracle_type(),
                column.nullable(),
            )?;
            if target.name() != field.name() || target.data_type() != field.data_type() {
                return Err(ElmError::TypeMapping(
                    "Oracle append requires matching column names, order, and canonical types"
                        .into(),
                ));
            }
        }
        Ok(())
    }
    fn begin(&mut self) -> Result<()> {
        if self.begun {
            return Err(ElmError::Conflict("Oracle sink already begun".into()));
        }
        let schema = self.schema.as_ref().ok_or_else(native_error)?;
        let schema_hash =
            blake3::hash(&serde_json::to_vec(schema.as_ref()).map_err(|_| native_error())?)
                .to_hex()
                .to_string();
        if let Some(journal) = self.plan.journal(self.connection)? {
            if journal.schema_hash != schema_hash {
                return Err(ElmError::TypeMapping(
                    "Oracle source schema changed since the staging load began".into(),
                ));
            }
            if journal.phase != "loading"
                || self.plan.id(self.connection, &self.plan.staging)? != Some(journal.staged)
                || journal.original != self.original
            {
                return Err(ElmError::Conflict(
                    "Oracle publication requires reconciliation before loading".into(),
                ));
            }
            let cutoff = self
                .checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.sequence);
            let stage = self.plan.relation(&self.plan.staging)?;
            if let Some(cutoff) = cutoff {
                self.connection
                    .execute(
                        &format!("DELETE FROM {stage} WHERE \"__elm_batch\" > :1"),
                        &[&cutoff],
                    )
                    .map_err(|_| native_error())?;
            } else {
                execute(self.connection, &format!("DELETE FROM {stage}"))?;
            }
            self.rows = self
                .checkpoint
                .as_ref()
                .map_or(0, |checkpoint| checkpoint.rows_committed);
            self.sequence = match &self.checkpoint {
                Some(checkpoint) => checkpoint
                    .sequence
                    .checked_add(1)
                    .ok_or_else(native_error)?,
                None => 0,
            };
            let count: u64 = self
                .connection
                .query_row_as(&format!("SELECT COUNT(*) FROM {stage}"), &[])
                .map_err(|_| native_error())?;
            if count != self.rows {
                return Err(ElmError::State(
                    "Oracle staging row count differs from checkpoint".into(),
                ));
            }
            self.plan.phase(self.connection, "loading", self.rows)?;
        } else {
            if self.checkpoint.is_some()
                || self.plan.id(self.connection, &self.plan.staging)?.is_some()
                || self.plan.id(self.connection, &self.plan.backup)?.is_some()
            {
                return Err(ElmError::Conflict("Oracle staging resources exist without a valid journal, or checkpoint staging is missing".into()));
            }
            let columns = schema
                .fields()
                .iter()
                .map(|field| {
                    Ok(format!(
                        "{} {}{}",
                        quote(field.name())?,
                        column_type(field.data_type())?.0,
                        if field.is_nullable() { "" } else { " NOT NULL" }
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .join(", ");
            execute(
                self.connection,
                &format!(
                    "CREATE TABLE {} ({columns}, \"__elm_batch\" NUMBER(20,0) DEFAULT 0 NOT NULL)",
                    self.plan.relation(&self.plan.staging)?
                ),
            )?;
            self.plan.mark(self.connection, &self.plan.staging)?;
            execute(
                self.connection,
                &format!(
                    "CREATE TABLE {} (staged_id NUMBER(20,0) NOT NULL, original_id NUMBER(20,0), committed_rows NUMBER(20,0) NOT NULL, phase VARCHAR2(20) NOT NULL, schema_hash VARCHAR2(64) NOT NULL)",
                    self.plan.relation(&self.plan.receipt)?
                ),
            )?;
            self.plan.mark(self.connection, &self.plan.receipt)?;
            let staged = self
                .plan
                .id(self.connection, &self.plan.staging)?
                .ok_or_else(native_error)?;
            self.connection
                .execute(
                    &format!(
                        "INSERT INTO {} VALUES (:1, :2, 0, 'loading', :3)",
                        self.plan.relation(&self.plan.receipt)?
                    ),
                    &[&staged, &self.original, &schema_hash],
                )
                .map_err(native_statement_error)?;
            self.connection.commit().map_err(native_statement_error)?;
        }
        self.begun = true;
        Ok(())
    }
    fn write(&mut self, record: &RecordBatch) -> Result<()> {
        if !self.begun || self.pending || self.schema.as_deref() != Some(record.schema().as_ref()) {
            return Err(ElmError::State(
                "Oracle sink is not ready for this batch".into(),
            ));
        }
        let mut widths = Vec::new();
        for field in record.schema().fields() {
            widths.push(column_type(field.data_type())?.1);
        }
        let row_bytes = widths.iter().sum::<usize>() + 128;
        let capacity = (2 * 1024 * 1024 / row_bytes).clamp(1, 4096);
        let placeholders = (1..=record.num_columns() + 1)
            .map(|index| format!(":{index}"))
            .collect::<Vec<_>>()
            .join(",");
        let columns = record
            .schema()
            .fields()
            .iter()
            .map(|field| quote(field.name()))
            .collect::<Result<Vec<_>>>()?
            .join(",");
        let sql = format!(
            "INSERT INTO {} ({columns}, \"__elm_batch\") VALUES ({placeholders})",
            self.plan.relation(&self.plan.staging)?
        );
        let mut batch = self
            .connection
            .batch(&sql, capacity)
            .build()
            .map_err(|_| native_error())?;
        for (index, field) in record.schema().fields().iter().enumerate() {
            if field.data_type() == &DataType::Utf8 {
                batch
                    .set_type(index + 1, &OracleType::NVarchar2(2000))
                    .map_err(|_| native_error())?;
            }
        }
        for row in 0..record.num_rows() {
            let mut values = record
                .columns()
                .iter()
                .map(|array| bind_value(array.as_ref(), row))
                .collect::<Result<Vec<_>>>()?;
            values.push(Box::new(self.sequence));
            let bindings = values
                .iter()
                .map(|value| value.as_ref())
                .collect::<Vec<_>>();
            batch.append_row(&bindings).map_err(|_| native_error())?;
        }
        batch.execute().map_err(|_| native_error())?;
        self.rows = self
            .rows
            .checked_add(record.num_rows() as u64)
            .ok_or_else(native_error)?;
        self.pending = true;
        Ok(())
    }
    fn publish(&mut self) -> Result<()> {
        if !self.begun || self.pending {
            return Err(ElmError::State(
                "Oracle sink has uncommitted staging data".into(),
            ));
        }
        let journal = self
            .plan
            .journal(self.connection)?
            .ok_or_else(native_error)?;
        if journal.phase != "loading"
            || self.plan.id(self.connection, &self.plan.staging)? != Some(journal.staged)
            || self
                .plan
                .id(self.connection, self.plan.target.name.as_str())?
                != self.original
        {
            return Err(ElmError::Conflict(
                "Oracle publication identities changed".into(),
            ));
        }
        let stage = self.plan.relation(&self.plan.staging)?;
        let rows: u64 = self
            .connection
            .query_row_as(&format!("SELECT COUNT(*) FROM {stage}"), &[])
            .map_err(|_| native_error())?;
        if rows != self.rows {
            return Err(ElmError::State(
                "Oracle staging row count validation failed".into(),
            ));
        }
        if self.mode == WriteMode::Append && self.original.is_some() {
            let target = self.plan.relation(self.plan.target.name.as_str())?;
            execute(
                self.connection,
                &format!("LOCK TABLE {target} IN EXCLUSIVE MODE NOWAIT"),
            )?;
            if self
                .plan
                .id(self.connection, self.plan.target.name.as_str())?
                != self.original
            {
                return Err(ElmError::Conflict(
                    "Oracle append target was replaced".into(),
                ));
            }
            let schema = self.schema.as_ref().ok_or_else(native_error)?;
            self.validate_append(schema)?;
            let columns = schema
                .fields()
                .iter()
                .map(|field| quote(field.name()))
                .collect::<Result<Vec<_>>>()?
                .join(",");
            execute(
                self.connection,
                &format!("INSERT INTO {target} ({columns}) SELECT {columns} FROM {stage}"),
            )?;
            // Receipt and append commit in the same transaction; failed inserts
            // roll back both, including duplicate-key and permission failures.
            self.plan.phase(self.connection, "published", self.rows)?;
            self.begun = false;
            return Ok(());
        }
        // Persist the irreversible publication boundary before the first DDL.
        // The private sequence column is made invisible, not dropped: a crash
        // before 'ready' can still restart or trim the checkpointed staging load.
        execute(
            self.connection,
            &format!("ALTER TABLE {stage} MODIFY (\"__elm_batch\" INVISIBLE)"),
        )?;
        self.plan.phase(self.connection, "ready", self.rows)?;
        let journal = self
            .plan
            .journal(self.connection)?
            .ok_or_else(native_error)?;
        self.plan.finish(self.connection, &journal)?;
        self.begun = false;
        Ok(())
    }
}

fn column_type(data_type: &DataType) -> Result<(String, usize)> {
    Ok(match data_type {
        DataType::Utf8 => ("NVARCHAR2(2000)".into(), 16_128),
        DataType::Binary => ("RAW(2000)".into(), 2_128),
        DataType::Decimal128(precision, scale) if (1..=38).contains(precision) && (-38..=*precision as i8).contains(scale) => (format!("NUMBER({precision},{scale})"), 256),
        DataType::Float32 => ("BINARY_FLOAT".into(), 128),
        DataType::Float64 => ("BINARY_DOUBLE".into(), 128),
        DataType::Timestamp(TimeUnit::Microsecond, None) => ("TIMESTAMP(6)".into(), 128),
        _ => return Err(ElmError::TypeMapping("Oracle sink requires bounded strings/binary, Decimal128, floats, or timezone-free microsecond timestamps; supply explicit conversions for other types".into())),
    })
}
fn bind_value(array: &dyn Array, row: usize) -> Result<Box<dyn ToSql>> {
    macro_rules! value {
        ($kind:ty) => {
            array
                .as_any()
                .downcast_ref::<$kind>()
                .ok_or_else(native_error)?
                .value(row)
        };
    }
    let native_type = match array.data_type() {
        DataType::Utf8 => OracleType::NVarchar2(2000),
        DataType::Binary => OracleType::Raw(2000),
        DataType::Decimal128(_, _) => OracleType::Varchar2(128),
        DataType::Float32 => OracleType::BinaryFloat,
        DataType::Float64 => OracleType::BinaryDouble,
        DataType::Timestamp(TimeUnit::Microsecond, None) => OracleType::Timestamp(6),
        _ => return Err(native_error()),
    };
    if array.is_null(row) {
        return Ok(Box::new(native_type));
    }
    Ok(match array.data_type() {
        DataType::Utf8 => {
            let text = value!(StringArray);
            if text.is_empty() || text.encode_utf16().count() > 2000 {
                return Err(ElmError::TypeMapping("Oracle strings must be nonempty and fit NVARCHAR2(2000); Oracle would convert an empty string into NULL".into()));
            }
            Box::new(text.to_owned())
        }
        DataType::Binary => {
            let bytes = value!(BinaryArray);
            if bytes.is_empty() || bytes.len() > 2000 {
                return Err(ElmError::TypeMapping(
                    "Oracle binary values must be nonempty and fit RAW(2000)".into(),
                ));
            }
            Box::new(bytes.to_vec())
        }
        DataType::Decimal128(_, scale) => Box::new(decimal_text(value!(Decimal128Array), *scale)),
        DataType::Float32 => Box::new(value!(Float32Array)),
        DataType::Float64 => Box::new(value!(Float64Array)),
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let value = chrono::DateTime::from_timestamp_micros(value!(TimestampMicrosecondArray))
                .ok_or_else(native_error)?;
            if !(1..=9999).contains(&value.year()) {
                return Err(ElmError::TypeMapping(
                    "Oracle timestamp requires years 1..9999".into(),
                ));
            }
            Box::new(
                Timestamp::new(
                    value.year(),
                    value.month(),
                    value.day(),
                    value.hour(),
                    value.minute(),
                    value.second(),
                    value.nanosecond(),
                )
                .map_err(|_| native_error())?,
            )
        }
        _ => return Err(native_error()),
    })
}
fn decimal_text(value: i128, scale: i8) -> String {
    let digits = value.unsigned_abs().to_string();
    let sign = if value < 0 { "-" } else { "" };
    if scale <= 0 {
        return format!("{sign}{digits}{}", "0".repeat((-i16::from(scale)) as usize));
    }
    let scale = scale as usize;
    if digits.len() <= scale {
        format!("{sign}0.{}{digits}", "0".repeat(scale - digits.len()))
    } else {
        let split = digits.len() - scale;
        format!("{sign}{}.{}", &digits[..split], &digits[split..])
    }
}

/// Reconcile a validated publication before reopening a potentially changed source.
pub async fn oracle_published_rows(
    environment: &Environment,
    password: &str,
    target: Relation,
    job: JobId,
) -> Result<Option<u64>> {
    let descriptor = oracle_connect_descriptor(environment)?;
    let timeout = std::time::Duration::from_millis(environment.oracle_call_timeout_ms()?);
    let username = environment.username.clone();
    let password = SecretString::from(password.to_owned());
    let plan = Plan::new(target, job)?;
    tokio::task::spawn_blocking(move || {
        let connection = connect(&username, &password, &descriptor, timeout)?;
        plan.check_owner(&connection)?;
        let Some(journal) = plan.journal(&connection)? else {
            return Ok(None);
        };
        if journal.phase == "loading" {
            return Ok(None);
        }
        plan.finish(&connection, &journal).map(Some)
    })
    .await
    .map_err(|_| native_error())?
}

/// Delete only journal-owned private objects. An interrupted swap is restored
/// before its staging is removed; deletion never publishes unfinished work.
pub async fn cleanup_oracle_staging(
    environment: &Environment,
    password: &str,
    target: Relation,
    job: JobId,
) -> Result<()> {
    let descriptor = oracle_connect_descriptor(environment)?;
    let timeout = std::time::Duration::from_millis(environment.oracle_call_timeout_ms()?);
    let username = environment.username.clone();
    let password = SecretString::from(password.to_owned());
    let plan = Plan::new(target, job)?;
    tokio::task::spawn_blocking(move || {
        let connection = connect(&username, &password, &descriptor, timeout)?;
        plan.check_owner(&connection)?;
        let Some(journal) = plan.journal(&connection)? else {
            if plan.id(&connection, &plan.staging)?.is_some()
                || plan.id(&connection, &plan.backup)?.is_some()
            {
                return Err(ElmError::Conflict(
                    "Oracle private objects have no valid journal; manual inspection is required"
                        .into(),
                ));
            }
            return Ok(());
        };
        let mut target_id = plan.id(&connection, plan.target.name.as_str())?;
        let stage_id = plan.id(&connection, &plan.staging)?;
        let backup_id = plan.id(&connection, &plan.backup)?;
        if stage_id.is_some_and(|id| id != journal.staged)
            || backup_id.is_some_and(|id| Some(id) != journal.original)
        {
            return Err(ElmError::Conflict(
                "Oracle cleanup refuses unowned staging or backup objects".into(),
            ));
        }
        if target_id.is_none() && backup_id.is_some() {
            plan.rename(&connection, &plan.backup, plan.target.name.as_str())?;
            target_id = backup_id;
        } else if backup_id.is_some() {
            if target_id != Some(journal.staged) || stage_id.is_some() {
                return Err(ElmError::Conflict(
                    "Oracle backup cannot be removed until the replacement is confirmed".into(),
                ));
            }
            execute(
                &connection,
                &format!("DROP TABLE {}", plan.relation(&plan.backup)?),
            )?;
        }
        if stage_id.is_some() {
            if target_id == Some(journal.staged) {
                return Err(native_error());
            }
            execute(
                &connection,
                &format!("DROP TABLE {}", plan.relation(&plan.staging)?),
            )?;
        }
        execute(
            &connection,
            &format!("DROP TABLE {}", plan.relation(&plan.receipt)?),
        )
    })
    .await
    .map_err(|_| native_error())?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[allow(deprecated)] // rust-oracle 0.6 exposes native error construction through enum variants.
    fn native_diagnostics_expose_codes_without_message_contents() {
        let errors = [
            (
                oracle::Error::OciError(oracle::DbError::new(
                    1,
                    0,
                    "secret-row-value",
                    "private-sql",
                    "private-action",
                )),
                "ORA-00001",
            ),
            (
                oracle::Error::DpiError(oracle::DbError::new(
                    0,
                    0,
                    "DPI-1067: secret-row-value",
                    "private-sql",
                    "private-action",
                )),
                "DPI-1067",
            ),
        ];
        for (error, code) in errors {
            let message = native_statement_error(error).to_string();
            assert!(message.contains(code));
            for secret in ["secret-row-value", "private-sql", "private-action"] {
                assert!(!message.contains(secret));
            }
        }
    }
    #[test]
    fn decimal_bindings_are_exact_and_do_not_use_locale() {
        assert_eq!(decimal_text(-1234, 3), "-1.234");
        assert_eq!(decimal_text(12, -2), "1200");
        assert_eq!(decimal_text(-1, 4), "-0.0001");
        assert_eq!(decimal_text(0, 2), "0.00");
    }
    #[test]
    fn empty_strings_and_oversized_binary_fail_instead_of_losing_values() {
        assert!(bind_value(&StringArray::from(vec![""]), 0).is_err());
        assert!(bind_value(&BinaryArray::from(vec![vec![0u8; 2001].as_slice()]), 0).is_err());
    }
}
