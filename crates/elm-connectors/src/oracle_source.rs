//! Experimental Oracle reader. Native handles never leave their blocking worker.
use crate::{dialect_for, native_diagnostics::oracle_connect_descriptor};
use arrow_array::{
    RecordBatch,
    builder::{
        ArrayBuilder, BinaryBuilder, Decimal128Builder, Float32Builder, Float64Builder,
        StringBuilder, TimestampMicrosecondBuilder, make_builder,
    },
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use elm_core::{DataSource, DatabaseKind, DatabaseSelection, ElmError, Environment, Result};
use oracle::{
    Row,
    sql_type::{OracleType, RefCursor, Timestamp},
};
use secrecy::{ExposeSecret, SecretString};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{mpsc, oneshot};

struct Request {
    bytes: usize,
    reply: oneshot::Sender<Result<Option<RecordBatch>>>,
}
struct SourceQuery {
    sql: String,
    timeout: std::time::Duration,
}

// A cancelled future may discard a batch after the native cursor advanced. Close
// the worker channel and poison this source rather than permit a later fetch.
struct PendingRequest {
    sender: Option<mpsc::Sender<Request>>,
    cancelled: Arc<AtomicBool>,
}
impl Drop for PendingRequest {
    fn drop(&mut self) {
        if self.sender.is_some() {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

/// Byte-bounded native array reader; all cursor handles stay on one worker.
pub struct OracleSource {
    schema: Arc<Schema>,
    requests: Option<mpsc::Sender<Request>>,
    cancelled: Arc<AtomicBool>,
    rows: u64,
    exhausted: bool,
}
impl Drop for OracleSource {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
impl OracleSource {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        selection: &DatabaseSelection,
    ) -> Result<Self> {
        let descriptor = oracle_connect_descriptor(environment)?;
        let username = environment.username.clone();
        let password = SecretString::from(password.to_owned());
        let query = SourceQuery {
            sql: source_sql(selection)?,
            timeout: std::time::Duration::from_millis(environment.oracle_call_timeout_ms()?),
        };
        let (requests, receiver) = mpsc::channel(1);
        let (ready, schema_receiver) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut source = Self {
            schema: Arc::new(Schema::empty()),
            requests: Some(requests),
            cancelled: cancelled.clone(),
            rows: 0,
            exhausted: false,
        };
        tokio::task::spawn_blocking(move || {
            let mut ready = Some(ready);
            if let Err(error) = worker(
                &username,
                &password,
                &descriptor,
                &query,
                receiver,
                &mut ready,
                &cancelled,
            ) && let Some(ready) = ready.take()
            {
                let _sent = ready.send(Err(error));
            }
        });
        source.schema = schema_receiver.await.map_err(|_| source_error())??;
        Ok(source)
    }
}
#[async_trait]
impl DataSource for OracleSource {
    async fn schema(&mut self) -> Result<Arc<Schema>> {
        Ok(self.schema.clone())
    }
    async fn next_batch(&mut self, target_bytes: usize) -> Result<Option<RecordBatch>> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(ElmError::Cancelled);
        }
        if self.exhausted {
            return Ok(None);
        }
        let mut pending = PendingRequest {
            sender: Some(self.requests.take().ok_or(ElmError::Cancelled)?),
            cancelled: self.cancelled.clone(),
        };
        let (reply, result) = oneshot::channel();
        pending
            .sender
            .as_ref()
            .ok_or(ElmError::Cancelled)?
            .send(Request {
                bytes: target_bytes,
                reply,
            })
            .await
            .map_err(|_| source_error())?;
        let batch = result.await.map_err(|_| source_error())??;
        if let Some(batch) = &batch {
            self.rows = self
                .rows
                .checked_add(batch.num_rows() as u64)
                .ok_or_else(source_error)?;
        } else {
            self.exhausted = true;
        }
        self.requests = pending.sender.take();
        Ok(batch)
    }
    async fn checkpoint(&mut self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"row": self.rows}))
    }
}
fn source_error() -> ElmError {
    ElmError::Connection { message: "Oracle source connection, query, or fetch failed; check native client, privileges, TLS configuration, and timeout settings".into(), retryable: false }
}
fn source_sql(selection: &DatabaseSelection) -> Result<String> {
    match selection {
        DatabaseSelection::Query { sql } => Ok(sql.clone()),
        DatabaseSelection::Table { relation } => {
            if relation.catalog.is_some() {
                return Err(ElmError::Unsupported(
                    "Oracle source tables use owner/schema and table identifiers, not catalogs"
                        .into(),
                ));
            }
            Ok(format!(
                "SELECT * FROM {}",
                dialect_for(DatabaseKind::Oracle).quote_relation(relation)
            ))
        }
    }
}
fn worker(
    username: &str,
    password: &SecretString,
    descriptor: &str,
    query: &SourceQuery,
    mut requests: mpsc::Receiver<Request>,
    ready: &mut Option<oneshot::Sender<Result<Arc<Schema>>>>,
    cancelled: &AtomicBool,
) -> Result<()> {
    let sql = query.sql.as_str();
    if cancelled.load(Ordering::Acquire) {
        return Err(ElmError::Cancelled);
    }
    oracle::Version::client().map_err(|_| {
        ElmError::NativeClientMissing("Oracle Instant Client is not installed".into())
    })?;
    let connection = oracle::Connection::connect(username, password.expose_secret(), descriptor)
        .map_err(|_| source_error())?;
    connection
        .set_call_timeout(Some(query.timeout))
        .map_err(|_| source_error())?;
    if cancelled.load(Ordering::Acquire) {
        return Err(ElmError::Cancelled);
    }
    // Preparing with OCI does not execute the statement. Check SELECT before
    // DBMS_SQL.PARSE, which would otherwise execute user-supplied DDL.
    let probe = connection
        .statement(sql)
        .build()
        .map_err(|_| source_error())?;
    if !probe.is_query() {
        return Err(ElmError::Validation(
            "Oracle sources must be SELECT queries".into(),
        ));
    }
    drop(probe);
    let describe_sql = format!(
        "DECLARE c INTEGER; n INTEGER; d DBMS_SQL.DESC_TAB3; b NUMBER := 0; \
         BEGIN c := DBMS_SQL.OPEN_CURSOR; DBMS_SQL.PARSE(c, :sql, DBMS_SQL.NATIVE); \
         {DESCRIBE_BOUNDED} :cursor_id := c; :row_bytes := b; END;"
    );
    let mut describe = connection
        .statement(&describe_sql)
        .build()
        .map_err(|_| source_error())?;
    describe
        .execute(&[&sql, &OracleType::Number(10, 0), &OracleType::Number(10, 0)])
        .map_err(|error| metadata_error(&error))?;
    let cursor_id: i64 = describe
        .bind_value("cursor_id")
        .map_err(|_| source_error())?;
    let declared_bytes: usize = describe
        .bind_value("row_bytes")
        .map_err(|_| source_error())?;
    let fetch_capacity = batch_capacity(declared_bytes, 8 * 1024 * 1024)? as u32;
    if cancelled.load(Ordering::Acquire) {
        return Err(ElmError::Cancelled);
    }
    // Execute once, recheck metadata before allocating client buffers, then
    // transfer the same server cursor into a correctly sized native REF CURSOR.
    // On any failure the dedicated connection is dropped, closing owned cursors.
    let execute_sql = format!(
        "DECLARE c INTEGER := :cursor_id; n INTEGER; d DBMS_SQL.DESC_TAB3; \
         b NUMBER := 0; ignored INTEGER; BEGIN ignored := DBMS_SQL.EXECUTE(c); \
         {DESCRIBE_BOUNDED} IF b > :row_bytes THEN \
         RAISE_APPLICATION_ERROR(-20001, 'ELM source metadata changed'); END IF; \
         :result := DBMS_SQL.TO_REFCURSOR(c); END;"
    );
    let mut statement = connection
        .statement(&execute_sql)
        .fetch_array_size(fetch_capacity)
        .prefetch_rows(0)
        .lob_locator()
        .build()
        .map_err(|_| source_error())?;
    statement
        .execute(&[&cursor_id, &declared_bytes, &OracleType::RefCursor])
        .map_err(|error| metadata_error(&error))?;
    let mut cursor: RefCursor = statement.bind_value("result").map_err(|_| source_error())?;
    let mut result = cursor.query().map_err(|_| source_error())?;
    let mut fields = Vec::new();
    let mut row_bytes = 0usize;
    for column in result.column_info() {
        let (field, bytes) = map_column(column.name(), column.oracle_type(), column.nullable())?;
        if fields
            .iter()
            .any(|other: &Field| other.name() == field.name())
        {
            return Err(ElmError::TypeMapping(
                "Oracle source requires unique column aliases".into(),
            ));
        }
        row_bytes = row_bytes.checked_add(bytes).ok_or_else(source_error)?;
        fields.push(field);
    }
    let _capacity = batch_capacity(row_bytes, 8 * 1024 * 1024)?;
    if row_bytes > declared_bytes {
        return Err(ElmError::TypeMapping(
            "Oracle source metadata exceeded its declared buffer bound".into(),
        ));
    }
    let schema = Arc::new(Schema::new(fields));
    if let Some(ready) = ready.take()
        && ready.send(Ok(schema.clone())).is_err()
    {
        return Ok(());
    }
    let mut exhausted = false;
    while let Some(request) = requests.blocking_recv() {
        let batch = (|| {
            let capacity = batch_capacity(row_bytes, request.bytes)?;
            let mut builders = schema
                .fields()
                .iter()
                .map(|field| make_builder(field.data_type(), capacity))
                .collect::<Vec<_>>();
            let mut count = 0;
            for _ in 0..capacity {
                if cancelled.load(Ordering::Acquire) {
                    return Err(ElmError::Cancelled);
                }
                let Some(row) = result.next() else {
                    exhausted = true;
                    break;
                };
                let row = row.map_err(|_| source_error())?;
                append_row(&row, &schema, &mut builders)?;
                count += 1;
            }
            if count == 0 {
                return Ok(None);
            }
            let batch = RecordBatch::try_new(
                schema.clone(),
                builders
                    .iter_mut()
                    .map(|builder| builder.finish())
                    .collect(),
            )
            .map_err(|_| source_error())?;
            if batch.get_array_memory_size() > request.bytes.min(8 * 1024 * 1024) {
                return Err(ElmError::TypeMapping(
                    "Oracle decoded batch exceeded its requested byte budget".into(),
                ));
            }
            Ok(Some(batch))
        })();
        let stop = batch.is_err() || matches!(batch, Ok(None));
        if request.reply.send(batch).is_err() || stop {
            break;
        }
        if exhausted {
            // Return the final partial batch first, then one durable EOF reply.
            if let Some(request) = requests.blocking_recv() {
                let _sent = request.reply.send(Ok(None));
            }
            break;
        }
    }
    Ok(())
}
fn batch_capacity(row_bytes: usize, requested: usize) -> Result<usize> {
    let budget = requested.min(8 * 1024 * 1024);
    let reserve = row_bytes
        .checked_mul(4)
        .filter(|value| *value > 0)
        .ok_or_else(source_error)?;
    if reserve > budget {
        return Err(ElmError::TypeMapping("Oracle declared row exceeds the batch budget; select narrower columns or increase the batch size".into()));
    }
    Ok((budget / reserve).min(65_536))
}

// Shared by pre-execution and post-execution description. Unsupported/unbounded
// values are rejected on the server before rust-oracle allocates result arrays.
// col_max_len is bytes; x4 also covers conversion into the client character set.
const DESCRIBE_BOUNDED: &str = "
    DBMS_SQL.DESCRIBE_COLUMNS3(c, n, d);
    IF n < 1 OR n > 1000 THEN
        RAISE_APPLICATION_ERROR(-20001, 'ELM unsupported column count');
    END IF;
    FOR i IN 1..n LOOP
        IF d(i).col_type NOT IN (1, 2, 12, 23, 96, 100, 101, 180)
           OR (d(i).col_type IN (1, 23, 96)
               AND (d(i).col_max_len < 1 OR d(i).col_max_len > 32767))
           OR (d(i).col_type = 2 AND
               (d(i).col_precision < 1 OR d(i).col_precision > 38
                OR d(i).col_scale < -38 OR d(i).col_scale > d(i).col_precision))
           OR (d(i).col_type = 180 AND d(i).col_scale > 6) THEN
            RAISE_APPLICATION_ERROR(-20001, 'ELM unsupported source type');
        END IF;
        b := b + GREATEST(NVL(d(i).col_max_len, 0), 0) * 4 + 128;
        IF b > 2097152 THEN
            RAISE_APPLICATION_ERROR(-20001, 'ELM source row too wide');
        END IF;
    END LOOP;";

fn metadata_error(error: &oracle::Error) -> ElmError {
    if error.db_error().is_some_and(|error| error.code() == 20001) {
        ElmError::TypeMapping("Oracle source has unsupported or oversized columns; cast unconstrained numbers, LOBs, and timestamps explicitly".into())
    } else {
        source_error()
    }
}
pub(crate) fn map_column(
    name: &str,
    native: &OracleType,
    nullable: bool,
) -> Result<(Field, usize)> {
    if name.is_empty() {
        return Err(ElmError::TypeMapping(
            "Oracle source returned an unnamed column".into(),
        ));
    }
    let (data_type, bytes) = match native {
        OracleType::Varchar2(size) | OracleType::NVarchar2(size) | OracleType::Char(size) | OracleType::NChar(size) if (1..=32767).contains(size) => (DataType::Utf8, *size as usize * 4 + 64),
        OracleType::Raw(size) if (1..=32767).contains(size) => (DataType::Binary, *size as usize + 64),
        OracleType::BinaryFloat => (DataType::Float32, 64),
        OracleType::BinaryDouble => (DataType::Float64, 64),
        OracleType::Number(precision, scale) if (1..=38).contains(precision) && (-38..=*precision as i8).contains(scale) => (DataType::Decimal128(*precision, *scale), 128),
        OracleType::Date | OracleType::Timestamp(0..=6) => (DataType::Timestamp(TimeUnit::Microsecond, None), 64),
        _ => return Err(ElmError::TypeMapping("Unsupported Oracle source type: cast unconstrained NUMBER/FLOAT and high-precision timestamps explicitly; LOBs, timezones, intervals, and custom types are not implemented".into())),
    };
    Ok((Field::new(name, data_type, nullable), bytes))
}
fn append_row(row: &Row, schema: &Schema, builders: &mut [Box<dyn ArrayBuilder>]) -> Result<()> {
    for (index, (field, builder)) in schema.fields().iter().zip(builders).enumerate() {
        macro_rules! append {
            ($builder:ty, $value:ty) => {{
                builder
                    .as_any_mut()
                    .downcast_mut::<$builder>()
                    .ok_or_else(source_error)?
                    .append_option(
                        row.get::<_, Option<$value>>(index)
                            .map_err(|_| source_error())?,
                    );
            }};
        }
        match field.data_type() {
            DataType::Utf8 => append!(StringBuilder, String),
            DataType::Binary => append!(BinaryBuilder, Vec<u8>),
            DataType::Float32 => append!(Float32Builder, f32),
            DataType::Float64 => append!(Float64Builder, f64),
            DataType::Decimal128(precision, scale) => {
                let value = row
                    .get::<_, Option<String>>(index)
                    .map_err(|_| source_error())?
                    .map(|value| decimal_coefficient(&value, *precision, *scale))
                    .transpose()?;
                builder
                    .as_any_mut()
                    .downcast_mut::<Decimal128Builder>()
                    .ok_or_else(source_error)?
                    .append_option(value);
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                let value = row
                    .get::<_, Option<Timestamp>>(index)
                    .map_err(|_| source_error())?
                    .map(timestamp_micros)
                    .transpose()?;
                builder
                    .as_any_mut()
                    .downcast_mut::<TimestampMicrosecondBuilder>()
                    .ok_or_else(source_error)?
                    .append_option(value);
            }
            _ => return Err(source_error()),
        }
    }
    Ok(())
}
fn timestamp_micros(value: Timestamp) -> Result<i64> {
    if value.with_tz()
        || !value.nanosecond().is_multiple_of(1000)
        || !(1..=9999).contains(&value.year())
    {
        return Err(ElmError::TypeMapping(
            "Oracle timestamp requires an AD date, no timezone, and exact microseconds".into(),
        ));
    }
    chrono::NaiveDate::from_ymd_opt(value.year(), value.month(), value.day())
        .and_then(|date| {
            date.and_hms_nano_opt(
                value.hour(),
                value.minute(),
                value.second(),
                value.nanosecond(),
            )
        })
        .map(|value| value.and_utc().timestamp_micros())
        .ok_or_else(|| ElmError::TypeMapping("Invalid Oracle timestamp".into()))
}
fn decimal_coefficient(text: &str, precision: u8, scale: i8) -> Result<i128> {
    let invalid = || {
        ElmError::TypeMapping(
            "Oracle NUMBER cannot be represented exactly by its declared Decimal128 type".into(),
        )
    };
    if text.len() > 256
        || !(1..=38).contains(&precision)
        || !(-38..=precision as i8).contains(&scale)
    {
        return Err(invalid());
    }
    let (negative, text) = if let Some(text) = text.strip_prefix('-') {
        (true, text)
    } else {
        (false, text.strip_prefix('+').unwrap_or(text))
    };
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().map_err(|_| invalid())?),
        None => (text, 0),
    };
    let mut digits = String::new();
    let mut fractional = 0i32;
    let mut point = false;
    for byte in mantissa.bytes() {
        if byte == b'.' && !point {
            point = true;
        } else if byte.is_ascii_digit() {
            digits.push(char::from(byte));
            if point {
                fractional += 1;
            }
        } else {
            return Err(invalid());
        }
    }
    if digits.is_empty() {
        return Err(invalid());
    }
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Ok(0);
    }
    let significant = digits.trim_end_matches('0');
    let power = exponent
        .checked_add(i32::from(scale))
        .and_then(|value| value.checked_sub(fractional))
        .and_then(|value| value.checked_add((digits.len() - significant.len()) as i32))
        .ok_or_else(invalid)?;
    if power < 0 || significant.len() as i64 + i64::from(power) > i64::from(precision) {
        return Err(invalid());
    }
    let value = significant
        .parse::<i128>()
        .map_err(|_| invalid())?
        .checked_mul(10i128.pow(power as u32))
        .ok_or_else(invalid)?;
    Ok(if negative { -value } else { value })
}

pub(crate) async fn resolve_physical_identity(
    environment: &Environment,
    password: &str,
    relation: &elm_core::Relation,
) -> Result<Option<crate::physical_identity::PhysicalIdentity>> {
    if relation.catalog.is_some() {
        return Ok(None);
    }
    let descriptor = oracle_connect_descriptor(environment)?;
    let timeout = std::time::Duration::from_millis(environment.oracle_call_timeout_ms()?);
    let username = environment.username.clone();
    let password = SecretString::from(password.to_owned());
    let explicit_schema = relation
        .schema
        .as_ref()
        .map(|value| value.as_str().to_owned());
    let name = relation.name.as_str().to_owned();
    let database = environment.database.clone();
    tokio::task::spawn_blocking(move || {
        resolve_oracle_identity_blocking(
            &username,
            &password,
            &descriptor,
            timeout,
            explicit_schema.as_deref(),
            &name,
            &database,
        )
    })
    .await
    .map_err(|_| ElmError::Internal("Oracle physical-identity worker stopped".into()))?
}

fn resolve_oracle_identity_blocking(
    username: &str,
    password: &SecretString,
    descriptor: &str,
    timeout: std::time::Duration,
    explicit_schema: Option<&str>,
    name: &str,
    database: &str,
) -> Result<Option<crate::physical_identity::PhysicalIdentity>> {
    oracle::Version::client().map_err(|_| {
        ElmError::NativeClientMissing("Oracle Instant Client is not installed".into())
    })?;
    let connection = oracle::Connection::connect(username, password.expose_secret(), descriptor)
        .map_err(|_| oracle_identity_error())?;
    connection
        .set_call_timeout(Some(timeout))
        .map_err(|_| oracle_identity_error())?;

    let connected_user: String = connection
        .query_row_as("SELECT USER FROM dual", &[])
        .map_err(|_| oracle_identity_error())?;
    let owner = explicit_schema.map_or_else(|| connected_user.clone(), str::to_owned);

    let table_count: i64 = connection
        .query_row_as(
            "SELECT COUNT(*) FROM all_tables WHERE owner = :1 AND table_name = :2",
            &[&owner, &name],
        )
        .map_err(|_| oracle_identity_error())?;
    let (base_schema, base_name) = if table_count == 1 {
        (owner.clone(), name.to_owned())
    } else {
        let mut synonym = oracle_synonym_base(&connection, &owner, name)?;
        if synonym.is_none() && explicit_schema.is_none() {
            synonym = oracle_synonym_base(&connection, "PUBLIC", name)?;
        }
        if let Some((base_owner, base_name, db_link)) = synonym {
            if db_link.is_some() {
                return Ok(None);
            }
            (base_owner, base_name)
        } else {
            let dependents = oracle_view_dependents(&connection, &owner, name)?;
            let [only] = dependents.as_slice() else {
                return Ok(None);
            };
            only.clone()
        }
    };

    let db_unique_name: String = connection
        .query_row_as(
            "SELECT SYS_CONTEXT('USERENV','DB_UNIQUE_NAME') FROM dual",
            &[],
        )
        .map_err(|_| oracle_identity_error())?;
    let instance_name: String = connection
        .query_row_as(
            "SELECT SYS_CONTEXT('USERENV','INSTANCE_NAME') FROM dual",
            &[],
        )
        .map_err(|_| oracle_identity_error())?;

    Ok(Some(crate::physical_identity::PhysicalIdentity {
        server_fingerprint: format!("oracle:{db_unique_name}:{instance_name}"),
        database: database.to_owned(),
        base_schema: Some(base_schema),
        base_relation: base_name,
    }))
}

fn oracle_synonym_base(
    connection: &oracle::Connection,
    owner: &str,
    name: &str,
) -> Result<Option<(String, String, Option<String>)>> {
    match connection.query_row_as::<(String, String, Option<String>)>(
        "SELECT table_owner, table_name, db_link FROM all_synonyms WHERE owner = :1 AND synonym_name = :2",
        &[&owner, &name],
    ) {
        Ok(row) => Ok(Some(row)),
        Err(error) if error.kind() == oracle::ErrorKind::NoDataFound => Ok(None),
        Err(_error) => Err(oracle_identity_error()),
    }
}

fn oracle_view_dependents(
    connection: &oracle::Connection,
    owner: &str,
    name: &str,
) -> Result<Vec<(String, String)>> {
    let rows = connection
        .query_as::<(String, String)>(
            "SELECT DISTINCT referenced_owner, referenced_name
             FROM all_dependencies
             WHERE owner = :1 AND name = :2 AND type = 'VIEW' AND referenced_type = 'TABLE'",
            &[&owner, &name],
        )
        .map_err(|_| oracle_identity_error())?;
    let mut dependents = Vec::new();
    for row in rows {
        dependents.push(row.map_err(|_| oracle_identity_error())?);
    }
    Ok(dependents)
}

fn oracle_identity_error() -> ElmError {
    ElmError::Connection {
        message: "Oracle physical-identity query failed".into(),
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulated_source() -> (OracleSource, mpsc::Receiver<Request>, RecordBatch) {
        let batch = RecordBatch::try_from_iter([(
            "value",
            Arc::new(arrow_array::Float64Array::from(vec![1.0])) as arrow_array::ArrayRef,
        )])
        .unwrap_or_else(|error| panic!("{error}"));
        let (sender, receiver) = mpsc::channel(1);
        (
            OracleSource {
                schema: batch.schema(),
                requests: Some(sender),
                cancelled: Arc::new(AtomicBool::new(false)),
                rows: 0,
                exhausted: false,
            },
            receiver,
            batch,
        )
    }

    #[tokio::test]
    async fn cancelled_request_closes_worker_and_cannot_skip_an_unobserved_batch() {
        for reply_delivered in [false, true] {
            let (mut source, mut receiver, batch) = simulated_source();
            let mut fetch = Box::pin(source.next_batch(4096));
            let request = tokio::select! {
                biased;
                _ = &mut fetch => panic!("fetch must wait for reply"),
                request = receiver.recv() => request.unwrap_or_else(|| panic!("missing request")),
            };
            if reply_delivered {
                assert!(request.reply.send(Ok(Some(batch))).is_ok());
            }
            drop(fetch);
            assert!(source.cancelled.load(Ordering::Acquire));
            assert!(receiver.recv().await.is_none());
            assert!(matches!(
                source.next_batch(4096).await,
                Err(ElmError::Cancelled)
            ));
            assert_eq!(
                source
                    .checkpoint()
                    .await
                    .unwrap_or_else(|error| panic!("{error}")),
                serde_json::json!({"row": 0})
            );
        }
    }

    #[tokio::test]
    async fn completed_requests_keep_source_live_and_checkpoint_only_observed_rows() {
        let (mut source, mut receiver, batch) = simulated_source();
        let worker = tokio::spawn(async move {
            for reply in [Some(batch), None] {
                let request = receiver
                    .recv()
                    .await
                    .unwrap_or_else(|| panic!("missing request"));
                assert!(request.reply.send(Ok(reply)).is_ok());
            }
        });
        assert_eq!(
            source
                .next_batch(4096)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .map(|batch| batch.num_rows()),
            Some(1)
        );
        assert!(!source.cancelled.load(Ordering::Acquire));
        assert!(
            source
                .next_batch(4096)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .is_none()
        );
        assert!(
            source
                .next_batch(4096)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .is_none()
        );
        assert_eq!(source.rows, 1);
        worker.await.unwrap_or_else(|error| panic!("{error}"));
    }

    #[tokio::test]
    async fn failed_request_cannot_reuse_an_advanced_cursor() {
        let (mut source, mut receiver, _) = simulated_source();
        let worker = tokio::spawn(async move {
            let request = receiver
                .recv()
                .await
                .unwrap_or_else(|| panic!("missing request"));
            assert!(
                request
                    .reply
                    .send(Err(ElmError::TypeMapping("invalid value".into())))
                    .is_ok()
            );
            assert!(receiver.recv().await.is_none());
        });
        assert!(matches!(
            source.next_batch(4096).await,
            Err(ElmError::TypeMapping(_))
        ));
        assert!(matches!(
            source.next_batch(4096).await,
            Err(ElmError::Cancelled)
        ));
        assert_eq!(source.rows, 0);
        worker.await.unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn declared_metadata_is_exact_and_unbounded_types_fail_closed() {
        assert_eq!(
            map_column("n", &OracleType::Number(38, -2), true)
                .unwrap_or_else(|error| panic!("{error}"))
                .0
                .data_type(),
            &DataType::Decimal128(38, -2)
        );
        assert_eq!(
            map_column("d", &OracleType::Date, false)
                .unwrap_or_else(|error| panic!("{error}"))
                .0
                .data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        for native in [
            OracleType::Number(0, 0),
            OracleType::Number(3, 4),
            OracleType::Number(38, -39),
            OracleType::Float(126),
            OracleType::CLOB,
            OracleType::BLOB,
            OracleType::Long,
            OracleType::Timestamp(9),
            OracleType::TimestampTZ(6),
            OracleType::TimestampLTZ(6),
        ] {
            assert!(map_column("value", &native, true).is_err());
        }
        assert!(map_column("", &OracleType::BinaryDouble, true).is_err());
        let (_, bytes) = map_column("text", &OracleType::NVarchar2(100), true)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(bytes, 464);
    }
    #[test]
    fn decimal_parser_handles_exponents_without_rounding() {
        for (text, precision, scale, expected) in [
            ("-12345.6700", 38, 4, -123_456_700),
            ("1.2e3", 38, -2, 12),
            ("1e-38", 38, 38, 1),
            ("-0.000E+12", 1, 0, 0),
            ("1200.00", 38, -2, 12),
        ] {
            assert_eq!(
                decimal_coefficient(text, precision, scale)
                    .unwrap_or_else(|error| panic!("{error}")),
                expected
            );
        }
        for text in [
            "1.001", "1000", "NaN", "1E9999", "1E-9999", "", ".", "1.2.3", "1e2e3", "1,23", "--1",
        ] {
            assert!(decimal_coefficient(text, 3, 2).is_err(), "accepted {text}");
        }
        for precision in 1..=38 {
            for scale in -38..=precision as i8 {
                for coefficient in [
                    0,
                    1,
                    -1,
                    10i128.pow(u32::from(precision)) - 1,
                    1 - 10i128.pow(u32::from(precision)),
                ] {
                    let text = format!("{coefficient}e{}", -i32::from(scale));
                    assert_eq!(
                        decimal_coefficient(&text, precision, scale)
                            .unwrap_or_else(|error| panic!("{error}")),
                        coefficient
                    );
                }
            }
        }
    }
    #[test]
    fn native_date_keeps_time_and_rejects_precision_loss() {
        let value = Timestamp::new(1969, 12, 31, 23, 59, 59, 999_999_000)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            timestamp_micros(value).unwrap_or_else(|error| panic!("{error}")),
            -1
        );
        let too_precise =
            Timestamp::new(2024, 2, 29, 12, 0, 0, 1).unwrap_or_else(|error| panic!("{error}"));
        assert!(timestamp_micros(too_precise).is_err());
        assert!(
            timestamp_micros(
                value
                    .and_tz_offset(0)
                    .unwrap_or_else(|error| panic!("{error}"))
            )
            .is_err()
        );
    }
    #[test]
    fn batching_reserves_native_and_decoded_memory() {
        assert_eq!(
            batch_capacity(128, 1024).unwrap_or_else(|error| panic!("{error}")),
            2
        );
        assert!(batch_capacity(128, 511).is_err());
        assert!(batch_capacity(0, 1024).is_err());
        assert!(batch_capacity(usize::MAX, usize::MAX).is_err());
        assert!(batch_capacity(1, usize::MAX).unwrap_or_else(|error| panic!("{error}")) <= 65536);
    }
    #[test]
    fn structured_identifiers_are_quoted_and_query_text_is_unchanged() {
        let selection = DatabaseSelection::Table {
            relation: elm_core::Relation {
                catalog: None,
                schema: Some(
                    elm_core::Identifier::new("Owner").unwrap_or_else(|error| panic!("{error}")),
                ),
                name: elm_core::Identifier::new("a\"b").unwrap_or_else(|error| panic!("{error}")),
            },
        };
        assert_eq!(
            source_sql(&selection).unwrap_or_else(|error| panic!("{error}")),
            "SELECT * FROM \"Owner\".\"a\"\"b\""
        );
        let sql = "SELECT :1 FROM dual";
        assert_eq!(
            source_sql(&DatabaseSelection::Query { sql: sql.into() })
                .unwrap_or_else(|error| panic!("{error}")),
            sql
        );
    }
}
