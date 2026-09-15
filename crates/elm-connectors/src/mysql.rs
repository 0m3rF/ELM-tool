use std::{collections::BTreeSet, path::PathBuf, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray, UInt8Array, UInt16Array,
    UInt32Array, UInt64Array,
    builder::{
        BinaryBuilder, Date32Builder, Decimal128Builder, Float32Builder, Float64Builder,
        Int8Builder, Int16Builder, Int32Builder, Int64Builder, StringBuilder,
        TimestampMicrosecondBuilder, UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
    },
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use bytes::Bytes;
use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike, Utc};
use futures::{StreamExt, stream};
use mysql_async::{
    Column, Conn, Error as MySqlError, Opts, OptsBuilder, Params, Row, SslOpts, Statement, Value,
    consts::{ColumnFlags, ColumnType},
    prelude::Queryable,
};
use tokio::sync::{mpsc, oneshot};

use elm_core::{
    BatchCheckpoint, ConnectorCapabilities, ConsistencyMode, DataSink, DataSource, DatabaseKind,
    DatabaseSelection, ElmError, Environment, Identifier, JobId, PreflightReport, Relation, Result,
    WriteMode,
};

use crate::dialect_for;

const STAGING_COMMENT_PREFIX: &str = "elm-tool staging job:";

enum SourceRequest {
    Next {
        target_bytes: usize,
        response: oneshot::Sender<Result<Option<RecordBatch>>>,
    },
}

/// A prepared binary-protocol MySQL query streamed through a request-driven worker.
pub struct MySqlSource {
    schema: Arc<Schema>,
    requests: mpsc::Sender<SourceRequest>,
    rows_read: u64,
    worker: tokio::task::JoinHandle<()>,
}

impl Drop for MySqlSource {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

impl MySqlSource {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        selection: &DatabaseSelection,
    ) -> Result<Self> {
        let (mut connection, _tls_enabled) = connect_mysql(environment, password).await?;
        let query = selection_query(selection)?;
        let statement = connection
            .prep(query)
            .await
            .map_err(mysql_source_preflight_error)?;
        let (schema, columns) = schema_from_columns(statement.columns())?;
        let schema = Arc::new(schema);
        let (requests, request_receiver) = mpsc::channel(1);
        let (ready_sender, ready_receiver) = oneshot::channel();
        let worker = tokio::spawn(mysql_source_worker(
            connection,
            statement,
            schema.clone(),
            columns,
            request_receiver,
            ready_sender,
        ));
        let source = Self {
            schema,
            requests,
            rows_read: 0,
            worker,
        };
        ready_receiver.await.map_err(|_| ElmError::Connection {
            message: "MySQL source worker stopped during query startup".into(),
            retryable: true,
        })??;
        Ok(source)
    }
}

#[async_trait]
impl DataSource for MySqlSource {
    async fn schema(&mut self) -> Result<Arc<Schema>> {
        Ok(self.schema.clone())
    }

    async fn next_batch(&mut self, target_bytes: usize) -> Result<Option<RecordBatch>> {
        let (response, receiver) = oneshot::channel();
        self.requests
            .send(SourceRequest::Next {
                target_bytes,
                response,
            })
            .await
            .map_err(|_| ElmError::Connection {
                message: "MySQL source worker is no longer available".into(),
                retryable: true,
            })?;
        let batch = receiver.await.map_err(|_| ElmError::Connection {
            message: "MySQL source worker stopped while reading a batch".into(),
            retryable: true,
        })??;
        if let Some(batch) = &batch {
            self.rows_read = self
                .rows_read
                .saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX));
        }
        Ok(batch)
    }

    async fn checkpoint(&mut self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({ "rows_read": self.rows_read }))
    }
}

async fn mysql_source_worker(
    mut connection: Conn,
    statement: Statement,
    schema: Arc<Schema>,
    columns: Vec<MySqlColumn>,
    mut requests: mpsc::Receiver<SourceRequest>,
    ready: oneshot::Sender<Result<()>>,
) {
    let mut result = match connection.exec_iter(&statement, ()).await {
        Ok(result) => result,
        Err(error) => {
            let _sent = ready.send(Err(mysql_source_start_error(error)));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    let mut exhausted = false;
    while let Some(request) = requests.recv().await {
        match request {
            SourceRequest::Next {
                target_bytes,
                response,
            } => {
                if exhausted {
                    let _sent = response.send(Ok(None));
                    continue;
                }
                let capacity = (target_bytes / 128).clamp(1, 65_536);
                let builders = columns
                    .iter()
                    .map(|column| MySqlBuilder::new(&column.kind, capacity))
                    .collect::<Result<Vec<_>>>();
                let mut builders = match builders {
                    Ok(builders) => builders,
                    Err(error) => {
                        let _sent = response.send(Err(error));
                        break;
                    }
                };
                let mut rows = 0_usize;
                let mut estimated_bytes = 0_usize;
                let mut read_error = None;
                while estimated_bytes < target_bytes.max(1) {
                    match result.next().await {
                        Ok(Some(row)) => match append_mysql_row(&row, &mut builders) {
                            Ok(bytes) => {
                                estimated_bytes = estimated_bytes.saturating_add(bytes);
                                rows = rows.saturating_add(1);
                            }
                            Err(error) => {
                                read_error = Some(error);
                                break;
                            }
                        },
                        Ok(None) => {
                            exhausted = true;
                            break;
                        }
                        Err(_error) => {
                            read_error = Some(ElmError::Connection {
                                message: "MySQL source stream was interrupted".into(),
                                retryable: true,
                            });
                            break;
                        }
                    }
                }
                if let Some(error) = read_error {
                    let _sent = response.send(Err(error));
                    break;
                }
                if rows == 0 {
                    let _sent = response.send(Ok(None));
                    continue;
                }
                let arrays = builders
                    .into_iter()
                    .map(MySqlBuilder::finish)
                    .collect::<Vec<_>>();
                let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(|error| {
                    ElmError::Internal(format!("cannot build MySQL Arrow batch: {error}"))
                });
                let _sent = response.send(batch.map(Some));
            }
        }
    }
}

/// Performs an authenticated connection and a round-trip query without exposing server details.
pub async fn test_mysql_environment(environment: &Environment, password: &str) -> Result<()> {
    let (mut connection, _tls_enabled) = connect_mysql(environment, password).await?;
    connection
        .query_drop("SELECT 1")
        .await
        .map_err(|_| ElmError::Connection {
            message: "MySQL diagnostic query failed".into(),
            retryable: true,
        })?;
    connection
        .disconnect()
        .await
        .map_err(mysql_connection_error)
}

async fn connect_mysql(environment: &Environment, password: &str) -> Result<(Conn, bool)> {
    if environment.kind != DatabaseKind::MySql {
        return Err(ElmError::Validation(
            "MySQL connector received a different database kind".into(),
        ));
    }
    let (options, tls_enabled) = mysql_options(environment, password)?;
    let connection = Conn::new(options).await.map_err(mysql_connection_error)?;
    Ok((connection, tls_enabled))
}

fn mysql_options(environment: &Environment, password: &str) -> Result<(Opts, bool)> {
    let builder = OptsBuilder::default()
        .ip_or_hostname(environment.host.clone())
        .tcp_port(environment.port)
        .user(Some(environment.username.clone()))
        .pass(Some(password.to_owned()))
        .db_name(Some(environment.database.clone()))
        .prefer_socket(false)
        .setup(vec![
            "SET NAMES utf8mb4",
            "SET SESSION time_zone = '+00:00'",
        ]);
    let ssl_mode = environment
        .options
        .get("ssl_mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("require");
    let (builder, tls_enabled) = match ssl_mode {
        "require" => {
            let mut ssl = SslOpts::default();
            if let Some(path) = environment
                .options
                .get("root_certificate")
                .and_then(serde_json::Value::as_str)
                .filter(|path| !path.trim().is_empty())
            {
                ssl = ssl.with_root_certs(vec![PathBuf::from(path).into()]);
            }
            (builder.ssl_opts(Some(ssl)), true)
        }
        "disable" => (builder.ssl_opts(None), false),
        _ => {
            return Err(ElmError::Validation(
                "MySQL ssl_mode must be 'require' or 'disable'".into(),
            ));
        }
    };
    Ok((Opts::from(builder), tls_enabled))
}

fn selection_query(selection: &DatabaseSelection) -> Result<String> {
    match selection {
        DatabaseSelection::Table { relation } => {
            if relation.catalog.is_some() {
                return Err(ElmError::Validation(
                    "MySQL relations cannot contain a catalog; use schema for a database name"
                        .into(),
                ));
            }
            Ok(format!(
                "SELECT * FROM {}",
                dialect_for(DatabaseKind::MySql).quote_relation(relation)
            ))
        }
        DatabaseSelection::Query { sql } => {
            let trimmed = sql.trim();
            if trimmed.is_empty() {
                return Err(ElmError::Validation(
                    "MySQL source query cannot be empty".into(),
                ));
            }
            if trimmed.ends_with(';') {
                return Err(ElmError::Validation(
                    "MySQL source query must omit its trailing semicolon".into(),
                ));
            }
            Ok(sql.clone())
        }
    }
}

#[derive(Debug, Clone)]
struct MySqlColumn {
    kind: MySqlColumnKind,
}

#[derive(Debug, Clone)]
enum MySqlColumnKind {
    Null,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Decimal128 { precision: u8, scale: i8 },
    Text,
    Binary,
    Date,
    Timestamp,
    TimestampTz,
}

fn schema_from_columns(columns: &[Column]) -> Result<(Schema, Vec<MySqlColumn>)> {
    if columns.is_empty() {
        return Err(ElmError::Validation(
            "MySQL source query must return at least one column".into(),
        ));
    }
    let mut names = BTreeSet::new();
    let mut fields = Vec::with_capacity(columns.len());
    let mut mapped = Vec::with_capacity(columns.len());
    for column in columns {
        let name = std::str::from_utf8(column.name_ref())
            .map_err(|_| ElmError::TypeMapping("MySQL returned a non-UTF-8 column name".into()))?;
        if !names.insert(name.to_owned()) {
            return Err(ElmError::TypeMapping(format!(
                "MySQL source returned duplicate column '{name}'"
            )));
        }
        let (data_type, kind) = mysql_column_to_arrow(column)
            .map_err(|error| ElmError::TypeMapping(format!("column '{name}': {error}")))?;
        let nullable = !column.flags().contains(ColumnFlags::NOT_NULL_FLAG);
        fields.push(Field::new(name, data_type, nullable));
        mapped.push(MySqlColumn { kind });
    }
    Ok((Schema::new(fields), mapped))
}

fn mysql_column_to_arrow(
    column: &Column,
) -> std::result::Result<(DataType, MySqlColumnKind), String> {
    use ColumnType::{
        MYSQL_TYPE_BLOB, MYSQL_TYPE_DATE, MYSQL_TYPE_DATETIME, MYSQL_TYPE_DATETIME2,
        MYSQL_TYPE_DECIMAL, MYSQL_TYPE_DOUBLE, MYSQL_TYPE_FLOAT, MYSQL_TYPE_INT24, MYSQL_TYPE_LONG,
        MYSQL_TYPE_LONG_BLOB, MYSQL_TYPE_LONGLONG, MYSQL_TYPE_MEDIUM_BLOB, MYSQL_TYPE_NEWDATE,
        MYSQL_TYPE_NEWDECIMAL, MYSQL_TYPE_NULL, MYSQL_TYPE_SHORT, MYSQL_TYPE_STRING,
        MYSQL_TYPE_TIMESTAMP, MYSQL_TYPE_TIMESTAMP2, MYSQL_TYPE_TINY, MYSQL_TYPE_TINY_BLOB,
        MYSQL_TYPE_VAR_STRING, MYSQL_TYPE_VARCHAR, MYSQL_TYPE_YEAR,
    };
    let unsigned = column.flags().contains(ColumnFlags::UNSIGNED_FLAG);
    if column
        .flags()
        .intersects(ColumnFlags::ENUM_FLAG | ColumnFlags::SET_FLAG)
    {
        return Err("MySQL ENUM and SET require an explicit conversion".into());
    }
    let mapped = match column.column_type() {
        MYSQL_TYPE_NULL => (DataType::Null, MySqlColumnKind::Null),
        MYSQL_TYPE_TINY if unsigned => (DataType::UInt8, MySqlColumnKind::UInt8),
        MYSQL_TYPE_TINY => (DataType::Int8, MySqlColumnKind::Int8),
        MYSQL_TYPE_SHORT if unsigned => (DataType::UInt16, MySqlColumnKind::UInt16),
        MYSQL_TYPE_SHORT => (DataType::Int16, MySqlColumnKind::Int16),
        MYSQL_TYPE_INT24 | MYSQL_TYPE_LONG if unsigned => {
            (DataType::UInt32, MySqlColumnKind::UInt32)
        }
        MYSQL_TYPE_INT24 | MYSQL_TYPE_LONG => (DataType::Int32, MySqlColumnKind::Int32),
        MYSQL_TYPE_LONGLONG if unsigned => (DataType::UInt64, MySqlColumnKind::UInt64),
        MYSQL_TYPE_LONGLONG => (DataType::Int64, MySqlColumnKind::Int64),
        MYSQL_TYPE_YEAR => (DataType::UInt16, MySqlColumnKind::UInt16),
        MYSQL_TYPE_FLOAT => (DataType::Float32, MySqlColumnKind::Float32),
        MYSQL_TYPE_DOUBLE => (DataType::Float64, MySqlColumnKind::Float64),
        MYSQL_TYPE_DECIMAL | MYSQL_TYPE_NEWDECIMAL => {
            let scale = column.decimals();
            let punctuation = u32::from(scale > 0) + u32::from(!unsigned);
            let precision = column
                .column_length()
                .checked_sub(punctuation)
                .ok_or_else(|| "MySQL DECIMAL metadata has an invalid display length".to_owned())?;
            let precision = u8::try_from(precision)
                .map_err(|_| "MySQL DECIMAL precision exceeds Arrow Decimal128".to_owned())?;
            if precision == 0 || precision > 38 || scale > precision {
                return Err(format!(
                    "MySQL DECIMAL({precision},{scale}) has no valid Arrow Decimal128 mapping"
                ));
            }
            let scale = i8::try_from(scale)
                .map_err(|_| "MySQL DECIMAL scale exceeds Arrow Decimal128".to_owned())?;
            (
                DataType::Decimal128(precision, scale),
                MySqlColumnKind::Decimal128 { precision, scale },
            )
        }
        MYSQL_TYPE_STRING
        | MYSQL_TYPE_VAR_STRING
        | MYSQL_TYPE_VARCHAR
        | MYSQL_TYPE_TINY_BLOB
        | MYSQL_TYPE_MEDIUM_BLOB
        | MYSQL_TYPE_LONG_BLOB
        | MYSQL_TYPE_BLOB => {
            if column.character_set() == 63 {
                (DataType::Binary, MySqlColumnKind::Binary)
            } else {
                (DataType::Utf8, MySqlColumnKind::Text)
            }
        }
        MYSQL_TYPE_DATE | MYSQL_TYPE_NEWDATE => (DataType::Date32, MySqlColumnKind::Date),
        MYSQL_TYPE_DATETIME | MYSQL_TYPE_DATETIME2 => (
            DataType::Timestamp(TimeUnit::Microsecond, None),
            MySqlColumnKind::Timestamp,
        ),
        MYSQL_TYPE_TIMESTAMP | MYSQL_TYPE_TIMESTAMP2 => (
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            MySqlColumnKind::TimestampTz,
        ),
        other => {
            return Err(format!(
                "MySQL type {other:?} has no lossless Arrow mapping"
            ));
        }
    };
    Ok(mapped)
}

enum MySqlBuilder {
    Null(usize),
    Int8(Int8Builder),
    Int16(Int16Builder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    UInt8(UInt8Builder),
    UInt16(UInt16Builder),
    UInt32(UInt32Builder),
    UInt64(UInt64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Decimal128(Decimal128Builder, u8, i8),
    Text(StringBuilder),
    Binary(BinaryBuilder),
    Date(Date32Builder),
    Timestamp(TimestampMicrosecondBuilder),
    TimestampTz(TimestampMicrosecondBuilder),
}

impl MySqlBuilder {
    fn new(kind: &MySqlColumnKind, capacity: usize) -> Result<Self> {
        Ok(match kind {
            MySqlColumnKind::Null => Self::Null(0),
            MySqlColumnKind::Int8 => Self::Int8(Int8Builder::with_capacity(capacity)),
            MySqlColumnKind::Int16 => Self::Int16(Int16Builder::with_capacity(capacity)),
            MySqlColumnKind::Int32 => Self::Int32(Int32Builder::with_capacity(capacity)),
            MySqlColumnKind::Int64 => Self::Int64(Int64Builder::with_capacity(capacity)),
            MySqlColumnKind::UInt8 => Self::UInt8(UInt8Builder::with_capacity(capacity)),
            MySqlColumnKind::UInt16 => Self::UInt16(UInt16Builder::with_capacity(capacity)),
            MySqlColumnKind::UInt32 => Self::UInt32(UInt32Builder::with_capacity(capacity)),
            MySqlColumnKind::UInt64 => Self::UInt64(UInt64Builder::with_capacity(capacity)),
            MySqlColumnKind::Float32 => Self::Float32(Float32Builder::with_capacity(capacity)),
            MySqlColumnKind::Float64 => Self::Float64(Float64Builder::with_capacity(capacity)),
            MySqlColumnKind::Decimal128 { precision, scale } => Self::Decimal128(
                Decimal128Builder::with_capacity(capacity)
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|error| {
                        ElmError::TypeMapping(format!(
                            "invalid MySQL decimal precision or scale: {error}"
                        ))
                    })?,
                *precision,
                *scale,
            ),
            MySqlColumnKind::Text => Self::Text(StringBuilder::with_capacity(
                capacity,
                capacity.saturating_mul(16),
            )),
            MySqlColumnKind::Binary => Self::Binary(BinaryBuilder::with_capacity(
                capacity,
                capacity.saturating_mul(16),
            )),
            MySqlColumnKind::Date => Self::Date(Date32Builder::with_capacity(capacity)),
            MySqlColumnKind::Timestamp => {
                Self::Timestamp(TimestampMicrosecondBuilder::with_capacity(capacity))
            }
            MySqlColumnKind::TimestampTz => Self::TimestampTz(
                TimestampMicrosecondBuilder::with_capacity(capacity).with_timezone("UTC"),
            ),
        })
    }

    fn append(&mut self, value: &Value) -> Result<usize> {
        if matches!(value, Value::NULL) {
            self.append_null();
            return Ok(1);
        }
        Ok(match self {
            Self::Null(_) => {
                return Err(ElmError::TypeMapping(
                    "MySQL NULL-typed column returned a non-null value".into(),
                ));
            }
            Self::Int8(builder) => append_signed(builder, value, 1)?,
            Self::Int16(builder) => append_signed(builder, value, 2)?,
            Self::Int32(builder) => append_signed(builder, value, 4)?,
            Self::Int64(builder) => append_signed(builder, value, 8)?,
            Self::UInt8(builder) => append_unsigned(builder, value, 1)?,
            Self::UInt16(builder) => append_unsigned(builder, value, 2)?,
            Self::UInt32(builder) => append_unsigned(builder, value, 4)?,
            Self::UInt64(builder) => append_unsigned(builder, value, 8)?,
            Self::Float32(builder) => {
                builder.append_value(mysql_float(value)? as f32);
                4
            }
            Self::Float64(builder) => {
                builder.append_value(mysql_float(value)?);
                8
            }
            Self::Decimal128(builder, precision, scale) => {
                let value = parse_mysql_decimal(mysql_bytes(value)?, *scale)?;
                ensure_decimal_precision(value, *precision)?;
                builder.append_value(value);
                16
            }
            Self::Text(builder) => {
                let value = std::str::from_utf8(mysql_bytes(value)?).map_err(|_| {
                    ElmError::TypeMapping("MySQL text value was not valid UTF-8".into())
                })?;
                let bytes = value.len();
                builder.append_value(value);
                bytes.saturating_add(4)
            }
            Self::Binary(builder) => {
                let value = mysql_bytes(value)?;
                let bytes = value.len();
                builder.append_value(value);
                bytes.saturating_add(4)
            }
            Self::Date(builder) => {
                let days = mysql_date(value)?
                    .signed_duration_since(unix_epoch())
                    .num_days();
                builder.append_value(i32::try_from(days).map_err(|_| {
                    ElmError::TypeMapping("MySQL DATE exceeded Arrow Date32 range".into())
                })?);
                4
            }
            Self::Timestamp(builder) | Self::TimestampTz(builder) => {
                builder.append_value(mysql_datetime(value)?.and_utc().timestamp_micros());
                8
            }
        })
    }

    fn append_null(&mut self) {
        match self {
            Self::Null(rows) => *rows = rows.saturating_add(1),
            Self::Int8(builder) => builder.append_null(),
            Self::Int16(builder) => builder.append_null(),
            Self::Int32(builder) => builder.append_null(),
            Self::Int64(builder) => builder.append_null(),
            Self::UInt8(builder) => builder.append_null(),
            Self::UInt16(builder) => builder.append_null(),
            Self::UInt32(builder) => builder.append_null(),
            Self::UInt64(builder) => builder.append_null(),
            Self::Float32(builder) => builder.append_null(),
            Self::Float64(builder) => builder.append_null(),
            Self::Decimal128(builder, _, _) => builder.append_null(),
            Self::Text(builder) => builder.append_null(),
            Self::Binary(builder) => builder.append_null(),
            Self::Date(builder) => builder.append_null(),
            Self::Timestamp(builder) | Self::TimestampTz(builder) => builder.append_null(),
        }
    }

    fn finish(mut self) -> ArrayRef {
        match &mut self {
            Self::Null(rows) => Arc::new(arrow_array::NullArray::new(*rows)),
            Self::Int8(builder) => Arc::new(builder.finish()),
            Self::Int16(builder) => Arc::new(builder.finish()),
            Self::Int32(builder) => Arc::new(builder.finish()),
            Self::Int64(builder) => Arc::new(builder.finish()),
            Self::UInt8(builder) => Arc::new(builder.finish()),
            Self::UInt16(builder) => Arc::new(builder.finish()),
            Self::UInt32(builder) => Arc::new(builder.finish()),
            Self::UInt64(builder) => Arc::new(builder.finish()),
            Self::Float32(builder) => Arc::new(builder.finish()),
            Self::Float64(builder) => Arc::new(builder.finish()),
            Self::Decimal128(builder, _, _) => Arc::new(builder.finish()),
            Self::Text(builder) => Arc::new(builder.finish()),
            Self::Binary(builder) => Arc::new(builder.finish()),
            Self::Date(builder) => Arc::new(builder.finish()),
            Self::Timestamp(builder) | Self::TimestampTz(builder) => Arc::new(builder.finish()),
        }
    }
}

fn append_mysql_row(row: &Row, builders: &mut [MySqlBuilder]) -> Result<usize> {
    if row.len() != builders.len() {
        return Err(ElmError::State(
            "MySQL row width differs from prepared metadata".into(),
        ));
    }
    let mut bytes = 0_usize;
    for (index, builder) in builders.iter_mut().enumerate() {
        let value = row
            .as_ref(index)
            .ok_or_else(|| ElmError::State("MySQL row contained a missing column value".into()))?;
        bytes = bytes.saturating_add(builder.append(value)?);
    }
    Ok(bytes)
}

fn append_signed<T>(
    builder: &mut arrow_array::builder::PrimitiveBuilder<T>,
    value: &Value,
    width: usize,
) -> Result<usize>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: TryFrom<i64>,
{
    let value = T::Native::try_from(mysql_signed(value)?).map_err(|_| {
        ElmError::TypeMapping("MySQL signed integer exceeded its Arrow range".into())
    })?;
    builder.append_value(value);
    Ok(width)
}

fn append_unsigned<T>(
    builder: &mut arrow_array::builder::PrimitiveBuilder<T>,
    value: &Value,
    width: usize,
) -> Result<usize>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: TryFrom<u64>,
{
    let value = T::Native::try_from(mysql_unsigned(value)?).map_err(|_| {
        ElmError::TypeMapping("MySQL unsigned integer exceeded its Arrow range".into())
    })?;
    builder.append_value(value);
    Ok(width)
}

fn mysql_signed(value: &Value) -> Result<i64> {
    match value {
        Value::Int(value) => Ok(*value),
        Value::UInt(value) => i64::try_from(*value).map_err(|_| {
            ElmError::TypeMapping("MySQL unsigned value exceeded signed Arrow range".into())
        }),
        Value::Bytes(value) => std::str::from_utf8(value)
            .ok()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| ElmError::TypeMapping("MySQL signed integer was malformed".into())),
        _ => Err(ElmError::TypeMapping(
            "MySQL value was not a signed integer".into(),
        )),
    }
}

fn mysql_unsigned(value: &Value) -> Result<u64> {
    match value {
        Value::UInt(value) => Ok(*value),
        Value::Int(value) => u64::try_from(*value)
            .map_err(|_| ElmError::TypeMapping("MySQL integer was unexpectedly negative".into())),
        Value::Bytes(value) => std::str::from_utf8(value)
            .ok()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| ElmError::TypeMapping("MySQL unsigned integer was malformed".into())),
        _ => Err(ElmError::TypeMapping(
            "MySQL value was not an unsigned integer".into(),
        )),
    }
}

fn mysql_float(value: &Value) -> Result<f64> {
    match value {
        Value::Float(value) => Ok(f64::from(*value)),
        Value::Double(value) => Ok(*value),
        Value::Bytes(value) => std::str::from_utf8(value)
            .ok()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| ElmError::TypeMapping("MySQL floating value was malformed".into())),
        _ => Err(ElmError::TypeMapping(
            "MySQL value was not floating point".into(),
        )),
    }
}

fn mysql_bytes(value: &Value) -> Result<&[u8]> {
    match value {
        Value::Bytes(value) => Ok(value),
        _ => Err(ElmError::TypeMapping(
            "MySQL value was not returned as bytes".into(),
        )),
    }
}

fn mysql_date(value: &Value) -> Result<NaiveDate> {
    match value {
        Value::Date(year, month, day, 0, 0, 0, 0) => {
            NaiveDate::from_ymd_opt(i32::from(*year), u32::from(*month), u32::from(*day))
                .ok_or_else(|| ElmError::TypeMapping("MySQL DATE value was invalid or zero".into()))
        }
        _ => Err(ElmError::TypeMapping("MySQL value was not a DATE".into())),
    }
}

fn mysql_datetime(value: &Value) -> Result<NaiveDateTime> {
    match value {
        Value::Date(year, month, day, hour, minute, second, micros) => {
            let date =
                NaiveDate::from_ymd_opt(i32::from(*year), u32::from(*month), u32::from(*day))
                    .ok_or_else(|| {
                        ElmError::TypeMapping("MySQL DATETIME value was invalid or zero".into())
                    })?;
            date.and_hms_micro_opt(
                u32::from(*hour),
                u32::from(*minute),
                u32::from(*second),
                *micros,
            )
            .ok_or_else(|| ElmError::TypeMapping("MySQL DATETIME time was invalid".into()))
        }
        _ => Err(ElmError::TypeMapping(
            "MySQL value was not a DATETIME or TIMESTAMP".into(),
        )),
    }
}

fn parse_mysql_decimal(value: &[u8], scale: i8) -> Result<i128> {
    let value = std::str::from_utf8(value)
        .map_err(|_| ElmError::TypeMapping("MySQL DECIMAL was not valid ASCII".into()))?;
    let (negative, value) = value
        .strip_prefix('-')
        .map_or((false, value), |value| (true, value));
    let (integer, fraction) = value.split_once('.').unwrap_or((value, ""));
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(ElmError::TypeMapping(
            "MySQL DECIMAL text was malformed".into(),
        ));
    }
    let scale = usize::try_from(scale)
        .map_err(|_| ElmError::TypeMapping("MySQL DECIMAL scale was negative".into()))?;
    if fraction.len() > scale {
        return Err(ElmError::TypeMapping(
            "MySQL DECIMAL had more fractional digits than its Arrow scale".into(),
        ));
    }
    let coefficient = integer
        .bytes()
        .chain(fraction.bytes())
        .chain(std::iter::repeat_n(b'0', scale - fraction.len()))
        .try_fold(0_i128, |coefficient, digit| {
            coefficient
                .checked_mul(10)
                .and_then(|coefficient| coefficient.checked_add(i128::from(digit - b'0')))
                .ok_or_else(|| {
                    ElmError::TypeMapping("MySQL DECIMAL exceeds Decimal128 range".into())
                })
        })?;
    if negative {
        coefficient
            .checked_neg()
            .ok_or_else(|| ElmError::TypeMapping("MySQL DECIMAL exceeds Decimal128 range".into()))
    } else {
        Ok(coefficient)
    }
}

fn ensure_decimal_precision(value: i128, precision: u8) -> Result<()> {
    let digits = value
        .checked_abs()
        .map_or(39, |value| value.to_string().len());
    if digits > usize::from(precision) {
        return Err(ElmError::TypeMapping(format!(
            "decimal coefficient requires {digits} digits but precision is {precision}"
        )));
    }
    Ok(())
}

fn unix_epoch() -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap_or(NaiveDate::MIN)
}

#[derive(Debug)]
struct MySqlSinkColumn {
    ddl: String,
}

impl MySqlSinkColumn {
    fn from_field(field: &Field) -> Result<Self> {
        let ddl = match field.data_type() {
            DataType::Boolean => "BOOLEAN".into(),
            DataType::Int8 => "TINYINT".into(),
            DataType::Int16 => "SMALLINT".into(),
            DataType::Int32 => "INTEGER".into(),
            DataType::Int64 => "BIGINT".into(),
            DataType::UInt8 => "TINYINT UNSIGNED".into(),
            DataType::UInt16 => "SMALLINT UNSIGNED".into(),
            DataType::UInt32 => "INTEGER UNSIGNED".into(),
            DataType::UInt64 => "BIGINT UNSIGNED".into(),
            DataType::Float32 => "FLOAT".into(),
            DataType::Float64 => "DOUBLE".into(),
            DataType::Decimal128(precision, scale)
                if *precision > 0
                    && *precision <= 38
                    && *scale >= 0
                    && scale.unsigned_abs() <= *precision =>
            {
                format!("DECIMAL({precision}, {scale})")
            }
            DataType::Utf8 | DataType::LargeUtf8 => "LONGTEXT CHARACTER SET utf8mb4".into(),
            DataType::Binary | DataType::LargeBinary => "LONGBLOB".into(),
            DataType::Date32 => "DATE".into(),
            DataType::Timestamp(TimeUnit::Microsecond, None) => "DATETIME(6)".into(),
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => "TIMESTAMP(6)".into(),
            other => {
                return Err(ElmError::TypeMapping(format!(
                    "column '{}': Arrow type {other} has no lossless MySQL mapping",
                    field.name()
                )));
            }
        };
        Ok(Self { ddl })
    }
}

/// An atomic MySQL sink using an InnoDB staging table and per-batch LOCAL INFILE.
pub struct MySqlSink {
    connection: Conn,
    default_schema: String,
    tls_enabled: bool,
    target: Relation,
    staging: Relation,
    backup: Relation,
    batch_column: Identifier,
    marker: String,
    resume_rows: Option<u64>,
    resume_sequence: Option<u64>,
    schema: Option<Arc<Schema>>,
    columns: Vec<MySqlSinkColumn>,
    mode: Option<WriteMode>,
    target_existed: bool,
    staged_rows: u64,
    next_sequence: u64,
    pending_sequence: Option<u64>,
    local_infile_enabled: bool,
    begun: bool,
}

impl MySqlSink {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        target: Relation,
        job_id: JobId,
        resume_checkpoint: Option<&BatchCheckpoint>,
    ) -> Result<Self> {
        reject_mysql_catalog(&target)?;
        let (connection, tls_enabled) = connect_mysql(environment, password).await?;
        let compact_id = job_id.0.simple().to_string();
        let staging = Relation {
            catalog: None,
            schema: target.schema.clone(),
            name: Identifier::new(format!("elm_stage_{compact_id}"))?,
        };
        let backup = Relation {
            catalog: None,
            schema: target.schema.clone(),
            name: Identifier::new(format!("elm_backup_{compact_id}"))?,
        };
        let batch_column = Identifier::new(format!("elm_batch_{compact_id}"))?;
        let next_sequence = match resume_checkpoint {
            Some(checkpoint) => checkpoint
                .sequence
                .checked_add(1)
                .ok_or_else(|| ElmError::State("MySQL checkpoint sequence overflow".into()))?,
            None => 0,
        };
        Ok(Self {
            connection,
            default_schema: environment.database.clone(),
            tls_enabled,
            target,
            staging,
            backup,
            batch_column,
            marker: format!("{STAGING_COMMENT_PREFIX}{job_id}"),
            resume_rows: resume_checkpoint.map(|checkpoint| checkpoint.rows_committed),
            resume_sequence: resume_checkpoint.map(|checkpoint| checkpoint.sequence),
            schema: None,
            columns: Vec::new(),
            mode: None,
            target_existed: false,
            staged_rows: resume_checkpoint.map_or(0, |checkpoint| checkpoint.rows_committed),
            next_sequence,
            pending_sequence: None,
            local_infile_enabled: false,
            begun: false,
        })
    }

    #[must_use]
    pub fn staging_relation(&self) -> &Relation {
        &self.staging
    }

    async fn verify_staging_marker(&mut self) -> Result<()> {
        let marker =
            mysql_relation_comment(&mut self.connection, &self.staging, &self.default_schema)
                .await?;
        if marker.as_deref() != Some(&self.marker) {
            return Err(ElmError::Conflict(format!(
                "refusing to reuse or remove unrecognized MySQL relation {}",
                dialect_for(DatabaseKind::MySql).quote_relation(&self.staging)
            )));
        }
        Ok(())
    }

    async fn validate_existing_schema(
        &mut self,
        relation: &Relation,
        expected: &Schema,
    ) -> Result<()> {
        let query = format!(
            "SELECT {} FROM {} LIMIT 0",
            quoted_mysql_columns(expected)?,
            dialect_for(DatabaseKind::MySql).quote_relation(relation)
        );
        let statement = self
            .connection
            .prep(query)
            .await
            .map_err(mysql_metadata_error)?;
        let (actual, _) = schema_from_columns(statement.columns())?;
        self.connection
            .close(statement)
            .await
            .map_err(mysql_metadata_error)?;
        if actual.fields().len() != expected.fields().len() {
            return Err(ElmError::TypeMapping(
                "MySQL destination column count differs from the input".into(),
            ));
        }
        for (actual, expected) in actual.fields().iter().zip(expected.fields()) {
            let expected_type = normalized_mysql_arrow_type(expected.data_type());
            if actual.name() != expected.name()
                || actual.data_type() != &expected_type
                || actual.is_nullable() != expected.is_nullable()
            {
                return Err(ElmError::TypeMapping(format!(
                    "MySQL destination column '{}' does not exactly match input column '{}' ({}, nullable={} versus {}, nullable={})",
                    actual.name(),
                    expected.name(),
                    actual.data_type(),
                    actual.is_nullable(),
                    expected.data_type(),
                    expected.is_nullable()
                )));
            }
        }
        Ok(())
    }
}

/// Returns the committed row count without replaying a transfer. An existing but
/// inconclusive publication fence fails closed instead of reporting no publication.
pub async fn mysql_published_rows(
    environment: &Environment,
    password: &str,
    target: &Relation,
    job_id: JobId,
) -> Result<Option<u64>> {
    reject_mysql_catalog(target)?;
    let (mut connection, _) = connect_mysql(environment, password).await?;
    let staging = Relation {
        catalog: None,
        schema: target.schema.clone(),
        name: Identifier::new(format!("elm_stage_{}", job_id.0.simple()))?,
    };
    let publication = mysql_publication_relation(&staging)?;
    if !mysql_relation_exists(&mut connection, &publication, &environment.database).await? {
        return Ok(None);
    }
    let marker = format!("{STAGING_COMMENT_PREFIX}{job_id}");
    if mysql_relation_comment(&mut connection, &publication, &environment.database)
        .await?
        .as_deref()
        != Some(&marker)
    {
        return Err(ElmError::Conflict(
            "MySQL publication receipt has an unrecognized job marker".into(),
        ));
    }
    let receipt: Option<(String, String, u64, u8)> = connection
        .query_first(format!(
            "SELECT target_name, strategy, published_rows, committed FROM {} WHERE receipt_id = 1",
            dialect_for(DatabaseKind::MySql).quote_relation(&publication)
        ))
        .await
        .map_err(mysql_publication_error)?;
    let Some((target_name, strategy, rows, committed)) = receipt else {
        return Err(mysql_publication_error(()));
    };
    if target_name != target.name.as_str() {
        return Err(ElmError::Conflict(
            "MySQL publication receipt belongs to a different destination".into(),
        ));
    }
    if committed == 1 {
        return Ok(Some(rows));
    }
    // A rename preserves the stage's job comment. Its disappearance from the
    // private name and appearance at the target prove the atomic rename completed.
    if strategy == "rename"
        && !mysql_relation_exists(&mut connection, &staging, &environment.database).await?
        && mysql_relation_comment(&mut connection, target, &environment.database)
            .await?
            .as_deref()
            == Some(&marker)
    {
        return Ok(Some(rows));
    }
    Err(mysql_publication_error(()))
}

/// Removes only staging resources carrying this job's exact MySQL marker.
pub async fn cleanup_mysql_staging(
    environment: &Environment,
    password: &str,
    target: &Relation,
    job_id: JobId,
) -> Result<bool> {
    reject_mysql_catalog(target)?;
    let (mut connection, _tls_enabled) = connect_mysql(environment, password).await?;
    let staging = Relation {
        catalog: None,
        schema: target.schema.clone(),
        name: Identifier::new(format!("elm_stage_{}", job_id.0.simple()))?,
    };
    let publication = mysql_publication_relation(&staging)?;
    let expected = format!("{STAGING_COMMENT_PREFIX}{job_id}");
    let mut recognized = Vec::new();
    for resource in [&staging, &publication] {
        if !mysql_relation_exists(&mut connection, resource, &environment.database).await? {
            continue;
        }
        if mysql_relation_comment(&mut connection, resource, &environment.database)
            .await?
            .as_deref()
            != Some(&expected)
        {
            return Err(ElmError::Conflict(format!(
                "refusing to remove unrecognized MySQL relation {}",
                dialect_for(DatabaseKind::MySql).quote_relation(resource)
            )));
        }
        recognized.push(resource);
    }
    for resource in &recognized {
        connection
            .query_drop(format!(
                "DROP TABLE {}",
                dialect_for(DatabaseKind::MySql).quote_relation(resource)
            ))
            .await
            .map_err(mysql_staging_error)?;
    }
    Ok(!recognized.is_empty())
}

#[async_trait]
impl DataSink for MySqlSink {
    async fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        _consistency: ConsistencyMode,
    ) -> Result<PreflightReport> {
        if self.schema.is_some() {
            return Err(ElmError::Conflict(
                "MySQL sink has already been preflighted".into(),
            ));
        }
        if schema.fields().is_empty() {
            return Err(ElmError::TypeMapping(
                "MySQL destinations require at least one column".into(),
            ));
        }
        if schema
            .fields()
            .iter()
            .any(|field| field.name() == self.batch_column.as_str())
        {
            return Err(ElmError::Conflict(
                "input schema collides with the job-scoped MySQL checkpoint column".into(),
            ));
        }
        let columns = schema
            .fields()
            .iter()
            .map(|field| MySqlSinkColumn::from_field(field))
            .collect::<Result<Vec<_>>>()?;
        let target = self.target.clone();
        let target_existed =
            mysql_relation_exists(&mut self.connection, &target, &self.default_schema).await?;
        if mode == WriteMode::Fail && target_existed {
            return Err(ElmError::Conflict(format!(
                "destination {} already exists",
                dialect_for(DatabaseKind::MySql).quote_relation(&self.target)
            )));
        }
        if target_existed && mode == WriteMode::Append {
            let engine =
                mysql_relation_engine(&mut self.connection, &target, &self.default_schema).await?;
            if !engine.eq_ignore_ascii_case("InnoDB") {
                return Err(ElmError::Unsupported(
                    "atomic MySQL APPEND requires an InnoDB destination".into(),
                ));
            }
            self.validate_existing_schema(&target, &schema).await?;
        }
        self.local_infile_enabled = self
            .connection
            .query_first::<u8, _>("SELECT @@local_infile")
            .await
            .map_err(mysql_metadata_error)?
            .unwrap_or(0)
            != 0;

        self.schema = Some(schema);
        self.columns = columns;
        self.mode = Some(mode);
        self.target_existed = target_existed;
        let mut warnings = Vec::new();
        if !self.tls_enabled {
            warnings.push(
                "MySQL transport encryption is explicitly disabled for this environment".into(),
            );
        }
        if !self.local_infile_enabled {
            warnings.push(
                "MySQL LOCAL INFILE is disabled; the prepared-batch fallback was selected".into(),
            );
        }
        Ok(PreflightReport {
            capabilities: ConnectorCapabilities {
                database: Some(DatabaseKind::MySql),
                native_bulk_read: true,
                native_bulk_write: true,
                atomic_append: true,
                atomic_replace: true,
                non_atomic_replace: false,
                checkpointed_write: false,
                resumable_keyset_read: false,
                supports_lob_spill: false,
            },
            staging_relation: Some(self.staging.clone()),
            warnings,
        })
    }

    async fn begin(&mut self) -> Result<()> {
        if self.begun {
            return Err(ElmError::Conflict("MySQL sink has already begun".into()));
        }
        let schema = self
            .schema
            .clone()
            .ok_or_else(|| ElmError::Internal("MySQL sink was not preflighted".into()))?;
        let mode = self
            .mode
            .ok_or_else(|| ElmError::Internal("MySQL sink was not preflighted".into()))?;
        let target = self.target.clone();
        let staging = self.staging.clone();
        let publication = mysql_publication_relation(&staging)?;
        if mysql_relation_exists(&mut self.connection, &publication, &self.default_schema).await? {
            return Err(ElmError::Conflict(
                "MySQL publication was previously attempted; reconcile the destination before restarting this transfer".into(),
            ));
        }
        let target_now =
            mysql_relation_exists(&mut self.connection, &target, &self.default_schema).await?;
        if target_now != self.target_existed {
            return Err(ElmError::Conflict(
                "MySQL destination changed after preflight".into(),
            ));
        }
        if mode == WriteMode::Fail && target_now {
            return Err(ElmError::Conflict(
                "MySQL destination appeared after preflight".into(),
            ));
        }
        if mysql_relation_exists(&mut self.connection, &staging, &self.default_schema).await? {
            self.verify_staging_marker().await?;
            if let Some(expected_rows) = self.resume_rows {
                self.validate_existing_schema(&staging, &schema).await?;
                let sequence = self.resume_sequence.ok_or_else(|| {
                    ElmError::State("MySQL resume row count has no batch sequence".into())
                })?;
                self.connection
                    .exec_drop(
                        format!(
                            "DELETE FROM {} WHERE {} > ?",
                            dialect_for(DatabaseKind::MySql).quote_relation(&staging),
                            dialect_for(DatabaseKind::MySql).quote_identifier(&self.batch_column)
                        ),
                        (sequence,),
                    )
                    .await
                    .map_err(mysql_staging_error)?;
                let actual_rows = mysql_relation_row_count(&mut self.connection, &staging).await?;
                if actual_rows != expected_rows {
                    return Err(ElmError::Conflict(format!(
                        "MySQL staging has {actual_rows} rows but checkpoint expects {expected_rows}"
                    )));
                }
                self.begun = true;
                return Ok(());
            }
            self.connection
                .query_drop(format!(
                    "DROP TABLE {}",
                    dialect_for(DatabaseKind::MySql).quote_relation(&staging)
                ))
                .await
                .map_err(mysql_staging_error)?;
        } else if self.resume_rows.is_some() {
            return Err(ElmError::Conflict(
                "MySQL staging relation required by the checkpoint is missing".into(),
            ));
        }
        let create_sql = create_mysql_table_sql(
            &staging,
            &schema,
            &self.columns,
            &self.batch_column,
            &self.marker,
        )?;
        self.connection
            .query_drop(create_sql)
            .await
            .map_err(mysql_staging_error)?;
        self.begun = true;
        Ok(())
    }

    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if !self.begun {
            return Err(ElmError::Conflict("MySQL sink has not begun".into()));
        }
        if self.pending_sequence.is_some() {
            return Err(ElmError::State(
                "MySQL staging batch has not been checkpointed".into(),
            ));
        }
        let schema = self
            .schema
            .clone()
            .ok_or_else(|| ElmError::Internal("MySQL sink has no schema".into()))?;
        ensure_mysql_batch_schema(batch.schema().as_ref(), &schema)?;
        let sequence = self.next_sequence;
        let written = if self.local_infile_enabled {
            match load_mysql_batch(
                &mut self.connection,
                &self.staging,
                &self.batch_column,
                batch,
                sequence,
            )
            .await
            {
                Ok(written) => written,
                Err(error) if mysql_local_infile_unavailable(&error) => {
                    delete_mysql_batch_sequence(
                        &mut self.connection,
                        &self.staging,
                        &self.batch_column,
                        sequence,
                    )
                    .await?;
                    return Err(ElmError::Conflict(
                        "MySQL LOCAL INFILE policy changed after preflight; restart to negotiate the prepared fallback explicitly".into(),
                    ));
                }
                Err(_error) => {
                    delete_mysql_batch_sequence(
                        &mut self.connection,
                        &self.staging,
                        &self.batch_column,
                        sequence,
                    )
                    .await?;
                    return Err(ElmError::Connection {
                        message: "MySQL LOCAL INFILE batch failed".into(),
                        retryable: true,
                    });
                }
            }
        } else {
            insert_mysql_prepared_batch(
                &mut self.connection,
                &self.staging,
                &self.batch_column,
                batch,
                sequence,
            )
            .await?
        };
        let expected = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        if written != expected || self.connection.get_warnings() > 0 {
            delete_mysql_batch_sequence(
                &mut self.connection,
                &self.staging,
                &self.batch_column,
                sequence,
            )
            .await?;
            return Err(ElmError::TypeMapping(format!(
                "MySQL accepted {written} of {expected} rows or reported conversion warnings; the batch was removed"
            )));
        }
        self.staged_rows = self.staged_rows.saturating_add(written);
        self.pending_sequence = Some(sequence);
        Ok(())
    }

    async fn commit_checkpoint(&mut self, checkpoint: &BatchCheckpoint) -> Result<()> {
        if self.pending_sequence != Some(checkpoint.sequence) {
            return Err(ElmError::State(format!(
                "MySQL sink expected checkpoint sequence {}, received {}",
                self.next_sequence, checkpoint.sequence
            )));
        }
        if checkpoint.rows_committed != self.staged_rows {
            return Err(ElmError::State(format!(
                "MySQL staging has committed {} rows but checkpoint records {}",
                self.staged_rows, checkpoint.rows_committed
            )));
        }
        self.pending_sequence = None;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| ElmError::State("MySQL checkpoint sequence overflow".into()))?;
        Ok(())
    }

    async fn publish(&mut self) -> Result<()> {
        if !self.begun {
            return Err(ElmError::Conflict("MySQL sink has not begun".into()));
        }
        if self.pending_sequence.is_some() {
            return Err(ElmError::State(
                "MySQL staging batch has not been checkpointed".into(),
            ));
        }
        let mode = self
            .mode
            .ok_or_else(|| ElmError::Internal("MySQL sink has no write mode".into()))?;
        let schema = self
            .schema
            .clone()
            .ok_or_else(|| ElmError::Internal("MySQL sink has no schema".into()))?;
        let target = self.target.clone();
        let staging = self.staging.clone();
        let backup = self.backup.clone();
        let actual_rows = mysql_relation_row_count(&mut self.connection, &staging).await?;
        if actual_rows != self.staged_rows {
            return Err(ElmError::Conflict(format!(
                "MySQL staging row count changed from {} to {actual_rows} before publication",
                self.staged_rows
            )));
        }
        let target_exists_now =
            mysql_relation_exists(&mut self.connection, &target, &self.default_schema).await?;
        if target_exists_now != self.target_existed {
            return Err(ElmError::Conflict(
                "MySQL destination changed before publication".into(),
            ));
        }
        let dialect = dialect_for(DatabaseKind::MySql);
        let stage_sql = dialect.quote_relation(&staging);
        let target_sql = dialect.quote_relation(&target);
        let data_columns = quoted_mysql_columns(&schema)?;
        // DDL commits this fence before any target mutation. It survives lost COMMIT
        // acknowledgements and prevents a restarted job from publishing twice.
        let publication = mysql_publication_relation(&staging)?;
        self.connection
            .query_drop(format!(
                "CREATE TABLE {} (receipt_id TINYINT PRIMARY KEY, target_name VARCHAR(64) NOT NULL, strategy VARCHAR(6) NOT NULL, published_rows BIGINT UNSIGNED NOT NULL, committed TINYINT NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin COMMENT={}",
                dialect.quote_relation(&publication),
                mysql_string_literal(&self.marker)
            ))
            .await
            .map_err(mysql_publication_error)?;
        let publication_sql = dialect.quote_relation(&publication);
        let strategy = if mode == WriteMode::Append && self.target_existed {
            "append"
        } else {
            "rename"
        };
        self.connection
            .exec_drop(
                format!("INSERT INTO {publication_sql} VALUES (1, ?, ?, ?, 0)"),
                (target.name.as_str(), strategy, self.staged_rows),
            )
            .await
            .map_err(mysql_publication_error)?;
        if mode == WriteMode::Append && self.target_existed {
            self.connection
                .query_drop("START TRANSACTION")
                .await
                .map_err(mysql_publication_error)?;
            if self
                .connection
                .query_drop(format!(
                    "INSERT INTO {target_sql} ({data_columns}) SELECT {data_columns} FROM {stage_sql}"
                ))
                .await
                .is_err()
                || self.connection.get_warnings() > 0
            {
                let _rollback = self.connection.query_drop("ROLLBACK").await;
                return Err(mysql_publication_error(()));
            }
            if self
                .connection
                .query_drop(format!(
                    "UPDATE {publication_sql} SET committed = 1 WHERE receipt_id = 1"
                ))
                .await
                .is_err()
            {
                let _rollback = self.connection.query_drop("ROLLBACK").await;
                return Err(mysql_publication_error(()));
            }
            self.connection
                .query_drop("COMMIT")
                .await
                .map_err(mysql_publication_error)?;
            let _cleanup = self
                .connection
                .query_drop(format!("DROP TABLE {stage_sql}"))
                .await;
            self.begun = false;
            return Ok(());
        }

        self.connection
            .query_drop(format!(
                "ALTER TABLE {stage_sql} DROP COLUMN {}",
                dialect.quote_identifier(&self.batch_column)
            ))
            .await
            .map_err(mysql_publication_error)?;
        if mode == WriteMode::Replace && self.target_existed {
            if mysql_relation_exists(&mut self.connection, &backup, &self.default_schema).await? {
                return Err(ElmError::Conflict(
                    "MySQL recovery backup relation already exists".into(),
                ));
            }
            self.connection
                .query_drop(format!(
                    "RENAME TABLE {target_sql} TO {}, {stage_sql} TO {target_sql}",
                    dialect.quote_relation(&backup)
                ))
                .await
                .map_err(mysql_publication_error)?;
            let _cleanup = self
                .connection
                .query_drop(format!("DROP TABLE {}", dialect.quote_relation(&backup)))
                .await;
        } else {
            self.connection
                .query_drop(format!("RENAME TABLE {stage_sql} TO {target_sql}"))
                .await
                .map_err(mysql_publication_error)?;
        }
        self.connection
            .query_drop(format!(
                "UPDATE {publication_sql} SET committed = 1 WHERE receipt_id = 1"
            ))
            .await
            .map_err(mysql_publication_error)?;
        self.begun = false;
        Ok(())
    }

    async fn abort(&mut self) -> Result<()> {
        // Completed batches remain in the marked staging table for checkpoint reconciliation.
        Ok(())
    }
}

fn normalized_mysql_arrow_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Boolean => DataType::Int8,
        DataType::LargeUtf8 => DataType::Utf8,
        DataType::LargeBinary => DataType::Binary,
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        other => other.clone(),
    }
}

fn create_mysql_table_sql(
    relation: &Relation,
    schema: &Schema,
    columns: &[MySqlSinkColumn],
    batch_column: &Identifier,
    marker: &str,
) -> Result<String> {
    let dialect = dialect_for(DatabaseKind::MySql);
    let mut definitions = schema
        .fields()
        .iter()
        .zip(columns)
        .map(|(field, column)| {
            let identifier = Identifier::new(field.name())?;
            Ok(format!(
                "{} {}{}",
                dialect.quote_identifier(&identifier),
                column.ddl,
                if field.is_nullable() { "" } else { " NOT NULL" }
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    definitions.push(format!(
        "{} BIGINT UNSIGNED NOT NULL",
        dialect.quote_identifier(batch_column)
    ));
    Ok(format!(
        "CREATE TABLE {} ({}) ENGINE=InnoDB COMMENT={}",
        dialect.quote_relation(relation),
        definitions.join(", "),
        mysql_string_literal(marker)
    ))
}

fn quoted_mysql_columns(schema: &Schema) -> Result<String> {
    let dialect = dialect_for(DatabaseKind::MySql);
    schema
        .fields()
        .iter()
        .map(|field| Ok(dialect.quote_identifier(&Identifier::new(field.name())?)))
        .collect::<Result<Vec<_>>>()
        .map(|columns| columns.join(", "))
}

fn quoted_mysql_columns_with_batch(schema: &Schema, batch_column: &Identifier) -> Result<String> {
    let mut columns = quoted_mysql_columns(schema)?;
    columns.push_str(", ");
    columns.push_str(&dialect_for(DatabaseKind::MySql).quote_identifier(batch_column));
    Ok(columns)
}

async fn load_mysql_batch(
    connection: &mut Conn,
    staging: &Relation,
    batch_column: &Identifier,
    batch: &RecordBatch,
    sequence: u64,
) -> std::result::Result<u64, MySqlError> {
    let input = batch.clone();
    connection.set_infile_handler(async move {
        Ok(
            stream::try_unfold((input, 0), move |(input, mut offset)| async move {
                if offset == input.num_rows() {
                    return Ok::<_, std::io::Error>(None);
                }
                let payload = encode_mysql_load_chunk(&input, sequence, &mut offset)
                    .map_err(std::io::Error::other)?;
                Ok(Some((Bytes::from(payload), (input, offset))))
            })
            .boxed(),
        )
    });
    let sql = format!(
        "LOAD DATA LOCAL INFILE 'elm-tool-memory' INTO TABLE {} CHARACTER SET binary FIELDS TERMINATED BY '\\t' ESCAPED BY '\\\\' LINES TERMINATED BY '\\n' ({})",
        dialect_for(DatabaseKind::MySql).quote_relation(staging),
        quoted_mysql_columns_with_batch(batch.schema().as_ref(), batch_column)
            .map_err(|error| MySqlError::Other(Box::new(error)))?
    );
    connection.query_drop(sql).await?;
    Ok(connection.affected_rows())
}

async fn insert_mysql_prepared_batch(
    connection: &mut Conn,
    staging: &Relation,
    batch_column: &Identifier,
    batch: &RecordBatch,
    sequence: u64,
) -> Result<u64> {
    let placeholders = std::iter::repeat_n("?", batch.num_columns().saturating_add(1))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "INSERT INTO {} ({}) VALUES ({placeholders})",
        dialect_for(DatabaseKind::MySql).quote_relation(staging),
        quoted_mysql_columns_with_batch(batch.schema().as_ref(), batch_column)?
    );
    let statement = connection.prep(sql).await.map_err(mysql_staging_error)?;
    connection
        .query_drop("START TRANSACTION")
        .await
        .map_err(mysql_staging_error)?;
    for row in 0..batch.num_rows() {
        let mut values = match mysql_values(batch, row) {
            Ok(values) => values,
            Err(error) => {
                let _rollback = connection.query_drop("ROLLBACK").await;
                return Err(error);
            }
        };
        values.push(Value::UInt(sequence));
        let result = connection
            .exec_drop(&statement, Params::Positional(values))
            .await;
        if result.is_ok() && connection.get_warnings() == 0 {
            continue;
        }
        let _rollback = connection.query_drop("ROLLBACK").await;
        return Err(ElmError::TypeMapping(
            "MySQL prepared staging batch failed or reported conversion warnings; the batch was rolled back".into(),
        ));
    }
    connection
        .close(statement)
        .await
        .map_err(mysql_staging_error)?;
    connection
        .query_drop("COMMIT")
        .await
        .map_err(mysql_staging_error)?;
    Ok(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX))
}

fn encode_mysql_load_chunk(
    batch: &RecordBatch,
    sequence: u64,
    offset: &mut usize,
) -> Result<Vec<u8>> {
    const TARGET_BYTES: usize = 64 * 1024;
    let mut output = Vec::with_capacity(TARGET_BYTES);
    while *offset < batch.num_rows() && output.len() < TARGET_BYTES {
        let row = *offset;
        for column in 0..batch.num_columns() {
            if column > 0 {
                output.push(b'\t');
            }
            let value = mysql_value(batch.column(column), row, batch.schema().field(column))?;
            encode_mysql_load_value(&value, &mut output)?;
        }
        output.push(b'\t');
        output.extend_from_slice(sequence.to_string().as_bytes());
        output.push(b'\n');
        *offset += 1;
    }
    Ok(output)
}

fn encode_mysql_load_value(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::NULL => output.extend_from_slice(b"\\N"),
        Value::Bytes(value) => escape_mysql_load_bytes(value, output),
        Value::Int(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::UInt(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::Float(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::Double(value) => output.extend_from_slice(value.to_string().as_bytes()),
        Value::Date(year, month, day, hour, minute, second, micros) => {
            let value = if *hour == 0 && *minute == 0 && *second == 0 && *micros == 0 {
                format!("{year:04}-{month:02}-{day:02}")
            } else {
                format!(
                    "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}.{micros:06}"
                )
            };
            output.extend_from_slice(value.as_bytes());
        }
        Value::Time(..) => {
            return Err(ElmError::Internal(
                "MySQL TIME reached a sink that does not support it".into(),
            ));
        }
    }
    Ok(())
}

fn escape_mysql_load_bytes(value: &[u8], output: &mut Vec<u8>) {
    for byte in value {
        match byte {
            0 => output.extend_from_slice(b"\\0"),
            b'\n' => output.extend_from_slice(b"\\n"),
            b'\r' => output.extend_from_slice(b"\\r"),
            b'\t' => output.extend_from_slice(b"\\t"),
            b'\\' => output.extend_from_slice(b"\\\\"),
            0x1a => output.extend_from_slice(b"\\Z"),
            byte => output.push(*byte),
        }
    }
}

fn mysql_values(batch: &RecordBatch, row: usize) -> Result<Vec<Value>> {
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| mysql_value(array, row, field))
        .collect()
}

fn mysql_value(array: &ArrayRef, row: usize, field: &Field) -> Result<Value> {
    if array.is_null(row) {
        return Ok(Value::NULL);
    }
    Ok(match field.data_type() {
        DataType::Boolean => Value::Int(i64::from(
            array_as::<BooleanArray>(array, field)?.value(row),
        )),
        DataType::Int8 => Value::Int(i64::from(array_as::<Int8Array>(array, field)?.value(row))),
        DataType::Int16 => Value::Int(i64::from(array_as::<Int16Array>(array, field)?.value(row))),
        DataType::Int32 => Value::Int(i64::from(array_as::<Int32Array>(array, field)?.value(row))),
        DataType::Int64 => Value::Int(array_as::<Int64Array>(array, field)?.value(row)),
        DataType::UInt8 => Value::UInt(u64::from(array_as::<UInt8Array>(array, field)?.value(row))),
        DataType::UInt16 => {
            Value::UInt(u64::from(array_as::<UInt16Array>(array, field)?.value(row)))
        }
        DataType::UInt32 => {
            Value::UInt(u64::from(array_as::<UInt32Array>(array, field)?.value(row)))
        }
        DataType::UInt64 => Value::UInt(array_as::<UInt64Array>(array, field)?.value(row)),
        DataType::Float32 => Value::Float(array_as::<Float32Array>(array, field)?.value(row)),
        DataType::Float64 => Value::Double(array_as::<Float64Array>(array, field)?.value(row)),
        DataType::Decimal128(precision, scale) => {
            let value = array_as::<Decimal128Array>(array, field)?.value(row);
            ensure_decimal_precision(value, *precision)?;
            Value::Bytes(format_decimal_coefficient(value, *scale)?.into_bytes())
        }
        DataType::Utf8 => Value::Bytes(
            array_as::<StringArray>(array, field)?
                .value(row)
                .as_bytes()
                .to_vec(),
        ),
        DataType::LargeUtf8 => Value::Bytes(
            array_as::<LargeStringArray>(array, field)?
                .value(row)
                .as_bytes()
                .to_vec(),
        ),
        DataType::Binary => {
            Value::Bytes(array_as::<BinaryArray>(array, field)?.value(row).to_vec())
        }
        DataType::LargeBinary => Value::Bytes(
            array_as::<LargeBinaryArray>(array, field)?
                .value(row)
                .to_vec(),
        ),
        DataType::Date32 => {
            let date = epoch_days_to_date(array_as::<Date32Array>(array, field)?.value(row))?;
            Value::Date(
                u16::try_from(date.year()).map_err(|_| {
                    ElmError::TypeMapping(format!(
                        "column '{}' contains a date outside MySQL's year range",
                        field.name()
                    ))
                })?,
                u8::try_from(date.month()).unwrap_or(0),
                u8::try_from(date.day()).unwrap_or(0),
                0,
                0,
                0,
                0,
            )
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let instant = chrono::DateTime::<Utc>::from_timestamp_micros(
                array_as::<TimestampMicrosecondArray>(array, field)?.value(row),
            )
            .ok_or_else(|| {
                ElmError::TypeMapping(format!(
                    "column '{}' contains an out-of-range timestamp",
                    field.name()
                ))
            })?;
            Value::Date(
                u16::try_from(instant.year()).map_err(|_| {
                    ElmError::TypeMapping(format!(
                        "column '{}' contains a timestamp outside MySQL's year range",
                        field.name()
                    ))
                })?,
                u8::try_from(instant.month()).unwrap_or(0),
                u8::try_from(instant.day()).unwrap_or(0),
                u8::try_from(instant.hour()).unwrap_or(0),
                u8::try_from(instant.minute()).unwrap_or(0),
                u8::try_from(instant.second()).unwrap_or(0),
                instant.timestamp_subsec_micros(),
            )
        }
        other => {
            return Err(ElmError::TypeMapping(format!(
                "column '{}': Arrow type {other} has no MySQL value encoder",
                field.name()
            )));
        }
    })
}

fn array_as<'a, T: Array + 'static>(array: &'a ArrayRef, field: &Field) -> Result<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        ElmError::Internal(format!(
            "column '{}' array does not match its Arrow schema",
            field.name()
        ))
    })
}

fn format_decimal_coefficient(value: i128, scale: i8) -> Result<String> {
    let scale = usize::try_from(scale).map_err(|_| {
        ElmError::TypeMapping("negative Decimal128 scales cannot be written to MySQL".into())
    })?;
    let negative = value.is_negative();
    let digits = value
        .checked_abs()
        .ok_or_else(|| ElmError::TypeMapping("Decimal128 value is out of range".into()))?
        .to_string();
    let mut output = String::new();
    if negative {
        output.push('-');
    }
    if scale == 0 {
        output.push_str(&digits);
    } else if digits.len() <= scale {
        output.push_str("0.");
        output.push_str(&"0".repeat(scale - digits.len()));
        output.push_str(&digits);
    } else {
        let split = digits.len() - scale;
        output.push_str(&digits[..split]);
        output.push('.');
        output.push_str(&digits[split..]);
    }
    Ok(output)
}

fn epoch_days_to_date(days: i32) -> Result<NaiveDate> {
    unix_epoch()
        .checked_add_signed(chrono::Duration::days(i64::from(days)))
        .ok_or_else(|| ElmError::TypeMapping("Arrow Date32 is outside MySQL's range".into()))
}

fn ensure_mysql_batch_schema(actual: &Schema, expected: &Schema) -> Result<()> {
    if actual != expected {
        return Err(ElmError::TypeMapping(
            "MySQL batch schema changed after sink preflight".into(),
        ));
    }
    Ok(())
}

async fn delete_mysql_batch_sequence(
    connection: &mut Conn,
    staging: &Relation,
    batch_column: &Identifier,
    sequence: u64,
) -> Result<()> {
    connection
        .exec_drop(
            format!(
                "DELETE FROM {} WHERE {} = ?",
                dialect_for(DatabaseKind::MySql).quote_relation(staging),
                dialect_for(DatabaseKind::MySql).quote_identifier(batch_column)
            ),
            (sequence,),
        )
        .await
        .map_err(mysql_staging_error)
}

fn mysql_local_infile_unavailable(error: &MySqlError) -> bool {
    matches!(error, MySqlError::Server(server) if matches!(server.code, 1148 | 3948 | 4166))
}

fn mysql_publication_relation(staging: &Relation) -> Result<Relation> {
    Ok(Relation {
        catalog: staging.catalog.clone(),
        schema: staging.schema.clone(),
        name: Identifier::new(format!("{}_publish", staging.name.as_str()))?,
    })
}

async fn mysql_relation_exists(
    connection: &mut Conn,
    relation: &Relation,
    default_schema: &str,
) -> Result<bool> {
    let schema = relation
        .schema
        .as_ref()
        .map_or(default_schema, Identifier::as_str);
    connection
        .exec_first::<u8, _, _>(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
            (schema, relation.name.as_str()),
        )
        .await
        .map(|value| value.is_some())
        .map_err(mysql_metadata_error)
}

async fn mysql_relation_comment(
    connection: &mut Conn,
    relation: &Relation,
    default_schema: &str,
) -> Result<Option<String>> {
    let schema = relation
        .schema
        .as_ref()
        .map_or(default_schema, Identifier::as_str);
    connection
        .exec_first(
            "SELECT table_comment FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
            (schema, relation.name.as_str()),
        )
        .await
        .map_err(mysql_metadata_error)
}

async fn mysql_relation_engine(
    connection: &mut Conn,
    relation: &Relation,
    default_schema: &str,
) -> Result<String> {
    let schema = relation
        .schema
        .as_ref()
        .map_or(default_schema, Identifier::as_str);
    connection
        .exec_first(
            "SELECT engine FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
            (schema, relation.name.as_str()),
        )
        .await
        .map_err(mysql_metadata_error)?
        .ok_or_else(|| ElmError::Conflict("MySQL destination disappeared during preflight".into()))
}

async fn mysql_relation_row_count(connection: &mut Conn, relation: &Relation) -> Result<u64> {
    connection
        .query_first(format!(
            "SELECT COUNT(*) FROM {}",
            dialect_for(DatabaseKind::MySql).quote_relation(relation)
        ))
        .await
        .map_err(mysql_metadata_error)?
        .ok_or_else(|| ElmError::State("MySQL row-count query returned no value".into()))
}

fn reject_mysql_catalog(relation: &Relation) -> Result<()> {
    if relation.catalog.is_some() {
        return Err(ElmError::Validation(
            "MySQL relations cannot contain a catalog; use schema for a database name".into(),
        ));
    }
    Ok(())
}

fn mysql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "''"))
}

fn mysql_connection_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "MySQL connection, authentication, or TLS verification failed".into(),
        retryable: true,
    }
}

fn mysql_metadata_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "MySQL metadata query failed".into(),
        retryable: true,
    }
}

fn mysql_staging_error<T>(_error: T) -> ElmError {
    ElmError::PermissionDenied(
        "MySQL staging table could not be created, written, or cleaned with this account".into(),
    )
}

fn mysql_publication_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "MySQL publication could not be confirmed; reconcile the destination before retrying because publication may have committed".into(),
        retryable: false,
    }
}

fn mysql_source_preflight_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "MySQL source query preflight failed".into(),
        retryable: false,
    }
}

fn mysql_source_start_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "MySQL source query could not start".into(),
        retryable: true,
    }
}

#[cfg(test)]
mod tests {
    use mysql_async::{
        Column,
        consts::{ColumnFlags, ColumnType},
    };

    use super::{mysql_column_to_arrow, parse_mysql_decimal};
    use arrow_schema::DataType;

    #[test]
    fn maps_signed_unsigned_binary_and_decimal_metadata() {
        let signed = Column::new(ColumnType::MYSQL_TYPE_LONG).with_name(b"signed");
        assert_eq!(
            mysql_column_to_arrow(&signed).map(|mapped| mapped.0),
            Ok(DataType::Int32)
        );
        let unsigned = Column::new(ColumnType::MYSQL_TYPE_LONGLONG)
            .with_name(b"unsigned")
            .with_flags(ColumnFlags::UNSIGNED_FLAG);
        assert_eq!(
            mysql_column_to_arrow(&unsigned).map(|mapped| mapped.0),
            Ok(DataType::UInt64)
        );
        let binary = Column::new(ColumnType::MYSQL_TYPE_BLOB)
            .with_name(b"payload")
            .with_character_set(63);
        assert_eq!(
            mysql_column_to_arrow(&binary).map(|mapped| mapped.0),
            Ok(DataType::Binary)
        );
        let binary_collation_text = Column::new(ColumnType::MYSQL_TYPE_VAR_STRING)
            .with_character_set(46)
            .with_flags(ColumnFlags::BINARY_FLAG);
        assert_eq!(
            mysql_column_to_arrow(&binary_collation_text).map(|mapped| mapped.0),
            Ok(DataType::Utf8)
        );
        let enumeration =
            Column::new(ColumnType::MYSQL_TYPE_STRING).with_flags(ColumnFlags::ENUM_FLAG);
        assert!(mysql_column_to_arrow(&enumeration).is_err());
        let decimal = Column::new(ColumnType::MYSQL_TYPE_NEWDECIMAL)
            .with_name(b"amount")
            .with_column_length(20)
            .with_decimals(4);
        assert_eq!(
            mysql_column_to_arrow(&decimal).map(|mapped| mapped.0),
            Ok(DataType::Decimal128(18, 4))
        );
    }

    #[test]
    fn parses_mysql_decimal_without_floating_point() {
        assert_eq!(
            parse_mysql_decimal(b"-12345.6700", 4).unwrap_or_else(|error| panic!("{error}")),
            -123_456_700
        );
        assert_eq!(
            parse_mysql_decimal(b"0.0012", 4).unwrap_or_else(|error| panic!("{error}")),
            12
        );
        assert_eq!(
            parse_mysql_decimal(b"42", 3).unwrap_or_else(|error| panic!("{error}")),
            42_000
        );
        assert!(parse_mysql_decimal(b"1.234", 2).is_err());
    }

    #[test]
    fn local_infile_chunks_preserve_rows_escaping_and_sequence() {
        use arrow_array::{RecordBatch, StringArray};
        use arrow_schema::{Field, Schema};
        use std::sync::Arc;

        let text = "İ\t\\\n".repeat(10_000);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("label", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![
                Some(text.as_str()),
                None,
                Some("end"),
            ]))],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let mut offset = 0;
        let first = super::encode_mysql_load_chunk(&batch, 7, &mut offset)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(offset, 1);
        assert_eq!(
            first,
            format!("{}\t7\n", "İ\\t\\\\\\n".repeat(10_000)).as_bytes()
        );
        let last = super::encode_mysql_load_chunk(&batch, 7, &mut offset)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(offset, 3);
        assert_eq!(last, b"\\N\t7\nend\t7\n");
    }

    #[tokio::test]
    #[ignore = "requires an isolated MySQL test instance"]
    async fn prepared_insert_rolls_back_early_row_conversion_warnings() {
        use arrow_array::{RecordBatch, StringArray};
        use arrow_schema::{Field, Schema};
        use elm_core::{Identifier, Relation};
        use mysql_async::{Conn, OptsBuilder, prelude::Queryable};
        use std::sync::Arc;

        let options = OptsBuilder::default()
            .ip_or_hostname("127.0.0.1")
            .tcp_port(
                std::env::var("ELM_TEST_MYSQL_PORT")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(53306),
            )
            .user(Some("root"))
            .pass(Some(
                std::env::var("ELM_TEST_MYSQL_PASSWORD")
                    .unwrap_or_else(|_| "elm_integration_only".into()),
            ))
            .db_name(Some("elm_test"));
        let mut connection = Conn::new(options)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        connection
            .query_drop("SET SESSION sql_mode = ''")
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        connection.query_drop("CREATE TEMPORARY TABLE elm_warning_test (label VARCHAR(3), batch BIGINT UNSIGNED) ENGINE=InnoDB")
            .await.unwrap_or_else(|error| panic!("{error}"));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "label",
                DataType::Utf8,
                false,
            )])),
            vec![Arc::new(StringArray::from(vec!["too long", "ok"]))],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let relation = Relation {
            catalog: None,
            schema: None,
            name: Identifier::new("elm_warning_test").unwrap_or_else(|error| panic!("{error}")),
        };
        let result = super::insert_mysql_prepared_batch(
            &mut connection,
            &relation,
            &Identifier::new("batch").unwrap_or_else(|error| panic!("{error}")),
            &batch,
            0,
        )
        .await;
        assert!(matches!(result, Err(elm_core::ElmError::TypeMapping(_))));
        let count: Option<u64> = connection
            .query_first("SELECT COUNT(*) FROM elm_warning_test")
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(count, Some(0));
        connection
            .disconnect()
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    }
}
