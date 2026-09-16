use std::{collections::BTreeSet, pin::Pin, sync::Arc};

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
    RecordBatch, StringArray, TimestampMicrosecondArray,
    builder::{
        BinaryBuilder, BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder,
        Float64Builder, Int16Builder, Int32Builder, Int64Builder, StringBuilder,
        TimestampMicrosecondBuilder,
    },
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use bytes::{BufMut, BytesMut};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, Utc};
use futures::StreamExt;
use native_tls::{Certificate, TlsConnector};
use postgres_native_tls::MakeTlsConnector;
use postgres_types::{FromSql, IsNull, ToSql, Type};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tokio_postgres::{
    Client, GenericClient, NoTls, Row,
    binary_copy::{BinaryCopyInWriter, BinaryCopyOutRow, BinaryCopyOutStream},
    config::SslMode,
};

use elm_core::{
    BatchCheckpoint, ConnectorCapabilities, ConsistencyMode, DataSink, DataSource, DatabaseKind,
    DatabaseSelection, ElmError, Environment, Identifier, JobId, PreflightReport, Relation, Result,
    WriteMode,
};

use crate::dialect_for;

const STAGING_COMMENT_PREFIX: &str = "elm-tool staging job:";

struct PostgresSession {
    client: Client,
    connection: JoinHandle<()>,
    tls_enabled: bool,
}

impl Drop for PostgresSession {
    fn drop(&mut self) {
        self.connection.abort();
    }
}

async fn connect(environment: &Environment, password: &str) -> Result<PostgresSession> {
    environment.validate()?;
    if environment.kind != DatabaseKind::PostgreSql {
        return Err(ElmError::Validation(format!(
            "environment '{}' is not PostgreSQL",
            environment.name
        )));
    }
    let ssl_mode = environment
        .options
        .get("ssl_mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("require");

    let mut configuration = tokio_postgres::Config::new();
    configuration
        .host(&environment.host)
        .port(environment.port)
        .dbname(&environment.database)
        .user(&environment.username)
        .password(password)
        .application_name("elm-tool-v2");
    let (client, connection, tls_enabled) = match ssl_mode {
        "disable" => {
            configuration.ssl_mode(SslMode::Disable);
            let (client, connection) = configuration
                .connect(NoTls)
                .await
                .map_err(postgres_connection_error)?;
            let task = tokio::spawn(async move {
                let _connection_result = connection.await;
            });
            (client, task, false)
        }
        "require" => {
            configuration.ssl_mode(SslMode::Require);
            let mut builder = TlsConnector::builder();
            if let Some(path) = environment
                .options
                .get("root_certificate")
                .and_then(serde_json::Value::as_str)
            {
                let pem = std::fs::read(path).map_err(|_| ElmError::Connection {
                    message: "PostgreSQL root certificate could not be read".into(),
                    retryable: false,
                })?;
                let certificate =
                    Certificate::from_pem(&pem).map_err(|_| ElmError::Connection {
                        message: "PostgreSQL root certificate is not valid PEM".into(),
                        retryable: false,
                    })?;
                builder.add_root_certificate(certificate);
            }
            let connector = builder.build().map_err(postgres_connection_error)?;
            let (client, connection) = configuration
                .connect(MakeTlsConnector::new(connector))
                .await
                .map_err(postgres_connection_error)?;
            let task = tokio::spawn(async move {
                let _connection_result = connection.await;
            });
            (client, task, true)
        }
        _ => {
            return Err(ElmError::Validation(
                "PostgreSQL ssl_mode must be 'require' or 'disable'".into(),
            ));
        }
    };
    Ok(PostgresSession {
        client,
        connection,
        tls_enabled,
    })
}

/// Performs an authenticated connection and a round-trip query without returning server details.
pub async fn test_postgres_environment(environment: &Environment, password: &str) -> Result<()> {
    let session = connect(environment, password).await?;
    session
        .client
        .simple_query("SELECT 1")
        .await
        .map_err(|_| ElmError::Connection {
            message: "PostgreSQL diagnostic query failed".into(),
            retryable: true,
        })?;
    Ok(())
}

pub struct PostgresSource {
    schema: Arc<Schema>,
    stream: Pin<Box<BinaryCopyOutStream>>,
    column_types: Vec<Type>,
    rows_read: u64,
    exhausted: bool,
    keyset: Option<PostgresKeysetState>,
    // Kept last so an active COPY stream is dropped before its connection task is aborted.
    _session: PostgresSession,
}

impl PostgresSource {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        selection: &DatabaseSelection,
    ) -> Result<Self> {
        Self::connect_resumable(environment, password, selection, &[], None).await
    }

    pub async fn connect_resumable(
        environment: &Environment,
        password: &str,
        selection: &DatabaseSelection,
        resume_key: &[Identifier],
        checkpoint: Option<&serde_json::Value>,
    ) -> Result<Self> {
        let session = connect(environment, password).await?;
        let (query, keyset) = match (selection, resume_key.is_empty()) {
            (_, true) => {
                if checkpoint.is_some() {
                    return Err(ElmError::State(
                        "PostgreSQL source checkpoint has no configured resume key".into(),
                    ));
                }
                (selection_query(selection)?, None)
            }
            (DatabaseSelection::Table { relation }, false) => {
                build_keyset_query(&session.client, relation, resume_key, checkpoint).await?
            }
            (DatabaseSelection::Query { .. }, false) => {
                return Err(ElmError::Validation(
                    "PostgreSQL keyset resume is available only for structured table sources"
                        .into(),
                ));
            }
        };
        let statement = session
            .client
            .prepare(&query)
            .await
            .map_err(|_| ElmError::Connection {
                message: "PostgreSQL source query preflight failed".into(),
                retryable: false,
            })?;
        if statement.columns().is_empty() {
            return Err(ElmError::Validation(
                "PostgreSQL source query must return at least one column".into(),
            ));
        }

        let mut names = BTreeSet::new();
        let mut fields = Vec::with_capacity(statement.columns().len());
        let mut column_types = Vec::with_capacity(statement.columns().len());
        for column in statement.columns() {
            if !names.insert(column.name().to_owned()) {
                return Err(ElmError::TypeMapping(format!(
                    "PostgreSQL source returned duplicate column '{}'",
                    column.name()
                )));
            }
            let data_type = postgres_type_to_arrow(column.type_(), column.type_modifier())
                .map_err(|error| {
                    ElmError::TypeMapping(format!("column '{}': {error}", column.name()))
                })?;
            fields.push(Field::new(column.name(), data_type, true));
            column_types.push(column.type_().clone());
        }
        let copy_sql = format!("COPY ({query}) TO STDOUT (FORMAT BINARY)");
        let copy = session
            .client
            .copy_out(&copy_sql)
            .await
            .map_err(|_| ElmError::Connection {
                message: "PostgreSQL binary COPY source could not start".into(),
                retryable: true,
            })?;
        let stream = Box::pin(BinaryCopyOutStream::new(copy, &column_types));
        Ok(Self {
            schema: Arc::new(Schema::new(fields)),
            stream,
            column_types,
            rows_read: 0,
            exhausted: false,
            keyset,
            _session: session,
        })
    }
}

#[async_trait]
impl DataSource for PostgresSource {
    async fn schema(&mut self) -> Result<Arc<Schema>> {
        Ok(self.schema.clone())
    }

    async fn next_batch(&mut self, target_bytes: usize) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        let capacity = (target_bytes / 128).clamp(1, 65_536);
        let mut builders = self
            .column_types
            .iter()
            .zip(self.schema.fields())
            .map(|(type_, field)| PgBuilder::new(type_, field.data_type(), capacity))
            .collect::<Result<Vec<_>>>()?;
        let mut rows = 0_usize;
        let mut estimated_bytes = 0_usize;

        while estimated_bytes < target_bytes.max(1) {
            let Some(row) = self.stream.next().await else {
                self.exhausted = true;
                break;
            };
            let row = row.map_err(|_| ElmError::Connection {
                message: "PostgreSQL binary COPY source was interrupted".into(),
                retryable: true,
            })?;
            for (index, builder) in builders.iter_mut().enumerate() {
                estimated_bytes = estimated_bytes.saturating_add(builder.append(&row, index)?);
            }
            if let Some(keyset) = &mut self.keyset {
                keyset.checkpoint.last_key = Some(
                    keyset
                        .column_indices
                        .iter()
                        .zip(&keyset.column_types)
                        .map(|(index, type_)| checkpoint_value_from_copy(&row, *index, type_))
                        .collect::<Result<Vec<_>>>()?,
                );
            }
            rows = rows.saturating_add(1);
        }

        if rows == 0 {
            return Ok(None);
        }
        self.rows_read = self
            .rows_read
            .saturating_add(u64::try_from(rows).unwrap_or(u64::MAX));
        let arrays = builders.into_iter().map(PgBuilder::finish).collect();
        RecordBatch::try_new(self.schema.clone(), arrays)
            .map(Some)
            .map_err(|error| ElmError::Internal(format!("cannot build PostgreSQL batch: {error}")))
    }

    async fn checkpoint(&mut self) -> Result<serde_json::Value> {
        if let Some(keyset) = &self.keyset {
            return serde_json::to_value(&keyset.checkpoint).map_err(|error| {
                ElmError::State(format!(
                    "cannot encode PostgreSQL keyset checkpoint: {error}"
                ))
            });
        }
        Ok(serde_json::json!({ "rows_read": self.rows_read }))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PostgresKeysetCheckpoint {
    version: u16,
    resume_keys: Vec<String>,
    upper_bound: Vec<PgCheckpointValue>,
    last_key: Option<Vec<PgCheckpointValue>>,
}

struct PostgresKeysetState {
    checkpoint: PostgresKeysetCheckpoint,
    column_indices: Vec<usize>,
    column_types: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
enum PgCheckpointValue {
    Boolean(bool),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Text(String),
    Date(i32),
    Timestamp(i64),
    TimestampTz(i64),
}

enum PgBuilder {
    Boolean(BooleanBuilder),
    Int16(Int16Builder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Decimal128(Decimal128Builder, u8, i8),
    Text(StringBuilder),
    Binary(BinaryBuilder),
    Date(Date32Builder),
    Timestamp(TimestampMicrosecondBuilder),
    TimestampTz(TimestampMicrosecondBuilder),
}

impl PgBuilder {
    fn new(type_: &Type, data_type: &DataType, capacity: usize) -> Result<Self> {
        Ok(match *type_ {
            Type::BOOL => Self::Boolean(BooleanBuilder::with_capacity(capacity)),
            Type::INT2 => Self::Int16(Int16Builder::with_capacity(capacity)),
            Type::INT4 => Self::Int32(Int32Builder::with_capacity(capacity)),
            Type::INT8 => Self::Int64(Int64Builder::with_capacity(capacity)),
            Type::FLOAT4 => Self::Float32(Float32Builder::with_capacity(capacity)),
            Type::FLOAT8 => Self::Float64(Float64Builder::with_capacity(capacity)),
            Type::NUMERIC => {
                let DataType::Decimal128(precision, scale) = data_type else {
                    return Err(ElmError::Internal(
                        "PostgreSQL NUMERIC column has a non-decimal Arrow schema".into(),
                    ));
                };
                let builder = Decimal128Builder::with_capacity(capacity)
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|error| {
                        ElmError::TypeMapping(format!(
                            "invalid PostgreSQL decimal precision or scale: {error}"
                        ))
                    })?;
                Self::Decimal128(builder, *precision, *scale)
            }
            Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => Self::Text(
                StringBuilder::with_capacity(capacity, capacity.saturating_mul(16)),
            ),
            Type::BYTEA => Self::Binary(BinaryBuilder::with_capacity(
                capacity,
                capacity.saturating_mul(16),
            )),
            Type::DATE => Self::Date(Date32Builder::with_capacity(capacity)),
            Type::TIMESTAMP => {
                Self::Timestamp(TimestampMicrosecondBuilder::with_capacity(capacity))
            }
            Type::TIMESTAMPTZ => Self::TimestampTz(
                TimestampMicrosecondBuilder::with_capacity(capacity).with_timezone("UTC"),
            ),
            _ => {
                return Err(ElmError::TypeMapping(format!(
                    "PostgreSQL type '{}' has no lossless Arrow mapping",
                    type_.name()
                )));
            }
        })
    }

    fn append(&mut self, row: &BinaryCopyOutRow, index: usize) -> Result<usize> {
        macro_rules! scalar {
            ($builder:expr, $type:ty, $size:expr) => {{
                let value = row
                    .try_get::<Option<$type>>(index)
                    .map_err(copy_decode_error)?;
                $builder.append_option(value);
                $size
            }};
        }
        Ok(match self {
            Self::Boolean(builder) => scalar!(builder, bool, 1),
            Self::Int16(builder) => scalar!(builder, i16, 2),
            Self::Int32(builder) => scalar!(builder, i32, 4),
            Self::Int64(builder) => scalar!(builder, i64, 8),
            Self::Float32(builder) => scalar!(builder, f32, 4),
            Self::Float64(builder) => scalar!(builder, f64, 8),
            Self::Decimal128(builder, precision, scale) => {
                let value = row
                    .try_get::<Option<PgNumeric>>(index)
                    .map_err(copy_decode_error)?;
                let value = value
                    .map(|value| decode_pg_numeric(&value.0, *scale))
                    .transpose()?;
                if let Some(value) = value {
                    ensure_decimal_precision(value, *precision)?;
                }
                builder.append_option(value);
                16
            }
            Self::Text(builder) => {
                let value = row
                    .try_get::<Option<&str>>(index)
                    .map_err(copy_decode_error)?;
                let bytes = value.map_or(0, str::len);
                builder.append_option(value);
                bytes.saturating_add(4)
            }
            Self::Binary(builder) => {
                let value = row
                    .try_get::<Option<&[u8]>>(index)
                    .map_err(copy_decode_error)?;
                let bytes = value.map_or(0, <[u8]>::len);
                builder.append_option(value);
                bytes.saturating_add(4)
            }
            Self::Date(builder) => {
                let value = row
                    .try_get::<Option<NaiveDate>>(index)
                    .map_err(copy_decode_error)?;
                builder.append_option(value.map(date_to_epoch_days).transpose()?);
                4
            }
            Self::Timestamp(builder) => {
                let value = row
                    .try_get::<Option<NaiveDateTime>>(index)
                    .map_err(copy_decode_error)?;
                builder.append_option(value.map(|value| value.and_utc().timestamp_micros()));
                8
            }
            Self::TimestampTz(builder) => {
                let value = row
                    .try_get::<Option<DateTime<Utc>>>(index)
                    .map_err(copy_decode_error)?;
                builder.append_option(value.map(|value| value.timestamp_micros()));
                8
            }
        })
    }

    fn finish(mut self) -> ArrayRef {
        match &mut self {
            Self::Boolean(builder) => Arc::new(builder.finish()),
            Self::Int16(builder) => Arc::new(builder.finish()),
            Self::Int32(builder) => Arc::new(builder.finish()),
            Self::Int64(builder) => Arc::new(builder.finish()),
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

#[derive(Debug)]
struct PgNumeric(Vec<u8>);

impl<'a> FromSql<'a> for PgNumeric {
    fn from_sql(
        _type_: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(Self(raw.to_vec()))
    }

    fn accepts(type_: &Type) -> bool {
        type_ == &Type::NUMERIC
    }
}

fn decode_pg_numeric(raw: &[u8], scale: i8) -> Result<i128> {
    if raw.len() < 8 || !(raw.len() - 8).is_multiple_of(2) {
        return Err(ElmError::TypeMapping(
            "PostgreSQL NUMERIC binary value is malformed".into(),
        ));
    }
    let read_i16 = |offset: usize| i16::from_be_bytes([raw[offset], raw[offset + 1]]);
    let digit_count = usize::try_from(read_i16(0)).map_err(|_| {
        ElmError::TypeMapping("PostgreSQL NUMERIC has a negative digit count".into())
    })?;
    if raw.len() != 8_usize.saturating_add(digit_count.saturating_mul(2)) {
        return Err(ElmError::TypeMapping(
            "PostgreSQL NUMERIC binary digit count is inconsistent".into(),
        ));
    }
    let weight = i32::from(read_i16(2));
    let sign = u16::from_be_bytes([raw[4], raw[5]]);
    if !matches!(sign, 0x0000 | 0x4000) {
        return Err(ElmError::TypeMapping(
            "PostgreSQL NUMERIC NaN and infinity are unsupported".into(),
        ));
    }
    let mut coefficient = 0_i128;
    for index in 0..digit_count {
        let offset = 8 + index * 2;
        let digit = i16::from_be_bytes([raw[offset], raw[offset + 1]]);
        if !(0..10_000).contains(&digit) {
            return Err(ElmError::TypeMapping(
                "PostgreSQL NUMERIC contains an invalid base-10000 digit".into(),
            ));
        }
        if digit == 0 {
            continue;
        }
        // Scale each base-10000 group before accumulating. A full mantissa may
        // contain up to three padding zeros beyond a valid Decimal128 coefficient.
        let exponent = 4 * (weight - index as i32) + i32::from(scale);
        let digit = i128::from(digit);
        let contribution = if exponent >= 0 {
            digit
                .checked_mul(pow10_i128(exponent as u32)?)
                .ok_or_else(|| {
                    ElmError::TypeMapping("PostgreSQL NUMERIC exceeds Decimal128 range".into())
                })?
        } else {
            let divisor = pow10_i128(exponent.unsigned_abs())?;
            if digit % divisor != 0 {
                return Err(ElmError::TypeMapping(
                    "PostgreSQL NUMERIC has more fractional digits than its Arrow scale".into(),
                ));
            }
            digit / divisor
        };
        coefficient = coefficient.checked_add(contribution).ok_or_else(|| {
            ElmError::TypeMapping("PostgreSQL NUMERIC exceeds Decimal128 range".into())
        })?;
    }
    if sign == 0x4000 {
        coefficient = coefficient.checked_neg().ok_or_else(|| {
            ElmError::TypeMapping("PostgreSQL NUMERIC exceeds Decimal128 range".into())
        })?;
    }
    Ok(coefficient)
}

fn encode_pg_numeric(
    value: i128,
    scale: i8,
    output: &mut BytesMut,
) -> std::result::Result<IsNull, Box<dyn std::error::Error + Sync + Send>> {
    let negative = value.is_negative();
    let absolute = value
        .checked_abs()
        .ok_or_else(|| std::io::Error::other("Decimal128 value is out of range"))?;
    let digits = absolute.to_string();
    let (mut integer, mut fractional) = if scale >= 0 {
        let scale = usize::try_from(scale).unwrap_or_default();
        if digits.len() <= scale {
            (
                "0".to_owned(),
                format!("{}{}", "0".repeat(scale - digits.len()), digits),
            )
        } else {
            let split = digits.len() - scale;
            (digits[..split].to_owned(), digits[split..].to_owned())
        }
    } else {
        (
            format!(
                "{}{}",
                digits,
                "0".repeat(usize::from(scale.unsigned_abs()))
            ),
            String::new(),
        )
    };
    let integer_padding = (4 - integer.len() % 4) % 4;
    integer.insert_str(0, &"0".repeat(integer_padding));
    let fractional_padding = (4 - fractional.len() % 4) % 4;
    fractional.push_str(&"0".repeat(fractional_padding));
    let integer_groups = integer.len() / 4;
    let mut groups = integer
        .as_bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .chain(fractional.as_bytes().as_chunks::<4>().0.iter())
        .map(|group| parse_decimal_group(group))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut weight = i16::try_from(integer_groups)
        .map_err(|_| std::io::Error::other("Decimal128 weight is out of range"))?
        - 1;
    let leading_zero_groups = groups.iter().take_while(|group| **group == 0).count();
    groups.drain(..leading_zero_groups);
    weight -= i16::try_from(leading_zero_groups)
        .map_err(|_| std::io::Error::other("Decimal128 weight is out of range"))?;
    while groups.last() == Some(&0) {
        groups.pop();
    }
    if groups.is_empty() {
        weight = 0;
    }
    output.put_i16(
        i16::try_from(groups.len()).map_err(|_| {
            std::io::Error::other("Decimal128 has too many PostgreSQL digit groups")
        })?,
    );
    output.put_i16(weight);
    output.put_u16(if negative && absolute != 0 { 0x4000 } else { 0 });
    output.put_i16(i16::from(scale.max(0)));
    for group in groups {
        output.put_i16(group);
    }
    Ok(IsNull::No)
}

fn parse_decimal_group(
    group: &[u8],
) -> std::result::Result<i16, Box<dyn std::error::Error + Sync + Send>> {
    let text = std::str::from_utf8(group)?;
    Ok(text.parse()?)
}

fn pow10_i128(exponent: u32) -> Result<i128> {
    10_i128.checked_pow(exponent).ok_or_else(|| {
        ElmError::TypeMapping("PostgreSQL NUMERIC exponent exceeds Decimal128 range".into())
    })
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

pub struct PostgresSink {
    session: PostgresSession,
    target: Relation,
    staging: Relation,
    backup: Relation,
    batch_column: Identifier,
    marker: String,
    resume_rows: Option<u64>,
    resume_sequence: Option<u64>,
    schema: Option<Arc<Schema>>,
    columns: Vec<PgColumn>,
    mode: Option<WriteMode>,
    target_existed: bool,
    staged_rows: u64,
    next_sequence: u64,
    pending_sequence: Option<u64>,
    begun: bool,
}

impl PostgresSink {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        target: Relation,
        job_id: JobId,
        resume_checkpoint: Option<&BatchCheckpoint>,
    ) -> Result<Self> {
        reject_catalog(&target)?;
        let session = connect(environment, password).await?;
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
                .ok_or_else(|| ElmError::State("PostgreSQL checkpoint sequence overflow".into()))?,
            None => 0,
        };
        Ok(Self {
            session,
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
            begun: false,
        })
    }

    #[must_use]
    pub fn staging_relation(&self) -> &Relation {
        &self.staging
    }

    async fn validate_existing_schema(
        &self,
        relation: &Relation,
        schema: &Schema,
        staging: bool,
    ) -> Result<()> {
        let projection = if staging {
            quoted_columns(schema)?
        } else {
            "*".into()
        };
        let query = format!(
            "SELECT {projection} FROM {} LIMIT 0",
            dialect_for(DatabaseKind::PostgreSql).quote_relation(relation)
        );
        let statement =
            self.session
                .client
                .prepare(&query)
                .await
                .map_err(|_| ElmError::Connection {
                    message: "PostgreSQL destination schema discovery failed".into(),
                    retryable: true,
                })?;
        if statement.columns().len() != schema.fields().len() {
            return Err(ElmError::TypeMapping(
                "PostgreSQL append destination column count differs from the input".into(),
            ));
        }
        for (column, field) in statement.columns().iter().zip(schema.fields()) {
            let existing = postgres_type_to_arrow(column.type_(), column.type_modifier()).map_err(
                |error| {
                    ElmError::TypeMapping(format!(
                        "destination column '{}': {error}",
                        column.name()
                    ))
                },
            )?;
            let expected = normalized_postgres_arrow_type(field.data_type());
            if column.name() != field.name() || existing != expected {
                return Err(ElmError::TypeMapping(format!(
                    "destination column '{}' does not exactly match input column '{}' ({existing} versus {})",
                    column.name(),
                    field.name(),
                    field.data_type()
                )));
            }
        }
        if staging {
            let batch_query = format!(
                "SELECT {} FROM {} LIMIT 0",
                dialect_for(DatabaseKind::PostgreSql).quote_identifier(&self.batch_column),
                dialect_for(DatabaseKind::PostgreSql).quote_relation(relation)
            );
            let batch_statement = self
                .session
                .client
                .prepare(&batch_query)
                .await
                .map_err(metadata_error)?;
            if batch_statement.columns().len() != 1
                || batch_statement.columns()[0].type_() != &Type::INT8
            {
                return Err(ElmError::Conflict(
                    "PostgreSQL staging checkpoint column is missing or has changed".into(),
                ));
            }
        }
        Ok(())
    }

    async fn verify_staging_marker(&self) -> Result<()> {
        let comment = relation_comment(&self.session.client, &self.staging).await?;
        if comment.as_deref() != Some(&self.marker) {
            return Err(ElmError::Conflict(format!(
                "refusing to reuse or remove unrecognized staging relation {}",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging)
            )));
        }
        Ok(())
    }
}

fn normalized_postgres_arrow_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::LargeUtf8 => DataType::Utf8,
        DataType::LargeBinary => DataType::Binary,
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        }
        other => other.clone(),
    }
}

/// Removes only the staging table marked for this job. Unmarked relations are never touched.
pub async fn cleanup_postgres_staging(
    environment: &Environment,
    password: &str,
    target: &Relation,
    job_id: JobId,
) -> Result<bool> {
    reject_catalog(target)?;
    let session = connect(environment, password).await?;
    let staging = Relation {
        catalog: None,
        schema: target.schema.clone(),
        name: Identifier::new(format!("elm_stage_{}", job_id.0.simple()))?,
    };
    if !relation_exists(&session.client, &staging).await? {
        return Ok(false);
    }
    let expected = format!("{STAGING_COMMENT_PREFIX}{job_id}");
    if relation_comment(&session.client, &staging)
        .await?
        .as_deref()
        != Some(&expected)
    {
        return Err(ElmError::Conflict(format!(
            "refusing to remove unrecognized PostgreSQL relation {}",
            dialect_for(DatabaseKind::PostgreSql).quote_relation(&staging)
        )));
    }
    let sql = format!(
        "DROP TABLE {}",
        dialect_for(DatabaseKind::PostgreSql).quote_relation(&staging)
    );
    session
        .client
        .batch_execute(&sql)
        .await
        .map_err(staging_permission_error)?;
    Ok(true)
}

#[async_trait]
impl DataSink for PostgresSink {
    async fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        _consistency: ConsistencyMode,
    ) -> Result<PreflightReport> {
        if self.schema.is_some() {
            return Err(ElmError::Conflict(
                "PostgreSQL sink has already been preflighted".into(),
            ));
        }
        if schema.fields().is_empty() {
            return Err(ElmError::TypeMapping(
                "PostgreSQL destinations require at least one column".into(),
            ));
        }
        if schema
            .fields()
            .iter()
            .any(|field| field.name() == self.batch_column.as_str())
        {
            return Err(ElmError::Conflict(
                "input schema collides with the job-scoped PostgreSQL checkpoint column".into(),
            ));
        }
        let columns = schema
            .fields()
            .iter()
            .map(|field| PgColumn::from_field(field))
            .collect::<Result<Vec<_>>>()?;
        let target_existed = relation_exists(&self.session.client, &self.target).await?;
        if mode == WriteMode::Fail && target_existed {
            return Err(ElmError::Conflict(format!(
                "destination {} already exists",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.target)
            )));
        }
        require_create_privilege(&self.session.client, &self.target).await?;
        if target_existed && mode == WriteMode::Append {
            require_table_privilege(&self.session.client, &self.target, "INSERT").await?;
            self.validate_existing_schema(&self.target, &schema, false)
                .await?;
        }
        if target_existed && mode == WriteMode::Replace {
            require_relation_ownership(&self.session.client, &self.target).await?;
        }

        self.schema = Some(schema);
        self.columns = columns;
        self.mode = Some(mode);
        self.target_existed = target_existed;
        Ok(PreflightReport {
            capabilities: ConnectorCapabilities {
                database: Some(DatabaseKind::PostgreSql),
                native_bulk_read: true,
                native_bulk_write: true,
                atomic_append: true,
                atomic_replace: true,
                non_atomic_replace: false,
                checkpointed_write: false,
                resumable_keyset_read: true,
                supports_lob_spill: false,
            },
            staging_relation: Some(self.staging.clone()),
            warnings: if self.session.tls_enabled {
                Vec::new()
            } else {
                vec![
                    "PostgreSQL transport encryption is explicitly disabled for this environment"
                        .into(),
                ]
            },
        })
    }

    async fn begin(&mut self) -> Result<()> {
        if self.begun {
            return Err(ElmError::Conflict(
                "PostgreSQL sink has already begun".into(),
            ));
        }
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| ElmError::Internal("PostgreSQL sink was not preflighted".into()))?;
        let mode = self
            .mode
            .ok_or_else(|| ElmError::Internal("PostgreSQL sink was not preflighted".into()))?;
        let target_now = relation_exists(&self.session.client, &self.target).await?;
        if target_now != self.target_existed {
            return Err(ElmError::Conflict(
                "PostgreSQL destination changed after preflight".into(),
            ));
        }
        if mode == WriteMode::Fail && target_now {
            return Err(ElmError::Conflict(
                "PostgreSQL destination appeared after preflight".into(),
            ));
        }

        if relation_exists(&self.session.client, &self.staging).await? {
            self.verify_staging_marker().await?;
            if let Some(expected_rows) = self.resume_rows {
                self.validate_existing_schema(&self.staging, schema, true)
                    .await?;
                let sequence = self.resume_sequence.ok_or_else(|| {
                    ElmError::State("PostgreSQL resume row count has no batch sequence".into())
                })?;
                let sequence = i64::try_from(sequence).map_err(|_| {
                    ElmError::State("PostgreSQL checkpoint sequence exceeds BIGINT".into())
                })?;
                let trim_sql = format!(
                    "DELETE FROM {} WHERE {} > $1",
                    dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging),
                    dialect_for(DatabaseKind::PostgreSql).quote_identifier(&self.batch_column)
                );
                self.session
                    .client
                    .execute(&trim_sql, &[&sequence])
                    .await
                    .map_err(staging_permission_error)?;
                let actual_rows = relation_row_count(&self.session.client, &self.staging).await?;
                if actual_rows != expected_rows {
                    return Err(ElmError::Conflict(format!(
                        "PostgreSQL staging has {actual_rows} rows but checkpoint expects {expected_rows}"
                    )));
                }
                self.begun = true;
                return Ok(());
            }
            let drop_sql = format!(
                "DROP TABLE {}",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging)
            );
            self.session
                .client
                .batch_execute(&drop_sql)
                .await
                .map_err(staging_permission_error)?;
        } else if self.resume_rows.is_some() {
            return Err(ElmError::Conflict(
                "PostgreSQL staging relation required by the checkpoint is missing".into(),
            ));
        }

        let create_sql = if mode == WriteMode::Append && self.target_existed {
            format!(
                "CREATE TABLE {} (LIKE {} INCLUDING ALL)",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging),
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.target)
            )
        } else {
            create_table_sql(&self.staging, schema, &self.columns)?
        };
        self.session
            .client
            .batch_execute(&create_sql)
            .await
            .map_err(staging_permission_error)?;
        let checkpoint_sql = format!(
            "ALTER TABLE {} ADD COLUMN {} BIGINT NOT NULL",
            dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging),
            dialect_for(DatabaseKind::PostgreSql).quote_identifier(&self.batch_column)
        );
        if self
            .session
            .client
            .batch_execute(&checkpoint_sql)
            .await
            .is_err()
        {
            let cleanup = format!(
                "DROP TABLE IF EXISTS {}",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging)
            );
            let _cleanup_result = self.session.client.batch_execute(&cleanup).await;
            return Err(ElmError::PermissionDenied(
                "PostgreSQL staging checkpoint column could not be created".into(),
            ));
        }
        let comment_sql = format!(
            "COMMENT ON TABLE {} IS {}",
            dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging),
            sql_string_literal(&self.marker)
        );
        if self
            .session
            .client
            .batch_execute(&comment_sql)
            .await
            .is_err()
        {
            let cleanup = format!(
                "DROP TABLE IF EXISTS {}",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging)
            );
            let _cleanup_result = self.session.client.batch_execute(&cleanup).await;
            return Err(ElmError::PermissionDenied(
                "PostgreSQL staging table could not be marked for safe recovery".into(),
            ));
        }
        self.begun = true;
        Ok(())
    }

    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if !self.begun {
            return Err(ElmError::Conflict("PostgreSQL sink has not begun".into()));
        }
        if self.pending_sequence.is_some() {
            return Err(ElmError::State(
                "PostgreSQL staging batch has not been checkpointed".into(),
            ));
        }
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| ElmError::Internal("PostgreSQL sink has no schema".into()))?;
        ensure_batch_schema(batch.schema().as_ref(), schema)?;
        let copy_sql = format!(
            "COPY {} ({}) FROM STDIN (FORMAT BINARY)",
            dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging),
            quoted_columns_with_batch(schema, &self.batch_column)?
        );
        let copy =
            self.session
                .client
                .copy_in(&copy_sql)
                .await
                .map_err(|_| ElmError::Connection {
                    message: "PostgreSQL binary COPY destination could not start".into(),
                    retryable: true,
                })?;
        let mut types = self
            .columns
            .iter()
            .map(|column| column.pg_type.clone())
            .collect::<Vec<_>>();
        types.push(Type::INT8);
        let batch_sequence = i64::try_from(self.next_sequence)
            .map_err(|_| ElmError::State("PostgreSQL batch sequence exceeds BIGINT".into()))?;
        let mut writer = Box::pin(BinaryCopyInWriter::new(copy, &types));
        for row in 0..batch.num_rows() {
            let mut cells = pg_cells(batch, row)?;
            cells.push(PgCell::Int64(batch_sequence));
            let values = cells
                .iter()
                .map(|cell| cell as &(dyn ToSql + Sync))
                .collect::<Vec<_>>();
            if writer.as_mut().write(&values).await.is_err() {
                // Dropping an unfinished BinaryCopyInWriter explicitly aborts COPY by contract.
                drop(writer);
                return Err(ElmError::Connection {
                    message: "PostgreSQL binary COPY destination write failed".into(),
                    retryable: true,
                });
            }
        }
        let written = writer
            .as_mut()
            .finish()
            .await
            .map_err(|_| ElmError::Connection {
                message: "PostgreSQL binary COPY destination commit failed".into(),
                retryable: true,
            })?;
        let expected = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        if written != expected {
            return Err(ElmError::State(format!(
                "PostgreSQL COPY reported {written} rows for a batch containing {expected}"
            )));
        }
        self.staged_rows = self.staged_rows.saturating_add(written);
        self.pending_sequence = Some(self.next_sequence);
        Ok(())
    }

    async fn commit_checkpoint(&mut self, checkpoint: &BatchCheckpoint) -> Result<()> {
        if self.pending_sequence != Some(checkpoint.sequence) {
            return Err(ElmError::State(format!(
                "PostgreSQL sink expected checkpoint sequence {}, received {}",
                self.next_sequence, checkpoint.sequence
            )));
        }
        if checkpoint.rows_committed != self.staged_rows {
            return Err(ElmError::State(format!(
                "PostgreSQL staging has committed {} rows but checkpoint records {}",
                self.staged_rows, checkpoint.rows_committed
            )));
        }
        self.pending_sequence = None;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| ElmError::State("PostgreSQL checkpoint sequence overflow".into()))?;
        Ok(())
    }

    async fn publish(&mut self) -> Result<()> {
        if !self.begun {
            return Err(ElmError::Conflict("PostgreSQL sink has not begun".into()));
        }
        if self.pending_sequence.is_some() {
            return Err(ElmError::State(
                "PostgreSQL staging batch has not been checkpointed".into(),
            ));
        }
        let mode = self
            .mode
            .ok_or_else(|| ElmError::Internal("PostgreSQL sink has no write mode".into()))?;
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| ElmError::Internal("PostgreSQL sink has no schema".into()))?;
        let data_columns = quoted_columns(schema)?;
        let transaction = self
            .session
            .client
            .transaction()
            .await
            .map_err(publication_error)?;
        let stage = dialect_for(DatabaseKind::PostgreSql).quote_relation(&self.staging);
        transaction
            .batch_execute(&format!("LOCK TABLE {stage} IN ACCESS EXCLUSIVE MODE"))
            .await
            .map_err(publication_error)?;
        let actual_rows = relation_row_count(&transaction, &self.staging).await?;
        if actual_rows != self.staged_rows {
            return Err(ElmError::Conflict(format!(
                "PostgreSQL staging row count changed from {} to {actual_rows} before publication",
                self.staged_rows
            )));
        }
        let target_exists_now = relation_exists(&transaction, &self.target).await?;
        if target_exists_now != self.target_existed {
            return Err(ElmError::Conflict(
                "PostgreSQL destination changed before publication".into(),
            ));
        }

        let dialect = dialect_for(DatabaseKind::PostgreSql);
        let target = dialect.quote_relation(&self.target);
        let drop_batch_column = format!(
            "ALTER TABLE {stage} DROP COLUMN {}",
            dialect.quote_identifier(&self.batch_column)
        );
        match mode {
            WriteMode::Append if self.target_existed => {
                transaction
                    .batch_execute(&format!(
                        "LOCK TABLE {target} IN ACCESS EXCLUSIVE MODE;
                         INSERT INTO {target} ({data_columns}) SELECT {data_columns} FROM {stage};
                         DROP TABLE {stage}"
                    ))
                    .await
                    .map_err(publication_error)?;
            }
            WriteMode::Append | WriteMode::Fail => {
                transaction
                    .batch_execute(&format!(
                        "{drop_batch_column}; ALTER TABLE {stage} RENAME TO {}",
                        dialect.quote_identifier(&self.target.name)
                    ))
                    .await
                    .map_err(publication_error)?;
            }
            WriteMode::Replace if self.target_existed => {
                if relation_exists(&transaction, &self.backup).await? {
                    return Err(ElmError::Conflict(
                        "PostgreSQL recovery backup relation already exists".into(),
                    ));
                }
                transaction
                    .batch_execute(&format!(
                        "LOCK TABLE {target} IN ACCESS EXCLUSIVE MODE;
                         {drop_batch_column};
                         ALTER TABLE {target} RENAME TO {};
                         ALTER TABLE {stage} RENAME TO {};
                         DROP TABLE {}",
                        dialect.quote_identifier(&self.backup.name),
                        dialect.quote_identifier(&self.target.name),
                        dialect.quote_relation(&self.backup)
                    ))
                    .await
                    .map_err(publication_error)?;
            }
            WriteMode::Replace => {
                transaction
                    .batch_execute(&format!(
                        "{drop_batch_column}; ALTER TABLE {stage} RENAME TO {}",
                        dialect.quote_identifier(&self.target.name)
                    ))
                    .await
                    .map_err(publication_error)?;
            }
        }
        transaction.commit().await.map_err(publication_error)?;
        self.begun = false;
        Ok(())
    }

    async fn abort(&mut self) -> Result<()> {
        // Completed per-batch COPY operations remain in the marked staging table for resume.
        Ok(())
    }
}

#[derive(Debug)]
struct PgColumn {
    pg_type: Type,
    ddl: String,
}

impl PgColumn {
    fn from_field(field: &Field) -> Result<Self> {
        let (pg_type, ddl) = match field.data_type() {
            DataType::Boolean => (Type::BOOL, "BOOLEAN".into()),
            DataType::Int16 => (Type::INT2, "SMALLINT".into()),
            DataType::Int32 => (Type::INT4, "INTEGER".into()),
            DataType::Int64 => (Type::INT8, "BIGINT".into()),
            DataType::Float32 => (Type::FLOAT4, "REAL".into()),
            DataType::Float64 => (Type::FLOAT8, "DOUBLE PRECISION".into()),
            DataType::Decimal128(precision, scale) => {
                (Type::NUMERIC, format!("NUMERIC({precision}, {scale})"))
            }
            DataType::Utf8 | DataType::LargeUtf8 => (Type::TEXT, "TEXT".into()),
            DataType::Binary | DataType::LargeBinary => (Type::BYTEA, "BYTEA".into()),
            DataType::Date32 => (Type::DATE, "DATE".into()),
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                (Type::TIMESTAMP, "TIMESTAMP WITHOUT TIME ZONE".into())
            }
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
                (Type::TIMESTAMPTZ, "TIMESTAMP WITH TIME ZONE".into())
            }
            other => {
                return Err(ElmError::TypeMapping(format!(
                    "column '{}': Arrow type {other} has no lossless PostgreSQL mapping",
                    field.name()
                )));
            }
        };
        Ok(Self { pg_type, ddl })
    }
}

#[derive(Debug)]
enum PgCell<'a> {
    Null,
    Boolean(bool),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Decimal128(i128, i8),
    Text(&'a str),
    Binary(&'a [u8]),
    Date(NaiveDate),
    Timestamp(NaiveDateTime),
    TimestampTz(DateTime<Utc>),
}

impl ToSql for PgCell<'_> {
    fn to_sql(
        &self,
        type_: &Type,
        output: &mut BytesMut,
    ) -> std::result::Result<IsNull, Box<dyn std::error::Error + Sync + Send>> {
        match self {
            Self::Null => Ok(IsNull::Yes),
            Self::Boolean(value) => value.to_sql(type_, output),
            Self::Int16(value) => value.to_sql(type_, output),
            Self::Int32(value) => value.to_sql(type_, output),
            Self::Int64(value) => value.to_sql(type_, output),
            Self::Float32(value) => value.to_sql(type_, output),
            Self::Float64(value) => value.to_sql(type_, output),
            Self::Decimal128(value, scale) => encode_pg_numeric(*value, *scale, output),
            Self::Text(value) => value.to_sql(type_, output),
            Self::Binary(value) => value.to_sql(type_, output),
            Self::Date(value) => value.to_sql(type_, output),
            Self::Timestamp(value) => value.to_sql(type_, output),
            Self::TimestampTz(value) => value.to_sql(type_, output),
        }
    }

    fn accepts(_type_: &Type) -> bool {
        true
    }

    postgres_types::to_sql_checked!();
}

fn pg_cells(batch: &RecordBatch, row: usize) -> Result<Vec<PgCell<'_>>> {
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| {
            if array.is_null(row) {
                return Ok(PgCell::Null);
            }
            Ok(match field.data_type() {
                DataType::Boolean => PgCell::Boolean(array_as::<BooleanArray>(array, field)?.value(row)),
                DataType::Int16 => PgCell::Int16(array_as::<Int16Array>(array, field)?.value(row)),
                DataType::Int32 => PgCell::Int32(array_as::<Int32Array>(array, field)?.value(row)),
                DataType::Int64 => PgCell::Int64(array_as::<Int64Array>(array, field)?.value(row)),
                DataType::Float32 => PgCell::Float32(array_as::<Float32Array>(array, field)?.value(row)),
                DataType::Float64 => PgCell::Float64(array_as::<Float64Array>(array, field)?.value(row)),
                DataType::Decimal128(precision, scale) => {
                    let value = array_as::<Decimal128Array>(array, field)?.value(row);
                    ensure_decimal_precision(value, *precision)?;
                    PgCell::Decimal128(value, *scale)
                }
                DataType::Utf8 => PgCell::Text(array_as::<StringArray>(array, field)?.value(row)),
                DataType::LargeUtf8 => {
                    PgCell::Text(array_as::<LargeStringArray>(array, field)?.value(row))
                }
                DataType::Binary => PgCell::Binary(array_as::<BinaryArray>(array, field)?.value(row)),
                DataType::LargeBinary => {
                    PgCell::Binary(array_as::<LargeBinaryArray>(array, field)?.value(row))
                }
                DataType::Date32 => PgCell::Date(epoch_days_to_date(
                    array_as::<Date32Array>(array, field)?.value(row),
                )?),
                DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
                    let value = array_as::<TimestampMicrosecondArray>(array, field)?.value(row);
                    let instant = DateTime::<Utc>::from_timestamp_micros(value).ok_or_else(|| {
                        ElmError::TypeMapping(format!(
                            "column '{}' contains an out-of-range timestamp",
                            field.name()
                        ))
                    })?;
                    if timezone.is_some() {
                        PgCell::TimestampTz(instant)
                    } else {
                        PgCell::Timestamp(instant.naive_utc())
                    }
                }
                other => {
                    return Err(ElmError::TypeMapping(format!(
                        "column '{}': Arrow type {other} cannot be encoded by PostgreSQL binary COPY",
                        field.name()
                    )));
                }
            })
        })
        .collect()
}

async fn build_keyset_query(
    client: &Client,
    relation: &Relation,
    resume_keys: &[Identifier],
    checkpoint: Option<&serde_json::Value>,
) -> Result<(String, Option<PostgresKeysetState>)> {
    reject_catalog(relation)?;
    let dialect = dialect_for(DatabaseKind::PostgreSql);
    let quoted_relation = dialect.quote_relation(relation);
    let discovery = client
        .prepare(&format!("SELECT * FROM {quoted_relation} LIMIT 0"))
        .await
        .map_err(metadata_error)?;
    let mut seen = BTreeSet::new();
    let mut column_indices = Vec::with_capacity(resume_keys.len());
    let mut column_types = Vec::with_capacity(resume_keys.len());
    for key in resume_keys {
        if !seen.insert(key.as_str()) {
            return Err(ElmError::Validation(format!(
                "PostgreSQL resume key '{}' is duplicated",
                key.as_str()
            )));
        }
        let (index, column) = discovery
            .columns()
            .iter()
            .enumerate()
            .find(|(_, column)| column.name() == key.as_str())
            .ok_or_else(|| {
                ElmError::Validation(format!(
                    "PostgreSQL resume key '{}' is not a source column",
                    key.as_str()
                ))
            })?;
        require_keyset_type(column.type_(), key)?;
        column_indices.push(index);
        column_types.push(column.type_().clone());
    }

    let quoted_keys = resume_keys
        .iter()
        .map(|key| dialect.quote_identifier(key))
        .collect::<Vec<_>>();
    let key_list = quoted_keys.join(", ");
    let null_predicate = quoted_keys
        .iter()
        .map(|key| format!("{key} IS NULL"))
        .collect::<Vec<_>>()
        .join(" OR ");
    let keys_are_non_null: bool = client
        .query_one(
            &format!("SELECT NOT EXISTS (SELECT 1 FROM {quoted_relation} WHERE {null_predicate})"),
            &[],
        )
        .await
        .map_err(metadata_error)?
        .try_get(0)
        .map_err(|_| metadata_error(()))?;
    if !keys_are_non_null {
        return Err(ElmError::Validation(
            "PostgreSQL resume keys must be non-null for every source row".into(),
        ));
    }
    let keys_are_unique: bool = client
        .query_one(
            &format!(
                "SELECT NOT EXISTS (
                    SELECT 1 FROM {quoted_relation}
                    GROUP BY {key_list} HAVING COUNT(*) > 1
                )"
            ),
            &[],
        )
        .await
        .map_err(metadata_error)?
        .try_get(0)
        .map_err(|_| metadata_error(()))?;
    if !keys_are_unique {
        return Err(ElmError::Validation(
            "PostgreSQL resume keys must uniquely identify every source row".into(),
        ));
    }

    let expected_names = resume_keys
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let is_resume = checkpoint.is_some();
    let checkpoint = match checkpoint {
        Some(value) => {
            let checkpoint: PostgresKeysetCheckpoint = serde_json::from_value(value.clone())
                .map_err(|error| {
                    ElmError::State(format!(
                        "invalid PostgreSQL keyset checkpoint document: {error}"
                    ))
                })?;
            if checkpoint.version != 1 || checkpoint.resume_keys != expected_names {
                return Err(ElmError::Conflict(
                    "PostgreSQL resume keys do not match the saved checkpoint".into(),
                ));
            }
            checkpoint
        }
        None => {
            let upper_row = client
                .query_opt(
                    &format!(
                        "SELECT {key_list} FROM {quoted_relation}
                         ORDER BY {} LIMIT 1",
                        quoted_keys
                            .iter()
                            .map(|key| format!("{key} DESC"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    &[],
                )
                .await
                .map_err(metadata_error)?;
            let upper_bound = upper_row
                .as_ref()
                .map(|row| checkpoint_values_from_row(row, &column_types))
                .transpose()?
                .unwrap_or_default();
            PostgresKeysetCheckpoint {
                version: 1,
                resume_keys: expected_names,
                upper_bound,
                last_key: None,
            }
        }
    };
    if checkpoint.upper_bound.is_empty() {
        if is_resume {
            return Err(ElmError::Conflict(
                "PostgreSQL keyset checkpoint has no upper bound".into(),
            ));
        }
    } else {
        validate_checkpoint_values(&checkpoint.upper_bound, &column_types, "upper bound")?;
    }
    if let Some(last_key) = &checkpoint.last_key {
        validate_checkpoint_values(last_key, &column_types, "last key")?;
    }

    if checkpoint.upper_bound.is_empty() {
        return Ok((
            format!("SELECT * FROM {quoted_relation} WHERE FALSE"),
            Some(PostgresKeysetState {
                checkpoint,
                column_indices,
                column_types,
            }),
        ));
    }

    let bounds_name = Identifier::new("elm_resume_bounds")?;
    let bounds_relation = Relation {
        catalog: None,
        schema: Some(Identifier::new("pg_temp")?),
        name: bounds_name.clone(),
    };
    let quoted_bounds = dialect.quote_relation(&bounds_relation);
    let role = Identifier::new("elm_bound_role")?;
    let quoted_role = dialect.quote_identifier(&role);
    client
        .batch_execute(&format!(
            "CREATE TEMP TABLE {} AS SELECT {key_list} FROM {quoted_relation} WITH NO DATA;
             ALTER TABLE {quoted_bounds} ADD COLUMN {quoted_role} TEXT NOT NULL",
            dialect.quote_identifier(&bounds_name)
        ))
        .await
        .map_err(metadata_error)?;
    insert_keyset_bound(
        client,
        &bounds_relation,
        &role,
        resume_keys,
        "upper",
        &checkpoint.upper_bound,
    )
    .await?;
    if let Some(last_key) = &checkpoint.last_key {
        insert_keyset_bound(
            client,
            &bounds_relation,
            &role,
            resume_keys,
            "lower",
            last_key,
        )
        .await?;
    }

    let source_alias = dialect.quote_identifier(&Identifier::new("elm_source")?);
    let upper_alias = dialect.quote_identifier(&Identifier::new("elm_upper")?);
    let lower_alias = dialect.quote_identifier(&Identifier::new("elm_lower")?);
    let qualified = |alias: &str| {
        quoted_keys
            .iter()
            .map(|key| format!("{alias}.{key}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let source_keys = qualified(&source_alias);
    let upper_keys = qualified(&upper_alias);
    let lower_keys = qualified(&lower_alias);
    let lower_join = checkpoint.last_key.as_ref().map_or_else(String::new, |_| {
        format!(" CROSS JOIN {quoted_bounds} AS {lower_alias}")
    });
    let lower_predicate = checkpoint.last_key.as_ref().map_or_else(String::new, |_| {
        format!(
            " AND {lower_alias}.{quoted_role} = 'lower' AND ROW({source_keys}) > ROW({lower_keys})"
        )
    });
    let query = format!(
        "SELECT {source_alias}.* FROM {quoted_relation} AS {source_alias}
         CROSS JOIN {quoted_bounds} AS {upper_alias}{lower_join}
         WHERE {upper_alias}.{quoted_role} = 'upper'
           AND ROW({source_keys}) <= ROW({upper_keys}){lower_predicate}
         ORDER BY {source_keys}"
    );
    Ok((
        query,
        Some(PostgresKeysetState {
            checkpoint,
            column_indices,
            column_types,
        }),
    ))
}

async fn insert_keyset_bound(
    client: &Client,
    bounds: &Relation,
    role_column: &Identifier,
    keys: &[Identifier],
    role: &str,
    values: &[PgCheckpointValue],
) -> Result<()> {
    let dialect = dialect_for(DatabaseKind::PostgreSql);
    let columns = std::iter::once(dialect.quote_identifier(role_column))
        .chain(keys.iter().map(|key| dialect.quote_identifier(key)))
        .collect::<Vec<_>>()
        .join(", ");
    let parameters = (1..=values.len().saturating_add(1))
        .map(|index| dialect.parameter(index))
        .collect::<Vec<_>>()
        .join(", ");
    let mut bind: Vec<&(dyn ToSql + Sync)> = Vec::with_capacity(values.len().saturating_add(1));
    bind.push(&role);
    bind.extend(values.iter().map(|value| value as &(dyn ToSql + Sync)));
    client
        .execute(
            &format!(
                "INSERT INTO {} ({columns}) VALUES ({parameters})",
                dialect.quote_relation(bounds)
            ),
            &bind,
        )
        .await
        .map_err(|_| {
            ElmError::Conflict(
                "PostgreSQL checkpoint values no longer match the resume-key column types".into(),
            )
        })?;
    Ok(())
}

fn require_keyset_type(type_: &Type, key: &Identifier) -> Result<()> {
    if matches!(
        *type_,
        Type::BOOL
            | Type::INT2
            | Type::INT4
            | Type::INT8
            | Type::TEXT
            | Type::VARCHAR
            | Type::BPCHAR
            | Type::NAME
            | Type::DATE
            | Type::TIMESTAMP
            | Type::TIMESTAMPTZ
    ) {
        Ok(())
    } else {
        Err(ElmError::TypeMapping(format!(
            "PostgreSQL resume key '{}' uses unsupported type '{}'",
            key.as_str(),
            type_.name()
        )))
    }
}

fn checkpoint_values_from_row(row: &Row, types: &[Type]) -> Result<Vec<PgCheckpointValue>> {
    types
        .iter()
        .enumerate()
        .map(|(index, type_)| checkpoint_value_from_row(row, index, type_))
        .collect()
}

fn checkpoint_value_from_row(row: &Row, index: usize, type_: &Type) -> Result<PgCheckpointValue> {
    let value = match *type_ {
        Type::BOOL => PgCheckpointValue::Boolean(row.try_get(index).map_err(key_decode_error)?),
        Type::INT2 => PgCheckpointValue::Int16(row.try_get(index).map_err(key_decode_error)?),
        Type::INT4 => PgCheckpointValue::Int32(row.try_get(index).map_err(key_decode_error)?),
        Type::INT8 => PgCheckpointValue::Int64(row.try_get(index).map_err(key_decode_error)?),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => {
            PgCheckpointValue::Text(row.try_get(index).map_err(key_decode_error)?)
        }
        Type::DATE => PgCheckpointValue::Date(date_to_epoch_days(
            row.try_get(index).map_err(key_decode_error)?,
        )?),
        Type::TIMESTAMP => {
            let value: NaiveDateTime = row.try_get(index).map_err(key_decode_error)?;
            PgCheckpointValue::Timestamp(value.and_utc().timestamp_micros())
        }
        Type::TIMESTAMPTZ => {
            let value: DateTime<Utc> = row.try_get(index).map_err(key_decode_error)?;
            PgCheckpointValue::TimestampTz(value.timestamp_micros())
        }
        _ => {
            return Err(ElmError::TypeMapping(format!(
                "PostgreSQL type '{}' cannot be checkpointed",
                type_.name()
            )));
        }
    };
    Ok(value)
}

fn checkpoint_value_from_copy(
    row: &BinaryCopyOutRow,
    index: usize,
    type_: &Type,
) -> Result<PgCheckpointValue> {
    let value = match *type_ {
        Type::BOOL => PgCheckpointValue::Boolean(row.try_get(index).map_err(copy_decode_error)?),
        Type::INT2 => PgCheckpointValue::Int16(row.try_get(index).map_err(copy_decode_error)?),
        Type::INT4 => PgCheckpointValue::Int32(row.try_get(index).map_err(copy_decode_error)?),
        Type::INT8 => PgCheckpointValue::Int64(row.try_get(index).map_err(copy_decode_error)?),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => {
            PgCheckpointValue::Text(row.try_get(index).map_err(copy_decode_error)?)
        }
        Type::DATE => PgCheckpointValue::Date(date_to_epoch_days(
            row.try_get(index).map_err(copy_decode_error)?,
        )?),
        Type::TIMESTAMP => {
            let value: NaiveDateTime = row.try_get(index).map_err(copy_decode_error)?;
            PgCheckpointValue::Timestamp(value.and_utc().timestamp_micros())
        }
        Type::TIMESTAMPTZ => {
            let value: DateTime<Utc> = row.try_get(index).map_err(copy_decode_error)?;
            PgCheckpointValue::TimestampTz(value.timestamp_micros())
        }
        _ => {
            return Err(ElmError::TypeMapping(format!(
                "PostgreSQL type '{}' cannot be checkpointed",
                type_.name()
            )));
        }
    };
    Ok(value)
}

fn validate_checkpoint_values(
    values: &[PgCheckpointValue],
    types: &[Type],
    label: &str,
) -> Result<()> {
    if values.len() != types.len()
        || !values
            .iter()
            .zip(types)
            .all(|(value, type_)| checkpoint_value_matches_type(value, type_))
    {
        return Err(ElmError::Conflict(format!(
            "PostgreSQL checkpoint {label} does not match the configured resume keys"
        )));
    }
    Ok(())
}

fn checkpoint_value_matches_type(value: &PgCheckpointValue, type_: &Type) -> bool {
    matches!(
        (value, type_),
        (PgCheckpointValue::Boolean(_), &Type::BOOL)
            | (PgCheckpointValue::Int16(_), &Type::INT2)
            | (PgCheckpointValue::Int32(_), &Type::INT4)
            | (PgCheckpointValue::Int64(_), &Type::INT8)
            | (
                PgCheckpointValue::Text(_),
                &Type::TEXT | &Type::VARCHAR | &Type::BPCHAR | &Type::NAME
            )
            | (PgCheckpointValue::Date(_), &Type::DATE)
            | (PgCheckpointValue::Timestamp(_), &Type::TIMESTAMP)
            | (PgCheckpointValue::TimestampTz(_), &Type::TIMESTAMPTZ)
    )
}

impl ToSql for PgCheckpointValue {
    fn to_sql(
        &self,
        type_: &Type,
        output: &mut BytesMut,
    ) -> std::result::Result<IsNull, Box<dyn std::error::Error + Sync + Send>> {
        match self {
            Self::Boolean(value) => value.to_sql(type_, output),
            Self::Int16(value) => value.to_sql(type_, output),
            Self::Int32(value) => value.to_sql(type_, output),
            Self::Int64(value) => value.to_sql(type_, output),
            Self::Text(value) => value.to_sql(type_, output),
            Self::Date(value) => epoch_days_to_date(*value)?.to_sql(type_, output),
            Self::Timestamp(value) => DateTime::<Utc>::from_timestamp_micros(*value)
                .ok_or("checkpoint timestamp is out of range")?
                .naive_utc()
                .to_sql(type_, output),
            Self::TimestampTz(value) => DateTime::<Utc>::from_timestamp_micros(*value)
                .ok_or("checkpoint timestamp is out of range")?
                .to_sql(type_, output),
        }
    }

    fn accepts(type_: &Type) -> bool {
        matches!(
            *type_,
            Type::BOOL
                | Type::INT2
                | Type::INT4
                | Type::INT8
                | Type::TEXT
                | Type::VARCHAR
                | Type::BPCHAR
                | Type::NAME
                | Type::DATE
                | Type::TIMESTAMP
                | Type::TIMESTAMPTZ
        )
    }

    postgres_types::to_sql_checked!();
}

fn key_decode_error(_error: tokio_postgres::Error) -> ElmError {
    ElmError::Conflict("PostgreSQL resume key changed or became null while reading".into())
}

fn array_as<'a, T: Array + 'static>(array: &'a ArrayRef, field: &Field) -> Result<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        ElmError::Internal(format!(
            "column '{}' array does not match its Arrow schema",
            field.name()
        ))
    })
}

fn selection_query(selection: &DatabaseSelection) -> Result<String> {
    match selection {
        DatabaseSelection::Table { relation } => {
            reject_catalog(relation)?;
            Ok(format!(
                "SELECT * FROM {}",
                dialect_for(DatabaseKind::PostgreSql).quote_relation(relation)
            ))
        }
        DatabaseSelection::Query { sql } => {
            let trimmed = sql.trim();
            if trimmed.is_empty() {
                return Err(ElmError::Validation(
                    "PostgreSQL source query may not be empty".into(),
                ));
            }
            if trimmed.ends_with(';') {
                return Err(ElmError::Validation(
                    "PostgreSQL source query must omit its trailing semicolon for binary COPY"
                        .into(),
                ));
            }
            Ok(sql.clone())
        }
    }
}

fn reject_catalog(relation: &Relation) -> Result<()> {
    if relation.catalog.is_some() {
        return Err(ElmError::Validation(
            "PostgreSQL relations cannot contain a catalog; choose the database in the environment"
                .into(),
        ));
    }
    Ok(())
}

fn postgres_type_to_arrow(
    type_: &Type,
    type_modifier: i32,
) -> std::result::Result<DataType, String> {
    match *type_ {
        Type::BOOL => Ok(DataType::Boolean),
        Type::INT2 => Ok(DataType::Int16),
        Type::INT4 => Ok(DataType::Int32),
        Type::INT8 => Ok(DataType::Int64),
        Type::FLOAT4 => Ok(DataType::Float32),
        Type::FLOAT8 => Ok(DataType::Float64),
        Type::NUMERIC => {
            let (precision, scale) = numeric_precision_scale(type_modifier)?;
            Ok(DataType::Decimal128(precision, scale))
        }
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => Ok(DataType::Utf8),
        Type::BYTEA => Ok(DataType::Binary),
        Type::DATE => Ok(DataType::Date32),
        Type::TIMESTAMP => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        Type::TIMESTAMPTZ => Ok(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("UTC".into()),
        )),
        _ => Err(format!(
            "PostgreSQL type '{}' has no lossless Arrow mapping",
            type_.name()
        )),
    }
}

fn numeric_precision_scale(type_modifier: i32) -> std::result::Result<(u8, i8), String> {
    if type_modifier < 4 {
        return Err(
            "unconstrained PostgreSQL NUMERIC has no bounded Arrow Decimal128 mapping".into(),
        );
    }
    let modifier = type_modifier - 4;
    let precision = u8::try_from((modifier >> 16) & 0xffff)
        .map_err(|_| "PostgreSQL NUMERIC precision exceeds Arrow Decimal128".to_owned())?;
    let raw_scale = modifier & 0x7ff;
    let scale = if raw_scale & 0x400 != 0 {
        raw_scale - 0x800
    } else {
        raw_scale
    };
    let scale = i8::try_from(scale)
        .map_err(|_| "PostgreSQL NUMERIC scale exceeds Arrow Decimal128".to_owned())?;
    if precision == 0 || precision > 38 {
        return Err(format!(
            "PostgreSQL NUMERIC precision {precision} exceeds Arrow Decimal128"
        ));
    }
    if scale > 0 && scale.unsigned_abs() > precision {
        return Err(format!(
            "PostgreSQL NUMERIC scale {scale} exceeds precision {precision} and has no valid Arrow Decimal128 mapping"
        ));
    }
    Ok((precision, scale))
}

fn create_table_sql(relation: &Relation, schema: &Schema, columns: &[PgColumn]) -> Result<String> {
    let dialect = dialect_for(DatabaseKind::PostgreSql);
    let definitions = schema
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
    Ok(format!(
        "CREATE TABLE {} ({})",
        dialect.quote_relation(relation),
        definitions.join(", ")
    ))
}

fn quoted_columns(schema: &Schema) -> Result<String> {
    let dialect = dialect_for(DatabaseKind::PostgreSql);
    schema
        .fields()
        .iter()
        .map(|field| Identifier::new(field.name()).map(|name| dialect.quote_identifier(&name)))
        .collect::<Result<Vec<_>>>()
        .map(|columns| columns.join(", "))
}

fn quoted_columns_with_batch(schema: &Schema, batch_column: &Identifier) -> Result<String> {
    let dialect = dialect_for(DatabaseKind::PostgreSql);
    let columns = quoted_columns(schema)?;
    Ok(format!(
        "{columns}, {}",
        dialect.quote_identifier(batch_column)
    ))
}

fn ensure_batch_schema(actual: &Schema, expected: &Schema) -> Result<()> {
    let matches = actual.fields().len() == expected.fields().len()
        && actual
            .fields()
            .iter()
            .zip(expected.fields())
            .all(|(left, right)| {
                left.name() == right.name() && left.data_type() == right.data_type()
            });
    matches.then_some(()).ok_or_else(|| {
        ElmError::TypeMapping("PostgreSQL batch schema changed after preflight".into())
    })
}

async fn relation_exists<C>(client: &C, relation: &Relation) -> Result<bool>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            "SELECT EXISTS (
                SELECT 1 FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = COALESCE($1, current_schema())
                  AND c.relname = $2
                  AND c.relkind IN ('r', 'p')
            )",
            &[
                &relation.schema.as_ref().map(Identifier::as_str),
                &relation.name.as_str(),
            ],
        )
        .await
        .map_err(metadata_error)?;
    row.try_get(0).map_err(|_| metadata_error(()))
}

async fn relation_row_count<C>(client: &C, relation: &Relation) -> Result<u64>
where
    C: GenericClient + Sync,
{
    let query = format!(
        "SELECT COUNT(*)::bigint FROM {}",
        dialect_for(DatabaseKind::PostgreSql).quote_relation(relation)
    );
    let row = client
        .query_one(&query, &[])
        .await
        .map_err(metadata_error)?;
    let count: i64 = row.try_get(0).map_err(|_| metadata_error(()))?;
    u64::try_from(count)
        .map_err(|_| ElmError::State("PostgreSQL returned a negative row count".into()))
}

async fn relation_comment<C>(client: &C, relation: &Relation) -> Result<Option<String>>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            "SELECT pg_catalog.obj_description(c.oid, 'pg_class')
             FROM pg_catalog.pg_class c
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = COALESCE($1, current_schema()) AND c.relname = $2",
            &[
                &relation.schema.as_ref().map(Identifier::as_str),
                &relation.name.as_str(),
            ],
        )
        .await
        .map_err(metadata_error)?;
    row.try_get(0).map_err(|_| metadata_error(()))
}

async fn require_create_privilege(client: &Client, relation: &Relation) -> Result<()> {
    let row = client
        .query_one(
            "SELECT has_schema_privilege(COALESCE($1, current_schema()), 'CREATE')",
            &[&relation.schema.as_ref().map(Identifier::as_str)],
        )
        .await
        .map_err(metadata_error)?;
    let allowed: bool = row.try_get(0).map_err(|_| metadata_error(()))?;
    allowed.then_some(()).ok_or_else(|| {
        ElmError::PermissionDenied(
            "PostgreSQL account lacks CREATE privilege in the destination schema".into(),
        )
    })
}

async fn require_table_privilege(
    client: &Client,
    relation: &Relation,
    privilege: &str,
) -> Result<()> {
    let row = client
        .query_one(
            "SELECT has_table_privilege(c.oid, $3)
             FROM pg_catalog.pg_class c
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = COALESCE($1, current_schema()) AND c.relname = $2",
            &[
                &relation.schema.as_ref().map(Identifier::as_str),
                &relation.name.as_str(),
                &privilege,
            ],
        )
        .await
        .map_err(metadata_error)?;
    let allowed: bool = row.try_get(0).map_err(|_| metadata_error(()))?;
    allowed.then_some(()).ok_or_else(|| {
        ElmError::PermissionDenied(format!(
            "PostgreSQL account lacks {privilege} privilege on the destination"
        ))
    })
}

async fn require_relation_ownership(client: &Client, relation: &Relation) -> Result<()> {
    let row = client
        .query_one(
            "SELECT c.relowner = current_user::regrole OR pg_has_role(c.relowner, 'MEMBER')
             FROM pg_catalog.pg_class c
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = COALESCE($1, current_schema()) AND c.relname = $2",
            &[
                &relation.schema.as_ref().map(Identifier::as_str),
                &relation.name.as_str(),
            ],
        )
        .await
        .map_err(metadata_error)?;
    let allowed: bool = row.try_get(0).map_err(|_| metadata_error(()))?;
    allowed.then_some(()).ok_or_else(|| {
        ElmError::PermissionDenied(
            "PostgreSQL REPLACE requires ownership of the destination relation".into(),
        )
    })
}

fn date_to_epoch_days(value: NaiveDate) -> Result<i32> {
    let epoch = postgres_epoch_date()?;
    i32::try_from((value - epoch).num_days())
        .map_err(|_| ElmError::TypeMapping("PostgreSQL date is outside Arrow Date32 range".into()))
}

fn epoch_days_to_date(value: i32) -> Result<NaiveDate> {
    postgres_epoch_date()?
        .checked_add_signed(Duration::days(i64::from(value)))
        .ok_or_else(|| {
            ElmError::TypeMapping("Arrow Date32 is outside PostgreSQL date range".into())
        })
}

fn postgres_epoch_date() -> Result<NaiveDate> {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .ok_or_else(|| ElmError::Internal("cannot construct Unix epoch date".into()))
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn copy_decode_error(_error: tokio_postgres::Error) -> ElmError {
    ElmError::Connection {
        message: "PostgreSQL binary COPY row decoding failed".into(),
        retryable: false,
    }
}

fn postgres_connection_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "PostgreSQL connection, authentication, or TLS verification failed".into(),
        retryable: true,
    }
}

fn metadata_error<T>(_error: T) -> ElmError {
    ElmError::Connection {
        message: "PostgreSQL metadata query failed".into(),
        retryable: true,
    }
}

fn staging_permission_error(_error: tokio_postgres::Error) -> ElmError {
    ElmError::PermissionDenied(
        "PostgreSQL staging table could not be created or cleaned with this account".into(),
    )
}

fn publication_error(_error: tokio_postgres::Error) -> ElmError {
    ElmError::Connection {
        message: "PostgreSQL atomic publication failed; the original destination was retained"
            .into(),
        retryable: false,
    }
}

/// Resolves `relation`'s physical identity, unwinding one level of view indirection through
/// `pg_depend`/`pg_rewrite` (never by parsing view SQL text). Returns `Ok(None)` rather than a
/// guess whenever the relation is missing, is a view depending on more than one base table, or
/// the connected role cannot read `pg_control_system()`'s cluster-wide `system_identifier`.
pub(crate) async fn resolve_physical_identity(
    environment: &Environment,
    password: &str,
    relation: &Relation,
) -> Result<Option<crate::physical_identity::PhysicalIdentity>> {
    let session = connect(environment, password).await?;
    let client = &session.client;
    let Some(row) = client
        .query_opt(
            "SELECT n.nspname, c.relname, c.relkind::text, c.oid
             FROM pg_catalog.pg_class c
             JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = COALESCE($1, current_schema()) AND c.relname = $2",
            &[
                &relation.schema.as_ref().map(Identifier::as_str),
                &relation.name.as_str(),
            ],
        )
        .await
        .map_err(metadata_error)?
    else {
        return Ok(None);
    };
    let mut schema: String = row.try_get(0).map_err(|_| metadata_error(()))?;
    let mut name: String = row.try_get(1).map_err(|_| metadata_error(()))?;
    let relkind: String = row.try_get(2).map_err(|_| metadata_error(()))?;
    let oid: tokio_postgres::types::Oid = row.try_get(3).map_err(|_| metadata_error(()))?;
    match relkind.as_str() {
        "r" | "p" => {}
        "v" => {
            let dependents = client
                .query(
                    "SELECT DISTINCT dep_ns.nspname, dep_class.relname
                     FROM pg_rewrite r
                     JOIN pg_depend d ON d.objid = r.oid AND d.refobjid <> r.ev_class
                     JOIN pg_catalog.pg_class dep_class ON dep_class.oid = d.refobjid
                     JOIN pg_catalog.pg_namespace dep_ns ON dep_ns.oid = dep_class.relnamespace
                     WHERE r.ev_class = $1 AND dep_class.relkind IN ('r', 'p')",
                    &[&oid],
                )
                .await
                .map_err(metadata_error)?;
            let [only] = dependents.as_slice() else {
                return Ok(None);
            };
            schema = only.try_get(0).map_err(|_| metadata_error(()))?;
            name = only.try_get(1).map_err(|_| metadata_error(()))?;
        }
        _ => return Ok(None),
    }
    let Ok(identity_row) = client
        .query_one(
            "SELECT system_identifier::text FROM pg_control_system()",
            &[],
        )
        .await
    else {
        return Ok(None);
    };
    let Ok(system_identifier) = identity_row.try_get::<_, String>(0) else {
        return Ok(None);
    };
    Ok(Some(crate::physical_identity::PhysicalIdentity {
        server_fingerprint: format!("postgresql:{system_identifier}"),
        database: environment.database.clone(),
        base_schema: Some(schema),
        base_relation: name,
    }))
}

#[cfg(test)]
mod tests {
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use bytes::BytesMut;
    use elm_core::{Identifier, Relation};

    use super::{
        PgColumn, create_table_sql, decode_pg_numeric, encode_pg_numeric, postgres_type_to_arrow,
        selection_query,
    };

    fn numeric_modifier(precision: i32, scale: i32) -> i32 {
        ((precision << 16) | (scale & 0x7ff)) + 4
    }

    #[test]
    fn maps_supported_postgres_types_without_loss() {
        assert_eq!(
            postgres_type_to_arrow(&postgres_types::Type::INT8, -1),
            Ok(DataType::Int64)
        );
        assert_eq!(
            postgres_type_to_arrow(&postgres_types::Type::TIMESTAMPTZ, -1),
            Ok(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into())
            ))
        );
        assert_eq!(
            postgres_type_to_arrow(&postgres_types::Type::NUMERIC, numeric_modifier(18, 4)),
            Ok(DataType::Decimal128(18, 4))
        );
        assert_eq!(
            postgres_type_to_arrow(&postgres_types::Type::NUMERIC, numeric_modifier(8, -2)),
            Ok(DataType::Decimal128(8, -2))
        );
        assert!(postgres_type_to_arrow(&postgres_types::Type::NUMERIC, -1).is_err());
        assert!(
            postgres_type_to_arrow(&postgres_types::Type::NUMERIC, numeric_modifier(3, 5)).is_err()
        );
    }

    #[test]
    fn postgres_numeric_binary_round_trips_exact_decimal_coefficients() {
        for (coefficient, scale) in [
            (0_i128, 4_i8),
            (12, 4),
            (12_000, 4),
            (-123_456_700, 4),
            (123, -2),
            (99_999_999_999_999_999, 0),
            (10_i128.pow(38) - 1, 1),
            (-(10_i128.pow(38) - 1), 3),
            (10_i128.pow(38) - 1, -2),
            (10_i128.pow(38) - 1, 38),
        ] {
            let mut encoded = BytesMut::new();
            encode_pg_numeric(coefficient, scale, &mut encoded)
                .unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(
                decode_pg_numeric(&encoded, scale).unwrap_or_else(|error| panic!("{error}")),
                coefficient
            );
        }
    }

    #[test]
    fn rejects_catalogs_and_query_terminators() {
        let selection = elm_core::DatabaseSelection::Table {
            relation: Relation {
                catalog: Some(Identifier::new("other").unwrap_or_else(|error| panic!("{error}"))),
                schema: None,
                name: Identifier::new("items").unwrap_or_else(|error| panic!("{error}")),
            },
        };
        assert!(selection_query(&selection).is_err());
        assert!(
            selection_query(&elm_core::DatabaseSelection::Query {
                sql: "SELECT 1;".into()
            })
            .is_err()
        );
    }

    #[test]
    fn generates_quoted_staging_ddl() {
        let relation = Relation {
            catalog: None,
            schema: Some(Identifier::new("Mixed Schema").unwrap_or_else(|error| panic!("{error}"))),
            name: Identifier::new("stage").unwrap_or_else(|error| panic!("{error}")),
        };
        let schema = Schema::new(vec![
            Field::new("select", DataType::Int64, false),
            Field::new("payload", DataType::Binary, true),
        ]);
        let columns = schema
            .fields()
            .iter()
            .map(|field| PgColumn::from_field(field))
            .collect::<elm_core::Result<Vec<_>>>()
            .unwrap_or_else(|error| panic!("{error}"));
        let ddl = create_table_sql(&relation, &schema, &columns)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            ddl,
            "CREATE TABLE \"Mixed Schema\".\"stage\" (\"select\" BIGINT NOT NULL, \"payload\" BYTEA)"
        );
    }
}
