use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, RecordBatch, StringArray, UInt8Array,
};
use arrow_array::{Date32Array, Decimal128Array, TimestampMicrosecondArray};
use arrow_schema::TimeUnit;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use chrono::NaiveDate;
use elm_core::{DataSource, DatabaseKind, DatabaseSelection, ElmError, Environment, Result};
use odbc_api::{
    ColumnDescription, Cursor, ResultSetMetadata,
    buffers::{AnySlice, BufferDesc, ColumnarAnyBuffer},
};
use secrecy::{ExposeSecret, SecretString};
use tokio::sync::{mpsc, oneshot};

use crate::{
    dialect_for,
    native_diagnostics::{sql_server_connection_string, sql_server_manager},
};

type BatchReply = oneshot::Sender<Result<Option<RecordBatch>>>;
struct Request {
    target_bytes: usize,
    reply: BatchReply,
}

/// Request-driven, typed ODBC block fetches owned by one blocking worker.
pub struct SqlServerSource {
    schema: Arc<Schema>,
    requests: mpsc::Sender<Request>,
    cancelled: Arc<AtomicBool>,
    rows: u64,
    exhausted: bool,
}

impl Drop for SqlServerSource {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl SqlServerSource {
    pub async fn connect(
        environment: &Environment,
        password: &str,
        selection: &DatabaseSelection,
    ) -> Result<Self> {
        let connection = sql_server_connection_string(environment, password)?;
        let query = match selection {
            DatabaseSelection::Table { relation } => format!(
                "SELECT * FROM {}",
                dialect_for(DatabaseKind::SqlServer).quote_relation(relation)
            ),
            DatabaseSelection::Query { sql } => sql.clone(),
        };
        let (requests, receiver) = mpsc::channel(1);
        let (ready, schema_receiver) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut source = Self {
            schema: Arc::new(Schema::empty()),
            requests,
            cancelled: cancelled.clone(),
            rows: 0,
            exhausted: false,
        };
        tokio::task::spawn_blocking(move || {
            let mut ready = Some(ready);
            if let Err(error) = source_worker(connection, query, receiver, &mut ready, cancelled)
                && let Some(ready) = ready.take()
            {
                let _sent = ready.send(Err(error));
            }
        });
        source.schema = schema_receiver.await.map_err(|_| source_error())??;
        Ok(source)
    }
}

#[async_trait]
impl DataSource for SqlServerSource {
    async fn schema(&mut self) -> Result<Arc<Schema>> {
        Ok(self.schema.clone())
    }
    async fn next_batch(&mut self, target_bytes: usize) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        let (reply, receiver) = oneshot::channel();
        self.requests
            .send(Request {
                target_bytes,
                reply,
            })
            .await
            .map_err(|_| source_error())?;
        let batch = receiver.await.map_err(|_| source_error())??;
        if let Some(batch) = &batch {
            self.rows = self.rows.saturating_add(batch.num_rows() as u64);
        } else {
            self.exhausted = true;
        }
        Ok(batch)
    }
    async fn checkpoint(&mut self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"row": self.rows}))
    }
}

fn source_error() -> ElmError {
    ElmError::Connection { message: "SQL Server source query or block fetch failed; verify connectivity, query permissions, and driver timeout settings".into(), retryable: false }
}

fn source_worker(
    connection_string: SecretString,
    query: String,
    mut requests: mpsc::Receiver<Request>,
    ready: &mut Option<oneshot::Sender<Result<Arc<Schema>>>>,
    cancelled: Arc<AtomicBool>,
) -> Result<()> {
    let manager = sql_server_manager()?;
    if cancelled.load(Ordering::Acquire) {
        return Err(ElmError::Cancelled);
    }
    let connection = manager
        .connect_with_connection_string(
            connection_string.expose_secret(),
            odbc_api::ConnectionOptions {
                login_timeout_sec: Some(15),
                ..Default::default()
            },
        )
        .map_err(|_| source_error())?;
    if cancelled.load(Ordering::Acquire) {
        return Err(ElmError::Cancelled);
    }
    let mut cursor = connection
        .execute(&query, (), Some(15))
        .map_err(|_| source_error())?
        .ok_or_else(|| {
            ElmError::Validation("SQL Server source query did not return a result set".into())
        })?;
    let count = cursor.num_result_cols().map_err(|_| source_error())?;
    let mut fields = Vec::new();
    let mut descriptions = Vec::new();
    for index in 1..=count {
        let mut column = ColumnDescription::default();
        cursor
            .describe_col(index as u16, &mut column)
            .map_err(|_| source_error())?;
        let (field, description) = map_column(&column)?;
        if fields
            .iter()
            .any(|other: &Field| other.name() == field.name())
        {
            return Err(ElmError::TypeMapping(
                "SQL Server query returned duplicate column names; supply unique aliases".into(),
            ));
        }
        fields.push(field);
        descriptions.push(description);
    }
    let schema = Arc::new(Schema::new(fields));
    let row_bytes = descriptions
        .iter()
        .try_fold(0usize, |total, desc| {
            total.checked_add(desc.bytes_per_row())
        })
        .ok_or_else(|| ElmError::TypeMapping("SQL Server row buffer size overflow".into()))?;
    if row_bytes == 0 {
        return Err(ElmError::TypeMapping(
            "SQL Server source has no supported columns".into(),
        ));
    }
    if let Some(ready) = ready.take()
        && ready.send(Ok(schema.clone())).is_err()
    {
        return Ok(());
    }
    let mut cached = None;
    while let Some(request) = requests.blocking_recv() {
        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        let capacity = batch_capacity(row_bytes, request.target_bytes);
        let capacity = match capacity {
            Ok(capacity) => capacity,
            Err(error) => {
                let _sent = request.reply.send(Err(error));
                return Ok(());
            }
        };
        let buffer = match cached.take() {
            Some((previous_capacity, buffer)) if previous_capacity == capacity => buffer,
            _ => ColumnarAnyBuffer::try_from_descs(capacity, descriptions.iter().copied())
                .map_err(|_| source_error())?,
        };
        let mut bound = cursor.bind_buffer(buffer).map_err(|_| source_error())?;
        let result = bound.fetch_with_truncation_check(true)
            .map_err(|_| ElmError::TypeMapping("SQL Server block fetch failed or truncated a value; no partial batch was accepted".into()))
            .and_then(|batch| batch.map(|batch| convert_batch(batch, schema.clone())).transpose());
        let stop = result.is_err();
        let exhausted = matches!(result, Ok(None));
        let (unbound, buffer) = bound.unbind().map_err(|_| source_error())?;
        if exhausted {
            let result = match unbound.more_results().map_err(|_| source_error())? {
                None => Ok(None),
                Some(_) => Err(ElmError::Validation(
                    "SQL Server source queries must return exactly one result set".into(),
                )),
            };
            let _sent = request.reply.send(result);
            return Ok(());
        }
        cursor = unbound;
        cached = Some((capacity, buffer));
        if request.reply.send(result).is_err() || stop {
            return Ok(());
        }
    }
    Ok(())
}

fn batch_capacity(row_bytes: usize, requested: usize) -> Result<usize> {
    // Account conservatively for native storage, Arrow output, and UTF-16 decoding.
    let bytes_per_row = row_bytes
        .checked_mul(4)
        .ok_or_else(|| ElmError::TypeMapping("SQL Server row size overflow".into()))?;
    let capacity = requested
        .min(8 * 1024 * 1024)
        .checked_div(bytes_per_row)
        .unwrap_or(0)
        .min(65_536);
    if capacity == 0 {
        return Err(ElmError::TypeMapping("SQL Server declared row width exceeds the requested batch budget; project smaller columns or increase the batch size".into()));
    }
    Ok(capacity)
}

pub(crate) fn map_column(column: &ColumnDescription) -> Result<(Field, BufferDesc)> {
    use odbc_api::DataType as Sql;
    let (data_type, buffer) = match column.data_type {
        Sql::Integer => (DataType::Int32, BufferDesc::I32 { nullable: true }),
        Sql::SmallInt => (DataType::Int16, BufferDesc::I16 { nullable: true }),
        Sql::BigInt => (DataType::Int64, BufferDesc::I64 { nullable: true }),
        Sql::TinyInt => (DataType::UInt8, BufferDesc::U8 { nullable: true }),
        Sql::Bit => (DataType::Boolean, BufferDesc::Bit { nullable: true }),
        Sql::Real | Sql::Float { precision: 1..=24 } => (DataType::Float32, BufferDesc::F32 { nullable: true }),
        Sql::Double | Sql::Float { precision: 25..=53 } => (DataType::Float64, BufferDesc::F64 { nullable: true }),
        Sql::Date => (DataType::Date32, BufferDesc::Date { nullable: true }),
        Sql::Timestamp { precision: 0..=6 } => (DataType::Timestamp(TimeUnit::Microsecond, None), BufferDesc::Timestamp { nullable: true }),
        Sql::Decimal { precision: precision @ 1..=38, scale } | Sql::Numeric { precision: precision @ 1..=38, scale }
            if scale >= 0 && scale as usize <= precision => {
                // ODBC's exact character representation avoids the default precision/scale
                // of SQL_C_NUMERIC and never passes a decimal through floating point.
                (DataType::Decimal128(precision as u8, scale as i8), BufferDesc::Text { max_str_len: 41 })
            }
        Sql::Char { length: Some(length) } | Sql::WChar { length: Some(length) } |
        Sql::Varchar { length: Some(length) } | Sql::WVarchar { length: Some(length) } if length.get() <= 8000 =>
            (DataType::Utf8, BufferDesc::WText { max_str_len: length.get() }),
        Sql::Binary { length: Some(length) } | Sql::Varbinary { length: Some(length) } if length.get() <= 8000 =>
            (DataType::Binary, BufferDesc::Binary { length: length.get() }),
        _ => return Err(ElmError::TypeMapping("SQL Server column has no enabled lossless mapping; project a supported type explicitly (LOBs, TIME, DATETIMEOFFSET, and sub-microsecond timestamps are pending)".into())),
    };
    let name = column
        .name_to_string()
        .map_err(|_| ElmError::TypeMapping("SQL Server column name is not valid Unicode".into()))?;
    if name.is_empty() {
        return Err(ElmError::TypeMapping(
            "SQL Server expressions require named aliases".into(),
        ));
    }
    Ok((
        Field::new(name, data_type, column.could_be_nullable()),
        buffer,
    ))
}

fn convert_batch(batch: &ColumnarAnyBuffer, schema: Arc<Schema>) -> Result<RecordBatch> {
    let columns = (0..schema.fields().len())
        .map(|index| convert_typed_column(batch.column(index), schema.field(index).data_type()))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(schema, columns).map_err(|_| {
        ElmError::TypeMapping("SQL Server batch does not match its discovered schema".into())
    })
}

fn convert_typed_column(column: AnySlice<'_>, data_type: &DataType) -> Result<ArrayRef> {
    if let DataType::Decimal128(precision, scale) = *data_type {
        let AnySlice::Text(values) = column else {
            return Err(ElmError::Internal(
                "SQL Server decimal was not bound as exact text".into(),
            ));
        };
        let values = values
            .iter()
            .map(|value| {
                value
                    .map(|value| decimal_coefficient(value, precision, scale))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let array = Decimal128Array::from(values)
            .with_precision_and_scale(precision, scale)
            .map_err(|_| ElmError::TypeMapping("SQL Server decimal metadata is invalid".into()))?;
        return Ok(Arc::new(array));
    }
    convert_column(column)
}

pub(crate) fn decimal_coefficient(value: &[u8], precision: u8, scale: i8) -> Result<i128> {
    let invalid = || {
        ElmError::TypeMapping(
            "SQL Server decimal is invalid, out of range, or would lose precision".into(),
        )
    };
    if !(1..=38).contains(&precision) || scale < 0 || scale as u8 > precision {
        return Err(invalid());
    }
    let (negative, digits) = match value.first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    let mut coefficient = 0i128;
    let mut fractional = None;
    let mut count = 0;
    for &digit in digits {
        if digit == b'.' && fractional.is_none() {
            fractional = Some(0u32);
            continue;
        }
        if !digit.is_ascii_digit() {
            return Err(invalid());
        }
        coefficient = coefficient
            .checked_mul(10)
            .and_then(|value| value.checked_add(i128::from(digit - b'0')))
            .ok_or_else(invalid)?;
        count += 1;
        if let Some(fractional) = &mut fractional {
            *fractional += 1;
        }
    }
    let fractional = fractional.unwrap_or(0);
    if count == 0 || fractional > scale as u32 {
        return Err(invalid());
    }
    coefficient = coefficient
        .checked_mul(10_i128.pow(scale as u32 - fractional))
        .ok_or_else(invalid)?;
    if coefficient >= 10_i128.pow(u32::from(precision)) {
        return Err(invalid());
    }
    Ok(if negative { -coefficient } else { coefficient })
}

fn native_date(year: i16, month: u16, day: u16) -> Result<NaiveDate> {
    if !(1..=9999).contains(&year) {
        return Err(ElmError::TypeMapping(
            "SQL Server date year is out of range".into(),
        ));
    }
    NaiveDate::from_ymd_opt(i32::from(year), u32::from(month), u32::from(day))
        .ok_or_else(|| ElmError::TypeMapping("SQL Server date is invalid".into()))
}

pub(crate) fn native_timestamp(value: &odbc_api::sys::Timestamp) -> Result<i64> {
    if value.fraction >= 1_000_000_000 || !value.fraction.is_multiple_of(1000) {
        return Err(ElmError::TypeMapping(
            "SQL Server timestamp cannot be represented exactly in microseconds".into(),
        ));
    }
    let date = native_date(value.year, value.month, value.day)?;
    let timestamp = date
        .and_hms_nano_opt(
            u32::from(value.hour),
            u32::from(value.minute),
            u32::from(value.second),
            value.fraction,
        )
        .ok_or_else(|| ElmError::TypeMapping("SQL Server timestamp is invalid".into()))?;
    Ok(timestamp.and_utc().timestamp_micros())
}

fn convert_column(column: AnySlice<'_>) -> Result<ArrayRef> {
    Ok(match column {
        AnySlice::NullableDate(values) => {
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)
                .ok_or_else(|| ElmError::Internal("invalid epoch".into()))?;
            let values = values
                .map(|value| {
                    value
                        .map(|value| {
                            native_date(value.year, value.month, value.day).and_then(|date| {
                                i32::try_from(date.signed_duration_since(epoch).num_days()).map_err(
                                    |_| {
                                        ElmError::TypeMapping(
                                            "SQL Server date exceeds Date32".into(),
                                        )
                                    },
                                )
                            })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(Date32Array::from(values))
        }
        AnySlice::NullableTimestamp(values) => {
            let values = values
                .map(|value| value.map(native_timestamp).transpose())
                .collect::<Result<Vec<_>>>()?;
            Arc::new(TimestampMicrosecondArray::from(values))
        }
        AnySlice::NullableI16(values) => {
            Arc::new(Int16Array::from_iter(values.map(|value| value.copied())))
        }
        AnySlice::NullableI32(values) => {
            Arc::new(Int32Array::from_iter(values.map(|value| value.copied())))
        }
        AnySlice::NullableI64(values) => {
            Arc::new(Int64Array::from_iter(values.map(|value| value.copied())))
        }
        AnySlice::NullableU8(values) => {
            Arc::new(UInt8Array::from_iter(values.map(|value| value.copied())))
        }
        AnySlice::NullableF32(values) => {
            Arc::new(Float32Array::from_iter(values.map(|value| value.copied())))
        }
        AnySlice::NullableF64(values) => {
            Arc::new(Float64Array::from_iter(values.map(|value| value.copied())))
        }
        AnySlice::NullableBit(values) => Arc::new(BooleanArray::from_iter(
            values.map(|value| value.map(|bit| bit.0 != 0)),
        )),
        AnySlice::Binary(values) => Arc::new(BinaryArray::from_iter(values.iter())),
        AnySlice::WText(values) => {
            let values = values
                .iter()
                .map(|value| {
                    value
                        .map(|value| String::from_utf16(value.as_slice()))
                        .transpose()
                        .map_err(|_| {
                            ElmError::TypeMapping("SQL Server text contains invalid UTF-16".into())
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(StringArray::from(values))
        }
        _ => {
            return Err(ElmError::Internal(
                "SQL Server returned an unexpected native buffer type".into(),
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use odbc_api::{
        DataType as Sql, Nullability,
        buffers::{AnyBuffer, BinColumn, ColumnBuffer, WCharColumn},
    };

    #[test]
    fn decimals_preserve_scale_and_precision_boundaries() {
        for (value, precision, scale, expected) in [
            ("-12345.6700", 18, 4, -123_456_700),
            ("0.00000000000000000000000000000000000001", 38, 38, 1),
            (
                "99999999999999999999999999999999999999",
                38,
                0,
                10_i128.pow(38) - 1,
            ),
            (
                "-99999999999999999999999999999999999999",
                38,
                0,
                -(10_i128.pow(38) - 1),
            ),
            ("12", 5, 2, 1200),
        ] {
            assert_eq!(
                decimal_coefficient(value.as_bytes(), precision, scale)
                    .unwrap_or_else(|error| panic!("{error}")),
                expected
            );
        }
        for value in [
            "",
            "-",
            ".",
            "NaN",
            "1e2",
            "1.2.3",
            "1.001",
            "1000",
            "999999999999999999999999999999999999999999",
        ] {
            assert!(
                decimal_coefficient(value.as_bytes(), 5, 2).is_err(),
                "accepted {value}"
            );
        }
        let column = ColumnDescription::new(
            "amount",
            Sql::Decimal {
                precision: 38,
                scale: 38,
            },
            Nullability::Nullable,
        );
        let (field, _) = map_column(&column).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(field.data_type(), &DataType::Decimal128(38, 38));
        let mut text = odbc_api::buffers::CharColumn::new(2, 41);
        text.set_value(0, Some(b"-12345.6700"));
        text.set_value(1, None);
        let array =
            convert_typed_column(AnySlice::Text(text.view(2)), &DataType::Decimal128(18, 4))
                .unwrap_or_else(|error| panic!("{error}"));
        let array = array
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap_or_else(|| panic!("expected decimal"));
        assert_eq!(array.value(0), -123_456_700);
        assert!(array.is_null(1));
    }

    #[test]
    fn timestamps_preserve_microseconds_and_reject_rounding_and_invalid_dates() {
        let mut timestamp = odbc_api::sys::Timestamp {
            year: 2024,
            month: 2,
            day: 29,
            hour: 12,
            minute: 34,
            second: 56,
            fraction: 123_456_000,
        };
        let expected = NaiveDate::from_ymd_opt(2024, 2, 29)
            .and_then(|date| date.and_hms_micro_opt(12, 34, 56, 123_456))
            .unwrap_or_else(|| panic!("invalid fixture"))
            .and_utc()
            .timestamp_micros();
        assert_eq!(
            native_timestamp(&timestamp).unwrap_or_else(|error| panic!("{error}")),
            expected
        );
        timestamp.fraction = 123_456_100;
        assert!(native_timestamp(&timestamp).is_err());
        timestamp.fraction = 1_000_000_000;
        timestamp.second = 59;
        assert!(native_timestamp(&timestamp).is_err());
        assert!(native_date(2023, 2, 29).is_err());
        assert!(native_date(0, 1, 1).is_err());
        assert!(native_date(9999, 12, 31).is_ok());
        for precision in 0..=6 {
            let (field, _) = map_column(&ColumnDescription::new(
                "created",
                Sql::Timestamp { precision },
                Nullability::Nullable,
            ))
            .unwrap_or_else(|error| panic!("{error}"));
            assert_eq!(
                field.data_type(),
                &DataType::Timestamp(TimeUnit::Microsecond, None)
            );
        }
    }

    #[test]
    fn schema_preserves_unsigned_tinyint_and_rejects_unbounded_types() {
        let (field, _) = map_column(&ColumnDescription::new(
            "tiny",
            Sql::TinyInt,
            Nullability::NoNulls,
        ))
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(field.data_type(), &DataType::UInt8);
        assert!(!field.is_nullable());
        for data_type in [
            Sql::WVarchar { length: None },
            Sql::WLongVarchar { length: None },
            Sql::Decimal {
                precision: 39,
                scale: 2,
            },
            Sql::Timestamp { precision: 7 },
        ] {
            assert!(
                map_column(&ColumnDescription::new(
                    "value",
                    data_type,
                    Nullability::Nullable
                ))
                .is_err()
            );
        }
        assert!(
            map_column(&ColumnDescription::new(
                "",
                Sql::Integer,
                Nullability::Nullable
            ))
            .is_err()
        );
    }

    #[test]
    fn batch_capacity_reserves_native_and_decoded_memory() {
        assert_eq!(
            batch_capacity(1024, 8192).unwrap_or_else(|error| panic!("{error}")),
            2
        );
        assert!(batch_capacity(8192, 1024).is_err());
        assert!(batch_capacity(usize::MAX, usize::MAX).is_err());
        assert_eq!(
            batch_capacity(1, usize::MAX).unwrap_or_else(|error| panic!("{error}")),
            65_536
        );
    }

    #[test]
    fn native_buffers_preserve_unicode_binary_nulls_and_integer_boundaries() {
        let text: Vec<u16> = "İstanbul 🌍\0end".encode_utf16().collect();
        let mut column = WCharColumn::new(3, 32);
        column.set_value(0, Some(&text));
        column.set_value(1, None);
        column.set_value(2, Some(&[]));
        let array = convert_column(AnySlice::WText(column.view(3)))
            .unwrap_or_else(|error| panic!("{error}"));
        let array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("expected text"));
        assert_eq!(array.value(0), "İstanbul 🌍\0end");
        assert!(array.is_null(1));
        assert_eq!(array.value(2), "");
        column.set_value(0, Some(&[0xD800]));
        assert!(convert_column(AnySlice::WText(column.view(1))).is_err());

        let mut binary = BinColumn::new(2, 4);
        binary.set_value(0, Some(&[0, 255, 92]));
        binary.set_value(1, None);
        let array = convert_column(AnySlice::Binary(binary.view(2)))
            .unwrap_or_else(|error| panic!("{error}"));
        let array = array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap_or_else(|| panic!("expected binary"));
        assert_eq!(array.value(0), &[0, 255, 92]);
        assert!(array.is_null(1));

        let AnyBuffer::NullableI64(mut integers) =
            AnyBuffer::from_desc(3, BufferDesc::I64 { nullable: true })
        else {
            panic!("expected integer buffer");
        };
        integers
            .writer_n(3)
            .write([Some(i64::MIN), None, Some(i64::MAX)].into_iter());
        let array = convert_column(AnySlice::NullableI64(integers.iter(3)))
            .unwrap_or_else(|error| panic!("{error}"));
        let array = array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("expected integers"));
        assert_eq!(array.value(0), i64::MIN);
        assert!(array.is_null(1));
        assert_eq!(array.value(2), i64::MAX);
    }
}
