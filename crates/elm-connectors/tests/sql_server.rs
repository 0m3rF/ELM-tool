#![cfg(feature = "sql-server")]

use arrow_array::{
    Array, BinaryArray, Date32Array, Decimal128Array, Int64Array, StringArray,
    TimestampMicrosecondArray, UInt8Array,
};
use elm_connectors::SqlServerSource;
use elm_core::{DataSource, DatabaseKind, DatabaseSelection, Environment};

fn environment() -> Environment {
    serde_json::from_value(serde_json::json!({
        "id": "00000000-0000-4000-8000-000000000001", "name": "SQL Server test", "kind": DatabaseKind::SqlServer,
        "host": std::env::var("ELM_TEST_SQL_SERVER_HOST").unwrap_or_else(|_| "localhost".into()),
        "port": std::env::var("ELM_TEST_SQL_SERVER_PORT").ok().and_then(|value| value.parse::<u16>().ok()).unwrap_or(1433),
        "database": std::env::var("ELM_TEST_SQL_SERVER_DATABASE").unwrap_or_else(|_| "elm_test".into()),
        "username": std::env::var("ELM_TEST_SQL_SERVER_USER").unwrap_or_else(|_| "sa".into()),
        "credential_ref": "integration-only",
        "options": {"ssl_mode": std::env::var("ELM_TEST_SQL_SERVER_SSL_MODE").unwrap_or_else(|_| "require".into())},
        "created_at": "2026-09-09T00:00:00Z", "updated_at": "2026-09-09T00:00:00Z"
    })).unwrap_or_else(|error| panic!("{error}"))
}

#[tokio::test]
#[ignore = "requires SQL Server Driver 18 and an isolated SQL Server test database"]
async fn block_source_preserves_typed_values_and_repeated_eof() {
    let environment = environment();
    let password = std::env::var("ELM_TEST_SQL_SERVER_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_SQL_SERVER_PASSWORD"));
    let selection = DatabaseSelection::Query { sql:
        "SELECT CAST(-9223372036854775808 AS BIGINT) AS id, CAST(255 AS TINYINT) AS tiny, CAST(N'İstanbul 🌍' AS NVARCHAR(32)) AS label, CAST(0x00FF5C AS VARBINARY(4)) AS payload, CAST(-12345.6700 AS DECIMAL(38,4)) AS amount, CAST('0001-01-01' AS DATE) AS day, CAST('2024-02-29T12:34:56.123456' AS DATETIME2(6)) AS created UNION ALL SELECT CAST(9223372036854775807 AS BIGINT), CAST(NULL AS TINYINT), CAST(NULL AS NVARCHAR(32)), CAST(NULL AS VARBINARY(4)), CAST(NULL AS DECIMAL(38,4)), CAST(NULL AS DATE), CAST(NULL AS DATETIME2(6))".into()
    };
    let mut source = SqlServerSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let first = source
        .next_batch(1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("missing first batch"));
    assert_eq!(first.num_rows(), 1);
    assert_eq!(
        first
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("integer"))
            .value(0),
        i64::MIN
    );
    assert_eq!(
        first
            .column(1)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap_or_else(|| panic!("tinyint"))
            .value(0),
        255
    );
    assert_eq!(
        first
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("text"))
            .value(0),
        "İstanbul 🌍"
    );
    assert_eq!(
        first
            .column(3)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap_or_else(|| panic!("binary"))
            .value(0),
        &[0, 255, 92]
    );
    assert_eq!(
        first
            .column(4)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap_or_else(|| panic!("decimal"))
            .value(0),
        -123_456_700
    );
    assert_eq!(
        first
            .column(5)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap_or_else(|| panic!("date"))
            .value(0),
        -719_162
    );
    let timestamp = chrono::NaiveDate::from_ymd_opt(2024, 2, 29)
        .and_then(|date| date.and_hms_micro_opt(12, 34, 56, 123_456))
        .unwrap_or_else(|| panic!("invalid timestamp fixture"))
        .and_utc()
        .timestamp_micros();
    assert_eq!(
        first
            .column(6)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap_or_else(|| panic!("timestamp"))
            .value(0),
        timestamp
    );
    let last = source
        .next_batch(1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("missing last batch"));
    assert_eq!(last.num_rows(), 1);
    assert_eq!(
        last.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("integer"))
            .value(0),
        i64::MAX
    );
    for column in 1..7 {
        assert!(last.column(column).is_null(0));
    }
    assert!(
        source
            .next_batch(1024)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .is_none()
    );
    assert!(
        source
            .next_batch(1024)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .is_none()
    );
    assert_eq!(
        source
            .checkpoint()
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        serde_json::json!({"row": 2})
    );
}

async fn target_rows(environment: &Environment, password: &str, table: &str) -> i64 {
    let mut source = SqlServerSource::connect(
        environment,
        password,
        &DatabaseSelection::Query {
            sql: format!("SELECT COUNT_BIG(*) AS rows FROM {table}"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let batch = source
        .next_batch(8192)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("missing count"));
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| panic!("count type"))
        .value(0)
}

#[tokio::test]
#[ignore = "requires SQL Server Driver 18 and an isolated database with CREATE TABLE and ALTER schema privileges"]
async fn staged_sink_recovers_batches_and_publishes_all_modes() {
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use elm_connectors::sql_server_sink::{
        SqlServerSink, cleanup_sql_server_staging, sql_server_published_rows,
    };
    use elm_core::{
        BatchCheckpoint, ConsistencyMode, DataSink, Identifier, JobId, Relation, WriteMode,
    };
    use std::sync::Arc;
    let environment = environment();
    let password = std::env::var("ELM_TEST_SQL_SERVER_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_SQL_SERVER_PASSWORD"));
    let name = format!("elm_sink_test_{}", JobId::new().0.simple());
    let target = Relation {
        catalog: None,
        schema: Some(Identifier::new("dbo").unwrap_or_else(|error| panic!("{error}"))),
        name: Identifier::new(&name).unwrap_or_else(|error| panic!("{error}")),
    };
    let table = format!("[dbo].[{name}]");
    let manager = odbc_api::Environment::new().unwrap_or_else(|_| panic!("ODBC unavailable"));
    let connection = fixture_connection(&manager, &environment, &password);
    connection
        .execute(
            &format!(
                "CREATE TABLE {table} (id BIGINT NOT NULL PRIMARY KEY, label NVARCHAR(4000) NULL)"
            ),
            (),
            Some(15),
        )
        .unwrap_or_else(|_| panic!("fixture creation failed"));
    connection
        .execute(
            &format!("INSERT INTO {table} VALUES (99, N'original')"),
            (),
            Some(15),
        )
        .unwrap_or_else(|_| panic!("fixture insert failed"));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("İstanbul 🌍"), None])),
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let checkpoint = BatchCheckpoint {
        sequence: 0,
        rows_committed: 2,
        bytes_committed: 0,
        source_fingerprint: None,
        source_position: serde_json::json!({"row": 2}),
    };
    let failed_job_id = JobId::new();
    let mut failed =
        SqlServerSink::connect(&environment, &password, target.clone(), failed_job_id, None)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    failed
        .preflight(schema.clone(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    failed
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let duplicates = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 1])),
            batch.column(1).clone(),
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    failed
        .write_batch(&duplicates)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    failed
        .commit_checkpoint(&checkpoint)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(failed.publish().await.is_err());
    assert_eq!(target_rows(&environment, &password, &table).await, 1);
    assert_eq!(
        sql_server_published_rows(&environment, &password, target.clone(), failed_job_id)
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        None
    );
    cleanup_sql_server_staging(&environment, &password, target.clone(), failed_job_id)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let mut sink = SqlServerSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema.clone(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.commit_checkpoint(&checkpoint)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(sink); // The second staging commit has no durable checkpoint.
    assert_eq!(target_rows(&environment, &password, &table).await, 1);
    let mut resumed = SqlServerSink::connect(
        &environment,
        &password,
        target.clone(),
        job_id,
        Some(&checkpoint),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .preflight(schema.clone(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(target_rows(&environment, &password, &table).await, 3);
    assert_eq!(
        sql_server_published_rows(&environment, &password, target.clone(), job_id)
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        Some(2)
    );
    cleanup_sql_server_staging(&environment, &password, target.clone(), job_id)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    for empty in [false, true] {
        let job_id = JobId::new();
        let mut sink =
            SqlServerSink::connect(&environment, &password, target.clone(), job_id, None)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
        sink.preflight(schema.clone(), WriteMode::Replace, ConsistencyMode::Atomic)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
        if !empty {
            sink.write_batch(&batch)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
            sink.commit_checkpoint(&checkpoint)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
        }
        sink.publish()
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            target_rows(&environment, &password, &table).await,
            if empty { 0 } else { 2 }
        );
        cleanup_sql_server_staging(&environment, &password, target.clone(), job_id)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    }
    let mut fail = SqlServerSink::connect(&environment, &password, target, JobId::new(), None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        fail.preflight(schema, WriteMode::Fail, ConsistencyMode::Atomic)
            .await
            .is_err()
    );
    connection
        .execute(&format!("DROP TABLE {table}"), (), Some(15))
        .unwrap_or_else(|_| panic!("fixture cleanup failed"));
}

fn fixture_connection<'a>(
    manager: &'a odbc_api::Environment,
    environment: &Environment,
    password: &str,
) -> odbc_api::Connection<'a> {
    let escape = odbc_api::escape_attribute_value;
    let connection_string = format!(
        "DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={};DATABASE={};UID={};PWD={};Encrypt={};TrustServerCertificate=no;",
        escape(&format!("tcp:{},{}", environment.host, environment.port)),
        escape(&environment.database),
        escape(&environment.username),
        escape(password),
        if environment.options["ssl_mode"] == "disable" {
            "no"
        } else {
            "yes"
        }
    );
    manager
        .connect_with_connection_string(
            &connection_string,
            odbc_api::ConnectionOptions {
                login_timeout_sec: Some(15),
                ..Default::default()
            },
        )
        .unwrap_or_else(|_| panic!("SQL Server test connection failed"))
}

#[tokio::test]
#[ignore = "requires SQL Server Driver 18 and an isolated writable SQL Server test database"]
async fn bulk_chunks_round_trip_every_supported_sink_type() {
    use arrow_array::{
        BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};
    use elm_connectors::sql_server_sink::{SqlServerSink, cleanup_sql_server_staging};
    use elm_core::{
        BatchCheckpoint, ConsistencyMode, DataSink, Identifier, JobId, Relation, WriteMode,
    };
    use std::sync::Arc;

    let environment = environment();
    let password = std::env::var("ELM_TEST_SQL_SERVER_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_SQL_SERVER_PASSWORD"));
    let job_id = JobId::new();
    let target = Relation {
        catalog: None,
        schema: Some(Identifier::new("dbo").unwrap_or_else(|error| panic!("{error}"))),
        name: Identifier::new(format!("elm_bulk_test_{}", job_id.0.simple()))
            .unwrap_or_else(|error| panic!("{error}")),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("tiny", DataType::UInt8, true),
        Field::new("small", DataType::Int16, true),
        Field::new("medium", DataType::Int32, true),
        Field::new("single", DataType::Float32, true),
        Field::new("double", DataType::Float64, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("payload", DataType::Binary, true),
        Field::new("amount", DataType::Decimal128(38, 4), true),
        Field::new("fraction", DataType::Decimal128(38, 38), true),
        Field::new("whole", DataType::Decimal128(38, 0), true),
        Field::new("day", DataType::Date32, true),
        Field::new(
            "created",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            true,
        ),
    ]));
    let nullable = |row| (row % 3 != 0).then_some(row);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter((0..513).map(Some))),
            Arc::new(BooleanArray::from_iter(
                (0..513).map(|row| nullable(row).map(|row| row % 2 == 0)),
            )),
            Arc::new(UInt8Array::from_iter(
                (0..513).map(|row| nullable(row).map(|_| u8::MAX)),
            )),
            Arc::new(Int16Array::from_iter(
                (0..513).map(|row| nullable(row).map(|_| i16::MIN)),
            )),
            Arc::new(Int32Array::from_iter(
                (0..513).map(|row| nullable(row).map(|_| i32::MAX)),
            )),
            Arc::new(Float32Array::from_iter(
                (0..513).map(|row| nullable(row).map(|_| -12.5)),
            )),
            Arc::new(Float64Array::from_iter(
                (0..513).map(|row| nullable(row).map(|_| 1.25e100)),
            )),
            Arc::new(StringArray::from_iter(
                (0..513).map(|row| nullable(row).map(|_| "İstanbul 🌍\0end")),
            )),
            Arc::new(BinaryArray::from_iter(
                (0..513).map(|row| nullable(row).map(|_| &[0, 255, 92][..])),
            )),
            Arc::new(
                Decimal128Array::from_iter((0..513).map(|row| {
                    nullable(row).map(|row| {
                        if row % 2 == 0 {
                            10i128.pow(38) - 1
                        } else {
                            1 - 10i128.pow(38)
                        }
                    })
                }))
                .with_precision_and_scale(38, 4)
                .unwrap_or_else(|error| panic!("{error}")),
            ),
            Arc::new(
                Decimal128Array::from_iter((0..513).map(|row| {
                    nullable(row).map(|row| if row % 2 == 0 { -1 } else { 10i128.pow(38) - 1 })
                }))
                .with_precision_and_scale(38, 38)
                .unwrap_or_else(|error| panic!("{error}")),
            ),
            Arc::new(
                Decimal128Array::from_iter((0..513).map(|row| {
                    nullable(row).map(|row| if row % 2 == 0 { 0 } else { 1 - 10i128.pow(38) })
                }))
                .with_precision_and_scale(38, 0)
                .unwrap_or_else(|error| panic!("{error}")),
            ),
            Arc::new(Date32Array::from_iter((0..513).map(|row| match row % 5 {
                0 => None,
                1 => Some(-719_162),
                2 => Some(2_932_896),
                3 => Some(-1),
                _ => Some(19_782),
            }))),
            Arc::new(TimestampMicrosecondArray::from_iter((0..513).map(
                |row| match row % 6 {
                    0 => None,
                    1 => Some(-62_135_596_800_000_000),
                    2 => Some(253_402_300_799_999_999),
                    3 => Some(-1),
                    4 => Some(0),
                    _ => Some(1_709_210_096_123_456),
                },
            ))),
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let mut sink = SqlServerSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema, WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    let mut invalid_columns = batch.columns().to_vec();
    invalid_columns[9] = Arc::new(
        Decimal128Array::from(vec![Some(10i128.pow(38)); 513])
            .with_precision_and_scale(38, 4)
            .unwrap_or_else(|error| panic!("{error}")),
    );
    let invalid = RecordBatch::try_new(batch.schema(), invalid_columns)
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(
        sink.write_batch(&invalid).await,
        Err(elm_core::ElmError::TypeMapping(_))
    ));
    // Rejection must leave the worker usable and staging empty, so the next valid
    // batch can commit with the original sequence and exact row count.
    for (column, array) in [
        (
            12,
            Arc::new(Date32Array::from(vec![Some(-719_163); 513])) as Arc<dyn Array>,
        ),
        (
            13,
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(
                    253_402_300_800_000_000
                );
                513
            ])) as Arc<dyn Array>,
        ),
    ] {
        let mut columns = batch.columns().to_vec();
        columns[column] = array;
        let invalid =
            RecordBatch::try_new(batch.schema(), columns).unwrap_or_else(|error| panic!("{error}"));
        assert!(matches!(
            sink.write_batch(&invalid).await,
            Err(elm_core::ElmError::TypeMapping(_))
        ));
    }
    sink.write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.commit_checkpoint(&BatchCheckpoint {
        sequence: 0,
        rows_committed: 513,
        bytes_committed: 0,
        source_fingerprint: None,
        source_position: serde_json::json!({"row": 513}),
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut source = SqlServerSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Query {
            sql: format!("SELECT * FROM [dbo].[{}] ORDER BY id", target.name.as_str()),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let mut offset = 0;
    while let Some(actual) = source
        .next_batch(32 * 1024 * 1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        assert_eq!(actual, batch.slice(offset, actual.num_rows()));
        offset += actual.num_rows();
    }
    assert_eq!(offset, 513);
    cleanup_sql_server_staging(&environment, &password, target.clone(), job_id)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let manager = odbc_api::Environment::new().unwrap_or_else(|_| panic!("ODBC unavailable"));
    let connection = fixture_connection(&manager, &environment, &password);
    connection
        .execute(
            &format!("DROP TABLE [dbo].[{}]", target.name.as_str()),
            (),
            Some(15),
        )
        .unwrap_or_else(|_| panic!("fixture cleanup failed"));
}

#[tokio::test]
#[ignore = "requires SQL Server Driver 18 and an isolated writable SQL Server test database"]
async fn timestamp_append_requires_exact_destination_precision() {
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use elm_connectors::sql_server_sink::{SqlServerSink, cleanup_sql_server_staging};
    use elm_core::{
        BatchCheckpoint, ConsistencyMode, DataSink, Identifier, JobId, Relation, WriteMode,
    };
    use std::sync::Arc;
    let environment = environment();
    let password = std::env::var("ELM_TEST_SQL_SERVER_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_SQL_SERVER_PASSWORD"));
    let manager = odbc_api::Environment::new().unwrap_or_else(|_| panic!("ODBC unavailable"));
    let connection = fixture_connection(&manager, &environment, &password);
    let schema = Arc::new(Schema::new(vec![Field::new(
        "created",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
    )]));
    for sql_type in ["DATETIME", "SMALLDATETIME", "DATETIME2(3)", "DATETIME2(6)"] {
        let job_id = JobId::new();
        let target = Relation {
            catalog: None,
            schema: Some(Identifier::new("dbo").unwrap_or_else(|error| panic!("{error}"))),
            name: Identifier::new(format!("elm_time_test_{}", job_id.0.simple()))
                .unwrap_or_else(|error| panic!("{error}")),
        };
        let table = format!("[dbo].[{}]", target.name.as_str());
        connection
            .execute(
                &format!("CREATE TABLE {table} (created {sql_type} NULL)"),
                (),
                Some(15),
            )
            .unwrap_or_else(|_| panic!("fixture creation failed"));
        let mut sink =
            SqlServerSink::connect(&environment, &password, target.clone(), job_id, None)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
        let preflight = sink
            .preflight(schema.clone(), WriteMode::Append, ConsistencyMode::Atomic)
            .await;
        if sql_type == "DATETIME2(6)" {
            preflight.unwrap_or_else(|error| panic!("{error}"));
            sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(-1),
                    None,
                ]))],
            )
            .unwrap_or_else(|error| panic!("{error}"));
            sink.write_batch(&batch)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
            sink.commit_checkpoint(&BatchCheckpoint {
                sequence: 0,
                rows_committed: 2,
                bytes_committed: 0,
                source_fingerprint: None,
                source_position: serde_json::json!({"row": 2}),
            })
            .await
            .unwrap_or_else(|error| panic!("{error}"));
            sink.publish()
                .await
                .unwrap_or_else(|error| panic!("{error}"));
            let mut source = SqlServerSource::connect(&environment, &password, &DatabaseSelection::Query {
                sql: format!("SELECT created FROM {table} ORDER BY CASE WHEN created IS NULL THEN 1 ELSE 0 END"),
            }).await.unwrap_or_else(|error| panic!("{error}"));
            let actual = source
                .next_batch(8192)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .unwrap_or_else(|| panic!("missing appended rows"));
            assert_eq!(actual, batch);
            drop(source);
            cleanup_sql_server_staging(&environment, &password, target, job_id)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
        } else {
            assert!(matches!(preflight, Err(elm_core::ElmError::TypeMapping(_))));
            assert_eq!(target_rows(&environment, &password, &table).await, 0);
        }
        connection
            .execute(&format!("DROP TABLE {table}"), (), Some(15))
            .unwrap_or_else(|_| panic!("fixture cleanup failed"));
    }
}

#[tokio::test]
#[ignore = "requires SQL Server Driver 18 and an isolated writable SQL Server test database"]
async fn append_rejects_nonunicode_fixed_width_and_narrow_targets() {
    use arrow_schema::{DataType, Field, Schema};
    use elm_connectors::sql_server_sink::SqlServerSink;
    use elm_core::{ConsistencyMode, DataSink, Identifier, JobId, Relation, WriteMode};
    use std::sync::Arc;
    let environment = environment();
    let password = std::env::var("ELM_TEST_SQL_SERVER_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_SQL_SERVER_PASSWORD"));
    let manager = odbc_api::Environment::new().unwrap_or_else(|_| panic!("ODBC unavailable"));
    let connection = fixture_connection(&manager, &environment, &password);
    for (sql_type, data_type) in [
        ("VARCHAR(4000)", DataType::Utf8),
        ("NCHAR(4000)", DataType::Utf8),
        ("NVARCHAR(20)", DataType::Utf8),
        ("BINARY(8000)", DataType::Binary),
        ("VARBINARY(20)", DataType::Binary),
    ] {
        let job_id = JobId::new();
        let target = Relation {
            catalog: None,
            schema: Some(Identifier::new("dbo").unwrap_or_else(|error| panic!("{error}"))),
            name: Identifier::new(format!("elm_width_test_{}", job_id.0.simple()))
                .unwrap_or_else(|error| panic!("{error}")),
        };
        let table = format!("[dbo].[{}]", target.name.as_str());
        connection
            .execute(
                &format!("CREATE TABLE {table} (value {sql_type} NULL)"),
                (),
                Some(15),
            )
            .unwrap_or_else(|_| panic!("fixture creation failed"));
        let mut sink = SqlServerSink::connect(&environment, &password, target, job_id, None)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, true)]));
        assert!(matches!(
            sink.preflight(schema, WriteMode::Append, ConsistencyMode::Atomic)
                .await,
            Err(elm_core::ElmError::TypeMapping(_))
        ));
        assert_eq!(target_rows(&environment, &password, &table).await, 0);
        connection
            .execute(&format!("DROP TABLE {table}"), (), Some(15))
            .unwrap_or_else(|_| panic!("fixture cleanup failed"));
    }
}
