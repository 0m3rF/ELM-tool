#![cfg(feature = "oracle")]
use arrow_array::{
    Array, BinaryArray, Decimal128Array, Float32Array, Float64Array, StringArray,
    TimestampMicrosecondArray,
};
use elm_connectors::oracle_sink::{OracleSink, cleanup_oracle_staging, oracle_published_rows};
use elm_connectors::oracle_source::OracleSource;
use elm_core::{
    BatchCheckpoint, ElmError, FileFormat, JobProgress, JobSpec, JobState, Result, SinkSpec,
    SourceSpec, WriteMode,
};
use elm_core::{ConsistencyMode, DataSink, Identifier, JobId, Relation};
use elm_core::{DataSource, DatabaseSelection, Environment};
use elm_engine::{EngineObserver, TransferEngine};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio_util::sync::CancellationToken;

// Force multiple engine batches without a large database fixture.
struct SmallBatches(OracleSource);
#[async_trait::async_trait]
impl DataSource for SmallBatches {
    async fn schema(&mut self) -> Result<Arc<arrow_schema::Schema>> {
        self.0.schema().await
    }
    async fn next_batch(&mut self, bytes: usize) -> Result<Option<arrow_array::RecordBatch>> {
        self.0.next_batch(bytes.min(4096)).await
    }
    async fn checkpoint(&mut self) -> Result<serde_json::Value> {
        self.0.checkpoint().await
    }
}

const ORIGINAL: &[u8] = b"existing destination must survive until publication";
struct PublicationObserver {
    path: PathBuf,
    fail: bool,
    checkpoints: Mutex<Vec<BatchCheckpoint>>,
    states: Mutex<Vec<JobState>>,
}
#[async_trait::async_trait]
impl EngineObserver for PublicationObserver {
    async fn progress(&self, event: JobProgress) -> Result<()> {
        self.states
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(event.state);
        Ok(())
    }
    async fn checkpoint(&self, checkpoint: &BatchCheckpoint) -> Result<()> {
        assert_eq!(std::fs::read(&self.path)?, ORIGINAL);
        assert_eq!(checkpoint.source_position["row"], checkpoint.rows_committed);
        self.checkpoints
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(checkpoint.clone());
        if self.fail {
            return Err(ElmError::State("injected checkpoint-store failure".into()));
        }
        Ok(())
    }
}

async fn export_fixture(format: FileFormat, empty: bool, fail: bool, mode: WriteMode) {
    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let output = directory.path().join("output");
    let stage = directory.path().join("stage");
    std::fs::write(&output, ORIGINAL).unwrap_or_else(|error| panic!("{error}"));
    let sql = if empty {
        "SELECT CAST(1 AS NUMBER(10,0)) AS \"id\", CAST(N'İstanbul 🌍' AS NVARCHAR2(32)) AS \"label\" FROM dual WHERE 1=0"
    } else {
        "SELECT CAST(LEVEL AS NUMBER(10,0)) AS \"id\", CAST(N'İstanbul 🌍' AS NVARCHAR2(32)) AS \"label\" FROM dual CONNECT BY LEVEL <= 9 ORDER BY LEVEL"
    };
    let selection = DatabaseSelection::Query { sql: sql.into() };
    let environment = environment();
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_ORACLE_PASSWORD"));
    let source = OracleSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut spec = JobSpec::new(
        SourceSpec::Database {
            environment_id: environment.id,
            selection,
            resume_key: Vec::new(),
        },
        SinkSpec::File {
            path: output.clone(),
            format,
        },
    );
    spec.write_mode = mode;
    let observer = Arc::new(PublicationObserver {
        path: output.clone(),
        fail,
        checkpoints: Mutex::new(Vec::new()),
        states: Mutex::new(Vec::new()),
    });
    let sink = elm_connectors::RecoverableFileSink::new(&output, format, &stage, None)
        .unwrap_or_else(|error| panic!("{error}"));
    let result = TransferEngine::new(spec, CancellationToken::new(), observer.clone())
        .run(Box::new(SmallBatches(source)), Box::new(sink))
        .await;
    let checkpoints = observer
        .checkpoints
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clone();
    if fail || mode == WriteMode::Fail {
        assert!(result.is_err());
        assert_eq!(
            std::fs::read(&output).unwrap_or_else(|error| panic!("{error}")),
            ORIGINAL
        );
        if fail {
            assert!(matches!(result, Err(ElmError::State(_))));
            assert_eq!(checkpoints.len(), 1);
            assert_eq!(
                observer
                    .states
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .last(),
                Some(&JobState::Failed)
            );
        } else {
            assert!(checkpoints.is_empty());
        }
        return;
    }
    let stats = result.unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stats.rows, if empty { 0 } else { 9 });
    assert!(!stage.exists());
    assert_eq!(
        observer
            .states
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .last(),
        Some(&JobState::Succeeded)
    );
    if empty {
        assert!(checkpoints.is_empty());
        match format {
            FileFormat::Csv => assert_eq!(
                std::fs::read_to_string(&output)
                    .unwrap_or_else(|error| panic!("{error}"))
                    .trim(),
                "id,label"
            ),
            FileFormat::Ndjson => assert!(
                std::fs::read(&output)
                    .unwrap_or_else(|error| panic!("{error}"))
                    .is_empty()
            ),
            FileFormat::Parquet => {
                let mut reader = elm_connectors::FileSource::open(&output, format)
                    .unwrap_or_else(|error| panic!("{error}"));
                assert_eq!(
                    reader
                        .schema()
                        .await
                        .unwrap_or_else(|error| panic!("{error}"))
                        .fields()
                        .len(),
                    2
                );
                assert!(
                    reader
                        .next_batch(4096)
                        .await
                        .unwrap_or_else(|error| panic!("{error}"))
                        .is_none()
                );
            }
        }
    } else {
        assert!(checkpoints.len() > 1);
        assert_eq!(
            checkpoints
                .last()
                .map(|checkpoint| checkpoint.rows_committed),
            Some(9)
        );
        let mut reader = elm_connectors::FileSource::open(&output, format)
            .unwrap_or_else(|error| panic!("{error}"));
        let mut rows = 0;
        while let Some(batch) = reader
            .next_batch(4096)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
        {
            let labels = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap_or_else(|| panic!("expected labels"));
            for index in 0..batch.num_rows() {
                assert_eq!(labels.value(index), "İstanbul 🌍");
                let expected = rows + index + 1;
                let ids = batch.column(0).as_any();
                if let Some(ids) = ids.downcast_ref::<Decimal128Array>() {
                    assert_eq!(ids.value(index), expected as i128);
                } else if let Some(ids) = ids.downcast_ref::<arrow_array::Int64Array>() {
                    assert_eq!(ids.value(index), expected as i64);
                } else if let Some(ids) = ids.downcast_ref::<StringArray>() {
                    assert_eq!(ids.value(index), expected.to_string());
                } else {
                    panic!("unexpected exported identifier type");
                }
            }
            rows += batch.num_rows();
        }
        assert_eq!(rows, 9);
    }
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn engine_exports_atomic_files_and_preserves_empty_schemas() {
    for format in [FileFormat::Csv, FileFormat::Ndjson, FileFormat::Parquet] {
        for empty in [false, true] {
            export_fixture(format, empty, false, WriteMode::Replace).await;
        }
    }
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn engine_failure_and_fail_mode_preserve_existing_destinations() {
    for format in [FileFormat::Csv, FileFormat::Ndjson, FileFormat::Parquet] {
        export_fixture(format, false, true, WriteMode::Replace).await;
        export_fixture(format, false, false, WriteMode::Fail).await;
    }
}

fn environment() -> Environment {
    let mut environment: Environment = serde_json::from_value(serde_json::json!({
        "id": "00000000-0000-4000-8000-000000000001", "name": "Oracle test", "kind": "oracle",
        "host": std::env::var("ELM_TEST_ORACLE_HOST").unwrap_or_else(|_| "localhost".into()),
        "port": std::env::var("ELM_TEST_ORACLE_PORT").ok().and_then(|value| value.parse::<u16>().ok()).unwrap_or(1521),
        "database": std::env::var("ELM_TEST_ORACLE_DATABASE").unwrap_or_else(|_| "XEPDB1".into()),
        "username": std::env::var("ELM_TEST_ORACLE_USER").unwrap_or_else(|_| "elm_test".into()),
        "credential_ref": "integration-only", "options": {"ssl_mode": std::env::var("ELM_TEST_ORACLE_SSL_MODE").unwrap_or_else(|_| "require".into())},
        "created_at": "2026-09-10T00:00:00Z", "updated_at": "2026-09-10T00:00:00Z"
    })).unwrap_or_else(|error| panic!("{error}"));
    if let Ok(timeout) = std::env::var("ELM_TEST_ORACLE_CALL_TIMEOUT_MS") {
        environment.options["oracle_call_timeout_ms"] = serde_json::json!(
            timeout
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid fixture timeout"))
        );
    }
    environment
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn typed_source_preserves_values_nulls_date_time_and_repeated_eof() {
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_ORACLE_PASSWORD"));
    let query = DatabaseSelection::Query { sql: "SELECT CAST(N'İstanbul 🌍' AS NVARCHAR2(32)) AS label, HEXTORAW('00FF5C') AS payload, CAST(-12345.6700 AS NUMBER(38,4)) AS amount, CAST(1e-38 AS NUMBER(38,38)) AS fraction, CAST(12300 AS NUMBER(5,-2)) AS rounded, CAST(TIMESTAMP '2024-02-29 12:34:56' AS DATE) AS day, CAST(TIMESTAMP '1969-12-31 23:59:59.999999' AS TIMESTAMP(6)) AS created, CAST(1.25 AS BINARY_FLOAT) AS single_value, CAST(-3.5 AS BINARY_DOUBLE) AS double_value FROM dual UNION ALL SELECT CAST(NULL AS NVARCHAR2(32)), CAST(NULL AS RAW(3)), CAST(NULL AS NUMBER(38,4)), CAST(NULL AS NUMBER(38,38)), CAST(NULL AS NUMBER(5,-2)), CAST(NULL AS DATE), CAST(NULL AS TIMESTAMP(6)), CAST(NULL AS BINARY_FLOAT), CAST(NULL AS BINARY_DOUBLE) FROM dual".into() };
    let mut source = OracleSource::connect(&environment(), &password, &query)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let first = source
        .next_batch(4096)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("missing first row"));
    assert_eq!(first.num_rows(), 1);
    macro_rules! column {
        ($index:expr, $array:ty) => {
            first
                .column($index)
                .as_any()
                .downcast_ref::<$array>()
                .unwrap_or_else(|| panic!("column type"))
        };
    }
    assert_eq!(column!(0, StringArray).value(0), "İstanbul 🌍");
    assert_eq!(column!(1, BinaryArray).value(0), &[0, 255, 92]);
    assert_eq!(column!(2, Decimal128Array).value(0), -123_456_700);
    assert_eq!(column!(3, Decimal128Array).value(0), 1);
    assert_eq!(column!(4, Decimal128Array).value(0), 123);
    let date = chrono::NaiveDate::from_ymd_opt(2024, 2, 29)
        .and_then(|date| date.and_hms_opt(12, 34, 56))
        .unwrap_or_else(|| panic!("fixture date"));
    assert_eq!(
        column!(5, TimestampMicrosecondArray).value(0),
        date.and_utc().timestamp_micros()
    );
    assert_eq!(column!(6, TimestampMicrosecondArray).value(0), -1);
    assert_eq!(column!(7, Float32Array).value(0), 1.25);
    assert_eq!(column!(8, Float64Array).value(0), -3.5);
    let last = source
        .next_batch(4096)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("missing null row"));
    assert_eq!(last.num_rows(), 1);
    assert!(last.columns().iter().all(|column| column.is_null(0)));
    for _ in 0..2 {
        assert!(
            source
                .next_batch(4096)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .is_none()
        );
    }
    assert_eq!(
        source
            .checkpoint()
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        serde_json::json!({"row": 2})
    );
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn source_rejects_unsupported_lobs_and_unconstrained_numbers() {
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_ORACLE_PASSWORD"));
    for sql in [
        "SELECT TO_CLOB('unsupported') AS value FROM dual",
        "SELECT CAST(1 AS NUMBER) AS value FROM dual",
        "SELECT CAST(SYSTIMESTAMP AS TIMESTAMP WITH TIME ZONE) AS value FROM dual",
    ] {
        let result = OracleSource::connect(
            &environment(),
            &password,
            &DatabaseSelection::Query { sql: sql.into() },
        )
        .await;
        assert!(matches!(result, Err(elm_core::ElmError::TypeMapping(_))));
    }
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn empty_source_preserves_schema_and_native_errors_are_sanitized() {
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_ORACLE_PASSWORD"));
    let environment = environment();
    let mut source = OracleSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Query {
            sql: "SELECT CAST(1 AS NUMBER(5,0)) AS id FROM dual WHERE 1=0".into(),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        source
            .schema()
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .fields()
            .len(),
        1
    );
    for _ in 0..2 {
        assert!(
            source
                .next_batch(4096)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .is_none()
        );
    }
    assert_eq!(
        source
            .checkpoint()
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        serde_json::json!({"row": 0})
    );
    let error = match OracleSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Query {
            sql: "SELECT elm_private_query_marker FROM dual".into(),
        },
    )
    .await
    {
        Ok(_) => panic!("invalid query unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(error, elm_core::ElmError::Connection { .. }));
    let message = error.to_string();
    assert!(!message.contains("elm_private_query_marker"));
    assert!(!message.contains(&password));
    assert!(!message.contains("ORA-"));
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn requested_batches_preserve_order_and_do_not_skip_rows() {
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD")
        .unwrap_or_else(|_| panic!("set ELM_TEST_ORACLE_PASSWORD"));
    let mut source = OracleSource::connect(
        &environment(),
        &password,
        &DatabaseSelection::Query {
            sql: "SELECT CAST(LEVEL AS NUMBER(10,0)) AS id FROM dual CONNECT BY LEVEL <= 1025 ORDER BY id".into(),
        },
    ).await.unwrap_or_else(|error| panic!("{error}"));
    let mut expected = 1i128;
    let mut batches = 0;
    while let Some(batch) = source
        .next_batch(4096)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        assert!(batch.get_array_memory_size() <= 4096);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap_or_else(|| panic!("expected decimal"));
        for row in 0..batch.num_rows() {
            assert_eq!(values.value(row), expected);
            expected += 1;
        }
        batches += 1;
    }
    assert_eq!(expected, 1026);
    assert!(batches > 1);
    assert_eq!(
        source
            .checkpoint()
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        serde_json::json!({"row": 1025})
    );
}

fn native_connection(environment: &Environment, password: &str) -> oracle::Connection {
    assert_eq!(
        environment.options["ssl_mode"], "disable",
        "native write fixtures require explicit SSL_MODE=disable on an isolated service"
    );
    oracle::Connection::connect(
        &environment.username,
        password,
        format!(
            "//{}:{}/{}",
            environment.host, environment.port, environment.database
        ),
    )
    .unwrap_or_else(|_| panic!("isolated Oracle fixture connection failed"))
}
fn fixture_execute(connection: &oracle::Connection, sql: &str) {
    connection
        .execute(sql, &[])
        .unwrap_or_else(|error| panic!("fixture SQL failed: {error}"));
}
fn fixture_target(job: JobId) -> Relation {
    Relation {
        catalog: None,
        schema: None,
        name: Identifier::new(format!("elm_test_{}", job.0.simple()))
            .unwrap_or_else(|error| panic!("{error}")),
    }
}
fn fixture_batch() -> arrow_array::RecordBatch {
    let decimals =
        Decimal128Array::from(vec![Some(12345678901234567890123456789012345678i128), None])
            .with_precision_and_scale(38, 4)
            .unwrap_or_else(|error| panic!("{error}"));
    arrow_array::RecordBatch::try_from_iter(vec![
        ("exact", Arc::new(decimals) as arrow_array::ArrayRef),
        (
            "label",
            Arc::new(StringArray::from(vec![Some("İstanbul 🌍"), None])) as arrow_array::ArrayRef,
        ),
    ])
    .unwrap_or_else(|error| panic!("{error}"))
}
async fn stage_fixture(sink: &mut OracleSink, batch: &arrow_array::RecordBatch) {
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.commit_checkpoint(&BatchCheckpoint {
        sequence: 0,
        source_position: serde_json::json!({"row": 2}),
        rows_committed: 2,
        bytes_committed: batch.get_array_memory_size() as u64,
        source_fingerprint: None,
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn bulk_sink_atomic_new_table_append_and_empty_swap() {
    let environment = environment();
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD").unwrap_or_default();
    let connection = native_connection(&environment, &password);
    let job = JobId::new();
    let target = fixture_target(job);
    let batch = fixture_batch();
    let mut sink = OracleSink::connect(&environment, &password, target.clone(), job, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(batch.schema(), WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    println!("staging initial atomic new-table transfer");
    stage_fixture(&mut sink, &batch).await;
    let absent: u64 = connection
        .query_row_as(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = :1",
            &[&target.name.as_str()],
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(absent, 0);
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(sink);
    assert_eq!(
        oracle_published_rows(&environment, &password, target.clone(), job)
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        Some(2)
    );
    let mut readback = OracleSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: target.clone(),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let output = readback
        .next_batch(1024 * 1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("missing rows"));
    assert_eq!(output.column(0).as_ref(), batch.column(0).as_ref());
    assert_eq!(output.column(1).as_ref(), batch.column(1).as_ref());
    drop(readback);
    let append_job = JobId::new();
    let mut append = OracleSink::connect(&environment, &password, target.clone(), append_job, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    append
        .preflight(batch.schema(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    println!("staging atomic append transfer");
    stage_fixture(&mut append, &batch).await;
    append
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(append);
    let count: u64 = connection
        .query_row_as(&format!("SELECT COUNT(*) FROM \"{}\"", target.name), &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(count, 4);
    let replace_job = JobId::new();
    let mut replace =
        OracleSink::connect(&environment, &password, target.clone(), replace_job, None)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        replace
            .preflight(batch.schema(), WriteMode::Replace, ConsistencyMode::Atomic)
            .await
            .is_err()
    );
    assert!(
        replace
            .preflight(batch.schema(), WriteMode::Fail, ConsistencyMode::Atomic)
            .await
            .is_err()
    );
    replace
        .preflight(
            batch.schema(),
            WriteMode::Replace,
            ConsistencyMode::TableSwap,
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    replace
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    replace
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(replace);
    let count: u64 = connection
        .query_row_as(&format!("SELECT COUNT(*) FROM \"{}\"", target.name), &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(count, 0);
    for id in [job, append_job, replace_job] {
        cleanup_oracle_staging(&environment, &password, target.clone(), id)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    }
    fixture_execute(&connection, &format!("DROP TABLE \"{}\"", target.name));
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn interrupted_swap_finishes_once_and_retains_original_backup() {
    let environment = environment();
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD").unwrap_or_default();
    let connection = native_connection(&environment, &password);
    let job = JobId::new();
    let target = fixture_target(job);
    fixture_execute(
        &connection,
        &format!("CREATE TABLE \"{}\" (original NUMBER(10,0))", target.name),
    );
    fixture_execute(
        &connection,
        &format!("INSERT INTO \"{}\" VALUES (99)", target.name),
    );
    connection
        .commit()
        .unwrap_or_else(|error| panic!("{error}"));
    let batch = fixture_batch();
    let mut sink = OracleSink::connect(&environment, &password, target.clone(), job, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(
        batch.schema(),
        WriteMode::Replace,
        ConsistencyMode::TableSwap,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    stage_fixture(&mut sink, &batch).await;
    drop(sink);
    let suffix = job.0.simple();
    // Inject the real persistent state left by termination between the renames.
    fixture_execute(
        &connection,
        &format!("ALTER TABLE \"elm_stage_{suffix}\" MODIFY (\"__elm_batch\" INVISIBLE)"),
    );
    fixture_execute(
        &connection,
        &format!("UPDATE \"elm_receipt_{suffix}\" SET phase = 'ready'"),
    );
    connection
        .commit()
        .unwrap_or_else(|error| panic!("{error}"));
    fixture_execute(
        &connection,
        &format!(
            "ALTER TABLE \"{}\" RENAME TO \"elm_backup_{suffix}\"",
            target.name
        ),
    );
    for _ in 0..2 {
        assert_eq!(
            oracle_published_rows(&environment, &password, target.clone(), job)
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            Some(2)
        );
    }
    let old: u64 = connection
        .query_row_as(
            &format!("SELECT original FROM \"elm_backup_{suffix}\""),
            &[],
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(old, 99);
    cleanup_oracle_staging(&environment, &password, target.clone(), job)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    fixture_execute(&connection, &format!("DROP TABLE \"{}\"", target.name));
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn failed_atomic_append_preserves_target_and_has_no_publication_receipt() {
    let environment = environment();
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD").unwrap_or_default();
    let connection = native_connection(&environment, &password);
    let job = JobId::new();
    let target = fixture_target(job);
    fixture_execute(
        &connection,
        &format!(
            "CREATE TABLE \"{}\" (\"exact\" NUMBER(38,4) UNIQUE, \"label\" NVARCHAR2(2000))",
            target.name
        ),
    );
    fixture_execute(
        &connection,
        &format!(
            "INSERT INTO \"{}\" VALUES (1234567890123456789012345678901234.5678, N'original')",
            target.name
        ),
    );
    connection
        .commit()
        .unwrap_or_else(|error| panic!("{error}"));
    let batch = fixture_batch();
    let mut sink = OracleSink::connect(&environment, &password, target.clone(), job, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(batch.schema(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    stage_fixture(&mut sink, &batch).await;
    let error = sink
        .publish()
        .await
        .err()
        .unwrap_or_else(|| panic!("duplicate key must fail"));
    let public = error.to_string();
    assert!(!public.contains(&password));
    assert!(!public.contains("1234567890123456789012345678901234"));
    sink.abort().await.unwrap_or_else(|error| panic!("{error}"));
    drop(sink);
    let (count, label): (u64, String) = connection
        .query_row_as(
            &format!("SELECT COUNT(*), MAX(\"label\") FROM \"{}\"", target.name),
            &[],
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!((count, label.as_str()), (1, "original"));
    assert_eq!(
        oracle_published_rows(&environment, &password, target.clone(), job)
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        None
    );
    cleanup_oracle_staging(&environment, &password, target.clone(), job)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    fixture_execute(&connection, &format!("DROP TABLE \"{}\"", target.name));
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn resume_trims_a_committed_batch_missing_from_the_durable_checkpoint() {
    let environment = environment();
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD").unwrap_or_default();
    let connection = native_connection(&environment, &password);
    let job = JobId::new();
    let target = fixture_target(job);
    let batch = fixture_batch();
    let checkpoint = BatchCheckpoint {
        sequence: 0,
        source_position: serde_json::json!({"row": 2}),
        rows_committed: 2,
        bytes_committed: batch.get_array_memory_size() as u64,
        source_fingerprint: Some("unchanged-fixture".into()),
    };
    let mut sink = OracleSink::connect(&environment, &password, target.clone(), job, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(batch.schema(), WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    stage_fixture(&mut sink, &batch).await;
    // Oracle commits a second batch, but the daemon dies before saving its
    // SQLite checkpoint. Recovery must trust the older durable checkpoint.
    let mut newer = checkpoint.clone();
    newer.sequence = 1;
    newer.rows_committed = 4;
    newer.bytes_committed *= 2;
    newer.source_position = serde_json::json!({"row": 4});
    sink.write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.commit_checkpoint(&newer)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(sink);
    let stage = format!("\"elm_stage_{}\"", job.0.simple());
    let before: u64 = connection
        .query_row_as(&format!("SELECT COUNT(*) FROM {stage}"), &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(before, 4);
    let changed_schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("changed", arrow_schema::DataType::Decimal128(38, 4), true),
        batch.schema().field(1).clone(),
    ]));
    let mut changed = OracleSink::connect(
        &environment,
        &password,
        target.clone(),
        job,
        Some(&checkpoint),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    changed
        .preflight(changed_schema, WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(
        changed.begin().await,
        Err(ElmError::TypeMapping(_))
    ));
    drop(changed);
    let mut invalid = checkpoint.clone();
    invalid.rows_committed = 999;
    let mut inconsistent =
        OracleSink::connect(&environment, &password, target.clone(), job, Some(&invalid))
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    inconsistent
        .preflight(batch.schema(), WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(
        inconsistent.begin().await,
        Err(ElmError::State(_))
    ));
    drop(inconsistent);
    let preserved: u64 = connection
        .query_row_as(&format!("SELECT COUNT(*) FROM {stage}"), &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        preserved, 4,
        "invalid recovery must roll back its attempted trim"
    );
    let mut resumed = OracleSink::connect(
        &environment,
        &password,
        target.clone(),
        job,
        Some(&checkpoint),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .preflight(batch.schema(), WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let trimmed: u64 = connection
        .query_row_as(&format!("SELECT COUNT(*) FROM {stage}"), &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(trimmed, 2);
    resumed
        .write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .commit_checkpoint(&newer)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(resumed);
    let (total, nonnull): (u64, u64) = connection
        .query_row_as(
            &format!("SELECT COUNT(*), COUNT(\"exact\") FROM \"{}\"", target.name),
            &[],
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!((total, nonnull), (4, 2));
    assert_eq!(
        oracle_published_rows(&environment, &password, target.clone(), job)
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        Some(4)
    );
    cleanup_oracle_staging(&environment, &password, target.clone(), job)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    fixture_execute(&connection, &format!("DROP TABLE \"{}\"", target.name));
}

#[tokio::test]
#[ignore = "requires Oracle Instant Client and an isolated Oracle database"]
async fn deleting_an_interrupted_swap_restores_the_original_without_publication() {
    let environment = environment();
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD").unwrap_or_default();
    let connection = native_connection(&environment, &password);
    let job = JobId::new();
    let target = fixture_target(job);
    fixture_execute(
        &connection,
        &format!("CREATE TABLE \"{}\" (original NUMBER(10,0))", target.name),
    );
    fixture_execute(
        &connection,
        &format!("INSERT INTO \"{}\" VALUES (99)", target.name),
    );
    connection
        .commit()
        .unwrap_or_else(|error| panic!("{error}"));
    let original_id: u64 = connection
        .query_row_as(
            "SELECT object_id FROM user_objects WHERE object_name = :1 AND object_type = 'TABLE'",
            &[&target.name.as_str()],
        )
        .unwrap_or_else(|error| panic!("{error}"));
    let batch = fixture_batch();
    let mut sink = OracleSink::connect(&environment, &password, target.clone(), job, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(
        batch.schema(),
        WriteMode::Replace,
        ConsistencyMode::TableSwap,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    stage_fixture(&mut sink, &batch).await;
    drop(sink);
    let suffix = job.0.simple();
    fixture_execute(
        &connection,
        &format!("ALTER TABLE \"elm_stage_{suffix}\" MODIFY (\"__elm_batch\" INVISIBLE)"),
    );
    fixture_execute(
        &connection,
        &format!("UPDATE \"elm_receipt_{suffix}\" SET phase = 'ready'"),
    );
    connection
        .commit()
        .unwrap_or_else(|error| panic!("{error}"));
    fixture_execute(
        &connection,
        &format!(
            "ALTER TABLE \"{}\" RENAME TO \"elm_backup_{suffix}\"",
            target.name
        ),
    );
    cleanup_oracle_staging(&environment, &password, target.clone(), job)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let restored_id: u64 = connection
        .query_row_as(
            "SELECT object_id FROM user_objects WHERE object_name = :1 AND object_type = 'TABLE'",
            &[&target.name.as_str()],
        )
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(restored_id, original_id);
    let value: u64 = connection
        .query_row_as(&format!("SELECT original FROM \"{}\"", target.name), &[])
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(value, 99);
    for name in [
        format!("elm_stage_{suffix}"),
        format!("elm_backup_{suffix}"),
        format!("elm_receipt_{suffix}"),
    ] {
        let remaining: u64 = connection
            .query_row_as(
                "SELECT COUNT(*) FROM user_tables WHERE table_name = :1",
                &[&name],
            )
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(remaining, 0);
    }
    // Cleanup is idempotent and must never remove the restored user table.
    cleanup_oracle_staging(&environment, &password, target.clone(), job)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    fixture_execute(&connection, &format!("DROP TABLE \"{}\"", target.name));
}
