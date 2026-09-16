#![cfg(feature = "mysql")]

use arrow_array::{
    Array, BinaryArray, Date32Array, Decimal128Array, Int8Array, Int64Array, StringArray,
    TimestampMicrosecondArray, UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};
use chrono::Utc;
use elm_connectors::{MySqlSink, MySqlSource, mysql_published_rows, test_mysql_environment};
use elm_core::{
    BatchCheckpoint, ConsistencyMode, DataSink, DataSource, DatabaseKind, DatabaseSelection,
    ElmError, Environment, EnvironmentId, Identifier, JobId, Relation, WriteMode,
};
use mysql_async::{Conn, Opts, OptsBuilder, prelude::Queryable};

const DEFAULT_PASSWORD: &str = "elm_integration_only";
type SinkOutputRow = (
    i64,
    u64,
    i8,
    Option<String>,
    Option<Vec<u8>>,
    Option<String>,
);

fn environment() -> Environment {
    let port = std::env::var("ELM_TEST_MYSQL_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(53_306);
    Environment {
        id: EnvironmentId::new(),
        name: "MySQL integration".into(),
        kind: DatabaseKind::MySql,
        host: "127.0.0.1".into(),
        port,
        database: "elm_test".into(),
        username: "root".into(),
        credential_ref: "integration-only".into(),
        options: serde_json::json!({ "ssl_mode": "disable" }),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn password() -> String {
    std::env::var("ELM_TEST_MYSQL_PASSWORD").unwrap_or_else(|_| DEFAULT_PASSWORD.into())
}

fn relation(name: &str) -> Relation {
    Relation {
        catalog: None,
        schema: None,
        name: Identifier::new(name).unwrap_or_else(|error| panic!("{error}")),
    }
}

fn connection_options(environment: &Environment, password: &str) -> Opts {
    Opts::from(
        OptsBuilder::default()
            .ip_or_hostname(environment.host.clone())
            .tcp_port(environment.port)
            .user(Some(environment.username.clone()))
            .pass(Some(password.to_owned()))
            .db_name(Some(environment.database.clone()))
            .prefer_socket(false),
    )
}

async fn stage_all(source: &mut MySqlSource, sink: &mut MySqlSink) -> u64 {
    let mut rows = 0_u64;
    let mut sequence = 0_u64;
    while let Some(batch) = source
        .next_batch(64)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        sink.write_batch(&batch)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        rows = rows.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX));
        sink.commit_checkpoint(&BatchCheckpoint {
            sequence,
            source_position: source
                .checkpoint()
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            rows_committed: rows,
            bytes_committed: 0,
            source_fingerprint: None,
        })
        .await
        .unwrap_or_else(|error| panic!("{error}"));
        sequence = sequence.saturating_add(1);
    }
    rows
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn prepared_source_streams_canonical_arrow_batches() {
    let environment = environment();
    let password = password();
    test_mysql_environment(&environment, &password)
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_mysql_source (
                id BIGINT NOT NULL,
                unsigned_id BIGINT UNSIGNED NOT NULL,
                enabled BOOLEAN NOT NULL,
                label VARCHAR(100) CHARACTER SET utf8mb4,
                payload VARBINARY(100),
                amount DECIMAL(18, 4),
                day DATE,
                happened_at DATETIME(6),
                occurred_at TIMESTAMP(6) NULL
             )",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "INSERT INTO elm_mysql_source VALUES
             (-1, 18446744073709551615, TRUE, 'İstanbul', X'0001FF', -12345.6700,
              DATE '2024-02-29', '2024-03-01 10:20:30.123456', '2024-03-01 07:20:30.654321'),
             (2, 3, FALSE, NULL, NULL, NULL, NULL, NULL, NULL);",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut source = MySqlSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_mysql_source"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(schema.field(0).data_type(), &DataType::Int64);
    assert!(!schema.field(0).is_nullable());
    assert_eq!(schema.field(1).data_type(), &DataType::UInt64);
    // MySQL BOOLEAN is only a TINYINT(1) alias with no 0/1 constraint, so Int8 is the
    // only lossless canonical mapping.
    assert_eq!(schema.field(2).data_type(), &DataType::Int8);
    assert_eq!(schema.field(3).data_type(), &DataType::Utf8);
    assert_eq!(schema.field(4).data_type(), &DataType::Binary);
    assert_eq!(schema.field(5).data_type(), &DataType::Decimal128(18, 4));
    assert_eq!(schema.field(6).data_type(), &DataType::Date32);
    assert_eq!(
        schema.field(7).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert_eq!(
        schema.field(8).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );

    let first = source
        .next_batch(64)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected first MySQL batch"));
    assert_eq!(first.num_rows(), 1);
    assert_eq!(
        first
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("id was not Int64"))
            .value(0),
        -1
    );
    assert_eq!(
        first
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap_or_else(|| panic!("unsigned_id was not UInt64"))
            .value(0),
        u64::MAX
    );
    assert_eq!(
        first
            .column(2)
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap_or_else(|| panic!("enabled was not Int8"))
            .value(0),
        1
    );
    assert_eq!(
        first
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("label was not Utf8"))
            .value(0),
        "İstanbul"
    );
    assert_eq!(
        first
            .column(4)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap_or_else(|| panic!("payload was not Binary"))
            .value(0),
        &[0, 1, 255]
    );
    assert_eq!(
        first
            .column(5)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap_or_else(|| panic!("amount was not Decimal128"))
            .value(0),
        -123_456_700
    );
    assert_eq!(
        first
            .column(6)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap_or_else(|| panic!("day was not Date32"))
            .value(0),
        19_782
    );
    assert!(
        first
            .column(7)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .is_some()
    );

    let second = source
        .next_batch(64)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected second MySQL batch"));
    assert_eq!(second.num_rows(), 1);
    assert!(second.column(3).is_null(0));
    assert!(
        source
            .next_batch(64)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .is_none()
    );
    assert!(
        source
            .next_batch(64)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .is_none()
    );

    drop(source);
    connection
        .query_drop("DROP TABLE elm_mysql_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .disconnect()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance with local_infile enabled"]
async fn local_infile_sink_publishes_all_write_modes_atomically() {
    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_sink_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_mysql_sink_source (
                id BIGINT NOT NULL,
                unsigned_id BIGINT UNSIGNED NOT NULL,
                enabled BOOLEAN NOT NULL,
                label VARCHAR(100) CHARACTER SET utf8mb4,
                payload VARBINARY(100),
                amount DECIMAL(18, 4),
                day DATE,
                happened_at DATETIME(6),
                occurred_at TIMESTAMP(6) NULL
             ) ENGINE=InnoDB",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "INSERT INTO elm_mysql_sink_source VALUES
             (-1, 18446744073709551615, TRUE, 'tab\\tline\\nİstanbul', X'00095C0A1AFF',
              -12345.6700, DATE '2024-02-29', '2024-03-01 10:20:30.123456',
              '2024-03-01 07:20:30.654321'),
             (2, 3, FALSE, NULL, NULL, NULL, NULL, NULL, NULL)",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE TABLE elm_mysql_target (legacy INTEGER) ENGINE=InnoDB")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("INSERT INTO elm_mysql_target VALUES (99)")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_mysql_sink_source"),
    };
    let mut source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let replacement_job_id = JobId::new();
    let mut sink = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_target"),
        replacement_job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let report = sink
        .preflight(schema, WriteMode::Replace, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        report
            .warnings
            .iter()
            .all(|warning| !warning.contains("fallback"))
    );
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stage_all(&mut source, &mut sink).await, 2);
    let legacy: Option<i32> = connection
        .query_first("SELECT legacy FROM elm_mysql_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(legacy, Some(99));
    assert_eq!(
        mysql_published_rows(
            &environment,
            &password,
            &relation("elm_mysql_target"),
            replacement_job_id
        )
        .await
        .unwrap_or_else(|error| panic!("{error}")),
        None
    );
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        mysql_published_rows(
            &environment,
            &password,
            &relation("elm_mysql_target"),
            replacement_job_id
        )
        .await
        .unwrap_or_else(|error| panic!("{error}")),
        Some(2)
    );
    // Simulate termination between the atomic rename and its receipt update.
    connection
        .query_drop(format!(
            "UPDATE elm_stage_{}_publish SET committed = 0",
            replacement_job_id.0.simple()
        ))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        mysql_published_rows(
            &environment,
            &password,
            &relation("elm_mysql_target"),
            replacement_job_id
        )
        .await
        .unwrap_or_else(|error| panic!("{error}")),
        Some(2)
    );
    let output: Vec<SinkOutputRow> = connection
        .query(
            "SELECT id, unsigned_id, enabled, label, payload, CAST(amount AS CHAR)
             FROM elm_mysql_target ORDER BY id",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].0, -1);
    assert_eq!(output[0].1, u64::MAX);
    assert_eq!(output[0].2, 1);
    assert_eq!(output[0].3.as_deref(), Some("tab\tline\nİstanbul"));
    assert_eq!(output[0].4.as_deref(), Some(&[0, 9, 92, 10, 26, 255][..]));
    assert_eq!(output[0].5.as_deref(), Some("-12345.6700"));

    let mut append_source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let append_schema = append_source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let append_job_id = JobId::new();
    let mut append_sink = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_target"),
        append_job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    append_sink
        .preflight(append_schema, WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    append_sink
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stage_all(&mut append_source, &mut append_sink).await, 2);
    let before_append: Option<u64> = connection
        .query_first("SELECT COUNT(*) FROM elm_mysql_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(before_append, Some(2));
    append_sink
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let after_append: Option<u64> = connection
        .query_first("SELECT COUNT(*) FROM elm_mysql_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(after_append, Some(4));
    assert_eq!(
        mysql_published_rows(
            &environment,
            &password,
            &relation("elm_mysql_target"),
            append_job_id
        )
        .await
        .unwrap_or_else(|error| panic!("{error}")),
        Some(2)
    );
    assert!(
        mysql_published_rows(
            &environment,
            &password,
            &relation("another_target"),
            append_job_id
        )
        .await
        .is_err()
    );
    // A fence without a committed APPEND receipt must never infer success from
    // row counts or from an unrelated target's marker.
    connection
        .query_drop(format!(
            "UPDATE elm_stage_{}_publish SET committed = 0",
            append_job_id.0.simple()
        ))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        mysql_published_rows(
            &environment,
            &password,
            &relation("elm_mysql_target"),
            append_job_id
        )
        .await
        .is_err()
    );

    // Publication committed, but pretend the daemon never recorded success.
    // Even a restart from zero must not append these rows a second time.
    let mut replay = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_target"),
        append_job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    replay
        .preflight(
            append_source
                .schema()
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            WriteMode::Append,
            ConsistencyMode::Atomic,
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(replay.begin().await, Err(ElmError::Conflict(_))));
    assert!(
        elm_connectors::cleanup_mysql_staging(
            &environment,
            &password,
            &relation("elm_mysql_target"),
            append_job_id,
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    );

    let mut fail_sink = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_target"),
        JobId::new(),
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let fail = fail_sink
        .preflight(
            append_source
                .schema()
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            WriteMode::Fail,
            ConsistencyMode::Atomic,
        )
        .await;
    assert!(matches!(fail, Err(ElmError::Conflict(_))));

    drop(source);
    drop(append_source);
    connection
        .query_drop("TRUNCATE TABLE elm_mysql_sink_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut empty_source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let empty_schema = empty_source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut empty_sink = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_target"),
        JobId::new(),
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    empty_sink
        .preflight(empty_schema, WriteMode::Replace, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    empty_sink
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stage_all(&mut empty_source, &mut empty_sink).await, 0);
    empty_sink
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let empty_count: Option<u64> = connection
        .query_first("SELECT COUNT(*) FROM elm_mysql_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(empty_count, Some(0));

    drop(empty_source);
    connection
        .query_drop("DROP TABLE elm_mysql_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE elm_mysql_sink_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .disconnect()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn failed_append_preserves_target_and_does_not_confirm_publication() {
    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_conflict_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_conflict_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_mysql_conflict_target (id BIGINT NOT NULL PRIMARY KEY) ENGINE=InnoDB",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("INSERT INTO elm_mysql_conflict_target VALUES (99)")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE TABLE elm_mysql_conflict_source (id BIGINT NOT NULL) ENGINE=InnoDB")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("INSERT INTO elm_mysql_conflict_source VALUES (1), (1)")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut source = MySqlSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_mysql_conflict_source"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let target = relation("elm_mysql_conflict_target");
    let mut sink = MySqlSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema, WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stage_all(&mut source, &mut sink).await, 2);
    assert!(sink.publish().await.is_err());
    let rows: Vec<i64> = connection
        .query("SELECT id FROM elm_mysql_conflict_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(rows, vec![99]);
    assert!(
        mysql_published_rows(&environment, &password, &target, job_id)
            .await
            .is_err()
    );
    elm_connectors::cleanup_mysql_staging(&environment, &password, &target, job_id)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(source);
    connection
        .query_drop("DROP TABLE elm_mysql_conflict_source, elm_mysql_conflict_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .disconnect()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn prepared_fallback_is_explicit_when_local_infile_is_disabled() {
    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("SET GLOBAL local_infile = OFF")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_fallback_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_fallback_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_mysql_fallback_source (id BIGINT NOT NULL, label VARCHAR(50));
             ",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("INSERT INTO elm_mysql_fallback_source VALUES (1, 'first'), (2, 'second')")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_mysql_fallback_source"),
    };
    let mut source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut sink = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_fallback_target"),
        JobId::new(),
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let report = sink
        .preflight(schema, WriteMode::Replace, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        report
            .warnings
            .iter()
            .any(|warning| warning.contains("fallback"))
    );
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stage_all(&mut source, &mut sink).await, 2);
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let values: Vec<(i64, String)> = connection
        .query("SELECT id, label FROM elm_mysql_fallback_target ORDER BY id")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(values, [(1, "first".into()), (2, "second".into())]);

    drop(source);
    connection
        .query_drop("DROP TABLE elm_mysql_fallback_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE elm_mysql_fallback_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("SET GLOBAL local_infile = ON")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .disconnect()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance with local_infile enabled"]
async fn sink_resume_trims_only_the_uncheckpointed_batch() {
    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_resume_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_resume_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE TABLE elm_mysql_resume_source (id BIGINT NOT NULL, label VARCHAR(50));")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("INSERT INTO elm_mysql_resume_source VALUES (1, 'first'), (2, 'second')")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_mysql_resume_target (id BIGINT NOT NULL, label VARCHAR(50)) ENGINE=InnoDB",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_mysql_resume_source"),
    };
    let mut source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let first = source
        .next_batch(1)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected first MySQL batch"));
    let second = source
        .next_batch(1)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected second MySQL batch"));
    let job_id = JobId::new();
    let mut first_attempt = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_resume_target"),
        job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    first_attempt
        .preflight(schema.clone(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    first_attempt
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    first_attempt
        .write_batch(&first)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let checkpoint = BatchCheckpoint {
        sequence: 0,
        source_position: serde_json::json!({ "row": 1 }),
        rows_committed: 1,
        bytes_committed: 0,
        source_fingerprint: None,
    };
    first_attempt
        .commit_checkpoint(&checkpoint)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    // Simulate a database commit followed by termination before SQLite persisted sequence 1.
    first_attempt
        .write_batch(&second)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    first_attempt
        .abort()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(first_attempt);
    let visible_before: Option<u64> = connection
        .query_first("SELECT COUNT(*) FROM elm_mysql_resume_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(visible_before, Some(0));

    let mut resumed = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_resume_target"),
        job_id,
        Some(&checkpoint),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .preflight(schema, WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .begin()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .write_batch(&second)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .commit_checkpoint(&BatchCheckpoint {
            sequence: 1,
            source_position: serde_json::json!({ "row": 2 }),
            rows_committed: 2,
            bytes_committed: 0,
            source_fingerprint: None,
        })
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    resumed
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let values: Vec<(i64, String)> = connection
        .query("SELECT id, label FROM elm_mysql_resume_target ORDER BY id")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(values, [(1, "first".into()), (2, "second".into())]);

    drop(source);
    connection
        .query_drop("DROP TABLE elm_mysql_resume_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE elm_mysql_resume_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .disconnect()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn physical_identity_resolves_through_a_view_and_rejects_unrelated_tables() {
    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP VIEW IF EXISTS elm_my_identity_view")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_identity_base")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_identity_other")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE TABLE elm_my_identity_base (id BIGINT)")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE TABLE elm_my_identity_other (id BIGINT)")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE VIEW elm_my_identity_view AS SELECT * FROM elm_my_identity_base")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let base = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_my_identity_base"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable base table identity"));
    let via_view = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_my_identity_view"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable view identity"));
    let other = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_my_identity_other"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable unrelated table identity"));
    let missing = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_my_identity_missing"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));

    assert!(base.same_physical_table(&via_view));
    assert!(!base.same_physical_table(&other));
    assert!(missing.is_none());

    connection
        .query_drop("DROP VIEW elm_my_identity_view")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE elm_my_identity_base")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE elm_my_identity_other")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn sink_rejects_timestamps_outside_native_timestamp_range() {
    use arrow_array::RecordBatch;
    use arrow_schema::{Field, Schema};
    use elm_core::{ConsistencyMode, JobId, WriteMode};

    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_mysql_timestamp_range")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let schema = std::sync::Arc::new(Schema::new(vec![Field::new(
        "occurred_at",
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        true,
    )]));
    let job_id = JobId::new();
    let mut sink = MySqlSink::connect(
        &environment,
        &password,
        relation("elm_mysql_timestamp_range"),
        job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema.clone(), WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));

    // 2040-01-01 is outside MySQL TIMESTAMP's 1970-2038 supported range.
    let out_of_range = chrono::NaiveDate::from_ymd_opt(2040, 1, 1)
        .unwrap_or_else(|| panic!("invalid fixture date"))
        .and_hms_opt(0, 0, 0)
        .unwrap_or_else(|| panic!("invalid fixture time"))
        .and_utc()
        .timestamp_micros();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![std::sync::Arc::new(
            TimestampMicrosecondArray::from(vec![Some(out_of_range)]).with_timezone("UTC"),
        )],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let result = sink.write_batch(&batch).await;
    assert!(matches!(result, Err(elm_core::ElmError::TypeMapping(_))));

    // A value inside the supported range still succeeds and round-trips.
    let in_range = chrono::NaiveDate::from_ymd_opt(2024, 3, 1)
        .unwrap_or_else(|| panic!("invalid fixture date"))
        .and_hms_micro_opt(7, 20, 30, 654_321)
        .unwrap_or_else(|| panic!("invalid fixture time"))
        .and_utc()
        .timestamp_micros();
    let ok_batch = RecordBatch::try_new(
        schema.clone(),
        vec![std::sync::Arc::new(
            TimestampMicrosecondArray::from(vec![Some(in_range)]).with_timezone("UTC"),
        )],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&ok_batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.commit_checkpoint(&BatchCheckpoint {
        sequence: 0,
        rows_committed: 1,
        bytes_committed: 0,
        source_fingerprint: None,
        source_position: serde_json::json!({"row": 1}),
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let stored: Option<String> = connection
        .query_first("SELECT occurred_at FROM elm_mysql_timestamp_range")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(stored, Some("2024-03-01 07:20:30.654321".into()));

    connection
        .query_drop("DROP TABLE elm_mysql_timestamp_range")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn sink_begin_rejects_a_user_without_create_privilege() {
    use elm_core::ConsistencyMode;

    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP USER IF EXISTS elm_restricted_user")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("CREATE USER elm_restricted_user IDENTIFIED BY 'elm-restricted-only'")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("GRANT SELECT ON elm_test.* TO elm_restricted_user")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut restricted_environment = environment.clone();
    restricted_environment.username = "elm_restricted_user".into();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let job_id = JobId::new();
    let mut sink = MySqlSink::connect(
        &restricted_environment,
        "elm-restricted-only",
        relation("elm_mysql_privilege_target"),
        job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema, WriteMode::Fail, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let result = sink.begin().await;
    assert!(matches!(result, Err(ElmError::PermissionDenied(_))));

    connection
        .query_drop("DROP USER elm_restricted_user")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn engine_round_trips_through_parquet_with_reserved_identifiers_and_canonical_types() {
    use elm_core::{FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_file_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_file_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_my_file_source (
                `select` BIGINT,
                label VARCHAR(100) CHARACTER SET utf8mb4,
                amount DECIMAL(18, 4),
                payload VARBINARY(100),
                occurred_at TIMESTAMP(6) NULL
             )",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "INSERT INTO elm_my_file_source VALUES
             (1, 'İstanbul 🌍', -12345.6700, X'0001FF', '2024-03-01 07:20:30.654321'),
             (2, NULL, NULL, NULL, NULL)",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("export.parquet");
    let stage = directory.path().join("stage");

    let selection = DatabaseSelection::Table {
        relation: relation("elm_my_file_source"),
    };
    let source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let export_spec = JobSpec::new(
        SourceSpec::Database {
            environment_id: environment.id,
            selection,
            resume_key: Vec::new(),
        },
        SinkSpec::File {
            path: path.clone(),
            format: FileFormat::Parquet,
        },
    );
    let export_sink =
        elm_connectors::RecoverableFileSink::new(&path, FileFormat::Parquet, &stage, None)
            .unwrap_or_else(|error| panic!("{error}"));
    TransferEngine::new(
        export_spec,
        CancellationToken::new(),
        std::sync::Arc::new(NoopObserver),
    )
    .run(Box::new(source), Box::new(export_sink))
    .await
    .unwrap_or_else(|error| panic!("{error}"));

    let target = relation("elm_my_file_target");
    let file_source = elm_connectors::FileSource::open(&path, FileFormat::Parquet)
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let import_sink = MySqlSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let import_spec = JobSpec::new(
        SourceSpec::File {
            path: path.clone(),
            format: FileFormat::Parquet,
        },
        SinkSpec::Database {
            environment_id: environment.id,
            relation: target.clone(),
        },
    );
    TransferEngine::new(
        import_spec,
        CancellationToken::new(),
        std::sync::Arc::new(NoopObserver),
    )
    .run(Box::new(file_source), Box::new(import_sink))
    .await
    .unwrap_or_else(|error| panic!("{error}"));

    let mut readback = MySqlSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: target.clone(),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let batch = readback
        .next_batch(8192)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected reimported rows"));
    assert_eq!(batch.num_rows(), 2);
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| panic!("expected select id"));
    let labels = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| panic!("expected label"));
    let amounts = batch
        .column(2)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap_or_else(|| panic!("expected amount"));
    let payloads = batch
        .column(3)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap_or_else(|| panic!("expected payload"));
    for row in 0..2 {
        match ids.value(row) {
            1 => {
                assert_eq!(labels.value(row), "İstanbul 🌍");
                assert_eq!(amounts.value(row), -123_456_700);
                assert_eq!(payloads.value(row), [0x00, 0x01, 0xff]);
            }
            2 => {
                assert!(labels.is_null(row));
                assert!(amounts.is_null(row));
                assert!(payloads.is_null(row));
            }
            other => panic!("unexpected reimported id {other}"),
        }
    }

    connection
        .query_drop("DROP TABLE elm_my_file_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE elm_my_file_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn engine_round_trips_csv_and_ndjson_including_empty_results() {
    use elm_core::{FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_text_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_my_text_source (`select` BIGINT, label VARCHAR(100) CHARACTER SET utf8mb4, event_day DATE)",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "INSERT INTO elm_my_text_source VALUES
             (1, 'İstanbul 🌍', DATE '2024-02-29'),
             (2, NULL, NULL)",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    for format in [FileFormat::Csv, FileFormat::Ndjson] {
        for empty in [false, true] {
            let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
            let path = directory.path().join("export");
            let stage = directory.path().join("stage");
            let sql = if empty {
                "SELECT * FROM elm_my_text_source WHERE 1=0"
            } else {
                "SELECT * FROM elm_my_text_source ORDER BY `select`"
            };
            let selection = DatabaseSelection::Query { sql: sql.into() };
            let source = MySqlSource::connect(&environment, &password, &selection)
                .await
                .unwrap_or_else(|error| panic!("{error}"));
            let export_spec = JobSpec::new(
                SourceSpec::Database {
                    environment_id: environment.id,
                    selection,
                    resume_key: Vec::new(),
                },
                SinkSpec::File {
                    path: path.clone(),
                    format,
                },
            );
            let export_sink = elm_connectors::RecoverableFileSink::new(&path, format, &stage, None)
                .unwrap_or_else(|error| panic!("{error}"));
            TransferEngine::new(
                export_spec,
                CancellationToken::new(),
                std::sync::Arc::new(NoopObserver),
            )
            .run(Box::new(source), Box::new(export_sink))
            .await
            .unwrap_or_else(|error| panic!("{error}"));

            if empty {
                match format {
                    FileFormat::Csv => assert_eq!(
                        std::fs::read_to_string(&path)
                            .unwrap_or_else(|error| panic!("{error}"))
                            .trim(),
                        "select,label,event_day"
                    ),
                    FileFormat::Ndjson => assert!(
                        std::fs::read(&path)
                            .unwrap_or_else(|error| panic!("{error}"))
                            .is_empty()
                    ),
                    FileFormat::Parquet => unreachable!(),
                }
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{error}"));
            assert!(content.contains("İstanbul 🌍"));
            assert!(content.contains("2024-02-29"));

            let target = relation("elm_my_text_target");
            connection
                .query_drop("DROP TABLE IF EXISTS elm_my_text_target")
                .await
                .unwrap_or_else(|error| panic!("{error}"));
            let file_source = elm_connectors::FileSource::open(&path, format)
                .unwrap_or_else(|error| panic!("{error}"));
            let job_id = JobId::new();
            let import_sink =
                MySqlSink::connect(&environment, &password, target.clone(), job_id, None)
                    .await
                    .unwrap_or_else(|error| panic!("{error}"));
            let import_spec = JobSpec::new(
                SourceSpec::File {
                    path: path.clone(),
                    format,
                },
                SinkSpec::Database {
                    environment_id: environment.id,
                    relation: target.clone(),
                },
            );
            TransferEngine::new(
                import_spec,
                CancellationToken::new(),
                std::sync::Arc::new(NoopObserver),
            )
            .run(Box::new(file_source), Box::new(import_sink))
            .await
            .unwrap_or_else(|error| panic!("{error}"));
            let mut readback = MySqlSource::connect(
                &environment,
                &password,
                &DatabaseSelection::Table {
                    relation: target.clone(),
                },
            )
            .await
            .unwrap_or_else(|error| panic!("{error}"));
            let batch = readback
                .next_batch(8192)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .unwrap_or_else(|| panic!("expected reimported rows"));
            assert_eq!(batch.num_rows(), 2);
            let labels = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap_or_else(|| panic!("expected label"));
            assert!((0..2).any(|row| !labels.is_null(row) && labels.value(row) == "İstanbul 🌍"));
            connection
                .query_drop("DROP TABLE elm_my_text_target")
                .await
                .unwrap_or_else(|error| panic!("{error}"));
        }
    }

    connection
        .query_drop("DROP TABLE elm_my_text_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn engine_rejects_an_explicit_lossy_conversion_of_an_unparseable_value() {
    use elm_core::{ConversionRule, FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_conversion_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("bad.csv");
    std::fs::write(&path, "value\n1\nnot-a-number\n").unwrap_or_else(|error| panic!("{error}"));

    let target = relation("elm_my_conversion_target");
    let file_source = elm_connectors::FileSource::open(&path, FileFormat::Csv)
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let sink = MySqlSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut spec = JobSpec::new(
        SourceSpec::File {
            path: path.clone(),
            format: FileFormat::Csv,
        },
        SinkSpec::Database {
            environment_id: environment.id,
            relation: target.clone(),
        },
    );
    spec.conversions.push(ConversionRule {
        column: Identifier::new("value").unwrap_or_else(|error| panic!("{error}")),
        target_type: "int64".into(),
        allow_lossy: true,
    });
    let result = TransferEngine::new(
        spec,
        CancellationToken::new(),
        std::sync::Arc::new(NoopObserver),
    )
    .run(Box::new(file_source), Box::new(sink))
    .await;
    assert!(matches!(result, Err(ElmError::TypeMapping(_))));

    let exists: Option<u8> = connection
        .exec_first(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
            (environment.database.as_str(), "elm_my_conversion_target"),
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        exists.is_none(),
        "a failed conversion must not leave a partial target"
    );
}

#[tokio::test]
#[ignore = "requires a MySQL test instance"]
async fn engine_cancellation_mid_transfer_leaves_no_new_target() {
    use elm_core::{JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

    let environment = environment();
    let password = password();
    let mut connection = Conn::new(connection_options(&environment, &password))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_cancel_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("DROP TABLE IF EXISTS elm_my_cancel_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "CREATE TABLE elm_my_cancel_source (id BIGINT NOT NULL, label TEXT) ENGINE=InnoDB",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop("SET SESSION cte_max_recursion_depth = 60000")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection
        .query_drop(
            "INSERT INTO elm_my_cancel_source
             WITH RECURSIVE seq AS (
                 SELECT 1 AS n
                 UNION ALL
                 SELECT n + 1 FROM seq WHERE n < 50000
             )
             SELECT n, REPEAT('x', 2000) FROM seq",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_my_cancel_source"),
    };
    let source = MySqlSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let target = relation("elm_my_cancel_target");
    let job_id = JobId::new();
    let sink = MySqlSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut spec = JobSpec::new(
        SourceSpec::Database {
            environment_id: environment.id,
            selection,
            resume_key: Vec::new(),
        },
        SinkSpec::Database {
            environment_id: environment.id,
            relation: target.clone(),
        },
    );
    spec.batch_target_bytes = elm_core::MIN_BATCH_TARGET_BYTES;
    let cancellation = CancellationToken::new();
    let engine = TransferEngine::new(
        spec,
        cancellation.clone(),
        std::sync::Arc::new(NoopObserver),
    );
    let handle = tokio::spawn(engine.run(Box::new(source), Box::new(sink)));
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    cancellation.cancel();
    let result = handle.await.unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(result, Err(ElmError::Cancelled)));

    let exists: Option<u8> = connection
        .exec_first(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
            (environment.database.as_str(), "elm_my_cancel_target"),
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        exists.is_none(),
        "a cancelled transfer must not leave a new target"
    );

    connection
        .query_drop("DROP TABLE elm_my_cancel_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
}
