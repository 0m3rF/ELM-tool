#![cfg(feature = "postgresql")]

use arrow_array::Int64Array;
use chrono::Utc;
use elm_connectors::{PostgresSink, PostgresSource, test_postgres_environment};
use elm_core::{
    BatchCheckpoint, ConsistencyMode, DataSink, DataSource, DatabaseKind, DatabaseSelection,
    ElmError, Environment, EnvironmentId, Identifier, JobId, Relation, WriteMode,
};
use tokio_postgres::NoTls;

const DEFAULT_PASSWORD: &str = "elm_integration_only";

fn relation(name: &str) -> Relation {
    Relation {
        catalog: None,
        schema: Some(Identifier::new("public").unwrap_or_else(|error| panic!("{error}"))),
        name: Identifier::new(name).unwrap_or_else(|error| panic!("{error}")),
    }
}

fn environment() -> Environment {
    let port = std::env::var("ELM_TEST_POSTGRES_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(55432);
    Environment {
        id: EnvironmentId::new(),
        name: "PostgreSQL integration".into(),
        kind: DatabaseKind::PostgreSql,
        host: "127.0.0.1".into(),
        port,
        database: "elm_test".into(),
        username: "postgres".into(),
        credential_ref: "integration-only".into(),
        options: serde_json::json!({ "ssl_mode": "disable" }),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    }
}

fn password() -> String {
    std::env::var("ELM_TEST_POSTGRES_PASSWORD").unwrap_or_else(|_| DEFAULT_PASSWORD.into())
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn binary_numeric_reads_full_precision_with_group_padding() {
    let maximum = 10_i128.pow(38) - 1;
    let selection = DatabaseSelection::Query {
        sql: format!(
            "SELECT '{}.9'::numeric(38,1) AS positive, '-{}.999'::numeric(38,3) AS negative, '{}00'::numeric(38,-2) AS negative_scale",
            "9".repeat(37),
            "9".repeat(35),
            "9".repeat(38),
        ),
    };
    let mut source = PostgresSource::connect(&environment(), &password(), &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let batch = source
        .next_batch(8192)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected numeric row"));
    assert_eq!(batch.num_rows(), 1);
    for (index, expected) in [maximum, -maximum, maximum].into_iter().enumerate() {
        let array = batch
            .column(index)
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .unwrap_or_else(|| panic!("expected decimal"));
        assert_eq!(array.value(0), expected);
    }
    assert!(
        source
            .next_batch(8192)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .is_none()
    );
}

async fn stage_all(source: &mut PostgresSource, sink: &mut PostgresSink) -> u64 {
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
        let checkpoint = BatchCheckpoint {
            sequence,
            source_position: source
                .checkpoint()
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            rows_committed: rows,
            bytes_committed: 0,
            source_fingerprint: None,
        };
        sink.commit_checkpoint(&checkpoint)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        sequence = sequence.saturating_add(1);
    }
    rows
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn binary_copy_replace_is_invisible_until_atomic_publish() {
    let environment = environment();
    let password = password();
    test_postgres_environment(&environment, &password)
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut configuration = tokio_postgres::Config::new();
    configuration
        .host(&environment.host)
        .port(environment.port)
        .dbname(&environment.database)
        .user(&environment.username)
        .password(&password);
    let (client, connection) = configuration
        .connect(NoTls)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let connection_task = tokio::spawn(async move {
        let _result = connection.await;
    });
    client
        .batch_execute(
            "DROP TABLE IF EXISTS public.elm_pg_target;
             DROP TABLE IF EXISTS public.elm_pg_source;
             CREATE TABLE public.elm_pg_source (
                \"select\" BIGINT,
                label TEXT,
                enabled BOOLEAN,
                payload BYTEA,
                day DATE,
                occurred_at TIMESTAMP WITH TIME ZONE,
                amount NUMERIC(18, 4),
                bucket NUMERIC(8, -2)
             );
             INSERT INTO public.elm_pg_source VALUES
                (1, 'İstanbul', true, decode('0001ff', 'hex'), DATE '2024-02-29', TIMESTAMPTZ '2024-03-01 10:20:30+03', -12345.6700, 12345),
                (2, NULL, false, NULL, NULL, NULL, NULL, NULL);
             CREATE TABLE public.elm_pg_target (legacy INTEGER);
             INSERT INTO public.elm_pg_target VALUES (99);",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut source = PostgresSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_pg_source"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let mut sink = PostgresSink::connect(
        &environment,
        &password,
        relation("elm_pg_target"),
        job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema, WriteMode::Replace, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));

    let rows = stage_all(&mut source, &mut sink).await;
    assert_eq!(rows, 2);

    let visible_before: i32 = client
        .query_one("SELECT legacy FROM public.elm_pg_target", &[])
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert_eq!(visible_before, 99);

    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let output = client
        .query(
            "SELECT \"select\", label, enabled, encode(payload, 'hex'), day::text,
                    occurred_at AT TIME ZONE 'UTC', amount::text, bucket::text
             FROM public.elm_pg_target ORDER BY \"select\"",
            &[],
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(output.len(), 2);
    assert_eq!(output[0].get::<_, i64>(0), 1);
    assert_eq!(output[0].get::<_, Option<&str>>(1), Some("İstanbul"));
    assert!(output[0].get::<_, bool>(2));
    assert_eq!(output[0].get::<_, Option<&str>>(3), Some("0001ff"));
    assert_eq!(output[0].get::<_, Option<&str>>(4), Some("2024-02-29"));
    assert_eq!(output[0].get::<_, Option<&str>>(6), Some("-12345.6700"));
    assert_eq!(output[0].get::<_, Option<&str>>(7), Some("12300"));
    assert_eq!(output[1].get::<_, Option<&str>>(1), None);
    assert_eq!(output[1].get::<_, Option<&str>>(6), None);

    let mut append_source = PostgresSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_pg_source"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let append_schema = append_source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut append_sink = PostgresSink::connect(
        &environment,
        &password,
        relation("elm_pg_target"),
        JobId::new(),
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
    let count_before_append: i64 = client
        .query_one("SELECT COUNT(*) FROM public.elm_pg_target", &[])
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert_eq!(count_before_append, 2);
    append_sink
        .publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let count_after_append: i64 = client
        .query_one("SELECT COUNT(*) FROM public.elm_pg_target", &[])
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert_eq!(count_after_append, 4);

    let mut fail_sink = PostgresSink::connect(
        &environment,
        &password,
        relation("elm_pg_target"),
        JobId::new(),
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let fail_result = fail_sink
        .preflight(
            append_source
                .schema()
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            WriteMode::Fail,
            ConsistencyMode::Atomic,
        )
        .await;
    assert!(matches!(fail_result, Err(ElmError::Conflict(_))));

    drop(source);
    drop(append_source);
    client
        .batch_execute("TRUNCATE TABLE public.elm_pg_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut empty_source = PostgresSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_pg_source"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let mut empty_sink = PostgresSink::connect(
        &environment,
        &password,
        relation("elm_pg_target"),
        JobId::new(),
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    empty_sink
        .preflight(
            empty_source
                .schema()
                .await
                .unwrap_or_else(|error| panic!("{error}")),
            WriteMode::Replace,
            ConsistencyMode::Atomic,
        )
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
    let empty_count: i64 = client
        .query_one("SELECT COUNT(*) FROM public.elm_pg_target", &[])
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert_eq!(empty_count, 0);

    client
        .batch_execute(
            "DROP TABLE public.elm_pg_target;
             DROP TABLE public.elm_pg_source;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(client);
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn sink_resume_reuses_only_the_marked_committed_stage() {
    let environment = environment();
    let password = password();
    let mut configuration = tokio_postgres::Config::new();
    configuration
        .host(&environment.host)
        .port(environment.port)
        .dbname(&environment.database)
        .user(&environment.username)
        .password(&password);
    let (client, connection) = configuration
        .connect(NoTls)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let connection_task = tokio::spawn(async move {
        let _result = connection.await;
    });
    client
        .batch_execute(
            "DROP TABLE IF EXISTS public.elm_pg_resume_target;
             DROP TABLE IF EXISTS public.elm_pg_resume_source;
             CREATE TABLE public.elm_pg_resume_source (id BIGINT, label TEXT);
             INSERT INTO public.elm_pg_resume_source VALUES (1, 'first'), (2, 'second');
             CREATE TABLE public.elm_pg_resume_target (id BIGINT, label TEXT);",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut source = PostgresSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_pg_resume_source"),
        },
    )
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
        .unwrap_or_else(|| panic!("expected first source batch"));
    let job_id = JobId::new();
    let mut first_attempt = PostgresSink::connect(
        &environment,
        &password,
        relation("elm_pg_resume_target"),
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
    let first_checkpoint = BatchCheckpoint {
        sequence: 0,
        source_position: serde_json::json!({ "row": 1 }),
        rows_committed: 1,
        bytes_committed: 0,
        source_fingerprint: None,
    };
    first_attempt
        .commit_checkpoint(&first_checkpoint)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let second = source
        .next_batch(1)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected second source batch"));
    // Simulate termination after PostgreSQL committed a COPY but before SQLite saved its
    // checkpoint. Resume must discard only this job-marked tail batch and write it once.
    first_attempt
        .write_batch(&second)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    first_attempt
        .abort()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(first_attempt);

    let visible_rows: i64 = client
        .query_one("SELECT COUNT(*) FROM public.elm_pg_resume_target", &[])
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert_eq!(visible_rows, 0);

    let mut resumed = PostgresSink::connect(
        &environment,
        &password,
        relation("elm_pg_resume_target"),
        job_id,
        Some(&first_checkpoint),
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

    let values = client
        .query(
            "SELECT id, label FROM public.elm_pg_resume_target ORDER BY id",
            &[],
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].get::<_, i64>(0), 1);
    assert_eq!(values[1].get::<_, &str>(1), "second");
    client
        .batch_execute(
            "DROP TABLE public.elm_pg_resume_target;
             DROP TABLE public.elm_pg_resume_source;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(client);
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn keyset_resume_uses_parameterized_bounds_and_excludes_newer_rows() {
    let environment = environment();
    let password = password();
    let mut configuration = tokio_postgres::Config::new();
    configuration
        .host(&environment.host)
        .port(environment.port)
        .dbname(&environment.database)
        .user(&environment.username)
        .password(&password);
    let (client, connection) = configuration
        .connect(NoTls)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let connection_task = tokio::spawn(async move {
        let _result = connection.await;
    });
    client
        .batch_execute(
            "DROP TABLE IF EXISTS public.elm_pg_keyset_source;
             CREATE TABLE public.elm_pg_keyset_source (id BIGINT NOT NULL, label TEXT);
             INSERT INTO public.elm_pg_keyset_source
             SELECT value, 'row-' || value::text FROM generate_series(1, 5) AS value;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_pg_keyset_source"),
    };
    let resume_key = vec![Identifier::new("id").unwrap_or_else(|error| panic!("{error}"))];
    let mut initial =
        PostgresSource::connect_resumable(&environment, &password, &selection, &resume_key, None)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    let first = initial
        .next_batch(1)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected a first keyset batch"));
    let first_ids = first
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| panic!("id column was not Int64"));
    assert_eq!(first_ids.values(), &[1]);
    let checkpoint = initial
        .checkpoint()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(initial);

    client
        .execute(
            "INSERT INTO public.elm_pg_keyset_source VALUES ($1, $2)",
            &[&6_i64, &"newer"],
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut resumed = PostgresSource::connect_resumable(
        &environment,
        &password,
        &selection,
        &resume_key,
        Some(&checkpoint),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let mut resumed_ids: Vec<i64> = Vec::new();
    while let Some(batch) = resumed
        .next_batch(64)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("id column was not Int64"));
        resumed_ids.extend(ids.values().iter().copied());
    }
    assert_eq!(resumed_ids, [2, 3, 4, 5]);

    client
        .batch_execute("DROP TABLE public.elm_pg_keyset_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    drop(client);
    connection_task.abort();
}
