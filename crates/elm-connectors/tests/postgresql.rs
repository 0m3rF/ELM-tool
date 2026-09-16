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

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn physical_identity_resolves_through_a_view_an_alternate_address_and_rejects_unrelated_tables()
 {
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
            "DROP VIEW IF EXISTS public.elm_pg_identity_view;
             DROP TABLE IF EXISTS public.elm_pg_identity_base;
             DROP TABLE IF EXISTS public.elm_pg_identity_other;
             CREATE TABLE public.elm_pg_identity_base (id BIGINT);
             CREATE TABLE public.elm_pg_identity_other (id BIGINT);
             CREATE VIEW public.elm_pg_identity_view AS SELECT * FROM public.elm_pg_identity_base;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let base = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_pg_identity_base"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable base table identity"));
    let via_view = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_pg_identity_view"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable view identity"));
    let other = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_pg_identity_other"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable unrelated table identity"));
    let missing = elm_connectors::physical_identity::resolve(
        &environment,
        &password,
        &relation("elm_pg_identity_missing"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));

    // A second environment record reaching the same physical server through a different
    // configured hostname must resolve to the same fingerprint, not a different one.
    let mut aliased_environment = environment.clone();
    aliased_environment.host = "localhost".into();
    let via_alternate_address = elm_connectors::physical_identity::resolve(
        &aliased_environment,
        &password,
        &relation("elm_pg_identity_base"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|| panic!("expected a resolvable identity through the alternate host"));

    assert!(base.same_physical_table(&via_view));
    assert!(base.same_physical_table(&via_alternate_address));
    assert!(!base.same_physical_table(&other));
    assert!(missing.is_none());

    client
        .batch_execute(
            "DROP VIEW public.elm_pg_identity_view;
             DROP TABLE public.elm_pg_identity_base;
             DROP TABLE public.elm_pg_identity_other;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn sink_preflight_rejects_a_role_without_create_privilege() {
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
    let restricted_password = "elm-restricted-only";
    let _ = client
        .batch_execute("DROP OWNED BY elm_restricted_role")
        .await;
    client
        .batch_execute("DROP ROLE IF EXISTS elm_restricted_role")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute(&format!(
            "CREATE ROLE elm_restricted_role LOGIN PASSWORD '{restricted_password}'"
        ))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute("REVOKE CREATE ON SCHEMA public FROM elm_restricted_role")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute("GRANT USAGE ON SCHEMA public TO elm_restricted_role")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut restricted_environment = environment.clone();
    restricted_environment.username = "elm_restricted_role".into();
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int64,
        false,
    )]));
    let job_id = JobId::new();
    let mut sink = PostgresSink::connect(
        &restricted_environment,
        restricted_password,
        relation("elm_pg_privilege_target"),
        job_id,
        None,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let result = sink
        .preflight(schema, WriteMode::Fail, ConsistencyMode::Atomic)
        .await;
    assert!(matches!(result, Err(ElmError::PermissionDenied(_))));
    drop(sink);

    client
        .batch_execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
             WHERE usename = 'elm_restricted_role'",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute("DROP OWNED BY elm_restricted_role")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute("DROP ROLE elm_restricted_role")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn engine_round_trips_through_parquet_with_reserved_identifiers_and_canonical_types() {
    use arrow_array::Array;
    use elm_core::{FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

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
            "DROP TABLE IF EXISTS public.elm_pg_file_source;
             DROP TABLE IF EXISTS public.elm_pg_file_target;
             CREATE TABLE public.elm_pg_file_source (
                 \"select\" BIGINT,
                 label TEXT,
                 amount NUMERIC(18,4),
                 payload BYTEA,
                 occurred_at TIMESTAMPTZ
             );
             INSERT INTO public.elm_pg_file_source VALUES
                 (1, 'İstanbul 🌍', -12345.6700, decode('0001ff', 'hex'), TIMESTAMPTZ '2024-03-01 10:20:30+03'),
                 (2, NULL, NULL, NULL, NULL);",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("export.parquet");
    let stage = directory.path().join("stage");

    let selection = DatabaseSelection::Table {
        relation: relation("elm_pg_file_source"),
    };
    let source = PostgresSource::connect(&environment, &password, &selection)
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

    let target = relation("elm_pg_file_target");
    let file_source = elm_connectors::FileSource::open(&path, FileFormat::Parquet)
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let import_sink = PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
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

    let mut readback = PostgresSource::connect(
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
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap_or_else(|| panic!("expected label"));
    let amounts = batch
        .column(2)
        .as_any()
        .downcast_ref::<arrow_array::Decimal128Array>()
        .unwrap_or_else(|| panic!("expected amount"));
    let payloads = batch
        .column(3)
        .as_any()
        .downcast_ref::<arrow_array::BinaryArray>()
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

    client
        .batch_execute(
            "DROP TABLE public.elm_pg_file_source;
             DROP TABLE public.elm_pg_file_target;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn engine_round_trips_csv_and_ndjson_including_empty_results() {
    use elm_core::{FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

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
            "DROP TABLE IF EXISTS public.elm_pg_text_source;
             CREATE TABLE public.elm_pg_text_source (\"select\" BIGINT, label TEXT, event_day DATE);
             INSERT INTO public.elm_pg_text_source VALUES
                 (1, 'İstanbul 🌍', DATE '2024-02-29'),
                 (2, NULL, NULL);",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    for format in [FileFormat::Csv, FileFormat::Ndjson] {
        for empty in [false, true] {
            let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
            let path = directory.path().join("export");
            let stage = directory.path().join("stage");
            let sql = if empty {
                "SELECT * FROM public.elm_pg_text_source WHERE 1=0"
            } else {
                "SELECT * FROM public.elm_pg_text_source ORDER BY \"select\""
            };
            let selection = DatabaseSelection::Query { sql: sql.into() };
            let source = PostgresSource::connect(&environment, &password, &selection)
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

            let target = relation("elm_pg_text_target");
            client
                .batch_execute("DROP TABLE IF EXISTS public.elm_pg_text_target")
                .await
                .unwrap_or_else(|error| panic!("{error}"));
            let file_source = elm_connectors::FileSource::open(&path, format)
                .unwrap_or_else(|error| panic!("{error}"));
            let job_id = JobId::new();
            let import_sink =
                PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
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
            let mut readback = PostgresSource::connect(
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
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap_or_else(|| panic!("expected label"));
            assert!((0..2).any(|row| !arrow_array::Array::is_null(labels, row)
                && labels.value(row) == "İstanbul 🌍"));
            client
                .batch_execute("DROP TABLE public.elm_pg_text_target")
                .await
                .unwrap_or_else(|error| panic!("{error}"));
        }
    }

    client
        .batch_execute("DROP TABLE public.elm_pg_text_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn engine_rejects_an_explicit_lossy_conversion_of_an_unparseable_value() {
    use elm_core::{ConversionRule, FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

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
        .batch_execute("DROP TABLE IF EXISTS public.elm_pg_conversion_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("bad.csv");
    std::fs::write(&path, "\"select\"\n1\nnot-a-number\n")
        .unwrap_or_else(|error| panic!("{error}"));

    let target = relation("elm_pg_conversion_target");
    let file_source = elm_connectors::FileSource::open(&path, FileFormat::Csv)
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let sink = PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
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
        column: Identifier::new("select").unwrap_or_else(|error| panic!("{error}")),
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

    let exists: bool = client
        .query_one(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relname = 'elm_pg_conversion_target')",
            &[],
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert!(
        !exists,
        "a failed conversion must not leave a partial target"
    );

    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn duplicate_source_rows_fail_closed_and_preserve_the_target() {
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
            "DROP TABLE IF EXISTS public.elm_pg_conflict_target;
             DROP TABLE IF EXISTS public.elm_pg_conflict_source;
             CREATE TABLE public.elm_pg_conflict_target (id BIGINT NOT NULL PRIMARY KEY);
             INSERT INTO public.elm_pg_conflict_target VALUES (99);
             CREATE TABLE public.elm_pg_conflict_source (id BIGINT NOT NULL);
             INSERT INTO public.elm_pg_conflict_source VALUES (1), (1);",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut source = PostgresSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Table {
            relation: relation("elm_pg_conflict_source"),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let job_id = JobId::new();
    let target = relation("elm_pg_conflict_target");
    let mut sink = PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema, WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));

    let mut sequence = 0_u64;
    let mut rows = 0_u64;
    let mut staging_rejected = false;
    while let Some(batch) = source
        .next_batch(64)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        match sink.write_batch(&batch).await {
            Ok(()) => {
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
                sequence += 1;
            }
            Err(_) => {
                staging_rejected = true;
                break;
            }
        }
    }
    // PostgreSQL's staging table for APPEND inherits the target's constraints
    // (`CREATE TABLE ... LIKE target INCLUDING ALL`), so a duplicate primary key is rejected
    // while staging the batch, not at publish; either failure point must preserve the target.
    if !staging_rejected {
        assert!(sink.publish().await.is_err());
    }

    let rows: Vec<i64> = client
        .query("SELECT id FROM public.elm_pg_conflict_target", &[])
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .iter()
        .map(|row| row.get(0))
        .collect();
    assert_eq!(rows, vec![99]);

    elm_connectors::cleanup_postgres_staging(&environment, &password, &target, job_id)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute(
            "DROP TABLE public.elm_pg_conflict_source;
             DROP TABLE public.elm_pg_conflict_target;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn engine_cancellation_mid_transfer_leaves_no_new_target() {
    use elm_core::{JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

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
            "DROP TABLE IF EXISTS public.elm_pg_cancel_source;
             DROP TABLE IF EXISTS public.elm_pg_cancel_target;
             CREATE TABLE public.elm_pg_cancel_source (id BIGINT NOT NULL, label TEXT);
             INSERT INTO public.elm_pg_cancel_source
                 SELECT generator, repeat('x', 2000)
                 FROM generate_series(1, 50000) AS generator;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_pg_cancel_source"),
    };
    let source = PostgresSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let target = relation("elm_pg_cancel_target");
    let job_id = JobId::new();
    let sink = PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
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

    let exists: bool = client
        .query_one(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relname = 'elm_pg_cancel_target')",
            &[],
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert!(!exists, "a cancelled transfer must not leave a new target");

    elm_connectors::cleanup_postgres_staging(&environment, &password, &target, job_id)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .batch_execute("DROP TABLE public.elm_pg_cancel_source")
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    connection_task.abort();
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance and Docker control over its container"]
async fn engine_reports_a_clean_error_when_the_connection_is_severed_mid_transfer() {
    use elm_core::{JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

    let container = std::env::var("ELM_TEST_POSTGRES_DOCKER_CONTAINER").unwrap_or_else(|_| {
        panic!("set ELM_TEST_POSTGRES_DOCKER_CONTAINER to the disposable fixture container name")
    });
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
            "DROP TABLE IF EXISTS public.elm_pg_severed_source;
             DROP TABLE IF EXISTS public.elm_pg_severed_target;
             CREATE TABLE public.elm_pg_severed_source (id BIGINT NOT NULL, label TEXT);
             INSERT INTO public.elm_pg_severed_source
                 SELECT generator, repeat('x', 4000)
                 FROM generate_series(1, 200000) AS generator;",
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let selection = DatabaseSelection::Table {
        relation: relation("elm_pg_severed_source"),
    };
    let source = PostgresSource::connect(&environment, &password, &selection)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let target = relation("elm_pg_severed_target");
    let job_id = JobId::new();
    let sink = PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
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
    let engine = TransferEngine::new(
        spec,
        CancellationToken::new(),
        std::sync::Arc::new(NoopObserver),
    );
    let handle = tokio::spawn(engine.run(Box::new(source), Box::new(sink)));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let restart = std::process::Command::new("docker")
        .args(["restart", "--timeout", "0", &container])
        .status()
        .unwrap_or_else(|error| panic!("failed to invoke docker: {error}"));
    assert!(restart.success(), "failed to restart the fixture container");

    let result = tokio::time::timeout(std::time::Duration::from_secs(30), handle)
        .await
        .unwrap_or_else(|_| panic!("engine did not report the severed connection in time"))
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        result.is_err(),
        "a severed connection must surface as a job failure, not a silent success"
    );
    assert!(
        !matches!(result, Err(ElmError::Cancelled)),
        "the failure must be reported as a connection error, not spurious cancellation"
    );
    connection_task.abort();

    // This test deliberately disrupts a shared fixture; block until it is genuinely healthy
    // again so later tests in the same run do not see spurious connection failures.
    for attempt in 0..60 {
        if test_postgres_environment(&environment, &password)
            .await
            .is_ok()
        {
            return;
        }
        let _ = attempt;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    panic!("PostgreSQL fixture did not become healthy again after the restart");
}

#[tokio::test]
#[ignore = "requires a PostgreSQL test instance"]
async fn engine_rejects_a_malformed_row_without_leaving_a_partial_target() {
    use elm_core::{FileFormat, JobSpec, SinkSpec, SourceSpec};
    use elm_engine::{NoopObserver, TransferEngine};
    use tokio_util::sync::CancellationToken;

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
        .batch_execute("DROP TABLE IF EXISTS public.elm_pg_malformed_target")
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("malformed.csv");
    // The third row declares an extra field: this is a structurally malformed row, distinct
    // from a value that merely fails an explicit type conversion.
    std::fs::write(&path, "id,label\n1,first\n2,second\n3,third,extra\n")
        .unwrap_or_else(|error| panic!("{error}"));

    let target = relation("elm_pg_malformed_target");
    let job_id = JobId::new();
    let file_source = elm_connectors::FileSource::open(&path, FileFormat::Csv);
    let malformed_at_open = file_source.is_err();
    let (result, target_touched) = if let Ok(file_source) = file_source {
        let sink = PostgresSink::connect(&environment, &password, target.clone(), job_id, None)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        let spec = JobSpec::new(
            SourceSpec::File {
                path: path.clone(),
                format: FileFormat::Csv,
            },
            SinkSpec::Database {
                environment_id: environment.id,
                relation: target.clone(),
            },
        );
        let result = TransferEngine::new(
            spec,
            CancellationToken::new(),
            std::sync::Arc::new(NoopObserver),
        )
        .run(Box::new(file_source), Box::new(sink))
        .await;
        (Some(result), true)
    } else {
        (None, false)
    };

    assert!(
        malformed_at_open || matches!(result, Some(Err(_))),
        "a structurally malformed row must fail closed, either at open or during the transfer"
    );
    if target_touched {
        elm_connectors::cleanup_postgres_staging(&environment, &password, &target, job_id)
            .await
            .ok();
    }
    let exists: bool = client
        .query_one(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relname = 'elm_pg_malformed_target')",
            &[],
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .get(0);
    assert!(
        !exists,
        "a malformed source must not leave a partial target"
    );

    connection_task.abort();
}
