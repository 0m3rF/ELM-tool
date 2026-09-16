use std::time::Duration;

use elm_connectors::PostgresSource;
use elm_core::{
    ConsistencyMode, DataSource, DatabaseKind, DatabaseSelection, Environment, EnvironmentId,
    Identifier, JobSpec, Relation, SinkSpec, SourceSpec, WriteMode,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, DaemonRuntime, RuntimePaths};
use elm_state::{CredentialVault, KeyringVault};
use secrecy::SecretString;

/// Live-tested against the disposable PostgreSQL fixture in `tests/cross-database/compose.yml`
/// (`docker compose -p elm-daemon-preview -f tests/cross-database/compose.yml up -d postgres
/// --wait`, `ELM_TEST_POSTGRES_PASSWORD` set to the fixture's `ELM_MATRIX_PASSWORD`). Ignored by
/// default like the Oracle daemon test, since it needs a running fixture and a real OS keychain.
#[tokio::test]
#[ignore = "requires the disposable PostgreSQL fixture and an unlocked native OS keychain"]
async fn daemon_preview_runs_real_preflight_without_creating_a_table_or_a_job() {
    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let runtime = DaemonRuntime::open(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    let daemon = tokio::spawn(runtime.serve());
    let client = DaemonClient::connect(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match client.request(Operation::DaemonStatus).await {
                Ok(Response::Status { .. }) => break,
                Err(elm_core::ElmError::Io(error))
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                    ) =>
                {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
                response => panic!("unexpected daemon readiness response: {response:?}"),
            }
        }
    })
    .await
    .unwrap_or_else(|error| panic!("daemon did not become ready: {error}"));

    let id = EnvironmentId::new();
    let environment = Environment {
        id,
        name: format!("PostgreSQL preview fixture {id}"),
        kind: DatabaseKind::PostgreSql,
        host: std::env::var("ELM_TEST_POSTGRES_HOST").unwrap_or_else(|_| "127.0.0.1".into()),
        port: std::env::var("ELM_TEST_POSTGRES_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(55441),
        database: std::env::var("ELM_TEST_POSTGRES_DATABASE")
            .unwrap_or_else(|_| "elm_matrix".into()),
        username: std::env::var("ELM_TEST_POSTGRES_USER").unwrap_or_else(|_| "postgres".into()),
        credential_ref: format!("postgres-preview-fixture-{id}"),
        options: serde_json::json!({"ssl_mode": "disable"}),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    let password = std::env::var("ELM_TEST_POSTGRES_PASSWORD")
        .unwrap_or_else(|_| panic!("set isolated fixture password"));
    let vault = KeyringVault::default();
    vault
        .put(
            &environment.credential_ref,
            &SecretString::from(password.clone()),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .request(Operation::EnvironmentAdd(environment.clone()))
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let target = Relation {
        catalog: None,
        schema: None,
        name: Identifier::new(format!("elm_daemon_preview_{}", id.0.simple()))
            .unwrap_or_else(|error| panic!("{error}")),
    };
    let mut spec = JobSpec::new(
        SourceSpec::Database {
            environment_id: id,
            selection: DatabaseSelection::Query {
                sql: "SELECT CAST(1 AS BIGINT) AS id, CAST('Ankara' AS TEXT) AS city".into(),
            },
            resume_key: Vec::new(),
        },
        SinkSpec::Database {
            environment_id: id,
            relation: target.clone(),
        },
    );
    spec.write_mode = WriteMode::Append;
    spec.consistency = ConsistencyMode::Atomic;

    let response = client
        .request(Operation::JobPreview(spec))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let Response::Preview(preview) = response else {
        panic!("expected a preview response, got {response:?}");
    };
    assert_eq!(preview.columns.len(), 2);
    assert_eq!(preview.columns[0].name, "id");
    assert_eq!(preview.columns[1].name, "city");
    assert!(
        preview.publication_error.is_none(),
        "unexpected publication error: {:?}",
        preview.publication_error
    );
    assert_eq!(
        preview.report.capabilities.database,
        Some(DatabaseKind::PostgreSql)
    );

    // The preview must never create the destination table.
    let mut existence_check = PostgresSource::connect(
        &environment,
        &password,
        &DatabaseSelection::Query {
            sql: format!(
                "SELECT to_regclass('{}') IS NOT NULL AS exists",
                target.name.as_str()
            ),
        },
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    let batch = existence_check
        .next_batch(1024 * 1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("existence check returned no rows"));
    let exists = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::BooleanArray>()
        .unwrap_or_else(|| panic!("expected a boolean column"))
        .value(0);
    assert!(
        !exists,
        "preview must not create the destination table {}",
        target.name.as_str()
    );

    // A preview must never be visible through job.list or job.get.
    let jobs = match client
        .request(Operation::JobList)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Jobs(jobs) => jobs,
        other => panic!("expected a jobs response, got {other:?}"),
    };
    assert!(jobs.is_empty());

    client
        .request(Operation::EnvironmentRemove(id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    vault
        .remove(&environment.credential_ref)
        .unwrap_or_else(|error| panic!("{error}"));
    client
        .request(Operation::DaemonStop)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    tokio::time::timeout(Duration::from_secs(5), daemon)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|error| panic!("{error}"));
}
