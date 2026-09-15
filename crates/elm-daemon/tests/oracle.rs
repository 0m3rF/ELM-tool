use elm_core::{
    ConsistencyMode, DataSource, DatabaseKind, DatabaseSelection, Environment, EnvironmentId,
    FileFormat, Identifier, JobSpec, JobState, Relation, SinkSpec, SourceSpec, WriteMode,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, DaemonRuntime, RuntimePaths};
use elm_state::{CredentialVault, KeyringVault};
use secrecy::SecretString;
use std::{sync::Arc, time::Duration};

async fn run_job(client: &DaemonClient, spec: JobSpec) {
    let id = spec.id;
    assert!(matches!(
        client
            .request(Operation::JobSubmit(spec))
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        Response::Job(_)
    ));
    let mut events = client
        .watch(Operation::JobWatch(id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let terminal = tokio::time::timeout(Duration::from_secs(60), async {
        while let Some(event) = events.recv().await {
            let progress = match event.unwrap_or_else(|error| panic!("{error}")) {
                Response::Job(job) => job.progress,
                Response::Event(progress) => progress,
                _ => continue,
            };
            if progress.state.is_terminal() {
                return progress;
            }
        }
        panic!("Oracle watch ended without a terminal event")
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(terminal.state, JobState::Succeeded, "{:?}", terminal.error);
    assert_eq!(terminal.rows, 9);
}

#[tokio::test]
#[ignore = "requires isolated Oracle, Instant Client, and an unlocked native OS keychain"]
async fn daemon_runs_oracle_bulk_transfer_swap_and_file_export() {
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
    let mut environment = Environment {
        id,
        name: format!("Oracle daemon fixture {id}"),
        kind: DatabaseKind::Oracle,
        host: std::env::var("ELM_TEST_ORACLE_HOST").unwrap_or_else(|_| "localhost".into()),
        port: 1521,
        database: std::env::var("ELM_TEST_ORACLE_DATABASE").unwrap_or_else(|_| "XEPDB1".into()),
        username: std::env::var("ELM_TEST_ORACLE_USER").unwrap_or_else(|_| "elm_test".into()),
        credential_ref: format!("oracle-daemon-fixture-{id}"),
        options: serde_json::json!({"ssl_mode": "disable"}),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    if let Ok(timeout) = std::env::var("ELM_TEST_ORACLE_CALL_TIMEOUT_MS") {
        environment.options["oracle_call_timeout_ms"] = serde_json::json!(
            timeout
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("invalid fixture timeout"))
        );
    }
    let password = std::env::var("ELM_TEST_ORACLE_PASSWORD")
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
    let source = SourceSpec::Database { environment_id: id, selection: DatabaseSelection::Query {
        sql: "SELECT CAST(LEVEL AS NUMBER(10,0)) AS \"id\", CAST(N'İstanbul 🌍' AS NVARCHAR2(32)) AS \"label\" FROM dual CONNECT BY LEVEL <= 9 ORDER BY LEVEL".into(),
    }, resume_key: Vec::new() };
    let target = Relation {
        catalog: None,
        schema: None,
        name: Identifier::new(format!("elm_daemon_{}", id.0.simple()))
            .unwrap_or_else(|error| panic!("{error}")),
    };
    let sink = SinkSpec::Database {
        environment_id: id,
        relation: target.clone(),
    };
    let mut first = JobSpec::new(source.clone(), sink.clone());
    first.write_mode = WriteMode::Fail;
    let first_id = first.id;
    run_job(&client, first).await;
    let mut second = JobSpec::new(source, sink);
    second.write_mode = WriteMode::Replace;
    second.consistency = ConsistencyMode::TableSwap;
    let second_id = second.id;
    run_job(&client, second).await;
    let output = directory.path().join("oracle.parquet");
    let export = JobSpec::new(
        SourceSpec::Database {
            environment_id: id,
            selection: DatabaseSelection::Table {
                relation: target.clone(),
            },
            resume_key: Vec::new(),
        },
        SinkSpec::File {
            path: output.clone(),
            format: FileFormat::Parquet,
        },
    );
    let export_id = export.id;
    run_job(&client, export).await;
    let mut reader = elm_connectors::FileSource::open(&output, FileFormat::Parquet)
        .unwrap_or_else(|error| panic!("{error}"));
    let mut rows = 0;
    while let Some(batch) = reader
        .next_batch(1024 * 1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        rows += batch.num_rows();
    }
    assert_eq!(rows, 9);
    // A fresh client reattaches to persisted history; serialized definitions and
    // progress contain credential references, never the database password.
    let reconnected = DaemonClient::connect(paths).unwrap_or_else(|error| panic!("{error}"));
    let history = reconnected
        .request(Operation::JobList)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(
        !serde_json::to_string(&history)
            .unwrap_or_default()
            .contains(&password)
    );
    for job in [first_id, second_id, export_id] {
        client
            .request(Operation::JobDelete(job))
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    }
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
    tokio::time::timeout(Duration::from_secs(10), daemon)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|error| panic!("{error}"));
    let environment = Arc::new(environment);
    tokio::task::spawn_blocking(move || {
        let connection = oracle_connection(&environment, &password);
        connection
            .execute(&format!("DROP TABLE \"{}\"", target.name), &[])
            .unwrap_or_else(|_| panic!("fixture cleanup failed"));
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
}

fn oracle_connection(environment: &Environment, password: &str) -> oracle::Connection {
    oracle::Connection::connect(
        &environment.username,
        password,
        format!(
            "//{}:{}/{}",
            environment.host, environment.port, environment.database
        ),
    )
    .unwrap_or_else(|_| panic!("fixture connection failed"))
}
