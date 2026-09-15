use std::{fs, time::Duration};

use elm_connectors::{FileSource, RecoverableFileSink};
use elm_core::{
    BatchCheckpoint, ConsistencyMode, DataSink, DataSource, FileFormat, JobSpec, JobState,
    SinkSpec, SourceSpec, WriteMode,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, DaemonRuntime, RuntimePaths};
use elm_state::StateStore;
use tempfile::tempdir;

#[tokio::test]
async fn authenticated_ipc_runs_and_watches_a_durable_transfer() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let runtime = DaemonRuntime::open(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    let daemon = tokio::spawn(runtime.serve());

    let client = DaemonClient::connect(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    let input = directory.path().join("input.csv");
    let output = directory.path().join("output.ndjson");
    fs::write(&input, "id,name\n1,Alice\n2,İstanbul 🌍\n")
        .unwrap_or_else(|error| panic!("{error}"));
    let spec = JobSpec::new(
        SourceSpec::File {
            path: input,
            format: FileFormat::Csv,
        },
        SinkSpec::File {
            path: output.clone(),
            format: FileFormat::Ndjson,
        },
    );
    let id = spec.id;

    let submitted = client
        .request(Operation::JobSubmit(spec))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(matches!(submitted, Response::Job(_)));
    let mut events = client
        .watch(Operation::JobWatch(id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let terminal = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(response) = events.recv().await {
            match response.unwrap_or_else(|error| panic!("{error}")) {
                Response::Job(job) if job.progress.state.is_terminal() => {
                    return job.progress.state;
                }
                Response::Event(progress) if progress.state.is_terminal() => return progress.state,
                _ => {}
            }
        }
        JobState::Interrupted
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(terminal, JobState::Succeeded);
    assert!(
        fs::read_to_string(output)
            .unwrap_or_default()
            .contains("İstanbul 🌍")
    );

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

#[tokio::test]
async fn daemon_resume_reuses_checkpointed_file_staging_without_duplicate_rows() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let input = directory.path().join("input.csv");
    let output = directory.path().join("output.ndjson");
    fs::write(&input, "id,name\n1,Alice\n2,Bob\n3,İstanbul 🌍\n")
        .unwrap_or_else(|error| panic!("{error}"));
    let spec = JobSpec::new(
        SourceSpec::File {
            path: input.clone(),
            format: FileFormat::Csv,
        },
        SinkSpec::File {
            path: output.clone(),
            format: FileFormat::Ndjson,
        },
    );
    let id = spec.id;
    let store = StateStore::open(&paths.database).unwrap_or_else(|error| panic!("{error}"));
    let mut progress = store
        .create_job(&spec)
        .unwrap_or_else(|error| panic!("{error}"))
        .progress;
    progress.state = JobState::Preflighting;
    store
        .record_progress(&progress)
        .unwrap_or_else(|error| panic!("{error}"));
    progress.state = JobState::Running;
    store
        .record_progress(&progress)
        .unwrap_or_else(|error| panic!("{error}"));

    let mut source =
        FileSource::open(&input, FileFormat::Csv).unwrap_or_else(|error| panic!("{error}"));
    let schema = source
        .schema()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let first = source
        .next_batch(1)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected a source row"));
    let staging = paths.job_staging_directory(id);
    store
        .register_recovery_resource(id, "file", &staging.to_string_lossy())
        .unwrap_or_else(|error| panic!("{error}"));
    let mut sink = RecoverableFileSink::new(&output, FileFormat::Ndjson, &staging, None)
        .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema, WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&first)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let checkpoint = BatchCheckpoint {
        sequence: 0,
        source_position: source
            .checkpoint()
            .await
            .unwrap_or_else(|error| panic!("{error}")),
        rows_committed: 1,
        bytes_committed: u64::try_from(first.get_array_memory_size()).unwrap_or(u64::MAX),
        source_fingerprint: None,
    };
    sink.commit_checkpoint(&checkpoint)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.abort().await.unwrap_or_else(|error| panic!("{error}"));
    store
        .save_checkpoint(id, &checkpoint)
        .unwrap_or_else(|error| panic!("{error}"));
    progress.rows = checkpoint.rows_committed;
    progress.bytes = checkpoint.bytes_committed;
    progress.batches = 1;
    progress.state = JobState::Interrupted;
    store
        .record_progress(&progress)
        .unwrap_or_else(|error| panic!("{error}"));
    drop(store);

    let runtime = DaemonRuntime::open(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    let daemon = tokio::spawn(runtime.serve());
    let client = DaemonClient::connect(paths).unwrap_or_else(|error| panic!("{error}"));
    client
        .request(Operation::JobResume(id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let mut events = client
        .watch(Operation::JobWatch(id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let terminal = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(response) = events.recv().await {
            match response.unwrap_or_else(|error| panic!("{error}")) {
                Response::Job(job) if job.progress.state.is_terminal() => return job,
                Response::Event(progress) if progress.state.is_terminal() => {
                    return match client
                        .request(Operation::JobGet(id))
                        .await
                        .unwrap_or_else(|error| panic!("{error}"))
                    {
                        Response::Job(job) => job,
                        _ => panic!("expected a job response"),
                    };
                }
                _ => {}
            }
        }
        panic!("job watch closed before a terminal event")
    })
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(terminal.progress.state, JobState::Succeeded);
    assert_eq!(terminal.progress.rows, 3);
    assert_eq!(terminal.progress.batches, 2);
    assert_eq!(terminal.attempt, 2);
    let contents = fs::read_to_string(&output).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(contents.lines().count(), 3);
    assert_eq!(contents.matches("Alice").count(), 1);
    assert_eq!(contents.matches("Bob").count(), 1);
    assert_eq!(contents.matches("İstanbul 🌍").count(), 1);
    assert!(!staging.exists());

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
