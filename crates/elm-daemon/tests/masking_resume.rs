//! Masking has a unit-level determinism guarantee (`elm_core::masking`'s own tests: the same job
//! seed and batch sequence reproduce the same random mask) but, before this file, nothing proved
//! that guarantee holds through an actual interrupted-and-resumed daemon transfer end to end.
//! This submits one identical `JobSpec` (same id, same derived seed, same masking rule) twice:
//! once straight through, and once killed mid-transfer as a real OS process and resumed, and
//! asserts the two runs' published output is byte-for-byte identical.

use std::{
    fs,
    process::{Child, Command, Stdio},
    time::Duration,
};

use elm_core::{
    FileFormat, Identifier, JobId, JobSpec, JobState, MaskAlgorithm, MaskRule, MaskRuleId,
    SinkSpec, SourceSpec,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, RuntimePaths};
use tempfile::tempdir;

fn spawn_daemon(paths: &RuntimePaths) -> Child {
    Command::new(env!("CARGO_BIN_EXE_elm-daemon"))
        .arg("--data-directory")
        .arg(&paths.data_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn elm-daemon: {error}"))
}

async fn connect_with_retry(paths: &RuntimePaths) -> DaemonClient {
    let mut last_error = None;
    for _ in 0..100 {
        match DaemonClient::connect(paths.clone()) {
            Ok(client) => match client.request(Operation::DaemonStatus).await {
                Ok(_) => return client,
                Err(error) => last_error = Some(error.to_string()),
            },
            Err(error) => last_error = Some(error.to_string()),
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("daemon did not become ready: {last_error:?}");
}

async fn job_state(client: &DaemonClient, job_id: JobId) -> (JobState, u64) {
    match client
        .request(Operation::JobGet(job_id))
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Job(job) => (job.progress.state, job.progress.rows),
        other => panic!("expected a job response, got {other:?}"),
    }
}

async fn run_to_success(client: &DaemonClient, job_id: JobId) {
    let mut events = client
        .watch(Operation::JobWatch(job_id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let terminal = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            match events.recv().await {
                Some(Ok(Response::Job(job))) if job.progress.state.is_terminal() => {
                    return job.progress.state;
                }
                Some(Ok(Response::Event(progress))) if progress.state.is_terminal() => {
                    return progress.state;
                }
                Some(Ok(_)) => continue,
                Some(Err(error)) => panic!("{error}"),
                None => panic!("watch stream closed before a terminal event"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("job did not reach a terminal state within 60s"));
    assert_eq!(terminal, JobState::Succeeded);
}

fn build_input(directory: &std::path::Path, rows: usize) -> std::path::PathBuf {
    let padding = "x".repeat(150);
    let mut content = String::from("id,email\n");
    for row in 0..rows {
        content.push_str(&format!("{row},person{row}@example.com-{padding}\n"));
    }
    let input = directory.join("input.csv");
    fs::write(&input, content).unwrap_or_else(|error| panic!("{error}"));
    input
}

fn mask_rule() -> MaskRule {
    MaskRule {
        id: MaskRuleId::new(),
        name: "email".into(),
        column: Identifier::new("email").unwrap_or_else(|error| panic!("{error}")),
        algorithm: MaskAlgorithm::Random,
        environment_id: None,
    }
}

#[tokio::test]
async fn masked_output_is_identical_whether_or_not_the_transfer_was_interrupted_and_resumed() {
    const ROWS: usize = 60_000;

    // Run A: straight through, no interruption.
    let directory_a = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths_a = RuntimePaths::for_data_directory(directory_a.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let input_a = build_input(directory_a.path(), ROWS);
    let output_a = directory_a.path().join("output.ndjson");
    let mut spec = JobSpec::new(
        SourceSpec::File {
            path: input_a,
            format: FileFormat::Csv,
        },
        SinkSpec::File {
            path: output_a.clone(),
            format: FileFormat::Ndjson,
        },
    );
    spec.batch_target_bytes = elm_core::MIN_BATCH_TARGET_BYTES;
    spec.masks = vec![mask_rule()];
    let job_id = spec.id;

    let mut daemon_a = spawn_daemon(&paths_a);
    let client_a = connect_with_retry(&paths_a).await;
    client_a
        .request(Operation::JobSubmit(spec.clone()))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    run_to_success(&client_a, job_id).await;
    let content_a = fs::read_to_string(&output_a).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(content_a.lines().count(), ROWS);
    client_a
        .request(Operation::DaemonStop)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    tokio::time::timeout(
        Duration::from_secs(5),
        tokio::task::spawn_blocking(move || daemon_a.wait()),
    )
    .await
    .unwrap_or_else(|error| panic!("daemon did not stop in time: {error}"))
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|error| panic!("{error}"));

    // Run B: the identical spec (same job id, same derived seed, same masking rule), but this
    // time a real OS process kill mid-transfer forces a genuine resume from a checkpoint.
    let directory_b = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths_b = RuntimePaths::for_data_directory(directory_b.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let input_b = build_input(directory_b.path(), ROWS);
    let output_b = directory_b.path().join("output.ndjson");
    spec.source = SourceSpec::File {
        path: input_b,
        format: FileFormat::Csv,
    };
    spec.sink = SinkSpec::File {
        path: output_b.clone(),
        format: FileFormat::Ndjson,
    };

    let mut first_daemon = spawn_daemon(&paths_b);
    let client = connect_with_retry(&paths_b).await;
    client
        .request(Operation::JobSubmit(spec))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    for _ in 0..400 {
        let (state, rows) = job_state(&client, job_id).await;
        if state.is_terminal() || (state == JobState::Running && rows > 0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (state_before_kill, _) = job_state(&client, job_id).await;
    first_daemon
        .kill()
        .unwrap_or_else(|error| panic!("failed to kill daemon: {error}"));
    first_daemon
        .wait()
        .unwrap_or_else(|error| panic!("{error}"));

    let mut second_daemon = spawn_daemon(&paths_b);
    let client = connect_with_retry(&paths_b).await;
    if !state_before_kill.is_terminal() {
        client
            .request(Operation::JobResume(job_id))
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    }
    run_to_success(&client, job_id).await;
    let content_b = fs::read_to_string(&output_b).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(content_b.lines().count(), ROWS);

    assert_eq!(
        content_a, content_b,
        "masked output must be byte-identical whether or not the transfer was interrupted and \
         resumed (kill landed with job state {state_before_kill:?}); a mismatch means retrying a \
         checkpoint re-randomizes masked values instead of reproducing them"
    );

    client
        .request(Operation::DaemonStop)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    tokio::time::timeout(
        Duration::from_secs(5),
        tokio::task::spawn_blocking(move || second_daemon.wait()),
    )
    .await
    .unwrap_or_else(|error| panic!("daemon did not stop in time: {error}"))
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|error| panic!("{error}"));
}
