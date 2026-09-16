//! Kills the real `elm-daemon` binary as an OS process (not an in-process task) and restarts
//! it, verifying the daemon's own reconciliation — not the engine/connector recovery boundary
//! exercised elsewhere — recovers an active job without duplicating rows.

use std::{
    fs,
    process::{Child, Command, Stdio},
    time::Duration,
};

use elm_core::{
    FileFormat, JobId, JobSpec, JobState, SinkSpec, SourceSpec,
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

#[tokio::test]
async fn daemon_process_restart_recovers_an_unclean_stop_without_duplicate_rows() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));

    let input = directory.path().join("input.csv");
    let output = directory.path().join("output.ndjson");
    let padding = "x".repeat(400);
    let mut content = String::from("id,name\n");
    const ROWS: usize = 150_000;
    for row in 1..=ROWS {
        content.push_str(&format!("{row},row-{row}-{padding}\n"));
    }
    fs::write(&input, content).unwrap_or_else(|error| panic!("{error}"));

    let mut spec = JobSpec::new(
        SourceSpec::File {
            path: input.clone(),
            format: FileFormat::Csv,
        },
        SinkSpec::File {
            path: output.clone(),
            format: FileFormat::Ndjson,
        },
    );
    spec.batch_target_bytes = elm_core::MIN_BATCH_TARGET_BYTES;
    let job_id = spec.id;

    let mut first_daemon = spawn_daemon(&paths);
    let client = connect_with_retry(&paths).await;
    client
        .request(Operation::JobSubmit(spec))
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    // Poll briefly for the job to leave `Queued` so the kill lands genuinely mid-transfer,
    // not before the daemon has even started it; then kill the process uncleanly (no
    // DaemonStop, no graceful shutdown) rather than requesting a clean stop.
    for _ in 0..200 {
        let (state, rows) = job_state(&client, job_id).await;
        if state.is_terminal() || (state == JobState::Running && rows > 0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let (state_before_kill, rows_before_kill) = job_state(&client, job_id).await;
    first_daemon
        .kill()
        .unwrap_or_else(|error| panic!("failed to kill daemon: {error}"));
    first_daemon
        .wait()
        .unwrap_or_else(|error| panic!("{error}"));

    let mut second_daemon = spawn_daemon(&paths);
    let client = connect_with_retry(&paths).await;

    let (state_after_restart, _) = job_state(&client, job_id).await;
    if state_before_kill.is_terminal() {
        // The transfer outran the kill; still a legitimate run of this test, just not the
        // interesting case. Reconciliation must still leave the job's terminal state alone.
        assert_eq!(state_after_restart, state_before_kill);
    } else {
        assert_eq!(
            state_after_restart,
            JobState::Interrupted,
            "an unclean stop must mark the active job interrupted on the next daemon startup \
             (was {state_before_kill:?} with {rows_before_kill} rows before the kill)"
        );
        client
            .request(Operation::JobResume(job_id))
            .await
            .unwrap_or_else(|error| panic!("{error}"));
    }

    let mut events = client
        .watch(Operation::JobWatch(job_id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let terminal = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            match job_state(&client, job_id).await {
                (state, _) if state.is_terminal() => return state,
                _ => {}
            }
            tokio::select! {
                response = events.recv() => {
                    if let Some(Ok(response)) = response
                        && let Response::Job(job) = response
                        && job.progress.state.is_terminal()
                    {
                        return job.progress.state;
                    }
                }
                () = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("job did not reach a terminal state in time"));
    assert_eq!(terminal, JobState::Succeeded);

    let contents = fs::read_to_string(&output).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        contents.lines().count(),
        ROWS,
        "resume after an unclean daemon process kill must publish every row exactly once, \
         with no duplicates and none missing"
    );
    for row in [1, ROWS / 2, ROWS] {
        assert!(
            contents.contains(&format!("\"id\":{row}"))
                || contents.contains(&format!("row-{row}-")),
            "expected row {row} to be present exactly once"
        );
    }

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
