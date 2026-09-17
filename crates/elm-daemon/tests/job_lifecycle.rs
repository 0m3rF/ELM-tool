//! Exercises the desktop app's job-control surface (cancel, delete, reconnect, history) against
//! the real `elm-daemon` binary as a genuine OS process, using a fresh `DaemonClient` connection
//! for each step to stand in for the desktop window being closed and reopened. An in-process
//! `tokio::spawn(runtime.serve())` on a single-threaded test runtime cannot reliably race a real
//! transfer (the job and the cancelling client cooperatively share one thread, so the transfer
//! usually finishes before a second client can even connect); a real subprocess gives the job
//! genuine OS-level concurrency against the polling client, matching `process_restart.rs`'s
//! established pattern for exactly this class of timing concern.

use std::{
    fs,
    process::{Child, Command, Stdio},
    time::Duration,
};

use elm_core::{
    ElmError, FileFormat, JobId, JobSpec, JobState, SinkSpec, SourceSpec,
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

/// Covers, in one real run against a real daemon process: an active job rejects deletion from
/// any client; cancellation issued from one client stops a real in-flight transfer; the
/// cancelled terminal state is visible to a brand-new client afterward (a "reconnect" after the
/// desktop window closed); and a terminal job can then be deleted and disappears from history.
#[tokio::test]
async fn cancel_mid_transfer_is_observable_across_client_reconnects_and_then_deletable() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));

    let input = directory.path().join("input.csv");
    let output = directory.path().join("output.ndjson");
    let padding = "x".repeat(400);
    let mut content = String::from("id,label\n");
    const ROWS: usize = 150_000;
    for row in 0..ROWS {
        content.push_str(&format!("{row},{padding}\n"));
    }
    fs::write(&input, content).unwrap_or_else(|error| panic!("{error}"));

    let mut spec = JobSpec::new(
        SourceSpec::File {
            path: input,
            format: FileFormat::Csv,
        },
        SinkSpec::File {
            path: output.clone(),
            format: FileFormat::Ndjson,
        },
    );
    spec.batch_target_bytes = elm_core::MIN_BATCH_TARGET_BYTES;
    let id = spec.id;

    let mut daemon = spawn_daemon(&paths);

    // "Window 1" submits the job.
    let client_window1 = connect_with_retry(&paths).await;
    client_window1
        .request(Operation::JobSubmit(spec))
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    // Poll briefly so cancellation lands genuinely mid-transfer, not before the daemon has even
    // started it. If the machine is fast enough that the whole transfer completes first, that is
    // still a legitimate run: fall back to asserting the terminal-job behavior instead of the
    // interrupted-job behavior, rather than failing on hardware-dependent timing.
    let mut state_before_cancel = JobState::Queued;
    let mut rows_before_cancel = 0u64;
    for _ in 0..400 {
        let (state, rows) = job_state(&client_window1, id).await;
        state_before_cancel = state;
        rows_before_cancel = rows;
        if state.is_terminal() || (state == JobState::Running && rows > 0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    if !state_before_cancel.is_terminal() {
        // Genuinely still running: deleting it from any client must be rejected.
        let client_probe = connect_with_retry(&paths).await;
        let delete_while_active = client_probe.request(Operation::JobDelete(id)).await;
        assert!(
            matches!(delete_while_active, Err(ElmError::Conflict(_))),
            "expected a conflict deleting an active job, got {delete_while_active:?}"
        );
    }

    // The user closed "window 1" and cancels from what is effectively a new app instance.
    let client_window2 = connect_with_retry(&paths).await;
    let cancel_result = client_window2.request(Operation::JobCancel(id)).await;
    let expect_cancelled = cancel_result.is_ok();
    if !expect_cancelled {
        assert!(
            matches!(cancel_result, Err(ElmError::Conflict(_))),
            "a non-conflict error cancelling a job is unexpected: {cancel_result:?}"
        );
    }

    // A third, independent client (another "reconnect") observes the real terminal outcome.
    let client_window3 = connect_with_retry(&paths).await;
    let (terminal_state, terminal_rows) = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let (state, rows) = job_state(&client_window3, id).await;
            if state.is_terminal() {
                return (state, rows);
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("job did not reach a terminal state within 30s"));

    if expect_cancelled {
        assert_eq!(terminal_state, JobState::Cancelled);
        assert!(
            terminal_rows < ROWS as u64,
            "a cancelled job must not report every source row as transferred \
             (had {rows_before_cancel} rows when cancel was issued)"
        );
    } else {
        // The transfer outran the cancel attempt; still a legitimate, if less interesting, run.
        assert_eq!(terminal_state, JobState::Succeeded);
    }

    // The terminal state persists and is visible to yet another, freshly opened client.
    let client_window4 = connect_with_retry(&paths).await;
    let (reread_state, _) = job_state(&client_window4, id).await;
    assert_eq!(reread_state, terminal_state);

    // A terminal job can now be deleted, and disappears from job history.
    client_window4
        .request(Operation::JobDelete(id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let jobs = match client_window4
        .request(Operation::JobList)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Jobs(jobs) => jobs,
        other => panic!("expected a jobs response, got {other:?}"),
    };
    assert!(!jobs.iter().any(|job| job.spec.id == id));

    client_window4
        .request(Operation::DaemonStop)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    tokio::time::timeout(
        Duration::from_secs(5),
        tokio::task::spawn_blocking(move || daemon.wait()),
    )
    .await
    .unwrap_or_else(|error| panic!("daemon did not stop in time: {error}"))
    .unwrap_or_else(|error| panic!("{error}"))
    .unwrap_or_else(|error| panic!("{error}"));
}
