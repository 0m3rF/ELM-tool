#![cfg(target_os = "windows")]

use elm_core::{
    DatabaseKind,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, DaemonRuntime, RuntimePaths};

/// Live-tested on Windows with ODBC Driver 18 for SQL Server actually installed: confirms
/// `Operation::NativeClientStatus` reports real presence, not a placeholder, and that both
/// Oracle and SQL Server are marked provisionable on this platform. Ignored by default because
/// asserting `present` depends on this specific host's installed drivers, not just the platform.
#[tokio::test]
#[ignore = "requires ODBC Driver 18 for SQL Server to be installed on this host"]
async fn reports_real_driver_presence() {
    let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let runtime = DaemonRuntime::open(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    let _daemon = tokio::spawn(runtime.serve());
    let client = DaemonClient::connect(paths).unwrap_or_else(|error| panic!("{error}"));

    let response = client
        .request(Operation::NativeClientStatus)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    let Response::NativeClientStatuses(statuses) = response else {
        panic!("expected NativeClientStatuses, got {response:?}");
    };

    let sql_server = statuses
        .iter()
        .find(|status| status.kind == DatabaseKind::SqlServer)
        .unwrap_or_else(|| panic!("no SQL Server status reported"));
    assert!(
        sql_server.present,
        "ODBC Driver 18 for SQL Server is installed on this host; status should reflect that"
    );
    assert!(
        sql_server.provisionable,
        "SQL Server is provisionable on Windows"
    );

    let oracle = statuses
        .iter()
        .find(|status| status.kind == DatabaseKind::Oracle)
        .unwrap_or_else(|| panic!("no Oracle status reported"));
    assert!(oracle.provisionable, "Oracle is provisionable on Windows");
}
