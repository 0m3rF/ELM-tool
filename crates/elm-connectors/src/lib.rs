//! Connector implementations and safe publication plans.

pub mod database;
pub mod dialect;
pub mod file;
#[cfg(feature = "mysql")]
pub mod mysql;
#[cfg(any(feature = "sql-server", feature = "oracle"))]
pub mod native_diagnostics;
#[cfg(feature = "oracle")]
pub use native_diagnostics::test_oracle_environment;
#[cfg(feature = "oracle")]
mod oracle_publication;
#[cfg(feature = "oracle")]
pub mod oracle_sink;
#[cfg(feature = "oracle")]
pub mod oracle_source;
#[cfg(feature = "sql-server")]
pub use native_diagnostics::test_sql_server_environment;
pub mod physical_identity;
#[cfg(feature = "postgresql")]
pub mod postgresql;
#[cfg(feature = "sql-server")]
pub mod sql_server;
#[cfg(feature = "sql-server")]
pub mod sql_server_sink;
#[cfg(feature = "sql-server")]
pub use sql_server::SqlServerSource;

pub use database::{DatabaseDiagnostic, NativeFastPath, connector_diagnostic};
pub use dialect::{Dialect, PublicationPlan, dialect_for};
pub use file::{
    FileSink, FileSource, FileSourceCheckpoint, RecoverableFileSink, cleanup_recoverable_staging,
    fingerprint_file,
};
#[cfg(feature = "mysql")]
pub use mysql::{
    MySqlSink, MySqlSource, cleanup_mysql_staging, mysql_published_rows, test_mysql_environment,
};
#[cfg(feature = "postgresql")]
pub use postgresql::{
    PostgresSink, PostgresSource, cleanup_postgres_staging, test_postgres_environment,
};
