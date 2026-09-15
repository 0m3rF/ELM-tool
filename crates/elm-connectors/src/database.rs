use elm_core::DatabaseKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeFastPath {
    PostgreSqlBinaryCopy,
    MySqlLocalInfile,
    SqlServerColumnarBulk,
    OracleBatchDml,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatabaseDiagnostic {
    pub kind: DatabaseKind,
    pub compiled: bool,
    pub fast_path: NativeFastPath,
    pub external_requirement: Option<&'static str>,
    pub installation_guidance: Option<&'static str>,
}

/// Reports build/runtime prerequisites without attempting a connection or exposing credentials.
#[must_use]
pub fn connector_diagnostic(kind: DatabaseKind) -> DatabaseDiagnostic {
    match kind {
        DatabaseKind::PostgreSql => DatabaseDiagnostic {
            kind,
            compiled: cfg!(feature = "postgresql"),
            fast_path: NativeFastPath::PostgreSqlBinaryCopy,
            external_requirement: None,
            installation_guidance: None,
        },
        DatabaseKind::MySql => DatabaseDiagnostic {
            kind,
            compiled: cfg!(feature = "mysql"),
            fast_path: NativeFastPath::MySqlLocalInfile,
            external_requirement: None,
            installation_guidance: None,
        },
        DatabaseKind::SqlServer => DatabaseDiagnostic {
            kind,
            compiled: cfg!(feature = "sql-server"),
            fast_path: NativeFastPath::SqlServerColumnarBulk,
            external_requirement: Some("Microsoft ODBC Driver 18 for SQL Server"),
            installation_guidance: Some(
                "Install Microsoft ODBC Driver 18 and verify it is registered with the OS ODBC manager.",
            ),
        },
        DatabaseKind::Oracle => DatabaseDiagnostic {
            kind,
            compiled: cfg!(feature = "oracle"),
            fast_path: NativeFastPath::OracleBatchDml,
            external_requirement: Some("Oracle Instant Client"),
            installation_guidance: Some(
                "Install a supported Oracle Instant Client and expose its native libraries to the process.",
            ),
        },
    }
}
