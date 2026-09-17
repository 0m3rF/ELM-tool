//! Native client checks and connection diagnostics. Transfer execution is separate.

use elm_core::{DatabaseKind, ElmError, Environment, Result};
use secrecy::{ExposeSecret, SecretString};

#[cfg(feature = "sql-server")]
const SQL_SERVER_DRIVER: &str = "ODBC Driver 18 for SQL Server";

fn connection_error(database: &str) -> ElmError {
    ElmError::Connection {
        message: format!(
            "{database} connection, authentication, or certificate verification failed"
        ),
        retryable: false,
    }
}

fn tls_required(environment: &Environment) -> Result<bool> {
    match environment.options.get("ssl_mode") {
        None => Ok(true),
        Some(value) if value.as_str() == Some("require") => Ok(true),
        Some(value) if value.as_str() == Some("disable") => Ok(false),
        _ => Err(ElmError::Validation(
            "ssl_mode must be require or disable".into(),
        )),
    }
}

fn validate_endpoint(environment: &Environment, expected: DatabaseKind) -> Result<()> {
    if !environment.options.is_object() {
        return Err(ElmError::Validation(
            "Connection options must be a JSON object".into(),
        ));
    }
    if environment.kind != expected
        || environment.port == 0
        || environment.host.is_empty()
        || !environment
            .host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".-_:[]".contains(&byte))
    {
        return Err(ElmError::Validation(
            "Provide a hostname or IP address and a nonzero port for this connector".into(),
        ));
    }
    if environment.database.is_empty()
        || environment.username.is_empty()
        || environment.database.contains('\0')
        || environment.username.contains('\0')
    {
        return Err(ElmError::Validation(
            "Database and username must be nonempty and contain no NUL characters".into(),
        ));
    }
    if environment
        .options
        .get("root_certificate")
        .is_some_and(|value| value.as_str() != Some(""))
    {
        return Err(ElmError::Unsupported("Configure certificate trust through the native client; root_certificate files are currently supported only by PostgreSQL and MySQL".into()));
    }
    Ok(())
}

#[cfg(feature = "sql-server")]
pub(crate) fn sql_server_connection_string(
    environment: &Environment,
    password: &str,
) -> Result<SecretString> {
    validate_endpoint(environment, DatabaseKind::SqlServer)?;
    if password.contains('\0') {
        return Err(ElmError::Validation(
            "ODBC passwords cannot contain NUL characters".into(),
        ));
    }
    let attribute = |value: &str| format!("{{{}}}", value.replace('}', "}}"));
    Ok(format!(
        "DRIVER={};SERVER={};DATABASE={};UID={};PWD={};Encrypt={};TrustServerCertificate=no;",
        attribute(SQL_SERVER_DRIVER),
        attribute(&format!("tcp:{},{}", environment.host, environment.port)),
        attribute(&environment.database),
        attribute(&environment.username),
        attribute(password),
        if tls_required(environment)? {
            "yes"
        } else {
            "no"
        },
    )
    .into())
}

/// Checks the registered driver before attempting a bounded SQL-authenticated login.
#[cfg(feature = "sql-server")]
pub async fn test_sql_server_environment(environment: &Environment, password: &str) -> Result<()> {
    let connection_string = sql_server_connection_string(environment, password)?;
    tokio::task::spawn_blocking(move || {
        let manager = sql_server_manager()?;
        let _connection = manager
            .connect_with_connection_string(
                connection_string.expose_secret(),
                odbc_api::ConnectionOptions {
                    login_timeout_sec: Some(15),
                    ..Default::default()
                },
            )
            .map_err(|_| connection_error("SQL Server"))?;
        Ok(())
    })
    .await
    .map_err(|_| ElmError::Internal("SQL Server diagnostic worker stopped".into()))?
}

#[cfg(feature = "sql-server")]
pub(crate) fn sql_server_manager() -> Result<odbc_api::Environment> {
    let manager = odbc_api::Environment::new().map_err(|_| ElmError::Unsupported(
            "ODBC manager unavailable. Install Microsoft ODBC Driver 18; Linux/macOS also require unixODBC".into(),
        ))?;
    let drivers = manager.drivers().map_err(|_| {
        ElmError::PermissionDenied("Cannot enumerate registered ODBC drivers for this user".into())
    })?;
    if !drivers
        .iter()
        .any(|driver| driver.description == SQL_SERVER_DRIVER)
    {
        return Err(ElmError::NativeClientMissing(
            "Microsoft ODBC Driver 18 for SQL Server is not installed".into(),
        ));
    }
    Ok(manager)
}

#[cfg(feature = "oracle")]
pub(crate) fn oracle_connect_descriptor(environment: &Environment) -> Result<String> {
    validate_endpoint(environment, DatabaseKind::Oracle)?;
    if !environment
        .database
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"._-$".contains(&byte))
    {
        return Err(ElmError::Validation("Oracle database must be a service name containing letters, digits, dot, underscore, hyphen, or dollar sign".into()));
    }
    Ok(format!(
        "(DESCRIPTION=(CONNECT_TIMEOUT=15)(TRANSPORT_CONNECT_TIMEOUT=10)(RETRY_COUNT=0)(ADDRESS=(PROTOCOL={})(HOST={})(PORT={}))(CONNECT_DATA=(SERVICE_NAME={}))(SECURITY=(SSL_SERVER_DN_MATCH=YES)))",
        if tls_required(environment)? {
            "TCPS"
        } else {
            "TCP"
        },
        environment.host,
        environment.port,
        environment.database,
    ))
}

/// Loads Instant Client before connecting on a dedicated blocking worker.
#[cfg(feature = "oracle")]
pub async fn test_oracle_environment(environment: &Environment, password: &str) -> Result<()> {
    let timeout = std::time::Duration::from_millis(environment.oracle_call_timeout_ms()?);
    let descriptor = oracle_connect_descriptor(environment)?;
    let username = environment.username.clone();
    let password = SecretString::from(password.to_owned());
    tokio::task::spawn_blocking(move || {
        oracle::Version::client().map_err(|_| {
            ElmError::NativeClientMissing("Oracle Instant Client is not installed".into())
        })?;
        let connection =
            oracle::Connection::connect(&username, password.expose_secret(), &descriptor)
                .map_err(|_| connection_error("Oracle"))?;
        connection
            .set_call_timeout(Some(timeout))
            .map_err(|_| connection_error("Oracle"))?;
        connection.ping().map_err(|_| connection_error("Oracle"))
    })
    .await
    .map_err(|_| ElmError::Internal("Oracle diagnostic worker stopped".into()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment(kind: DatabaseKind) -> Environment {
        serde_json::from_value(serde_json::json!({
            "id": "00000000-0000-4000-8000-000000000001", "name": "test", "kind": kind,
            "host": "localhost", "port": 1433, "database": "test", "username": "test",
            "credential_ref": "test", "options": {},
            "created_at": "2026-09-09T00:00:00Z", "updated_at": "2026-09-09T00:00:00Z"
        }))
        .unwrap_or_else(|error| panic!("{error}"))
    }

    #[cfg(all(feature = "sql-server", feature = "oracle"))]
    #[tokio::test]
    #[ignore = "probes installed native clients and attempts loopback port 1 if present"]
    async fn native_diagnostic_errors_do_not_expose_credentials() {
        const PASSWORD: &str = "diagnostic-only-secret};injection=sentinel";
        for kind in [DatabaseKind::SqlServer, DatabaseKind::Oracle] {
            let mut environment = environment(kind);
            environment.port = 1;
            let result = if kind == DatabaseKind::SqlServer {
                test_sql_server_environment(&environment, PASSWORD).await
            } else {
                test_oracle_environment(&environment, PASSWORD).await
            };
            let error = result
                .err()
                .unwrap_or_else(|| panic!("unexpected database at loopback port 1"));
            assert!(matches!(
                error,
                ElmError::Unsupported(_)
                    | ElmError::Connection { .. }
                    | ElmError::PermissionDenied(_)
            ));
            assert!(!error.to_string().contains(PASSWORD));
            assert!(!error.to_string().contains("sentinel"));
            eprintln!("{kind}: {error}");
        }
    }

    #[cfg(feature = "sql-server")]
    #[test]
    fn odbc_attributes_cannot_override_transport_settings() {
        let mut environment = environment(DatabaseKind::SqlServer);
        environment.username = "name};Encrypt=no;UID={other".into();
        let secret = sql_server_connection_string(&environment, "pass};PWD=oops;{")
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(
            secret
                .expose_secret()
                .contains("UID={name}};Encrypt=no;UID={other};")
        );
        assert!(secret.expose_secret().contains("PWD={pass}};PWD=oops;{};"));
        assert!(
            secret
                .expose_secret()
                .ends_with("Encrypt=yes;TrustServerCertificate=no;")
        );
        environment.host = "localhost;Encrypt=no".into();
        assert!(sql_server_connection_string(&environment, "password").is_err());
    }

    #[cfg(feature = "oracle")]
    #[test]
    fn oracle_descriptor_enforces_transport_and_rejects_descriptor_injection() {
        let mut environment = environment(DatabaseKind::Oracle);
        let descriptor =
            oracle_connect_descriptor(&environment).unwrap_or_else(|error| panic!("{error}"));
        assert!(descriptor.contains("PROTOCOL=TCPS"));
        assert!(descriptor.contains("SSL_SERVER_DN_MATCH=YES"));
        environment.options = serde_json::json!({"ssl_mode": "disable"});
        assert!(
            oracle_connect_descriptor(&environment)
                .unwrap_or_else(|error| panic!("{error}"))
                .contains("PROTOCOL=TCP)")
        );
        environment.database = "service))(ADDRESS=(HOST=evil".into();
        assert!(oracle_connect_descriptor(&environment).is_err());
    }
}
