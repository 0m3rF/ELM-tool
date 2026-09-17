#![cfg(feature = "postgresql")]

//! Verifies that `ssl_mode=require` performs real certificate-chain verification rather than
//! accepting any TLS handshake. Live-tested against a disposable fixture
//! (`tests/tls-postgres/Dockerfile`) whose server certificate is a self-signed leaf that is
//! never in any OS trust store, so a client that trusted it without an explicit
//! `root_certificate` would prove verification is being skipped.
//!
//! ```powershell
//! docker build -t elm-pg-tls-fixture tests/tls-postgres
//! docker run -d --rm --name elm-pg-tls -p 127.0.0.1:55443:5432 `
//!   -e POSTGRES_DB=elm_tls_test -e POSTGRES_PASSWORD=elm_tls_only elm-pg-tls-fixture
//! docker cp elm-pg-tls:/certs/ca.crt ./elm-tls-ca.crt
//! $env:ELM_TEST_TLS_CA_PATH = (Resolve-Path ./elm-tls-ca.crt)
//! cargo test -p elm-connectors --features postgresql --test tls -- --ignored
//! docker stop elm-pg-tls
//! ```

use elm_connectors::PostgresSource;
use elm_core::{DatabaseKind, DatabaseSelection, Environment, EnvironmentId};

fn environment(ssl_mode: &str, root_certificate: Option<&str>) -> Environment {
    let port = std::env::var("ELM_TEST_TLS_POSTGRES_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(55443);
    let mut options = serde_json::json!({ "ssl_mode": ssl_mode });
    if let Some(path) = root_certificate {
        options["root_certificate"] = serde_json::Value::String(path.into());
    }
    Environment {
        id: EnvironmentId::new(),
        name: "PostgreSQL TLS fixture".into(),
        kind: DatabaseKind::PostgreSql,
        host: "127.0.0.1".into(),
        port,
        database: "elm_tls_test".into(),
        username: "postgres".into(),
        credential_ref: "tls-fixture-only".into(),
        options,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}

fn password() -> String {
    std::env::var("ELM_TEST_TLS_POSTGRES_PASSWORD").unwrap_or_else(|_| "elm_tls_only".into())
}

fn ca_path() -> String {
    std::env::var("ELM_TEST_TLS_CA_PATH")
        .unwrap_or_else(|_| panic!("set ELM_TEST_TLS_CA_PATH to the fixture's extracted ca.crt"))
}

#[tokio::test]
#[ignore = "requires the disposable TLS PostgreSQL fixture"]
async fn untrusted_self_signed_certificate_is_rejected_without_an_explicit_root() {
    let selection = DatabaseSelection::Query {
        sql: "SELECT 1".into(),
    };
    let result =
        PostgresSource::connect(&environment("require", None), &password(), &selection).await;
    assert!(
        result.is_err(),
        "a self-signed certificate absent from any trust store must not be accepted implicitly; \
         if this connects, certificate verification is being skipped"
    );
}

#[tokio::test]
#[ignore = "requires the disposable TLS PostgreSQL fixture"]
async fn a_correctly_configured_root_certificate_is_trusted() {
    let path = ca_path();
    let selection = DatabaseSelection::Query {
        sql: "SELECT 1".into(),
    };
    PostgresSource::connect(
        &environment("require", Some(&path)),
        &password(),
        &selection,
    )
    .await
    .unwrap_or_else(|error| {
        panic!("connection with the correct root_certificate should succeed: {error}")
    });
}
