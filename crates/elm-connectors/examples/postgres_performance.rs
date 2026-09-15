//! Destructive only inside a fresh, dedicated elm_perf fixture; never use production credentials.
use std::{sync::Arc, time::Instant};

use elm_connectors::{PostgresSink, PostgresSource};
use elm_core::{
    DatabaseKind, DatabaseSelection, Environment, EnvironmentId, Identifier, JobSpec, Relation,
    SinkSpec, SourceSpec, WriteMode,
};
use elm_engine::{NoopObserver, TransferEngine};
use tokio_util::sync::CancellationToken;

fn relation(name: &str) -> Result<Relation, Box<dyn std::error::Error>> {
    Ok(Relation {
        catalog: None,
        schema: Some(Identifier::new("public")?),
        name: Identifier::new(name)?,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ELM_PERF_DISPOSABLE").as_deref() != Ok("yes") {
        return Err("requires ELM_PERF_DISPOSABLE=yes and a fresh isolated fixture".into());
    }
    let environment = Environment {
        id: EnvironmentId::new(),
        name: "Disposable performance fixture".into(),
        kind: DatabaseKind::PostgreSql,
        host: "127.0.0.1".into(),
        port: 55439,
        database: "elm_perf".into(),
        username: "postgres".into(),
        credential_ref: "fixture-trust-only".into(),
        options: serde_json::json!({"ssl_mode":"disable"}),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    let (client, connection) = tokio_postgres::Config::new()
        .host(&environment.host)
        .port(environment.port)
        .user(&environment.username)
        .dbname(&environment.database)
        .connect(tokio_postgres::NoTls)
        .await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    // CREATE (not DROP/overwrite) deliberately refuses a previously used fixture.
    client
        .batch_execute(
            "CREATE TABLE perf_source (id bigint PRIMARY KEY, payload text NOT NULL);
        INSERT INTO perf_source SELECT i, repeat(md5(i::text),32) FROM generate_series(1,250000) i;
        ANALYZE perf_source;",
        )
        .await?;
    let version: String = client.query_one("SELECT version()", &[]).await?.get(0);
    let mut runs = Vec::new();
    for trial in 1..=3 {
        let selection = DatabaseSelection::Table {
            relation: relation("perf_source")?,
        };
        let target = relation(&format!("perf_target_{trial}"))?;
        let mut spec = JobSpec::new(
            SourceSpec::Database {
                environment_id: environment.id,
                selection: selection.clone(),
                resume_key: vec![],
            },
            SinkSpec::Database {
                environment_id: environment.id,
                relation: target.clone(),
            },
        );
        spec.write_mode = WriteMode::Replace;
        let started = Instant::now();
        let source = PostgresSource::connect(&environment, "", &selection).await?;
        let sink = PostgresSink::connect(&environment, "", target.clone(), spec.id, None).await?;
        let stats = TransferEngine::new(spec, CancellationToken::new(), Arc::new(NoopObserver))
            .run(Box::new(source), Box::new(sink))
            .await?;
        let seconds = started.elapsed().as_secs_f64();
        // The only interpolated part is an internally generated integer, never user SQL.
        let verification = client
            .query_one(
                &format!(
                    "SELECT count(*)::bigint, count(DISTINCT id)::bigint,
            count(*) FILTER (WHERE id IS NULL OR id < 1 OR id > 250000 OR payload IS NULL
              OR payload <> repeat(md5(id::text),32))::bigint FROM perf_target_{trial}"
                ),
                &[],
            )
            .await?;
        let count: i64 = verification.get(0);
        let distinct: i64 = verification.get(1);
        let invalid: i64 = verification.get(2);
        if count != 250000 || distinct != 250000 || invalid != 0 || stats.rows != 250000 {
            return Err("performance fixture data verification failed".into());
        }
        let result = serde_json::json!({"trial":trial,"seconds":seconds,"rows":stats.rows,
            "logical_bytes":258000000_u64,"rows_per_second":stats.rows as f64 / seconds,
            "logical_mib_per_second":258000000_f64 / 1048576.0 / seconds,
            "arrow_bytes":stats.bytes,"batches":stats.batches,"verified":true});
        eprintln!("Completed trial {trial}: {seconds:.3}s, verified all rows");
        runs.push(result);
        client
            .batch_execute(&format!("DROP TABLE perf_target_{trial}"))
            .await?;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "captured_at":chrono::Utc::now(),"database_version":version,
            "profile":if cfg!(debug_assertions) {"debug"} else {"release"},
            "scope":"PostgreSQL connector + engine; no daemon/SQLite observer; atomic staged new-target REPLACE",
            "generator":"generate_series(1,250000), bigint id + repeat(md5(id::text),32) UTF8 text",
            "memory_budget_bytes":536870912_u64,"runs":runs,
            "notes":"Sequential warm-cache trials; setup and verification excluded from transfer wall time. Not a 10GB release gate or Python comparison."
        }))?
    );
    Ok(())
}
