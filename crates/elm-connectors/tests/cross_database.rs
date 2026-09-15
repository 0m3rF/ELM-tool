#![cfg(all(feature = "postgresql", feature = "mysql"))]

use arrow_array::{
    Array, BinaryArray, Decimal128Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use elm_connectors::{MySqlSink, MySqlSource, PostgresSink, PostgresSource};
use elm_core::{
    DataSink, DataSource, DatabaseKind, DatabaseSelection, Environment, EnvironmentId, Identifier,
    JobId, JobSpec, Relation, SinkSpec, SourceSpec, WriteMode,
};
use elm_engine::{NoopObserver, TransferEngine};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn environment(kind: DatabaseKind) -> Environment {
    Environment {
        id: EnvironmentId::new(),
        name: "isolated cross-database fixture".into(),
        kind,
        host: "127.0.0.1".into(),
        port: if kind == DatabaseKind::PostgreSql {
            55441
        } else {
            53341
        },
        database: "elm_matrix".into(),
        username: if kind == DatabaseKind::PostgreSql {
            "postgres"
        } else {
            "root"
        }
        .into(),
        credential_ref: "fixture-only".into(),
        options: serde_json::json!({"ssl_mode":"disable"}),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}

fn fixture_query(kind: DatabaseKind, empty: bool) -> String {
    let query = if kind == DatabaseKind::PostgreSql {
        "SELECT CAST(1 AS BIGINT) AS id, CAST('İstanbul 🌍' AS TEXT) AS label,
         decode('00ff5c','hex') AS payload, CAST(-12345.6700 AS NUMERIC(18,4)) AS amount,
         TIMESTAMP '2024-02-29 12:34:56.123456' AS occurred
         UNION ALL SELECT CAST(2 AS BIGINT), NULL, NULL, NULL, NULL"
    } else {
        "SELECT CAST(1 AS SIGNED) AS id, _utf8mb4'İstanbul 🌍' AS label,
         UNHEX('00ff5c') AS payload, CAST(-12345.6700 AS DECIMAL(18,4)) AS amount,
         CAST('2024-02-29 12:34:56.123456' AS DATETIME(6)) AS occurred
         UNION ALL SELECT CAST(2 AS SIGNED), NULL, NULL, NULL, NULL"
    };
    // PostgreSQL UNION loses NUMERIC typmods. Re-declare the bounded type at the
    // query boundary; unconstrained NUMERIC must still fail connector preflight.
    let projection = if kind == DatabaseKind::PostgreSql {
        "id,label,payload,CAST(amount AS NUMERIC(18,4)) AS amount,occurred"
    } else {
        "*"
    };
    format!(
        "SELECT {projection} FROM ({query}) AS fixture WHERE {} ORDER BY id",
        if empty { "1=0" } else { "1=1" }
    )
}

async fn source(env: &Environment, password: &str, sql: String) -> TestResult<Box<dyn DataSource>> {
    let selection = DatabaseSelection::Query { sql };
    Ok(match env.kind {
        DatabaseKind::PostgreSql => {
            Box::new(PostgresSource::connect(env, password, &selection).await?)
        }
        _ => Box::new(MySqlSource::connect(env, password, &selection).await?),
    })
}

async fn sink(
    env: &Environment,
    password: &str,
    target: Relation,
    job: JobId,
) -> TestResult<Box<dyn DataSink>> {
    Ok(match env.kind {
        DatabaseKind::PostgreSql => {
            Box::new(PostgresSink::connect(env, password, target, job, None).await?)
        }
        _ => Box::new(MySqlSink::connect(env, password, target, job, None).await?),
    })
}

async fn verify(env: &Environment, password: &str, table: &str, copies: i64) -> TestResult {
    let mut reader = source(
        env,
        password,
        format!("SELECT id,label,payload,amount,occurred FROM {table} ORDER BY id"),
    )
    .await?;
    let schema = reader.schema().await?;
    assert_eq!(
        schema.fields().len(),
        5,
        "empty replacements must retain schema"
    );
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|field| field.data_type().clone())
            .collect::<Vec<_>>(),
        vec![
            DataType::Int64,
            DataType::Utf8,
            DataType::Binary,
            DataType::Decimal128(18, 4),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        ],
        "canonical types must survive publication, including an empty replacement"
    );
    let mut rows = 0;
    while let Some(batch) = reader.next_batch(1024 * 1024).await? {
        for row in 0..batch.num_rows() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("id type changed")?;
            assert!(copies > 0, "empty replacement returned rows");
            assert_eq!(ids.value(row), (rows / copies) + 1);
            if rows < copies {
                let labels = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or("text type changed")?;
                let payload = batch
                    .column(2)
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .ok_or("binary type changed")?;
                let amount = batch
                    .column(3)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or("decimal type changed")?;
                let occurred = batch
                    .column(4)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or("timestamp type changed")?;
                assert_eq!(labels.value(row), "İstanbul 🌍");
                assert_eq!(payload.value(row), &[0, 255, 92]);
                assert_eq!(amount.value(row), -123456700);
                assert_eq!(amount.scale(), 4);
                assert_eq!(occurred.value(row), 1709210096123456);
            } else {
                for column in batch.columns().iter().skip(1) {
                    assert!(column.is_null(row));
                }
            }
            rows += 1;
        }
    }
    assert_eq!(rows, copies * 2);
    Ok(())
}

async fn exercise(from: DatabaseKind, to: DatabaseKind) -> TestResult {
    let password = std::env::var("ELM_MATRIX_PASSWORD")?;
    let from = environment(from);
    let to = environment(to);
    let table = format!("elm_matrix_{}", JobId::new().0.simple());
    let target = Relation {
        catalog: None,
        schema: None,
        name: Identifier::new(&table)?,
    };
    // REPLACE creates a new target; FAIL must leave it intact; empty REPLACE must publish an empty schema.
    for (mode, empty, copies) in [
        (WriteMode::Replace, false, 1),
        (WriteMode::Append, false, 2),
        (WriteMode::Fail, false, 2),
        (WriteMode::Replace, true, 0),
    ] {
        let query = fixture_query(from.kind, empty);
        let mut spec = JobSpec::new(
            SourceSpec::Database {
                environment_id: from.id,
                selection: DatabaseSelection::Query { sql: query.clone() },
                resume_key: vec![],
            },
            SinkSpec::Database {
                environment_id: to.id,
                relation: target.clone(),
            },
        );
        spec.write_mode = mode;
        let input = source(&from, &password, query).await?;
        let output = sink(&to, &password, target.clone(), spec.id).await?;
        let result = TransferEngine::new(spec, CancellationToken::new(), Arc::new(NoopObserver))
            .run(input, output)
            .await;
        if mode == WriteMode::Fail {
            assert!(
                matches!(result, Err(elm_core::ElmError::Conflict(_))),
                "FAIL must reject an existing target with a conflict"
            );
        } else {
            assert_eq!(result?.rows, if empty { 0 } else { 2 });
        }
        verify(&to, &password, &table, copies).await?;
    }
    // Fixture volumes are discarded by the runner, including journals and uniquely named targets.
    Ok(())
}

#[tokio::test]
#[ignore = "requires disposable PostgreSQL/MySQL matrix fixtures and ELM_MATRIX_PASSWORD"]
async fn postgres_to_mysql_preserves_typed_values_and_publication_modes() -> TestResult {
    exercise(DatabaseKind::PostgreSql, DatabaseKind::MySql).await
}

#[tokio::test]
#[ignore = "requires disposable PostgreSQL/MySQL matrix fixtures and ELM_MATRIX_PASSWORD"]
async fn mysql_to_postgres_preserves_typed_values_and_publication_modes() -> TestResult {
    exercise(DatabaseKind::MySql, DatabaseKind::PostgreSql).await
}
