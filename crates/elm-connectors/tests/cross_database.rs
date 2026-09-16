#![cfg(all(feature = "postgresql", feature = "mysql"))]

use arrow_array::{
    Array, BinaryArray, Decimal128Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use elm_connectors::{MySqlSink, MySqlSource, PostgresSink, PostgresSource, dialect_for};
#[cfg(feature = "sql-server")]
use elm_connectors::{SqlServerSource, sql_server_sink::SqlServerSink};
#[cfg(feature = "oracle")]
use elm_connectors::{oracle_sink::OracleSink, oracle_source::OracleSource};
use elm_core::{
    ConsistencyMode, DataSink, DataSource, DatabaseKind, DatabaseSelection, Environment,
    EnvironmentId, Identifier, JobId, JobSpec, Relation, SinkSpec, SourceSpec, WriteMode,
};
use elm_engine::{NoopObserver, TransferEngine};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn environment(kind: DatabaseKind) -> Environment {
    let (host, port, database, username) = match kind {
        DatabaseKind::PostgreSql => ("postgres", 55441, "elm_matrix", "postgres"),
        DatabaseKind::MySql => ("mysql", 53341, "elm_matrix", "root"),
        DatabaseKind::Oracle => ("oracle", 51541, "XEPDB1", "elm_matrix"),
        DatabaseKind::SqlServer => ("sqlserver", 51441, "master", "sa"),
    };
    let container = std::env::var("ELM_MATRIX_CONTAINER").as_deref() == Ok("1");
    let mut options = serde_json::json!({"ssl_mode":"disable"});
    if kind == DatabaseKind::Oracle {
        options["oracle_call_timeout_ms"] = 120000.into();
    }
    Environment {
        id: EnvironmentId::new(),
        name: "isolated cross-database fixture".into(),
        kind,
        host: if container { host } else { "127.0.0.1" }.into(),
        port: if container {
            match kind {
                DatabaseKind::PostgreSql => 5432,
                DatabaseKind::MySql => 3306,
                DatabaseKind::Oracle => 1521,
                DatabaseKind::SqlServer => 1433,
            }
        } else {
            port
        },
        database: database.into(),
        username: username.into(),
        credential_ref: "fixture-only".into(),
        options,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    }
}

fn fixture_query(kind: DatabaseKind, empty: bool, decimal_id: bool) -> String {
    let query = if kind == DatabaseKind::PostgreSql {
        "SELECT CAST(1 AS BIGINT) AS id, CAST('İstanbul 🌍' AS TEXT) AS label,
         decode('00ff5c','hex') AS payload, CAST(-12345.6700 AS NUMERIC(18,4)) AS amount,
         TIMESTAMP '2024-02-29 12:34:56.123456' AS occurred
         UNION ALL SELECT CAST(2 AS BIGINT), NULL, NULL, NULL, NULL"
    } else if kind == DatabaseKind::SqlServer {
        "SELECT CAST(1 AS BIGINT) AS id, CAST(N'İstanbul 🌍' AS NVARCHAR(32)) AS label,
         CAST(0x00FF5C AS VARBINARY(3)) AS payload, CAST(-12345.6700 AS DECIMAL(18,4)) AS amount,
         CAST('2024-02-29T12:34:56.123456' AS DATETIME2(6)) AS occurred
         UNION ALL SELECT CAST(2 AS BIGINT), NULL, NULL, NULL, NULL"
    } else if kind == DatabaseKind::Oracle {
        r#"SELECT CAST(1 AS NUMBER(18,0)) AS "id", CAST(N'İstanbul 🌍' AS NVARCHAR2(32)) AS "label",
         HEXTORAW('00FF5C') AS "payload", CAST(-12345.6700 AS NUMBER(18,4)) AS "amount",
         CAST(TIMESTAMP '2024-02-29 12:34:56.123456' AS TIMESTAMP(6)) AS "occurred" FROM dual
         UNION ALL SELECT CAST(2 AS NUMBER(18,0)), NULL, NULL, NULL, NULL FROM dual"#
    } else {
        "SELECT CAST(1 AS SIGNED) AS id, _utf8mb4'İstanbul 🌍' AS label,
         UNHEX('00ff5c') AS payload, CAST(-12345.6700 AS DECIMAL(18,4)) AS amount,
         CAST('2024-02-29 12:34:56.123456' AS DATETIME(6)) AS occurred
         UNION ALL SELECT CAST(2 AS SIGNED), NULL, NULL, NULL, NULL"
    };
    // PostgreSQL UNION loses NUMERIC typmods. Re-declare the bounded type at the
    // query boundary; unconstrained NUMERIC must still fail connector preflight.
    let dialect = dialect_for(kind);
    let quote = |name: &str| {
        dialect.quote_identifier(&Identifier::new(name).unwrap_or_else(|e| panic!("{e}")))
    };
    let id = quote("id");
    let amount = quote("amount");
    let numeric = if kind == DatabaseKind::Oracle {
        "NUMBER"
    } else {
        "DECIMAL"
    };
    // Oracle has no canonical Int64 mapping. Convert IDs explicitly for pairs
    // involving Oracle and retain the original integer coverage for other pairs.
    let projection = format!(
        "{}, {}, {}, CAST({amount} AS {numeric}(18,4)) AS {amount}, {}",
        if decimal_id {
            format!("CAST({id} AS {numeric}(18,0)) AS {id}")
        } else {
            id.clone()
        },
        quote("label"),
        quote("payload"),
        quote("occurred")
    );
    let alias = if kind == DatabaseKind::Oracle {
        "fixture"
    } else {
        "AS fixture"
    };
    format!(
        "SELECT {projection} FROM ({query}) {alias} WHERE {} ORDER BY {id}",
        if empty { "1=0" } else { "1=1" }
    )
}

async fn source(env: &Environment, password: &str, sql: String) -> TestResult<Box<dyn DataSource>> {
    let selection = DatabaseSelection::Query { sql };
    selected_source(env, password, &selection).await
}

async fn selected_source(
    env: &Environment,
    password: &str,
    selection: &DatabaseSelection,
) -> TestResult<Box<dyn DataSource>> {
    Ok(match env.kind {
        DatabaseKind::PostgreSql => {
            Box::new(PostgresSource::connect(env, password, selection).await?)
        }
        DatabaseKind::MySql => Box::new(MySqlSource::connect(env, password, selection).await?),
        #[cfg(feature = "oracle")]
        DatabaseKind::Oracle => Box::new(OracleSource::connect(env, password, selection).await?),
        #[cfg(feature = "sql-server")]
        DatabaseKind::SqlServer => {
            Box::new(SqlServerSource::connect(env, password, selection).await?)
        }
        #[allow(unreachable_patterns)]
        _ => return Err("matrix database feature is disabled".into()),
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
        DatabaseKind::MySql => {
            Box::new(MySqlSink::connect(env, password, target, job, None).await?)
        }
        #[cfg(feature = "oracle")]
        DatabaseKind::Oracle => {
            Box::new(OracleSink::connect(env, password, target, job, None).await?)
        }
        #[cfg(feature = "sql-server")]
        DatabaseKind::SqlServer => {
            Box::new(SqlServerSink::connect(env, password, target, job, None).await?)
        }
        #[allow(unreachable_patterns)]
        _ => return Err("matrix database feature is disabled".into()),
    })
}

async fn verify(
    env: &Environment,
    password: &str,
    target: &Relation,
    copies: i64,
    decimal_id: bool,
) -> TestResult {
    let dialect = dialect_for(env.kind);
    let columns = ["id", "label", "payload", "amount", "occurred"]
        .into_iter()
        .map(|name| Identifier::new(name).map(|name| dialect.quote_identifier(&name)))
        .collect::<elm_core::Result<Vec<_>>>()?
        .join(",");
    let table = dialect.quote_relation(target);
    let mut reader = source(
        env,
        password,
        format!("SELECT {columns} FROM {table} ORDER BY 1"),
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
            if decimal_id {
                DataType::Decimal128(18, 0)
            } else {
                DataType::Int64
            },
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
            assert!(!batch.column(0).is_null(row));
            let id = if decimal_id {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or("id type changed")?
                    .value(row)
            } else {
                i128::from(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or("id type changed")?
                        .value(row),
                )
            };
            assert!(copies > 0, "empty replacement returned rows");
            assert_eq!(id, i128::from((rows / copies) + 1));
            if rows < copies {
                for column in batch.columns() {
                    assert!(!column.is_null(row));
                }
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

async fn exercise(from: DatabaseKind, to: DatabaseKind, initial_mode: WriteMode) -> TestResult {
    let password = std::env::var("ELM_MATRIX_PASSWORD")?;
    let decimal_id = from == DatabaseKind::Oracle || to == DatabaseKind::Oracle;
    let from = environment(from);
    let to = environment(to);
    let table = format!("elm_matrix_{}", JobId::new().0.simple());
    let target = Relation {
        catalog: None,
        schema: None,
        name: Identifier::new(&table)?,
    };
    // Exercise both new-target creation and existing-target publication.
    for (step, (mode, empty, copies)) in [
        (initial_mode, false, 1),
        (WriteMode::Append, false, 2),
        (WriteMode::Fail, false, 2),
        (WriteMode::Replace, false, 1),
        (WriteMode::Replace, true, 0),
    ]
    .into_iter()
    .enumerate()
    {
        eprintln!(
            "{:?} -> {:?}: initial={initial_mode:?}, step={step}, mode={mode:?}, empty={empty}",
            from.kind, to.kind
        );
        let query = fixture_query(from.kind, empty, decimal_id);
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
        if to.kind == DatabaseKind::Oracle && mode == WriteMode::Replace && step > 0 {
            // Default atomic REPLACE must fail closed and preserve the old table.
            let mut input = source(&from, &password, query.clone()).await?;
            let schema = input.schema().await?;
            let mut output = sink(&to, &password, target.clone(), spec.id).await?;
            assert!(matches!(
                output
                    .preflight(schema.clone(), mode, ConsistencyMode::Atomic)
                    .await,
                Err(elm_core::ElmError::Unsupported(_))
            ));
            let report = output
                .preflight(schema, mode, ConsistencyMode::TableSwap)
                .await?;
            assert!(!report.capabilities.atomic_replace);
            assert!(report.capabilities.non_atomic_replace);
            assert!(!report.warnings.is_empty());
            drop(output);
            drop(input);
            verify(
                &to,
                &password,
                &target,
                if empty { 1 } else { 2 },
                decimal_id,
            )
            .await?;
            spec.consistency = ConsistencyMode::TableSwap;
        }
        let input = source(&from, &password, query).await?;
        let output = sink(&to, &password, target.clone(), spec.id).await?;
        let result = TransferEngine::new(spec, CancellationToken::new(), Arc::new(NoopObserver))
            .run(input, output)
            .await;
        if mode == WriteMode::Fail && step > 0 {
            assert!(
                matches!(result, Err(elm_core::ElmError::Conflict(_))),
                "FAIL must reject an existing target with a conflict"
            );
        } else {
            assert_eq!(result?.rows, if empty { 0 } else { 2 });
        }
        verify(&to, &password, &target, copies, decimal_id).await?;
    }
    // Fixture volumes are discarded by the runner, including journals and uniquely named targets.
    Ok(())
}

// Read the public target after each committed staging batch. A self-copy must
// finish reading its original rows before publishing any replacement or append.
struct UnchangedTarget {
    environment: Environment,
    password: String,
    target: Relation,
    copies: i64,
}

struct FailCheckpoint;

#[async_trait::async_trait]
impl elm_engine::EngineObserver for FailCheckpoint {
    async fn progress(&self, _: elm_core::JobProgress) -> elm_core::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self, _: &elm_core::BatchCheckpoint) -> elm_core::Result<()> {
        Err(elm_core::ElmError::State(
            "injected durable checkpoint failure".into(),
        ))
    }
}

#[async_trait::async_trait]
impl elm_engine::EngineObserver for UnchangedTarget {
    async fn progress(&self, _: elm_core::JobProgress) -> elm_core::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self, _: &elm_core::BatchCheckpoint) -> elm_core::Result<()> {
        verify(
            &self.environment,
            &self.password,
            &self.target,
            self.copies,
            self.environment.kind == DatabaseKind::Oracle,
        )
        .await
        .map_err(|error| elm_core::ElmError::State(error.to_string()))
    }
}

async fn same_table(kind: DatabaseKind) -> TestResult {
    let password = std::env::var("ELM_MATRIX_PASSWORD")?;
    let destination = environment(kind);
    let decimal_id = kind == DatabaseKind::Oracle;
    for alias in ["direct", "qualified", "query"] {
        let target = Relation {
            catalog: None,
            schema: None,
            name: Identifier::new(format!("elm_self_{}", JobId::new().0.simple()))?,
        };
        let mut from = destination.clone();
        if alias != "direct" {
            // Separate saved environments can refer to the same physical table.
            from.id = EnvironmentId::new();
            from.name = "second fixture environment".into();
        }
        let mut qualified = target.clone();
        qualified.schema = Some(Identifier::new(match kind {
            DatabaseKind::PostgreSql => "public",
            DatabaseKind::MySql => "elm_matrix",
            DatabaseKind::SqlServer => "dbo",
            DatabaseKind::Oracle => "ELM_MATRIX",
        })?);
        if kind == DatabaseKind::SqlServer {
            qualified.catalog = Some(Identifier::new("master")?);
        }
        let quoted = dialect_for(kind).quote_relation(&qualified);
        let table_selection = match alias {
            "direct" => DatabaseSelection::Table {
                relation: target.clone(),
            },
            "qualified" => DatabaseSelection::Table {
                relation: qualified,
            },
            _ => DatabaseSelection::Query {
                sql: format!("SELECT original.* FROM {quoted} original"),
            },
        };
        let mut before = 0;
        for (step, (mode, expected)) in [
            (WriteMode::Fail, 1),    // Seed a new table.
            (WriteMode::Replace, 1), // Read and replace that same table.
            (WriteMode::Append, 2),  // Exactly double it, without feeding back new rows.
            (WriteMode::Fail, 2),    // Preserve the original on conflict.
            (WriteMode::Replace, 2), // Preserve all four rows.
            (WriteMode::Replace, 0), // Empty selection must still replace.
        ]
        .into_iter()
        .enumerate()
        {
            eprintln!("{kind:?}: same-table {alias}, step={step}, mode={mode:?}");
            let selection = if step == 0 {
                DatabaseSelection::Query {
                    sql: fixture_query(kind, false, decimal_id),
                }
            } else if step == 5 {
                DatabaseSelection::Query {
                    sql: format!("SELECT original.* FROM {quoted} original WHERE 1=0"),
                }
            } else {
                table_selection.clone()
            };
            let mut spec = JobSpec::new(
                SourceSpec::Database {
                    environment_id: from.id,
                    selection: selection.clone(),
                    resume_key: vec![],
                },
                SinkSpec::Database {
                    environment_id: destination.id,
                    relation: target.clone(),
                },
            );
            spec.write_mode = mode;
            if kind == DatabaseKind::Oracle && mode == WriteMode::Replace {
                spec.consistency = ConsistencyMode::TableSwap;
            }
            let input = selected_source(&from, &password, &selection).await?;
            let output = sink(&destination, &password, target.clone(), spec.id).await?;
            let observer: Arc<dyn elm_engine::EngineObserver> = if step == 0 {
                Arc::new(NoopObserver)
            } else {
                Arc::new(UnchangedTarget {
                    environment: destination.clone(),
                    password: password.clone(),
                    target: target.clone(),
                    copies: before,
                })
            };
            let cancellation = CancellationToken::new();
            let deadline = cancellation.clone();
            // Cancel cooperatively so native workers and private stages can close.
            let watchdog = tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                deadline.cancel();
            });
            let result = TransferEngine::new(spec, cancellation, observer)
                .run(input, output)
                .await;
            watchdog.abort();
            if step == 3 {
                assert!(matches!(result, Err(elm_core::ElmError::Conflict(_))));
            } else {
                assert_eq!(
                    result?.rows,
                    if step == 0 {
                        2
                    } else if step == 5 {
                        0
                    } else {
                        before as u64 * 2
                    }
                );
            }
            verify(&destination, &password, &target, expected, decimal_id).await?;
            before = expected;
        }
    }
    Ok(())
}

async fn interrupted_same_table(kind: DatabaseKind) -> TestResult {
    let password = std::env::var("ELM_MATRIX_PASSWORD")?;
    let environment = environment(kind);
    let decimal_id = kind == DatabaseKind::Oracle;
    for mode in [WriteMode::Append, WriteMode::Replace] {
        let target = Relation {
            catalog: None,
            schema: None,
            name: Identifier::new(format!("elm_resume_{}", JobId::new().0.simple()))?,
        };

        // Seed the table through the same sink implementation used by the test.
        let seed_query = fixture_query(kind, false, decimal_id);
        let mut seed_spec = JobSpec::new(
            SourceSpec::Database {
                environment_id: environment.id,
                selection: DatabaseSelection::Query {
                    sql: seed_query.clone(),
                },
                resume_key: vec![],
            },
            SinkSpec::Database {
                environment_id: environment.id,
                relation: target.clone(),
            },
        );
        seed_spec.write_mode = WriteMode::Fail;
        let seed_input = source(&environment, &password, seed_query).await?;
        let seed_output = sink(&environment, &password, target.clone(), seed_spec.id).await?;
        assert_eq!(
            TransferEngine::new(seed_spec, CancellationToken::new(), Arc::new(NoopObserver),)
                .run(seed_input, seed_output)
                .await?
                .rows,
            2
        );

        let selection = DatabaseSelection::Table {
            relation: target.clone(),
        };
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
        spec.write_mode = mode;
        if kind == DatabaseKind::Oracle && mode == WriteMode::Replace {
            spec.consistency = ConsistencyMode::TableSwap;
        }

        // The sink commits its private staging batch before the observer persists
        // the checkpoint. This models daemon termination at the recovery boundary.
        let first_input = selected_source(&environment, &password, &selection).await?;
        let first_output = sink(&environment, &password, target.clone(), spec.id).await?;
        let first = TransferEngine::new(
            spec.clone(),
            CancellationToken::new(),
            Arc::new(FailCheckpoint),
        )
        .run(first_input, first_output)
        .await;
        assert!(matches!(first, Err(elm_core::ElmError::State(_))));
        verify(&environment, &password, &target, 1, decimal_id).await?;

        // Database sources without resume keys intentionally restart from zero.
        // The same job identity makes the sink trim or recreate its private stage.
        let resumed_input = selected_source(&environment, &password, &selection).await?;
        let resumed_output = sink(&environment, &password, target.clone(), spec.id).await?;
        let resumed = TransferEngine::new(
            spec,
            CancellationToken::new(),
            Arc::new(UnchangedTarget {
                environment: environment.clone(),
                password: password.clone(),
                target: target.clone(),
                copies: 1,
            }),
        )
        .run(resumed_input, resumed_output)
        .await?;
        assert_eq!(resumed.rows, 2);
        verify(
            &environment,
            &password,
            &target,
            if mode == WriteMode::Append { 2 } else { 1 },
            decimal_id,
        )
        .await?;
    }
    Ok(())
}

macro_rules! same_table_test {
    ($name:ident, $kind:ident) => {
        #[tokio::test]
        #[ignore = "requires disposable matrix fixtures, native drivers, and ELM_MATRIX_PASSWORD"]
        async fn $name() -> TestResult {
            same_table(DatabaseKind::$kind).await
        }
    };
}
same_table_test!(same_table_postgres, PostgreSql);
same_table_test!(same_table_mysql, MySql);
#[cfg(feature = "oracle")]
same_table_test!(same_table_oracle, Oracle);
#[cfg(feature = "sql-server")]
same_table_test!(same_table_sql_server, SqlServer);

macro_rules! interrupted_same_table_test {
    ($name:ident, $kind:ident) => {
        #[tokio::test]
        #[ignore = "requires disposable matrix fixtures, native drivers, and ELM_MATRIX_PASSWORD"]
        async fn $name() -> TestResult {
            interrupted_same_table(DatabaseKind::$kind).await
        }
    };
}
interrupted_same_table_test!(interrupted_same_table_postgres, PostgreSql);
interrupted_same_table_test!(interrupted_same_table_mysql, MySql);
#[cfg(feature = "oracle")]
interrupted_same_table_test!(interrupted_same_table_oracle, Oracle);
#[cfg(feature = "sql-server")]
interrupted_same_table_test!(interrupted_same_table_sql_server, SqlServer);

macro_rules! matrix_pair {
    ($name:ident, $from:ident, $to:ident) => {
        #[tokio::test]
        #[ignore = "requires disposable matrix fixtures, native drivers, and ELM_MATRIX_PASSWORD"]
        async fn $name() -> TestResult {
            exercise(DatabaseKind::$from, DatabaseKind::$to, WriteMode::Replace).await?;
            exercise(DatabaseKind::$from, DatabaseKind::$to, WriteMode::Fail).await
        }
    };
}
matrix_pair!(postgres_to_postgres, PostgreSql, PostgreSql);
matrix_pair!(postgres_to_mysql, PostgreSql, MySql);
#[cfg(feature = "oracle")]
matrix_pair!(postgres_to_oracle, PostgreSql, Oracle);
#[cfg(feature = "sql-server")]
matrix_pair!(postgres_to_sql_server, PostgreSql, SqlServer);
matrix_pair!(mysql_to_postgres, MySql, PostgreSql);
matrix_pair!(mysql_to_mysql, MySql, MySql);
#[cfg(feature = "oracle")]
matrix_pair!(mysql_to_oracle, MySql, Oracle);
#[cfg(feature = "sql-server")]
matrix_pair!(mysql_to_sql_server, MySql, SqlServer);
#[cfg(feature = "oracle")]
matrix_pair!(oracle_to_postgres, Oracle, PostgreSql);
#[cfg(feature = "oracle")]
matrix_pair!(oracle_to_mysql, Oracle, MySql);
#[cfg(feature = "oracle")]
matrix_pair!(oracle_to_oracle, Oracle, Oracle);
#[cfg(all(feature = "oracle", feature = "sql-server"))]
matrix_pair!(oracle_to_sql_server, Oracle, SqlServer);
#[cfg(feature = "sql-server")]
matrix_pair!(sql_server_to_postgres, SqlServer, PostgreSql);
#[cfg(feature = "sql-server")]
matrix_pair!(sql_server_to_mysql, SqlServer, MySql);
#[cfg(all(feature = "oracle", feature = "sql-server"))]
matrix_pair!(sql_server_to_oracle, SqlServer, Oracle);
#[cfg(feature = "sql-server")]
matrix_pair!(sql_server_to_sql_server, SqlServer, SqlServer);
