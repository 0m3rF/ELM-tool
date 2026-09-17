use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::broadcast,
};
use tokio_util::sync::CancellationToken;

use elm_connectors::{
    FileSink, FileSource, FileSourceCheckpoint, MySqlSink, MySqlSource, PostgresSink,
    PostgresSource, RecoverableFileSink, cleanup_mysql_staging, cleanup_postgres_staging,
    cleanup_recoverable_staging, connector_diagnostic, mysql_published_rows,
    oracle_sink::{OracleSink, cleanup_oracle_staging, oracle_published_rows},
    sql_server_sink::{SqlServerSink, cleanup_sql_server_staging, sql_server_published_rows},
    test_mysql_environment, test_postgres_environment,
};
use elm_core::{
    BatchCheckpoint, ConversionPlan, DataSink, DataSource, DatabaseKind, DatabaseSelection,
    ElmError, EnvironmentId, JobId, JobProgress, JobSpec, JobState, PreviewColumn, Result,
    SinkSpec, SourceSpec, TransferPreview,
    masking::mask_text,
    protocol::{IPC_PROTOCOL_VERSION, Operation, RequestEnvelope, Response, ResponseEnvelope},
};
use elm_engine::{EngineObserver, TransferEngine};
use elm_state::{CredentialVault, KeyringVault, StateStore};
use secrecy::ExposeSecret;

use crate::{
    RuntimePaths,
    transport::{self, BoxedIo, ConnectionHandler},
};

pub struct DaemonRuntime {
    paths: RuntimePaths,
    store: StateStore,
    auth_token: String,
    shutdown: CancellationToken,
    active: Arc<Mutex<HashMap<JobId, ActiveJob>>>,
}

struct ActiveJob {
    cancellation: CancellationToken,
    events: broadcast::Sender<JobProgress>,
}

impl DaemonRuntime {
    pub fn open(paths: RuntimePaths) -> Result<Self> {
        let auth_token = paths.load_or_create_token()?;
        let store = StateStore::open(&paths.database)?;
        store.mark_active_jobs_interrupted()?;
        Ok(Self {
            paths,
            store,
            auth_token,
            shutdown: CancellationToken::new(),
            active: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub async fn serve(self) -> Result<()> {
        let runtime = Arc::new(self);
        let cloned = runtime.clone();
        let handler: ConnectionHandler = Arc::new(move |stream| {
            let runtime = cloned.clone();
            Box::pin(async move {
                if let Err(error) = runtime.handle_connection(stream).await {
                    tracing::warn!(code = ?error.code(), "IPC request failed");
                }
            })
        });
        transport::serve(&runtime.paths, runtime.shutdown.clone(), handler).await
    }

    async fn handle_connection(&self, stream: BoxedIo) -> Result<()> {
        let (reader, mut writer) = tokio::io::split(stream);
        let mut reader = BufReader::new(reader);
        let mut request_line = String::new();
        let bytes = reader.read_line(&mut request_line).await?;
        if bytes == 0 || bytes > 1024 * 1024 {
            return Err(ElmError::Validation("invalid IPC request length".into()));
        }
        let request: RequestEnvelope = serde_json::from_str(&request_line)
            .map_err(|error| ElmError::Validation(format!("invalid IPC request: {error}")))?;
        if request.version != IPC_PROTOCOL_VERSION {
            return self
                .write_response(
                    &mut writer,
                    request.request_id,
                    Err(ElmError::Validation(
                        "incompatible IPC protocol version".into(),
                    )),
                )
                .await;
        }
        if !tokens_equal(&request.auth_token, &self.auth_token) {
            return self
                .write_response(
                    &mut writer,
                    request.request_id,
                    Err(ElmError::Authentication),
                )
                .await;
        }

        if let Operation::JobWatch(job_id) = request.operation {
            return self
                .watch_job(&mut writer, request.request_id, job_id)
                .await;
        }

        let result = self.dispatch(request.operation).await;
        self.write_response(&mut writer, request.request_id, result)
            .await
    }

    async fn dispatch(&self, operation: Operation) -> Result<Response> {
        match operation {
            Operation::DaemonStatus => Ok(Response::Status {
                pid: std::process::id(),
                active_jobs: self
                    .active
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .len(),
            }),
            Operation::DaemonStop => {
                self.shutdown.cancel();
                Ok(Response::Ack)
            }
            Operation::EnvironmentAdd(environment) | Operation::EnvironmentEdit(environment) => {
                self.store.upsert_environment(&environment)?;
                Ok(Response::Environment(environment))
            }
            Operation::EnvironmentList => {
                Ok(Response::Environments(self.store.list_environments()?))
            }
            Operation::EnvironmentRemove(id) => {
                if self
                    .store
                    .list_jobs()?
                    .iter()
                    .any(|job| job_references_environment(&job.spec, id))
                {
                    return Err(ElmError::Conflict(
                        "environment is referenced by job history; delete those jobs first so staging recovery remains possible"
                            .into(),
                    ));
                }
                self.store.remove_environment(id)?;
                Ok(Response::Ack)
            }
            Operation::EnvironmentTest(id) => self.test_environment(id).await,
            Operation::MaskAdd(rule) | Operation::MaskEdit(rule) => {
                self.store.upsert_mask(&rule)?;
                Ok(Response::Masks(vec![rule]))
            }
            Operation::MaskList => Ok(Response::Masks(self.store.list_masks()?)),
            Operation::MaskRemove(id) => {
                self.store.remove_mask(id)?;
                Ok(Response::Ack)
            }
            Operation::MaskTest { rule, value } => Ok(Response::MaskedValue(mask_text(
                &value,
                &rule.algorithm,
                &[0; 32],
            ))),
            Operation::SettingsGet => Ok(Response::Settings(self.store.load_settings()?)),
            Operation::SettingsSet(settings) => {
                self.store.save_settings(settings)?;
                Ok(Response::Settings(settings))
            }
            Operation::JobPreview(spec) => self.preview_job(spec).await,
            Operation::JobSubmit(spec) => {
                let record = self.store.create_job(&spec)?;
                self.spawn_job(spec, None)?;
                Ok(Response::Job(record))
            }
            Operation::JobGet(id) => Ok(Response::Job(self.store.get_job(id)?)),
            Operation::JobList => Ok(Response::Jobs(self.store.list_jobs()?)),
            Operation::JobCancel(id) => self.cancel_job(id),
            Operation::JobResume(id) => {
                let checkpoint = self.store.load_checkpoint(id)?;
                let record = self.store.queue_retry(id)?;
                self.spawn_job(record.spec.clone(), checkpoint)?;
                Ok(Response::Job(record))
            }
            Operation::JobDelete(id) => {
                let job = self.store.get_job(id)?;
                if !job.progress.state.is_terminal() && job.progress.state != JobState::Interrupted
                {
                    return Err(ElmError::Conflict(
                        "active or queued jobs cannot be deleted".into(),
                    ));
                }
                if let SinkSpec::File {
                    path: destination,
                    format,
                } = &job.spec.sink
                {
                    let staging = self.paths.job_staging_directory(id);
                    if cleanup_recoverable_staging(
                        &staging,
                        destination,
                        *format,
                        job.spec.write_mode,
                    )? {
                        let _cleanup_record = self.store.mark_recovery_resource_cleaned(
                            id,
                            "file",
                            &staging.to_string_lossy(),
                        );
                    }
                }
                if let SinkSpec::Database {
                    environment_id,
                    relation,
                } = &job.spec.sink
                {
                    let environment = self.store.get_environment(*environment_id)?;
                    let password = KeyringVault::default().get(&environment.credential_ref)?;
                    match environment.kind {
                        DatabaseKind::PostgreSql => {
                            let _removed = cleanup_postgres_staging(
                                &environment,
                                password.expose_secret(),
                                relation,
                                id,
                            )
                            .await?;
                            let resource = database_recovery_resource(*environment_id, id);
                            let _cleanup_record = self.store.mark_recovery_resource_cleaned(
                                id,
                                "postgresql",
                                &resource,
                            );
                        }
                        DatabaseKind::MySql => {
                            let _removed = cleanup_mysql_staging(
                                &environment,
                                password.expose_secret(),
                                relation,
                                id,
                            )
                            .await?;
                            let resource = database_recovery_resource(*environment_id, id);
                            let _cleanup_record = self
                                .store
                                .mark_recovery_resource_cleaned(id, "mysql", &resource);
                        }
                        DatabaseKind::SqlServer => {
                            cleanup_sql_server_staging(
                                &environment,
                                password.expose_secret(),
                                relation.clone(),
                                id,
                            )
                            .await?;
                            let resource = database_recovery_resource(*environment_id, id);
                            let _cleanup_record = self.store.mark_recovery_resource_cleaned(
                                id,
                                "sql_server",
                                &resource,
                            );
                        }
                        DatabaseKind::Oracle => {
                            cleanup_oracle_staging(
                                &environment,
                                password.expose_secret(),
                                relation.clone(),
                                id,
                            )
                            .await?;
                            let resource = database_recovery_resource(*environment_id, id);
                            self.store
                                .mark_recovery_resource_cleaned(id, "oracle", &resource)?;
                        }
                    }
                }
                self.store.delete_job(id)?;
                Ok(Response::Ack)
            }
            Operation::JobWatch(_) => Err(ElmError::Internal(
                "job.watch must be dispatched as a streaming request".into(),
            )),
        }
    }

    async fn test_environment(&self, id: EnvironmentId) -> Result<Response> {
        let environment = self.store.get_environment(id)?;
        let diagnostic = connector_diagnostic(environment.kind);
        if !diagnostic.compiled {
            return Err(ElmError::Unsupported(format!(
                "{} connector is not compiled into this build",
                environment.kind
            )));
        }
        match environment.kind {
            DatabaseKind::PostgreSql => {
                let password = KeyringVault::default().get(&environment.credential_ref)?;
                test_postgres_environment(&environment, password.expose_secret()).await?;
                Ok(Response::Ack)
            }
            DatabaseKind::MySql => {
                let password = KeyringVault::default().get(&environment.credential_ref)?;
                test_mysql_environment(&environment, password.expose_secret()).await?;
                Ok(Response::Ack)
            }
            DatabaseKind::SqlServer | DatabaseKind::Oracle => {
                let password = KeyringVault::default().get(&environment.credential_ref)?;
                if environment.kind == DatabaseKind::SqlServer {
                    elm_connectors::test_sql_server_environment(
                        &environment,
                        password.expose_secret(),
                    )
                    .await?;
                } else {
                    elm_connectors::test_oracle_environment(&environment, password.expose_secret())
                        .await?;
                }
                Ok(Response::Ack)
            }
        }
    }

    /// Runs real preflight against the job's actual source and destination without executing
    /// the transfer: no staging is created, no rows move, and no recovery resource is recorded.
    /// Every connector's `preflight()` is a read-only privilege/schema check by contract (the
    /// same one `execute_job` runs immediately before `begin()`), so this call has no more
    /// side effects than `EnvironmentTest` does.
    async fn preview_job(&self, spec: JobSpec) -> Result<Response> {
        spec.validate()?;
        let mut source = open_preview_source(&spec, &self.store).await?;
        let schema = source.schema().await?;
        let plan = ConversionPlan::new(schema, &spec.conversions)?;
        let mut sink = open_preview_sink(&spec, &self.store).await?;
        let report = sink
            .preflight(plan.output_schema(), spec.write_mode, spec.consistency)
            .await?;
        let publication_error = report
            .require_safe_publication(spec.write_mode, spec.consistency)
            .err()
            .map(|error| error.to_public().message);
        let columns = plan
            .input_schema()
            .fields()
            .iter()
            .zip(plan.output_schema().fields())
            .map(|(input, output)| PreviewColumn {
                name: input.name().clone(),
                source_type: input.data_type().to_string(),
                output_type: output.data_type().to_string(),
                nullable: output.is_nullable(),
            })
            .collect();
        Ok(Response::Preview(TransferPreview {
            columns,
            report,
            publication_error,
        }))
    }

    fn spawn_job(&self, spec: JobSpec, checkpoint: Option<BatchCheckpoint>) -> Result<()> {
        let cancellation = CancellationToken::new();
        let (events, _) = broadcast::channel(64);
        let staging_directory = matches!(&spec.sink, SinkSpec::File { .. })
            .then(|| self.paths.job_staging_directory(spec.id));
        if let Some(directory) = &staging_directory {
            self.store
                .register_recovery_resource(spec.id, "file", &directory.to_string_lossy())?;
        }
        {
            let mut active = self
                .active
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if active.contains_key(&spec.id) {
                return Err(ElmError::Conflict(format!(
                    "job {} is already active",
                    spec.id
                )));
            }
            active.insert(
                spec.id,
                ActiveJob {
                    cancellation: cancellation.clone(),
                    events: events.clone(),
                },
            );
        }
        let store = self.store.clone();
        let active = self.active.clone();
        tokio::spawn(async move {
            let source_kind = match &spec.source {
                SourceSpec::Database { environment_id, .. } => store
                    .get_environment(*environment_id)
                    .ok()
                    .map(|environment| environment.kind),
                SourceSpec::File { .. } => None,
            };
            let source_warnings =
                database_source_warnings(&spec, checkpoint.is_some(), source_kind);
            let observer = Arc::new(StateObserver {
                store: store.clone(),
                events,
                job_id: spec.id,
                injected_warnings: source_warnings,
            });
            let cancellation_requested = cancellation.clone();
            let result = execute_job(
                spec.clone(),
                checkpoint,
                staging_directory.clone(),
                cancellation,
                observer.clone(),
                store.clone(),
            )
            .await;
            if result.is_ok()
                && let Some(directory) = &staging_directory
                && !directory.exists()
            {
                let _cleanup_record = store.mark_recovery_resource_cleaned(
                    spec.id,
                    "file",
                    &directory.to_string_lossy(),
                );
            }
            if let Err(error) = result
                && let Ok(record) = store.get_job(spec.id)
                && !record.progress.state.is_terminal()
                && record.progress.state != JobState::Interrupted
            {
                let mut progress = record.progress;
                if progress.state == JobState::Queued {
                    progress.state = JobState::Preflighting;
                    progress.occurred_at = chrono::Utc::now();
                    let _record_result = store.record_progress(&progress);
                }
                // A cancellation request can race a batch that was already mid-flight: that
                // batch's own progress write then loses to `cancel_job`'s `Cancelling` write and
                // comes back as a state-transition `Conflict`, not `ElmError::Cancelled`. Once
                // cancellation has been requested, any resulting error reflects that request,
                // not a genuine failure, so it must still resolve to `Cancelled`.
                progress.state = if matches!(error, ElmError::Cancelled)
                    || cancellation_requested.is_cancelled()
                {
                    JobState::Cancelled
                } else {
                    JobState::Failed
                };
                progress.error = Some(error.to_public());
                progress.occurred_at = chrono::Utc::now();
                let _record_result = store.record_progress(&progress);
                let _send_result = observer.events.send(progress);
            }
            active
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .remove(&spec.id);
        });
        Ok(())
    }

    fn cancel_job(&self, id: JobId) -> Result<Response> {
        let job = self.store.get_job(id)?;
        if job.progress.state.is_terminal() {
            return Err(ElmError::Conflict(
                "completed jobs cannot be cancelled".into(),
            ));
        }
        let active = self
            .active
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let active_job = active
            .get(&id)
            .ok_or_else(|| ElmError::Conflict("job is not active in this daemon".into()))?;
        // Flip the in-memory flag before persisting `Cancelling`: a batch that is mid-flight can
        // race this call and lose the store write (its own progress update conflicts with the
        // `Cancelling` state that just committed). That losing batch's error handling checks
        // this same flag to tell a genuine failure from a cancellation artifact, so the flag
        // must already be visible by the time the `Cancelling` write can possibly be observed.
        active_job.cancellation.cancel();
        let mut progress = job.progress;
        progress.state = JobState::Cancelling;
        progress.occurred_at = chrono::Utc::now();
        self.store.record_progress(&progress)?;
        let _send_result = active_job.events.send(progress);
        Ok(Response::Ack)
    }

    async fn watch_job<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        writer: &mut W,
        request_id: uuid::Uuid,
        id: JobId,
    ) -> Result<()> {
        let mut receiver = self
            .active
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&id)
            .map(|job| job.events.subscribe());
        let current = self.store.get_job(id)?;
        self.write_response(writer, request_id, Ok(Response::Job(current.clone())))
            .await?;
        if current.progress.state.is_terminal() || current.progress.state == JobState::Interrupted {
            return Ok(());
        }
        let Some(receiver) = receiver.as_mut() else {
            return Ok(());
        };
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    let terminal =
                        event.state.is_terminal() || event.state == JobState::Interrupted;
                    self.write_response(writer, request_id, Ok(Response::Event(event)))
                        .await?;
                    if terminal {
                        return Ok(());
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let current = self.store.get_job(id)?;
                    self.write_response(writer, request_id, Ok(Response::Event(current.progress)))
                        .await?;
                }
                Err(broadcast::error::RecvError::Closed) => return Ok(()),
            }
        }
    }

    async fn write_response<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        writer: &mut W,
        request_id: uuid::Uuid,
        response: Result<Response>,
    ) -> Result<()> {
        let envelope = ResponseEnvelope {
            version: IPC_PROTOCOL_VERSION,
            request_id,
            result: response.map_err(|error| error.to_public()),
        };
        let mut payload = serde_json::to_vec(&envelope)
            .map_err(|error| ElmError::Internal(format!("cannot encode IPC response: {error}")))?;
        payload.push(b'\n');
        writer.write_all(&payload).await?;
        writer.flush().await?;
        Ok(())
    }
}

/// Opens a fresh (non-resuming) source connector for preview purposes only. Mirrors the
/// `SourceSpec::Database`/`SourceSpec::File` construction in `execute_job`, minus checkpoint
/// resume: a preview always reads the source's current state, never a saved resume position.
async fn open_preview_source(spec: &JobSpec, store: &StateStore) -> Result<Box<dyn DataSource>> {
    Ok(match &spec.source {
        SourceSpec::File {
            path: source_path,
            format: source_format,
        } => Box::new(FileSource::open(source_path, *source_format)?),
        SourceSpec::Database {
            environment_id,
            selection,
            resume_key,
        } => {
            let environment = store.get_environment(*environment_id)?;
            validate_source_resume(environment.kind, !resume_key.is_empty())?;
            let password = KeyringVault::default().get(&environment.credential_ref)?;
            match environment.kind {
                DatabaseKind::PostgreSql => Box::new(
                    PostgresSource::connect(&environment, password.expose_secret(), selection)
                        .await?,
                ),
                DatabaseKind::MySql => Box::new(
                    MySqlSource::connect(&environment, password.expose_secret(), selection).await?,
                ),
                DatabaseKind::SqlServer => Box::new(
                    elm_connectors::SqlServerSource::connect(
                        &environment,
                        password.expose_secret(),
                        selection,
                    )
                    .await?,
                ),
                DatabaseKind::Oracle => Box::new(
                    elm_connectors::oracle_source::OracleSource::connect(
                        &environment,
                        password.expose_secret(),
                        selection,
                    )
                    .await?,
                ),
            }
        }
    })
}

/// Opens a fresh sink connector for preview purposes only: connects and, for database sinks,
/// computes staging/backup names, but never registers a recovery resource and is never
/// `begin()`-ed, so no staging DDL runs and nothing needs to be cleaned up afterward.
async fn open_preview_sink(spec: &JobSpec, store: &StateStore) -> Result<Box<dyn DataSink>> {
    Ok(match &spec.sink {
        SinkSpec::File {
            path: sink_path,
            format: sink_format,
        } => Box::new(FileSink::new(sink_path, *sink_format)),
        SinkSpec::Database {
            environment_id,
            relation,
        } => {
            let environment = store.get_environment(*environment_id)?;
            let password = KeyringVault::default().get(&environment.credential_ref)?;
            match environment.kind {
                DatabaseKind::PostgreSql => Box::new(
                    PostgresSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        None,
                    )
                    .await?,
                ),
                DatabaseKind::MySql => Box::new(
                    MySqlSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        None,
                    )
                    .await?,
                ),
                DatabaseKind::SqlServer => Box::new(
                    SqlServerSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        None,
                    )
                    .await?,
                ),
                DatabaseKind::Oracle => Box::new(
                    OracleSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        None,
                    )
                    .await?,
                ),
            }
        }
    })
}

async fn execute_job(
    spec: JobSpec,
    checkpoint: Option<BatchCheckpoint>,
    staging_directory: Option<std::path::PathBuf>,
    cancellation: CancellationToken,
    observer: Arc<StateObserver>,
    store: StateStore,
) -> Result<()> {
    // Reconcile publication before opening the source: it may be unavailable or
    // changed, and a confirmed commit must never trigger another transfer.
    if let SinkSpec::Database {
        environment_id,
        relation,
    } = &spec.sink
    {
        let environment = store.get_environment(*environment_id)?;
        if matches!(
            environment.kind,
            DatabaseKind::MySql | DatabaseKind::SqlServer | DatabaseKind::Oracle
        ) {
            let password = KeyringVault::default().get(&environment.credential_ref)?;
            let published = match environment.kind {
                DatabaseKind::MySql => {
                    mysql_published_rows(&environment, password.expose_secret(), relation, spec.id)
                        .await?
                }
                DatabaseKind::SqlServer => {
                    sql_server_published_rows(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                    )
                    .await?
                }
                DatabaseKind::Oracle => {
                    oracle_published_rows(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                    )
                    .await?
                }
                _ => None,
            };
            if let Some(rows) = published {
                return observer.reconcile_publication(rows, environment.kind).await;
            }
        }
    }
    // Database sources without a configured keyset restart their private staging load from zero.
    // This never exposes partial target data or pretends that a row count is a snapshot boundary.
    let effective_checkpoint = source_resume_checkpoint(&spec.source, checkpoint);
    let source: Box<dyn DataSource> = match &spec.source {
        SourceSpec::File {
            path: source_path,
            format: source_format,
        } => {
            let source = match &effective_checkpoint {
                Some(checkpoint) => {
                    let file_checkpoint: FileSourceCheckpoint = serde_json::from_value(
                        checkpoint.source_position.clone(),
                    )
                    .map_err(|error| {
                        ElmError::State(format!(
                            "invalid file source checkpoint for resume: {error}"
                        ))
                    })?;
                    if file_checkpoint.logical_row_offset != checkpoint.rows_committed {
                        return Err(ElmError::State(
                            "file source checkpoint row count does not match job progress".into(),
                        ));
                    }
                    FileSource::resume(source_path, *source_format, &file_checkpoint)?
                }
                None => FileSource::open(source_path, *source_format)?,
            };
            Box::new(source)
        }
        SourceSpec::Database {
            environment_id,
            selection,
            resume_key,
        } => {
            let environment = store.get_environment(*environment_id)?;
            validate_source_resume(environment.kind, !resume_key.is_empty())?;
            let password = KeyringVault::default().get(&environment.credential_ref)?;
            match environment.kind {
                DatabaseKind::PostgreSql => Box::new(
                    PostgresSource::connect_resumable(
                        &environment,
                        password.expose_secret(),
                        selection,
                        resume_key,
                        effective_checkpoint
                            .as_ref()
                            .map(|checkpoint| &checkpoint.source_position),
                    )
                    .await?,
                ),
                DatabaseKind::MySql => Box::new(
                    MySqlSource::connect(&environment, password.expose_secret(), selection).await?,
                ),
                DatabaseKind::SqlServer => Box::new(
                    elm_connectors::SqlServerSource::connect(
                        &environment,
                        password.expose_secret(),
                        selection,
                    )
                    .await?,
                ),
                DatabaseKind::Oracle => Box::new(
                    elm_connectors::oracle_source::OracleSource::connect(
                        &environment,
                        password.expose_secret(),
                        selection,
                    )
                    .await?,
                ),
            }
        }
    };

    let mut database_resource = None;
    let sink: Box<dyn DataSink> = match &spec.sink {
        SinkSpec::File {
            path: sink_path,
            format: sink_format,
        } => {
            let staging_directory = staging_directory.ok_or_else(|| {
                ElmError::Internal("file destination has no durable staging directory".into())
            })?;
            Box::new(RecoverableFileSink::new(
                sink_path,
                *sink_format,
                staging_directory,
                effective_checkpoint.as_ref(),
            )?)
        }
        SinkSpec::Database {
            environment_id,
            relation,
        } => {
            let environment = store.get_environment(*environment_id)?;
            let password = KeyringVault::default().get(&environment.credential_ref)?;
            let resource = database_recovery_resource(*environment_id, spec.id);
            match environment.kind {
                DatabaseKind::PostgreSql => {
                    let sink = PostgresSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        effective_checkpoint.as_ref(),
                    )
                    .await?;
                    store.register_recovery_resource(spec.id, "postgresql", &resource)?;
                    database_resource = Some(("postgresql", resource));
                    Box::new(sink)
                }
                DatabaseKind::MySql => {
                    let sink = MySqlSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        effective_checkpoint.as_ref(),
                    )
                    .await?;
                    store.register_recovery_resource(spec.id, "mysql", &resource)?;
                    database_resource = Some(("mysql", resource));
                    Box::new(sink)
                }
                DatabaseKind::SqlServer => {
                    let sink = SqlServerSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        effective_checkpoint.as_ref(),
                    )
                    .await?;
                    store.register_recovery_resource(spec.id, "sql_server", &resource)?;
                    database_resource = Some(("sql_server", resource));
                    Box::new(sink)
                }
                DatabaseKind::Oracle => {
                    store.register_recovery_resource(spec.id, "oracle", &resource)?;
                    let sink = OracleSink::connect(
                        &environment,
                        password.expose_secret(),
                        relation.clone(),
                        spec.id,
                        effective_checkpoint.as_ref(),
                    )
                    .await?;
                    database_resource = Some(("oracle", resource));
                    Box::new(sink)
                }
            }
        }
    };
    let extra_warnings = physical_identity_warnings(&store, &spec).await;
    let job_id = spec.id;
    let mut engine =
        TransferEngine::new(spec, cancellation, observer).with_extra_warnings(extra_warnings);
    if let Some(checkpoint) = effective_checkpoint {
        engine = engine.with_resume_checkpoint(checkpoint);
    }
    engine.run(source, sink).await?;
    if let Some((resource_kind, resource)) = database_resource {
        // Publication receipts remain until job deletion, including after success.
        if !matches!(resource_kind, "mysql" | "sql_server" | "oracle") {
            store.mark_recovery_resource_cleaned(job_id, resource_kind, &resource)?;
        }
    }
    Ok(())
}

/// Best-effort diagnostic: warns when a job's source and destination resolve to the same
/// physical table (directly, through a view/synonym, or through a second environment record
/// for the same physical server). Never fails the job; any lookup error is treated the same as
/// "identity unknown" and produces no warning.
async fn physical_identity_warnings(store: &StateStore, spec: &JobSpec) -> Vec<String> {
    let SourceSpec::Database {
        environment_id: source_environment_id,
        selection: DatabaseSelection::Table {
            relation: source_relation,
        },
        ..
    } = &spec.source
    else {
        return Vec::new();
    };
    let SinkSpec::Database {
        environment_id: sink_environment_id,
        relation: sink_relation,
    } = &spec.sink
    else {
        return Vec::new();
    };
    let identities = async {
        let source_environment = store.get_environment(*source_environment_id)?;
        let source_password = KeyringVault::default().get(&source_environment.credential_ref)?;
        let source_identity = elm_connectors::physical_identity::resolve(
            &source_environment,
            source_password.expose_secret(),
            source_relation,
        )
        .await?;
        let sink_environment = store.get_environment(*sink_environment_id)?;
        let sink_password = KeyringVault::default().get(&sink_environment.credential_ref)?;
        let sink_identity = elm_connectors::physical_identity::resolve(
            &sink_environment,
            sink_password.expose_secret(),
            sink_relation,
        )
        .await?;
        Result::Ok((source_identity, sink_identity))
    }
    .await;
    let Ok((Some(source_identity), Some(sink_identity))) = identities else {
        return Vec::new();
    };
    if source_identity.same_physical_table(&sink_identity) {
        vec![elm_connectors::physical_identity::same_table_warning(
            source_relation,
            sink_relation,
        )]
    } else {
        Vec::new()
    }
}

fn database_recovery_resource(environment_id: EnvironmentId, job_id: JobId) -> String {
    format!("environment:{environment_id};job:{job_id}")
}

fn job_references_environment(spec: &JobSpec, environment_id: EnvironmentId) -> bool {
    matches!(
        &spec.source,
        SourceSpec::Database {
            environment_id: id,
            ..
        } if *id == environment_id
    ) || matches!(
        &spec.sink,
        SinkSpec::Database {
            environment_id: id,
            ..
        } if *id == environment_id
    )
}

fn validate_source_resume(kind: DatabaseKind, has_resume_key: bool) -> Result<()> {
    if has_resume_key && kind != DatabaseKind::PostgreSql {
        return Err(ElmError::Unsupported(format!(
            "{kind} keyset resume is not implemented; omit resume keys to restart the private staging load from zero"
        )));
    }
    Ok(())
}

fn source_resume_checkpoint(
    source: &SourceSpec,
    checkpoint: Option<BatchCheckpoint>,
) -> Option<BatchCheckpoint> {
    match source {
        SourceSpec::Database { resume_key, .. } if resume_key.is_empty() => None,
        _ => checkpoint,
    }
}

fn database_source_warnings(
    spec: &JobSpec,
    is_resume: bool,
    kind: Option<DatabaseKind>,
) -> Vec<String> {
    let mut warnings = match &spec.source {
        SourceSpec::Database { resume_key, .. } if !resume_key.is_empty() => vec![
            "Database resume is key-bounded, not a recoverable snapshot; keep the source stable for exact point-in-time results"
                .into(),
        ],
        SourceSpec::Database { .. } if is_resume => vec![
            "This database source has no resume key; its private staging load restarted from zero"
                .into(),
        ],
        _ => Vec::new(),
    };
    if matches!(spec.source, SourceSpec::Database { .. }) && kind == Some(DatabaseKind::Oracle) {
        warnings.push("Experimental Oracle source uses byte-bounded native array fetching; throughput is not performance-qualified. Oracle DATE preserves time of day as a timezone-free timestamp".into());
    }
    warnings
}

struct StateObserver {
    store: StateStore,
    events: broadcast::Sender<JobProgress>,
    job_id: JobId,
    injected_warnings: Vec<String>,
}

impl StateObserver {
    async fn reconcile_publication(&self, rows: u64, database: DatabaseKind) -> Result<()> {
        let mut progress = self.store.get_job(self.job_id)?.progress;
        progress.rows = rows;
        progress.error = None;
        progress.eta = None;
        progress.rows_per_second = 0.0;
        progress.bytes_per_second = 0.0;
        progress.warnings.push(format!(
            "Recovered a confirmed {database} publication; no source rows were read or written again"
        ));
        for state in [
            JobState::Preflighting,
            JobState::Running,
            JobState::Publishing,
            JobState::Succeeded,
        ] {
            progress.state = state;
            progress.occurred_at = chrono::Utc::now();
            self.store.record_progress(&progress)?;
            let _sent = self.events.send(progress.clone());
        }
        Ok(())
    }
}

#[async_trait]
impl EngineObserver for StateObserver {
    async fn progress(&self, mut event: JobProgress) -> Result<()> {
        for warning in &self.injected_warnings {
            if !event.warnings.contains(warning) {
                event.warnings.push(warning.clone());
            }
        }
        self.store.record_progress(&event)?;
        let _send_result = self.events.send(event);
        Ok(())
    }

    async fn checkpoint(&self, checkpoint: &BatchCheckpoint) -> Result<()> {
        self.store.save_checkpoint(self.job_id, checkpoint)
    }
}

fn tokens_equal(left: &str, right: &str) -> bool {
    let left = blake3::hash(left.as_bytes());
    let right = blake3::hash(right.as_bytes());
    constant_time_eq::constant_time_eq(left.as_bytes(), right.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use elm_core::FileFormat;

    #[test]
    fn oracle_source_warns_and_cannot_reuse_a_row_count_as_a_resume_boundary() {
        let source = SourceSpec::Database {
            environment_id: EnvironmentId(uuid::Uuid::new_v4()),
            selection: elm_core::DatabaseSelection::Query {
                sql: "SELECT 1 FROM dual".into(),
            },
            resume_key: Vec::new(),
        };
        let checkpoint = BatchCheckpoint {
            sequence: 3,
            source_position: serde_json::json!({"row": 100}),
            rows_committed: 100,
            bytes_committed: 4096,
            source_fingerprint: None,
        };
        assert!(source_resume_checkpoint(&source, Some(checkpoint.clone())).is_none());
        assert!(validate_source_resume(DatabaseKind::Oracle, false).is_ok());
        assert!(matches!(
            validate_source_resume(DatabaseKind::Oracle, true),
            Err(ElmError::Unsupported(_))
        ));
        let spec = JobSpec::new(
            source,
            SinkSpec::File {
                path: "output.parquet".into(),
                format: FileFormat::Parquet,
            },
        );
        let initial = database_source_warnings(&spec, false, Some(DatabaseKind::Oracle));
        assert_eq!(initial.len(), 1);
        assert!(initial[0].contains("byte-bounded native array"));
        assert!(initial[0].contains("not performance-qualified"));
        let resumed = database_source_warnings(&spec, true, Some(DatabaseKind::Oracle));
        assert_eq!(resumed.len(), 2);
        assert!(
            resumed
                .iter()
                .any(|warning| warning.contains("restarted from zero"))
        );
        let file = SourceSpec::File {
            path: "input.parquet".into(),
            format: FileFormat::Parquet,
        };
        assert_eq!(
            source_resume_checkpoint(&file, Some(checkpoint.clone())),
            Some(checkpoint)
        );
        assert!(validate_source_resume(DatabaseKind::PostgreSql, true).is_ok());
        for kind in [
            DatabaseKind::MySql,
            DatabaseKind::SqlServer,
            DatabaseKind::Oracle,
        ] {
            assert!(validate_source_resume(kind, false).is_ok());
            assert!(matches!(
                validate_source_resume(kind, true),
                Err(ElmError::Unsupported(_))
            ));
        }
    }

    #[tokio::test]
    async fn experimental_source_warning_is_persisted_and_broadcast_once() {
        let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
        let store = StateStore::open(directory.path().join("state.sqlite3"))
            .unwrap_or_else(|error| panic!("{error}"));
        let spec = JobSpec::new(
            SourceSpec::Database {
                environment_id: EnvironmentId(uuid::Uuid::new_v4()),
                selection: elm_core::DatabaseSelection::Query {
                    sql: "SELECT 1 FROM dual".into(),
                },
                resume_key: Vec::new(),
            },
            SinkSpec::File {
                path: directory.path().join("output.parquet"),
                format: FileFormat::Parquet,
            },
        );
        let mut progress = store
            .create_job(&spec)
            .unwrap_or_else(|error| panic!("{error}"))
            .progress;
        let (events, mut receiver) = broadcast::channel(4);
        let warnings = database_source_warnings(&spec, false, Some(DatabaseKind::Oracle));
        let observer = StateObserver {
            store: store.clone(),
            events,
            job_id: spec.id,
            injected_warnings: warnings.clone(),
        };
        progress.state = JobState::Preflighting;
        progress.warnings = warnings.clone();
        observer
            .progress(progress)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            receiver
                .recv()
                .await
                .unwrap_or_else(|error| panic!("{error}"))
                .warnings,
            warnings
        );
        assert_eq!(
            store
                .get_job(spec.id)
                .unwrap_or_else(|error| panic!("{error}"))
                .progress
                .warnings,
            warnings
        );
    }

    #[tokio::test]
    async fn confirmed_publication_finishes_retry_and_broadcasts_without_transfer() {
        for database in [DatabaseKind::MySql, DatabaseKind::SqlServer] {
            assert_publication_reconciliation(database).await;
        }
    }

    async fn assert_publication_reconciliation(database: DatabaseKind) {
        let directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
        let store = StateStore::open(directory.path().join("state.sqlite3"))
            .unwrap_or_else(|error| panic!("{error}"));
        let spec = JobSpec::new(
            SourceSpec::File {
                path: directory.path().join("missing.csv"),
                format: FileFormat::Csv,
            },
            SinkSpec::File {
                path: directory.path().join("untouched.csv"),
                format: FileFormat::Csv,
            },
        );
        let mut progress = store
            .create_job(&spec)
            .unwrap_or_else(|error| panic!("{error}"))
            .progress;
        for state in [
            JobState::Preflighting,
            JobState::Running,
            JobState::Publishing,
        ] {
            progress.state = state;
            progress.rows = 12;
            progress.bytes = 1200;
            progress.batches = 3;
            store
                .record_progress(&progress)
                .unwrap_or_else(|error| panic!("{error}"));
        }
        store
            .mark_active_jobs_interrupted()
            .unwrap_or_else(|error| panic!("{error}"));
        store
            .queue_retry(spec.id)
            .unwrap_or_else(|error| panic!("{error}"));
        let (events, mut receiver) = broadcast::channel(16);
        let observer = StateObserver {
            store: store.clone(),
            events,
            job_id: spec.id,
            injected_warnings: vec!["must not claim the source restarted".into()],
        };
        observer
            .reconcile_publication(12, database)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        let job = store
            .get_job(spec.id)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(job.progress.state, JobState::Succeeded);
        assert_eq!(
            (job.progress.rows, job.progress.bytes, job.progress.batches),
            (12, 1200, 3)
        );
        assert_eq!(job.attempt, 2);
        assert!(
            job.progress
                .warnings
                .iter()
                .any(|warning| warning.contains(&database.to_string()))
        );
        assert!(
            job.progress
                .warnings
                .iter()
                .any(|warning| warning.contains("no source rows"))
        );
        assert!(
            !job.progress
                .warnings
                .iter()
                .any(|warning| warning.contains("must not claim"))
        );
        for state in [
            JobState::Preflighting,
            JobState::Running,
            JobState::Publishing,
            JobState::Succeeded,
        ] {
            assert_eq!(
                receiver
                    .try_recv()
                    .unwrap_or_else(|error| panic!("{error}"))
                    .state,
                state
            );
        }
        assert!(!directory.path().join("untouched.csv").exists());
    }
}
