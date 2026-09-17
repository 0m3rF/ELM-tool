use std::{
    fs,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use directories::ProjectDirs;
use rusqlite::{Connection, OptionalExtension, Transaction, params};

use elm_core::{
    BatchCheckpoint, ElmError, Environment, EnvironmentId, JobId, JobProgress, JobRecord, JobSpec,
    JobState, MaskRule, MaskRuleId, Result, RuntimeSettings,
};

const DATABASE_FILE: &str = "elm-v2.sqlite3";
const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct StateStore {
    path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    pub sequence: i64,
    pub job_id: JobId,
    pub progress: JobProgress,
    pub occurred_at: DateTime<Utc>,
}

#[must_use]
pub fn default_data_directory() -> Option<PathBuf> {
    ProjectDirs::from("dev", "ELM Tool", "ELM Tool").map(|dirs| dirs.data_local_dir().to_path_buf())
}

impl StateStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let store = Self { path };
        let mut connection = store.connect()?;
        migrate(&mut connection)?;
        Ok(store)
    }

    pub fn open_default() -> Result<Self> {
        let directory = default_data_directory().ok_or_else(|| {
            ElmError::State("operating system did not provide a per-user data directory".into())
        })?;
        Self::open(directory.join(DATABASE_FILE))
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn connect(&self) -> Result<Connection> {
        let connection = Connection::open(&self.path).map_err(sql_error)?;
        connection
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = FULL;
                 PRAGMA busy_timeout = 5000;",
            )
            .map_err(sql_error)?;
        Ok(connection)
    }

    pub fn upsert_environment(&self, environment: &Environment) -> Result<()> {
        environment.validate()?;
        let document = to_json(environment)?;
        self.connect()?
            .execute(
                "INSERT INTO environments (id, name, document, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    document = excluded.document,
                    updated_at = excluded.updated_at",
                params![
                    environment.id.to_string(),
                    environment.name,
                    document,
                    environment.created_at.to_rfc3339(),
                    environment.updated_at.to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        Ok(())
    }

    pub fn list_environments(&self) -> Result<Vec<Environment>> {
        query_documents(
            &self.connect()?,
            "SELECT document FROM environments ORDER BY name COLLATE NOCASE, id",
            [],
        )
    }

    pub fn get_environment(&self, id: EnvironmentId) -> Result<Environment> {
        query_document(
            &self.connect()?,
            "SELECT document FROM environments WHERE id = ?1",
            [id.to_string()],
            "environment",
        )
    }

    pub fn remove_environment(&self, id: EnvironmentId) -> Result<()> {
        let changed = self
            .connect()?
            .execute("DELETE FROM environments WHERE id = ?1", [id.to_string()])
            .map_err(sql_error)?;
        if changed == 0 {
            return Err(ElmError::NotFound(format!("environment {id}")));
        }
        Ok(())
    }

    pub fn upsert_mask(&self, rule: &MaskRule) -> Result<()> {
        if rule.name.trim().is_empty() {
            return Err(ElmError::Validation("mask name is required".into()));
        }
        self.connect()?
            .execute(
                "INSERT INTO masks (id, document, updated_at) VALUES (?1, ?2, ?3)
                 ON CONFLICT(id) DO UPDATE SET
                    document = excluded.document,
                    updated_at = excluded.updated_at",
                params![rule.id.to_string(), to_json(rule)?, Utc::now().to_rfc3339()],
            )
            .map_err(sql_error)?;
        Ok(())
    }

    pub fn list_masks(&self) -> Result<Vec<MaskRule>> {
        query_documents(
            &self.connect()?,
            "SELECT document FROM masks ORDER BY updated_at, id",
            [],
        )
    }

    pub fn remove_mask(&self, id: MaskRuleId) -> Result<()> {
        let changed = self
            .connect()?
            .execute("DELETE FROM masks WHERE id = ?1", [id.to_string()])
            .map_err(sql_error)?;
        if changed == 0 {
            return Err(ElmError::NotFound(format!("mask rule {id}")));
        }
        Ok(())
    }

    pub fn load_settings(&self) -> Result<RuntimeSettings> {
        let document = self
            .connect()?
            .query_row("SELECT document FROM settings WHERE id = 1", [], |row| {
                row.get::<_, String>(0)
            })
            .optional()
            .map_err(sql_error)?;
        document
            .map(|value| from_json(&value))
            .transpose()
            .map(Option::unwrap_or_default)
    }

    pub fn save_settings(&self, settings: RuntimeSettings) -> Result<()> {
        settings.validate()?;
        self.connect()?
            .execute(
                "INSERT INTO settings (id, document, updated_at) VALUES (1, ?1, ?2)
                 ON CONFLICT(id) DO UPDATE SET document = excluded.document, updated_at = excluded.updated_at",
                params![to_json(&settings)?, Utc::now().to_rfc3339()],
            )
            .map_err(sql_error)?;
        Ok(())
    }

    pub fn create_job(&self, spec: &JobSpec) -> Result<JobRecord> {
        spec.validate()?;
        let progress = JobProgress::initial(spec.id);
        let now = Utc::now();
        let mut connection = self.connect()?;
        let transaction = connection.transaction().map_err(sql_error)?;
        transaction
            .execute(
                "INSERT INTO jobs
                 (id, spec, progress, state, attempt, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, 1, ?5, ?5)",
                params![
                    spec.id.to_string(),
                    to_json(spec)?,
                    to_json(&progress)?,
                    state_name(progress.state)?,
                    now.to_rfc3339(),
                ],
            )
            .map_err(map_insert_conflict("job", spec.id))?;
        transaction
            .execute(
                "INSERT INTO attempts (id, job_id, number, state, started_at)
                 VALUES (?1, ?2, 1, ?3, ?4)",
                params![
                    elm_core::AttemptId::new().to_string(),
                    spec.id.to_string(),
                    state_name(JobState::Queued)?,
                    now.to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        insert_event(&transaction, spec.id, &progress)?;
        transaction.commit().map_err(sql_error)?;
        Ok(JobRecord {
            spec: spec.clone(),
            progress,
            attempt: 1,
            updated_at: now,
        })
    }

    pub fn get_job(&self, id: JobId) -> Result<JobRecord> {
        let connection = self.connect()?;
        connection
            .query_row(
                "SELECT spec, progress, attempt, updated_at FROM jobs WHERE id = ?1",
                [id.to_string()],
                job_from_row,
            )
            .optional()
            .map_err(sql_error)?
            .ok_or_else(|| ElmError::NotFound(format!("job {id}")))
    }

    pub fn list_jobs(&self) -> Result<Vec<JobRecord>> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare(
                "SELECT spec, progress, attempt, updated_at
                 FROM jobs ORDER BY created_at DESC, id DESC",
            )
            .map_err(sql_error)?;
        let rows = statement.query_map([], job_from_row).map_err(sql_error)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql_error)
    }

    pub fn record_progress(&self, progress: &JobProgress) -> Result<()> {
        let mut connection = self.connect()?;
        // Must take the write lock up front (not the default deferred/upgrade behavior): this
        // transaction reads the current state and then writes based on it, so if two callers
        // (e.g. a running job's own progress updates and a concurrent job.cancel) both start as
        // deferred readers, one's later upgrade-to-write can hit an immediate "database is
        // locked" even with `busy_timeout` set, because WAL only retries a connection that is
        // still waiting for its *first* write lock, not one whose read snapshot has gone stale.
        let transaction = connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(sql_error)?;
        let current: String = transaction
            .query_row(
                "SELECT state FROM jobs WHERE id = ?1",
                [progress.job_id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(sql_error)?
            .ok_or_else(|| ElmError::NotFound(format!("job {}", progress.job_id)))?;
        let current = parse_state(&current)?;
        if current != progress.state && !current.can_transition_to(progress.state) {
            return Err(ElmError::Conflict(format!(
                "invalid job state transition from {current:?} to {:?}",
                progress.state
            )));
        }
        let now = Utc::now();
        transaction
            .execute(
                "UPDATE jobs SET progress = ?2, state = ?3, updated_at = ?4 WHERE id = ?1",
                params![
                    progress.job_id.to_string(),
                    to_json(progress)?,
                    state_name(progress.state)?,
                    now.to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        insert_event(&transaction, progress.job_id, progress)?;
        let finished_at = (progress.state.is_terminal() || progress.state == JobState::Interrupted)
            .then(|| now.to_rfc3339());
        transaction
            .execute(
                "UPDATE attempts SET state = ?2, finished_at = ?3
                 WHERE job_id = ?1 AND number = (SELECT attempt FROM jobs WHERE id = ?1)",
                params![
                    progress.job_id.to_string(),
                    state_name(progress.state)?,
                    finished_at,
                ],
            )
            .map_err(sql_error)?;
        transaction.commit().map_err(sql_error)
    }

    pub fn save_checkpoint(&self, job_id: JobId, checkpoint: &BatchCheckpoint) -> Result<()> {
        self.connect()?
            .execute(
                "INSERT INTO checkpoints (job_id, sequence, document, committed_at)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(job_id) DO UPDATE SET
                    sequence = excluded.sequence,
                    document = excluded.document,
                    committed_at = excluded.committed_at",
                params![
                    job_id.to_string(),
                    i64::try_from(checkpoint.sequence).map_err(|_| {
                        ElmError::State("checkpoint sequence exceeds SQLite integer range".into())
                    })?,
                    to_json(checkpoint)?,
                    Utc::now().to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        Ok(())
    }

    pub fn load_checkpoint(&self, job_id: JobId) -> Result<Option<BatchCheckpoint>> {
        let document = self
            .connect()?
            .query_row(
                "SELECT document FROM checkpoints WHERE job_id = ?1",
                [job_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(sql_error)?;
        document.map(|value| from_json(&value)).transpose()
    }

    pub fn register_recovery_resource(
        &self,
        job_id: JobId,
        connector: &str,
        resource: &str,
    ) -> Result<()> {
        if connector.trim().is_empty() || resource.trim().is_empty() {
            return Err(ElmError::Validation(
                "recovery connector and resource are required".into(),
            ));
        }
        self.connect()?
            .execute(
                "INSERT INTO recovery_resources (job_id, connector, resource, created_at)
                 SELECT ?1, ?2, ?3, ?4
                 WHERE NOT EXISTS (
                    SELECT 1 FROM recovery_resources
                    WHERE job_id = ?1 AND connector = ?2 AND resource = ?3 AND cleaned_at IS NULL
                 )",
                params![
                    job_id.to_string(),
                    connector,
                    resource,
                    Utc::now().to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        Ok(())
    }

    pub fn mark_recovery_resource_cleaned(
        &self,
        job_id: JobId,
        connector: &str,
        resource: &str,
    ) -> Result<()> {
        let changed = self
            .connect()?
            .execute(
                "UPDATE recovery_resources SET cleaned_at = ?4
                 WHERE job_id = ?1 AND connector = ?2 AND resource = ?3 AND cleaned_at IS NULL",
                params![
                    job_id.to_string(),
                    connector,
                    resource,
                    Utc::now().to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        if changed == 0 {
            return Err(ElmError::NotFound(format!(
                "open recovery resource for job {job_id}"
            )));
        }
        Ok(())
    }

    pub fn events_after(&self, job_id: JobId, after: i64) -> Result<Vec<EventRecord>> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare(
                "SELECT sequence, progress, occurred_at FROM events
                 WHERE job_id = ?1 AND sequence > ?2 ORDER BY sequence",
            )
            .map_err(sql_error)?;
        let rows = statement
            .query_map(params![job_id.to_string(), after], |row| {
                let progress_json: String = row.get(1)?;
                let occurred_at: String = row.get(2)?;
                Ok((row.get::<_, i64>(0)?, progress_json, occurred_at))
            })
            .map_err(sql_error)?;
        rows.map(|row| {
            let (sequence, progress, occurred_at) = row.map_err(sql_error)?;
            Ok(EventRecord {
                sequence,
                job_id,
                progress: from_json(&progress)?,
                occurred_at: parse_time(&occurred_at)?,
            })
        })
        .collect()
    }

    pub fn mark_active_jobs_interrupted(&self) -> Result<usize> {
        let jobs = self.list_jobs()?;
        let mut changed = 0;
        for job in jobs.into_iter().filter(|job| {
            matches!(
                job.progress.state,
                JobState::Queued
                    | JobState::Preflighting
                    | JobState::Running
                    | JobState::Publishing
                    | JobState::Cancelling
            )
        }) {
            let mut progress = job.progress;
            progress.state = JobState::Interrupted;
            progress.occurred_at = Utc::now();
            progress.warnings.push(
                "The daemon stopped during this attempt. Resume uses the last committed checkpoint."
                    .into(),
            );
            self.record_progress(&progress)?;
            changed += 1;
        }
        Ok(changed)
    }

    pub fn queue_retry(&self, id: JobId) -> Result<JobRecord> {
        let job = self.get_job(id)?;
        if !matches!(job.progress.state, JobState::Interrupted | JobState::Failed) {
            return Err(ElmError::Conflict(
                "only interrupted or failed jobs can be resumed".into(),
            ));
        }
        let attempt = job.attempt.saturating_add(1);
        let now = Utc::now();
        let mut progress = job.progress;
        progress.state = JobState::Queued;
        progress.error = None;
        progress.occurred_at = now;
        let mut connection = self.connect()?;
        let transaction = connection.transaction().map_err(sql_error)?;
        transaction
            .execute(
                "UPDATE jobs SET progress = ?2, state = ?3, attempt = ?4, updated_at = ?5
                 WHERE id = ?1",
                params![
                    id.to_string(),
                    to_json(&progress)?,
                    state_name(JobState::Queued)?,
                    attempt,
                    now.to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        transaction
            .execute(
                "INSERT INTO attempts (id, job_id, number, state, started_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    elm_core::AttemptId::new().to_string(),
                    id.to_string(),
                    attempt,
                    state_name(JobState::Queued)?,
                    now.to_rfc3339(),
                ],
            )
            .map_err(sql_error)?;
        insert_event(&transaction, id, &progress)?;
        transaction.commit().map_err(sql_error)?;
        Ok(JobRecord {
            spec: job.spec,
            progress,
            attempt,
            updated_at: now,
        })
    }

    pub fn delete_job(&self, id: JobId) -> Result<()> {
        let job = self.get_job(id)?;
        if !job.progress.state.is_terminal() && job.progress.state != JobState::Interrupted {
            return Err(ElmError::Conflict(
                "active or queued jobs cannot be deleted".into(),
            ));
        }
        self.connect()?
            .execute("DELETE FROM jobs WHERE id = ?1", [id.to_string()])
            .map_err(sql_error)?;
        Ok(())
    }
}

fn migrate(connection: &mut Connection) -> Result<()> {
    let transaction = connection.transaction().map_err(sql_error)?;
    transaction
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS environments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                document TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS masks (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                document TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                spec TEXT NOT NULL,
                progress TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS jobs_state_idx ON jobs(state, updated_at);
             CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                number INTEGER NOT NULL,
                state TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                UNIQUE(job_id, number)
             );
             CREATE TABLE IF NOT EXISTS checkpoints (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                document TEXT NOT NULL,
                committed_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                progress TEXT NOT NULL,
                occurred_at TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id, sequence);
             CREATE TABLE IF NOT EXISTS recovery_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                connector TEXT NOT NULL,
                resource TEXT NOT NULL,
                created_at TEXT NOT NULL,
                cleaned_at TEXT
             );",
        )
        .map_err(sql_error)?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
            params![SCHEMA_VERSION, Utc::now().to_rfc3339()],
        )
        .map_err(sql_error)?;
    transaction.commit().map_err(sql_error)
}

fn insert_event(transaction: &Transaction<'_>, id: JobId, progress: &JobProgress) -> Result<()> {
    transaction
        .execute(
            "INSERT INTO events (job_id, progress, occurred_at) VALUES (?1, ?2, ?3)",
            params![
                id.to_string(),
                to_json(progress)?,
                progress.occurred_at.to_rfc3339(),
            ],
        )
        .map_err(sql_error)?;
    Ok(())
}

fn job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<JobRecord> {
    let spec: String = row.get(0)?;
    let progress: String = row.get(1)?;
    let updated_at: String = row.get(3)?;
    let spec = serde_json::from_str(&spec).map_err(json_sql_error)?;
    let progress = serde_json::from_str(&progress).map_err(json_sql_error)?;
    let updated_at = DateTime::parse_from_rfc3339(&updated_at)
        .map_err(time_sql_error)?
        .with_timezone(&Utc);
    Ok(JobRecord {
        spec,
        progress,
        attempt: row.get(2)?,
        updated_at,
    })
}

fn query_document<T, P>(
    connection: &Connection,
    sql: &str,
    parameters: P,
    entity: &str,
) -> Result<T>
where
    T: serde::de::DeserializeOwned,
    P: rusqlite::Params,
{
    let value = connection
        .query_row(sql, parameters, |row| row.get::<_, String>(0))
        .optional()
        .map_err(sql_error)?
        .ok_or_else(|| ElmError::NotFound(entity.into()))?;
    from_json(&value)
}

fn query_documents<T, P>(connection: &Connection, sql: &str, parameters: P) -> Result<Vec<T>>
where
    T: serde::de::DeserializeOwned,
    P: rusqlite::Params,
{
    let mut statement = connection.prepare(sql).map_err(sql_error)?;
    let rows = statement
        .query_map(parameters, |row| row.get::<_, String>(0))
        .map_err(sql_error)?;
    rows.map(|row| from_json(&row.map_err(sql_error)?))
        .collect()
}

fn state_name(state: JobState) -> Result<String> {
    serde_json::to_value(state)
        .map_err(|error| ElmError::State(format!("cannot encode job state: {error}")))?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| ElmError::State("job state did not encode as a string".into()))
}

fn parse_state(value: &str) -> Result<JobState> {
    serde_json::from_value(serde_json::Value::String(value.into()))
        .map_err(|error| ElmError::State(format!("invalid persisted job state: {error}")))
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|error| ElmError::State(format!("cannot encode state document: {error}")))
}

fn from_json<T: serde::de::DeserializeOwned>(value: &str) -> Result<T> {
    serde_json::from_str(value)
        .map_err(|error| ElmError::State(format!("cannot decode state document: {error}")))
}

fn parse_time(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|time| time.with_timezone(&Utc))
        .map_err(|error| ElmError::State(format!("invalid persisted timestamp: {error}")))
}

fn sql_error(error: rusqlite::Error) -> ElmError {
    ElmError::State(error.to_string())
}

fn map_insert_conflict(
    entity: &'static str,
    id: impl std::fmt::Display,
) -> impl FnOnce(rusqlite::Error) -> ElmError {
    move |error| match error {
        rusqlite::Error::SqliteFailure(code, _)
            if code.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            ElmError::Conflict(format!("{entity} {id} already exists"))
        }
        other => sql_error(other),
    }
}

fn json_sql_error(error: serde_json::Error) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

fn time_sql_error(error: chrono::ParseError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use elm_core::{FileFormat, SinkSpec, SourceSpec};

    fn store() -> (tempfile::TempDir, StateStore) {
        let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
        let store = StateStore::open(directory.path().join("state.sqlite3"))
            .unwrap_or_else(|error| panic!("{error}"));
        (directory, store)
    }

    fn job() -> JobSpec {
        JobSpec::new(
            SourceSpec::File {
                path: "input.csv".into(),
                format: FileFormat::Csv,
            },
            SinkSpec::File {
                path: "output.parquet".into(),
                format: FileFormat::Parquet,
            },
        )
    }

    #[test]
    fn persists_jobs_events_and_checkpoints_transactionally() {
        let (_directory, store) = store();
        let spec = job();
        let record = store
            .create_job(&spec)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(record.progress.state, JobState::Queued);

        let mut progress = record.progress;
        progress.state = JobState::Preflighting;
        store
            .record_progress(&progress)
            .unwrap_or_else(|error| panic!("{error}"));
        progress.state = JobState::Running;
        store
            .record_progress(&progress)
            .unwrap_or_else(|error| panic!("{error}"));
        let checkpoint = BatchCheckpoint {
            sequence: 7,
            source_position: serde_json::json!({"row": 20}),
            rows_committed: 20,
            bytes_committed: 800,
            source_fingerprint: Some("hash".into()),
        };
        store
            .save_checkpoint(spec.id, &checkpoint)
            .unwrap_or_else(|error| panic!("{error}"));

        assert_eq!(
            store
                .load_checkpoint(spec.id)
                .unwrap_or_else(|error| panic!("{error}")),
            Some(checkpoint)
        );
        assert_eq!(
            store
                .events_after(spec.id, 0)
                .unwrap_or_else(|error| panic!("{error}"))
                .len(),
            3
        );
    }

    #[test]
    fn startup_marks_only_active_jobs_interrupted() {
        let (_directory, store) = store();
        let spec = job();
        let mut progress = store
            .create_job(&spec)
            .unwrap_or_else(|error| panic!("{error}"))
            .progress;
        progress.state = JobState::Preflighting;
        store
            .record_progress(&progress)
            .unwrap_or_else(|error| panic!("{error}"));
        progress.state = JobState::Running;
        store
            .record_progress(&progress)
            .unwrap_or_else(|error| panic!("{error}"));
        let queued = job();
        store
            .create_job(&queued)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(store.mark_active_jobs_interrupted().unwrap_or(0), 2);
        assert_eq!(
            store
                .get_job(spec.id)
                .unwrap_or_else(|error| panic!("{error}"))
                .progress
                .state,
            JobState::Interrupted
        );
        assert_eq!(
            store
                .get_job(queued.id)
                .unwrap_or_else(|error| panic!("{error}"))
                .progress
                .state,
            JobState::Interrupted
        );
        let connection = store.connect().unwrap_or_else(|error| panic!("{error}"));
        let (attempt_state, finished_at): (String, Option<String>) = connection
            .query_row(
                "SELECT state, finished_at FROM attempts WHERE job_id = ?1 AND number = 1",
                [spec.id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(attempt_state, "interrupted");
        assert!(finished_at.is_some());
    }

    #[test]
    fn recovery_resources_are_idempotent_and_marked_cleaned() {
        let (_directory, store) = store();
        let spec = job();
        store
            .create_job(&spec)
            .unwrap_or_else(|error| panic!("{error}"));
        store
            .register_recovery_resource(spec.id, "file", "staging/job")
            .unwrap_or_else(|error| panic!("{error}"));
        store
            .register_recovery_resource(spec.id, "file", "staging/job")
            .unwrap_or_else(|error| panic!("{error}"));
        let connection = store.connect().unwrap_or_else(|error| panic!("{error}"));
        let open_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM recovery_resources WHERE job_id = ?1 AND cleaned_at IS NULL",
                [spec.id.to_string()],
                |row| row.get(0),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(open_count, 1);
        drop(connection);

        store
            .mark_recovery_resource_cleaned(spec.id, "file", "staging/job")
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(
            store
                .mark_recovery_resource_cleaned(spec.id, "file", "staging/job")
                .is_err()
        );
    }

    #[test]
    fn runtime_settings_round_trip_and_validate() {
        let (_directory, store) = store();
        assert_eq!(
            store
                .load_settings()
                .unwrap_or_else(|error| panic!("{error}")),
            RuntimeSettings::default()
        );
        let settings = RuntimeSettings {
            memory_budget_bytes: 768 * 1024 * 1024,
            batch_target_bytes: 32 * 1024 * 1024,
        };
        store
            .save_settings(settings)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            store
                .load_settings()
                .unwrap_or_else(|error| panic!("{error}")),
            settings
        );
        assert!(
            store
                .save_settings(RuntimeSettings {
                    memory_budget_bytes: 8 * 1024 * 1024,
                    batch_target_bytes: 8 * 1024 * 1024,
                })
                .is_err()
        );
    }
}
