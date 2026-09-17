use std::{collections::HashSet, fmt, path::PathBuf, str::FromStr, time::Duration};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{ElmError, Result};

pub const JOB_SPEC_VERSION: u16 = 1;
pub const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_BATCH_TARGET_BYTES: u64 = 16 * 1024 * 1024;
pub const MIN_BATCH_TARGET_BYTES: u64 = 8 * 1024 * 1024;
pub const MAX_BATCH_TARGET_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSettings {
    pub memory_budget_bytes: u64,
    pub batch_target_bytes: u64,
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            batch_target_bytes: DEFAULT_BATCH_TARGET_BYTES,
        }
    }
}

impl RuntimeSettings {
    pub fn validate(self) -> Result<()> {
        if !(MIN_BATCH_TARGET_BYTES..=MAX_BATCH_TARGET_BYTES).contains(&self.batch_target_bytes) {
            return Err(ElmError::Validation(format!(
                "batch target must be between {MIN_BATCH_TARGET_BYTES} and {MAX_BATCH_TARGET_BYTES} bytes"
            )));
        }
        if self.memory_budget_bytes < self.batch_target_bytes.saturating_mul(2) {
            return Err(ElmError::Validation(
                "memory budget must hold at least two target batches".into(),
            ));
        }
        Ok(())
    }
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl FromStr for $name {
            type Err = uuid::Error;

            fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
                Uuid::parse_str(value).map(Self)
            }
        }
    };
}

id_type!(JobId);
id_type!(EnvironmentId);
id_type!(MaskRuleId);
id_type!(AttemptId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseKind {
    PostgreSql,
    Oracle,
    MySql,
    SqlServer,
}

impl fmt::Display for DatabaseKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PostgreSql => "postgresql",
            Self::Oracle => "oracle",
            Self::MySql => "mysql",
            Self::SqlServer => "sql_server",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Identifier(String);

impl Identifier {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty() || value.contains('\0') {
            return Err(ElmError::Validation(
                "identifiers must be non-empty and may not contain NUL".into(),
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relation {
    pub catalog: Option<Identifier>,
    pub schema: Option<Identifier>,
    pub name: Identifier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Environment {
    pub id: EnvironmentId,
    pub name: String,
    pub kind: DatabaseKind,
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    /// Opaque keychain account name; never a password or connection URL.
    pub credential_ref: String,
    pub options: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Environment {
    /// Oracle native round-trip deadline. Zero (unbounded) is never accepted.
    pub fn oracle_call_timeout_ms(&self) -> Result<u64> {
        match self.options.get("oracle_call_timeout_ms") {
            None => Ok(15_000),
            Some(value) if self.kind == DatabaseKind::Oracle => value
                .as_u64()
                .filter(|value| (1..=300_000).contains(value))
                .ok_or_else(|| {
                    ElmError::Validation(
                        "oracle_call_timeout_ms must be an integer from 1 through 300000".into(),
                    )
                }),
            Some(_) => Err(ElmError::Validation(
                "oracle_call_timeout_ms is only supported by Oracle environments".into(),
            )),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.oracle_call_timeout_ms()?;
        if self.name.trim().is_empty()
            || self.host.trim().is_empty()
            || self.database.trim().is_empty()
            || self.username.trim().is_empty()
            || self.credential_ref.trim().is_empty()
        {
            return Err(ElmError::Validation(
                "environment name, host, database, username, and credential reference are required"
                    .into(),
            ));
        }
        if self.port == 0 {
            return Err(ElmError::Validation(
                "environment port must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    Csv,
    Ndjson,
    Parquet,
}

impl FromStr for FileFormat {
    type Err = ElmError;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "csv" => Ok(Self::Csv),
            "ndjson" | "jsonl" => Ok(Self::Ndjson),
            "parquet" => Ok(Self::Parquet),
            _ => Err(ElmError::Validation(format!(
                "unsupported file format: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatabaseSelection {
    Table {
        relation: Relation,
    },
    Query {
        /// Explicit user SQL. Implementations must execute it as-is and never append identifiers
        /// or checkpoint literals. Resumable query sources are intentionally unsupported.
        sql: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceSpec {
    Database {
        environment_id: EnvironmentId,
        selection: DatabaseSelection,
        resume_key: Vec<Identifier>,
    },
    File {
        path: PathBuf,
        format: FileFormat,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SinkSpec {
    Database {
        environment_id: EnvironmentId,
        relation: Relation,
    },
    File {
        path: PathBuf,
        format: FileFormat,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteMode {
    Append,
    Replace,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistencyMode {
    Atomic,
    Checkpointed,
    /// Explicit, non-atomic Oracle REPLACE with a recoverable backup.
    TableSwap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversionRule {
    pub column: Identifier,
    pub target_type: String,
    pub allow_lossy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "algorithm", rename_all = "snake_case")]
pub enum MaskAlgorithm {
    Star,
    StarLength { unmasked_prefix: usize },
    Random,
    Nullify,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskRule {
    pub id: MaskRuleId,
    pub name: String,
    pub column: Identifier,
    pub algorithm: MaskAlgorithm,
    pub environment_id: Option<EnvironmentId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSpec {
    pub version: u16,
    pub id: JobId,
    pub name: Option<String>,
    pub source: SourceSpec,
    pub sink: SinkSpec,
    pub write_mode: WriteMode,
    pub consistency: ConsistencyMode,
    pub memory_budget_bytes: u64,
    pub batch_target_bytes: u64,
    pub masks: Vec<MaskRule>,
    pub random_seed: [u8; 32],
    pub conversions: Vec<ConversionRule>,
    pub submitted_at: DateTime<Utc>,
}

impl JobSpec {
    #[must_use]
    pub fn new(source: SourceSpec, sink: SinkSpec) -> Self {
        let id = JobId::new();
        let seed = *blake3::hash(id.0.as_bytes()).as_bytes();
        Self {
            version: JOB_SPEC_VERSION,
            id,
            name: None,
            source,
            sink,
            write_mode: WriteMode::Append,
            consistency: ConsistencyMode::Atomic,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            batch_target_bytes: DEFAULT_BATCH_TARGET_BYTES,
            masks: Vec::new(),
            random_seed: seed,
            conversions: Vec::new(),
            submitted_at: Utc::now(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.consistency == ConsistencyMode::TableSwap
            && (self.write_mode != WriteMode::Replace
                || !matches!(self.sink, SinkSpec::Database { .. }))
        {
            return Err(ElmError::Validation(
                "table-swap consistency requires a database destination and REPLACE mode".into(),
            ));
        }
        if self.version != JOB_SPEC_VERSION {
            return Err(ElmError::Validation(format!(
                "unsupported job specification version {}; expected {JOB_SPEC_VERSION}",
                self.version
            )));
        }
        RuntimeSettings {
            memory_budget_bytes: self.memory_budget_bytes,
            batch_target_bytes: self.batch_target_bytes,
        }
        .validate()?;
        if let SourceSpec::Database {
            selection: DatabaseSelection::Query { .. },
            resume_key,
            ..
        } = &self.source
            && !resume_key.is_empty()
        {
            return Err(ElmError::Validation(
                "resume keys are supported only for structured table sources, not user SQL".into(),
            ));
        }
        if let (SourceSpec::File { path: source, .. }, SinkSpec::File { path: sink, .. }) =
            (&self.source, &self.sink)
            && source == sink
        {
            return Err(ElmError::Validation(
                "source and destination file must be different".into(),
            ));
        }
        let mut masked_columns = HashSet::with_capacity(self.masks.len());
        for rule in &self.masks {
            if !masked_columns.insert(rule.column.as_str()) {
                return Err(ElmError::Validation(format!(
                    "more than one masking rule targets column '{}'; apply_masks would silently \
                     use only the last one, so select at most one masking rule per column",
                    rule.column
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Preflighting,
    Running,
    Publishing,
    Succeeded,
    Failed,
    Cancelling,
    Cancelled,
    Interrupted,
}

impl JobState {
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    #[must_use]
    pub fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Queued,
                Self::Preflighting | Self::Cancelling | Self::Interrupted,
            ) | (
                Self::Preflighting,
                Self::Running | Self::Failed | Self::Cancelling | Self::Interrupted
            ) | (
                Self::Running,
                Self::Publishing | Self::Failed | Self::Cancelling | Self::Interrupted
            ) | (
                Self::Publishing,
                Self::Succeeded | Self::Failed | Self::Interrupted
            ) | (
                Self::Cancelling,
                Self::Cancelled | Self::Failed | Self::Interrupted,
            ) | (Self::Interrupted, Self::Queued | Self::Cancelled)
                | (Self::Failed, Self::Queued)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobProgress {
    pub job_id: JobId,
    pub state: JobState,
    pub rows: u64,
    pub bytes: u64,
    pub batches: u64,
    pub rows_per_second: f64,
    pub bytes_per_second: f64,
    pub elapsed: Duration,
    pub eta: Option<Duration>,
    pub warnings: Vec<String>,
    pub error: Option<crate::PublicError>,
    pub occurred_at: DateTime<Utc>,
}

impl JobProgress {
    #[must_use]
    pub fn initial(job_id: JobId) -> Self {
        Self {
            job_id,
            state: JobState::Queued,
            rows: 0,
            bytes: 0,
            batches: 0,
            rows_per_second: 0.0,
            bytes_per_second: 0.0,
            elapsed: Duration::ZERO,
            eta: None,
            warnings: Vec::new(),
            error: None,
            occurred_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    pub spec: JobSpec,
    pub progress: JobProgress,
    pub attempt: u32,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_timeout_is_bounded_and_driver_specific() {
        let mut environment = Environment {
            id: EnvironmentId::new(),
            name: "test".into(),
            kind: DatabaseKind::Oracle,
            host: "localhost".into(),
            port: 1521,
            database: "test".into(),
            username: "test".into(),
            credential_ref: "test-reference".into(),
            options: serde_json::json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        assert_eq!(environment.oracle_call_timeout_ms().ok(), Some(15000));
        for value in [1, 120000, 300000] {
            environment.options["oracle_call_timeout_ms"] = value.into();
            assert!(environment.validate().is_ok());
            assert_eq!(environment.oracle_call_timeout_ms().ok(), Some(value));
        }
        for value in [
            serde_json::json!(0),
            serde_json::json!(300001),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("120000"),
            serde_json::Value::Null,
        ] {
            environment.options["oracle_call_timeout_ms"] = value;
            assert!(environment.validate().is_err());
        }
        environment.options["oracle_call_timeout_ms"] = 120000.into();
        environment.kind = DatabaseKind::PostgreSql;
        assert!(environment.validate().is_err());
    }

    fn relation(name: &str) -> Relation {
        Relation {
            catalog: None,
            schema: None,
            name: Identifier::new(name).unwrap_or_else(|error| panic!("{error}")),
        }
    }

    #[test]
    fn query_resume_key_is_rejected() {
        let source = SourceSpec::Database {
            environment_id: EnvironmentId::new(),
            selection: DatabaseSelection::Query {
                sql: "select 1".into(),
            },
            resume_key: vec![Identifier::new("id").unwrap_or_else(|error| panic!("{error}"))],
        };
        let sink = SinkSpec::Database {
            environment_id: EnvironmentId::new(),
            relation: relation("target"),
        };
        assert!(JobSpec::new(source, sink).validate().is_err());
    }

    #[test]
    fn duplicate_masked_column_is_rejected() {
        let source = SourceSpec::File {
            path: PathBuf::from("in.csv"),
            format: FileFormat::Csv,
        };
        let sink = SinkSpec::File {
            path: PathBuf::from("out.csv"),
            format: FileFormat::Csv,
        };
        let mut spec = JobSpec::new(source, sink);
        let column = Identifier::new("email").unwrap_or_else(|error| panic!("{error}"));
        spec.masks = vec![
            MaskRule {
                id: MaskRuleId::new(),
                name: "star email".into(),
                column: column.clone(),
                algorithm: MaskAlgorithm::Star,
                environment_id: None,
            },
            MaskRule {
                id: MaskRuleId::new(),
                name: "nullify email".into(),
                column,
                algorithm: MaskAlgorithm::Nullify,
                environment_id: None,
            },
        ];
        match spec.validate() {
            Ok(()) => panic!("duplicate masked column must fail closed"),
            Err(error) => assert!(error.to_string().contains("email")),
        }
    }

    #[test]
    fn terminal_states_cannot_transition() {
        assert!(!JobState::Succeeded.can_transition_to(JobState::Running));
        assert!(JobState::Interrupted.can_transition_to(JobState::Queued));
        assert!(JobState::Queued.can_transition_to(JobState::Interrupted));
        assert!(JobState::Preflighting.can_transition_to(JobState::Interrupted));
        assert!(JobState::Cancelling.can_transition_to(JobState::Interrupted));
    }
}
