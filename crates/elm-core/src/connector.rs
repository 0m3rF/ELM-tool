use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{ConsistencyMode, DatabaseKind, ElmError, Relation, Result, WriteMode};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorCapabilities {
    pub database: Option<DatabaseKind>,
    pub native_bulk_read: bool,
    pub native_bulk_write: bool,
    pub atomic_append: bool,
    pub atomic_replace: bool,
    #[serde(default)]
    pub non_atomic_replace: bool,
    pub checkpointed_write: bool,
    pub resumable_keyset_read: bool,
    pub supports_lob_spill: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreflightReport {
    pub capabilities: ConnectorCapabilities,
    pub staging_relation: Option<Relation>,
    pub warnings: Vec<String>,
}

/// A column as it will actually reach the destination, after any configured conversions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreviewColumn {
    pub name: String,
    pub source_type: String,
    pub output_type: String,
    pub nullable: bool,
}

/// The result of running preflight against a job's real source and destination without
/// executing the transfer: no staging is created, no rows move, and no target is touched
/// beyond the read-only privilege/schema checks each connector's `preflight()` already performs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferPreview {
    pub columns: Vec<PreviewColumn>,
    pub report: PreflightReport,
    pub publication_error: Option<String>,
}

impl PreflightReport {
    pub fn require_safe_publication(
        &self,
        mode: WriteMode,
        consistency: ConsistencyMode,
    ) -> Result<()> {
        if consistency == ConsistencyMode::TableSwap {
            return (mode == WriteMode::Replace
                && self.capabilities.database == Some(DatabaseKind::Oracle)
                && self.capabilities.non_atomic_replace)
                .then_some(())
                .ok_or_else(|| {
                    ElmError::Unsupported(
                        "table-swap requires an Oracle REPLACE sink with recovery support".into(),
                    )
                });
        }
        if consistency == ConsistencyMode::Checkpointed {
            return self
                .capabilities
                .checkpointed_write
                .then_some(())
                .ok_or_else(|| {
                    ElmError::Unsupported("connector does not support checkpointed writes".into())
                });
        }
        let supported = match mode {
            WriteMode::Append => self.capabilities.atomic_append,
            WriteMode::Replace | WriteMode::Fail => self.capabilities.atomic_replace,
        };
        supported.then_some(()).ok_or_else(|| {
            ElmError::PermissionDenied(
                "safe staged publication is unavailable; no automatic downgrade was attempted"
                    .into(),
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchCheckpoint {
    pub sequence: u64,
    pub source_position: serde_json::Value,
    pub rows_committed: u64,
    pub bytes_committed: u64,
    pub source_fingerprint: Option<String>,
}

#[async_trait]
pub trait DataSource: Send {
    async fn schema(&mut self) -> Result<Arc<Schema>>;
    async fn next_batch(&mut self, target_bytes: usize) -> Result<Option<RecordBatch>>;
    async fn checkpoint(&mut self) -> Result<serde_json::Value>;
}

#[async_trait]
pub trait DataSink: Send {
    async fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        consistency: ConsistencyMode,
    ) -> Result<PreflightReport>;
    async fn begin(&mut self) -> Result<()>;
    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()>;
    async fn commit_checkpoint(&mut self, checkpoint: &BatchCheckpoint) -> Result<()>;
    async fn publish(&mut self) -> Result<()>;
    async fn abort(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_atomic_swap_never_satisfies_atomic_publication() {
        let mut report = PreflightReport {
            capabilities: ConnectorCapabilities {
                database: Some(DatabaseKind::Oracle),
                native_bulk_read: true,
                native_bulk_write: true,
                atomic_append: true,
                atomic_replace: false,
                non_atomic_replace: true,
                checkpointed_write: false,
                resumable_keyset_read: false,
                supports_lob_spill: false,
            },
            staging_relation: None,
            warnings: Vec::new(),
        };
        assert!(
            report
                .require_safe_publication(WriteMode::Replace, ConsistencyMode::Atomic)
                .is_err()
        );
        assert!(
            report
                .require_safe_publication(WriteMode::Replace, ConsistencyMode::TableSwap)
                .is_ok()
        );
        assert!(
            report
                .require_safe_publication(WriteMode::Append, ConsistencyMode::TableSwap)
                .is_err()
        );
        report.capabilities.database = Some(DatabaseKind::PostgreSql);
        assert!(
            report
                .require_safe_publication(WriteMode::Replace, ConsistencyMode::TableSwap)
                .is_err()
        );
        report.capabilities.database = Some(DatabaseKind::Oracle);
        report.capabilities.non_atomic_replace = false;
        assert!(
            report
                .require_safe_publication(WriteMode::Replace, ConsistencyMode::TableSwap)
                .is_err()
        );
    }
}
