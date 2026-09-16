use std::{sync::Arc, time::Instant};

use arrow_array::RecordBatch;
use async_trait::async_trait;
use chrono::Utc;
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use elm_core::{
    BatchCheckpoint, ConversionPlan, DataSink, DataSource, ElmError, JobProgress, JobSpec,
    JobState, Result, masking::apply_masks,
};

use crate::MemoryBudget;

struct SourceBatch {
    sequence: u64,
    batch: RecordBatch,
    source_position: serde_json::Value,
    memory: tokio::sync::OwnedSemaphorePermit,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub rows: u64,
    pub bytes: u64,
    pub batches: u64,
}

#[async_trait]
pub trait EngineObserver: Send + Sync {
    async fn progress(&self, event: JobProgress) -> Result<()>;
    async fn checkpoint(&self, checkpoint: &BatchCheckpoint) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct NoopObserver;

#[async_trait]
impl EngineObserver for NoopObserver {
    async fn progress(&self, _event: JobProgress) -> Result<()> {
        Ok(())
    }

    async fn checkpoint(&self, _checkpoint: &BatchCheckpoint) -> Result<()> {
        Ok(())
    }
}

pub struct TransferEngine {
    spec: JobSpec,
    cancellation: CancellationToken,
    observer: Arc<dyn EngineObserver>,
    resume_checkpoint: Option<BatchCheckpoint>,
    extra_warnings: Vec<String>,
}

impl TransferEngine {
    #[must_use]
    pub fn new(
        spec: JobSpec,
        cancellation: CancellationToken,
        observer: Arc<dyn EngineObserver>,
    ) -> Self {
        Self {
            spec,
            cancellation,
            observer,
            resume_checkpoint: None,
            extra_warnings: Vec::new(),
        }
    }

    /// Continues sequence numbers and cumulative progress from a durable sink checkpoint.
    #[must_use]
    pub fn with_resume_checkpoint(mut self, checkpoint: BatchCheckpoint) -> Self {
        self.resume_checkpoint = Some(checkpoint);
        self
    }

    /// Adds warnings determined before preflight (for example, best-effort physical-identity
    /// discovery) to every progress event alongside the sink's own preflight warnings.
    #[must_use]
    pub fn with_extra_warnings(mut self, warnings: Vec<String>) -> Self {
        self.extra_warnings = warnings;
        self
    }

    pub async fn run(
        self,
        mut source: Box<dyn DataSource>,
        mut sink: Box<dyn DataSink>,
    ) -> Result<PipelineStats> {
        self.spec.validate()?;
        let (initial_stats, initial_sequence) = match &self.resume_checkpoint {
            Some(checkpoint) => (
                PipelineStats {
                    rows: checkpoint.rows_committed,
                    bytes: checkpoint.bytes_committed,
                    batches: checkpoint.sequence.checked_add(1).ok_or_else(|| {
                        ElmError::State("checkpoint sequence cannot be resumed".into())
                    })?,
                },
                checkpoint.sequence.checked_add(1).ok_or_else(|| {
                    ElmError::State("checkpoint sequence cannot be resumed".into())
                })?,
            ),
            None => (PipelineStats::default(), 0),
        };
        let started = Instant::now();
        if self.cancellation.is_cancelled() {
            self.emit(
                JobState::Cancelled,
                initial_stats,
                started,
                Some(ElmError::Cancelled.to_public()),
                &[],
            )
            .await?;
            return Err(ElmError::Cancelled);
        }
        self.emit(JobState::Preflighting, initial_stats, started, None, &[])
            .await?;

        let schema = source.schema().await?;
        let conversion_plan = ConversionPlan::new(schema, &self.spec.conversions)?;
        let report = sink
            .preflight(
                conversion_plan.output_schema(),
                self.spec.write_mode,
                self.spec.consistency,
            )
            .await?;
        report.require_safe_publication(self.spec.write_mode, self.spec.consistency)?;
        let mut warnings = report.warnings;
        warnings.extend(self.extra_warnings.iter().cloned());
        sink.begin().await?;
        self.emit(JobState::Running, initial_stats, started, None, &warnings)
            .await?;

        let budget = MemoryBudget::new(self.spec.memory_budget_bytes)?;
        let (source_tx, source_rx) = mpsc::channel(2);
        let (transform_tx, mut transform_rx) = mpsc::channel(2);
        let source_cancel = self.cancellation.clone();
        let target_bytes = usize::try_from(self.spec.batch_target_bytes).map_err(|_| {
            ElmError::Validation("batch target does not fit this platform's address space".into())
        })?;
        let reserve_multiplier = if self.spec.masks.is_empty() && self.spec.conversions.is_empty() {
            1
        } else {
            2
        };

        let source_task = tokio::spawn(async move {
            produce_batches(
                source.as_mut(),
                source_tx,
                budget,
                target_bytes,
                reserve_multiplier,
                initial_sequence,
                source_cancel,
            )
            .await
        });
        let transform_cancel = self.cancellation.clone();
        let rules = self.spec.masks.clone();
        let seed = self.spec.random_seed;
        let transform_task = tokio::spawn(async move {
            transform_batches(
                source_rx,
                transform_tx,
                rules,
                seed,
                conversion_plan,
                transform_cancel,
            )
            .await
        });

        let mut stats = initial_stats;
        let result = loop {
            tokio::select! {
                () = self.cancellation.cancelled() => break Err(ElmError::Cancelled),
                value = transform_rx.recv() => {
                    match value {
                        Some(Ok(item)) => {
                            let batch_bytes = u64::try_from(item.batch.get_array_memory_size())
                                .unwrap_or(u64::MAX);
                            let batch_rows = u64::try_from(item.batch.num_rows()).unwrap_or(u64::MAX);
                            if let Err(error) = sink.write_batch(&item.batch).await {
                                break Err(error);
                            }
                            stats.rows = stats.rows.saturating_add(batch_rows);
                            stats.bytes = stats.bytes.saturating_add(batch_bytes);
                            stats.batches = stats.batches.saturating_add(1);
                            let checkpoint = BatchCheckpoint {
                                sequence: item.sequence,
                                source_position: item.source_position,
                                rows_committed: stats.rows,
                                bytes_committed: stats.bytes,
                                source_fingerprint: None,
                            };
                            if let Err(error) = sink.commit_checkpoint(&checkpoint).await {
                                break Err(error);
                            }
                            if let Err(error) = self.observer.checkpoint(&checkpoint).await {
                                break Err(error);
                            }
                            if let Err(error) = self.emit(JobState::Running, stats, started, None, &warnings).await {
                                break Err(error);
                            }
                            drop(item.memory);
                        }
                        Some(Err(error)) => break Err(error),
                        None => break Ok(()),
                    }
                }
            }
        };

        if result.is_err() {
            self.cancellation.cancel();
        }
        drop(transform_rx);
        join_stage(source_task).await?;
        join_stage(transform_task).await?;

        if let Err(error) = result {
            let _abort_result = sink.abort().await;
            let terminal = if matches!(error, ElmError::Cancelled) {
                JobState::Cancelled
            } else {
                JobState::Failed
            };
            self.emit(terminal, stats, started, Some(error.to_public()), &warnings)
                .await?;
            return Err(error);
        }

        if self.cancellation.is_cancelled() {
            let error = ElmError::Cancelled;
            let _abort_result = sink.abort().await;
            self.emit(
                JobState::Cancelled,
                stats,
                started,
                Some(error.to_public()),
                &warnings,
            )
            .await?;
            return Err(error);
        }

        self.emit(JobState::Publishing, stats, started, None, &warnings)
            .await?;
        if let Err(error) = sink.publish().await {
            let _abort_result = sink.abort().await;
            self.emit(
                JobState::Failed,
                stats,
                started,
                Some(error.to_public()),
                &warnings,
            )
            .await?;
            return Err(error);
        }
        self.emit(JobState::Succeeded, stats, started, None, &warnings)
            .await?;
        Ok(stats)
    }

    async fn emit(
        &self,
        state: JobState,
        stats: PipelineStats,
        started: Instant,
        error: Option<elm_core::PublicError>,
        warnings: &[String],
    ) -> Result<()> {
        let elapsed = started.elapsed();
        let seconds = elapsed.as_secs_f64();
        self.observer
            .progress(JobProgress {
                job_id: self.spec.id,
                state,
                rows: stats.rows,
                bytes: stats.bytes,
                batches: stats.batches,
                rows_per_second: if seconds > 0.0 {
                    stats.rows as f64 / seconds
                } else {
                    0.0
                },
                bytes_per_second: if seconds > 0.0 {
                    stats.bytes as f64 / seconds
                } else {
                    0.0
                },
                elapsed,
                eta: None,
                warnings: warnings.to_vec(),
                error,
                occurred_at: Utc::now(),
            })
            .await
    }
}

async fn produce_batches(
    source: &mut dyn DataSource,
    tx: mpsc::Sender<Result<SourceBatch>>,
    budget: MemoryBudget,
    target_bytes: usize,
    reserve_multiplier: u64,
    initial_sequence: u64,
    cancellation: CancellationToken,
) -> Result<()> {
    let mut sequence = initial_sequence;
    loop {
        let next = tokio::select! {
            () = cancellation.cancelled() => return Ok(()),
            value = source.next_batch(target_bytes) => value,
        };
        let batch = match next {
            Ok(Some(batch)) => batch,
            Ok(None) => return Ok(()),
            Err(error) => {
                let _send_result = tx.send(Err(error)).await;
                return Ok(());
            }
        };
        let bytes = u64::try_from(batch.get_array_memory_size())
            .unwrap_or(u64::MAX)
            .saturating_mul(reserve_multiplier);
        let memory = match budget.acquire(bytes).await {
            Ok(permit) => permit,
            Err(error) => {
                let _send_result = tx.send(Err(error)).await;
                return Ok(());
            }
        };
        let source_position = match source.checkpoint().await {
            Ok(position) => position,
            Err(error) => {
                let _send_result = tx.send(Err(error)).await;
                return Ok(());
            }
        };
        let item = SourceBatch {
            sequence,
            batch,
            source_position,
            memory,
        };
        tokio::select! {
            () = cancellation.cancelled() => return Ok(()),
            send_result = tx.send(Ok(item)) => {
                if send_result.is_err() {
                    return Ok(());
                }
            }
        }
        sequence = sequence.saturating_add(1);
    }
}

async fn transform_batches(
    mut rx: mpsc::Receiver<Result<SourceBatch>>,
    tx: mpsc::Sender<Result<SourceBatch>>,
    rules: Vec<elm_core::MaskRule>,
    seed: [u8; 32],
    conversion_plan: ConversionPlan,
    cancellation: CancellationToken,
) -> Result<()> {
    loop {
        let item = tokio::select! {
            () = cancellation.cancelled() => return Ok(()),
            item = rx.recv() => item,
        };
        let Some(item) = item else {
            return Ok(());
        };
        let transformed = item.and_then(|mut item| {
            item.batch = apply_masks(&item.batch, &rules, &seed, item.sequence)?;
            item.batch = conversion_plan.apply(&item.batch)?;
            Ok(item)
        });
        if tx.send(transformed).await.is_err() {
            return Ok(());
        }
    }
}

async fn join_stage(handle: JoinHandle<Result<()>>) -> Result<()> {
    handle
        .await
        .map_err(|error| ElmError::Internal(format!("pipeline task failed: {error}")))?
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow_array::{Int32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use elm_core::{
        ConnectorCapabilities, ConsistencyMode, ConversionRule, DataSink, DataSource, FileFormat,
        Identifier, PreflightReport, SinkSpec, SourceSpec, WriteMode,
    };

    struct Source {
        batch: Option<RecordBatch>,
    }

    #[async_trait]
    impl DataSource for Source {
        async fn schema(&mut self) -> Result<Arc<Schema>> {
            Ok(self
                .batch
                .as_ref()
                .map_or_else(|| Arc::new(Schema::empty()), RecordBatch::schema))
        }

        async fn next_batch(&mut self, _target_bytes: usize) -> Result<Option<RecordBatch>> {
            Ok(self.batch.take())
        }

        async fn checkpoint(&mut self) -> Result<serde_json::Value> {
            Ok(serde_json::json!({"row": 3}))
        }
    }

    struct Sink {
        rows: Arc<Mutex<usize>>,
        published: Arc<Mutex<bool>>,
    }

    #[async_trait]
    impl DataSink for Sink {
        async fn preflight(
            &mut self,
            _schema: Arc<Schema>,
            _mode: WriteMode,
            _consistency: ConsistencyMode,
        ) -> Result<PreflightReport> {
            Ok(PreflightReport {
                capabilities: ConnectorCapabilities {
                    database: None,
                    native_bulk_read: false,
                    native_bulk_write: false,
                    atomic_append: true,
                    atomic_replace: true,
                    non_atomic_replace: false,
                    checkpointed_write: true,
                    resumable_keyset_read: true,
                    supports_lob_spill: true,
                },
                staging_relation: None,
                warnings: vec![],
            })
        }

        async fn begin(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
            *self
                .rows
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) += batch.num_rows();
            Ok(())
        }

        async fn commit_checkpoint(&mut self, _checkpoint: &BatchCheckpoint) -> Result<()> {
            Ok(())
        }

        async fn publish(&mut self) -> Result<()> {
            *self
                .published
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            Ok(())
        }

        async fn abort(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn transfers_and_publishes_an_empty_safe_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))])
            .unwrap_or_else(|error| panic!("{error}"));
        let source = Box::new(Source { batch: Some(batch) });
        let rows = Arc::new(Mutex::new(0));
        let published = Arc::new(Mutex::new(false));
        let sink = Box::new(Sink {
            rows: rows.clone(),
            published: published.clone(),
        });
        let spec = JobSpec::new(
            SourceSpec::File {
                path: "input.csv".into(),
                format: FileFormat::Csv,
            },
            SinkSpec::File {
                path: "output.csv".into(),
                format: FileFormat::Csv,
            },
        );
        let engine = TransferEngine::new(spec, CancellationToken::new(), Arc::new(NoopObserver));
        let stats = engine
            .run(source, sink)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(stats.rows, 3);
        assert_eq!(
            *rows
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            3
        );
        assert!(
            *published
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        );
    }

    #[tokio::test]
    async fn executes_preflighted_conversion_rules() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
            .unwrap_or_else(|error| panic!("{error}"));
        let source = Box::new(Source { batch: Some(batch) });
        let sink = Box::new(Sink {
            rows: Arc::new(Mutex::new(0)),
            published: Arc::new(Mutex::new(false)),
        });
        let mut spec = JobSpec::new(
            SourceSpec::File {
                path: "input.csv".into(),
                format: FileFormat::Csv,
            },
            SinkSpec::File {
                path: "output.csv".into(),
                format: FileFormat::Csv,
            },
        );
        spec.conversions.push(ConversionRule {
            column: Identifier::new("id").unwrap_or_else(|error| panic!("{error}")),
            target_type: "Int64".into(),
            allow_lossy: false,
        });
        let stats = TransferEngine::new(spec, CancellationToken::new(), Arc::new(NoopObserver))
            .run(source, sink)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(stats.rows, 3);
    }
}
