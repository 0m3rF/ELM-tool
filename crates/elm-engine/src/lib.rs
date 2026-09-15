//! Bounded `Source -> Arrow -> Mask/Convert -> Sink` execution.

mod memory;
mod pipeline;
mod spill;

pub use memory::{AdaptiveBatchSizer, MemoryBudget};
pub use pipeline::{EngineObserver, NoopObserver, PipelineStats, TransferEngine};
pub use spill::{SpillFile, SpillStore};
