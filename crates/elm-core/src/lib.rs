//! Stable domain contracts shared by every ELM Tool interface.

pub mod connector;
pub mod conversion;
pub mod error;
pub mod masking;
pub mod protocol;
pub mod semantics;
pub mod types;

pub use connector::{
    BatchCheckpoint, ConnectorCapabilities, DataSink, DataSource, PreflightReport,
};
pub use conversion::ConversionPlan;
pub use error::{ElmError, ErrorCode, PublicError, Result};
pub use types::*;
