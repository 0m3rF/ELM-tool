use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use elm_core::{ElmError, Result};

const MEMORY_QUANTUM: u64 = 64 * 1024;

#[derive(Debug, Clone)]
pub struct MemoryBudget {
    semaphore: Arc<Semaphore>,
    max_bytes: u64,
}

impl MemoryBudget {
    pub fn new(max_bytes: u64) -> Result<Self> {
        let permit_count = max_bytes.div_ceil(MEMORY_QUANTUM);
        let permits = usize::try_from(permit_count)
            .map_err(|_| ElmError::ResourceExhausted("memory budget is too large".into()))?;
        if permits == 0 || permit_count > u64::from(u32::MAX) {
            return Err(ElmError::Validation(
                "memory budget must be non-zero".into(),
            ));
        }
        Ok(Self {
            semaphore: Arc::new(Semaphore::new(permits)),
            max_bytes,
        })
    }

    #[must_use]
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub async fn acquire(&self, bytes: u64) -> Result<OwnedSemaphorePermit> {
        if bytes > self.max_bytes {
            return Err(ElmError::ResourceExhausted(format!(
                "batch requires {bytes} bytes but the process budget is {} bytes",
                self.max_bytes
            )));
        }
        let permits = u32::try_from(bytes.max(1).div_ceil(MEMORY_QUANTUM))
            .map_err(|_| ElmError::ResourceExhausted("batch permit count overflow".into()))?;
        self.semaphore
            .clone()
            .acquire_many_owned(permits)
            .await
            .map_err(|_| ElmError::Interrupted)
    }
}

/// Converts a byte target into a row hint using a damped observed row width.
#[derive(Debug, Clone)]
pub struct AdaptiveBatchSizer {
    target_bytes: u64,
    estimated_row_bytes: f64,
}

impl AdaptiveBatchSizer {
    #[must_use]
    pub fn new(target_bytes: u64) -> Self {
        Self {
            target_bytes,
            estimated_row_bytes: 1024.0,
        }
    }

    pub fn observe(&mut self, rows: usize, bytes: usize) {
        if rows > 0 {
            let observed = bytes as f64 / rows as f64;
            self.estimated_row_bytes = self.estimated_row_bytes.mul_add(0.75, observed * 0.25);
        }
    }

    #[must_use]
    pub fn target_rows(&self) -> usize {
        ((self.target_bytes as f64 / self.estimated_row_bytes).floor() as usize).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::AdaptiveBatchSizer;

    #[test]
    fn adaptive_sizer_reacts_to_wider_rows() {
        let mut sizer = AdaptiveBatchSizer::new(8 * 1024 * 1024);
        let before = sizer.target_rows();
        sizer.observe(100, 2 * 1024 * 1024);
        assert!(sizer.target_rows() < before);
        assert!(sizer.target_rows() > 0);
    }
}
