use serde::{Deserialize, Serialize};

use crate::{ConsistencyMode, ElmError, Result, WriteMode};

/// Frozen v2 publication behavior. This decision is made before a source is opened for loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationDecision {
    pub create_new_target: bool,
    pub preserve_existing_until_publish: bool,
    pub publish_empty_source: bool,
    pub stage_before_publish: bool,
}

pub fn decide_publication(
    mode: WriteMode,
    consistency: ConsistencyMode,
    target_exists: bool,
) -> Result<PublicationDecision> {
    if consistency == ConsistencyMode::TableSwap && mode != WriteMode::Replace {
        return Err(ElmError::Validation(
            "table-swap requires REPLACE mode".into(),
        ));
    }
    if mode == WriteMode::Fail && target_exists {
        return Err(ElmError::Conflict(
            "destination exists and write mode is fail".into(),
        ));
    }
    Ok(PublicationDecision {
        create_new_target: !target_exists,
        preserve_existing_until_publish: target_exists
            && consistency != ConsistencyMode::Checkpointed,
        publish_empty_source: matches!(mode, WriteMode::Replace | WriteMode::Fail),
        stage_before_publish: consistency != ConsistencyMode::Checkpointed,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResumeGuarantee {
    FileFingerprintBound,
    StableSourceKeyBound,
    RestartFromZero,
}

impl ResumeGuarantee {
    #[must_use]
    pub fn warning(self) -> Option<&'static str> {
        match self {
            Self::StableSourceKeyBound => Some(
                "Database resume is bounded by the captured key range, not a recoverable snapshot; keep the source stable for point-in-time results.",
            ),
            Self::FileFingerprintBound | Self::RestartFromZero => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_mode_stops_before_loading_existing_target() {
        assert!(decide_publication(WriteMode::Fail, ConsistencyMode::Atomic, true).is_err());
    }

    #[test]
    fn replace_publishes_an_empty_source() {
        let decision = decide_publication(WriteMode::Replace, ConsistencyMode::Atomic, true)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(decision.publish_empty_source);
        assert!(decision.preserve_existing_until_publish);
        assert!(decision.stage_before_publish);
    }

    #[test]
    fn table_swap_stages_even_empty_replacements_and_rejects_other_modes() {
        let decision = decide_publication(WriteMode::Replace, ConsistencyMode::TableSwap, true)
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(decision.stage_before_publish);
        assert!(decision.preserve_existing_until_publish);
        assert!(decision.publish_empty_source);
        for mode in [WriteMode::Append, WriteMode::Fail] {
            assert!(decide_publication(mode, ConsistencyMode::TableSwap, false).is_err());
        }
    }
}
