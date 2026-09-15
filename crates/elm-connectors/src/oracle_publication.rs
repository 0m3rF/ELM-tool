//! Recovery decisions for the explicitly non-atomic Oracle rename swap.
//!
//! Names alone never prove ownership: the journal must retain Oracle OBJECT_IDs.
//! This module performs no DDL; callers must persist intent before each rename.
use elm_core::{ElmError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SwapIdentity {
    pub staged: u64,
    pub original: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SwapObjects {
    pub target: Option<u64>,
    pub staging: Option<u64>,
    pub backup: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SwapAction {
    RenameOriginalToBackup,
    RenameStagingToTarget,
    Published,
}

impl SwapIdentity {
    pub fn next_action(self, objects: SwapObjects) -> Result<SwapAction> {
        if self.staged == self.original || self.staged == 0 || self.original == 0 {
            return Err(ElmError::State(
                "invalid Oracle swap object identities".into(),
            ));
        }
        match objects {
            SwapObjects { target: Some(target), staging: Some(stage), backup: None }
                if target == self.original && stage == self.staged =>
                    Ok(SwapAction::RenameOriginalToBackup),
            SwapObjects { target: None, staging: Some(stage), backup: Some(backup) }
                if stage == self.staged && backup == self.original =>
                    Ok(SwapAction::RenameStagingToTarget),
            SwapObjects { target: Some(target), staging: None, backup: Some(backup) }
                if target == self.staged && backup == self.original =>
                    Ok(SwapAction::Published),
            _ => Err(ElmError::Conflict(
                "Oracle swap objects changed or are missing; preserve all tables and inspect the recovery journal before retrying".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_boundaries_recover_by_object_identity() {
        let identity = SwapIdentity {
            staged: 20,
            original: 10,
        };
        let cases = [
            (
                SwapObjects {
                    target: Some(10),
                    staging: Some(20),
                    backup: None,
                },
                SwapAction::RenameOriginalToBackup,
            ),
            (
                SwapObjects {
                    target: None,
                    staging: Some(20),
                    backup: Some(10),
                },
                SwapAction::RenameStagingToTarget,
            ),
            (
                SwapObjects {
                    target: Some(20),
                    staging: None,
                    backup: Some(10),
                },
                SwapAction::Published,
            ),
        ];
        for (objects, action) in cases {
            assert_eq!(identity.next_action(objects).ok(), Some(action));
        }
    }

    #[test]
    fn unknown_objects_never_authorize_a_rename_or_success() {
        let identity = SwapIdentity {
            staged: 20,
            original: 10,
        };
        let mut accepted = 0;
        for target in [None, Some(10), Some(20), Some(30)] {
            for staging in [None, Some(10), Some(20), Some(30)] {
                for backup in [None, Some(10), Some(20), Some(30)] {
                    accepted += usize::from(
                        identity
                            .next_action(SwapObjects {
                                target,
                                staging,
                                backup,
                            })
                            .is_ok(),
                    );
                }
            }
        }
        assert_eq!(accepted, 3);
        assert!(
            SwapIdentity {
                staged: 10,
                original: 10
            }
            .next_action(SwapObjects {
                target: Some(10),
                staging: Some(10),
                backup: None,
            })
            .is_err()
        );
    }
}
