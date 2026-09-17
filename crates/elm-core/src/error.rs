use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ElmError>;

/// Machine-readable error categories carried over IPC and persisted with jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Validation,
    Authentication,
    NotFound,
    Conflict,
    Unsupported,
    NativeClientMissing,
    PermissionDenied,
    Connection,
    TypeMapping,
    ResourceExhausted,
    Cancelled,
    Interrupted,
    Io,
    State,
    Internal,
}

/// The only error representation allowed to cross the process boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicError {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
    pub remediation: Option<String>,
}

#[derive(Debug, Error)]
pub enum ElmError {
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("authentication failed")]
    Authentication,
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// The native client library/driver this connector needs isn't installed. Distinct from
    /// `Unsupported` so a client can offer to provision it (`Operation::ProvisionNativeClient`)
    /// instead of just reporting a dead end. Carries a plain message, like its siblings here,
    /// because `PublicError` erases everything but the error code and message across IPC; the
    /// caller already knows which `DatabaseKind` it was testing when it gets this back.
    #[error("native client missing: {0}")]
    NativeClientMissing(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("connection failed: {message}")]
    Connection { message: String, retryable: bool },
    #[error("unsupported or lossy type mapping: {0}")]
    TypeMapping(String),
    #[error("resource budget exceeded: {0}")]
    ResourceExhausted(String),
    #[error("job cancelled")]
    Cancelled,
    #[error("job interrupted")]
    Interrupted,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("state store error: {0}")]
    State(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl ElmError {
    #[must_use]
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::Validation(_) => ErrorCode::Validation,
            Self::Authentication => ErrorCode::Authentication,
            Self::NotFound(_) => ErrorCode::NotFound,
            Self::Conflict(_) => ErrorCode::Conflict,
            Self::Unsupported(_) => ErrorCode::Unsupported,
            Self::NativeClientMissing(_) => ErrorCode::NativeClientMissing,
            Self::PermissionDenied(_) => ErrorCode::PermissionDenied,
            Self::Connection { .. } => ErrorCode::Connection,
            Self::TypeMapping(_) => ErrorCode::TypeMapping,
            Self::ResourceExhausted(_) => ErrorCode::ResourceExhausted,
            Self::Cancelled => ErrorCode::Cancelled,
            Self::Interrupted => ErrorCode::Interrupted,
            Self::Io(_) => ErrorCode::Io,
            Self::State(_) => ErrorCode::State,
            Self::Internal(_) => ErrorCode::Internal,
        }
    }

    #[must_use]
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::Connection {
                retryable: true,
                ..
            } | Self::Interrupted
        )
    }

    /// Redacts credentials and connection details before persistence or IPC.
    #[must_use]
    pub fn to_public(&self) -> PublicError {
        let raw = self.to_string();
        PublicError {
            code: self.code(),
            message: redact_secrets(&raw).into_owned(),
            retryable: self.retryable(),
            remediation: remediation(self).map(ToOwned::to_owned),
        }
    }
}

fn remediation(error: &ElmError) -> Option<&'static str> {
    match error {
        ElmError::NativeClientMissing(_) => Some(
            "Run 'elm native install <oracle|sql-server>' (CLI) or use the Settings screen's \
             native drivers panel (desktop) to install it; ELM asks for confirmation before \
             downloading or installing anything.",
        ),
        ElmError::PermissionDenied(_) => Some(
            "Grant staging-table create/drop privileges or explicitly choose checkpointed consistency.",
        ),
        ElmError::TypeMapping(_) => {
            Some("Add an explicit conversion rule for the affected column.")
        }
        ElmError::ResourceExhausted(_) => Some(
            "Increase the memory budget or lower the target batch size; oversized LOBs require spill support.",
        ),
        _ => None,
    }
}

/// Conservative redaction for URLs, common password fields, and Oracle-style connect strings.
#[must_use]
pub fn redact_secrets(input: &str) -> Cow<'_, str> {
    let mut output = input.to_owned();
    if let Some(scheme) = output.find("://") {
        let credentials_start = scheme + 3;
        let authority_end = output[credentials_start..]
            .find(['/', '?', '#', ' '])
            .map_or(output.len(), |offset| credentials_start + offset);
        if let Some(at_offset) = output[credentials_start..authority_end].rfind('@') {
            let at = credentials_start + at_offset;
            if output[credentials_start..at].contains(':') {
                output.replace_range(credentials_start..at, "[REDACTED]");
            }
        }
    }

    for marker in ["password=", "pwd=", "token=", "secret="] {
        loop {
            let lower = output.to_ascii_lowercase();
            let Some(start) = lower.find(marker) else {
                break;
            };
            let value_start = start + marker.len();
            let value_end = output[value_start..]
                .find([';', '&', ' ', '\n', '\r'])
                .map_or(output.len(), |offset| value_start + offset);
            output.replace_range(value_start..value_end, "[REDACTED]");
            // The replacement no longer contains the marker's value, but the marker remains.
            // Replace the marker too so the scan always makes progress.
            output.replace_range(start..value_start, "credential=");
        }
    }

    if output == input {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(output)
    }
}

#[cfg(test)]
mod tests {
    use super::redact_secrets;

    #[test]
    fn redacts_url_credentials_and_password_fields() {
        let value = "connect postgresql://alice:p@ss@db/prod password=hunter2; token=abc";
        let redacted = redact_secrets(value);
        assert!(!redacted.contains("alice:p@ss"));
        assert!(!redacted.contains("p@ss"));
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("abc"));
    }
}
