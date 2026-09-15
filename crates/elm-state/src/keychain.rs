use secrecy::{ExposeSecret, SecretString};

use elm_core::{ElmError, Result};

pub trait CredentialVault: Send + Sync {
    fn put(&self, reference: &str, secret: &SecretString) -> Result<()>;
    fn get(&self, reference: &str) -> Result<SecretString>;
    fn remove(&self, reference: &str) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct KeyringVault {
    service: String,
}

impl Default for KeyringVault {
    fn default() -> Self {
        Self {
            service: "dev.elm-tool.v2".into(),
        }
    }
}

impl KeyringVault {
    #[must_use]
    pub fn with_service(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
        }
    }

    fn entry(&self, reference: &str) -> Result<keyring::Entry> {
        keyring::Entry::new(&self.service, reference)
            .map_err(|error| ElmError::State(format!("cannot access OS keychain: {error}")))
    }
}

impl CredentialVault for KeyringVault {
    fn put(&self, reference: &str, secret: &SecretString) -> Result<()> {
        if reference.trim().is_empty() {
            return Err(ElmError::Validation(
                "credential reference may not be empty".into(),
            ));
        }
        self.entry(reference)?
            .set_password(secret.expose_secret())
            .map_err(|error| {
                ElmError::State(format!("cannot write OS keychain credential: {error}"))
            })
    }

    fn get(&self, reference: &str) -> Result<SecretString> {
        self.entry(reference)?
            .get_password()
            .map(SecretString::from)
            .map_err(|error| {
                ElmError::State(format!("cannot read OS keychain credential: {error}"))
            })
    }

    fn remove(&self, reference: &str) -> Result<()> {
        self.entry(reference)?.delete_credential().map_err(|error| {
            ElmError::State(format!("cannot remove OS keychain credential: {error}"))
        })
    }
}
