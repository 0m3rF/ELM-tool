//! Per-user durable state. Secrets are intentionally excluded from SQLite.

mod keychain;
mod store;

pub use keychain::{CredentialVault, KeyringVault};
pub use store::{EventRecord, StateStore, default_data_directory};
