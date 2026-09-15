//! Authenticated local IPC and persistent job execution.

mod client;
mod paths;
mod runtime;
mod transport;

pub use client::DaemonClient;
pub use paths::RuntimePaths;
pub use runtime::DaemonRuntime;
