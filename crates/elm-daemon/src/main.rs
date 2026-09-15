use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
};

use clap::Parser;
use fs2::FileExt;

use elm_core::{ElmError, Result};
use elm_daemon::{DaemonRuntime, RuntimePaths};

#[derive(Debug, Parser)]
#[command(name = "elm-daemon", version, about = "ELM Tool per-user job daemon")]
struct Arguments {
    #[arg(long)]
    data_directory: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("elm_daemon=info")),
        )
        .json()
        .init();
    let arguments = Arguments::parse();
    let paths = match arguments.data_directory {
        Some(path) => RuntimePaths::for_data_directory(path)?,
        None => RuntimePaths::discover()?,
    };
    let _lock = acquire_instance_lock(&paths.lock)?;
    DaemonRuntime::open(paths)?.serve().await?;
    Ok(())
}

fn acquire_instance_lock(path: &std::path::Path) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    file.try_lock_exclusive()
        .map_err(|_| ElmError::Conflict("an ELM daemon is already running for this user".into()))?;
    Ok(file)
}
