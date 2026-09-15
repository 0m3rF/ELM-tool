use std::{future::Future, pin::Pin, sync::Arc};

use tokio::io::{AsyncRead, AsyncWrite};
use tokio_util::sync::CancellationToken;

use elm_core::Result;

use crate::RuntimePaths;

pub trait IpcIo: AsyncRead + AsyncWrite + Send {}
impl<T> IpcIo for T where T: AsyncRead + AsyncWrite + Send {}

pub type BoxedIo = Box<dyn IpcIo + Unpin>;
pub type ConnectionHandler =
    Arc<dyn Fn(BoxedIo) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

#[cfg(windows)]
pub async fn connect(paths: &RuntimePaths) -> Result<BoxedIo> {
    use tokio::net::windows::named_pipe::ClientOptions;

    let mut last_error = None;
    for _ in 0..40 {
        match ClientOptions::new().open(paths.endpoint()) {
            Ok(client) => return Ok(Box::new(client)),
            Err(error) if matches!(error.raw_os_error(), Some(2 | 231)) => {
                last_error = Some(error);
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(last_error
        .unwrap_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "named pipe did not become ready",
            )
        })
        .into())
}

#[cfg(unix)]
pub async fn connect(paths: &RuntimePaths) -> Result<BoxedIo> {
    let stream = tokio::net::UnixStream::connect(paths.endpoint()).await?;
    Ok(Box::new(stream))
}

#[cfg(windows)]
pub async fn serve(
    paths: &RuntimePaths,
    shutdown: CancellationToken,
    handler: ConnectionHandler,
) -> Result<()> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let mut first = true;
    loop {
        let mut options = ServerOptions::new();
        if first {
            options.first_pipe_instance(true);
        }
        let server = options.create(paths.endpoint())?;
        tokio::select! {
            () = shutdown.cancelled() => return Ok(()),
            result = server.connect() => result?,
        }
        first = false;
        let handler = handler.clone();
        tokio::spawn(async move { handler(Box::new(server)).await });
    }
}

#[cfg(unix)]
pub async fn serve(
    paths: &RuntimePaths,
    shutdown: CancellationToken,
    handler: ConnectionHandler,
) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    if paths.endpoint().exists() {
        match tokio::net::UnixStream::connect(paths.endpoint()).await {
            Ok(_) => {
                return Err(elm_core::ElmError::Conflict(
                    "another daemon is already listening".into(),
                ));
            }
            Err(_) => std::fs::remove_file(paths.endpoint())?,
        }
    }
    let listener = tokio::net::UnixListener::bind(paths.endpoint())?;
    std::fs::set_permissions(paths.endpoint(), std::fs::Permissions::from_mode(0o600))?;
    loop {
        let (stream, _) = tokio::select! {
            () = shutdown.cancelled() => break,
            result = listener.accept() => result?,
        };
        let handler = handler.clone();
        tokio::spawn(async move { handler(Box::new(stream)).await });
    }
    match std::fs::remove_file(paths.endpoint()) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}
