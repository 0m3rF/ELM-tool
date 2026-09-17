use std::{process::Stdio, time::Duration};

use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    sync::mpsc,
    time::sleep,
};

use elm_core::{
    ElmError, Result,
    protocol::{Operation, RequestEnvelope, Response, ResponseEnvelope},
};

use crate::{RuntimePaths, transport};

#[derive(Debug, Clone)]
pub struct DaemonClient {
    paths: RuntimePaths,
    auth_token: String,
}

impl DaemonClient {
    pub fn connect(paths: RuntimePaths) -> Result<Self> {
        let auth_token = paths.load_token()?;
        Ok(Self { paths, auth_token })
    }

    pub async fn connect_or_start(paths: RuntimePaths) -> Result<Self> {
        if let Ok(client) = Self::connect(paths.clone())
            && client.request(Operation::DaemonStatus).await.is_ok()
        {
            return Ok(client);
        }
        start_daemon_process(&paths)?;
        let mut last_error = None;
        for _ in 0..40 {
            sleep(Duration::from_millis(50)).await;
            match Self::connect(paths.clone()) {
                Ok(client) => match client.request(Operation::DaemonStatus).await {
                    Ok(_) => return Ok(client),
                    Err(error) => last_error = Some(error),
                },
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| ElmError::Connection {
            message: "daemon did not become ready".into(),
            retryable: true,
        }))
    }

    pub async fn request(&self, operation: Operation) -> Result<Response> {
        let stream = transport::connect(&self.paths).await?;
        let (reader, mut writer) = tokio::io::split(stream);
        let request = RequestEnvelope::new(self.auth_token.clone(), operation);
        let mut payload = serde_json::to_vec(&request)
            .map_err(|error| ElmError::Internal(format!("cannot encode IPC request: {error}")))?;
        payload.push(b'\n');
        writer.write_all(&payload).await?;
        writer.shutdown().await?;
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Err(ElmError::Connection {
                message: "daemon closed the connection without a response".into(),
                retryable: true,
            });
        }
        decode_response(&line)
    }

    pub async fn watch(&self, operation: Operation) -> Result<mpsc::Receiver<Result<Response>>> {
        if !matches!(operation, Operation::JobWatch(_)) {
            return Err(ElmError::Validation(
                "streaming IPC is available only for job.watch".into(),
            ));
        }
        let stream = transport::connect(&self.paths).await?;
        let (reader, mut writer) = tokio::io::split(stream);
        let request = RequestEnvelope::new(self.auth_token.clone(), operation);
        let mut payload = serde_json::to_vec(&request)
            .map_err(|error| ElmError::Internal(format!("cannot encode IPC request: {error}")))?;
        payload.push(b'\n');
        writer.write_all(&payload).await?;
        writer.shutdown().await?;
        let (tx, rx) = mpsc::channel(16);
        tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if tx.send(decode_response(&line)).await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let _send_result = tx.send(Err(error.into())).await;
                        break;
                    }
                }
            }
        });
        Ok(rx)
    }

    #[must_use]
    pub fn paths(&self) -> &RuntimePaths {
        &self.paths
    }
}

fn decode_response(line: &str) -> Result<Response> {
    let envelope: ResponseEnvelope = serde_json::from_str(line)
        .map_err(|error| ElmError::Internal(format!("invalid IPC response: {error}")))?;
    envelope.result.map_err(|error| match error.code {
        elm_core::ErrorCode::Validation => ElmError::Validation(error.message),
        elm_core::ErrorCode::Authentication => ElmError::Authentication,
        elm_core::ErrorCode::NotFound => ElmError::NotFound(error.message),
        elm_core::ErrorCode::Conflict => ElmError::Conflict(error.message),
        elm_core::ErrorCode::Unsupported => ElmError::Unsupported(error.message),
        elm_core::ErrorCode::NativeClientMissing => ElmError::NativeClientMissing(error.message),
        elm_core::ErrorCode::PermissionDenied => ElmError::PermissionDenied(error.message),
        elm_core::ErrorCode::Connection => ElmError::Connection {
            message: error.message,
            retryable: error.retryable,
        },
        elm_core::ErrorCode::TypeMapping => ElmError::TypeMapping(error.message),
        elm_core::ErrorCode::ResourceExhausted => ElmError::ResourceExhausted(error.message),
        elm_core::ErrorCode::Cancelled => ElmError::Cancelled,
        elm_core::ErrorCode::Interrupted => ElmError::Interrupted,
        elm_core::ErrorCode::Io => ElmError::Io(std::io::Error::other(error.message)),
        elm_core::ErrorCode::State => ElmError::State(error.message),
        elm_core::ErrorCode::Internal => ElmError::Internal(error.message),
    })
}

fn start_daemon_process(paths: &RuntimePaths) -> Result<()> {
    let current = std::env::current_exe()?;
    let file_name = if cfg!(windows) {
        "elm-daemon.exe"
    } else {
        "elm-daemon"
    };
    let executable = current.with_file_name(file_name);
    let mut command = std::process::Command::new(executable);
    command
        .arg("--data-directory")
        .arg(&paths.data_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    command.spawn()?;
    Ok(())
}
