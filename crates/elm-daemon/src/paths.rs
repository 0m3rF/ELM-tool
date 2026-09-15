use std::{
    fs,
    path::{Path, PathBuf},
};

use directories::ProjectDirs;
use uuid::Uuid;

use elm_core::{ElmError, JobId, Result};

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub data_directory: PathBuf,
    pub database: PathBuf,
    pub auth_token: PathBuf,
    pub lock: PathBuf,
    #[cfg(unix)]
    pub socket: PathBuf,
    #[cfg(windows)]
    pub pipe: String,
}

impl RuntimePaths {
    pub fn discover() -> Result<Self> {
        let data_directory = ProjectDirs::from("dev", "ELM Tool", "ELM Tool")
            .map(|directories| directories.data_local_dir().to_path_buf())
            .ok_or_else(|| {
                ElmError::State("operating system did not provide a per-user data directory".into())
            })?;
        Self::for_data_directory(data_directory)
    }

    pub fn for_data_directory(data_directory: impl Into<PathBuf>) -> Result<Self> {
        let data_directory = data_directory.into();
        fs::create_dir_all(&data_directory)?;
        let database = data_directory.join("elm-v2.sqlite3");
        let auth_token = data_directory.join("daemon.token");
        let lock = data_directory.join("daemon.lock");
        #[cfg(unix)]
        let socket = data_directory.join("daemon.sock");
        #[cfg(windows)]
        let pipe = {
            let digest = blake3::hash(data_directory.to_string_lossy().as_bytes());
            format!(r"\\.\pipe\elm-tool-v2-{}", &digest.to_hex()[..16])
        };
        Ok(Self {
            data_directory,
            database,
            auth_token,
            lock,
            #[cfg(unix)]
            socket,
            #[cfg(windows)]
            pipe,
        })
    }

    pub fn load_token(&self) -> Result<String> {
        let token = fs::read_to_string(&self.auth_token)?;
        let token = token.trim().to_owned();
        if token.len() < 32 {
            return Err(ElmError::Authentication);
        }
        Ok(token)
    }

    pub fn load_or_create_token(&self) -> Result<String> {
        match self.load_token() {
            Ok(token) => Ok(token),
            Err(ElmError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                create_token_file(&self.auth_token)
            }
            Err(error) => Err(error),
        }
    }

    #[must_use]
    pub fn job_staging_directory(&self, job_id: JobId) -> PathBuf {
        self.data_directory.join("staging").join(job_id.to_string())
    }

    #[cfg(windows)]
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.pipe
    }

    #[cfg(unix)]
    #[must_use]
    pub fn endpoint(&self) -> &Path {
        &self.socket
    }
}

fn create_token_file(path: &Path) -> Result<String> {
    // Two independent UUIDv4 values provide 244 random bits after fixed version/variant bits.
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    let temporary = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
    fs::write(&temporary, token.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
    }
    fs::rename(&temporary, path)?;
    Ok(token)
}
