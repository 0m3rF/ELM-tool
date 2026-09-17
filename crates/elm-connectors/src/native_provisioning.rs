//! Downloads and installs the native database clients ELM cannot ship for licensing reasons
//! (Oracle Instant Client, Microsoft's ODBC Driver 18 for SQL Server). Every provisioning
//! function here only ever runs from an explicit `Operation::ProvisionNativeClient` request,
//! which the daemon never sends itself: the CLI and desktop are responsible for obtaining the
//! user's confirmation before making that request, so nothing here ever downloads or installs
//! anything the user hasn't been asked about first. Each download is pinned by URL and SHA-256;
//! a hash mismatch fails closed before anything is extracted or executed.
//!
//! Oracle Instant Client's DLLs are extracted directly into the same directory as the running
//! `elm-daemon` executable: Windows checks an application's own directory before `PATH` when
//! resolving an implicit library load, so this needs no environment-variable mutation (this
//! workspace forbids `unsafe` code, which `std::env::set_var` requires) and no elevation, since
//! it never writes outside a location the current user already owns. SQL Server's ODBC driver
//! registers itself with the Windows driver manager and genuinely cannot be "just files in a
//! folder"; installing it runs the vendor MSI elevated through a single Windows UAC prompt.

use elm_core::DatabaseKind;

/// True if "ODBC Driver 18 for SQL Server" is registered with the OS ODBC driver manager. Never
/// downloads or installs anything.
#[cfg(feature = "sql-server")]
#[must_use]
pub fn sql_server_odbc_driver_present() -> bool {
    crate::native_diagnostics::sql_server_manager().is_ok()
}

/// True if the Oracle client library is loadable right now: either the host already has a
/// system install, or a previous [`install::provision_oracle_instant_client`] call already
/// placed it next to this executable. Never downloads or installs anything.
#[cfg(feature = "oracle")]
#[must_use]
pub fn oracle_instant_client_present() -> bool {
    oracle::Version::client().is_ok()
}

/// Whether ELM knows how to provision this database's native client on the current OS. Oracle
/// Instant Client and SQL Server's ODBC driver are implemented for Windows only so far.
#[must_use]
pub fn is_provisionable(kind: DatabaseKind) -> bool {
    cfg!(target_os = "windows") && matches!(kind, DatabaseKind::Oracle | DatabaseKind::SqlServer)
}

#[cfg(all(target_os = "windows", feature = "native-client-provisioning"))]
mod install {
    use std::path::{Path, PathBuf};

    use elm_core::{ElmError, Result};
    use sha2::{Digest, Sha256};

    use super::sql_server_odbc_driver_present;

    const ORACLE_INSTANT_CLIENT_URL: &str = "https://download.oracle.com/otn_software/nt/instantclient/2123000/instantclient-basic-windows.x64-21.23.0.0.0dbru.zip";
    const ORACLE_INSTANT_CLIENT_SHA256: &str =
        "f01253058bf786678326b5a7aee8131bf559f210bba144f425fd8114eefcf0b9";

    const SQL_SERVER_ODBC_DRIVER_URL: &str = "https://download.microsoft.com/download/7bf9fad4-0f21-486d-a750-fc990ded5624/amd64/1033/msodbcsql.msi";
    const SQL_SERVER_ODBC_DRIVER_SHA256: &str =
        "20314529110da3365a252164a657bdc837a18be5839105aa5f5acf0a8d2f4b82";

    fn hex_encode(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// The directory Oracle Instant Client's DLLs are extracted into: the running
    /// `elm-daemon.exe`'s own directory, which Windows always checks before `PATH`.
    fn oracle_target_directory() -> Result<PathBuf> {
        let exe = std::env::current_exe().map_err(ElmError::Io)?;
        exe.parent().map(Path::to_path_buf).ok_or_else(|| {
            ElmError::Internal("could not determine the running executable's directory".into())
        })
    }

    async fn download_and_verify(url: &str, expected_sha256: &str) -> Result<Vec<u8>> {
        let response = reqwest::get(url)
            .await
            .map_err(|error| ElmError::Connection {
                message: format!("native client download failed: {error}"),
                retryable: true,
            })?;
        if !response.status().is_success() {
            return Err(ElmError::Connection {
                message: format!("native client download returned HTTP {}", response.status()),
                retryable: true,
            });
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| ElmError::Connection {
                message: format!("native client download was interrupted: {error}"),
                retryable: true,
            })?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual = hex_encode(&hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected_sha256) {
            return Err(ElmError::Validation(format!(
                "native client download does not match its pinned checksum (expected {expected_sha256}, got {actual}); refusing to install it"
            )));
        }
        Ok(bytes.to_vec())
    }

    /// Downloads, verifies, and extracts Oracle Instant Client Basic directly next to the
    /// running `elm-daemon` executable. No elevation and no environment-variable change is
    /// required: Windows already checks an application's own directory before `PATH` when
    /// resolving `oci.dll`. Fails closed with a clear error if that directory isn't writable
    /// (for example, a per-machine install under `Program Files`) rather than silently trying
    /// something riskier.
    pub async fn provision_oracle_instant_client() -> Result<PathBuf> {
        let destination = oracle_target_directory()?;
        // Fail fast, before spending a ~90 MB download, if we can't actually write here.
        let probe = destination.join(".elm-write-test");
        std::fs::write(&probe, b"").map_err(|_| {
            ElmError::PermissionDenied(format!(
                "cannot write to {} to install Oracle Instant Client; this location needs a \
                 writable, per-user install rather than a protected system directory",
                destination.display()
            ))
        })?;
        let _ = std::fs::remove_file(&probe);

        let bytes =
            download_and_verify(ORACLE_INSTANT_CLIENT_URL, ORACLE_INSTANT_CLIENT_SHA256).await?;
        let destination_clone = destination.clone();
        tokio::task::spawn_blocking(move || extract_instant_client_zip(&bytes, &destination_clone))
            .await
            .map_err(|error| ElmError::Internal(format!("extraction worker stopped: {error}")))??;
        Ok(destination)
    }

    fn extract_instant_client_zip(bytes: &[u8], destination: &Path) -> Result<()> {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|error| {
            ElmError::Validation(format!("native client archive is invalid: {error}"))
        })?;
        // The archive's own top-level folder (e.g. instantclient_21_23/...) is stripped so every
        // file lands directly in `destination` next to elm-daemon.exe.
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).map_err(|error| {
                ElmError::Validation(format!("native client archive entry is invalid: {error}"))
            })?;
            let Some(entry_path) = entry.enclosed_name() else {
                continue;
            };
            let relative: PathBuf = entry_path.components().skip(1).collect();
            if relative.as_os_str().is_empty() || entry.is_dir() {
                continue;
            }
            // Flatten: only the DLLs/executables at the archive's top level are needed next to
            // elm-daemon.exe; the `network` admin-config subfolder is skipped rather than
            // shadowing anything the user already has in that shared directory.
            if relative.components().count() > 1 {
                continue;
            }
            let target = destination.join(&relative);
            let mut out = std::fs::File::create(&target).map_err(ElmError::Io)?;
            std::io::copy(&mut entry, &mut out).map_err(ElmError::Io)?;
        }
        Ok(())
    }

    /// Downloads and verifies the ODBC Driver 18 MSI, then runs it elevated through a single
    /// Windows UAC consent prompt (PowerShell's `Start-Process -Verb RunAs -Wait`, a standard,
    /// fully safe way to elevate one specific action without running the whole daemon as admin)
    /// with the driver's documented silent-install property. Requires the user to have already
    /// confirmed this in the CLI or desktop -- ELM asks before this function is ever called, and
    /// the UAC prompt itself is a second, OS-level consent step that cannot be bypassed or
    /// pre-answered by ELM.
    pub async fn provision_sql_server_odbc_driver(staging_directory: &Path) -> Result<()> {
        let bytes =
            download_and_verify(SQL_SERVER_ODBC_DRIVER_URL, SQL_SERVER_ODBC_DRIVER_SHA256).await?;
        tokio::fs::create_dir_all(staging_directory)
            .await
            .map_err(ElmError::Io)?;
        let msi_path = staging_directory.join("msodbcsql18.msi");
        tokio::fs::write(&msi_path, &bytes)
            .await
            .map_err(ElmError::Io)?;
        run_elevated_odbc_install(&msi_path).await?;
        if !sql_server_odbc_driver_present() {
            return Err(ElmError::Internal(
                "ODBC Driver 18 install finished but the driver is still not registered".into(),
            ));
        }
        Ok(())
    }

    async fn run_elevated_odbc_install(msi_path: &Path) -> Result<()> {
        let msi_argument = format!(
            "/i \"{}\" IACCEPTMSODBCSQLLICENSETERMS=YES /quiet /norestart",
            msi_path.display()
        );
        let powershell_command = format!(
            "$p = Start-Process -FilePath msiexec.exe -ArgumentList '{}' -Verb RunAs -Wait -PassThru; exit $p.ExitCode",
            msi_argument.replace('\'', "''")
        );
        let output = tokio::process::Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                &powershell_command,
            ])
            .output()
            .await
            .map_err(ElmError::Io)?;
        // 1223 = ERROR_CANCELLED: the user declined the UAC prompt.
        if output.status.code() == Some(1223) {
            return Err(ElmError::PermissionDenied(
                "the Windows elevation (UAC) prompt for the ODBC driver install was declined"
                    .into(),
            ));
        }
        if !output.status.success() {
            return Err(ElmError::Internal(format!(
                "ODBC Driver 18 install failed with exit code {:?}",
                output.status.code()
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::hex_encode;

        #[test]
        fn hex_encode_matches_known_vector() {
            assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        }
    }
}

#[cfg(all(target_os = "windows", feature = "native-client-provisioning"))]
pub use install::{provision_oracle_instant_client, provision_sql_server_odbc_driver};
