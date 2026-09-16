use std::str::FromStr;

use chrono::Utc;
use elm_core::{
    DatabaseKind, Environment, EnvironmentId, JobId, JobProgress, JobSpec, MaskRule, MaskRuleId,
    RuntimeSettings,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, RuntimePaths};
use elm_state::{CredentialVault, KeyringVault};
use secrecy::SecretString;
use tauri::{State, ipc::Channel};

struct DesktopState {
    client: DaemonClient,
}

fn update_connection_options(options: &mut serde_json::Value, kind: DatabaseKind, ssl_mode: &str) {
    if !options.is_object() {
        *options = serde_json::json!({});
    }
    if let Some(values) = options.as_object_mut() {
        values.insert("ssl_mode".into(), ssl_mode.into());
        values.remove("root_certificate");
        if kind != DatabaseKind::Oracle {
            values.remove("oracle_call_timeout_ms");
        }
    }
}

#[cfg(test)]
mod connection_options_tests {
    use super::*;

    #[test]
    fn editing_preserves_advanced_options_and_clears_inapplicable_fields() {
        let mut options = serde_json::json!({
            "oracle_call_timeout_ms": 120000,
            "root_certificate": "old.pem",
            "custom_option": true
        });
        update_connection_options(&mut options, DatabaseKind::Oracle, "disable");
        assert_eq!(options["oracle_call_timeout_ms"], 120000);
        assert_eq!(options["ssl_mode"], "disable");
        assert_eq!(options["custom_option"], true);
        assert!(options.get("root_certificate").is_none());
        update_connection_options(&mut options, DatabaseKind::PostgreSql, "require");
        assert!(options.get("oracle_call_timeout_ms").is_none());
    }
}

#[derive(serde::Deserialize)]
struct EnvironmentDraft {
    id: Option<String>,
    name: String,
    kind: DatabaseKind,
    host: String,
    port: u16,
    database: String,
    username: String,
    password: Option<String>,
    ssl_mode: Option<String>,
    root_certificate: Option<String>,
}

#[tauri::command]
async fn list_environments(state: State<'_, DesktopState>) -> Result<Vec<Environment>, String> {
    match state
        .client
        .request(Operation::EnvironmentList)
        .await
        .map_err(public_message)?
    {
        Response::Environments(value) => Ok(value),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn save_environment(
    state: State<'_, DesktopState>,
    draft: EnvironmentDraft,
) -> Result<Environment, String> {
    let now = Utc::now();
    let (mut environment, operation_is_add) = match draft.id.as_deref() {
        Some(id) => {
            let id = EnvironmentId::from_str(id).map_err(|_| "invalid environment id")?;
            (find_environment(&state.client, id).await?, false)
        }
        None => {
            let id = EnvironmentId::new();
            (
                Environment {
                    id,
                    name: String::new(),
                    kind: draft.kind,
                    host: String::new(),
                    port: draft.port,
                    database: String::new(),
                    username: String::new(),
                    credential_ref: format!("environment-{id}"),
                    options: serde_json::json!({ "ssl_mode": "require" }),
                    created_at: now,
                    updated_at: now,
                },
                true,
            )
        }
    };
    environment.name = draft.name;
    environment.kind = draft.kind;
    environment.host = draft.host;
    environment.port = draft.port;
    environment.database = draft.database;
    environment.username = draft.username;
    update_connection_options(
        &mut environment.options,
        environment.kind,
        draft.ssl_mode.as_deref().unwrap_or("require"),
    );
    if matches!(
        environment.kind,
        DatabaseKind::PostgreSql | DatabaseKind::MySql
    ) && let Some(path) = draft
        .root_certificate
        .filter(|path| !path.trim().is_empty())
    {
        environment.options["root_certificate"] = serde_json::Value::String(path);
    }
    environment.updated_at = now;
    environment.validate().map_err(public_message)?;

    let password = draft
        .password
        .filter(|value| !value.is_empty())
        .map(SecretString::from);
    if operation_is_add && password.is_none() {
        return Err("a password is required for a new connection".into());
    }
    let vault = KeyringVault::default();
    let previous = if password.is_some() {
        vault.get(&environment.credential_ref).ok()
    } else {
        None
    };
    if let Some(password) = &password {
        vault
            .put(&environment.credential_ref, password)
            .map_err(public_message)?;
    }
    let credential_ref = environment.credential_ref.clone();
    let operation = if operation_is_add {
        Operation::EnvironmentAdd(environment)
    } else {
        Operation::EnvironmentEdit(environment)
    };
    match state.client.request(operation).await {
        Ok(Response::Environment(value)) => Ok(value),
        Ok(_) => {
            if password.is_some() {
                rollback_credential(&vault, &credential_ref, previous.as_ref());
            }
            Err("daemon returned an unexpected response".into())
        }
        Err(error) => {
            if password.is_some() {
                rollback_credential(&vault, &credential_ref, previous.as_ref());
            }
            Err(public_message(error))
        }
    }
}

#[tauri::command]
async fn test_environment(state: State<'_, DesktopState>, id: String) -> Result<(), String> {
    let id = EnvironmentId::from_str(&id).map_err(|_| "invalid environment id")?;
    match state
        .client
        .request(Operation::EnvironmentTest(id))
        .await
        .map_err(public_message)?
    {
        Response::Ack => Ok(()),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn remove_environment(state: State<'_, DesktopState>, id: String) -> Result<(), String> {
    let id = EnvironmentId::from_str(&id).map_err(|_| "invalid environment id")?;
    let environment = find_environment(&state.client, id).await?;
    let vault = KeyringVault::default();
    let previous = vault
        .get(&environment.credential_ref)
        .map_err(public_message)?;
    vault
        .remove(&environment.credential_ref)
        .map_err(public_message)?;
    match state.client.request(Operation::EnvironmentRemove(id)).await {
        Ok(Response::Ack) => Ok(()),
        Ok(_) => {
            let _restore_result = vault.put(&environment.credential_ref, &previous);
            Err("daemon returned an unexpected response".into())
        }
        Err(error) => {
            let _restore_result = vault.put(&environment.credential_ref, &previous);
            Err(public_message(error))
        }
    }
}

#[tauri::command]
async fn list_masks(state: State<'_, DesktopState>) -> Result<Vec<MaskRule>, String> {
    match state
        .client
        .request(Operation::MaskList)
        .await
        .map_err(public_message)?
    {
        Response::Masks(value) => Ok(value),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn save_mask(state: State<'_, DesktopState>, rule: MaskRule) -> Result<MaskRule, String> {
    match state
        .client
        .request(Operation::MaskEdit(rule))
        .await
        .map_err(public_message)?
    {
        Response::Masks(mut values) => values
            .pop()
            .ok_or_else(|| "daemon returned no saved masking rule".into()),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn remove_mask(state: State<'_, DesktopState>, id: String) -> Result<(), String> {
    let id = MaskRuleId::from_str(&id).map_err(|_| "invalid mask rule id")?;
    match state
        .client
        .request(Operation::MaskRemove(id))
        .await
        .map_err(public_message)?
    {
        Response::Ack => Ok(()),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn test_mask(
    state: State<'_, DesktopState>,
    rule: MaskRule,
    value: String,
) -> Result<String, String> {
    match state
        .client
        .request(Operation::MaskTest { rule, value })
        .await
        .map_err(public_message)?
    {
        Response::MaskedValue(value) => Ok(value),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn get_settings(state: State<'_, DesktopState>) -> Result<RuntimeSettings, String> {
    match state
        .client
        .request(Operation::SettingsGet)
        .await
        .map_err(public_message)?
    {
        Response::Settings(settings) => Ok(settings),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn save_settings(
    state: State<'_, DesktopState>,
    settings: RuntimeSettings,
) -> Result<RuntimeSettings, String> {
    settings.validate().map_err(public_message)?;
    match state
        .client
        .request(Operation::SettingsSet(settings))
        .await
        .map_err(public_message)?
    {
        Response::Settings(settings) => Ok(settings),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn preview_transfer(
    state: State<'_, DesktopState>,
    spec: JobSpec,
) -> Result<elm_core::TransferPreview, String> {
    match state
        .client
        .request(Operation::JobPreview(spec))
        .await
        .map_err(public_message)?
    {
        Response::Preview(value) => Ok(value),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn submit_job(
    state: State<'_, DesktopState>,
    spec: JobSpec,
) -> Result<elm_core::JobRecord, String> {
    match state
        .client
        .request(Operation::JobSubmit(spec))
        .await
        .map_err(public_message)?
    {
        Response::Job(value) => Ok(value),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn list_jobs(state: State<'_, DesktopState>) -> Result<Vec<elm_core::JobRecord>, String> {
    match state
        .client
        .request(Operation::JobList)
        .await
        .map_err(public_message)?
    {
        Response::Jobs(value) => Ok(value),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

#[tauri::command]
async fn job_action(
    state: State<'_, DesktopState>,
    id: String,
    action: String,
) -> Result<(), String> {
    let id = JobId::from_str(&id).map_err(|_| "invalid job id".to_owned())?;
    let operation = match action.as_str() {
        "cancel" => Operation::JobCancel(id),
        "resume" => Operation::JobResume(id),
        "delete" => Operation::JobDelete(id),
        _ => return Err("unsupported job action".into()),
    };
    state
        .client
        .request(operation)
        .await
        .map_err(public_message)?;
    Ok(())
}

#[tauri::command]
async fn watch_job(
    state: State<'_, DesktopState>,
    id: String,
    events: Channel<JobProgress>,
) -> Result<(), String> {
    let id = JobId::from_str(&id).map_err(|_| "invalid job id".to_owned())?;
    let mut stream = state
        .client
        .watch(Operation::JobWatch(id))
        .await
        .map_err(public_message)?;
    while let Some(response) = stream.recv().await {
        let progress = match response.map_err(public_message)? {
            Response::Job(job) => job.progress,
            Response::Event(progress) => progress,
            _ => continue,
        };
        let terminal =
            progress.state.is_terminal() || progress.state == elm_core::JobState::Interrupted;
        events.send(progress).map_err(|error| error.to_string())?;
        if terminal {
            break;
        }
    }
    Ok(())
}

fn public_message(error: elm_core::ElmError) -> String {
    error.to_public().message
}

async fn find_environment(client: &DaemonClient, id: EnvironmentId) -> Result<Environment, String> {
    match client
        .request(Operation::EnvironmentList)
        .await
        .map_err(public_message)?
    {
        Response::Environments(environments) => environments
            .into_iter()
            .find(|environment| environment.id == id)
            .ok_or_else(|| format!("environment {id} was not found")),
        _ => Err("daemon returned an unexpected response".into()),
    }
}

fn rollback_credential(vault: &KeyringVault, reference: &str, previous: Option<&SecretString>) {
    match previous {
        Some(previous) => {
            let _restore_result = vault.put(reference, previous);
        }
        None => {
            let _remove_result = vault.remove(reference);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::async_runtime::block_on(async {
        let paths = RuntimePaths::discover().unwrap_or_else(|error| panic!("{error}"));
        let client = DaemonClient::connect_or_start(paths)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        tauri::Builder::default()
            .manage(DesktopState { client })
            .invoke_handler(tauri::generate_handler![
                list_environments,
                save_environment,
                test_environment,
                remove_environment,
                list_masks,
                save_mask,
                remove_mask,
                test_mask,
                get_settings,
                save_settings,
                preview_transfer,
                submit_job,
                list_jobs,
                job_action,
                watch_job,
            ])
            .run(tauri::generate_context!())
            .unwrap_or_else(|error| panic!("error while running ELM desktop: {error}"));
    });
}
