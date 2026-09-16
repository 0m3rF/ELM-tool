use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Environment, EnvironmentId, JobId, JobProgress, JobRecord, JobSpec, MaskRule, MaskRuleId,
    PublicError, RuntimeSettings, TransferPreview,
};

pub const IPC_PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Serialize, Deserialize)]
pub struct RequestEnvelope {
    pub version: u16,
    pub request_id: Uuid,
    pub auth_token: String,
    pub operation: Operation,
}

impl RequestEnvelope {
    #[must_use]
    pub fn new(auth_token: String, operation: Operation) -> Self {
        Self {
            version: IPC_PROTOCOL_VERSION,
            request_id: Uuid::new_v4(),
            auth_token,
            operation,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum Operation {
    #[serde(rename = "daemon.status")]
    DaemonStatus,
    #[serde(rename = "daemon.stop")]
    DaemonStop,
    #[serde(rename = "environment.add")]
    EnvironmentAdd(Environment),
    #[serde(rename = "environment.edit")]
    EnvironmentEdit(Environment),
    #[serde(rename = "environment.list")]
    EnvironmentList,
    #[serde(rename = "environment.test")]
    EnvironmentTest(EnvironmentId),
    #[serde(rename = "environment.remove")]
    EnvironmentRemove(EnvironmentId),
    #[serde(rename = "mask.add")]
    MaskAdd(MaskRule),
    #[serde(rename = "mask.edit")]
    MaskEdit(MaskRule),
    #[serde(rename = "mask.list")]
    MaskList,
    #[serde(rename = "mask.remove")]
    MaskRemove(MaskRuleId),
    #[serde(rename = "mask.test")]
    MaskTest { rule: MaskRule, value: String },
    #[serde(rename = "settings.get")]
    SettingsGet,
    #[serde(rename = "settings.set")]
    SettingsSet(RuntimeSettings),
    #[serde(rename = "job.preview")]
    JobPreview(JobSpec),
    #[serde(rename = "job.submit")]
    JobSubmit(JobSpec),
    #[serde(rename = "job.get")]
    JobGet(JobId),
    #[serde(rename = "job.list")]
    JobList,
    #[serde(rename = "job.watch")]
    JobWatch(JobId),
    #[serde(rename = "job.cancel")]
    JobCancel(JobId),
    #[serde(rename = "job.resume")]
    JobResume(JobId),
    #[serde(rename = "job.delete")]
    JobDelete(JobId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEnvelope {
    pub version: u16,
    pub request_id: Uuid,
    pub result: std::result::Result<Response, PublicError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum Response {
    Ack,
    Status { pid: u32, active_jobs: usize },
    Environment(Environment),
    Environments(Vec<Environment>),
    Masks(Vec<MaskRule>),
    MaskedValue(String),
    Settings(RuntimeSettings),
    Job(JobRecord),
    Jobs(Vec<JobRecord>),
    Event(JobProgress),
    Preview(TransferPreview),
}
