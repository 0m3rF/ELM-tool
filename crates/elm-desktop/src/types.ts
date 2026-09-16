export type JobState =
  | "queued"
  | "preflighting"
  | "running"
  | "publishing"
  | "succeeded"
  | "failed"
  | "cancelling"
  | "cancelled"
  | "interrupted";

export interface PublicError {
  code: string;
  message: string;
  retryable: boolean;
  remediation?: string;
}

export interface JobProgress {
  job_id: string;
  state: JobState;
  rows: number;
  bytes: number;
  batches: number;
  rows_per_second: number;
  bytes_per_second: number;
  elapsed: { secs: number; nanos: number };
  eta?: { secs: number; nanos: number };
  warnings: string[];
  error?: PublicError;
  occurred_at: string;
}

export interface Relation {
  catalog?: string;
  schema?: string;
  name: string;
}

export type FileFormat = "csv" | "ndjson" | "parquet";

export type SourceSpec =
  | { kind: "file"; path: string; format: FileFormat }
  | {
      kind: "database";
      environment_id: string;
      selection:
        | { kind: "table"; relation: Relation }
        | { kind: "query"; sql: string };
      resume_key: string[];
    };

export type SinkSpec =
  | { kind: "file"; path: string; format: FileFormat }
  | { kind: "database"; environment_id: string; relation: Relation };

export interface JobSpec {
  version: number;
  id: string;
  name?: string;
  source: SourceSpec;
  sink: SinkSpec;
  write_mode: "append" | "replace" | "fail";
  consistency: "atomic" | "checkpointed" | "table_swap";
  memory_budget_bytes: number;
  batch_target_bytes: number;
  masks: MaskRule[];
  random_seed: number[];
  conversions: Array<{ column: string; target_type: string; allow_lossy: boolean }>;
  submitted_at: string;
}

export interface JobRecord {
  spec: JobSpec;
  progress: JobProgress;
  attempt: number;
  updated_at: string;
}

export interface Environment {
  id: string;
  name: string;
  kind: "postgre_sql" | "oracle" | "my_sql" | "sql_server";
  host: string;
  port: number;
  database: string;
  username: string;
  credential_ref: string;
  options: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface EnvironmentDraft {
  id?: string;
  name: string;
  kind: Environment["kind"];
  host: string;
  port: number;
  database: string;
  username: string;
  password?: string;
  ssl_mode?: "disable" | "require";
  root_certificate?: string;
}

export interface RuntimeSettings {
  memory_budget_bytes: number;
  batch_target_bytes: number;
}

export interface MaskRule {
  id: string;
  name: string;
  column: string;
  algorithm: Record<string, unknown> | string;
  environment_id?: string;
}

export interface PreviewColumn {
  name: string;
  source_type: string;
  output_type: string;
  nullable: boolean;
}

export interface ConnectorCapabilities {
  database: Environment["kind"] | null;
  native_bulk_read: boolean;
  native_bulk_write: boolean;
  atomic_append: boolean;
  atomic_replace: boolean;
  non_atomic_replace: boolean;
  checkpointed_write: boolean;
  resumable_keyset_read: boolean;
  supports_lob_spill: boolean;
}

export interface PreflightReport {
  capabilities: ConnectorCapabilities;
  staging_relation: Relation | null;
  warnings: string[];
}

export interface TransferPreview {
  columns: PreviewColumn[];
  report: PreflightReport;
  publication_error?: string;
}
