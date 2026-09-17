import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";
import type { Environment, EnvironmentDraft, FileFormat, JobProgress, JobRecord, JobSpec, MaskRule, RuntimeSettings, SinkSpec, SourceSpec, TransferPreview } from "./types";
import { filterJobs, findDuplicateMaskColumns, isActiveJobState, parseRelation } from "./logic";

type Screen = "connections" | "transfer" | "jobs" | "masks" | "settings";

const screens: Array<{ id: Screen; label: string; key: string }> = [
  { id: "connections", label: "Connections", key: "1" },
  { id: "transfer", label: "New transfer", key: "2" },
  { id: "jobs", label: "Jobs", key: "3" },
  { id: "masks", label: "Masking rules", key: "4" },
  { id: "settings", label: "Settings", key: "5" },
];

export default function App() {
  const [screen, setScreen] = useState<Screen>("connections");

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey) {
        const destination = screens.find((entry) => entry.key === event.key);
        if (destination) setScreen(destination.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="app-shell">
      <aside>
        <div className="brand" aria-label="ELM Tool">
          <span className="brand-mark">E</span>
          <div><strong>ELM Tool</strong><small>Local transfer studio</small></div>
        </div>
        <nav aria-label="Primary navigation">
          {screens.map((item) => (
            <button
              key={item.id}
              className={screen === item.id ? "active" : ""}
              onClick={() => setScreen(item.id)}
              aria-current={screen === item.id ? "page" : undefined}
            >
              <span>{item.label}</span><kbd>Alt+{item.key}</kbd>
            </button>
          ))}
        </nav>
        <div className="safety-note"><span className="status-dot" />Atomic staging is the default</div>
      </aside>
      <main>
        {screen === "connections" && <Connections />}
        {screen === "transfer" && <NewTransfer onSubmitted={() => setScreen("jobs")} />}
        {screen === "jobs" && <Jobs />}
        {screen === "masks" && <Masks />}
        {screen === "settings" && <Settings />}
      </main>
    </div>
  );
}

function Header({ eyebrow, title, detail }: { eyebrow: string; title: string; detail: string }) {
  return <header className="page-header"><p>{eyebrow}</p><h1>{title}</h1><span>{detail}</span></header>;
}

function Connections() {
  const [items, setItems] = useState<Environment[]>([]);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [draft, setDraft] = useState<EnvironmentDraft>();
  const load = useCallback(() => {
    invoke<Environment[]>("list_environments").then(setItems).catch((value) => setError(String(value)));
  }, []);
  useEffect(load, [load]);
  const add = () => setDraft({ name: "", kind: "postgre_sql", host: "localhost", port: 5432, database: "", username: "", password: "", ssl_mode: "require", root_certificate: "" });
  const edit = (item: Environment) => setDraft({ id: item.id, name: item.name, kind: item.kind, host: item.host, port: item.port, database: item.database, username: item.username, password: "", ssl_mode: item.options.ssl_mode === "disable" ? "disable" : "require", root_certificate: typeof item.options.root_certificate === "string" ? item.options.root_certificate : "" });
  const save = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setMessage("");
    if (!draft) return;
    try {
      await invoke<Environment>("save_environment", { draft });
      setDraft(undefined); setMessage("Connection saved."); load();
    } catch (value) { setError(String(value)); }
  };
  const test = async (item: Environment) => {
    setError(""); setMessage("");
    try { await invoke("test_environment", { id: item.id }); setMessage(`${item.name} connected successfully.`); }
    catch (value) { setError(String(value)); }
  };
  const remove = async (item: Environment) => {
    if (!window.confirm(`Remove ${item.name} and its keychain credential?`)) return;
    setError(""); setMessage("");
    try { await invoke("remove_environment", { id: item.id }); setMessage("Connection removed."); load(); }
    catch (value) { setError(String(value)); }
  };
  return <section>
    <Header eyebrow="Workspace" title="Connections" detail="Credentials stay in your operating system keychain." />
    {error && <div role="alert" className="alert">{error}</div>}
    {message && <div role="status" className="success">{message}</div>}
    <div className="toolbar"><button className="primary" onClick={add}>Add connection</button><button onClick={load}>Refresh</button></div>
    {draft && <form className="panel editor" onSubmit={save}>
      <div className="editor-heading"><h2>{draft.id ? "Edit connection" : "Add connection"}</h2><button type="button" onClick={() => setDraft(undefined)}>Close</button></div>
      <div className="two-column">
        <label>Name<input required value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label>
        <label>Database type<select value={draft.kind} onChange={(event) => setDraft({ ...draft, kind: event.target.value as Environment["kind"] })}><option value="postgre_sql">PostgreSQL</option><option value="my_sql">MySQL</option><option value="sql_server">SQL Server</option><option value="oracle">Oracle</option></select></label>
        <label>Host<input required value={draft.host} onChange={(event) => setDraft({ ...draft, host: event.target.value })} /></label>
        <label>Port<input required type="number" min="1" max="65535" value={draft.port} onChange={(event) => setDraft({ ...draft, port: Number(event.target.value) })} /></label>
        <label>Database or service<input required value={draft.database} onChange={(event) => setDraft({ ...draft, database: event.target.value })} /></label>
        <label>Username<input required autoComplete="username" value={draft.username} onChange={(event) => setDraft({ ...draft, username: event.target.value })} /></label>
        <label>TLS mode<select value={draft.ssl_mode ?? "require"} onChange={(event) => setDraft({ ...draft, ssl_mode: event.target.value as EnvironmentDraft["ssl_mode"] })}><option value="require">Required and verified</option><option value="disable">Disabled (trusted networks only)</option></select><small>TLS is required by default. Oracle uses the native client's wallet/trust configuration.</small></label>
        {(draft.kind === "postgre_sql" || draft.kind === "my_sql") && <label>Custom root certificate<input value={draft.root_certificate ?? ""} onChange={(event) => setDraft({ ...draft, root_certificate: event.target.value })} placeholder="Optional PEM or DER file path" /></label>}
        <label className="wide">Password{draft.id && <small>Leave blank to keep the current keychain credential.</small>}<input required={!draft.id} type="password" autoComplete="new-password" value={draft.password} onChange={(event) => setDraft({ ...draft, password: event.target.value })} /></label>
      </div>
      <div className="footer-actions"><button type="button" onClick={() => setDraft(undefined)}>Cancel</button><button className="primary" type="submit">Save connection</button></div>
    </form>}
    <div className="card-grid">
      {items.map((item) => <article className="card" key={item.id}>
        <div className="card-title"><h2>{item.name}</h2><span className="pill">{item.kind.replace("_", " ")}</span></div>
        <p>{item.username}@{item.host}:{item.port}</p><strong>{item.database}</strong>
        <div className="card-actions"><button onClick={() => void test(item)}>Test</button><button onClick={() => edit(item)}>Edit</button><button className="danger" onClick={() => void remove(item)}>Remove</button></div>
      </article>)}
      {!items.length && !error && <div className="empty"><h2>No connections yet</h2><p>Add a database connection to begin a transfer.</p></div>}
    </div>
  </section>;
}

function NewTransfer({ onSubmitted }: { onSubmitted: () => void }) {
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [direction, setDirection] = useState<"db-to-db" | "db-to-file" | "file-to-db">("db-to-db");
  const [sourceEnvironment, setSourceEnvironment] = useState("");
  const [targetEnvironment, setTargetEnvironment] = useState("");
  const [sourceValue, setSourceValue] = useState("");
  const [sourceIsQuery, setSourceIsQuery] = useState(false);
  const [targetTable, setTargetTable] = useState("");
  const [filePath, setFilePath] = useState("");
  const [format, setFormat] = useState<FileFormat>("csv");
  const [writeMode, setWriteMode] = useState<JobSpec["write_mode"]>("append");
  const [consistency, setConsistency] = useState<JobSpec["consistency"]>("atomic");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [preview, setPreview] = useState<TransferPreview>();
  const [settings, setSettings] = useState<RuntimeSettings>({ memory_budget_bytes: 512 * 1024 * 1024, batch_target_bytes: 16 * 1024 * 1024 });
  const [masks, setMasks] = useState<MaskRule[]>([]);
  const [selectedMaskIds, setSelectedMaskIds] = useState<Set<string>>(new Set());
  useEffect(() => { void invoke<Environment[]>("list_environments").then((items) => {
    setEnvironments(items);
    if (items[0]) { setSourceEnvironment((value) => value || items[0].id); setTargetEnvironment((value) => value || items[0].id); }
  }).catch((value) => setError(String(value))); }, []);
  useEffect(() => { void invoke<RuntimeSettings>("get_settings").then(setSettings).catch((value) => setError(String(value))); }, []);
  useEffect(() => { void invoke<MaskRule[]>("list_masks").then(setMasks).catch((value) => setError(String(value))); }, []);
  // Masking is opt-in per transfer: nothing is ever masked unless explicitly checked here. This
  // only seeds a suggested default (rules scoped to a relevant connection, plus global rules)
  // whenever the relevant connections change; the user's own checkbox choices are not otherwise
  // second-guessed.
  useEffect(() => {
    setSelectedMaskIds(new Set(
      masks
        .filter((rule) => !rule.environment_id || rule.environment_id === sourceEnvironment || rule.environment_id === targetEnvironment)
        .map((rule) => rule.id)
    ));
  }, [masks, sourceEnvironment, targetEnvironment, direction]);
  const toggleMask = (id: string) => setSelectedMaskIds((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  // Any field that changes what would be transferred invalidates a prior preview: the reviewed
  // columns and publication safety no longer describe what "Submit transfer" would actually run.
  useEffect(() => { setPreview(undefined); setPreviewError(""); }, [
    direction, sourceEnvironment, targetEnvironment, sourceValue, sourceIsQuery,
    targetTable, filePath, format, writeMode, consistency, selectedMaskIds,
  ]);
  const buildSpec = (): JobSpec => {
    const source: SourceSpec = direction === "file-to-db"
      ? { kind: "file", path: filePath, format }
      : { kind: "database", environment_id: sourceEnvironment, selection: sourceIsQuery ? { kind: "query", sql: sourceValue } : { kind: "table", relation: parseRelation(sourceValue) }, resume_key: [] };
    const sink: SinkSpec = direction === "db-to-file"
      ? { kind: "file", path: filePath, format }
      : { kind: "database", environment_id: targetEnvironment, relation: parseRelation(targetTable) };
    const seed = new Uint8Array(32); crypto.getRandomValues(seed);
    return {
      version: 1, id: crypto.randomUUID(), name: `${direction} transfer`, source, sink,
      write_mode: writeMode, consistency, memory_budget_bytes: settings.memory_budget_bytes,
      batch_target_bytes: settings.batch_target_bytes,
      masks: masks.filter((rule) => selectedMaskIds.has(rule.id)),
      random_seed: Array.from(seed),
      conversions: [], submitted_at: new Date().toISOString(),
    };
  };
  const runPreview = async () => {
    setPreviewError(""); setError(""); setPreviewing(true);
    try {
      const spec = buildSpec();
      setPreview(await invoke<TransferPreview>("preview_transfer", { spec }));
    } catch (value) { setPreviewError(String(value)); setPreview(undefined); }
    finally { setPreviewing(false); }
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setSubmitting(true);
    try {
      const spec = buildSpec();
      await invoke<JobRecord>("submit_job", { spec }); onSubmitted();
    } catch (value) { setError(String(value)); }
    finally { setSubmitting(false); }
  };
  const reviewed = Boolean(preview) && !preview?.publication_error;
  const duplicateMaskColumns = useMemo(
    () => findDuplicateMaskColumns(masks, selectedMaskIds),
    [masks, selectedMaskIds],
  );
  const hasMaskConflict = duplicateMaskColumns.length > 0;
  return <section>
    <Header eyebrow="Transfer wizard" title="New transfer" detail="Preflight catches unsafe privileges and lossy mappings before data moves." />
    {error && <div role="alert" className="alert">{error}</div>}
    <div className="wizard-steps" aria-label="Transfer steps"><b>1 Source</b><span>2 Destination</span><span>3 Mapping</span><span>4 Review</span></div>
    <form className="panel two-column" onSubmit={submit}>
      <label>Transfer direction<select value={direction} onChange={(event) => setDirection(event.target.value as typeof direction)}><option value="db-to-db">Database to database</option><option value="db-to-file">Database to file</option><option value="file-to-db">File to database</option></select></label>
      <label>Consistency<select value={consistency} onChange={(event) => setConsistency(event.target.value as JobSpec["consistency"])}><option value="atomic">Atomic staged</option><option value="checkpointed">Checkpointed (advanced)</option><option value="table_swap">Oracle table swap (non-atomic REPLACE)</option></select></label>
      {consistency === "table_swap" && <div role="note" className="alert wide">Oracle REPLACE only. The target name can be temporarily unavailable between renames; interruption requires recovery. The original table is retained as a backup until publication succeeds.</div>}
      {direction !== "file-to-db" && <label>Source connection<select required value={sourceEnvironment} onChange={(event) => setSourceEnvironment(event.target.value)}><option value="">Select a connection…</option>{environments.map((environment) => <option key={environment.id} value={environment.id}>{environment.name}</option>)}</select></label>}
      {direction !== "db-to-file" && <label>Target connection<select required value={targetEnvironment} onChange={(event) => setTargetEnvironment(event.target.value)}><option value="">Select a connection…</option>{environments.map((environment) => <option key={environment.id} value={environment.id}>{environment.name}</option>)}</select></label>}
      {direction !== "file-to-db" && <label className="wide">Source table or explicit query<input required value={sourceValue} onChange={(event) => setSourceValue(event.target.value)} placeholder={sourceIsQuery ? "SELECT …" : "public.customers"} /><span className="checkbox"><input type="checkbox" checked={sourceIsQuery} onChange={(event) => setSourceIsQuery(event.target.checked)} />Treat this as explicit SQL</span></label>}
      {direction !== "db-to-file" && <label className="wide">Target table<input required value={targetTable} onChange={(event) => setTargetTable(event.target.value)} placeholder="analytics.customers" /></label>}
      {direction !== "db-to-db" && <><label className="wide">{direction === "file-to-db" ? "Input file path" : "Output file path"}<input required value={filePath} onChange={(event) => setFilePath(event.target.value)} /></label><label>File format<select value={format} onChange={(event) => setFormat(event.target.value as FileFormat)}><option value="csv">CSV</option><option value="ndjson">NDJSON</option><option value="parquet">Parquet</option></select></label></>}
      <label>Write mode<select value={writeMode} onChange={(event) => setWriteMode(event.target.value as JobSpec["write_mode"])}><option value="append">Append</option><option value="replace">Replace</option><option value="fail">Fail if destination exists</option></select></label>
      <div className="wide">
        <div className="editor-heading"><h2>Masking</h2></div>
        {masks.length
          ? <>
              <p>No rule is applied unless checked here. Rules scoped to a connection used by this transfer are checked by default.</p>
              {hasMaskConflict && <div role="alert" className="alert">Only one masking rule can apply per column: uncheck one of the rules for {duplicateMaskColumns.map((column) => <code key={column}>{column}</code>)}. The last-checked rule would otherwise silently win.</div>}
              <div className="table-wrap"><table><thead><tr><th /><th>Name</th><th>Column</th><th>Algorithm</th><th>Scope</th></tr></thead><tbody>
                {masks.map((rule) => <tr key={rule.id}>
                  <td><input type="checkbox" aria-label={`Apply masking rule ${rule.name}`} checked={selectedMaskIds.has(rule.id)} onChange={() => toggleMask(rule.id)} /></td>
                  <td>{rule.name}</td>
                  <td><code>{rule.column}</code></td>
                  <td>{typeof rule.algorithm === "string" ? rule.algorithm : String(rule.algorithm.algorithm ?? Object.keys(rule.algorithm)[0])}</td>
                  <td>{environments.find((environment) => environment.id === rule.environment_id)?.name ?? "Global"}</td>
                </tr>)}
              </tbody></table></div>
            </>
          : <p>No masking rules saved yet. Add one on the Masking rules screen, then come back here to apply it to this transfer.</p>}
      </div>
      <div className="callout wide"><strong>Publication guarantee</strong><p>The target stays unchanged until staging validation succeeds. ELM will stop if your account cannot publish safely.</p></div>
      <div className="callout warning wide"><strong>Alpha database boundary</strong><p>Database connectors have limited type mappings. PostgreSQL supports keyset resume; other database sources restart from zero. SQL Server requires Driver 18; Oracle requires Instant Client. Oracle uses bounded native array fetching and batch DML staging. Existing Oracle targets support atomic APPEND or explicit non-atomic table-swap REPLACE. A swap retains the old table's indexes, grants, and constraints on its backup, not on the replacement. Backups remain until job deletion. LOB support and large-workload performance qualification remain pending.</p></div>
      <div className="wide">
        <div className="editor-heading"><h2>4. Review</h2><button type="button" onClick={() => void runPreview()} disabled={previewing || hasMaskConflict} title={hasMaskConflict ? "Resolve the conflicting masking rules above first" : undefined}>{previewing ? "Checking…" : "Run preview"}</button></div>
        {previewError && <div role="alert" className="alert">{previewError}</div>}
        {preview && <div className="panel">
          <table><thead><tr><th>Column</th><th>Source type</th><th>Destination type</th><th>Nullable</th></tr></thead><tbody>
            {preview.columns.map((column) => <tr key={column.name}><td><code>{column.name}</code></td><td>{column.source_type}</td><td>{column.output_type}</td><td>{column.nullable ? "yes" : "no"}</td></tr>)}
          </tbody></table>
          {preview.report.warnings.map((warning) => <div key={warning} role="status" className="alert wide">{warning}</div>)}
          {preview.publication_error
            ? <div role="alert" className="alert wide"><strong>This configuration cannot publish safely</strong><p>{preview.publication_error}</p></div>
            : <div role="status" className="success">Preflight passed. This configuration can publish safely with the requested write mode and consistency.</div>}
        </div>}
        {!preview && !previewError && <p>Run a preview to see the destination columns and confirm the requested write mode and consistency are safe, before any data moves.</p>}
      </div>
      <div className="wide footer-actions"><button className="primary" type="submit" disabled={submitting || !reviewed || hasMaskConflict} title={hasMaskConflict ? "Resolve the conflicting masking rules above first" : reviewed ? undefined : "Run a preview that passes before submitting"}>{submitting ? "Submitting…" : "Submit transfer"}</button></div>
    </form>
  </section>;
}

function Jobs() {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [selected, setSelected] = useState<string>();
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const load = useCallback(async () => {
    try { setJobs(await invoke<JobRecord[]>("list_jobs")); } catch (value) { setError(String(value)); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    const active = jobs.filter((job) => isActiveJobState(job.progress.state));
    const cleanups = active.map((job) => {
      const channel = new Channel<JobProgress>();
      channel.onmessage = (progress) => setJobs((current) => current.map((entry) => entry.spec.id === job.spec.id ? { ...entry, progress } : entry));
      void invoke("watch_job", { id: job.spec.id, events: channel }).catch((value) => setError(String(value)));
      return () => { channel.onmessage = () => undefined; };
    });
    return () => cleanups.forEach((cleanup) => cleanup());
  }, [jobs.length]);
  const current = useMemo(() => jobs.find((job) => job.spec.id === selected), [jobs, selected]);
  const filtered = useMemo(() => filterJobs(jobs, query, stateFilter), [jobs, query, stateFilter]);
  const action = async (id: string, actionName: "cancel" | "resume" | "delete") => {
    if (actionName === "cancel" && !window.confirm("Cancel this transfer? There is no pause: this cannot be resumed afterward, only deleted. Submit a new transfer to retry it.")) return;
    setError("");
    try { await invoke("job_action", { id, action: actionName }); if (actionName === "delete") setSelected(undefined); await load(); }
    catch (value) { setError(String(value)); }
  };
  return <section>
    <Header eyebrow="Activity" title="Jobs" detail="Close this window whenever you like—the daemon keeps active jobs running." />
    {error && <div role="alert" className="alert">{error}</div>}
    <div className="toolbar filters"><input aria-label="Filter jobs" placeholder="Filter by name or id" value={query} onChange={(event) => setQuery(event.target.value)} /><select aria-label="Filter by state" value={stateFilter} onChange={(event) => setStateFilter(event.target.value)}><option value="">All states</option>{["queued", "preflighting", "running", "publishing", "succeeded", "failed", "cancelling", "cancelled", "interrupted"].map((state) => <option key={state} value={state}>{state}</option>)}</select><button onClick={() => void load()}>Refresh</button></div>
    <div className="split-panel">
      <div className="job-list" role="list">
        {filtered.map((job) => <button role="listitem" key={job.spec.id} onClick={() => setSelected(job.spec.id)} className={selected === job.spec.id ? "selected" : ""}>
          <span><strong>{job.spec.name || job.spec.id.slice(0, 8)}</strong><small>{new Date(job.updated_at).toLocaleString()}</small></span>
          <span className={`state ${job.progress.state}`}>{job.progress.state}</span>
          {job.progress.state === "succeeded"
            ? <progress max="100" value="100" aria-label="Transfer complete" />
            : isActiveJobState(job.progress.state)
              ? <progress aria-label="Transfer in progress; exact completion percentage isn't tracked yet" />
              : null}
          <small>{job.progress.rows.toLocaleString()} rows · {(job.progress.bytes_per_second / 1048576).toFixed(1)} MiB/s</small>
        </button>)}
        {!filtered.length && <div className="empty"><h2>No matching jobs</h2><p>Submitted transfers that match the filters appear here.</p></div>}
      </div>
      <div className="job-detail">{current ? <>
        <h2>{current.spec.name || "Transfer details"}</h2><dl><dt>State</dt><dd>{current.progress.state}</dd><dt>Rows</dt><dd>{current.progress.rows.toLocaleString()}</dd><dt>Attempt</dt><dd>{current.attempt}</dd></dl>
        {current.progress.error && <div className="alert"><strong>{current.progress.error.message}</strong><p>{current.progress.error.remediation}</p></div>}
        <div className="card-actions">
          <button title="Stops the transfer permanently. There is no pause: it cannot be resumed afterward." disabled={!['queued', 'preflighting', 'running', 'publishing', 'cancelling'].includes(current.progress.state)} onClick={() => void action(current.spec.id, "cancel")}>Cancel</button>
          <button title={current.progress.state === 'cancelled' ? "A cancelled transfer cannot be resumed; submit a new transfer to retry it." : "Continues from the last durable checkpoint."} disabled={!['failed', 'interrupted'].includes(current.progress.state)} onClick={() => void action(current.spec.id, "resume")}>Resume</button>
          <button className="danger" disabled={!['succeeded', 'failed', 'cancelled', 'interrupted'].includes(current.progress.state)} onClick={() => void action(current.spec.id, "delete")}>Delete</button>
        </div>
      </> : <div className="empty"><p>Select a job to inspect checkpoints and recovery details.</p></div>}</div>
    </div>
  </section>;
}

function Masks() {
  const [rules, setRules] = useState<MaskRule[]>([]);
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [draft, setDraft] = useState<{ id?: string; name: string; column: string; algorithm: string; unmaskedPrefix: number; environmentId: string; testValue: string }>({ name: "", column: "", algorithm: "star", unmaskedPrefix: 0, environmentId: "", testValue: "" });
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const load = useCallback(() => {
    void invoke<MaskRule[]>("list_masks").then(setRules).catch((value) => setError(String(value)));
    void invoke<Environment[]>("list_environments").then(setEnvironments).catch((value) => setError(String(value)));
  }, []);
  useEffect(load, [load]);
  const openNew = () => { setDraft({ name: "", column: "", algorithm: "star", unmaskedPrefix: 0, environmentId: "", testValue: "" }); setEditing(true); };
  const openEdit = (rule: MaskRule) => {
    const algorithm = typeof rule.algorithm === "string" ? rule.algorithm : String(rule.algorithm.algorithm ?? Object.keys(rule.algorithm)[0]);
    const unmaskedPrefix = typeof rule.algorithm === "object" ? Number(rule.algorithm.unmasked_prefix ?? 0) : 0;
    setDraft({ id: rule.id, name: rule.name, column: rule.column, algorithm, unmaskedPrefix, environmentId: rule.environment_id ?? "", testValue: "" }); setEditing(true);
  };
  const buildRule = (): MaskRule => ({
    id: draft.id ?? crypto.randomUUID(), name: draft.name, column: draft.column,
    algorithm: draft.algorithm === "star_length" ? { algorithm: "star_length", unmasked_prefix: draft.unmaskedPrefix } : { algorithm: draft.algorithm },
    environment_id: draft.environmentId || undefined,
  });
  const save = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setMessage("");
    try { await invoke<MaskRule>("save_mask", { rule: buildRule() }); setEditing(false); setMessage("Masking rule saved."); load(); }
    catch (value) { setError(String(value)); }
  };
  const test = async () => {
    setError(""); setMessage("");
    try { const output = await invoke<string>("test_mask", { rule: buildRule(), value: draft.testValue }); setMessage(`Preview: ${output}`); }
    catch (value) { setError(String(value)); }
  };
  const remove = async (rule: MaskRule) => {
    if (!window.confirm(`Remove masking rule ${rule.name}?`)) return;
    setError(""); setMessage("");
    try { await invoke("remove_mask", { id: rule.id }); setMessage("Masking rule removed."); load(); }
    catch (value) { setError(String(value)); }
  };
  return <section><Header eyebrow="Protection" title="Masking rules" detail="Random masks are seeded per job for deterministic retries and resume." />
    {error && <div role="alert" className="alert">{error}</div>}
    {message && <div role="status" className="success">{message}</div>}
    <div className="toolbar"><button className="primary" onClick={openNew}>Add rule</button><button onClick={load}>Refresh</button></div>
    {editing && <form className="panel editor" onSubmit={save}>
      <div className="editor-heading"><h2>{draft.id ? "Edit masking rule" : "Add masking rule"}</h2><button type="button" onClick={() => setEditing(false)}>Close</button></div>
      <div className="two-column">
        <label>Name<input required value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label>
        <label>Column<input required value={draft.column} onChange={(event) => setDraft({ ...draft, column: event.target.value })} /></label>
        <label>Algorithm<select value={draft.algorithm} onChange={(event) => setDraft({ ...draft, algorithm: event.target.value })}><option value="star">Star</option><option value="star_length">Star length</option><option value="random">Random</option><option value="nullify">Nullify</option></select></label>
        <label>Scope<select value={draft.environmentId} onChange={(event) => setDraft({ ...draft, environmentId: event.target.value })}><option value="">Global</option>{environments.map((environment) => <option key={environment.id} value={environment.id}>{environment.name}</option>)}</select></label>
        {draft.algorithm === "star_length" && <label>Unmasked prefix<input type="number" min="0" value={draft.unmaskedPrefix} onChange={(event) => setDraft({ ...draft, unmaskedPrefix: Number(event.target.value) })} /></label>}
        <label className="wide">Preview value<input value={draft.testValue} onChange={(event) => setDraft({ ...draft, testValue: event.target.value })} /></label>
      </div>
      <div className="footer-actions"><button type="button" onClick={() => void test()}>Test value</button><button type="button" onClick={() => setEditing(false)}>Cancel</button><button className="primary" type="submit">Save rule</button></div>
    </form>}
    <div className="table-wrap"><table><thead><tr><th>Name</th><th>Column</th><th>Algorithm</th><th>Scope</th><th>Actions</th></tr></thead><tbody>
      {rules.map((rule) => <tr key={rule.id}><td>{rule.name}</td><td><code>{rule.column}</code></td><td>{typeof rule.algorithm === "string" ? rule.algorithm : String(rule.algorithm.algorithm ?? Object.keys(rule.algorithm)[0])}</td><td>{environments.find((environment) => environment.id === rule.environment_id)?.name ?? "Global"}</td><td className="row-actions"><button onClick={() => openEdit(rule)}>Edit</button><button className="danger" onClick={() => void remove(rule)}>Remove</button></td></tr>)}
    </tbody></table></div>
  </section>;
}

function Settings() {
  const mib = 1024 * 1024;
  const [settings, setSettings] = useState<RuntimeSettings>({ memory_budget_bytes: 512 * mib, batch_target_bytes: 16 * mib });
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  useEffect(() => { void invoke<RuntimeSettings>("get_settings").then(setSettings).catch((value) => setError(String(value))); }, []);
  const save = async (event: FormEvent) => {
    event.preventDefault(); setError(""); setMessage("");
    try { setSettings(await invoke<RuntimeSettings>("save_settings", { settings })); setMessage("Runtime settings saved."); }
    catch (value) { setError(String(value)); }
  };
  return <section><Header eyebrow="Runtime" title="Settings" detail="Tune bounded resource use without weakening consistency." />
    {error && <div role="alert" className="alert">{error}</div>}
    {message && <div role="status" className="success">{message}</div>}
    <form className="panel settings" onSubmit={save}><label>Process memory budget <span>{settings.memory_budget_bytes / mib} MiB</span><input type="range" min="128" max="2048" step="64" value={settings.memory_budget_bytes / mib} onChange={(event) => setSettings({ ...settings, memory_budget_bytes: Number(event.target.value) * mib })} /></label><label>Target batch size<select value={settings.batch_target_bytes / mib} onChange={(event) => setSettings({ ...settings, batch_target_bytes: Number(event.target.value) * mib })}><option value="8">8 MiB</option><option value="16">16 MiB</option><option value="32">32 MiB</option></select></label><div className="callout"><strong>Native prerequisites</strong><p>Oracle requires Instant Client. SQL Server requires Microsoft ODBC Driver 18. Live driver probes will activate with the native database backends.</p></div><div className="footer-actions"><button className="primary" type="submit">Save settings</button></div></form>
  </section>;
}
