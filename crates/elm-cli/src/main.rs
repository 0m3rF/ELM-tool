use std::{path::PathBuf, str::FromStr};

use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use secrecy::SecretString;

use elm_core::{
    ConsistencyMode, DatabaseKind, DatabaseSelection, Environment, EnvironmentId, FileFormat,
    Identifier, JobId, JobProgress, JobSpec, JobState, MaskAlgorithm, MaskRule, MaskRuleId,
    Relation, SinkSpec, SourceSpec, WriteMode,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, RuntimePaths};
use elm_state::{CredentialVault, KeyringVault};

#[derive(Debug, Parser)]
#[command(name = "elm", version, about = "Fast, safe local database transfers")]
struct Cli {
    #[arg(long, global = true, help = "Print machine-readable JSON")]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
    #[command(alias = "environment")]
    Env {
        #[command(subcommand)]
        command: EnvironmentCommand,
    },
    Mask {
        #[command(subcommand)]
        command: MaskCommand,
    },
    Copy {
        #[command(subcommand)]
        command: CopyCommand,
    },
    Jobs {
        #[command(subcommand)]
        command: JobsCommand,
    },
}

#[derive(Debug, Subcommand)]
enum DaemonCommand {
    Start,
    Status,
    Stop,
}

#[derive(Debug, Subcommand)]
enum EnvironmentCommand {
    Add(EnvironmentAdd),
    Edit(EnvironmentEdit),
    List,
    Test { environment: String },
    Remove { environment: String },
}

#[derive(Debug, Args)]
struct EnvironmentAdd {
    #[arg(long)]
    name: String,
    #[arg(long, value_enum)]
    kind: DatabaseArgument,
    #[arg(long)]
    host: String,
    #[arg(long)]
    port: u16,
    #[arg(long)]
    database: String,
    #[arg(long)]
    username: String,
    #[arg(
        long,
        help = "Read the password from stdin instead of an interactive prompt"
    )]
    password_stdin: bool,
    #[arg(long, value_enum, default_value = "require")]
    ssl_mode: SslModeArgument,
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=300_000), help = "Oracle native call timeout in milliseconds (default 15000; maximum 300000)")]
    oracle_call_timeout_ms: Option<u64>,
}

#[derive(Debug, Args)]
struct EnvironmentEdit {
    environment: String,
    #[arg(long)]
    name: Option<String>,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    port: Option<u16>,
    #[arg(long)]
    database: Option<String>,
    #[arg(long)]
    username: Option<String>,
    #[arg(long, help = "Replace the keychain password using stdin")]
    password_stdin: bool,
    #[arg(long, value_enum)]
    ssl_mode: Option<SslModeArgument>,
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=300_000))]
    oracle_call_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SslModeArgument {
    Disable,
    Require,
}

impl SslModeArgument {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disable => "disable",
            Self::Require => "require",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DatabaseArgument {
    Postgresql,
    Oracle,
    Mysql,
    SqlServer,
}

impl From<DatabaseArgument> for DatabaseKind {
    fn from(value: DatabaseArgument) -> Self {
        match value {
            DatabaseArgument::Postgresql => Self::PostgreSql,
            DatabaseArgument::Oracle => Self::Oracle,
            DatabaseArgument::Mysql => Self::MySql,
            DatabaseArgument::SqlServer => Self::SqlServer,
        }
    }
}

#[derive(Debug, Subcommand)]
enum MaskCommand {
    Add(MaskArguments),
    Edit(MaskEdit),
    List,
    Remove { id: String },
    Test(MaskTest),
}

#[derive(Debug, Args)]
struct MaskArguments {
    #[arg(long)]
    name: String,
    #[arg(long)]
    column: String,
    #[arg(long, value_enum)]
    algorithm: MaskArgument,
    #[arg(long, requires = "algorithm")]
    unmasked_prefix: Option<usize>,
    #[arg(long)]
    environment: Option<String>,
}

#[derive(Debug, Args)]
struct MaskEdit {
    id: String,
    #[command(flatten)]
    rule: MaskArguments,
}

#[derive(Debug, Args)]
struct MaskTest {
    #[arg(long)]
    column: String,
    #[arg(long)]
    value: String,
    #[arg(long, value_enum)]
    algorithm: MaskArgument,
    #[arg(long)]
    unmasked_prefix: Option<usize>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MaskArgument {
    Star,
    StarLength,
    Random,
    Nullify,
}

impl MaskArgument {
    fn algorithm(self, prefix: Option<usize>) -> anyhow::Result<MaskAlgorithm> {
        Ok(match self {
            Self::Star => MaskAlgorithm::Star,
            Self::StarLength => MaskAlgorithm::StarLength {
                unmasked_prefix: prefix.ok_or_else(|| {
                    anyhow::anyhow!("--unmasked-prefix is required for star-length")
                })?,
            },
            Self::Random => MaskAlgorithm::Random,
            Self::Nullify => MaskAlgorithm::Nullify,
        })
    }
}

#[derive(Debug, Subcommand)]
enum CopyCommand {
    DbToDb(DbToDb),
    DbToFile(DbToFile),
    FileToDb(FileToDb),
}

#[derive(Debug, Args)]
struct SourceDatabase {
    #[arg(long)]
    source: String,
    #[arg(long, conflicts_with = "query", required_unless_present = "query")]
    source_table: Option<String>,
    #[arg(
        long,
        conflicts_with = "source_table",
        required_unless_present = "source_table"
    )]
    query: Option<String>,
    #[arg(long, value_delimiter = ',')]
    resume_key: Vec<String>,
}

#[derive(Debug, Args)]
struct TransferOptions {
    #[arg(long, value_enum, default_value = "append")]
    mode: WriteArgument,
    #[arg(long, value_enum, default_value = "atomic")]
    consistency: ConsistencyArgument,
    #[arg(long, default_value_t = 512)]
    memory_mib: u64,
    #[arg(long, default_value_t = 16)]
    batch_mib: u64,
    #[arg(long)]
    detach: bool,
    /// Run preflight against the real source and destination and print the result without
    /// moving any data or creating a job.
    #[arg(long)]
    dry_run: bool,
    /// Attach a saved masking rule (by id, from `elm mask list`) to this transfer. Repeatable,
    /// or comma-separated. Masking is opt-in per transfer: a saved rule has no effect on any
    /// transfer unless named here.
    #[arg(long = "mask-rule", value_delimiter = ',')]
    mask_rules: Vec<String>,
}

#[derive(Debug, Args)]
struct DbToDb {
    #[command(flatten)]
    source_database: SourceDatabase,
    #[arg(long)]
    target: String,
    #[arg(long)]
    target_table: String,
    #[command(flatten)]
    transfer: TransferOptions,
}

#[derive(Debug, Args)]
struct DbToFile {
    #[command(flatten)]
    source_database: SourceDatabase,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum)]
    format: FormatArgument,
    #[command(flatten)]
    transfer: TransferOptions,
}

#[derive(Debug, Args)]
struct FileToDb {
    #[arg(long)]
    input: PathBuf,
    #[arg(long, value_enum)]
    format: FormatArgument,
    #[arg(long)]
    target: String,
    #[arg(long)]
    target_table: String,
    #[command(flatten)]
    transfer: TransferOptions,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FormatArgument {
    Csv,
    Ndjson,
    Parquet,
}

impl From<FormatArgument> for FileFormat {
    fn from(value: FormatArgument) -> Self {
        match value {
            FormatArgument::Csv => Self::Csv,
            FormatArgument::Ndjson => Self::Ndjson,
            FormatArgument::Parquet => Self::Parquet,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum WriteArgument {
    Append,
    Replace,
    Fail,
}

impl From<WriteArgument> for WriteMode {
    fn from(value: WriteArgument) -> Self {
        match value {
            WriteArgument::Append => Self::Append,
            WriteArgument::Replace => Self::Replace,
            WriteArgument::Fail => Self::Fail,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConsistencyArgument {
    Atomic,
    Checkpointed,
    /// Oracle REPLACE only: non-atomic rename swap with a recovery backup.
    TableSwap,
}

impl From<ConsistencyArgument> for ConsistencyMode {
    fn from(value: ConsistencyArgument) -> Self {
        match value {
            ConsistencyArgument::Atomic => Self::Atomic,
            ConsistencyArgument::Checkpointed => Self::Checkpointed,
            ConsistencyArgument::TableSwap => Self::TableSwap,
        }
    }
}

#[derive(Debug, Subcommand)]
enum JobsCommand {
    List,
    Show {
        id: String,
    },
    Watch {
        id: String,
    },
    /// Stops the job permanently. There is no pause: a cancelled job cannot be resumed, only
    /// deleted; submit a new transfer to retry it.
    Cancel {
        id: String,
    },
    /// Resumes a job left `interrupted` (e.g. by a daemon restart) or `failed`, continuing from
    /// its last durable checkpoint. Not available for a cancelled job.
    Resume {
        id: String,
    },
    Delete {
        id: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let paths = RuntimePaths::discover()?;
    match cli.command {
        Command::Daemon { command } => handle_daemon(command, paths, cli.json).await,
        command => {
            let client = DaemonClient::connect_or_start(paths).await?;
            handle_command(command, &client, cli.json).await
        }
    }
}

async fn handle_daemon(
    command: DaemonCommand,
    paths: RuntimePaths,
    json: bool,
) -> anyhow::Result<()> {
    match command {
        DaemonCommand::Start => {
            let client = DaemonClient::connect_or_start(paths).await?;
            print_response(client.request(Operation::DaemonStatus).await?, json)
        }
        DaemonCommand::Status => {
            let client = DaemonClient::connect(paths)?;
            print_response(client.request(Operation::DaemonStatus).await?, json)
        }
        DaemonCommand::Stop => {
            let client = DaemonClient::connect(paths)?;
            print_response(client.request(Operation::DaemonStop).await?, json)
        }
    }
}

async fn handle_command(command: Command, client: &DaemonClient, json: bool) -> anyhow::Result<()> {
    match command {
        Command::Env { command } => handle_environment(command, client, json).await,
        Command::Mask { command } => handle_mask(command, client, json).await,
        Command::Copy { command } => handle_copy(command, client, json).await,
        Command::Jobs { command } => handle_jobs(command, client, json).await,
        Command::Daemon { .. } => Err(anyhow::anyhow!(
            "daemon command reached the auto-start dispatcher unexpectedly"
        )),
    }
}

async fn handle_environment(
    command: EnvironmentCommand,
    client: &DaemonClient,
    json: bool,
) -> anyhow::Result<()> {
    match command {
        EnvironmentCommand::Add(arguments) => {
            let id = EnvironmentId::new();
            let reference = format!("environment-{id}");
            let password = read_password(arguments.password_stdin)?;
            let vault = KeyringVault::default();
            vault.put(&reference, &password)?;
            let now = Utc::now();
            let kind = arguments.kind.into();
            let mut environment = Environment {
                id,
                name: arguments.name,
                kind,
                host: arguments.host,
                port: arguments.port,
                database: arguments.database,
                username: arguments.username,
                credential_ref: reference.clone(),
                options: serde_json::json!({ "ssl_mode": arguments.ssl_mode.as_str() }),
                created_at: now,
                updated_at: now,
            };
            if let Some(timeout) = arguments.oracle_call_timeout_ms {
                environment.options["oracle_call_timeout_ms"] = serde_json::json!(timeout);
            }
            match client.request(Operation::EnvironmentAdd(environment)).await {
                Ok(response) => print_response(response, json),
                Err(error) => {
                    let _remove_result = vault.remove(&reference);
                    Err(error.into())
                }
            }
        }
        EnvironmentCommand::Edit(arguments) => {
            let mut environment = resolve_environment(client, &arguments.environment).await?;
            if let Some(name) = arguments.name {
                environment.name = name;
            }
            if let Some(host) = arguments.host {
                environment.host = host;
            }
            if let Some(port) = arguments.port {
                environment.port = port;
            }
            if let Some(database) = arguments.database {
                environment.database = database;
            }
            if let Some(username) = arguments.username {
                environment.username = username;
            }
            if let Some(ssl_mode) = arguments.ssl_mode {
                if !environment.options.is_object() {
                    environment.options = serde_json::json!({});
                }
                environment.options["ssl_mode"] = serde_json::json!(ssl_mode.as_str());
            }
            if let Some(timeout) = arguments.oracle_call_timeout_ms {
                if !environment.options.is_object() {
                    environment.options = serde_json::json!({});
                }
                environment.options["oracle_call_timeout_ms"] = serde_json::json!(timeout);
            }
            environment.validate()?;
            if arguments.password_stdin {
                KeyringVault::default().put(&environment.credential_ref, &read_password(true)?)?;
            }
            environment.updated_at = Utc::now();
            print_response(
                client
                    .request(Operation::EnvironmentEdit(environment))
                    .await?,
                json,
            )
        }
        EnvironmentCommand::List => {
            print_response(client.request(Operation::EnvironmentList).await?, json)
        }
        EnvironmentCommand::Test { environment } => {
            let environment = resolve_environment(client, &environment).await?;
            print_response(
                client
                    .request(Operation::EnvironmentTest(environment.id))
                    .await?,
                json,
            )
        }
        EnvironmentCommand::Remove { environment } => {
            let environment = resolve_environment(client, &environment).await?;
            let vault = KeyringVault::default();
            let previous = vault.get(&environment.credential_ref)?;
            vault.remove(&environment.credential_ref)?;
            match client
                .request(Operation::EnvironmentRemove(environment.id))
                .await
            {
                Ok(response) => print_response(response, json),
                Err(error) => {
                    let _restore_result = vault.put(&environment.credential_ref, &previous);
                    Err(error.into())
                }
            }
        }
    }
}

async fn handle_mask(
    command: MaskCommand,
    client: &DaemonClient,
    json: bool,
) -> anyhow::Result<()> {
    match command {
        MaskCommand::Add(arguments) => {
            let rule = build_rule(MaskRuleId::new(), arguments, client).await?;
            print_response(client.request(Operation::MaskAdd(rule)).await?, json)
        }
        MaskCommand::Edit(arguments) => {
            let id = MaskRuleId::from_str(&arguments.id)?;
            let rule = build_rule(id, arguments.rule, client).await?;
            print_response(client.request(Operation::MaskEdit(rule)).await?, json)
        }
        MaskCommand::List => print_response(client.request(Operation::MaskList).await?, json),
        MaskCommand::Remove { id } => print_response(
            client
                .request(Operation::MaskRemove(MaskRuleId::from_str(&id)?))
                .await?,
            json,
        ),
        MaskCommand::Test(arguments) => {
            let rule = MaskRule {
                id: MaskRuleId::new(),
                name: "preview".into(),
                column: Identifier::new(arguments.column)?,
                algorithm: arguments.algorithm.algorithm(arguments.unmasked_prefix)?,
                environment_id: None,
            };
            print_response(
                client
                    .request(Operation::MaskTest {
                        rule,
                        value: arguments.value,
                    })
                    .await?,
                json,
            )
        }
    }
}

async fn build_rule(
    id: MaskRuleId,
    arguments: MaskArguments,
    client: &DaemonClient,
) -> anyhow::Result<MaskRule> {
    let environment_id = match arguments.environment {
        Some(environment) => Some(resolve_environment(client, &environment).await?.id),
        None => None,
    };
    Ok(MaskRule {
        id,
        name: arguments.name,
        column: Identifier::new(arguments.column)?,
        algorithm: arguments.algorithm.algorithm(arguments.unmasked_prefix)?,
        environment_id,
    })
}

async fn handle_copy(
    command: CopyCommand,
    client: &DaemonClient,
    json: bool,
) -> anyhow::Result<()> {
    let (source, sink, options) = match command {
        CopyCommand::DbToDb(arguments) => {
            let source = database_source(client, arguments.source_database).await?;
            let target = resolve_environment(client, &arguments.target).await?;
            let sink = SinkSpec::Database {
                environment_id: target.id,
                relation: parse_relation(&arguments.target_table)?,
            };
            (source, sink, arguments.transfer)
        }
        CopyCommand::DbToFile(arguments) => (
            database_source(client, arguments.source_database).await?,
            SinkSpec::File {
                path: arguments.output,
                format: arguments.format.into(),
            },
            arguments.transfer,
        ),
        CopyCommand::FileToDb(arguments) => {
            let target = resolve_environment(client, &arguments.target).await?;
            (
                SourceSpec::File {
                    path: arguments.input,
                    format: arguments.format.into(),
                },
                SinkSpec::Database {
                    environment_id: target.id,
                    relation: parse_relation(&arguments.target_table)?,
                },
                arguments.transfer,
            )
        }
    };
    let mut spec = JobSpec::new(source, sink);
    spec.write_mode = options.mode.into();
    spec.consistency = options.consistency.into();
    spec.memory_budget_bytes = options.memory_mib.saturating_mul(1024 * 1024);
    spec.batch_target_bytes = options.batch_mib.saturating_mul(1024 * 1024);
    spec.masks = resolve_mask_rules(client, &options.mask_rules).await?;
    spec.validate()?;
    if options.dry_run {
        let response = client.request(Operation::JobPreview(spec)).await?;
        return print_response(response, json);
    }
    let response = client.request(Operation::JobSubmit(spec)).await?;
    let id = match &response {
        Response::Job(job) => job.spec.id,
        _ => anyhow::bail!("daemon returned an unexpected response to job.submit"),
    };
    print_response(response, json)?;
    if !options.detach {
        watch_job(client, id, json).await?;
    }
    Ok(())
}

/// Resolves `--mask-rule` ids against the daemon's saved masking rules. Fails closed on an
/// unknown id rather than silently submitting a transfer that masks less than the user asked for.
async fn resolve_mask_rules(
    client: &DaemonClient,
    ids: &[String],
) -> anyhow::Result<Vec<MaskRule>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let saved = match client.request(Operation::MaskList).await? {
        Response::Masks(rules) => rules,
        other => anyhow::bail!("expected a masks response to mask.list, got {other:?}"),
    };
    ids.iter()
        .map(|id| {
            let rule_id = MaskRuleId::from_str(id)
                .map_err(|_| anyhow::anyhow!("invalid --mask-rule id: {id}"))?;
            saved
                .iter()
                .find(|rule| rule.id == rule_id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("no saved masking rule with id {id}"))
        })
        .collect()
}

async fn database_source(
    client: &DaemonClient,
    arguments: SourceDatabase,
) -> anyhow::Result<SourceSpec> {
    let environment = resolve_environment(client, &arguments.source).await?;
    let selection = match (arguments.source_table, arguments.query) {
        (Some(table), None) => DatabaseSelection::Table {
            relation: parse_relation(&table)?,
        },
        (None, Some(sql)) => DatabaseSelection::Query { sql },
        _ => anyhow::bail!("exactly one of --source-table and --query is required"),
    };
    let resume_key = arguments
        .resume_key
        .into_iter()
        .map(Identifier::new)
        .collect::<elm_core::Result<Vec<_>>>()?;
    Ok(SourceSpec::Database {
        environment_id: environment.id,
        selection,
        resume_key,
    })
}

async fn handle_jobs(
    command: JobsCommand,
    client: &DaemonClient,
    json: bool,
) -> anyhow::Result<()> {
    match command {
        JobsCommand::List => print_response(client.request(Operation::JobList).await?, json),
        JobsCommand::Show { id } => print_response(
            client
                .request(Operation::JobGet(JobId::from_str(&id)?))
                .await?,
            json,
        ),
        JobsCommand::Watch { id } => watch_job(client, JobId::from_str(&id)?, json).await,
        JobsCommand::Cancel { id } => print_response(
            client
                .request(Operation::JobCancel(JobId::from_str(&id)?))
                .await?,
            json,
        ),
        JobsCommand::Resume { id } => print_response(
            client
                .request(Operation::JobResume(JobId::from_str(&id)?))
                .await?,
            json,
        ),
        JobsCommand::Delete { id } => print_response(
            client
                .request(Operation::JobDelete(JobId::from_str(&id)?))
                .await?,
            json,
        ),
    }
}

async fn watch_job(client: &DaemonClient, id: JobId, json: bool) -> anyhow::Result<()> {
    let mut responses = client.watch(Operation::JobWatch(id)).await?;
    while let Some(response) = responses.recv().await {
        match response? {
            Response::Job(job) => render_progress(&job.progress, json)?,
            Response::Event(progress) => {
                let terminal =
                    progress.state.is_terminal() || progress.state == JobState::Interrupted;
                render_progress(&progress, json)?;
                if terminal {
                    if !json {
                        println!();
                    }
                    if progress.state != JobState::Succeeded {
                        anyhow::bail!("job finished with state {:?}", progress.state);
                    }
                    break;
                }
            }
            response => print_response(response, json)?,
        }
    }
    Ok(())
}

fn render_progress(progress: &JobProgress, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string(progress)?);
    } else {
        eprint!(
            "\r{:?}: {} rows, {} batches, {:.1} MiB/s    ",
            progress.state,
            progress.rows,
            progress.batches,
            progress.bytes_per_second / (1024.0 * 1024.0),
        );
        if let Some(error) = &progress.error {
            eprintln!("\n{}", error.message);
        }
    }
    Ok(())
}

async fn resolve_environment(client: &DaemonClient, value: &str) -> anyhow::Result<Environment> {
    let environments = match client.request(Operation::EnvironmentList).await? {
        Response::Environments(environments) => environments,
        _ => anyhow::bail!("daemon returned an unexpected response to environment.list"),
    };
    if let Ok(id) = EnvironmentId::from_str(value)
        && let Some(environment) = environments.iter().find(|environment| environment.id == id)
    {
        return Ok(environment.clone());
    }
    let matches = environments
        .into_iter()
        .filter(|environment| environment.name.eq_ignore_ascii_case(value))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [environment] => Ok(environment.clone()),
        [] => anyhow::bail!("environment '{value}' was not found"),
        _ => anyhow::bail!("environment name '{value}' is ambiguous; use its id"),
    }
}

fn parse_relation(value: &str) -> anyhow::Result<Relation> {
    let parts = value.split('.').collect::<Vec<_>>();
    Ok(match parts.as_slice() {
        [name] => Relation {
            catalog: None,
            schema: None,
            name: Identifier::new(*name)?,
        },
        [schema, name] => Relation {
            catalog: None,
            schema: Some(Identifier::new(*schema)?),
            name: Identifier::new(*name)?,
        },
        [catalog, schema, name] => Relation {
            catalog: Some(Identifier::new(*catalog)?),
            schema: Some(Identifier::new(*schema)?),
            name: Identifier::new(*name)?,
        },
        _ => anyhow::bail!("relation must be name, schema.name, or catalog.schema.name"),
    })
}

fn read_password(from_stdin: bool) -> anyhow::Result<SecretString> {
    let password = if from_stdin {
        let mut value = String::new();
        std::io::stdin().read_line(&mut value)?;
        value.trim_end_matches(['\r', '\n']).to_owned()
    } else {
        rpassword::prompt_password("Password: ")?
    };
    if password.is_empty() {
        anyhow::bail!("password may not be empty");
    }
    Ok(SecretString::from(password))
}

fn print_response(response: Response, json: bool) -> anyhow::Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(&response)?);
        return Ok(());
    }
    match response {
        Response::Ack => println!("OK"),
        Response::Status { pid, active_jobs } => {
            println!("daemon running (pid {pid}, {active_jobs} active jobs)");
        }
        Response::Environment(environment) => {
            println!(
                "{}\t{}\t{}",
                environment.id, environment.name, environment.kind
            );
        }
        Response::Environments(environments) => {
            for environment in environments {
                println!(
                    "{}\t{}\t{}\t{}:{}\t{}",
                    environment.id,
                    environment.name,
                    environment.kind,
                    environment.host,
                    environment.port,
                    environment.database,
                );
            }
        }
        Response::Masks(rules) => {
            for rule in rules {
                println!(
                    "{}\t{}\t{}\t{:?}",
                    rule.id, rule.name, rule.column, rule.algorithm
                );
            }
        }
        Response::MaskedValue(value) => println!("{value}"),
        Response::Settings(settings) => println!(
            "memory={} MiB\tbatch={} MiB",
            settings.memory_budget_bytes / (1024 * 1024),
            settings.batch_target_bytes / (1024 * 1024)
        ),
        Response::Job(job) => println!(
            "{}\t{:?}\t{} rows\tattempt {}",
            job.spec.id, job.progress.state, job.progress.rows, job.attempt
        ),
        Response::Jobs(jobs) => {
            for job in jobs {
                println!(
                    "{}\t{:?}\t{} rows\t{}",
                    job.spec.id, job.progress.state, job.progress.rows, job.updated_at
                );
            }
        }
        Response::Event(progress) => render_progress(&progress, false)?,
        Response::Preview(preview) => {
            for column in &preview.columns {
                println!(
                    "{}\t{} -> {}\t{}",
                    column.name,
                    column.source_type,
                    column.output_type,
                    if column.nullable { "null" } else { "not null" }
                );
            }
            for warning in &preview.report.warnings {
                println!("warning: {warning}");
            }
            if let Some(error) = &preview.publication_error {
                println!("unsafe: {error}");
            } else {
                println!("publication is safe for the requested write mode and consistency");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod timeout_tests {
    use super::*;

    #[test]
    fn cli_timeout_accepts_only_bounded_integer_values() {
        for value in ["1", "120000", "300000"] {
            assert!(
                Cli::try_parse_from([
                    "elm",
                    "env",
                    "edit",
                    "test",
                    "--oracle-call-timeout-ms",
                    value
                ])
                .is_ok()
            );
        }
        for value in ["0", "300001", "-1", "1.5", "never"] {
            assert!(
                Cli::try_parse_from([
                    "elm",
                    "env",
                    "edit",
                    "test",
                    "--oracle-call-timeout-ms",
                    value
                ])
                .is_err()
            );
        }
    }
}
