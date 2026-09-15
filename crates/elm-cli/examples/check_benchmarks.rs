//! Read-only validation of reviewed 10 GB benchmark reports, not a benchmark runner.
use std::{collections::BTreeSet, path::PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Deserialize;

const CONNECTORS: [&str; 4] = ["postgresql", "oracle", "mysql", "sql_server"];
const MEMORY_BUDGET: u64 = 512 * 1024 * 1024;

#[derive(Parser)]
struct Arguments {
    report: PathBuf,
    /// Previously reviewed Rust baseline on the identical host and dataset.
    #[arg(long)]
    previous: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema_version: u32,
    captured_at: chrono::DateTime<chrono::FixedOffset>,
    host: Host,
    dataset: Dataset,
    results: Vec<Measurement>,
}

#[derive(Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Host {
    cpu: String,
    memory_bytes: u64,
    storage: String,
    operating_system: String,
}

#[derive(Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    generator_version: String,
    digest: String,
    logical_bytes: u64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurement {
    connector: String,
    python_rows_per_second: f64,
    rust_rows_per_second: f64,
    rust_peak_rss_bytes: u64,
    #[serde(default)]
    native_client_allowance_bytes: u64,
    #[serde(default)]
    native_client_allowance_evidence: Option<String>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate(report: &Report, previous: Option<&Report>) -> Result<()> {
    ensure!(
        report.schema_version == 1,
        "unsupported report schema version"
    );
    ensure!(
        [
            &report.host.cpu,
            &report.host.storage,
            &report.host.operating_system,
            &report.dataset.generator_version,
            &report.dataset.digest
        ]
        .into_iter()
        .all(|v| nonempty(v)),
        "host and dataset provenance must be nonempty"
    );
    ensure!(report.host.memory_bytes > 0, "host memory must be positive");
    ensure!(
        report.dataset.logical_bytes == 10_000_000_000,
        "requires the fixed 10 GB dataset"
    );
    let mut seen = BTreeSet::new();
    for result in &report.results {
        ensure!(
            CONNECTORS.contains(&result.connector.as_str()),
            "unknown connector"
        );
        ensure!(
            seen.insert(result.connector.as_str()),
            "duplicate connector"
        );
        let connector = &result.connector;
        for rate in [result.python_rows_per_second, result.rust_rows_per_second] {
            ensure!(
                rate.is_finite() && rate > 0.0,
                "{connector}: throughput must be finite and positive"
            );
        }
        ensure!(
            result.rust_rows_per_second / 2.0 >= result.python_rows_per_second,
            "{connector}: below the 2x Python throughput gate"
        );
        ensure!(
            result.rust_peak_rss_bytes > 0,
            "{connector}: peak RSS is missing"
        );
        if result.native_client_allowance_bytes > 0 {
            ensure!(
                result
                    .native_client_allowance_evidence
                    .as_deref()
                    .is_some_and(nonempty),
                "{connector}: native memory allowance requires reviewed evidence"
            );
        }
        let limit = MEMORY_BUDGET
            .checked_add(result.native_client_allowance_bytes)
            .context("native memory allowance overflows the RSS limit")?;
        ensure!(
            result.rust_peak_rss_bytes <= limit,
            "{connector}: exceeds the RSS budget"
        );
    }
    ensure!(
        seen.len() == CONNECTORS.len(),
        "all four connectors are required"
    );
    if let Some(previous) = previous {
        validate(previous, None).context("invalid previous baseline")?;
        ensure!(
            report.host == previous.host && report.dataset == previous.dataset,
            "previous baseline host or dataset differs"
        );
        ensure!(
            report.captured_at >= previous.captured_at,
            "candidate predates previous baseline"
        );
        for result in &report.results {
            let old = previous
                .results
                .iter()
                .find(|old| old.connector == result.connector)
                .context("previous connector missing")?;
            ensure!(
                result.python_rows_per_second == old.python_rows_per_second,
                "frozen Python baseline changed"
            );
            ensure!(
                result.native_client_allowance_bytes == old.native_client_allowance_bytes,
                "native memory allowance changed since baseline"
            );
            ensure!(
                result.rust_rows_per_second >= old.rust_rows_per_second * 0.9,
                "{}: exceeds the 10% Rust regression limit",
                result.connector
            );
        }
    }
    Ok(())
}

fn read_report(path: &std::path::Path) -> Result<Report> {
    let file = std::fs::File::open(path).context("cannot open benchmark report")?;
    // Do not print submitted values: report content may contain accidental secrets.
    serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|_| anyhow::anyhow!("invalid or incomplete benchmark report"))
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let report = read_report(&args.report)?;
    let previous = args.previous.as_deref().map(read_report).transpose()?;
    validate(&report, previous.as_ref())?;
    println!("10 GB throughput/RSS report gates passed for all four connectors.");
    if previous.is_none() {
        println!("Regression gate NOT evaluated: no --previous baseline supplied.");
    }
    println!(
        "Measurements require independent review; soak, correctness, and release gates are separate."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Report {
        // Synthetic numbers only exercise the checker; these are not performance evidence.
        Report {
            schema_version: 1,
            captured_at: chrono::DateTime::parse_from_rfc3339("2026-09-15T00:00:00Z")
                .unwrap_or_else(|error| panic!("{error}")),
            host: Host {
                cpu: "synthetic".into(),
                memory_bytes: 1,
                storage: "synthetic".into(),
                operating_system: "synthetic".into(),
            },
            dataset: Dataset {
                generator_version: "synthetic".into(),
                digest: "synthetic".into(),
                logical_bytes: 10_000_000_000,
            },
            results: CONNECTORS
                .iter()
                .map(|name| Measurement {
                    connector: (*name).into(),
                    python_rows_per_second: 100.0,
                    rust_rows_per_second: 300.0,
                    rust_peak_rss_bytes: MEMORY_BUDGET,
                    native_client_allowance_bytes: 0,
                    native_client_allowance_evidence: None,
                })
                .collect(),
        }
    }

    #[test]
    fn incomplete_template_is_not_evidence() {
        assert!(
            serde_json::from_str::<Report>(include_str!(
                "../../../benchmarks/baseline-template.json"
            ))
            .is_err()
        );
    }

    #[test]
    fn exact_thresholds_pass_but_regression_and_over_budget_fail() {
        let previous = fixture();
        let mut report = previous.clone();
        report.results[0].rust_rows_per_second = 270.0;
        assert!(validate(&report, Some(&previous)).is_ok());
        report.results[0].rust_rows_per_second = 269.99;
        assert!(validate(&report, Some(&previous)).is_err());
        report.results[0].rust_rows_per_second = 200.0;
        assert!(validate(&report, None).is_ok());
        report.results[0].rust_rows_per_second = 199.99;
        assert!(validate(&report, None).is_err());
        report = fixture();
        report.results[0].rust_peak_rss_bytes += 1;
        assert!(validate(&report, None).is_err());
    }

    #[test]
    fn rejects_missing_duplicate_and_invalid_measurements() {
        let mut report = fixture();
        report.results.pop();
        assert!(validate(&report, None).is_err());
        report = fixture();
        report.results[1] = report.results[0].clone();
        assert!(validate(&report, None).is_err());
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            report = fixture();
            report.results[0].rust_rows_per_second = rate;
            assert!(validate(&report, None).is_err());
        }
    }

    #[test]
    fn rejects_incomparable_baselines_and_undocumented_allowances() {
        let previous = fixture();
        let mut report = fixture();
        report.host.cpu = "different".into();
        assert!(validate(&report, Some(&previous)).is_err());
        report = fixture();
        report.results[0].native_client_allowance_bytes = 1;
        assert!(validate(&report, None).is_err());
        report.results[0].native_client_allowance_evidence =
            Some("reviewed evidence reference".into());
        assert!(validate(&report, None).is_ok());
        assert!(validate(&report, Some(&previous)).is_err());
        report.results[0].native_client_allowance_bytes = u64::MAX;
        assert!(validate(&report, None).is_err());
    }
}
