use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use elm_connectors::RecoverableFileSink;
use elm_core::{BatchCheckpoint, ConsistencyMode, DataSink, ElmError, FileFormat, WriteMode};

#[tokio::test]
#[ignore = "requires a size-limited filesystem mounted at ELM_TEST_TINY_DISK_DIR"]
async fn recoverable_file_sink_fails_closed_when_staging_runs_out_of_space() {
    let tiny_dir = std::env::var("ELM_TEST_TINY_DISK_DIR")
        .unwrap_or_else(|_| panic!("set ELM_TEST_TINY_DISK_DIR to a size-limited mount point"));
    let staging_directory = std::path::Path::new(&tiny_dir).join("staging");
    let output_directory = tempfile::tempdir().unwrap_or_else(|error| panic!("{error}"));
    let output_path = output_directory.path().join("output.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new(
        "payload",
        DataType::Binary,
        false,
    )]));
    let mut sink =
        RecoverableFileSink::new(&output_path, FileFormat::Parquet, &staging_directory, None)
            .unwrap_or_else(|error| panic!("{error}"));
    sink.preflight(schema.clone(), WriteMode::Replace, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));

    // Each batch is ~512 KiB of incompressible binary data; the mounted filesystem is capped
    // at a few MiB (see tests/full-disk/compose.yml), so this must exhaust it well before an
    // unreasonable number of iterations.
    let payload = vec![0xAB_u8; 512 * 1024];
    let mut sequence = 0_u64;
    let mut exhausted = false;
    for _ in 0..64 {
        let array: ArrayRef = Arc::new(BinaryArray::from(vec![Some(payload.as_slice())]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array])
            .unwrap_or_else(|error| panic!("{error}"));
        match sink.write_batch(&batch).await {
            Ok(()) => {
                sink.commit_checkpoint(&BatchCheckpoint {
                    sequence,
                    source_position: serde_json::json!({ "row": sequence }),
                    rows_committed: sequence + 1,
                    bytes_committed: 0,
                    source_fingerprint: None,
                })
                .await
                .unwrap_or_else(|error| panic!("{error}"));
                sequence += 1;
            }
            Err(error) => {
                assert!(
                    matches!(error, ElmError::Io(_)),
                    "expected a clean I/O error when staging space is exhausted, got {error:?}"
                );
                exhausted = true;
                break;
            }
        }
    }
    assert!(
        exhausted,
        "the size-limited filesystem never reported exhaustion; increase payload size or iteration count"
    );
    assert!(
        !output_path.exists(),
        "a failed staging write must never produce a partial destination file"
    );
    let _abort_result = sink.abort().await;
}
