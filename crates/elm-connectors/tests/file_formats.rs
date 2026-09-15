use std::{fs, sync::Arc};

use arrow_array::{
    ArrayRef, BinaryArray, Decimal128Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use elm_connectors::{FileSink, FileSource, FileSourceCheckpoint};
use elm_core::{ConsistencyMode, DataSink, DataSource, FileFormat, WriteMode};
use tempfile::tempdir;

fn canonical_batch() -> RecordBatch {
    let decimal = Decimal128Array::from(vec![
        Some(12_345_678_901_234_567_890_123_456_789_i128),
        Some(-1),
    ])
    .with_precision_and_scale(38, 9)
    .unwrap_or_else(|error| panic!("{error}"));
    let timestamp =
        TimestampNanosecondArray::from(vec![Some(1_788_528_896_123_456_789_i64), Some(0)])
            .with_timezone("UTC");
    let lob = format!("{}終", "x".repeat(1024 * 1024));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![1, 2])),
        Arc::new(StringArray::from(vec![None, Some("İstanbul 🌍")])),
        Arc::new(decimal),
        Arc::new(timestamp),
        Arc::new(BinaryArray::from(vec![Some(&b"\0\xff\x10"[..]), None])),
        Arc::new(LargeStringArray::from(vec![
            Some(lob.as_str()),
            Some("narrow"),
        ])),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("unicode", DataType::Utf8, true),
        Field::new("decimal_38_9", DataType::Decimal128(38, 9), true),
        Field::new(
            "timestamp_utc",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true,
        ),
        Field::new("binary", DataType::Binary, true),
        Field::new("lob", DataType::LargeUtf8, true),
    ]));
    RecordBatch::try_new(schema, arrays).unwrap_or_else(|error| panic!("{error}"))
}

#[tokio::test]
async fn parquet_preserves_canonical_values_and_schema() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("canonical.parquet");
    let expected = canonical_batch();
    let mut sink = FileSink::new(&path, FileFormat::Parquet);
    sink.preflight(
        expected.schema(),
        WriteMode::Replace,
        ConsistencyMode::Atomic,
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&expected)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));

    let mut source =
        FileSource::open(path, FileFormat::Parquet).unwrap_or_else(|error| panic!("{error}"));
    let actual = source
        .next_batch(8 * 1024 * 1024)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected parquet batch"));
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn abort_never_replaces_the_visible_target() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("target.ndjson");
    fs::write(&path, "{\"original\":true}\n").unwrap_or_else(|error| panic!("{error}"));
    let batch = RecordBatch::try_from_iter(vec![(
        "original",
        Arc::new(arrow_array::BooleanArray::from(vec![false])) as ArrayRef,
    )])
    .unwrap_or_else(|error| panic!("{error}"));
    let mut sink = FileSink::new(&path, FileFormat::Ndjson);
    // REPLACE does not need the existing file's unrelated schema; it remains visible until publish.
    sink.preflight(batch.schema(), WriteMode::Replace, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.abort().await.unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        fs::read_to_string(path).unwrap_or_default(),
        "{\"original\":true}\n"
    );
}

#[tokio::test]
async fn append_inserts_a_missing_newline_only_in_staging() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("target.ndjson");
    fs::write(&path, "{\"id\":1}").unwrap_or_else(|error| panic!("{error}"));
    let batch = RecordBatch::try_from_iter(vec![(
        "id",
        Arc::new(Int64Array::from(vec![2])) as ArrayRef,
    )])
    .unwrap_or_else(|error| panic!("{error}"));
    let mut sink = FileSink::new(&path, FileFormat::Ndjson);
    sink.preflight(batch.schema(), WriteMode::Append, ConsistencyMode::Atomic)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
    sink.write_batch(&batch)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    sink.publish()
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        fs::read_to_string(path).unwrap_or_default(),
        "{\"id\":1}\n{\"id\":2}\n"
    );
}

#[tokio::test]
async fn resume_uses_a_logical_row_offset_and_rejects_a_changed_file() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let path = directory.path().join("resume.csv");
    fs::write(&path, "id,name\n1,one\n2,two\n3,three\n").unwrap_or_else(|error| panic!("{error}"));

    let mut source =
        FileSource::open(&path, FileFormat::Csv).unwrap_or_else(|error| panic!("{error}"));
    let first = source
        .next_batch(1)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected the first row"));
    assert_eq!(first.num_rows(), 1);
    let checkpoint: FileSourceCheckpoint = serde_json::from_value(
        source
            .checkpoint()
            .await
            .unwrap_or_else(|error| panic!("{error}")),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(checkpoint.logical_row_offset, 1);

    let mut resumed = FileSource::resume(&path, FileFormat::Csv, &checkpoint)
        .unwrap_or_else(|error| panic!("{error}"));
    let remaining = resumed
        .next_batch(usize::MAX)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|| panic!("expected remaining rows"));
    let ids = remaining
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| panic!("expected inferred Int64 values"));
    assert_eq!(ids.values(), &[2, 3]);

    fs::write(&path, "id,name\n1,changed\n2,two\n3,three\n")
        .unwrap_or_else(|error| panic!("{error}"));
    assert!(FileSource::resume(&path, FileFormat::Csv, &checkpoint).is_err());
}
