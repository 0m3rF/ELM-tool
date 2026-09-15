use std::{
    collections::{BTreeSet, VecDeque},
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::UNIX_EPOCH,
};

use arrow_array::RecordBatch;
use arrow_csv::{ReaderBuilder as CsvReaderBuilder, WriterBuilder as CsvWriterBuilder};
use arrow_json::{
    LineDelimitedWriter,
    reader::{ReaderBuilder as JsonReaderBuilder, infer_json_schema_from_seekable},
};
use arrow_schema::Schema;
use async_trait::async_trait;
use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use serde_json::json;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

use elm_core::{
    BatchCheckpoint, ConnectorCapabilities, ConsistencyMode, DataSink, DataSource, ElmError,
    FileFormat, PreflightReport, Result, WriteMode,
};

const INFERENCE_ROWS: usize = 10_000;
const INITIAL_ROW_BATCH_SIZE: usize = 8_192;

type CsvReader = arrow_csv::Reader<File>;
type JsonReader = arrow_json::Reader<BufReader<File>>;
type ParquetReader = parquet::arrow::arrow_reader::ParquetRecordBatchReader;

enum ReaderKind {
    Csv(CsvReader),
    Ndjson(JsonReader),
    Parquet(ParquetReader),
}

pub struct FileSource {
    schema: Arc<Schema>,
    reader: ReaderKind,
    rows_read: u64,
    fingerprint: String,
    pending: VecDeque<RecordBatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileSourceCheckpoint {
    pub logical_row_offset: u64,
    pub fingerprint: String,
}

impl FileSource {
    pub fn open(path: impl Into<PathBuf>, format: FileFormat) -> Result<Self> {
        let path = path.into();
        let fingerprint = fingerprint_file(&path)?;
        let (schema, reader) = build_reader(&path, format)?;
        Ok(Self {
            schema,
            reader,
            rows_read: 0,
            fingerprint,
            pending: VecDeque::new(),
        })
    }

    pub fn resume(
        path: impl Into<PathBuf>,
        format: FileFormat,
        checkpoint: &FileSourceCheckpoint,
    ) -> Result<Self> {
        let mut source = Self::open(path, format)?;
        if source.fingerprint != checkpoint.fingerprint {
            return Err(ElmError::Conflict(
                "input file changed since the checkpoint was recorded".into(),
            ));
        }
        let mut remaining = checkpoint.logical_row_offset;
        while remaining > 0 {
            let batch = source.read_next()?.ok_or_else(|| {
                ElmError::Conflict("checkpoint is beyond the end of the input file".into())
            })?;
            let rows = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
            if rows > remaining {
                let consumed = usize::try_from(remaining).map_err(|_| {
                    ElmError::Conflict("checkpoint row offset exceeds platform limits".into())
                })?;
                source
                    .pending
                    .push_back(batch.slice(consumed, batch.num_rows() - consumed));
                source.rows_read = source.rows_read.saturating_add(remaining);
                remaining = 0;
            } else {
                remaining -= rows;
                source.rows_read = source.rows_read.saturating_add(rows);
            }
        }
        Ok(source)
    }

    fn read_next(&mut self) -> Result<Option<RecordBatch>> {
        let result = match &mut self.reader {
            ReaderKind::Csv(reader) => reader.next(),
            ReaderKind::Ndjson(reader) => reader.next(),
            ReaderKind::Parquet(reader) => reader.next(),
        };
        result
            .transpose()
            .map_err(|error| ElmError::Io(std::io::Error::other(error)))
    }
}

#[async_trait]
impl DataSource for FileSource {
    async fn schema(&mut self) -> Result<Arc<Schema>> {
        Ok(self.schema.clone())
    }

    async fn next_batch(&mut self, target_bytes: usize) -> Result<Option<RecordBatch>> {
        let batch = match self.pending.pop_front() {
            Some(batch) => Some(batch),
            None => self.read_next()?,
        };
        let batch = batch.map(|batch| self.split_for_target(batch, target_bytes));
        if let Some(batch) = &batch {
            self.rows_read = self
                .rows_read
                .saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX));
        }
        Ok(batch)
    }

    async fn checkpoint(&mut self) -> Result<serde_json::Value> {
        Ok(json!(FileSourceCheckpoint {
            logical_row_offset: self.rows_read,
            fingerprint: self.fingerprint.clone(),
        }))
    }
}

impl FileSource {
    fn split_for_target(&mut self, batch: RecordBatch, target_bytes: usize) -> RecordBatch {
        let bytes = batch.get_array_memory_size();
        if bytes <= target_bytes || batch.num_rows() <= 1 || target_bytes == 0 {
            return batch;
        }
        let rows = batch.num_rows().saturating_mul(target_bytes) / bytes;
        let rows = rows.clamp(1, batch.num_rows() - 1);
        self.pending
            .push_front(batch.slice(rows, batch.num_rows() - rows));
        batch.slice(0, rows)
    }
}

fn build_reader(path: &Path, format: FileFormat) -> Result<(Arc<Schema>, ReaderKind)> {
    match format {
        FileFormat::Csv => {
            let mut file = File::open(path)?;
            let format = arrow_csv::reader::Format::default().with_header(true);
            let (schema, _) = format
                .infer_schema(&mut file, Some(INFERENCE_ROWS))
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            file.rewind()?;
            let schema = Arc::new(schema);
            let reader = CsvReaderBuilder::new(schema.clone())
                .with_format(format)
                .with_batch_size(INITIAL_ROW_BATCH_SIZE)
                .build(file)
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            Ok((schema, ReaderKind::Csv(reader)))
        }
        FileFormat::Ndjson => {
            let file = File::open(path)?;
            let mut input = BufReader::new(file);
            let (schema, _) = infer_json_schema_from_seekable(&mut input, Some(INFERENCE_ROWS))
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            let schema = Arc::new(schema);
            let reader = JsonReaderBuilder::new(schema.clone())
                .with_batch_size(INITIAL_ROW_BATCH_SIZE)
                .build(input)
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            Ok((schema, ReaderKind::Ndjson(reader)))
        }
        FileFormat::Parquet => {
            let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?
                .with_batch_size(INITIAL_ROW_BATCH_SIZE);
            let schema = builder.schema().clone();
            let reader = builder
                .build()
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            Ok((schema, ReaderKind::Parquet(reader)))
        }
    }
}

enum WriterKind {
    Csv(arrow_csv::Writer<File>),
    Ndjson(LineDelimitedWriter<File>),
    Parquet(ArrowWriter<File>),
}

pub struct FileSink {
    path: PathBuf,
    format: FileFormat,
    schema: Option<Arc<Schema>>,
    mode: Option<WriteMode>,
    staging: Option<NamedTempFile>,
    writer: Option<WriterKind>,
    begun: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct FileStagingManifest {
    version: u16,
    destination: PathBuf,
    format: FileFormat,
    schema: Schema,
    write_mode: WriteMode,
}

/// A file sink whose committed batches survive daemon termination.
///
/// Batches are persisted as individually valid Parquet chunks. Publication streams those chunks
/// through `FileSink`, retaining its sibling-file atomic replacement behavior.
pub struct RecoverableFileSink {
    path: PathBuf,
    format: FileFormat,
    staging_directory: PathBuf,
    resume_sequence: Option<u64>,
    schema: Option<Arc<Schema>>,
    mode: Option<WriteMode>,
    next_sequence: u64,
    pending_chunk: Option<PathBuf>,
    begun: bool,
}

impl RecoverableFileSink {
    pub fn new(
        path: impl Into<PathBuf>,
        format: FileFormat,
        staging_directory: impl Into<PathBuf>,
        resume_checkpoint: Option<&BatchCheckpoint>,
    ) -> Result<Self> {
        let resume_sequence = resume_checkpoint.map(|checkpoint| checkpoint.sequence);
        let next_sequence = match resume_sequence {
            Some(sequence) => sequence
                .checked_add(1)
                .ok_or_else(|| ElmError::State("checkpoint sequence cannot be resumed".into()))?,
            None => 0,
        };
        Ok(Self {
            path: path.into(),
            format,
            staging_directory: staging_directory.into(),
            resume_sequence,
            schema: None,
            mode: None,
            next_sequence,
            pending_chunk: None,
            begun: false,
        })
    }

    fn manifest(&self, schema: &Schema, mode: WriteMode) -> FileStagingManifest {
        FileStagingManifest {
            version: 1,
            destination: self.path.clone(),
            format: self.format,
            schema: schema.clone(),
            write_mode: mode,
        }
    }

    fn manifest_path(&self) -> PathBuf {
        self.staging_directory.join("manifest.json")
    }

    fn prepare_staging(&self, expected: &FileStagingManifest) -> Result<()> {
        let manifest_path = self.manifest_path();
        if let Some(resume_sequence) = self.resume_sequence {
            let actual = read_staging_manifest(&manifest_path)?;
            validate_staging_manifest(&actual, expected)?;
            validate_and_trim_chunks(&self.staging_directory, resume_sequence)?;
            return Ok(());
        }

        if manifest_path.exists() {
            let actual = read_staging_manifest(&manifest_path)?;
            validate_staging_manifest(&actual, expected)?;
            fs::remove_dir_all(&self.staging_directory)?;
        } else if self.staging_directory.exists()
            && self.staging_directory.read_dir()?.next().is_some()
        {
            return Err(ElmError::Conflict(format!(
                "unrecognized staging directory '{}' is not safe to replace",
                self.staging_directory.display()
            )));
        }

        fs::create_dir_all(&self.staging_directory)?;
        let mut temporary = NamedTempFile::new_in(&self.staging_directory)?;
        serde_json::to_writer(temporary.as_file_mut(), expected)
            .map_err(|error| ElmError::State(format!("cannot encode staging manifest: {error}")))?;
        temporary.as_file_mut().sync_all()?;
        temporary
            .persist_noclobber(manifest_path)
            .map_err(|error| ElmError::Io(error.error))?;
        Ok(())
    }

    fn chunk_path(&self, sequence: u64) -> PathBuf {
        self.staging_directory
            .join(format!("chunk-{sequence:020}.parquet"))
    }
}

impl FileSink {
    #[must_use]
    pub fn new(path: impl Into<PathBuf>, format: FileFormat) -> Self {
        Self {
            path: path.into(),
            format,
            schema: None,
            mode: None,
            staging: None,
            writer: None,
            begun: false,
        }
    }

    fn close_writer(&mut self) -> Result<()> {
        let Some(writer) = self.writer.take() else {
            return Ok(());
        };
        match writer {
            WriterKind::Csv(writer) => writer.into_inner().sync_all().map_err(ElmError::Io),
            WriterKind::Ndjson(mut writer) => {
                writer
                    .finish()
                    .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
                writer.into_inner().sync_all().map_err(ElmError::Io)
            }
            WriterKind::Parquet(writer) => writer
                .into_inner()
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?
                .sync_all()
                .map_err(ElmError::Io),
        }
    }
}

#[async_trait]
impl DataSink for FileSink {
    async fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        _consistency: ConsistencyMode,
    ) -> Result<PreflightReport> {
        if mode == WriteMode::Fail && self.path.exists() {
            return Err(ElmError::Conflict(format!(
                "destination '{}' already exists",
                self.path.display()
            )));
        }
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
        {
            return Err(ElmError::NotFound(format!(
                "destination directory '{}' does not exist",
                parent.display()
            )));
        }
        if mode == WriteMode::Append && self.path.exists() {
            let (existing_schema, _) = build_reader(&self.path, self.format)?;
            if !append_schema_compatible(&existing_schema, &schema) {
                return Err(ElmError::TypeMapping(format!(
                    "append schema does not match existing destination '{}'",
                    self.path.display()
                )));
            }
        }
        self.schema = Some(schema);
        self.mode = Some(mode);
        Ok(PreflightReport {
            capabilities: ConnectorCapabilities {
                database: None,
                native_bulk_read: false,
                native_bulk_write: false,
                atomic_append: true,
                atomic_replace: true,
                non_atomic_replace: false,
                checkpointed_write: false,
                resumable_keyset_read: false,
                supports_lob_spill: false,
            },
            staging_relation: None,
            warnings: Vec::new(),
        })
    }

    async fn begin(&mut self) -> Result<()> {
        if self.begun {
            return Err(ElmError::Conflict("file sink has already begun".into()));
        }
        let schema = self
            .schema
            .clone()
            .ok_or_else(|| ElmError::Internal("file sink was not preflighted".into()))?;
        let parent = self
            .path
            .parent()
            .filter(|path| !path.as_os_str().is_empty());
        let staging = match parent {
            Some(parent) => NamedTempFile::new_in(parent)?,
            None => NamedTempFile::new()?,
        };
        let mut output = staging.as_file().try_clone()?;
        let appending = self.mode == Some(WriteMode::Append) && self.path.exists();

        if appending && matches!(self.format, FileFormat::Csv | FileFormat::Ndjson) {
            std::io::copy(&mut File::open(&self.path)?, &mut output)?;
            let mut input = File::open(&self.path)?;
            if input.metadata()?.len() > 0 {
                input.seek(SeekFrom::End(-1))?;
                let mut last = [0_u8; 1];
                input.read_exact(&mut last)?;
                if last[0] != b'\n' {
                    output.write_all(b"\n")?;
                }
            }
        }

        let mut writer = match self.format {
            FileFormat::Csv => WriterKind::Csv(
                CsvWriterBuilder::new()
                    .with_header(!appending)
                    .build(output),
            ),
            FileFormat::Ndjson => WriterKind::Ndjson(LineDelimitedWriter::new(output)),
            FileFormat::Parquet => {
                let mut writer = ArrowWriter::try_new(output, schema.clone(), None)
                    .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
                if appending {
                    let (_, mut existing) = build_reader(&self.path, FileFormat::Parquet)?;
                    while let ReaderKind::Parquet(reader) = &mut existing {
                        let Some(batch) = reader.next() else {
                            break;
                        };
                        writer
                            .write(
                                &batch
                                    .map_err(|error| ElmError::Io(std::io::Error::other(error)))?,
                            )
                            .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
                    }
                }
                WriterKind::Parquet(writer)
            }
        };
        if !appending {
            write_record_batch(&mut writer, &RecordBatch::new_empty(schema))?;
        }
        self.staging = Some(staging);
        self.writer = Some(writer);
        self.begun = true;
        Ok(())
    }

    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        match self.writer.as_mut() {
            Some(writer) => write_record_batch(writer, batch),
            None => Err(ElmError::Internal("file sink has not begun".into())),
        }
    }

    async fn commit_checkpoint(&mut self, _checkpoint: &BatchCheckpoint) -> Result<()> {
        // The staged file is durable at publication. A partially written file is never exposed.
        Ok(())
    }

    async fn publish(&mut self) -> Result<()> {
        self.close_writer()?;
        let staging = self
            .staging
            .take()
            .ok_or_else(|| ElmError::Internal("file sink has no staging file".into()))?;
        staging
            .persist(&self.path)
            .map_err(|error| ElmError::Io(error.error))?;
        Ok(())
    }

    async fn abort(&mut self) -> Result<()> {
        self.writer.take();
        self.staging.take();
        Ok(())
    }
}

#[async_trait]
impl DataSink for RecoverableFileSink {
    async fn preflight(
        &mut self,
        schema: Arc<Schema>,
        mode: WriteMode,
        consistency: ConsistencyMode,
    ) -> Result<PreflightReport> {
        let mut validator = FileSink::new(&self.path, self.format);
        let report = validator
            .preflight(schema.clone(), mode, consistency)
            .await?;
        self.schema = Some(schema);
        self.mode = Some(mode);
        Ok(report)
    }

    async fn begin(&mut self) -> Result<()> {
        if self.begun {
            return Err(ElmError::Conflict(
                "recoverable file sink has already begun".into(),
            ));
        }
        let schema = self
            .schema
            .as_deref()
            .ok_or_else(|| ElmError::Internal("file sink was not preflighted".into()))?;
        let mode = self
            .mode
            .ok_or_else(|| ElmError::Internal("file sink was not preflighted".into()))?;
        self.prepare_staging(&self.manifest(schema, mode))?;
        self.begun = true;
        Ok(())
    }

    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if !self.begun {
            return Err(ElmError::Internal("file sink has not begun".into()));
        }
        if self.pending_chunk.is_some() {
            return Err(ElmError::Conflict(
                "the previous staged batch has not been checkpointed".into(),
            ));
        }
        let schema = self
            .schema
            .as_ref()
            .ok_or_else(|| ElmError::Internal("file sink was not preflighted".into()))?;
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(ElmError::TypeMapping(
                "staged batch schema differs from the preflight schema".into(),
            ));
        }
        let final_path = self.chunk_path(self.next_sequence);
        if final_path.exists() {
            return Err(ElmError::Conflict(format!(
                "staging chunk {} already exists",
                self.next_sequence
            )));
        }
        let temporary = TempFileBuilder::new()
            .prefix("chunk-pending-")
            .suffix(".parquet")
            .tempfile_in(&self.staging_directory)?;
        let output = temporary.as_file().try_clone()?;
        let mut writer = ArrowWriter::try_new(output, schema.clone(), None)
            .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
        writer
            .write(batch)
            .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
        writer
            .into_inner()
            .map_err(|error| ElmError::Io(std::io::Error::other(error)))?
            .sync_all()?;
        temporary
            .persist_noclobber(&final_path)
            .map_err(|error| ElmError::Io(error.error))?;
        self.pending_chunk = Some(final_path);
        Ok(())
    }

    async fn commit_checkpoint(&mut self, checkpoint: &BatchCheckpoint) -> Result<()> {
        if checkpoint.sequence != self.next_sequence {
            return Err(ElmError::State(format!(
                "sink expected checkpoint sequence {}, received {}",
                self.next_sequence, checkpoint.sequence
            )));
        }
        let chunk = self
            .pending_chunk
            .take()
            .ok_or_else(|| ElmError::State("checkpoint has no durable staging chunk".into()))?;
        if !chunk.exists() {
            return Err(ElmError::State(
                "durable staging chunk disappeared before checkpoint commit".into(),
            ));
        }
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| ElmError::State("checkpoint sequence overflow".into()))?;
        Ok(())
    }

    async fn publish(&mut self) -> Result<()> {
        if self.pending_chunk.is_some() {
            return Err(ElmError::State(
                "cannot publish an uncheckpointed staging chunk".into(),
            ));
        }
        let schema = self
            .schema
            .clone()
            .ok_or_else(|| ElmError::Internal("file sink was not preflighted".into()))?;
        let mode = self
            .mode
            .ok_or_else(|| ElmError::Internal("file sink was not preflighted".into()))?;
        let mut publisher = FileSink::new(&self.path, self.format);
        publisher
            .preflight(schema, mode, ConsistencyMode::Atomic)
            .await?;
        publisher.begin().await?;
        for sequence in 0..self.next_sequence {
            let path = self.chunk_path(sequence);
            let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            let reader = builder
                .build()
                .map_err(|error| ElmError::Io(std::io::Error::other(error)))?;
            for batch in reader {
                publisher
                    .write_batch(
                        &batch.map_err(|error| ElmError::Io(std::io::Error::other(error)))?,
                    )
                    .await?;
            }
        }
        publisher.publish().await?;
        let _cleanup_result = fs::remove_dir_all(&self.staging_directory);
        Ok(())
    }

    async fn abort(&mut self) -> Result<()> {
        if let Some(path) = self.pending_chunk.take() {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

fn write_record_batch(writer: &mut WriterKind, batch: &RecordBatch) -> Result<()> {
    match writer {
        WriterKind::Csv(writer) => writer
            .write(batch)
            .map_err(|error| ElmError::Io(std::io::Error::other(error))),
        WriterKind::Ndjson(writer) => writer
            .write(batch)
            .map_err(|error| ElmError::Io(std::io::Error::other(error))),
        WriterKind::Parquet(writer) => writer
            .write(batch)
            .map_err(|error| ElmError::Io(std::io::Error::other(error))),
    }
}

fn read_staging_manifest(path: &Path) -> Result<FileStagingManifest> {
    let contents = fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ElmError::Conflict("the durable staging manifest is missing".into())
        } else {
            ElmError::Io(error)
        }
    })?;
    serde_json::from_str(&contents)
        .map_err(|error| ElmError::State(format!("invalid staging manifest: {error}")))
}

fn validate_staging_manifest(
    actual: &FileStagingManifest,
    expected: &FileStagingManifest,
) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(ElmError::Conflict(
        "durable staging metadata does not match this job attempt".into(),
    ))
}

fn validate_and_trim_chunks(directory: &Path, resume_sequence: u64) -> Result<()> {
    let mut committed = BTreeSet::new();
    for entry in directory.read_dir()? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "manifest.json" {
            continue;
        }
        if file_type.is_file() && name.starts_with("chunk-pending-") {
            fs::remove_file(entry.path())?;
            continue;
        }
        let sequence = name
            .strip_prefix("chunk-")
            .and_then(|value| value.strip_suffix(".parquet"))
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|_| file_type.is_file())
            .ok_or_else(|| {
                ElmError::Conflict(format!(
                    "unrecognized resource '{}' in durable staging",
                    entry.path().display()
                ))
            })?;
        if sequence <= resume_sequence {
            committed.insert(sequence);
        } else {
            fs::remove_file(entry.path())?;
        }
    }

    let mut expected = 0_u64;
    for sequence in committed {
        if sequence != expected {
            return Err(ElmError::Conflict(format!(
                "durable staging chunk {expected} is missing"
            )));
        }
        expected = expected
            .checked_add(1)
            .ok_or_else(|| ElmError::State("checkpoint sequence overflow".into()))?;
    }
    let required = resume_sequence
        .checked_add(1)
        .ok_or_else(|| ElmError::State("checkpoint sequence cannot be resumed".into()))?;
    if expected != required {
        return Err(ElmError::Conflict(format!(
            "durable staging chunk {} is missing",
            expected
        )));
    }
    Ok(())
}

fn append_schema_compatible(existing: &Schema, incoming: &Schema) -> bool {
    existing.fields().len() == incoming.fields().len()
        && existing
            .fields()
            .iter()
            .zip(incoming.fields())
            .all(|(existing, incoming)| {
                existing.name() == incoming.name()
                    && existing.data_type() == incoming.data_type()
                    && (!incoming.is_nullable() || existing.is_nullable())
            })
}

pub fn fingerprint_file(path: &Path) -> Result<String> {
    let metadata = fs::metadata(path)?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0_u128, |duration| duration.as_nanos());
    let mut file = File::open(path)?;
    let mut prefix = vec![0_u8; 64 * 1024];
    let prefix_len = file.read(&mut prefix)?;
    prefix.truncate(prefix_len);
    let mut hasher = blake3::Hasher::new();
    hasher.update(&metadata.len().to_le_bytes());
    hasher.update(&modified.to_le_bytes());
    hasher.update(&prefix);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Removes a job staging directory only after its manifest and every contained resource match the
/// expected file sink. Returns `false` when the directory is already absent.
pub fn cleanup_recoverable_staging(
    directory: &Path,
    destination: &Path,
    format: FileFormat,
    write_mode: WriteMode,
) -> Result<bool> {
    if !directory.exists() {
        return Ok(false);
    }
    let manifest = read_staging_manifest(&directory.join("manifest.json"))?;
    if manifest.version != 1
        || manifest.destination != destination
        || manifest.format != format
        || manifest.write_mode != write_mode
    {
        return Err(ElmError::Conflict(
            "staging manifest does not match the requested cleanup target".into(),
        ));
    }
    for entry in directory.read_dir()? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let recognized = name == "manifest.json"
            || name.starts_with("chunk-pending-")
            || (name.starts_with("chunk-") && name.ends_with(".parquet"));
        if !recognized || !entry.file_type()?.is_file() {
            return Err(ElmError::Conflict(format!(
                "refusing to remove unrecognized staging resource '{}'",
                entry.path().display()
            )));
        }
    }
    fs::remove_dir_all(directory)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::{fs, sync::Arc};

    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::tempdir;

    use super::*;

    fn test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("message", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec![Some("İstanbul 🌍"), None])),
            ],
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    #[tokio::test]
    async fn csv_round_trip_is_streamed_through_record_batches() {
        let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
        let path = directory.path().join("values.csv");
        let batch = test_batch();
        let mut sink = FileSink::new(&path, FileFormat::Csv);
        sink.preflight(batch.schema(), WriteMode::Replace, ConsistencyMode::Atomic)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
        sink.write_batch(&batch)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        sink.publish()
            .await
            .unwrap_or_else(|error| panic!("{error}"));

        let mut source =
            FileSource::open(&path, FileFormat::Csv).unwrap_or_else(|error| panic!("{error}"));
        let read = source
            .next_batch(8 * 1024 * 1024)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .unwrap_or_else(|| panic!("expected a batch"));
        assert_eq!(read.num_rows(), 2);
    }

    #[tokio::test]
    async fn fail_mode_does_not_touch_existing_file() {
        let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
        let path = directory.path().join("existing.ndjson");
        fs::write(&path, "untouched").unwrap_or_else(|error| panic!("{error}"));
        let mut sink = FileSink::new(&path, FileFormat::Ndjson);
        assert!(
            sink.preflight(
                test_batch().schema(),
                WriteMode::Fail,
                ConsistencyMode::Atomic
            )
            .await
            .is_err()
        );
        assert_eq!(fs::read_to_string(path).unwrap_or_default(), "untouched");
    }

    #[tokio::test]
    async fn recoverable_sink_reuses_only_checkpointed_chunks() {
        let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
        let path = directory.path().join("resumed.csv");
        let staging = directory.path().join("job-staging");
        let batch = test_batch();
        let checkpoint = BatchCheckpoint {
            sequence: 0,
            source_position: serde_json::json!({"logical_row_offset": 2}),
            rows_committed: 2,
            bytes_committed: u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX),
            source_fingerprint: None,
        };

        let mut first = RecoverableFileSink::new(&path, FileFormat::Csv, &staging, None)
            .unwrap_or_else(|error| panic!("{error}"));
        first
            .preflight(batch.schema(), WriteMode::Replace, ConsistencyMode::Atomic)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        first
            .begin()
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        first
            .write_batch(&batch)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        first
            .commit_checkpoint(&checkpoint)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        first
            .abort()
            .await
            .unwrap_or_else(|error| panic!("{error}"));

        let mut resumed =
            RecoverableFileSink::new(&path, FileFormat::Csv, &staging, Some(&checkpoint))
                .unwrap_or_else(|error| panic!("{error}"));
        resumed
            .preflight(batch.schema(), WriteMode::Replace, ConsistencyMode::Atomic)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        resumed
            .begin()
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        resumed
            .write_batch(&batch)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        let next_checkpoint = BatchCheckpoint {
            sequence: 1,
            source_position: serde_json::json!({"logical_row_offset": 4}),
            rows_committed: 4,
            bytes_committed: checkpoint.bytes_committed.saturating_mul(2),
            source_fingerprint: None,
        };
        resumed
            .commit_checkpoint(&next_checkpoint)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        resumed
            .publish()
            .await
            .unwrap_or_else(|error| panic!("{error}"));

        let mut source =
            FileSource::open(&path, FileFormat::Csv).unwrap_or_else(|error| panic!("{error}"));
        let output = source
            .next_batch(8 * 1024 * 1024)
            .await
            .unwrap_or_else(|error| panic!("{error}"))
            .unwrap_or_else(|| panic!("expected resumed output"));
        assert_eq!(output.num_rows(), 4);
        assert!(!staging.exists());
    }

    #[tokio::test]
    async fn empty_csv_replace_still_writes_the_schema_header() {
        let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
        let path = directory.path().join("empty.csv");
        let mut sink = FileSink::new(&path, FileFormat::Csv);
        sink.preflight(
            test_batch().schema(),
            WriteMode::Replace,
            ConsistencyMode::Atomic,
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"));
        sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
        sink.publish()
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(fs::read_to_string(path).unwrap_or_default(), "id,message\n");
    }

    #[tokio::test]
    async fn cleanup_refuses_unrecognized_staging_resources() {
        let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
        let path = directory.path().join("target.csv");
        let staging = directory.path().join("job-staging");
        let batch = test_batch();
        let mut sink = RecoverableFileSink::new(&path, FileFormat::Csv, &staging, None)
            .unwrap_or_else(|error| panic!("{error}"));
        sink.preflight(batch.schema(), WriteMode::Replace, ConsistencyMode::Atomic)
            .await
            .unwrap_or_else(|error| panic!("{error}"));
        sink.begin().await.unwrap_or_else(|error| panic!("{error}"));
        fs::write(staging.join("unrecognized.txt"), "do not remove")
            .unwrap_or_else(|error| panic!("{error}"));

        assert!(
            cleanup_recoverable_staging(&staging, &path, FileFormat::Csv, WriteMode::Replace)
                .is_err()
        );
        assert!(staging.exists());
        fs::remove_file(staging.join("unrecognized.txt")).unwrap_or_else(|error| panic!("{error}"));
        assert!(
            cleanup_recoverable_staging(&staging, &path, FileFormat::Csv, WriteMode::Replace)
                .unwrap_or(false)
        );
        assert!(!staging.exists());
    }
}
