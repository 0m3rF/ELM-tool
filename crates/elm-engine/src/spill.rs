use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use tempfile::TempDir;
use uuid::Uuid;

use elm_core::{ElmError, Result};

/// Job-scoped storage for values too large to share the in-memory batch budget.
/// Files are removed with the store and are never used as an implicit fallback: connectors must
/// advertise spill support during preflight before writing here.
#[derive(Debug)]
pub struct SpillStore {
    directory: TempDir,
}

impl SpillStore {
    pub fn new(parent: Option<&Path>) -> Result<Self> {
        let directory = match parent {
            Some(parent) => tempfile::Builder::new()
                .prefix("elm-spill-")
                .tempdir_in(parent),
            None => tempfile::Builder::new().prefix("elm-spill-").tempdir(),
        }?;
        Ok(Self { directory })
    }

    pub fn write_from(&self, input: &mut dyn Read) -> Result<SpillFile> {
        let path = self.directory.path().join(Uuid::new_v4().to_string());
        let mut output = File::create(&path)?;
        let length = std::io::copy(input, &mut output)?;
        output.flush()?;
        Ok(SpillFile { path, length })
    }
}

#[derive(Debug)]
pub struct SpillFile {
    path: PathBuf,
    length: u64,
}

impl SpillFile {
    #[must_use]
    pub fn len(&self) -> u64 {
        self.length
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn read_chunk(&self, offset: u64, maximum: usize) -> Result<Vec<u8>> {
        if offset > self.length {
            return Err(ElmError::Validation(
                "spill offset exceeds value length".into(),
            ));
        }
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let remaining = usize::try_from(self.length - offset).unwrap_or(usize::MAX);
        let mut buffer = vec![0; maximum.min(remaining)];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::SpillStore;

    #[test]
    fn spill_is_read_in_bounded_chunks() {
        let store = SpillStore::new(None).unwrap_or_else(|error| panic!("{error}"));
        let file = store
            .write_from(&mut Cursor::new(b"0123456789"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(file.read_chunk(3, 4).unwrap_or_default(), b"3456");
    }
}
