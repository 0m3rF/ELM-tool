use std::sync::Arc;

use arrow_array::{Array, ArrayRef, LargeStringArray, RecordBatch, StringArray, new_null_array};
use arrow_schema::DataType;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{ElmError, MaskAlgorithm, MaskRule, Result};

const RANDOM_ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Masks a single preview value with the same deterministic algorithm used for Arrow batches.
#[must_use]
pub fn mask_text(value: &str, algorithm: &MaskAlgorithm, job_seed: &[u8; 32]) -> String {
    mask_value(value, algorithm, job_seed, 0, 0, 0)
}

/// Applies masking without changing the input batch. Random values are derived from the job seed,
/// batch sequence, column index and row index, so retrying a checkpoint yields identical output.
pub fn apply_masks(
    batch: &RecordBatch,
    rules: &[MaskRule],
    job_seed: &[u8; 32],
    batch_sequence: u64,
) -> Result<RecordBatch> {
    if rules.is_empty() {
        return Ok(batch.clone());
    }

    let mut columns = batch.columns().to_vec();
    for rule in rules {
        let index = batch.schema().index_of(rule.column.as_str()).map_err(|_| {
            ElmError::Validation(format!("mask column '{}' does not exist", rule.column))
        })?;
        columns[index] = mask_array(
            columns[index].as_ref(),
            &rule.algorithm,
            job_seed,
            batch_sequence,
            index,
        )?;
    }

    RecordBatch::try_new(batch.schema(), columns)
        .map_err(|error| ElmError::Internal(format!("masked batch is invalid: {error}")))
}

fn mask_array(
    array: &dyn Array,
    algorithm: &MaskAlgorithm,
    job_seed: &[u8; 32],
    batch_sequence: u64,
    column_index: usize,
) -> Result<ArrayRef> {
    if matches!(algorithm, MaskAlgorithm::Nullify) {
        return Ok(new_null_array(array.data_type(), array.len()));
    }

    match array.data_type() {
        DataType::Utf8 => {
            let strings = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| ElmError::Internal("Utf8 array downcast failed".into()))?;
            let values = (0..strings.len()).map(|row| {
                strings.is_valid(row).then(|| {
                    mask_value(
                        strings.value(row),
                        algorithm,
                        job_seed,
                        batch_sequence,
                        column_index,
                        row,
                    )
                })
            });
            Ok(Arc::new(StringArray::from_iter(values)))
        }
        DataType::LargeUtf8 => {
            let strings = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| ElmError::Internal("LargeUtf8 array downcast failed".into()))?;
            let values = (0..strings.len()).map(|row| {
                strings.is_valid(row).then(|| {
                    mask_value(
                        strings.value(row),
                        algorithm,
                        job_seed,
                        batch_sequence,
                        column_index,
                        row,
                    )
                })
            });
            Ok(Arc::new(LargeStringArray::from_iter(values)))
        }
        data_type => Err(ElmError::Validation(format!(
            "mask algorithm {algorithm:?} requires a string column, found {data_type}"
        ))),
    }
}

fn mask_value(
    value: &str,
    algorithm: &MaskAlgorithm,
    job_seed: &[u8; 32],
    batch_sequence: u64,
    column_index: usize,
    row_index: usize,
) -> String {
    match algorithm {
        MaskAlgorithm::Star => "*****".into(),
        MaskAlgorithm::StarLength { unmasked_prefix } => {
            let prefix: String = value.chars().take(*unmasked_prefix).collect();
            let hidden = value.chars().count().saturating_sub(*unmasked_prefix);
            prefix + &"*".repeat(hidden)
        }
        MaskAlgorithm::Random => {
            let mut hasher = blake3::Hasher::new_keyed(job_seed);
            hasher.update(&batch_sequence.to_le_bytes());
            hasher.update(&(column_index as u64).to_le_bytes());
            hasher.update(&(row_index as u64).to_le_bytes());
            let seed = *hasher.finalize().as_bytes();
            let mut rng = ChaCha8Rng::from_seed(seed);
            value
                .chars()
                .map(|_| RANDOM_ALPHABET[rng.random_range(0..RANDOM_ALPHABET.len())] as char)
                .collect()
        }
        MaskAlgorithm::Nullify => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::{Identifier, MaskRuleId};

    fn rule(algorithm: MaskAlgorithm) -> MaskRule {
        MaskRule {
            id: MaskRuleId::new(),
            name: "email mask".into(),
            column: Identifier::new("secret").unwrap_or_else(|error| panic!("{error}")),
            algorithm,
            environment_id: None,
        }
    }

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "secret",
            DataType::Utf8,
            true,
        )]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec![
                Some("héllo"),
                None,
                Some("world"),
            ]))],
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn random_mask_is_deterministic_by_batch_position() {
        let input = batch();
        let seed = [42; 32];
        let first = apply_masks(&input, &[rule(MaskAlgorithm::Random)], &seed, 3)
            .unwrap_or_else(|error| panic!("{error}"));
        let retried = apply_masks(&input, &[rule(MaskAlgorithm::Random)], &seed, 3)
            .unwrap_or_else(|error| panic!("{error}"));
        let next = apply_masks(&input, &[rule(MaskAlgorithm::Random)], &seed, 4)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(first.column(0).to_data(), retried.column(0).to_data());
        assert_ne!(first.column(0).to_data(), next.column(0).to_data());
    }

    #[test]
    fn nullify_preserves_type_and_nulls_every_row() {
        let output = apply_masks(&batch(), &[rule(MaskAlgorithm::Nullify)], &[0; 32], 0)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(output.column(0).data_type(), &DataType::Utf8);
        assert_eq!(output.column(0).null_count(), 3);
    }
}
