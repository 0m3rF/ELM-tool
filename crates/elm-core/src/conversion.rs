use std::{collections::HashSet, str::FromStr, sync::Arc};

use arrow_array::RecordBatch;
use arrow_cast::{CastOptions, can_cast_types, cast_with_options};
use arrow_schema::{DataType, Field, Schema, TimeUnit};

use crate::{ConversionRule, ElmError, Result};

#[derive(Debug, Clone)]
struct PlannedConversion {
    column: String,
    target_type: DataType,
}

/// A schema-validated set of per-column Arrow casts.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    input_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    columns: Vec<Option<PlannedConversion>>,
}

impl ConversionPlan {
    pub fn new(input_schema: Arc<Schema>, rules: &[ConversionRule]) -> Result<Self> {
        let mut columns = vec![None; input_schema.fields().len()];
        let mut output_fields = input_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        let mut seen = HashSet::new();

        for rule in rules {
            if !seen.insert(rule.column.as_str()) {
                return Err(ElmError::Validation(format!(
                    "column '{}' has more than one conversion rule",
                    rule.column
                )));
            }
            let index = input_schema.index_of(rule.column.as_str()).map_err(|_| {
                ElmError::Validation(format!(
                    "conversion column '{}' does not exist",
                    rule.column
                ))
            })?;
            let source = input_schema.field(index);
            let target_type = parse_target_type(&rule.target_type)?;
            if !can_cast_types(source.data_type(), &target_type) {
                return Err(ElmError::TypeMapping(format!(
                    "column '{}' cannot be converted from {} to {}",
                    rule.column,
                    source.data_type(),
                    target_type
                )));
            }
            if !rule.allow_lossy && !is_lossless(source.data_type(), &target_type) {
                return Err(ElmError::TypeMapping(format!(
                    "column '{}' conversion from {} to {} may be lossy; set allow_lossy on this rule to proceed",
                    rule.column,
                    source.data_type(),
                    target_type
                )));
            }
            output_fields[index] =
                Field::new(source.name(), target_type.clone(), source.is_nullable())
                    .with_metadata(source.metadata().clone());
            columns[index] = Some(PlannedConversion {
                column: rule.column.to_string(),
                target_type,
            });
        }

        let output_metadata = input_schema.metadata().clone();
        Ok(Self {
            input_schema,
            output_schema: Arc::new(Schema::new_with_metadata(output_fields, output_metadata)),
            columns,
        })
    }

    #[must_use]
    pub fn output_schema(&self) -> Arc<Schema> {
        self.output_schema.clone()
    }

    #[must_use]
    pub fn input_schema(&self) -> Arc<Schema> {
        self.input_schema.clone()
    }

    pub fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if batch.schema().as_ref() != self.input_schema.as_ref() {
            return Err(ElmError::TypeMapping(
                "source batch schema changed after conversion preflight".into(),
            ));
        }
        if self.columns.iter().all(Option::is_none) {
            return Ok(batch.clone());
        }
        let options = CastOptions {
            // Arrow's `safe` mode replaces failed values with null. ELM instead fails the batch.
            safe: false,
            ..CastOptions::default()
        };
        let arrays = batch
            .columns()
            .iter()
            .zip(&self.columns)
            .map(|(array, conversion)| match conversion {
                Some(conversion) => {
                    let nulls_before = array.null_count();
                    let converted =
                        cast_with_options(array.as_ref(), &conversion.target_type, &options)
                            .map_err(|error| {
                                ElmError::TypeMapping(format!(
                                    "column '{}' conversion failed: {error}",
                                    conversion.column
                                ))
                            })?;
                    if converted.null_count() > nulls_before {
                        return Err(ElmError::TypeMapping(format!(
                            "column '{}' conversion introduced null values",
                            conversion.column
                        )));
                    }
                    Ok(converted)
                }
                None => Ok(array.clone()),
            })
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new(self.output_schema.clone(), arrays)
            .map_err(|error| ElmError::TypeMapping(format!("converted batch is invalid: {error}")))
    }
}

fn parse_target_type(value: &str) -> Result<DataType> {
    let value = value.trim();
    let alias = match value.to_ascii_lowercase().as_str() {
        "bool" | "boolean" => Some(DataType::Boolean),
        "i8" | "int8" => Some(DataType::Int8),
        "i16" | "int16" => Some(DataType::Int16),
        "i32" | "int32" => Some(DataType::Int32),
        "i64" | "int64" => Some(DataType::Int64),
        "u8" | "uint8" => Some(DataType::UInt8),
        "u16" | "uint16" => Some(DataType::UInt16),
        "u32" | "uint32" => Some(DataType::UInt32),
        "u64" | "uint64" => Some(DataType::UInt64),
        "f32" | "float32" => Some(DataType::Float32),
        "f64" | "float64" | "double" => Some(DataType::Float64),
        "string" | "utf8" => Some(DataType::Utf8),
        "large_string" | "large_utf8" => Some(DataType::LargeUtf8),
        "binary" => Some(DataType::Binary),
        "large_binary" => Some(DataType::LargeBinary),
        "date32" => Some(DataType::Date32),
        "date64" => Some(DataType::Date64),
        _ => None,
    };
    alias.map_or_else(
        || {
            DataType::from_str(value).map_err(|error| {
                ElmError::Validation(format!("invalid Arrow target type '{value}': {error}"))
            })
        },
        Ok,
    )
}

fn is_lossless(source: &DataType, target: &DataType) -> bool {
    if source == target || matches!(source, DataType::Null) {
        return true;
    }
    if let (Some((source_signed, source_bits)), Some((target_signed, target_bits))) =
        (integer_width(source), integer_width(target))
    {
        return source_signed == target_signed && target_bits >= source_bits
            || !source_signed && target_signed && target_bits > source_bits;
    }
    match (source, target) {
        (DataType::Float32, DataType::Float64)
        | (DataType::Utf8, DataType::LargeUtf8)
        | (DataType::Binary, DataType::LargeBinary)
        | (DataType::Date32, DataType::Date64) => true,
        (
            DataType::Timestamp(source_unit, source_zone),
            DataType::Timestamp(target_unit, target_zone),
        ) => {
            source_zone == target_zone && time_precision(target_unit) >= time_precision(source_unit)
        }
        (DataType::Duration(source_unit), DataType::Duration(target_unit)) => {
            time_precision(target_unit) >= time_precision(source_unit)
        }
        (
            DataType::Decimal128(source_precision, source_scale),
            DataType::Decimal128(target_precision, target_scale),
        ) => {
            target_scale >= source_scale
                && i16::from(*target_precision) - i16::from(*target_scale)
                    >= i16::from(*source_precision) - i16::from(*source_scale)
        }
        _ => false,
    }
}

fn integer_width(data_type: &DataType) -> Option<(bool, u8)> {
    match data_type {
        DataType::Int8 => Some((true, 8)),
        DataType::Int16 => Some((true, 16)),
        DataType::Int32 => Some((true, 32)),
        DataType::Int64 => Some((true, 64)),
        DataType::UInt8 => Some((false, 8)),
        DataType::UInt16 => Some((false, 16)),
        DataType::UInt32 => Some((false, 32)),
        DataType::UInt64 => Some((false, 64)),
        _ => None,
    }
}

fn time_precision(unit: &TimeUnit) -> u8 {
    match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Int32Array, Int64Array};

    use super::*;
    use crate::Identifier;

    fn schema(data_type: DataType) -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("value", data_type, true)]))
    }

    fn rule(target_type: &str, allow_lossy: bool) -> ConversionRule {
        ConversionRule {
            column: Identifier::new("value").unwrap_or_else(|error| panic!("{error}")),
            target_type: target_type.into(),
            allow_lossy,
        }
    }

    #[test]
    fn widens_integer_values_without_lossy_opt_in() {
        let input_schema = schema(DataType::Int32);
        let batch = RecordBatch::try_new(
            input_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![Some(7), None, Some(-9)]))],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        let plan = ConversionPlan::new(input_schema, &[rule("int64", false)])
            .unwrap_or_else(|error| panic!("{error}"));
        let output = plan.apply(&batch).unwrap_or_else(|error| panic!("{error}"));
        let values = output
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("expected Int64 output"));
        assert_eq!(values.value(0), 7);
        assert!(values.is_null(1));
        assert_eq!(values.value(2), -9);
    }

    #[test]
    fn narrowing_requires_column_level_lossy_opt_in() {
        assert!(ConversionPlan::new(schema(DataType::Int64), &[rule("int32", false)]).is_err());
        assert!(ConversionPlan::new(schema(DataType::Int64), &[rule("int32", true)]).is_ok());
    }

    #[test]
    fn duplicate_and_missing_columns_fail_preflight() {
        let input_schema = schema(DataType::Int32);
        assert!(
            ConversionPlan::new(
                input_schema.clone(),
                &[rule("int64", false), rule("int64", false)]
            )
            .is_err()
        );
        let missing = ConversionRule {
            column: Identifier::new("missing").unwrap_or_else(|error| panic!("{error}")),
            target_type: "int64".into(),
            allow_lossy: false,
        };
        assert!(ConversionPlan::new(input_schema, &[missing]).is_err());
    }
}
