use std::collections::HashMap;
use std::hash::BuildHasher;

use qdrant_client::qdrant::{Condition, Filter, Range};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    Keyword,
    Integer,
    Float,
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FilterCondition {
    #[serde(rename = "==")]
    Eq,
    #[serde(rename = "!=")]
    Ne,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    Gte,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    Lte,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "except")]
    Except,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterableField {
    pub name: String,
    pub description: String,
    pub field_type: FieldType,
    pub condition: Option<FilterCondition>,
    #[serde(default)]
    pub required: bool,
}

const METADATA_PATH: &str = "metadata";

#[allow(clippy::result_large_err, clippy::cast_precision_loss)]
pub fn make_filter<S: BuildHasher>(
    fields: &[FilterableField],
    values: &HashMap<String, serde_json::Value, S>,
) -> crate::errors::Result<Filter> {
    let mut must = Vec::new();
    let mut must_not = Vec::new();

    for field in fields {
        let value = values.get(&field.name);

        match value {
            None | Some(serde_json::Value::Null) => {
                if field.required {
                    return Err(crate::errors::Error::Config(format!(
                        "field '{}' is required",
                        field.name
                    )));
                }
            }
            Some(v) => {
                let key = format!("{METADATA_PATH}.{}", field.name);
                let condition = field.condition.as_ref().unwrap_or(&FilterCondition::Eq);

                match (&field.field_type, condition) {
                    (FieldType::Keyword, FilterCondition::Eq) => {
                        let s = v.as_str().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected string value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::matches(key, s.to_string()));
                    }
                    (FieldType::Keyword, FilterCondition::Ne) => {
                        let s = v.as_str().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected string value",
                                field.name
                            ))
                        })?;
                        must_not.push(Condition::matches(key, s.to_string()));
                    }
                    (FieldType::Integer, FilterCondition::Eq) => {
                        let n = v.as_i64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected integer value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::matches(key, n));
                    }
                    (FieldType::Integer, FilterCondition::Ne) => {
                        let n = v.as_i64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected integer value",
                                field.name
                            ))
                        })?;
                        must_not.push(Condition::matches(key, n));
                    }
                    (FieldType::Integer, FilterCondition::Gt) => {
                        let n = v.as_i64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected integer value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                gt: Some(n as f64),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Integer, FilterCondition::Gte) => {
                        let n = v.as_i64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected integer value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                gte: Some(n as f64),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Integer, FilterCondition::Lt) => {
                        let n = v.as_i64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected integer value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                lt: Some(n as f64),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Integer, FilterCondition::Lte) => {
                        let n = v.as_i64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected integer value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                lte: Some(n as f64),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Float, FilterCondition::Gt) => {
                        let n = v.as_f64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected float value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                gt: Some(n),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Float, FilterCondition::Gte) => {
                        let n = v.as_f64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected float value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                gte: Some(n),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Float, FilterCondition::Lt) => {
                        let n = v.as_f64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected float value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                lt: Some(n),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Float, FilterCondition::Lte) => {
                        let n = v.as_f64().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected float value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::range(
                            key,
                            Range {
                                lte: Some(n),
                                ..Default::default()
                            },
                        ));
                    }
                    (FieldType::Boolean, FilterCondition::Eq) => {
                        let b = v.as_bool().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected boolean value",
                                field.name
                            ))
                        })?;
                        must.push(Condition::matches(key, b));
                    }
                    (FieldType::Boolean, FilterCondition::Ne) => {
                        let b = v.as_bool().ok_or_else(|| {
                            crate::errors::Error::Config(format!(
                                "field '{}': expected boolean value",
                                field.name
                            ))
                        })?;
                        must_not.push(Condition::matches(key, b));
                    }
                    (ft, cond) => {
                        return Err(crate::errors::Error::Config(format!(
                            "unsupported condition {cond:?} for field type {ft:?} on field '{}'",
                            field.name
                        )));
                    }
                }
            }
        }
    }

    Ok(Filter {
        must,
        must_not,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_type_round_trip() {
        for (json, expected) in [
            ("\"keyword\"", FieldType::Keyword),
            ("\"integer\"", FieldType::Integer),
            ("\"float\"", FieldType::Float),
            ("\"boolean\"", FieldType::Boolean),
        ] {
            let deserialized: FieldType = serde_json::from_str(json).unwrap();
            assert_eq!(deserialized, expected);
            let serialized = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(serialized, json);
        }
    }

    #[test]
    fn filter_condition_round_trip() {
        for (json, expected) in [
            ("\"==\"", FilterCondition::Eq),
            ("\"!=\"", FilterCondition::Ne),
            ("\">\"", FilterCondition::Gt),
            ("\">=\"", FilterCondition::Gte),
            ("\"<\"", FilterCondition::Lt),
            ("\"<=\"", FilterCondition::Lte),
            ("\"any\"", FilterCondition::Any),
            ("\"except\"", FilterCondition::Except),
        ] {
            let deserialized: FilterCondition = serde_json::from_str(json).unwrap();
            assert_eq!(deserialized, expected);
            let serialized = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(serialized, json);
        }
    }

    #[test]
    fn filterable_field_round_trip() {
        let json = r#"{
            "name": "category",
            "description": "Product category",
            "field_type": "keyword",
            "condition": "==",
            "required": true
        }"#;

        let field: FilterableField = serde_json::from_str(json).unwrap();
        assert_eq!(field.name, "category");
        assert_eq!(field.description, "Product category");
        assert_eq!(field.field_type, FieldType::Keyword);
        assert_eq!(field.condition, Some(FilterCondition::Eq));
        assert!(field.required);

        let reserialized = serde_json::to_string(&field).unwrap();
        let round_tripped: FilterableField = serde_json::from_str(&reserialized).unwrap();
        assert_eq!(round_tripped.name, field.name);
        assert_eq!(round_tripped.field_type, field.field_type);
        assert_eq!(round_tripped.condition, field.condition);
        assert_eq!(round_tripped.required, field.required);
    }

    #[test]
    fn filterable_field_optional_condition() {
        let json = r#"{
            "name": "status",
            "description": "Item status",
            "field_type": "keyword"
        }"#;

        let field: FilterableField = serde_json::from_str(json).unwrap();
        assert_eq!(field.condition, None);
        assert!(!field.required);
    }

    #[test]
    fn filterable_fields_array_from_json() {
        let json = r#"[
            {
                "name": "category",
                "description": "Product category",
                "field_type": "keyword",
                "condition": "==",
                "required": true
            },
            {
                "name": "price",
                "description": "Product price",
                "field_type": "float",
                "condition": ">="
            }
        ]"#;

        let fields: Vec<FilterableField> = serde_json::from_str(json).unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].field_type, FieldType::Keyword);
        assert_eq!(fields[1].field_type, FieldType::Float);
        assert_eq!(fields[1].condition, Some(FilterCondition::Gte));
        assert!(!fields[1].required);
    }
}
