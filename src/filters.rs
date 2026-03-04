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
