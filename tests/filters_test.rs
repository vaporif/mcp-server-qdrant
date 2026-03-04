use std::collections::HashMap;

use mcp_server_qdrant::filters::{FieldType, FilterCondition, FilterableField, make_filter};

#[test]
fn deserialize_filterable_fields_from_json() {
    let json = r#"[
        {
            "name": "category",
            "description": "Item category",
            "field_type": "keyword",
            "condition": "==",
            "required": true
        },
        {
            "name": "price",
            "description": "Item price",
            "field_type": "float",
            "condition": ">=",
            "required": false
        }
    ]"#;

    let fields: Vec<FilterableField> = serde_json::from_str(json).unwrap();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].field_type, FieldType::Keyword);
    assert_eq!(fields[0].condition, Some(FilterCondition::Eq));
    assert!(fields[0].required);
    assert_eq!(fields[1].field_type, FieldType::Float);
    assert_eq!(fields[1].condition, Some(FilterCondition::Gte));
}

#[test]
fn make_filter_keyword_eq() {
    let fields = vec![FilterableField {
        name: "city".into(),
        description: "City name".into(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];
    let mut values = HashMap::new();
    values.insert("city".into(), serde_json::json!("London"));
    let filter = make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 1);
    assert!(filter.must_not.is_empty());
}

#[test]
fn make_filter_keyword_ne() {
    let fields = vec![FilterableField {
        name: "city".into(),
        description: "City name".into(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Ne),
        required: false,
    }];
    let mut values = HashMap::new();
    values.insert("city".into(), serde_json::json!("London"));
    let filter = make_filter(&fields, &values).unwrap();
    assert!(filter.must.is_empty());
    assert_eq!(filter.must_not.len(), 1);
}

#[test]
fn make_filter_integer_range() {
    let fields = vec![FilterableField {
        name: "price".into(),
        description: "Price".into(),
        field_type: FieldType::Integer,
        condition: Some(FilterCondition::Gte),
        required: false,
    }];
    let mut values = HashMap::new();
    values.insert("price".into(), serde_json::json!(100));
    let filter = make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 1);
}

#[test]
fn make_filter_boolean_eq() {
    let fields = vec![FilterableField {
        name: "active".into(),
        description: "Is active".into(),
        field_type: FieldType::Boolean,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];
    let mut values = HashMap::new();
    values.insert("active".into(), serde_json::json!(true));
    let filter = make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 1);
}

#[test]
fn make_filter_skips_null_optional_fields() {
    let fields = vec![FilterableField {
        name: "city".into(),
        description: "City name".into(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];
    let values = HashMap::new();
    let filter = make_filter(&fields, &values).unwrap();
    assert!(filter.must.is_empty());
    assert!(filter.must_not.is_empty());
}

#[test]
fn make_filter_errors_on_missing_required_field() {
    let fields = vec![FilterableField {
        name: "city".into(),
        description: "City name".into(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Eq),
        required: true,
    }];
    let values = HashMap::new();
    assert!(make_filter(&fields, &values).is_err());
}

#[test]
fn make_filter_multiple_fields() {
    let fields = vec![
        FilterableField {
            name: "city".into(),
            description: "City name".into(),
            field_type: FieldType::Keyword,
            condition: Some(FilterCondition::Eq),
            required: false,
        },
        FilterableField {
            name: "price".into(),
            description: "Min price".into(),
            field_type: FieldType::Float,
            condition: Some(FilterCondition::Gte),
            required: false,
        },
    ];
    let mut values = HashMap::new();
    values.insert("city".into(), serde_json::json!("London"));
    values.insert("price".into(), serde_json::json!(9.99));
    let filter = make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 2);
}

#[test]
fn make_filter_float_range_lt() {
    let fields = vec![FilterableField {
        name: "score".into(),
        description: "Max score".into(),
        field_type: FieldType::Float,
        condition: Some(FilterCondition::Lt),
        required: false,
    }];
    let mut values = HashMap::new();
    values.insert("score".into(), serde_json::json!(5.5));
    let filter = make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 1);
}
