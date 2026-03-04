use mcp_server_qdrant::filters::{FieldType, FilterCondition, FilterableField};

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
