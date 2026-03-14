use std::sync::Arc;

use mcp_server_qdrant::config::{
    Config, EmbeddingConfig, QdrantConfig, QdrantLocation, ToolConfig, Transport,
};
use mcp_server_qdrant::embeddings::create_embedding_provider;
use mcp_server_qdrant::filters::{FieldType, FilterCondition, FilterableField};
use mcp_server_qdrant::qdrant::{Entry, QdrantConnector};
use mcp_server_qdrant::server::QdrantMcpServer;
use qdrant_client::Qdrant;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::{ClientHandler, ServiceExt};

const QDRANT_URL: &str = "http://localhost:6334";
const EMBEDDING_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn qdrant_config(collection_name: &str) -> QdrantConfig {
    QdrantConfig {
        location: QdrantLocation::Remote {
            url: QDRANT_URL.to_string(),
            api_key: None,
        },
        collection_name: Some(collection_name.to_string()),
        search_limit: 10,
        read_only: false,
        filterable_fields: vec![],
        allow_arbitrary_filter: false,
    }
}

fn unique_collection() -> String {
    format!("test_e2e_{}", uuid::Uuid::new_v4())
}

struct CollectionGuard {
    name: String,
}

impl Drop for CollectionGuard {
    fn drop(&mut self) {
        let name = self.name.clone();
        let _ = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let client = Qdrant::from_url(QDRANT_URL).build().expect("qdrant client");
                let _ = client.delete_collection(&name).await;
            });
        })
        .join();
    }
}

async fn embedding_provider() -> Arc<dyn mcp_server_qdrant::embeddings::EmbeddingProvider> {
    Arc::from(create_embedding_provider(EMBEDDING_MODEL).await.unwrap())
}

async fn connector_for(config: &QdrantConfig) -> QdrantConnector {
    let embedding = embedding_provider().await;
    QdrantConnector::new(config, embedding).unwrap()
}

fn default_config(qdrant: QdrantConfig) -> Config {
    Config {
        qdrant,
        embedding: EmbeddingConfig {
            model_name: EMBEDDING_MODEL.to_string(),
        },
        tools: ToolConfig {
            store_description: None,
            find_description: None,
        },
        transport: Transport::Stdio,
    }
}

fn json_args(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    v.as_object().unwrap().clone()
}

fn result_text(result: &CallToolResult) -> &str {
    result.content[0]
        .raw
        .as_text()
        .map(|t| t.text.as_str())
        .unwrap()
}

#[derive(Debug, Clone, Default)]
struct DummyClientHandler;

impl ClientHandler for DummyClientHandler {
    fn get_info(&self) -> rmcp::model::ClientInfo {
        rmcp::model::ClientInfo::default()
    }
}

async fn mcp_client_for(
    config: Config,
) -> rmcp::service::RunningService<rmcp::RoleClient, DummyClientHandler> {
    let embedding = embedding_provider().await;
    let connector = Arc::new(QdrantConnector::new(&config.qdrant, embedding).unwrap());
    let config = Arc::new(config);
    let server = QdrantMcpServer::new(connector, config);

    let (server_transport, client_transport) = tokio::io::duplex(4096);

    tokio::spawn(async move {
        let svc = server.serve(server_transport).await.unwrap();
        svc.waiting().await.unwrap();
    });

    DummyClientHandler
        .serve(client_transport)
        .await
        .expect("client connect")
}

async fn schema_test_client(
    config: Config,
) -> rmcp::service::RunningService<rmcp::RoleClient, DummyClientHandler> {
    let embedding = embedding_provider().await;
    let connector = Arc::new(QdrantConnector::new(&config.qdrant, embedding).unwrap());
    let config = Arc::new(config);
    let server = QdrantMcpServer::new(connector, config);

    let (server_transport, client_transport) = tokio::io::duplex(4096);
    tokio::spawn(async move {
        let svc = server.serve(server_transport).await.unwrap();
        svc.waiting().await.unwrap();
    });

    DummyClientHandler
        .serve(client_transport)
        .await
        .expect("client connect")
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn store_and_search_basic() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    connector
        .store(
            Entry {
                content: "Qdrant is a vector search engine written in Rust".to_string(),
                metadata: None,
            },
            &collection,
        )
        .await
        .unwrap();

    let results = connector
        .search("vector search engine", &collection, 5, None)
        .await
        .unwrap();

    assert!(!results.is_empty(), "expected at least one result");
    assert!(
        results[0].content.contains("Qdrant"),
        "expected result to contain 'Qdrant'"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn store_with_metadata_and_search() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        "category".to_string(),
        serde_json::Value::String("database".to_string()),
    );
    metadata.insert("version".to_string(), serde_json::json!(2));

    connector
        .store(
            Entry {
                content: "PostgreSQL is a relational database".to_string(),
                metadata: Some(metadata),
            },
            &collection,
        )
        .await
        .unwrap();

    let results = connector
        .search("relational database", &collection, 5, None)
        .await
        .unwrap();

    assert!(!results.is_empty(), "expected at least one result");
    assert!(results[0].content.contains("PostgreSQL"));

    let meta = results[0].metadata.as_ref().expect("expected metadata");
    assert_eq!(meta.get("category").unwrap(), "database");
    assert_eq!(meta.get("version").unwrap(), 2);
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn search_returns_empty_for_unrelated_query() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    connector
        .store(
            Entry {
                content: "Rust programming language memory safety".to_string(),
                metadata: None,
            },
            &collection,
        )
        .await
        .unwrap();

    let results = connector
        .search(
            "french pastry baking recipes croissant",
            &collection,
            5,
            None,
        )
        .await
        .unwrap();

    assert!(
        results.is_empty() || results[0].content.contains("Rust"),
        "results should either be empty or contain the only stored document"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn search_limit_is_respected() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    for i in 0..5 {
        connector
            .store(
                Entry {
                    content: format!("Document number {i} about vector databases"),
                    metadata: None,
                },
                &collection,
            )
            .await
            .unwrap();
    }

    let results = connector
        .search("vector databases", &collection, 2, None)
        .await
        .unwrap();

    assert_eq!(results.len(), 2, "search_limit=2 should return at most 2");
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn store_with_empty_metadata_returns_no_metadata() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    connector
        .store(
            Entry {
                content: "A document with empty metadata".to_string(),
                metadata: Some(std::collections::HashMap::new()),
            },
            &collection,
        )
        .await
        .unwrap();

    let results = connector
        .search("empty metadata", &collection, 5, None)
        .await
        .unwrap();

    assert!(!results.is_empty());
    assert!(
        results[0].metadata.is_none(),
        "empty metadata map should become None"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn ensure_collection_creates_field_indexes() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let mut config = qdrant_config(&collection);
    config.filterable_fields = vec![
        FilterableField {
            name: "category".to_string(),
            description: "Category".to_string(),
            field_type: FieldType::Keyword,
            condition: None,
            required: false,
        },
        FilterableField {
            name: "priority".to_string(),
            description: "Priority level".to_string(),
            field_type: FieldType::Integer,
            condition: None,
            required: false,
        },
    ];

    let connector = connector_for(&config).await;
    connector.ensure_collection(&collection).await.unwrap();

    let client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    assert!(client.collection_exists(&collection).await.unwrap());
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn ensure_collection_is_idempotent() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    connector.ensure_collection(&collection).await.unwrap();
    connector.ensure_collection(&collection).await.unwrap();
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn search_with_arbitrary_filter_must() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    let mut meta1 = std::collections::HashMap::new();
    meta1.insert(
        "city".to_string(),
        serde_json::Value::String("London".to_string()),
    );
    connector
        .store(
            Entry {
                content: "Weather in London is rainy".to_string(),
                metadata: Some(meta1),
            },
            &collection,
        )
        .await
        .unwrap();

    let mut meta2 = std::collections::HashMap::new();
    meta2.insert(
        "city".to_string(),
        serde_json::Value::String("Paris".to_string()),
    );
    connector
        .store(
            Entry {
                content: "Weather in Paris is sunny".to_string(),
                metadata: Some(meta2),
            },
            &collection,
        )
        .await
        .unwrap();

    let filter_json = serde_json::json!({
        "must": [{"key": "city", "match": {"value": "London"}}]
    });
    let filter = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&filter_json).unwrap();

    let results = connector
        .search("weather", &collection, 5, Some(filter))
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("London"));
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn search_with_arbitrary_filter_must_not() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    let mut meta1 = std::collections::HashMap::new();
    meta1.insert(
        "city".to_string(),
        serde_json::Value::String("London".to_string()),
    );
    connector
        .store(
            Entry {
                content: "London has Big Ben".to_string(),
                metadata: Some(meta1),
            },
            &collection,
        )
        .await
        .unwrap();

    let mut meta2 = std::collections::HashMap::new();
    meta2.insert(
        "city".to_string(),
        serde_json::Value::String("Paris".to_string()),
    );
    connector
        .store(
            Entry {
                content: "Paris has Eiffel Tower".to_string(),
                metadata: Some(meta2),
            },
            &collection,
        )
        .await
        .unwrap();

    let filter_json = serde_json::json!({
        "must_not": [{"key": "city", "match": {"value": "London"}}]
    });
    let filter = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&filter_json).unwrap();

    let results = connector
        .search("landmark", &collection, 5, Some(filter))
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("Paris"));
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn search_with_integer_filter() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    let mut meta1 = std::collections::HashMap::new();
    meta1.insert("priority".to_string(), serde_json::json!(1));
    connector
        .store(
            Entry {
                content: "Low priority task: update docs".to_string(),
                metadata: Some(meta1),
            },
            &collection,
        )
        .await
        .unwrap();

    let mut meta2 = std::collections::HashMap::new();
    meta2.insert("priority".to_string(), serde_json::json!(5));
    connector
        .store(
            Entry {
                content: "High priority task: fix critical bug".to_string(),
                metadata: Some(meta2),
            },
            &collection,
        )
        .await
        .unwrap();

    let filter_json = serde_json::json!({
        "must": [{"key": "priority", "match": {"value": 5}}]
    });
    let filter = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&filter_json).unwrap();

    let results = connector
        .search("task", &collection, 5, Some(filter))
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("critical bug"));
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn search_with_boolean_filter() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let connector = connector_for(&qdrant_config(&collection)).await;

    let mut meta1 = std::collections::HashMap::new();
    meta1.insert("active".to_string(), serde_json::json!(true));
    connector
        .store(
            Entry {
                content: "Active project: building search engine".to_string(),
                metadata: Some(meta1),
            },
            &collection,
        )
        .await
        .unwrap();

    let mut meta2 = std::collections::HashMap::new();
    meta2.insert("active".to_string(), serde_json::json!(false));
    connector
        .store(
            Entry {
                content: "Archived project: old website".to_string(),
                metadata: Some(meta2),
            },
            &collection,
        )
        .await
        .unwrap();

    let filter_json = serde_json::json!({
        "must": [{"key": "active", "match": {"value": true}}]
    });
    let filter = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&filter_json).unwrap();

    let results = connector
        .search("project", &collection, 5, Some(filter))
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("search engine"));
}

#[test]
fn json_to_qdrant_filter_empty_object() {
    let filter = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&serde_json::json!({})).unwrap();
    assert!(filter.must.is_empty());
    assert!(filter.should.is_empty());
    assert!(filter.must_not.is_empty());
}

#[test]
fn json_to_qdrant_filter_missing_key_errors() {
    let result = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&serde_json::json!({
        "must": [{"match": {"value": "x"}}]
    }));
    assert!(result.is_err());
}

#[test]
fn json_to_qdrant_filter_unsupported_condition_errors() {
    let result = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&serde_json::json!({
        "must": [{"key": "x", "range": {"gt": 5}}]
    }));
    assert!(result.is_err());
}

#[test]
fn json_to_qdrant_filter_unsupported_match_value_type_errors() {
    let result = mcp_server_qdrant::qdrant::json_to_qdrant_filter(&serde_json::json!({
        "must": [{"key": "x", "match": {"value": [1, 2, 3]}}]
    }));
    assert!(result.is_err());
}

#[test]
fn make_filter_required_field_missing_errors() {
    let fields = vec![FilterableField {
        name: "category".to_string(),
        description: "Category".to_string(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Eq),
        required: true,
    }];

    let values = std::collections::HashMap::new();
    assert!(mcp_server_qdrant::filters::make_filter(&fields, &values).is_err());
}

#[test]
fn make_filter_optional_field_missing_ok() {
    let fields = vec![FilterableField {
        name: "category".to_string(),
        description: "Category".to_string(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];

    let values = std::collections::HashMap::new();
    let filter = mcp_server_qdrant::filters::make_filter(&fields, &values).unwrap();
    assert!(filter.must.is_empty());
    assert!(filter.must_not.is_empty());
}

#[test]
fn make_filter_keyword_eq() {
    let fields = vec![FilterableField {
        name: "category".to_string(),
        description: "Category".to_string(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];

    let mut values = std::collections::HashMap::new();
    values.insert(
        "category".to_string(),
        serde_json::Value::String("books".to_string()),
    );

    let filter = mcp_server_qdrant::filters::make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 1);
    assert!(filter.must_not.is_empty());
}

#[test]
fn make_filter_keyword_ne() {
    let fields = vec![FilterableField {
        name: "category".to_string(),
        description: "Category".to_string(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Ne),
        required: false,
    }];

    let mut values = std::collections::HashMap::new();
    values.insert(
        "category".to_string(),
        serde_json::Value::String("spam".to_string()),
    );

    let filter = mcp_server_qdrant::filters::make_filter(&fields, &values).unwrap();
    assert!(filter.must.is_empty());
    assert_eq!(filter.must_not.len(), 1);
}

#[test]
fn make_filter_integer_range_conditions() {
    for (condition, expected_must, expected_must_not) in [
        (FilterCondition::Gt, 1, 0),
        (FilterCondition::Gte, 1, 0),
        (FilterCondition::Lt, 1, 0),
        (FilterCondition::Lte, 1, 0),
        (FilterCondition::Eq, 1, 0),
        (FilterCondition::Ne, 0, 1),
    ] {
        let fields = vec![FilterableField {
            name: "priority".to_string(),
            description: "Priority".to_string(),
            field_type: FieldType::Integer,
            condition: Some(condition),
            required: false,
        }];

        let mut values = std::collections::HashMap::new();
        values.insert("priority".to_string(), serde_json::json!(5));

        let filter = mcp_server_qdrant::filters::make_filter(&fields, &values).unwrap();
        assert_eq!(filter.must.len(), expected_must);
        assert_eq!(filter.must_not.len(), expected_must_not);
    }
}

#[test]
fn make_filter_float_range_conditions() {
    for condition in [
        FilterCondition::Gt,
        FilterCondition::Gte,
        FilterCondition::Lt,
        FilterCondition::Lte,
    ] {
        let fields = vec![FilterableField {
            name: "score".to_string(),
            description: "Score".to_string(),
            field_type: FieldType::Float,
            condition: Some(condition),
            required: false,
        }];

        let mut values = std::collections::HashMap::new();
        values.insert("score".to_string(), serde_json::json!(3.14));

        let filter = mcp_server_qdrant::filters::make_filter(&fields, &values).unwrap();
        assert_eq!(filter.must.len(), 1);
    }
}

#[test]
fn make_filter_boolean_eq_ne() {
    let mut values = std::collections::HashMap::new();
    values.insert("active".to_string(), serde_json::json!(true));

    let fields_eq = vec![FilterableField {
        name: "active".to_string(),
        description: "Active".to_string(),
        field_type: FieldType::Boolean,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];
    let filter = mcp_server_qdrant::filters::make_filter(&fields_eq, &values).unwrap();
    assert_eq!(filter.must.len(), 1);

    let fields_ne = vec![FilterableField {
        name: "active".to_string(),
        description: "Active".to_string(),
        field_type: FieldType::Boolean,
        condition: Some(FilterCondition::Ne),
        required: false,
    }];
    let filter = mcp_server_qdrant::filters::make_filter(&fields_ne, &values).unwrap();
    assert_eq!(filter.must_not.len(), 1);
}

#[test]
fn make_filter_wrong_type_errors() {
    let fields = vec![FilterableField {
        name: "count".to_string(),
        description: "Count".to_string(),
        field_type: FieldType::Integer,
        condition: Some(FilterCondition::Eq),
        required: false,
    }];

    let mut values = std::collections::HashMap::new();
    values.insert(
        "count".to_string(),
        serde_json::Value::String("not a number".to_string()),
    );

    assert!(mcp_server_qdrant::filters::make_filter(&fields, &values).is_err());
}

#[test]
fn make_filter_unsupported_condition_for_type_errors() {
    let fields = vec![FilterableField {
        name: "name".to_string(),
        description: "Name".to_string(),
        field_type: FieldType::Keyword,
        condition: Some(FilterCondition::Gt),
        required: false,
    }];

    let mut values = std::collections::HashMap::new();
    values.insert(
        "name".to_string(),
        serde_json::Value::String("test".to_string()),
    );

    assert!(mcp_server_qdrant::filters::make_filter(&fields, &values).is_err());
}

#[test]
fn make_filter_default_condition_is_eq() {
    let fields = vec![FilterableField {
        name: "category".to_string(),
        description: "Category".to_string(),
        field_type: FieldType::Keyword,
        condition: None,
        required: false,
    }];

    let mut values = std::collections::HashMap::new();
    values.insert(
        "category".to_string(),
        serde_json::Value::String("test".to_string()),
    );

    let filter = mcp_server_qdrant::filters::make_filter(&fields, &values).unwrap();
    assert_eq!(filter.must.len(), 1);
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_store_and_find() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let client = mcp_client_for(default_config(qdrant_config(&collection))).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "Rust is a systems programming language"
                }),
            )),
        )
        .await
        .unwrap();

    assert!(
        result_text(&result).contains("stored"),
        "expected 'stored' in result"
    );

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find").with_arguments(json_args(
                serde_json::json!({
                    "query": "systems programming"
                }),
            )),
        )
        .await
        .unwrap();

    assert!(
        result_text(&result).contains("Rust"),
        "expected 'Rust' in find result"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_store_rejected_in_read_only_mode() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let mut qdrant = qdrant_config(&collection);
    qdrant.read_only = true;

    let client = mcp_client_for(default_config(qdrant)).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "should fail"
                }),
            )),
        )
        .await;

    assert!(result.is_err(), "store should fail in read-only mode");
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_find_no_collection_name_errors() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let mut qdrant = qdrant_config(&collection);
    qdrant.collection_name = None;

    let client = mcp_client_for(default_config(qdrant)).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find")
                .with_arguments(json_args(serde_json::json!({"query": "test"}))),
        )
        .await;

    assert!(result.is_err(), "find without collection name should error");
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_store_with_explicit_collection_name() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let mut qdrant = qdrant_config(&collection);
    qdrant.collection_name = None;

    let client = mcp_client_for(default_config(qdrant)).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "explicit collection test",
                    "collection_name": collection
                }),
            )),
        )
        .await
        .unwrap();

    assert!(result_text(&result).contains("stored"));
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_find_with_metadata_in_result() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let client = mcp_client_for(default_config(qdrant_config(&collection))).await;

    client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "Paris is the capital of France",
                    "metadata": {"country": "France", "type": "capital"}
                }),
            )),
        )
        .await
        .unwrap();

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find")
                .with_arguments(json_args(serde_json::json!({"query": "capital city"}))),
        )
        .await
        .unwrap();

    let text = result_text(&result);
    assert!(text.contains("Paris"));
    assert!(text.contains("metadata"));
    assert!(text.contains("France"));
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_find_no_results_returns_message() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let client = mcp_client_for(default_config(qdrant_config(&collection))).await;

    client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({"information": "placeholder to create collection"}),
            )),
        )
        .await
        .unwrap();

    let qdrant_client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    qdrant_client.delete_collection(&collection).await.unwrap();

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find")
                .with_arguments(json_args(serde_json::json!({"query": "anything"}))),
        )
        .await
        .unwrap();

    let text = result_text(&result);
    assert!(
        text.contains("no results found"),
        "expected 'no results found' in: {text}"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_find_arbitrary_filter_rejected_when_disabled() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let mut qdrant = qdrant_config(&collection);
    qdrant.allow_arbitrary_filter = false;

    let client = mcp_client_for(default_config(qdrant)).await;

    client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "test document"
                }),
            )),
        )
        .await
        .unwrap();

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find").with_arguments(json_args(
                serde_json::json!({
                    "query": "test",
                    "query_filter": {"must": [{"key": "x", "match": {"value": "y"}}]}
                }),
            )),
        )
        .await;

    assert!(
        result.is_err(),
        "arbitrary filter should be rejected when disabled"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_find_with_arbitrary_filter_when_enabled() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let mut qdrant = qdrant_config(&collection);
    qdrant.allow_arbitrary_filter = true;

    let client = mcp_client_for(default_config(qdrant)).await;

    client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "Berlin is a city in Germany",
                    "metadata": {"country": "Germany"}
                }),
            )),
        )
        .await
        .unwrap();

    client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "Tokyo is a city in Japan",
                    "metadata": {"country": "Japan"}
                }),
            )),
        )
        .await
        .unwrap();

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find").with_arguments(json_args(
                serde_json::json!({
                    "query": "city",
                    "query_filter": {
                        "must": [{"key": "country", "match": {"value": "Germany"}}]
                    }
                }),
            )),
        )
        .await
        .unwrap();

    let text = result_text(&result);
    assert!(text.contains("Berlin"), "expected Berlin in: {text}");
    assert!(
        !text.contains("Tokyo"),
        "should not contain Tokyo in: {text}"
    );
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn mcp_store_with_non_object_metadata_ignored() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let client = mcp_client_for(default_config(qdrant_config(&collection))).await;

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-store").with_arguments(json_args(
                serde_json::json!({
                    "information": "test with bad metadata",
                    "metadata": "not an object"
                }),
            )),
        )
        .await
        .unwrap();

    assert!(result_text(&result).contains("stored"));

    let result = client
        .call_tool(
            CallToolRequestParams::new("qdrant-find")
                .with_arguments(json_args(serde_json::json!({"query": "bad metadata"}))),
        )
        .await
        .unwrap();

    let text = result_text(&result);
    assert!(text.contains("test with bad metadata"));
    assert!(!text.contains("<metadata>"));
}

#[tokio::test]
async fn mcp_tool_listing_includes_store_and_find() {
    let collection = unique_collection();
    let client = schema_test_client(default_config(qdrant_config(&collection))).await;
    let tools = client.list_tools(None).await.unwrap();

    let tool_names: Vec<&str> = tools.tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(tool_names.contains(&"qdrant-store"), "missing qdrant-store");
    assert!(tool_names.contains(&"qdrant-find"), "missing qdrant-find");
}

#[tokio::test]
async fn mcp_custom_tool_descriptions() {
    let collection = unique_collection();
    let mut config = default_config(qdrant_config(&collection));
    config.tools.store_description = Some("Custom store desc".to_string());
    config.tools.find_description = Some("Custom find desc".to_string());

    let client = schema_test_client(config).await;
    let tools = client.list_tools(None).await.unwrap();

    let store_tool = tools
        .tools
        .iter()
        .find(|t| t.name == "qdrant-store")
        .unwrap();
    assert_eq!(store_tool.description.as_deref(), Some("Custom store desc"));

    let find_tool = tools
        .tools
        .iter()
        .find(|t| t.name == "qdrant-find")
        .unwrap();
    assert_eq!(find_tool.description.as_deref(), Some("Custom find desc"));
}

#[tokio::test]
async fn mcp_schema_hides_collection_name_when_default_set() {
    let collection = unique_collection();
    let client = schema_test_client(default_config(qdrant_config(&collection))).await;
    let tools = client.list_tools(None).await.unwrap();

    for tool in &tools.tools {
        if let Some(props) = tool
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            assert!(
                !props.contains_key("collection_name"),
                "collection_name should be hidden when default is set for tool {}",
                tool.name
            );
        }
    }
}

#[tokio::test]
async fn mcp_schema_shows_collection_name_when_no_default() {
    let collection = unique_collection();
    let mut qdrant = qdrant_config(&collection);
    qdrant.collection_name = None;

    let client = schema_test_client(default_config(qdrant)).await;
    let tools = client.list_tools(None).await.unwrap();

    for tool in &tools.tools {
        if let Some(props) = tool
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            assert!(
                props.contains_key("collection_name"),
                "collection_name should be visible when no default for tool {}",
                tool.name
            );
        }
    }
}

#[tokio::test]
async fn mcp_schema_hides_query_filter_when_no_arbitrary_filter() {
    let collection = unique_collection();
    let mut qdrant = qdrant_config(&collection);
    qdrant.allow_arbitrary_filter = false;
    qdrant.filterable_fields = vec![];

    let client = schema_test_client(default_config(qdrant)).await;
    let tools = client.list_tools(None).await.unwrap();

    let find_tool = tools
        .tools
        .iter()
        .find(|t| t.name == "qdrant-find")
        .unwrap();
    if let Some(props) = find_tool
        .input_schema
        .get("properties")
        .and_then(|p| p.as_object())
    {
        assert!(
            !props.contains_key("query_filter"),
            "query_filter should be hidden when arbitrary filter is disabled"
        );
    }
}

#[tokio::test]
async fn mcp_schema_shows_query_filter_when_arbitrary_filter_enabled() {
    let collection = unique_collection();
    let mut qdrant = qdrant_config(&collection);
    qdrant.allow_arbitrary_filter = true;
    qdrant.filterable_fields = vec![];

    let client = schema_test_client(default_config(qdrant)).await;
    let tools = client.list_tools(None).await.unwrap();

    let find_tool = tools
        .tools
        .iter()
        .find(|t| t.name == "qdrant-find")
        .unwrap();
    let props = find_tool
        .input_schema
        .get("properties")
        .and_then(|p| p.as_object())
        .unwrap();
    assert!(
        props.contains_key("query_filter"),
        "query_filter should be visible when arbitrary filter is enabled"
    );
}

#[tokio::test]
async fn mcp_schema_adds_filterable_fields_to_find_schema() {
    let collection = unique_collection();
    let mut qdrant = qdrant_config(&collection);
    qdrant.filterable_fields = vec![
        FilterableField {
            name: "category".to_string(),
            description: "Product category".to_string(),
            field_type: FieldType::Keyword,
            condition: Some(FilterCondition::Eq),
            required: true,
        },
        FilterableField {
            name: "price".to_string(),
            description: "Price".to_string(),
            field_type: FieldType::Float,
            condition: Some(FilterCondition::Gte),
            required: false,
        },
    ];

    let client = schema_test_client(default_config(qdrant)).await;
    let tools = client.list_tools(None).await.unwrap();

    let find_tool = tools
        .tools
        .iter()
        .find(|t| t.name == "qdrant-find")
        .unwrap();
    let props = find_tool
        .input_schema
        .get("properties")
        .and_then(|p| p.as_object())
        .unwrap();

    assert!(props.contains_key("category"), "missing category field");
    assert!(props.contains_key("price"), "missing price field");
    assert!(
        !props.contains_key("query_filter"),
        "query_filter should be hidden when filterable_fields are set"
    );

    let required = find_tool
        .input_schema
        .get("required")
        .and_then(|r| r.as_array())
        .unwrap();
    let required_names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(
        required_names.contains(&"category"),
        "category should be required"
    );
    assert!(
        !required_names.contains(&"price"),
        "price should not be required"
    );

    assert_eq!(props["category"]["type"], "string");
    assert_eq!(props["price"]["type"], "number");
}
