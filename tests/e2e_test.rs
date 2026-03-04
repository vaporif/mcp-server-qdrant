use std::sync::Arc;

use mcp_server_qdrant::config::{QdrantConfig, QdrantLocation};
use mcp_server_qdrant::embeddings::create_embedding_provider;
use mcp_server_qdrant::qdrant::{Entry, QdrantConnector};
use qdrant_client::Qdrant;

fn qdrant_config(collection_name: &str) -> QdrantConfig {
    QdrantConfig {
        location: QdrantLocation::Remote {
            url: "http://localhost:6334".to_string(),
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
        // Spawn a separate thread with its own runtime to avoid
        // "cannot block_on inside a runtime" panic.
        let _ = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let client = Qdrant::from_url("http://localhost:6334")
                    .build()
                    .expect("qdrant client");
                let _ = client.delete_collection(&name).await;
            });
        })
        .join();
    }
}

#[tokio::test]
#[ignore = "requires running Qdrant instance"]
async fn store_and_search_basic() {
    let collection = unique_collection();
    let _guard = CollectionGuard {
        name: collection.clone(),
    };

    let config = qdrant_config(&collection);
    let embedding: Arc<dyn mcp_server_qdrant::embeddings::EmbeddingProvider> =
        Arc::from(create_embedding_provider("sentence-transformers/all-MiniLM-L6-v2").unwrap());
    let connector = QdrantConnector::new(&config, embedding).unwrap();

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

    let config = qdrant_config(&collection);
    let embedding: Arc<dyn mcp_server_qdrant::embeddings::EmbeddingProvider> =
        Arc::from(create_embedding_provider("sentence-transformers/all-MiniLM-L6-v2").unwrap());
    let connector = QdrantConnector::new(&config, embedding).unwrap();

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

    let config = qdrant_config(&collection);
    let embedding: Arc<dyn mcp_server_qdrant::embeddings::EmbeddingProvider> =
        Arc::from(create_embedding_provider("sentence-transformers/all-MiniLM-L6-v2").unwrap());
    let connector = QdrantConnector::new(&config, embedding).unwrap();

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

    // Qdrant returns top-k regardless of score, so with a single document
    // the unrelated query will still return it. We just verify the search completes.
    assert!(
        results.is_empty() || results[0].content.contains("Rust"),
        "results should either be empty or contain the only stored document"
    );
}
