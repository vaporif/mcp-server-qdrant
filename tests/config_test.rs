use clap::Parser;
use mcp_server_qdrant::config::{Cli, Config, QdrantLocation};

#[test]
fn cli_parses_minimal_remote() {
    let cli = Cli::parse_from(["mcp-server-qdrant", "--qdrant-url", "http://localhost:6334"]);
    let config = Config::from_cli(cli).unwrap();
    assert!(matches!(
        config.qdrant.location,
        QdrantLocation::Remote { .. }
    ));
    assert_eq!(config.qdrant.search_limit, 10);
    assert!(!config.qdrant.read_only);
}

#[test]
fn cli_parses_all_options() {
    let cli = Cli::parse_from([
        "mcp-server-qdrant",
        "--qdrant-url",
        "http://localhost:6334",
        "--qdrant-api-key",
        "my-key",
        "--collection-name",
        "test-collection",
        "--search-limit",
        "5",
        "--read-only",
        "--embedding-model",
        "my-model",
        "--tool-store-description",
        "Custom store",
        "--tool-find-description",
        "Custom find",
        "--transport",
        "sse",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
    ]);
    let config = Config::from_cli(cli).unwrap();
    assert_eq!(config.qdrant.search_limit, 5);
    assert!(config.qdrant.read_only);
    assert_eq!(
        config.qdrant.collection_name.as_deref(),
        Some("test-collection")
    );
    assert_eq!(config.embedding.model_name, "my-model");
}
