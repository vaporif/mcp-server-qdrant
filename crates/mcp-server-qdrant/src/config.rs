use std::net::IpAddr;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::filters::FilterableField;

#[derive(Parser, Debug)]
#[command(
    name = "mcp-server-qdrant",
    about = "MCP server for Qdrant vector database"
)]
pub struct Cli {
    /// Transport protocol
    #[arg(long, default_value = "stdio", env = "MCP_TRANSPORT")]
    pub transport: TransportArg,

    /// Qdrant server URL (mutually exclusive with --qdrant-local-path)
    #[arg(long, env = "QDRANT_URL")]
    pub qdrant_url: Option<String>,

    /// Qdrant API key
    #[arg(long, env = "QDRANT_API_KEY")]
    pub qdrant_api_key: Option<String>,

    /// Local Qdrant storage path (mutually exclusive with --qdrant-url)
    #[arg(long, env = "QDRANT_LOCAL_PATH")]
    pub qdrant_local_path: Option<PathBuf>,

    /// Default collection name
    #[arg(long, env = "COLLECTION_NAME")]
    pub collection_name: Option<String>,

    /// Max search results
    #[arg(long, default_value = "10", env = "QDRANT_SEARCH_LIMIT")]
    pub search_limit: usize,

    /// Read-only mode (disable store tool)
    #[arg(long, default_value = "false", env = "QDRANT_READ_ONLY")]
    pub read_only: bool,

    /// Embedding model name
    #[arg(
        long,
        default_value = "sentence-transformers/all-MiniLM-L6-v2",
        env = "EMBEDDING_MODEL"
    )]
    pub embedding_model: String,

    /// Custom store tool description
    #[arg(long, env = "TOOL_STORE_DESCRIPTION")]
    pub tool_store_description: Option<String>,

    /// Custom find tool description
    #[arg(long, env = "TOOL_FIND_DESCRIPTION")]
    pub tool_find_description: Option<String>,

    /// Filterable fields as JSON array
    #[arg(long, env = "FILTERABLE_FIELDS")]
    pub filterable_fields: Option<String>,

    /// Allow arbitrary Qdrant filter syntax
    #[arg(long, default_value = "false", env = "QDRANT_ALLOW_ARBITRARY_FILTER")]
    pub allow_arbitrary_filter: bool,

    /// Host to bind for SSE/HTTP transports
    #[arg(long, default_value = "127.0.0.1", env = "HOST")]
    pub host: IpAddr,

    /// Port for SSE/HTTP transports
    #[arg(long, default_value = "8000", env = "PORT")]
    pub port: u16,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum TransportArg {
    Stdio,
    Sse,
    StreamableHttp,
}

pub enum Transport {
    Stdio,
    Sse { host: IpAddr, port: u16 },
    StreamableHttp { host: IpAddr, port: u16 },
}

pub enum QdrantLocation {
    Remote {
        url: String,
        api_key: Option<String>,
    },
    Local {
        path: PathBuf,
    },
}

pub struct QdrantConfig {
    pub location: QdrantLocation,
    pub collection_name: Option<String>,
    pub search_limit: usize,
    pub read_only: bool,
    pub filterable_fields: Vec<FilterableField>,
    pub allow_arbitrary_filter: bool,
}

pub struct EmbeddingConfig {
    pub model_name: String,
}

pub struct ToolConfig {
    pub store_description: Option<String>,
    pub find_description: Option<String>,
}

pub struct Config {
    pub qdrant: QdrantConfig,
    pub embedding: EmbeddingConfig,
    pub tools: ToolConfig,
    pub transport: Transport,
}

impl Config {
    #[allow(clippy::result_large_err)]
    pub fn from_cli(cli: Cli) -> Result<Self, crate::errors::Error> {
        let location = match (&cli.qdrant_url, &cli.qdrant_local_path) {
            (Some(url), None) => QdrantLocation::Remote {
                url: url.clone(),
                api_key: cli.qdrant_api_key.clone(),
            },
            (None, Some(path)) => QdrantLocation::Local { path: path.clone() },
            (Some(_), Some(_)) => {
                return Err(crate::errors::Error::Config(
                    "cannot set both QDRANT_URL and QDRANT_LOCAL_PATH".into(),
                ));
            }
            (None, None) => {
                return Err(crate::errors::Error::Config(
                    "must set either QDRANT_URL or QDRANT_LOCAL_PATH".into(),
                ));
            }
        };

        let filterable_fields = match &cli.filterable_fields {
            Some(json) => serde_json::from_str(json).map_err(|e| {
                crate::errors::Error::Config(format!("invalid FILTERABLE_FIELDS JSON: {e}"))
            })?,
            None => vec![],
        };

        let transport = match cli.transport {
            TransportArg::Stdio => Transport::Stdio,
            TransportArg::Sse => Transport::Sse {
                host: cli.host,
                port: cli.port,
            },
            TransportArg::StreamableHttp => Transport::StreamableHttp {
                host: cli.host,
                port: cli.port,
            },
        };

        Ok(Self {
            qdrant: QdrantConfig {
                location,
                collection_name: cli.collection_name,
                search_limit: cli.search_limit,
                read_only: cli.read_only,
                filterable_fields,
                allow_arbitrary_filter: cli.allow_arbitrary_filter,
            },
            embedding: EmbeddingConfig {
                model_name: cli.embedding_model,
            },
            tools: ToolConfig {
                store_description: cli.tool_store_description,
                find_description: cli.tool_find_description,
            },
            transport,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_cli_remote() {
        let cli = Cli::parse_from([
            "mcp-server-qdrant",
            "--qdrant-url",
            "http://localhost:6334",
            "--qdrant-api-key",
            "test-key",
        ]);
        let config = Config::from_cli(cli).unwrap();
        assert!(matches!(
            config.qdrant.location,
            QdrantLocation::Remote { .. }
        ));
        if let QdrantLocation::Remote { url, api_key } = &config.qdrant.location {
            assert_eq!(url, "http://localhost:6334");
            assert_eq!(api_key.as_deref(), Some("test-key"));
        }
    }

    #[test]
    fn config_from_cli_local() {
        let cli = Cli::parse_from(["mcp-server-qdrant", "--qdrant-local-path", "/tmp/qdrant"]);
        let config = Config::from_cli(cli).unwrap();
        assert!(matches!(
            config.qdrant.location,
            QdrantLocation::Local { .. }
        ));
        if let QdrantLocation::Local { path } = &config.qdrant.location {
            assert_eq!(path, &PathBuf::from("/tmp/qdrant"));
        }
    }

    #[test]
    fn config_rejects_both_url_and_path() {
        let cli = Cli::parse_from([
            "mcp-server-qdrant",
            "--qdrant-url",
            "http://localhost:6334",
            "--qdrant-local-path",
            "/tmp/qdrant",
        ]);
        assert!(Config::from_cli(cli).is_err());
    }

    #[test]
    fn config_rejects_neither_url_nor_path() {
        let cli = Cli::parse_from(["mcp-server-qdrant"]);
        assert!(Config::from_cli(cli).is_err());
    }

    #[test]
    fn config_parses_filterable_fields() {
        let cli = Cli::parse_from([
            "mcp-server-qdrant",
            "--qdrant-url",
            "http://localhost:6334",
            "--filterable-fields",
            r#"[{"name":"category","description":"Category","field_type":"keyword","required":false}]"#,
        ]);
        let config = Config::from_cli(cli).unwrap();
        assert_eq!(config.qdrant.filterable_fields.len(), 1);
        assert_eq!(config.qdrant.filterable_fields[0].name, "category");
    }

    #[test]
    fn config_defaults() {
        let cli = Cli::parse_from(["mcp-server-qdrant", "--qdrant-url", "http://localhost:6334"]);
        let config = Config::from_cli(cli).unwrap();
        assert_eq!(config.qdrant.search_limit, 10);
        assert!(!config.qdrant.read_only);
        assert!(!config.qdrant.allow_arbitrary_filter);
        assert_eq!(
            config.embedding.model_name,
            "sentence-transformers/all-MiniLM-L6-v2"
        );
        assert!(config.qdrant.filterable_fields.is_empty());
        assert!(config.tools.store_description.is_none());
        assert!(config.tools.find_description.is_none());
    }
}
