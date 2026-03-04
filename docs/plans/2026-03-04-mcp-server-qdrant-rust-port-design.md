# MCP Server Qdrant — Rust Port Design

Port of [mcp-server-qdrant-py](../mcp-server-qdrant-py) from Python to Rust.

## Decisions

- **MCP SDK:** rmcp (v1.1.0, official Rust MCP SDK)
- **Qdrant client:** qdrant-client (gRPC, official)
- **Embeddings:** Dual candle/ONNX backend (feature-flagged), same pattern as parry
- **Model:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Crate layout:** Single crate in workspace
- **Scope:** Full port of all Python features
- **Style:** Rust-idiomatic (enums for variants, type-level mutual exclusion)

## Module Layout

```
crates/mcp-server-qdrant/src/
├── main.rs          # CLI (clap), transport selection, server launch
├── config.rs        # Strongly-typed config from CLI args + env vars
├── server.rs        # MCP tool definitions via rmcp #[tool] macros
├── qdrant.rs        # QdrantConnector — store, search, collection mgmt
├── embeddings.rs    # EmbeddingProvider trait + candle/ONNX impls
├── filters.rs       # Typed filter system (enums, pattern matching)
└── errors.rs        # thiserror error types
```

## Dependencies

**Core:**
- `rmcp` — MCP server (stdio/SSE/streamable-http)
- `qdrant-client` — Qdrant gRPC client
- `clap` (derive) — CLI + env var parsing
- `serde` / `serde_json` — serialization
- `tokio` — async runtime
- `thiserror` — error types
- `tracing` — structured logging

**Embeddings (shared):**
- `tokenizers` — HuggingFace tokenization
- `hf-hub` — model download from HuggingFace

**Embeddings (candle, default):**
- `candle-core`, `candle-nn`

**Embeddings (onnx, optional):**
- `ort`, `ndarray`

## Config

```rust
struct Config {
    qdrant: QdrantConfig,
    embedding: EmbeddingConfig,
    tools: ToolConfig,
    transport: Transport,
}

enum Transport {
    Stdio,
    Sse { host: IpAddr, port: u16 },
    StreamableHttp { host: IpAddr, port: u16 },
}

enum QdrantLocation {
    Remote { url: String, api_key: Option<String> },
    Local { path: PathBuf },
}

struct QdrantConfig {
    location: QdrantLocation,
    collection_name: Option<String>,
    search_limit: usize,              // default: 10
    read_only: bool,
    filterable_fields: Vec<FilterableField>,
    allow_arbitrary_filter: bool,
}

struct EmbeddingConfig {
    model_name: String,                // default: sentence-transformers/all-MiniLM-L6-v2
}

struct ToolConfig {
    store_description: Option<String>,
    find_description: Option<String>,
}
```

`QdrantLocation` enum enforces mutual exclusion at the type level — no runtime validation needed.

Env vars match Python version: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_LOCAL_PATH`, `COLLECTION_NAME`, `QDRANT_SEARCH_LIMIT`, `QDRANT_READ_ONLY`, `EMBEDDING_MODEL`, `TOOL_STORE_DESCRIPTION`, `TOOL_FIND_DESCRIPTION`.

## Embeddings

Feature-flagged dual backend (compile-time, same as parry):
- `candle` (default) — pure Rust, portable
- `onnx` — user provides `ORT_DYLIB_PATH`
- `onnx-fetch` — auto-downloads ONNX runtime

```rust
#[async_trait]
trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

#[cfg(feature = "candle")]
struct CandleEmbedder { /* candle-core + candle-nn + tokenizers */ }

#[cfg(any(feature = "onnx", feature = "onnx-fetch"))]
struct OnnxEmbedder { /* ort + tokenizers + ndarray */ }
```

Model: all-MiniLM-L6-v2 (BERT encoder). Tokenize → forward pass → mean pooling → 384-dim vector.

## Filters

```rust
enum FieldType { Keyword, Integer, Float, Boolean }
enum FilterCondition { Eq, Ne, Gt, Gte, Lt, Lte, Any, Except }

struct FilterableField {
    name: String,
    description: String,
    field_type: FieldType,
    condition: Option<FilterCondition>,
    required: bool,
}
```

Enums replace Python's string literals. Pattern matching gives exhaustive compile-time checks when converting to Qdrant filter types.

## QdrantConnector

```rust
struct QdrantConnector {
    client: QdrantClient,
    embedding: Arc<dyn EmbeddingProvider>,
    default_collection: Option<String>,
    indexes: Vec<FilterableField>,
}

impl QdrantConnector {
    async fn store(&self, entry: Entry, collection: &str) -> Result<()>;
    async fn search(&self, query: &str, collection: &str, limit: usize,
                    filter: Option<qdrant_client::Filter>) -> Result<Vec<Entry>>;
    async fn ensure_collection(&self, collection: &str) -> Result<()>;
}

struct Entry {
    content: String,
    metadata: Option<serde_json::Value>,
}
```

## MCP Server

```rust
#[derive(Clone)]
struct QdrantMcpServer {
    connector: Arc<QdrantConnector>,
    config: Arc<Config>,
}

#[tool(tool_box)]
impl QdrantMcpServer {
    #[tool(description = "Store information in Qdrant")]
    async fn qdrant_store(&self, information: String, metadata: Option<Value>,
                          collection_name: Option<String>) -> Result<String>;

    #[tool(description = "Find relevant information in Qdrant")]
    async fn qdrant_find(&self, query: String, collection_name: Option<String>,
                         query_filter: Option<Value>) -> Result<String>;
}
```

- `collection_name` falls back to `config.qdrant.collection_name`, errors if neither set
- Tool descriptions overridable via `config.tools`
- Read-only mode: `qdrant_store` returns error if `config.qdrant.read_only`
- Results formatted as XML: `<entry><content>...</content><metadata>...</metadata></entry>`

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("qdrant: {0}")]
    Qdrant(#[from] qdrant_client::Error),
    #[error("embedding: {0}")]
    Embedding(String),
    #[error("config: {0}")]
    Config(String),
    #[error("no collection name provided and no default configured")]
    NoCollection,
    #[error("store not available in read-only mode")]
    ReadOnly,
}
```
