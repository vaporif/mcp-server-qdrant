# MCP Server Qdrant — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port mcp-server-qdrant from Python to Rust with dual candle/ONNX embedding backends.

**Architecture:** Single crate in workspace. rmcp for MCP protocol, qdrant-client for vector DB, candle/ort for embeddings (feature-flagged). Config via clap with env var fallbacks.

**Tech Stack:** rmcp 1.1.0, qdrant-client, candle-core/candle-nn (default), ort/ndarray (optional), tokenizers, hf-hub, clap, tokio, thiserror, tracing, serde, schemars

**Design doc:** `docs/plans/2026-03-04-mcp-server-qdrant-rust-port-design.md`

---

### Task 1: Scaffold Crate and Dependencies

**Files:**
- Delete: `crates/my-crate/` (template placeholder)
- Create: `crates/mcp-server-qdrant/Cargo.toml`
- Create: `crates/mcp-server-qdrant/src/lib.rs`
- Create: `crates/mcp-server-qdrant/src/main.rs`
- Modify: `Cargo.toml` (workspace deps)

**Step 1: Remove template crate**

```bash
rm -rf crates/my-crate
```

**Step 2: Create workspace Cargo.toml with all dependencies**

Update root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
cargo = { level = "warn", priority = -1 }

implicit_return = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
missing_safety_doc = "allow"
module_name_repetitions = "allow"
must_use_candidate = "allow"
similar_names = "allow"
too_many_lines = "allow"
cognitive_complexity = "allow"

[workspace.package]
version = "0.1.0"
edition = "2024"

[workspace.dependencies]
# MCP
rmcp = { version = "1.1", features = ["server", "transport-io", "transport-streamable-http-server"] }

# Qdrant
qdrant-client = "1"

# Embeddings (shared)
tokenizers = { version = "0.22", default-features = false, features = ["onig"] }
hf-hub = { version = "0.5", features = ["tokio"] }

# Embeddings (candle)
candle-core = "0.9"
candle-nn = "0.9"

# Embeddings (onnx)
ort = "2"
ndarray = "0.16"

# Async
tokio = { version = "1", features = ["rt-multi-thread", "macros", "signal"] }

# CLI
clap = { version = "4", features = ["derive", "env"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"

# Error handling
thiserror = "2"
anyhow = "1"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Misc
uuid = { version = "1", features = ["v4"] }
async-trait = "0.1"
```

**Step 3: Create crate Cargo.toml**

Create `crates/mcp-server-qdrant/Cargo.toml`:

```toml
[package]
name = "mcp-server-qdrant"
version.workspace = true
edition.workspace = true

[lints]
workspace = true

[features]
default = ["candle"]
candle = ["dep:candle-core", "dep:candle-nn"]
onnx = ["dep:ort", "dep:ndarray"]
onnx-fetch = ["onnx"]

[dependencies]
rmcp.workspace = true
qdrant-client.workspace = true
tokenizers.workspace = true
hf-hub.workspace = true
tokio.workspace = true
clap.workspace = true
serde.workspace = true
serde_json.workspace = true
schemars.workspace = true
thiserror.workspace = true
anyhow.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
uuid.workspace = true
async-trait.workspace = true

# Candle (default)
candle-core = { workspace = true, optional = true }
candle-nn = { workspace = true, optional = true }

# ONNX (optional)
ort = { workspace = true, optional = true }
ndarray = { workspace = true, optional = true }
```

**Step 4: Create minimal lib.rs and main.rs**

`crates/mcp-server-qdrant/src/lib.rs`:
```rust
pub mod config;
pub mod embeddings;
pub mod errors;
pub mod filters;
pub mod qdrant;
pub mod server;
```

`crates/mcp-server-qdrant/src/main.rs`:
```rust
fn main() {
    println!("mcp-server-qdrant");
}
```

Create stub files for each module (empty `mod` bodies):
- `crates/mcp-server-qdrant/src/config.rs`
- `crates/mcp-server-qdrant/src/embeddings.rs`
- `crates/mcp-server-qdrant/src/errors.rs`
- `crates/mcp-server-qdrant/src/filters.rs`
- `crates/mcp-server-qdrant/src/qdrant.rs`
- `crates/mcp-server-qdrant/src/server.rs`

**Step 5: Verify it compiles**

```bash
just build
```

Expected: Successful build with warnings about unused modules.

**Step 6: Commit**

```bash
git add -A && git commit -m "scaffold mcp-server-qdrant crate with dependencies"
```

---

### Task 2: Error Types

**Files:**
- Modify: `crates/mcp-server-qdrant/src/errors.rs`

**Step 1: Write error types**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("qdrant: {0}")]
    Qdrant(#[from] qdrant_client::QdrantError),

    #[error("embedding: {0}")]
    Embedding(String),

    #[error("config: {0}")]
    Config(String),

    #[error("no collection name provided and no default configured")]
    NoCollection,

    #[error("store not available in read-only mode")]
    ReadOnly,

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("model download: {0}")]
    ModelDownload(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

**Step 2: Verify it compiles**

```bash
just build
```

**Step 3: Commit**

```bash
git add -A && git commit -m "add error types"
```

---

### Task 3: Filter Types

**Files:**
- Modify: `crates/mcp-server-qdrant/src/filters.rs`

**Step 1: Write filter enums and structs**

```rust
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
#[serde(rename_all = "lowercase")]
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
    Any,
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
```

**Step 2: Write tests for serde round-trip**

Add to bottom of `filters.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_type_deserializes_from_lowercase() {
        let json = r#""keyword""#;
        let ft: FieldType = serde_json::from_str(json).unwrap();
        assert_eq!(ft, FieldType::Keyword);
    }

    #[test]
    fn filter_condition_deserializes_symbols() {
        let json = r#""==""#;
        let fc: FilterCondition = serde_json::from_str(json).unwrap();
        assert_eq!(fc, FilterCondition::Eq);

        let json = r#"">=""#;
        let fc: FilterCondition = serde_json::from_str(json).unwrap();
        assert_eq!(fc, FilterCondition::Gte);
    }

    #[test]
    fn filterable_field_round_trip() {
        let field = FilterableField {
            name: "category".into(),
            description: "Item category".into(),
            field_type: FieldType::Keyword,
            condition: Some(FilterCondition::Eq),
            required: false,
        };
        let json = serde_json::to_string(&field).unwrap();
        let back: FilterableField = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "category");
        assert_eq!(back.field_type, FieldType::Keyword);
    }
}
```

**Step 3: Run tests**

```bash
just test -E 'test(filters)'
```

Expected: 3 tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "add filter types with serde support"
```

---

### Task 4: Config System

**Files:**
- Modify: `crates/mcp-server-qdrant/src/config.rs`

**Step 1: Write config structs with clap derive**

```rust
use std::net::IpAddr;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::Deserialize;

use crate::filters::FilterableField;

#[derive(Parser, Debug)]
#[command(name = "mcp-server-qdrant", about = "MCP server for Qdrant vector database")]
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
    #[arg(long, default_value = "sentence-transformers/all-MiniLM-L6-v2", env = "EMBEDDING_MODEL")]
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
    Remote { url: String, api_key: Option<String> },
    Local { path: PathBuf },
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

        Ok(Config {
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
```

**Step 2: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_cli_remote() {
        let cli = Cli::parse_from([
            "mcp-server-qdrant",
            "--qdrant-url", "http://localhost:6334",
            "--qdrant-api-key", "test-key",
        ]);
        let config = Config::from_cli(cli).unwrap();
        assert!(matches!(config.qdrant.location, QdrantLocation::Remote { .. }));
    }

    #[test]
    fn config_from_cli_local() {
        let cli = Cli::parse_from([
            "mcp-server-qdrant",
            "--qdrant-local-path", "/tmp/qdrant",
        ]);
        let config = Config::from_cli(cli).unwrap();
        assert!(matches!(config.qdrant.location, QdrantLocation::Local { .. }));
    }

    #[test]
    fn config_rejects_both_url_and_path() {
        let cli = Cli::parse_from([
            "mcp-server-qdrant",
            "--qdrant-url", "http://localhost:6334",
            "--qdrant-local-path", "/tmp/qdrant",
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
            "--qdrant-url", "http://localhost:6334",
            "--filterable-fields",
            r#"[{"name":"category","description":"Category","field_type":"keyword","required":false}]"#,
        ]);
        let config = Config::from_cli(cli).unwrap();
        assert_eq!(config.qdrant.filterable_fields.len(), 1);
        assert_eq!(config.qdrant.filterable_fields[0].name, "category");
    }
}
```

**Step 3: Run tests**

```bash
just test -E 'test(config)'
```

Expected: 5 tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "add config system with clap + env var support"
```

---

### Task 5: Embedding Provider Trait and Candle Backend

**Files:**
- Modify: `crates/mcp-server-qdrant/src/embeddings.rs`

This is the most complex task. The embeddings module has:
1. The `EmbeddingProvider` trait
2. The Candle backend (default feature)
3. The ONNX backend (optional feature)
4. A factory function

**Step 1: Write the trait and factory**

```rust
use async_trait::async_trait;

use crate::errors::{Error, Result};

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}

#[cfg(feature = "candle")]
mod candle_backend;

#[cfg(any(feature = "onnx", feature = "onnx-fetch"))]
mod onnx_backend;

pub fn create_embedding_provider(model_name: &str) -> Result<Box<dyn EmbeddingProvider>> {
    #[cfg(feature = "candle")]
    {
        return Ok(Box::new(candle_backend::CandleEmbedder::new(model_name)?));
    }

    #[cfg(any(feature = "onnx", feature = "onnx-fetch"))]
    {
        return Ok(Box::new(onnx_backend::OnnxEmbedder::new(model_name)?));
    }

    #[allow(unreachable_code)]
    Err(Error::Embedding("no embedding backend enabled — compile with 'candle' or 'onnx' feature".into()))
}
```

**Step 2: Write the Candle backend**

Create `crates/mcp-server-qdrant/src/embeddings/candle_backend.rs`:

```rust
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;

use crate::errors::{Error, Result};
use super::EmbeddingProvider;

pub struct CandleEmbedder {
    model: candle_transformers::models::bert::BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
}

impl CandleEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let device = Device::Cpu;
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let repo = api.model(model_name.to_string());

        let config_path = repo.get("config.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let weights_path = repo.get("model.safetensors")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
        let config: candle_transformers::models::bert::Config = serde_json::from_str(&config_str)
            .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;

        let dimension = config.hidden_size;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
                .map_err(|e| Error::Embedding(format!("load weights: {e}")))?
        };

        let model = candle_transformers::models::bert::BertModel::load(vb, &config)
            .map_err(|e| Error::Embedding(format!("load model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Self { model, tokenizer, device, dimension })
    }

    fn embed_sync(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let token_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
        let attention_masks: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect();
        let type_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_type_ids().to_vec()).collect();

        let batch_size = texts.len();
        let seq_len = token_ids[0].len();

        let token_ids_flat: Vec<u32> = token_ids.into_iter().flatten().collect();
        let attention_flat: Vec<u32> = attention_masks.into_iter().flatten().collect();
        let type_ids_flat: Vec<u32> = type_ids.into_iter().flatten().collect();

        let token_ids_t = Tensor::from_vec(token_ids_flat, (batch_size, seq_len), &self.device)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let attention_t = Tensor::from_vec(attention_flat, (batch_size, seq_len), &self.device)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let type_ids_t = Tensor::from_vec(type_ids_flat, (batch_size, seq_len), &self.device)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let output = self.model.forward(&token_ids_t, &type_ids_t, Some(&attention_t))
            .map_err(|e| Error::Embedding(format!("forward pass: {e}")))?;

        // Mean pooling with attention mask
        let attention_f = attention_t.to_dtype(candle_core::DType::F32)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let attention_expanded = attention_f.unsqueeze(2)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let masked = output.broadcast_mul(&attention_expanded)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let summed = masked.sum(1)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let mask_sum = attention_f.sum(1)
            .map_err(|e| Error::Embedding(e.to_string()))?
            .unsqueeze(1)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let pooled = summed.broadcast_div(&mask_sum)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        // L2 normalize
        let norms = pooled.sqr()
            .map_err(|e| Error::Embedding(e.to_string()))?
            .sum(1)
            .map_err(|e| Error::Embedding(e.to_string()))?
            .sqrt()
            .map_err(|e| Error::Embedding(e.to_string()))?
            .unsqueeze(1)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let normalized = pooled.broadcast_div(&norms)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let vecs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                normalized.get(i)
                    .and_then(|row| row.to_vec1())
                    .map_err(|e| Error::Embedding(e.to_string()))
            })
            .collect::<Result<_>>()?;

        Ok(vecs)
    }
}

#[async_trait]
impl EmbeddingProvider for CandleEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut results = self.embed_sync(&texts)?;
        results.pop().ok_or_else(|| Error::Embedding("empty result".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_sync(texts)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
```

**Step 3: Restructure embeddings as a directory module**

Since we now have submodules, restructure:
- `crates/mcp-server-qdrant/src/embeddings.rs` → `crates/mcp-server-qdrant/src/embeddings/mod.rs`

Wait — the design says modern module syntax (`foo.rs` + `foo/bar.rs`). So:
- Keep `crates/mcp-server-qdrant/src/embeddings.rs` as the parent
- Create `crates/mcp-server-qdrant/src/embeddings/candle_backend.rs`
- Create `crates/mcp-server-qdrant/src/embeddings/onnx_backend.rs`

**Step 4: Verify it compiles**

```bash
just build
```

**Step 5: Commit**

```bash
git add -A && git commit -m "add embedding provider trait and candle backend"
```

---

### Task 6: ONNX Embedding Backend

**Files:**
- Create: `crates/mcp-server-qdrant/src/embeddings/onnx_backend.rs`

**Step 1: Write the ONNX backend**

```rust
use async_trait::async_trait;
use ndarray::{Array2, CowArray};
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::errors::{Error, Result};
use super::EmbeddingProvider;

pub struct OnnxEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    dimension: usize,
}

impl OnnxEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let repo = api.model(model_name.to_string());

        let onnx_path = repo.get("onnx/model.onnx")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;
        let config_path = repo.get("config.json")
            .map_err(|e| Error::ModelDownload(e.to_string()))?;

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::Embedding(format!("read config: {e}")))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| Error::Embedding(format!("parse config: {e}")))?;
        let dimension = config["hidden_size"].as_u64()
            .ok_or_else(|| Error::Embedding("missing hidden_size in config".into()))? as usize;

        let session = Session::builder()
            .map_err(|e| Error::Embedding(format!("ort session builder: {e}")))?
            .commit_from_file(&onnx_path)
            .map_err(|e| Error::Embedding(format!("load onnx model: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer, dimension })
    }

    fn embed_sync(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self.tokenizer
            .encode_batch(texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let batch_size = texts.len();
        let seq_len = encodings[0].get_ids().len();

        let input_ids: Vec<i64> = encodings.iter()
            .flat_map(|e| e.get_ids().iter().map(|&id| id as i64))
            .collect();
        let attention_mask: Vec<i64> = encodings.iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&m| m as i64))
            .collect();
        let token_type_ids: Vec<i64> = encodings.iter()
            .flat_map(|e| e.get_type_ids().iter().map(|&t| t as i64))
            .collect();

        let input_ids_arr = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let attention_arr = Array2::from_shape_vec((batch_size, seq_len), attention_mask)
            .map_err(|e| Error::Embedding(e.to_string()))?;
        let type_ids_arr = Array2::from_shape_vec((batch_size, seq_len), token_type_ids)
            .map_err(|e| Error::Embedding(e.to_string()))?;

        let outputs = self.session.run(
            ort::inputs![
                "input_ids" => input_ids_arr,
                "attention_mask" => attention_arr,
                "token_type_ids" => type_ids_arr,
            ].map_err(|e| Error::Embedding(format!("ort inputs: {e}")))?
        ).map_err(|e| Error::Embedding(format!("ort run: {e}")))?;

        // Output shape: (batch_size, seq_len, hidden_size)
        let output_tensor = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| Error::Embedding(format!("extract tensor: {e}")))?;
        let output_view = output_tensor.view();

        // Mean pooling with attention mask
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut embedding = vec![0.0f32; self.dimension];
            let mut mask_sum = 0.0f32;
            for j in 0..seq_len {
                let mask_val = encodings[i].get_attention_mask()[j] as f32;
                mask_sum += mask_val;
                for k in 0..self.dimension {
                    embedding[k] += output_view[[i, j, k]] * mask_val;
                }
            }
            for val in &mut embedding {
                *val /= mask_sum;
            }

            // L2 normalize
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }

            results.push(embedding);
        }

        Ok(results)
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let mut results = self.embed_sync(&texts)?;
        results.pop().ok_or_else(|| Error::Embedding("empty result".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_sync(texts)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
```

**Step 2: Verify it compiles with both features**

```bash
just build
cargo build --no-default-features --features onnx
```

**Step 3: Commit**

```bash
git add -A && git commit -m "add ONNX embedding backend"
```

---

### Task 7: Qdrant Connector

**Files:**
- Modify: `crates/mcp-server-qdrant/src/qdrant.rs`

**Step 1: Write the QdrantConnector**

```rust
use std::collections::HashMap;
use std::sync::Arc;

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, Filter, PointStruct,
    QueryPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    CreateFieldIndexCollectionBuilder, FieldType as QdrantFieldType,
    value::Kind, Value as QdrantValue,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::{QdrantConfig, QdrantLocation};
use crate::embeddings::EmbeddingProvider;
use crate::errors::{Error, Result};
use crate::filters::{FieldType, FilterableField};

pub type Metadata = HashMap<String, serde_json::Value>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

pub struct QdrantConnector {
    client: Qdrant,
    embedding: Arc<dyn EmbeddingProvider>,
    indexes: Vec<FilterableField>,
}

impl QdrantConnector {
    pub fn new(
        config: &QdrantConfig,
        embedding: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        let client = match &config.location {
            QdrantLocation::Remote { url, api_key } => {
                let mut builder = Qdrant::from_url(url);
                if let Some(key) = api_key {
                    builder = builder.api_key(key);
                }
                builder.build().map_err(|e| Error::Qdrant(e.into()))?
            }
            QdrantLocation::Local { path } => {
                // For local mode, qdrant-client needs a path-based setup
                // The Rust client uses gRPC so local mode means connecting to localhost
                // or using an embedded Qdrant instance
                Qdrant::from_url(&format!("http://localhost:6334"))
                    .build()
                    .map_err(|e| Error::Qdrant(e.into()))?
            }
        };

        Ok(Self {
            client,
            embedding,
            indexes: config.filterable_fields.clone(),
        })
    }

    pub async fn ensure_collection(&self, collection_name: &str) -> Result<()> {
        let exists = self.client
            .collection_exists(collection_name)
            .await
            .map_err(Error::Qdrant)?;

        if exists {
            return Ok(());
        }

        let dim = self.embedding.dimension() as u64;
        self.client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
            )
            .await
            .map_err(Error::Qdrant)?;

        // Create indexes for filterable fields
        for field in &self.indexes {
            let qdrant_type = match field.field_type {
                FieldType::Keyword => QdrantFieldType::Keyword,
                FieldType::Integer => QdrantFieldType::Integer,
                FieldType::Float => QdrantFieldType::Float,
                FieldType::Boolean => QdrantFieldType::Bool,
            };

            self.client
                .create_field_index(
                    CreateFieldIndexCollectionBuilder::new(
                        collection_name,
                        &field.name,
                        qdrant_type,
                    ),
                )
                .await
                .map_err(Error::Qdrant)?;
        }

        Ok(())
    }

    pub async fn store(&self, entry: Entry, collection_name: &str) -> Result<()> {
        self.ensure_collection(collection_name).await?;

        let vector = self.embedding.embed(&entry.content).await?;
        let point_id = Uuid::new_v4().to_string();

        let mut payload = qdrant_client::Payload::new();
        payload.insert("document", entry.content.clone());
        if let Some(metadata) = &entry.metadata {
            for (key, value) in metadata {
                payload.insert(
                    key.as_str(),
                    serde_json::to_string(value)
                        .unwrap_or_default(),
                );
            }
        }

        let point = PointStruct::new(point_id, vector, payload);

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection_name, vec![point]).wait(true))
            .await
            .map_err(Error::Qdrant)?;

        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        collection_name: &str,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<Entry>> {
        self.ensure_collection(collection_name).await?;

        let vector = self.embedding.embed(query).await?;

        let mut builder = QueryPointsBuilder::new(collection_name)
            .query(vector)
            .limit(limit as u64)
            .with_payload(true);

        if let Some(f) = filter {
            builder = builder.filter(f);
        }

        let results = self.client.query(builder).await.map_err(Error::Qdrant)?;

        let entries: Vec<Entry> = results
            .result
            .into_iter()
            .filter_map(|point| {
                let payload = point.payload;
                let content = payload.get("document")
                    .and_then(|v| match &v.kind {
                        Some(Kind::StringValue(s)) => Some(s.clone()),
                        _ => None,
                    })?;

                let metadata: Metadata = payload.iter()
                    .filter(|(k, _)| k.as_str() != "document")
                    .filter_map(|(k, v)| {
                        match &v.kind {
                            Some(Kind::StringValue(s)) => {
                                serde_json::from_str(s).ok()
                                    .map(|parsed| (k.clone(), parsed))
                            }
                            _ => None,
                        }
                    })
                    .collect();

                Some(Entry {
                    content,
                    metadata: if metadata.is_empty() { None } else { Some(metadata) },
                })
            })
            .collect();

        Ok(entries)
    }
}
```

**Step 2: Verify it compiles**

```bash
just build
```

**Step 3: Commit**

```bash
git add -A && git commit -m "add Qdrant connector with store and search"
```

---

### Task 8: MCP Server with rmcp Tools

**Files:**
- Modify: `crates/mcp-server-qdrant/src/server.rs`

**Step 1: Write the MCP server**

```rust
use std::sync::Arc;

use rmcp::{
    ServerHandler,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    ErrorData as McpError,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::Config;
use crate::qdrant::{Entry, QdrantConnector};

#[derive(Clone)]
pub struct QdrantMcpServer {
    connector: Arc<QdrantConnector>,
    config: Arc<Config>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreParams {
    /// The information/text content to store
    pub information: String,
    /// Optional metadata as JSON object
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Collection name (uses default if not provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindParams {
    /// Search query
    pub query: String,
    /// Collection name (uses default if not provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_name: Option<String>,
    /// Optional Qdrant filter as JSON
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_filter: Option<serde_json::Value>,
}

impl QdrantMcpServer {
    pub fn new(connector: Arc<QdrantConnector>, config: Arc<Config>) -> Self {
        Self {
            connector,
            config,
            tool_router: Self::tool_router(),
        }
    }

    fn resolve_collection(&self, provided: &Option<String>) -> std::result::Result<String, McpError> {
        provided
            .clone()
            .or_else(|| self.config.qdrant.collection_name.clone())
            .ok_or_else(|| McpError::invalid_params(
                "no collection name provided and no default configured", None,
            ))
    }

    fn format_entries(entries: &[Entry]) -> String {
        entries
            .iter()
            .map(|entry| {
                let metadata_xml = entry.metadata.as_ref().map_or(String::new(), |m| {
                    format!("<metadata>{}</metadata>", serde_json::to_string(m).unwrap_or_default())
                });
                format!("<entry><content>{}</content>{}</entry>", entry.content, metadata_xml)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[tool_router]
impl QdrantMcpServer {
    #[tool(name = "qdrant-store", description = "Store information in Qdrant vector database")]
    async fn qdrant_store(
        &self,
        Parameters(params): Parameters<StoreParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        if self.config.qdrant.read_only {
            return Err(McpError::invalid_request(
                "store not available in read-only mode", None,
            ));
        }

        let collection = self.resolve_collection(&params.collection_name)?;

        let metadata = params.metadata.and_then(|v| {
            serde_json::from_value(v).ok()
        });

        let entry = Entry {
            content: params.information,
            metadata,
        };

        self.connector
            .store(entry, &collection)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(
            format!("Successfully stored in collection '{collection}'"),
        )]))
    }

    #[tool(name = "qdrant-find", description = "Find relevant information in Qdrant using semantic search")]
    async fn qdrant_find(
        &self,
        Parameters(params): Parameters<FindParams>,
    ) -> std::result::Result<CallToolResult, McpError> {
        let collection = self.resolve_collection(&params.collection_name)?;

        let filter = if let Some(filter_json) = params.query_filter {
            // Convert JSON filter to Qdrant filter
            // For now, support arbitrary filter passthrough
            let filter_str = serde_json::to_string(&filter_json)
                .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
            serde_json::from_str(&filter_str).ok()
        } else {
            None
        };

        let entries = self.connector
            .search(&params.query, &collection, self.config.qdrant.search_limit, filter)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        if entries.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found".to_string(),
            )]));
        }

        let formatted = Self::format_entries(&entries);
        Ok(CallToolResult::success(vec![Content::text(formatted)]))
    }
}

#[tool_handler]
impl ServerHandler for QdrantMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder().enable_tools().build(),
        )
        .with_instructions("Qdrant MCP server for semantic memory storage and retrieval".to_string())
    }
}
```

**Step 2: Verify it compiles**

```bash
just build
```

**Step 3: Commit**

```bash
git add -A && git commit -m "add MCP server with qdrant-store and qdrant-find tools"
```

---

### Task 9: Main Entry Point with Transport Selection

**Files:**
- Modify: `crates/mcp-server-qdrant/src/main.rs`

**Step 1: Write main.rs**

```rust
use std::sync::Arc;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use mcp_server_qdrant::config::{Cli, Config, Transport};
use mcp_server_qdrant::embeddings::create_embedding_provider;
use mcp_server_qdrant::qdrant::QdrantConnector;
use mcp_server_qdrant::server::QdrantMcpServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let config = Config::from_cli(cli)?;

    let embedding = create_embedding_provider(&config.embedding.model_name)?;
    let embedding = Arc::from(embedding);

    let connector = QdrantConnector::new(&config.qdrant, embedding)?;
    let connector = Arc::new(connector);

    let config = Arc::new(config);
    let server = QdrantMcpServer::new(connector, config.clone());

    match &config.transport {
        Transport::Stdio => {
            use rmcp::{ServiceExt, transport::stdio};
            tracing::info!("starting MCP server on stdio");
            let service = server.serve(stdio()).await?;
            service.waiting().await?;
        }
        Transport::Sse { host, port } | Transport::StreamableHttp { host, port } => {
            use rmcp::transport::streamable_http_server::{
                StreamableHttpService,
                session::local::LocalSessionManager,
            };
            use axum::Router;

            let addr = format!("{host}:{port}");
            tracing::info!("starting MCP server on {addr}");

            let connector = connector.clone();
            let config_clone = config.clone();
            let service = StreamableHttpService::new(
                move || Ok(QdrantMcpServer::new(connector.clone(), config_clone.clone())),
                LocalSessionManager::default().into(),
                Default::default(),
            );

            let router = Router::new().nest_service("/mcp", service);
            let listener = tokio::net::TcpListener::bind(&addr).await?;

            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    tokio::signal::ctrl_c().await.ok();
                })
                .await?;
        }
    }

    Ok(())
}
```

**Step 2: Add axum dependency if needed**

Check if rmcp's `transport-streamable-http-server` feature already brings in axum. If not, add to workspace deps:

```toml
axum = "0.8"
```

**Step 3: Verify it compiles**

```bash
just build
```

**Step 4: Commit**

```bash
git add -A && git commit -m "add main entry point with stdio and HTTP transport"
```

---

### Task 10: Integration Testing

**Files:**
- Create: `crates/mcp-server-qdrant/tests/config_test.rs`
- Create: `crates/mcp-server-qdrant/tests/filters_test.rs`

Integration tests that don't require a running Qdrant instance or HF model download.

**Step 1: Config integration tests**

`crates/mcp-server-qdrant/tests/config_test.rs`:

```rust
use mcp_server_qdrant::config::{Cli, Config, QdrantLocation};
use clap::Parser;

#[test]
fn cli_parses_minimal_remote() {
    let cli = Cli::parse_from([
        "mcp-server-qdrant",
        "--qdrant-url", "http://localhost:6334",
    ]);
    let config = Config::from_cli(cli).unwrap();
    assert!(matches!(config.qdrant.location, QdrantLocation::Remote { .. }));
    assert_eq!(config.qdrant.search_limit, 10);
    assert!(!config.qdrant.read_only);
}

#[test]
fn cli_parses_all_options() {
    let cli = Cli::parse_from([
        "mcp-server-qdrant",
        "--qdrant-url", "http://localhost:6334",
        "--qdrant-api-key", "my-key",
        "--collection-name", "test-collection",
        "--search-limit", "5",
        "--read-only",
        "--embedding-model", "my-model",
        "--tool-store-description", "Custom store",
        "--tool-find-description", "Custom find",
        "--transport", "sse",
        "--host", "0.0.0.0",
        "--port", "9000",
    ]);
    let config = Config::from_cli(cli).unwrap();
    assert_eq!(config.qdrant.search_limit, 5);
    assert!(config.qdrant.read_only);
    assert_eq!(config.qdrant.collection_name.as_deref(), Some("test-collection"));
    assert_eq!(config.embedding.model_name, "my-model");
}
```

**Step 2: Filter integration tests**

`crates/mcp-server-qdrant/tests/filters_test.rs`:

```rust
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
```

**Step 3: Run tests**

```bash
just test
```

Expected: All tests pass.

**Step 4: Commit**

```bash
git add -A && git commit -m "add integration tests for config and filters"
```

---

### Task 11: Lint and Final Cleanup

**Step 1: Format**

```bash
just fmt
```

**Step 2: Lint**

```bash
just lint
```

Fix any clippy warnings.

**Step 3: Final build verification**

```bash
just build
just build --no-default-features --features onnx
```

**Step 4: Commit any cleanup**

```bash
git add -A && git commit -m "lint and clippy fixes"
```

---

### Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Update CLAUDE.md to reflect the actual crate name and any new commands.

**Step 1: Update**

Add the crate name change from `my-crate` to `mcp-server-qdrant` and note the feature flags.

**Step 2: Commit**

```bash
git add CLAUDE.md && git commit -m "update CLAUDE.md for mcp-server-qdrant"
```
