# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust MCP (Model Context Protocol) server for Qdrant vector database with local embedding support. Dual candle/ONNX embedding backends (feature-flagged).

## Commands

All commands use `just` (task runner). Run `just` to see available tasks.

- **Build:** `just build` (debug), `just build-release` (release)
- **Test:** `just test` — uses `cargo nextest`
- **Test single:** `just test -E 'test(my_test_name)'` (nextest filter expression)
- **Lint:** `just lint` — runs `cargo fmt --check` + `cargo clippy -D warnings`
- **Format:** `just fmt`
- **Run:** `just run`
- **Watch:** `just watch` (uses bacon)

### Feature flags

- `onnx-fetch` (default) — ONNX Runtime with auto-download of runtime binaries
- `onnx` — ONNX Runtime with user-provided `ORT_DYLIB_PATH`
- `candle` — Pure Rust candle-based BERT embedding backend (no native deps)

Build with candle instead of ONNX:
```bash
cargo build --no-default-features --features candle
```

## Architecture

- Single crate at repo root

### Modules

- `config` — CLI parsing (clap) with env var fallbacks, config structs
- `errors` — Error types (thiserror)
- `filters` — Filterable field types for Qdrant queries
- `embeddings` — `EmbeddingProvider` trait + feature-gated backends
  - `embeddings/candle_backend` — candle BERT embedder (default)
  - `embeddings/onnx_backend` — ONNX Runtime embedder
- `qdrant` — Qdrant client connector (placeholder)
- `server` — MCP server implementation (placeholder)

## Code Conventions

- Rust edition 2024
- Clippy: `all + pedantic + nursery + cargo` warnings enabled
- Allowed lints: `implicit_return`, `missing_errors_doc`, `missing_panics_doc`, `missing_safety_doc`, `module_name_repetitions`, `must_use_candidate`, `similar_names`, `too_many_lines`, `cognitive_complexity`, `multiple_crate_versions`, `significant_drop_tightening`
- Modern module syntax: `foo.rs` + `foo/bar.rs` (not `foo/mod.rs`)
- Nix dev environment via `flake.nix` + `direnv`
- `.envrc` auto-loads the dev shell

## Environment Variables

- `QDRANT_URL` — Qdrant server URL (mutually exclusive with `QDRANT_LOCAL_PATH`)
- `QDRANT_API_KEY` — Qdrant API key
- `QDRANT_LOCAL_PATH` — Local Qdrant storage path
- `COLLECTION_NAME` — Default collection name
- `QDRANT_SEARCH_LIMIT` — Max search results (default: 10)
- `QDRANT_READ_ONLY` — Read-only mode (default: false)
- `EMBEDDING_MODEL` — Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
- `MCP_TRANSPORT` — Transport protocol: stdio, sse, streamable-http (default: stdio)
- `HOST` — Bind host for SSE/HTTP (default: 127.0.0.1)
- `PORT` — Bind port for SSE/HTTP (default: 8000)
