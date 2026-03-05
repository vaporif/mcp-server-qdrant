# mcp-server-qdrant

[![Check](https://github.com/vaporif/mcp-server-qdrant/actions/workflows/check.yml/badge.svg)](https://github.com/vaporif/mcp-server-qdrant/actions/workflows/check.yml)

Rust MCP server for Qdrant with local BERT embeddings. Single binary, no Python.

## Why not the [official Python one](https://github.com/qdrant/mcp-server-qdrant)?

ONNX Python wheels are painful to package in Nix (especially aarch64-linux). This is a Rust rewrite with Nix-native packaging and a pure-Rust default backend (Candle) that has zero native dependencies.

## Install

```bash
cargo install --path .
```

```bash
nix run github:vaporif/mcp-server-qdrant

# ONNX backend
nix run github:vaporif/mcp-server-qdrant#onnx
```

As a flake input:

```nix
{
  inputs.mcp-server-qdrant.url = "github:vaporif/mcp-server-qdrant";

  # use the overlay
  nixpkgs.overlays = [ mcp-server-qdrant.overlays.default ];
}
```

## Usage

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "mcp-server-qdrant",
      "env": {
        "QDRANT_URL": "http://localhost:6334",
        "COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | — | Qdrant server URL |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `QDRANT_LOCAL_PATH` | — | Local storage path (instead of URL) |
| `COLLECTION_NAME` | — | Default collection name |
| `QDRANT_SEARCH_LIMIT` | 10 | Max search results |
| `QDRANT_READ_ONLY` | false | Read-only mode |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Model name |
| `MCP_TRANSPORT` | stdio | `stdio`, `sse`, `streamable-http` |
| `HOST` | 127.0.0.1 | Bind host for SSE/HTTP |
| `PORT` | 8000 | Bind port for SSE/HTTP |

## Embedding Backends

| Feature | Description |
|---------|-------------|
| `candle` (default) | Pure Rust. No native deps. |
| `onnx` | ONNX Runtime. Provide `ORT_DYLIB_PATH`. |
| `onnx-fetch` | ONNX with auto-download. |

## Development

```bash
nix develop    # dev shell
just test      # run tests
just lint      # clippy + fmt
just e2e       # e2e tests (needs Qdrant)
```

## License

MIT
