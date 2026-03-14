# mcp-server-qdrant

[![Check](https://github.com/vaporif/mcp-server-qdrant/actions/workflows/check.yml/badge.svg)](https://github.com/vaporif/mcp-server-qdrant/actions/workflows/check.yml)

Rust MCP server for Qdrant with local BERT embeddings. Single binary, no Python.

## Why not the [official Python one](https://github.com/qdrant/mcp-server-qdrant)?

ONNX Python wheels are painful to package in Nix (especially aarch64-linux). This is a Rust rewrite with Nix-native packaging and a pure-Rust default backend (Candle) that has zero native dependencies.

## Usage

### Claude Desktop / Claude Code

**With [uvx](https://docs.astral.sh/uv/) (recommended):**

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant-rs"],
      "env": {
        "QDRANT_URL": "http://localhost:6334",
        "COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

**With [rvx](https://github.com/vaporif/rvx):**

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "rvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6334",
        "COLLECTION_NAME": "my-collection"
      }
    }
  }
}
```

<details>
<summary>Other installation methods</summary>

**With Nix:**

```sh
nix run github:vaporif/mcp-server-qdrant

# ONNX backend
nix run github:vaporif/mcp-server-qdrant#onnx
```

As a flake input:

```nix
{
  inputs.mcp-server-qdrant.url = "github:vaporif/mcp-server-qdrant";
  nixpkgs.overlays = [ mcp-server-qdrant.overlays.default ];
}
```

**With cargo:**

```sh
cargo install mcp-server-qdrant
```

**From releases:**

Download a prebuilt binary from [GitHub Releases](https://github.com/vaporif/mcp-server-qdrant/releases).

**With Docker:**

```sh
docker build -t mcp-server-qdrant .
docker run -p 8000:8000 -e QDRANT_URL=http://host.docker.internal:6334 -e COLLECTION_NAME=my-collection mcp-server-qdrant
```

</details>

### HTTP Transport

```sh
mcp-server-qdrant --transport streamable-http --port 8000
```

### Debugging

```sh
RUST_LOG=debug mcp-server-qdrant
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

```sh
nix develop    # dev shell
just check     # clippy + test + fmt + taplo + typos
just test      # run tests
just lint      # clippy + fmt
just deny      # dependency audit
just e2e       # e2e tests (needs Qdrant)
```

## License

MIT
