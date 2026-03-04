set dotenv-load

# Default task lists all available tasks
default:
    just --list

# Build the project
[group('build')]
build *args:
    cargo build {{args}}

# Build release
[group('build')]
build-release:
    cargo build --release

# Run tests
[group('test')]
test *args:
    cargo nextest run {{args}}

# Run e2e tests (requires running Qdrant)
[group('test')]
e2e *args:
    cargo nextest run --run-ignored ignored-only {{args}}

# Lint code
[group('lint')]
lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

# Lint GitHub Actions
[group('lint')]
lint-actions:
    actionlint

# Format code
[group('lint')]
fmt:
    cargo fmt --all

# Clean build artifacts
[group('clean')]
clean:
    cargo clean

# Run the project
[group('run')]
run *args:
    cargo run {{args}}

# Watch for changes
[group('run')]
watch:
    bacon
