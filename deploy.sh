#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cargo install --force --path .
rm -f "$HOME/.cargo/bin/claude-memory-mcp"

echo "Installed claude-memory to ~/.cargo/bin"
echo "Removed retired claude-memory-mcp executable"
