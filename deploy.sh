#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building claude-memory..."
cargo build --release

echo "Done. Restart Claude Code to reload the MCP server."
