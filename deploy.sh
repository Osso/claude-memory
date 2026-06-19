#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cargo install --force --path .

echo "Installed claude-memory + claude-memory-mcp to ~/.cargo/bin"
echo "Restart Claude Code to reload the MCP server."
