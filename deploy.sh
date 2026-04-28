#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cargo install --force --path .

echo "Installed to ~/.cargo/bin/claude-memory"
echo "Restart Claude Code to reload the MCP server."
