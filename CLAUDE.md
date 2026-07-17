# claude-memory

Semantic memory search for Claude Code sessions and knowledge base.

## Architecture

- **Session-history vector store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Integration**: MCP server for Claude Code
- **Migration CLI**: `claude-memory-migrate` for guarded legacy Qdrant storage migration

## Usage

Manual CLI invocation + Claude Code hooks:

```bash
# Index active and archived Claude transcript chunks
claude-memory index

# Search
claude-memory search "query"
claude-memory search --type prompts "query"
claude-memory search --type answers "query"

# Stats
claude-memory stats
```

`index` reads active `.jsonl` sessions and archived `.jsonl.zst` sessions only.
Prompt and answer searches are filtered views over the shared
`claude-session-history` collection. There is no `index --kb` command; KB
Markdown uses the separate `kb-page-index` surface. The former `ingest-kb`
command is retired, so KB facts
are not actively duplicated into memory units.

Project summaries, KB Markdown, manual memories, and the
`claude-memory`, `claude-session-prompts`, and `claude-answers` stores are not
session-history indexing targets or alternate search paths. Legacy
`source=summary` and `source=kb` recognition remains for export/migration parity;
legacy Qdrant data and collections are not changed by this retirement.

## Build & Install

```bash
./deploy.sh
```

`deploy.sh` installs all four binaries to `~/.cargo/bin/` with `cargo install --force --path .`.

No systemd service — the MCP server runs as a stdio child process of Claude Code. After rebuilding, restart Claude Code to reload it.

The separate `claude-memory-migrate` binary exposes read-only `plan` and `verify` commands plus guarded `apply --backup-dir <directory>`. It backs up all four legacy collections before creating `claude-session-history`, preserves eligible history points, and verifies exact parity. No deletion or fallback command exists; this documents the contract and does not claim a live migration has been run.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
