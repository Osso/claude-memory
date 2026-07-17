# claude-memory

Semantic memory search for Claude Code sessions and the local knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Integration**: MCP server for Claude Code
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Migration/export**: guarded legacy migration and durable-memory KB export

## Usage

Manual CLI invocation + Claude Code hooks:

```bash
# Index active and archived Claude transcript chunks
claude-memory index

# Search unified prompt/answer history (--type is required)
claude-memory search --type prompts "query"
claude-memory search --type answers "query"

# Build/query PageIndex surfaces
claude-memory kb-page-index build --kb /syncthing/Sync/KB
claude-memory kb-page-index query "query"
claude-memory transcript-page-index build

# Enrich a prompt and show collection statistics
claude-memory enrich
claude-memory stats
```

`index` reads active `.jsonl` sessions and archived `.jsonl.zst` sessions only.
Prompt and answer searches are filtered views over the shared
`claude-session-history` collection. `enrich` uses those unified prompt/answer
history results plus KB PageIndex results only. Transcript PageIndex remains a
separate CLI navigation surface and is not injected by default.

The former memory-unit and graph runtime paths are retired. The
`deduplicate`, `build-graph`, `graph-clean`, and `graph-dump` commands are no
longer public commands. The `src/memory_unit.rs`, `src/dedup.rs`, `src/graph.rs`,
`src/graph/`, and `src/graph_cmds.rs` runtime modules were removed.

Legacy collection readers remain only where migration or KB export compatibility
requires them. Legacy `source=summary` and `source=kb` recognition remains for
classification and parity. This change does not claim that any Qdrant
collection or point was deleted.

## Build & Install

```bash
./deploy.sh
```

`deploy.sh` installs the remaining binaries to `~/.cargo/bin/` with
`cargo install --force --path .`.

No systemd service — the MCP server runs as a stdio child process of Claude
Code. After rebuilding, restart Claude Code to reload it.

The separate `claude-memory-migrate` binary exposes read-only `plan` and
`verify` commands plus guarded `apply --backup-dir <directory>`. It preserves
eligible prompt/answer history and verifies exact parity. The
`claude-memory-export-kb` flow writes canonical durable-memory Markdown and
manifest files under `/syncthing/Sync/KB/memory` and rebuilds KB PageIndex.
Neither migration nor export deletes source points or collections.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
