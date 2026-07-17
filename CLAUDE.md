# claude-memory

Semantic memory search for Claude Code sessions and the knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Integration**: MCP server for Claude Code
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Migration/export**: guarded legacy migration and durable-memory KB export

## Usage

```bash
claude-memory index
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory kb-page-index query "query"
claude-memory transcript-page-index build
claude-memory enrich
claude-memory stats
```

`index` reads active `.jsonl` sessions and archived `.jsonl.zst` sessions only.
Prompt and answer searches are filtered views over `claude-session-history`.
`enrich` reads unified prompt/answer history and KB PageIndex results only.
Transcript PageIndex remains CLI-only navigation and is not injected by default.

The memory-unit and graph runtime paths are retired. `deduplicate`,
`build-graph`, `graph-clean`, and `graph-dump` are retired commands. The
`src/memory_unit.rs`, `src/dedup.rs`, `src/graph.rs`, `src/graph/`, and
`src/graph_cmds.rs` runtime modules were deleted.

Migration/export compatibility readers still recognize legacy collection data
and legacy `source=summary` / `source=kb` values where required for parity.
This retirement does not claim deletion of any Qdrant collection or point.

## Build & Install

```bash
./deploy.sh
```

`deploy.sh` installs the remaining binaries to `~/.cargo/bin/` with
`cargo install --force --path .`.

No systemd service — the MCP server runs as a stdio child process of Claude
Code. After rebuilding, restart Claude Code to reload it.

`claude-memory-migrate` provides read-only `plan`/`verify` and guarded
`apply --backup-dir <directory>` for legacy history migration. The completed
`claude-memory-export-kb` flow writes canonical durable-memory Markdown and a
manifest under `/syncthing/Sync/KB/memory`, then rebuilds KB PageIndex.
Neither tool deletes source points or collections.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
