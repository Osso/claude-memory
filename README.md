# claude-memory

Semantic memory search for Claude Code sessions and the local knowledge base.

## Architecture

- **Session-history vector store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Integration**: MCP server for Claude Code
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Migration/export tools**: guarded legacy migration and completed durable-memory KB export

## Usage

Manual CLI invocation + Claude Code hooks:

```bash
# Index active and archived Claude transcript chunks
claude-memory index

# Search prompt/answer history (--type is required)
claude-memory search --type prompts "query"
claude-memory search --type answers "query"

# Build/query PageIndex surfaces
claude-memory kb-page-index build --kb /syncthing/Sync/KB
claude-memory kb-page-index query "query"
claude-memory transcript-page-index build

# Read and maintain remaining memory surfaces
claude-memory deduplicate
claude-memory enrich
claude-memory stats
```

`index` reads active `.jsonl` sessions and archived `.jsonl.zst` sessions only.
Prompt and answer searches are filtered views over the shared
`claude-session-history` collection. KB Markdown uses the separate KB PageIndex
surface. The former `ingest-kb` path is retired.

The transcript analyzer and its `analyze` and `backfill` CLI commands are
retired. The notable-fact analyzer/writer module is removed; no active
notable-fact writer remains. Existing memory-unit reader, listing, deletion,
deduplication, and enrich paths remain. Enrichment also reads prompt/answer
history and KB PageIndex results.

Project summaries, KB Markdown, manual memories, and the
`claude-memory`, `claude-session-prompts`, and `claude-answers` stores are not
session-history indexing targets or alternate search paths. Legacy
`source=summary` and `source=kb` recognition remains for migration/export
compatibility.

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
eligible prompt/answer history and verifies exact parity. The completed
`claude-memory-export-kb` flow writes canonical durable-memory Markdown and
manifest files under `/syncthing/Sync/KB/memory` and rebuilds KB PageIndex.

Neither migration nor export deletes legacy points or collections. This
retirement does not claim that any collection was deleted.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
