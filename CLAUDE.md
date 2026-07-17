# claude-memory

Semantic memory search for Claude Code sessions and the knowledge base.

## Architecture

- **Session-history vector store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Integration**: MCP server for Claude Code
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Migration/export**: guarded legacy migration and completed durable-memory KB export

## Usage

```bash
claude-memory index
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory deduplicate
claude-memory enrich
claude-memory stats
```

`index` reads active `.jsonl` sessions and archived `.jsonl.zst` sessions only.
Prompt and answer searches are filtered views over `claude-session-history`.
KB Markdown uses the separate `kb-page-index` surface. The former `ingest-kb`
command is retired.

The transcript analyzer and `analyze`/`backfill` commands are retired. The
notable-fact analyzer/writer module is removed; no active notable-fact writer
remains. Memory-unit readers, listing/deletion, deduplication, and enrich
retrieval remain active. Prompt/answer history, KB PageIndex, and transcript
PageIndex remain separate retrieval surfaces.

Project summaries, KB Markdown, manual memories, and the
`claude-memory`, `claude-session-prompts`, and `claude-answers` stores are not
session-history indexing targets or alternate search paths. Legacy
`source=summary` and `source=kb` recognition remains for migration/export
compatibility; legacy Qdrant data and collections are not changed by this
retirement.

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
Neither tool deletes source points or collections. No collection deletion is
claimed here.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
