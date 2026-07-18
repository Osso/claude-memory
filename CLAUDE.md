# claude-memory

Semantic memory search for Claude Code sessions and the knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Interface**: the `claude-memory` CLI binary
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Qdrant state**: only `claude-session-history` remains

## Usage

```bash
claude-memory index
claude-memory search "query"
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory search --limit 10 --json "query"
claude-memory kb-page-index query "query"
claude-memory transcript-page-index build
claude-memory enrich
claude-memory stats
```

Claude Code, Codex, and Pi session shutdown integrations automatically run
`claude-memory index-file <transcript_path>`. Manual `claude-memory index` is
incremental backfill and recovery across Claude active/archive, Codex
active/archive, and Pi session JSONL files. Existing hashes are skipped unless
`--fresh` is supplied. The default search runs one globally ranked prompt+answer
query over `claude-session-history`; `--type prompts|answers` provides optional
filtering and `--limit` applies globally. `--json` emits stable NDJSON fields
`type`, `text`, `source`, `path`, `session_id`, and `score`. UserPromptSubmit
runs `enrich` only to retrieve existing prompt/answer and KB PageIndex context;
it does not index. Transcript PageIndex remains CLI-only navigation and is not
injected by default.
Its query command is deterministic lexical-only; document, structure, and content
remain explicit source-inspection commands.

The memory-unit and graph runtime paths are retired. `deduplicate`,
`build-graph`, `graph-clean`, and `graph-dump` are retired commands. The
`src/memory_unit.rs`, `src/dedup.rs`, `src/graph.rs`, `src/graph/`, and
`src/graph_cmds.rs` runtime modules were deleted.

The canonical durable-memory KB Markdown export completed before the
compatibility code was removed. Its Markdown and manifest remain the editable
KB representation, and migration backups exist. No runtime migration or export
command remains. Qdrant now contains only `claude-session-history`.

## Build & Install

```bash
./deploy.sh
```

`deploy.sh` installs the `claude-memory` CLI binary to `~/.cargo/bin/` with
`cargo install --force --path .`.

The installed interface is the `claude-memory` CLI binary.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
