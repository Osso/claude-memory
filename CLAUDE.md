# claude-memory

Semantic memory search for Claude Code sessions and the knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: default Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions); OpenRouter is an optional alternate profile
- **Interface**: the `claude-memory` CLI binary
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Qdrant state**: the default collection is `claude-session-history`; alternate profiles use their configured collection

## Usage

```bash
claude-memory index
claude-memory search "query"
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory search --session 019fe2f2 "query"
claude-memory search --limit 10 --json "query"
claude-memory kb-page-index query "query"
claude-memory transcript-page-index build
claude-memory enrich "query"
printf '%s\n' '{"prompt":"query"}' | claude-memory enrich
claude-memory stats
```

Claude Code, Codex, and Pi session shutdown integrations automatically run
`claude-memory index-file <transcript_path>`. Manual `claude-memory index` is
incremental backfill and recovery across Claude active/archive, Codex
active/archive, and Pi session JSONL files. Existing hashes are skipped unless
`--fresh` is supplied. The default search runs one globally ranked prompt+answer
query over `claude-session-history`; `--type prompts|answers` provides optional
filtering, `--session <id-substring>` restricts results to matching indexed
session IDs, and `--limit` applies after those filters. `--json` emits stable
NDJSON fields `type`, `text`, `source`, `path`, `session_id`, and `score`.
`enrich` accepts optional prompt text for manual testing; when omitted, it reads
UserPromptSubmit JSON from stdin. It only retrieves existing prompt/answer and
KB PageIndex context; it does not index. Transcript PageIndex remains CLI-only
navigation and is not injected by default.
Its query command is deterministic lexical-only; document, structure, and content
remain explicit source-inspection commands.

The memory-unit and graph runtime paths are retired. `deduplicate`,
`build-graph`, `graph-clean`, and `graph-dump` are retired commands. The
`src/memory_unit.rs`, `src/dedup.rs`, `src/graph.rs`, `src/graph/`, and
`src/graph_cmds.rs` runtime modules were deleted.

The canonical durable-memory KB Markdown export completed before the
compatibility code was removed. Its Markdown and manifest remain the editable
KB representation, and migration backups exist. No runtime migration or export
command remains. The default runtime uses only `claude-session-history`; explicitly configured alternate embedding profiles use their own collection.

## Build & Install

```bash
./deploy.sh
```

`deploy.sh` installs the `claude-memory` CLI binary to `~/.cargo/bin/` with
`cargo install --force --path .`.

The installed interface is the `claude-memory` CLI binary.

## Embedding profiles

The default profile remains Ollama `qwen3-embedding:0.6b-ctx2048` with 1024
dimensions in `claude-session-history`. An OpenRouter profile can select
`qwen/qwen3-embedding-8b`, a positive vector size, a separate collection, and
an optional query instruction using:

- `CLAUDE_MEMORY_EMBEDDING_BACKEND`
- `CLAUDE_MEMORY_EMBEDDING_MODEL`
- `CLAUDE_MEMORY_VECTOR_SIZE`
- `CLAUDE_MEMORY_COLLECTION`
- `CLAUDE_MEMORY_QUERY_INSTRUCTION`

The query instruction applies only to query embeddings. OpenRouter reads
`api_key` only from `~/.config/openrouter/config.toml`. Its document requests
are batched and transient failures are retried. Dense-vector dimension
mismatches fail without replacing a collection; embedding failures stop
indexing.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama (default profile): `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
