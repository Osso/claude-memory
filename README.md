# claude-memory

Semantic memory search for Claude Code sessions and the local knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: production OpenRouter `qwen/qwen3-embedding-8b` (4096 dimensions) in `claude-session-history-qwen3-8b`; the intact Ollama `claude-session-history` collection remains the rollback profile
- **Interface**: the `claude-memory` CLI binary
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Qdrant state**: production uses `claude-session-history-qwen3-8b`; `claude-session-history` remains intact for rollback

## Usage

Manual CLI invocation plus Claude Code, Codex, and Pi lifecycle integration:

```bash
# Incrementally index missing Claude, Codex, and Pi transcript chunks
claude-memory index

# Search globally ranked prompt+answer history
claude-memory search "query"
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory search --session 019fe2f2 "query"
claude-memory search --limit 10 --json "query"

# Query KB text index; missing or stale indexes rebuild automatically
claude-memory kb-page-index query "query" --kb /syncthing/Sync/KB
# Optional explicit prewarm/diagnostic build
claude-memory kb-page-index build --kb /syncthing/Sync/KB

# Build transcript PageIndex from Claude and Codex sessions
claude-memory transcript-page-index build
claude-memory transcript-page-index query "query"

# Enrich a prompt manually, or omit it for JSON hook input on stdin
claude-memory enrich "query"
printf '%s\n' '{"prompt":"query"}' | claude-memory enrich
claude-memory stats
```

Claude Code, Codex, and Pi session shutdown integrations automatically run
`claude-memory index-file <transcript_path>`. Manual `claude-memory index` is
incremental backfill and recovery across Claude active/archive, Codex
active/archive, and Pi session JSONL files. Existing hashes are skipped unless
`--fresh` is supplied. Search runs one globally ranked prompt+answer
query over the configured session-history collection; `--type
prompts|answers` provides optional filtering, `--session <id-substring>`
restricts the ranked query to matching indexed session IDs, and `--limit`
applies after those filters. `--json` emits stable NDJSON fields `type`, `text`, `source`, `path`,
`session_id`, and `score`. `enrich` accepts optional prompt text for manual
testing; when omitted, it reads UserPromptSubmit JSON from stdin. It only
retrieves existing prompt/answer and KB PageIndex context; it does not index.
Transcript
PageIndex remains a separate CLI navigation surface and is not injected by
default.

KB `build` writes exactly `nodes.tsv` and `manifest.tsv`. KB `query` and
`enrich` synchronously rebuild missing or stale indexes under a lock. Completed
indexes replace the prior index atomically. Direct query fails if rebuilding
fails; enrich uses the prior index with an agent-context warning, or warns and
continues without KB results when no index exists. Query prints matched excerpts
directly. KB `document`, `structure`, `content`, and agentic query commands are
retired. Transcript PageIndex query is deterministic lexical-only;
its document, structure, and content commands remain explicit CLI operations.

The former memory-unit and graph runtime paths are retired. The
`deduplicate`, `build-graph`, `graph-clean`, and `graph-dump` commands are no
longer public commands. The `src/memory_unit.rs`, `src/dedup.rs`, `src/graph.rs`,
`src/graph/`, and `src/graph_cmds.rs` runtime modules were removed.

The canonical durable-memory KB Markdown export completed before the
compatibility code was removed. Its Markdown and manifest remain the editable
KB representation, and migration backups exist. No runtime migration or export
command remains. The runtime uses the configured embedding collection; production uses `claude-session-history-qwen3-8b`, while `claude-session-history` remains intact for rollback.

## Claude Code plugin install

Install the active setup skill in Claude Code:

```bash
claude plugin marketplace add Osso/claude-memory && claude plugin install claude-memory@claude-memory
```

Invoke the installed skill to inspect prerequisites, install the CLI, configure the embedding model, validate services, and apply KB rules. The skill does not enable or restart system services without approval.

## Build & Install

Install the public repository directly:

```bash
cargo install --git https://github.com/Osso/claude-memory
```

For a local checkout:

```bash
./deploy.sh
```

`deploy.sh` installs the `claude-memory` CLI binary to `~/.cargo/bin/` with
`cargo install --force --path .`.

The installed interface is the `claude-memory` CLI binary.

## Embedding profiles

The persistent profile lives in `~/.config/claude-memory/config.toml`:

```toml
[embedding]
backend = "openrouter"
model = "qwen/qwen3-embedding-8b"
vector_size = 4096
collection = "claude-session-history-qwen3-8b"
query_instruction = "Represent this query for retrieval"
```

Supported fields are `backend` (`ollama` or `openrouter`), `model`, positive
integer `vector_size`, `collection`, and optional `query_instruction`. The
corresponding `CLAUDE_MEMORY_*` environment variables override file values.
Built-in local Ollama defaults apply only when neither the file profile nor
embedding environment variables are configured: model
`qwen3-embedding:0.6b-ctx2048`, 1024 dimensions, and collection
`claude-session-history`. That collection remains intact as rollback.

OpenRouter reads `api_key` only from `~/.config/openrouter/config.toml`; do not
copy it into environment variables or project files. Every OpenRouter request
enforces zero data retention. Document requests are batched and transient
failures are retried. A dense-vector dimension mismatch fails without
replacing the existing collection, and an embedding failure stops indexing.

## Dependencies

Requires running services:
- Qdrant: `authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin` then `authsudo systemctl enable --now qdrant.service`
- Ollama (rollback profile): `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
