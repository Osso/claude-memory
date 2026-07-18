# claude-memory

Semantic memory search for Claude Code sessions and the local knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Interface**: the `claude-memory` CLI binary
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Qdrant state**: only `claude-session-history` remains

## Usage

Manual CLI invocation plus Claude Code, Codex, and Pi lifecycle integration:

```bash
# Incrementally index missing Claude, Codex, and Pi transcript chunks
claude-memory index

# Search globally ranked prompt+answer history
claude-memory search "query"
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory search --limit 10 --json "query"

# Build/query KB text index (build writes only nodes.tsv and manifest.tsv)
claude-memory kb-page-index build --kb /syncthing/Sync/KB
claude-memory kb-page-index query "query" --kb /syncthing/Sync/KB
# Fetch exact source lines from a fresh index
claude-memory kb-page-index content path/to/note.md 4-8 --kb /syncthing/Sync/KB

# Build transcript PageIndex from Claude and Codex sessions
claude-memory transcript-page-index build
claude-memory transcript-page-index query "query"

# Enrich a prompt and show collection statistics
claude-memory enrich
claude-memory stats
```

Claude Code, Codex, and Pi session shutdown integrations automatically run
`claude-memory index-file <transcript_path>`. Manual `claude-memory index` is
incremental backfill and recovery across Claude active/archive, Codex
active/archive, and Pi session JSONL files. Existing hashes are skipped unless
`--fresh` is supplied. The default search runs one globally ranked prompt+answer
query over the shared `claude-session-history` collection; `--type
prompts|answers` provides optional filtering and `--limit` applies globally.
`--json` emits stable NDJSON fields `type`, `text`, `source`, `path`,
`session_id`, and `score`. UserPromptSubmit runs `enrich` only to retrieve
existing prompt/answer and KB PageIndex context; it does not index. Transcript
PageIndex remains a separate CLI navigation surface and is not injected by
default.

KB `build` writes exactly `nodes.tsv` and `manifest.tsv`. KB `query` reads those
files and rejects a stale index without rebuilding; run `build` explicitly after
source changes. KB `content` requires the KB source and an inclusive line range.
Query results include a follow-up `content` command with explicit `--kb` and
`--index` paths. The former KB `document`, `structure`, and agentic query
commands are retired. Transcript PageIndex query is deterministic lexical-only;
its document, structure, and content commands remain explicit CLI operations.

The former memory-unit and graph runtime paths are retired. The
`deduplicate`, `build-graph`, `graph-clean`, and `graph-dump` commands are no
longer public commands. The `src/memory_unit.rs`, `src/dedup.rs`, `src/graph.rs`,
`src/graph/`, and `src/graph_cmds.rs` runtime modules were removed.

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
