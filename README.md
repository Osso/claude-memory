# claude-memory

Semantic memory search for Claude Code sessions and the local knowledge base.

## Architecture

- **Unified session-history store**: Qdrant collection `claude-session-history` (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Interface**: the `claude-memory` CLI binary
- **KB retrieval**: persistent KB PageIndex over Markdown
- **Qdrant state**: only `claude-session-history` remains

## Usage

Manual CLI invocation + Claude Code hooks:

```bash
# Index active and archived Claude transcript chunks
claude-memory index

# Search unified prompt/answer history (--type is required)
claude-memory search --type prompts "query"
claude-memory search --type answers "query"

# Build/query KB text index (build writes only nodes.tsv and manifest.tsv)
claude-memory kb-page-index build --kb /syncthing/Sync/KB
claude-memory kb-page-index query "query" --kb /syncthing/Sync/KB
# Fetch exact source lines from a fresh index
claude-memory kb-page-index content path/to/note.md 4-8 --kb /syncthing/Sync/KB

# Build transcript PageIndex (unchanged)
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

KB `build` writes exactly `nodes.tsv` and `manifest.tsv`. KB `query` reads those
files and rejects a stale index without rebuilding; run `build` explicitly after
source changes. KB `content` requires the KB source and an inclusive line range.
Query results include a follow-up `content` command with explicit `--kb` and
`--index` paths. The KB `document`, `structure`, and agentic query commands are
retired. Transcript PageIndex behavior is unchanged.

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
