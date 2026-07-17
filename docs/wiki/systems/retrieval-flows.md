# Retrieval Flows

`claude-memory` has two runtime retrieval sources for prompt enrichment. They
answer different questions and remain separately traceable.

## Active surfaces

### Session-history search

`claude-memory index` reads active Claude `.jsonl` sessions and archived
`.jsonl.zst` sessions. Prompt and answer searches are filtered views over the
shared `claude-session-history` collection. MCP exposes the same two views as
`prompt_search` and `answer_search`.

### `claude-memory enrich`

Enrichment reads only:

1. unified raw prompt/answer history chunks from `claude-session-history`;
2. deterministic KB PageIndex context.

The output labels the two sources separately. KB PageIndex remains the exact
Markdown retrieval path. Transcript PageIndex is CLI-only navigation and is not
injected by default.

### PageIndex

KB PageIndex retrieves canonical Markdown from `/syncthing/Sync/KB`.
Transcript PageIndex builds local outlines for Claude and Codex sessions. Neither
PageIndex path writes memory units or graph records.

## Retired runtime paths

The memory-unit runtime search, listing/deletion, enrich, and deduplication paths
are retired. The graph runtime and its `build-graph`, `graph-clean`, and
`graph-dump` commands are retired. The corresponding `src/memory_unit.rs`,
`src/dedup.rs`, `src/graph.rs`, `src/graph/`, and `src/graph_cmds.rs` modules
were deleted.

Legacy memory-unit, migration, and export compatibility paths are retired.
The canonical KB Markdown export completed before their removal; its Markdown
and manifest remain available through KB PageIndex, and migration backups exist.
Qdrant now contains only `claude-session-history`.

## Which surface to use

| Need | Surface |
| --- | --- |
| Automatic prompt context | `claude-memory enrich` hook |
| Find past user prompts/discussions | `prompt_search` or `search --type prompts` |
| Find prior assistant solutions | `answer_search` or `search --type answers` |
| Exact KB note context | `kb-page-index` |
| Inspect raw session history | `transcript-page-index` |
