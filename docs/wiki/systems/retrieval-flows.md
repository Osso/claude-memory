# Retrieval Flows

`claude-memory` has two runtime retrieval sources for prompt enrichment. They
answer different questions and remain separately traceable.

## Active surfaces

### Session-history search

`claude-memory index` reads active Claude `.jsonl` sessions and archived
`.jsonl.zst` sessions. Typed prompt and answer searches are CLI views over the
shared `claude-session-history` collection.

### `claude-memory enrich`

Enrichment reads only:

1. unified raw prompt/answer history chunks from `claude-session-history`;
2. deterministic KB PageIndex context.

The output labels the two sources separately. KB PageIndex uses the existing
fresh TSV text index; missing or stale indexes cause enrichment to omit KB output
until an explicit build. Transcript PageIndex is CLI-only navigation and is not
injected by default.

### PageIndex

KB PageIndex retrieves canonical Markdown from `/syncthing/Sync/KB`.
Transcript PageIndex builds local outlines for Claude and Codex sessions. Neither
PageIndex path writes memory units or graph records.

KB text search first filters matches to the best distinct query-term coverage
tier. It returns only that tier, up to the requested limit; it does not fill
remaining slots with weaker matches. If that tier contains any nonarchive
result, archive results are excluded. Before applying the limit, search keeps
at most one section per source document, making the returned documents distinct.
The real-KB query `frontend design skill load immediately` returns two relevant
documents, with `memory/corrections.md` first, rather than a noisy third;
`claude bash hook codex unsafe` returns three distinct documents. AMDGPU-first
ordering and empty-query behavior remain covered by quality gates. Broader
corpus quality and current TSV benchmark data remain open gaps.

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
| Find past user prompts/discussions | `search --type prompts` |
| Find prior assistant solutions | `search --type answers` |
| Exact KB note context | `kb-page-index` |
| Inspect raw session history | `transcript-page-index` |
