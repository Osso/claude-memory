# Retrieval Flows

`claude-memory` has several retrieval surfaces. They share storage and LLM
helpers, but answer different questions.

## Configuration gates

Runtime config lives at `~/.config/claude-memory/config.toml`.

```toml
[search]
enabled = true

[graph]
enabled = false
```

Defaults are conservative: both gates are false when missing.

## Active surfaces

### Session-history search

`claude-memory index` reads active Claude `.jsonl` sessions and archived
`.jsonl.zst` sessions. Prompt and answer searches are filtered views over the
shared `claude-session-history` collection. MCP exposes the same two views as
`prompt_search` and `answer_search`.

### `claude-memory enrich`

Enrichment can read:

1. memory-unit search results, above the configured relevance floor;
2. raw prompt/answer history chunks;
3. deterministic KB PageIndex context;
4. optional graph context when graph search is enabled.

Memory units are labeled as possibly useful hints. KB PageIndex remains the
exact Markdown retrieval path. Transcript PageIndex is CLI-only navigation and
is not injected by default.

### PageIndex

KB PageIndex retrieves canonical Markdown from `/syncthing/Sync/KB`.
Transcript PageIndex builds local outlines for Claude and Codex sessions. Neither
PageIndex path writes notable facts or memory units.

## Retired memory creation

The transcript analyzer and `analyze`/`backfill` commands are retired. No active
friction-memory or notable-fact analyzer writer remains. Existing memory-unit
read, deduplication, and enrich paths remain separate from transcript mining.

The completed durable-memory KB export remains the canonical Markdown export
path. Migration/export compatibility readers retain legacy `source=summary` and
`source=kb` recognition. These readers do not delete legacy points or
collections.

## Optional graph path

`build-graph`, `graph-dump`, and `graph-clean` maintain the optional CozoDB graph.
`enrich` and MCP search only read graph context when enabled.

## Which surface to use

| Need | Surface |
| --- | --- |
| Automatic prompt hints | `claude-memory enrich` hook |
| Find past user prompts/discussions | `prompt_search` or `search --type prompts` |
| Find prior assistant solutions | `answer_search` or `search --type answers` |
| Exact KB note context | `kb-page-index` |
| Inspect raw session history | `transcript-page-index` |
| Browse optional entity graph | `graph-dump`, graph-enabled enrich/search |
