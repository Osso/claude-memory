# Retrieval Flows

`claude-memory` has several retrieval surfaces. They share storage and LLM
helpers, but they intentionally answer different questions.

## Configuration Gates

Runtime config lives at `~/.config/claude-memory/config.toml`.

```toml
[search]
enabled = true

[graph]
enabled = false
```

Defaults are conservative: both `search.enabled` and `graph.enabled` are false
when the file or section is missing.

LLM calls go through `src/llm.rs` unless a path is explicitly noted otherwise.
`CLAUDE_MEMORY_LLM_BACKEND` selects `ollama`, `anthropic`, `openrouter`,
`claude`, or `codex`; unset and unknown values fall back to local Ollama.
`CLAUDE_MEMORY_LLM_MODEL` overrides the backend default model.

Known legacy gate: `claude-memory deduplicate` still checks for
`ANTHROPIC_API_KEY` before it calls the shared LLM merge path. That preflight is
not representative of the default backend behavior.

## `claude-memory enrich`

Purpose: inject small, relevant context into a Claude Code `UserPromptSubmit`
hook.

Flow in `src/enrich_cmd.rs`:

1. Read hook JSON from stdin and extract `prompt`.
2. Search memory units semantically with `memory_unit::search`.
   - Requires `[search].enabled = true`.
   - Uses Ollama embeddings and Qdrant memory-unit vectors.
   - Keeps only results with score `>= 0.75`.
3. Search KB PageIndex with `kb_search::search_default_kb_context`.
   - Does not use embeddings or an LLM.
   - Builds or refreshes the local KB PageIndex if stale.
   - Caps injected KB results to three sections and 500 chars each.
4. Optionally append graph context.
   - Requires `[graph].enabled = true`.
   - Reads CozoDB graph relationships; no extraction happens during enrich.
5. Emit Claude hook JSON with `additionalContext`, or `{}` if nothing matched.

Enrich never injects Transcript PageIndex results by default.

## CLI `search`

Purpose: manual lookup from the terminal.

`claude-memory search <query>` defaults to memory units:

- With `[search].enabled = true`, it calls `memory_unit::list(..., query=...)`
  for semantic vector lookup.
- With search disabled, it falls back to substring filtering over stored memory
  units.

`claude-memory search --type prompts <query>` searches prompt history and legacy
KB vector chunks. `--type answers` searches assistant responses. Both paths use
`src/index_search.rs` and require `[search].enabled = true`; if disabled they
return no results rather than using substring fallback.

## MCP Search Tools

Purpose: let Claude Code query indexed history through MCP.

Tools in `src/bin/mcp.rs`:

- `prompt_search`: user prompts, questions, and legacy KB vector chunks.
- `answer_search`: assistant responses and solutions.
- `memory_list`: exact category/project listing of memory entries.
- `memory_write`: storage disabled; returns guidance to write Markdown memory.

`prompt_search` and `answer_search` flow:

1. Require `[search].enabled = true`; otherwise no points are returned.
2. Embed query with Ollama.
3. Run hybrid Qdrant dense+BM25 search with over-fetch.
4. Ask the configured LLM backend to filter relevance.
5. Fall back to top results if LLM filtering fails.
6. Drop points below score `0.65`.
7. Optionally append graph context only when `[graph].enabled = true`.

## KB PageIndex

Purpose: traceable retrieval from `/syncthing/Sync/KB` Markdown.

CLI surface: `claude-memory kb-page-index ...`.

- `build` writes a persistent nested index under
  `~/.cache/claude-memory/kb-page-index` by default.
- `query` defaults to deterministic lexical lookup.
- `query --mode agentic` uses the configured LLM backend to inspect metadata and
  structure, choose content fetches, and answer from fetched content.
- `document`, `structure`, and `content` expose exact metadata and source text.

`claude-memory enrich` uses the deterministic KB PageIndex context path, not the
agentic mode.

## Transcript PageIndex

Purpose: local navigation of Claude and Codex session history.

CLI surface: `claude-memory transcript-page-index ...`.

- Builds nested outlines from Claude project/archive sessions and Codex session
  stores.
- Query defaults to deterministic lexical lookup.
- `--mode agentic` uses the configured LLM backend for tree-walk retrieval.
- It is CLI-only: no MCP tool and no default prompt enrichment injection.
- It does not write memory units.

## Memory Creation and Ingestion

Durable memory units come from analysis/ingestion paths, not from search:

- `analyze <session.jsonl>` parses one session, runs friction/notability LLM
  stages, and writes accepted memory units/notable facts with deduplication.
- `backfill` runs analysis over many sessions using a processed-state file.
- `ingest-kb` extracts facts from KB Markdown sections into memory units.
- Manual `memory_write` storage is disabled; project-local durable context should
  be written as editable Markdown under `docs/local/memory.md`.

## Graph Path

The graph is optional and disabled by default.

- `build-graph` iterates memory units, and optionally KB text, but extraction
  only runs when `[graph].enabled = true`.
- Extraction calls the configured LLM backend through `src/llm.rs`.
- Results are sanitized triplets stored in CozoDB at `~/.claude/memory/graph.db`.
- `enrich` and MCP search only read graph context when `[graph].enabled = true`.
- `graph-dump` and `graph-clean` are explicit graph maintenance commands.

## Which Surface To Use

| Need | Surface |
| --- | --- |
| Automatic prompt hints | `claude-memory enrich` hook |
| Find durable memory units | `claude-memory search <query>` |
| Find past user prompts/discussions | `prompt_search` or `search --type prompts` |
| Find prior assistant solutions | `answer_search` or `search --type answers` |
| Exact KB note context | `kb-page-index` |
| Inspect raw session history | `transcript-page-index` |
| Browse optional entity graph | `graph-dump`, graph-enabled enrich/search |
