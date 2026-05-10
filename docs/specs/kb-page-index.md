The KB PageIndex feature provides persistent, heading-aware retrieval over the local Markdown knowledge base at `/syncthing/Sync/KB`. It is the source of KB context for prompt enrichment without requiring KB facts to be duplicated into vector memory units; implementation details belong in [docs/wiki/systems/kb-page-index.md](../wiki/systems/kb-page-index.md).

## What it must do

### Index lifecycle

- [x] Build a persistent KB index from a Markdown directory into an index directory.
- [x] Store enough index metadata to report indexed file and node counts.
- [x] Preserve Markdown heading paths in search results, such as `Corrections > Process`.
- [x] Refresh the persistent index automatically when a source Markdown file changes.
- [ ] Refresh the persistent index automatically when a source Markdown file is deleted.
- [ ] Refresh the persistent index automatically when a new Markdown file is added.

### CLI surface

- [x] Accept `claude-memory kb-page-index build --kb <dir> --output <dir>`.
- [x] Accept `claude-memory kb-page-index query <query> --limit <n> --kb <dir> --index <dir>`.
- [ ] Print a clear no-results message when no KB section matches.
- [ ] Print query results with source path, heading path, score, and snippet.

### Retrieval behavior

- [ ] Query the persisted index without re-reading every Markdown file when the index is fresh.
- [x] Return the matching source path and heading for a query that targets a persisted section.
- [x] Avoid weak long-query matches that share only one or two incidental terms.
- [ ] Rank exact operational rules above generic bookmark/reference material for common agent prompts.
- [ ] Return useful results for the real KB query `frontend design skill load immediately`.
- [ ] Return useful results for the real KB query `claude bash hook codex unsafe`.

### Enrich integration

- [x] Format KB results under a distinct `Relevant KB notes` section.
- [x] Label enrich KB context as coming from `KB PageIndex`.
- [ ] Include KB PageIndex results alongside memory-unit preloads when both are relevant.
- [x] Auto-build or refresh the KB PageIndex from enrich when the index is missing or stale.
- [x] Cap enrich KB output to a small number of results.

## How it works

- [docs/wiki/systems/kb-page-index.md](../wiki/systems/kb-page-index.md) describes the on-disk index, stale-index detection, scoring, and enrich integration.

## Implementation inventory

- `src/kb_search.rs` — builds, stores, refreshes, loads, scores, and queries the persistent KB PageIndex.
- `src/enrich_cmd.rs` — reads prompt-hook input and injects formatted KB PageIndex results with memory preloads.
- `src/indexing_cmds.rs` — implements the `kb-page-index build` and `kb-page-index query` command handlers.
- `src/main.rs` — declares the `kb-page-index` CLI subcommands and dispatches them.
- `src/main_tests.rs` — covers CLI parsing for KB PageIndex commands.
- `README.md` — lists the user-facing KB PageIndex commands.

## Tests asserting this spec

- `src/kb_search.rs`
  - `build_and_search_persisted_kb_index`
  - `search_or_build_refreshes_stale_index`
  - `long_queries_require_three_distinct_terms`
- `src/enrich_cmd.rs`
  - `kb_results_include_source_path_and_heading`
- `src/main_tests.rs`
  - `kb_page_index_accepts_build_paths`
  - `kb_page_index_accepts_query_paths_and_limit`

## Known gaps (current cycle)

- [ ] Add tests for new-file and deleted-file stale-index refresh.
- [x] Add an integration-style test for `enrich` that proves KB PageIndex results are included and capped.
- [ ] Add a CLI output test or snapshot for query result formatting and no-results behavior.
- [ ] Decide whether KB PageIndex should index repo-local `AGENTS.md` / persistent rules as a separate source.

## Out of scope

- Vector embeddings for KB sections; KB PageIndex is the raw structured retrieval path.
- LLM-guided PageIndex traversal; current retrieval is deterministic lexical scoring over persisted heading nodes.
- Replacing transcript PageIndex; session PageIndex and KB PageIndex are separate surfaces.
