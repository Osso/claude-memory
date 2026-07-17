The KB PageIndex feature provides persistent, heading-aware retrieval over the local Markdown knowledge base at `/syncthing/Sync/KB`. It is the source of KB context for prompt enrichment without requiring KB facts to be duplicated into vector memory units. It is separate from [prompt and answer history](prompt-answer-history.md), which indexes transcript chunks only.

## What it must do

### Index lifecycle

- [x] Build a persistent KB text index from a Markdown directory into an index directory.
- [x] Write exactly `nodes.tsv` and `manifest.tsv` for the KB text index.
- [x] Store enough index metadata to report indexed file and node counts.
- [x] Preserve Markdown heading paths in search results, such as `Corrections > Process`.
- [x] Have query read the persisted TSV files and reject a stale index without rebuilding it.

### CLI surface

- [x] Accept `claude-memory kb-page-index build --kb <dir> --output <dir>`.
- [x] Accept `claude-memory kb-page-index query <query> --limit <n> --kb <dir> --index <dir>` without an implicit rebuild.
- [x] Accept `claude-memory kb-page-index content <doc-path> <start-end> --kb <dir> --index <dir>` for an inclusive source line range.
- [x] Print a clear no-results message when no KB section matches.
- [x] Print query results with source path, inclusive line range, heading path, score, and a follow-up content command.
- [x] Make the follow-up command explicit for custom roots: `claude-memory kb-page-index content <doc-path> <start-end> --kb <dir> --index <dir>`.
- [x] Retire the KB `document`, `structure`, and agentic query commands.

### Retrieval behavior

- [x] Query `nodes.tsv` and `manifest.tsv` without rebuilding the index.
- [x] Reject the query when the KB source is missing or the manifest is stale.
- [x] Return the matching source path and heading for a query that targets a persisted section.
- [x] Require content retrieval to name the KB source and an exact inclusive line range.
- [ ] Avoid weak long-query matches that share only one or two incidental terms.
- [ ] Rank exact operational rules above generic bookmark/reference material for common agent prompts.
- [ ] Return useful results for the real KB query `frontend design skill load immediately`.
- [ ] Return useful results for the real KB query `claude bash hook codex unsafe`.

### Enrich integration

- [x] Format KB results under a distinct `Relevant KB notes` section.
- [x] Label enrich KB context as coming from `KB PageIndex`.
- [x] Include KB PageIndex results alongside unified prompt/answer history when both are relevant.
- [x] Include KB results from an existing fresh text index only; omit the KB section when the index is missing or stale until an explicit rebuild.
- [x] Cap enrich KB output to a small number of results.

## Implementation inventory

- `src/kb_search.rs` — builds, stores, validates, loads, scores, and queries the persistent KB text index.
- `src/enrich_cmd.rs` — reads prompt-hook input and injects formatted unified prompt/answer history and KB PageIndex results.
- `src/indexing_cmds.rs` — implements the `kb-page-index build`, `query`, and exact-line-range `content` command handlers.
- `src/kb_page_index_cli.rs` — declares the `kb-page-index` CLI subcommands.
- `src/main.rs` — dispatches the `kb-page-index` CLI subcommands.
- `src/main_tests.rs` — covers CLI parsing for KB PageIndex commands.
- `README.md` — lists the user-facing KB PageIndex commands.

## Tests asserting this spec

- `tests/kb_page_index_cli.rs`
  - `explicit_build_writes_only_text_index_files`
  - `query_reads_explicit_text_index`
  - `stale_query_fails_without_rebuilding`
  - `query_rejects_added_deleted_and_missing_kb_files`
  - `build_creates_missing_nested_output_parents`
  - `json_only_kb_commands_are_retired`
  - `content_fetch_preserves_exact_line_endings`
  - `content_fetch_reads_exact_markdown_line_range`
- `src/kb_search_tests.rs`
  - `stale_text_search_rejects_without_automatic_rebuild`
  - `text_search_uses_deterministic_path_order_for_ties`
  - `search_kb_context_fetches_exact_node_content_for_enrich`
- `src/enrich_cmd.rs`
  - `fresh_text_index_result_formats_for_enrich_with_explicit_content_roots`

## Known gaps (current cycle)

- [x] Cover stale-index rejection when KB files are added, changed, or deleted.
- [x] Add an integration-style test for `enrich` that proves KB PageIndex results are included and capped.
- [ ] Add a CLI output test or snapshot for query result formatting and no-results behavior.
- [ ] Decide whether KB PageIndex should index repo-local `AGENTS.md` / persistent rules as a separate source.

## Out of scope

- Vector embeddings for KB sections; KB PageIndex is the deterministic TSV text retrieval path.
- JSON document metadata, structure, node-id content fetches, and KB agentic traversal.
- Session-history vector indexing; see [prompt-answer-history.md](prompt-answer-history.md).
- LLM-guided PageIndex traversal is not part of the KB CLI; KB query remains deterministic lexical scoring over persisted heading nodes.
- Replacing transcript PageIndex; session PageIndex and KB PageIndex are separate surfaces.
