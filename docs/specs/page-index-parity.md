PageIndex parity defines the contract for matching the useful architecture of `VectifyAI/PageIndex` reference commit `f50e529` inside `claude-memory`. The target is not a line-for-line port; it is a Rust-native PageIndex surface for local KB Markdown and Claude/Codex transcript history. KB uses a deterministic text index; Transcript PageIndex retains the nested document model and deterministic lexical query. Implementation details belong in [docs/wiki/systems/page-index-parity.md](../wiki/systems/page-index-parity.md).

Decision: PDF parsing and OCR support stay out of scope until there is a
concrete need for document formats beyond Markdown KB notes and transcript
history. Do not add PDF/OCR dependencies speculatively.

Decision: corpus-level PageIndex file-system routing stays out of scope until
single-document query quality and corpus-scale build costs are proven good
enough. Current routing remains explicit: KB PageIndex for Markdown KB,
Transcript PageIndex for Claude/Codex sessions.

## What it must do

### Reference baseline

Reference implementation: `VectifyAI/PageIndex` at commit `f50e529`.

- [x] Preserve traceable Transcript PageIndex document, structure, and exact-content surfaces.
- [x] Keep the reference distinction between transcript structure and transcript content: structure output must not require dumping full turn text.
- [x] Keep stable transcript node identifiers suitable for follow-up content fetches.
- [x] Keep KB retrieval outside that JSON/nested-document CLI contract: KB uses deterministic TSV text search and exact source line ranges.

### Supported source families

- [x] Support KB Markdown from `/syncthing/Sync/KB` through the deterministic TSV text index.
- [x] Support live Claude transcripts from `~/.claude/projects` as PageIndex documents.
- [x] Support archived Claude transcripts from `~/.claude/archive` as PageIndex documents.
- [x] Support live Codex transcripts from `~/.codex/sessions` as PageIndex documents.
- [x] Support archived Codex transcripts from `~/.codex/archived_sessions` as PageIndex documents.
- [x] Keep KB PageIndex and Transcript PageIndex as separate surfaces with separate default output directories.

### Document model

- [x] Store transcript document metadata and nested nodes with stable locators.
- [x] Store KB heading-aware nodes in `nodes.tsv` and source freshness metadata in `manifest.tsv`.
- [x] Provide exact KB content retrieval from the source Markdown file by an inclusive line range.
- [x] Preserve the transcript distinction between structure without full text and exact content fetch.

### KB Markdown behavior

- [x] Build heading-aware KB nodes from Markdown into `nodes.tsv`.
- [x] Ignore Markdown headings inside fenced code blocks.
- [x] Preserve heading source line numbers.
- [x] Record source freshness in `manifest.tsv`.
- [x] Reject query/content when Markdown files are added, changed, deleted, or otherwise make the manifest stale; rebuild is explicit.
- [x] Keep only results in the best distinct query-term coverage tier up to the requested limit; do not fill remaining slots with weaker matches, and exclude archive results when nonarchive matches exist in that tier.
- [x] Keep at most one matching section per source document before applying the limit.
- [x] Preserve the frontend two-result, bash-hook document-diversity, AMDGPU-first, and absent-query quality gates.

### Transcript behavior

- [x] Build transcript documents from Claude and Codex session JSONL formats.
- [x] Group turns into navigable exchange nodes with exact turn ranges.
- [x] Preserve assistant tool-call counts in transcript node summaries.
- [x] Fetch exact turn text for a node id or turn range.
- [x] Transcript PageIndex must not write memory units; legacy memory-unit creation and retrieval are retired.
- [x] Keep Transcript PageIndex CLI-only; broader integrations remain out of scope.

### CLI and retrieval surfaces

- [x] `claude-memory kb-page-index build` writes only `nodes.tsv` and `manifest.tsv`.
- [x] `claude-memory kb-page-index query <query>` reads the persisted TSV index and rejects stale data without rebuilding.
- [x] `claude-memory kb-page-index content <doc-path> <start-end> --kb <dir> --index <dir>` requires the KB source and returns the exact inclusive line range.
- [x] KB query results include a custom follow-up content command with the selected `--kb` and `--index` paths.
- [x] Retire the KB `document`, `structure`, and agentic query commands.
- [x] `claude-memory transcript-page-index build` builds the transcript PageIndex.
- [x] `claude-memory transcript-page-index document <doc-id-or-path>` prints transcript metadata.
- [x] `claude-memory transcript-page-index structure <doc-id-or-path>` prints transcript outline without full turn text.
- [x] `claude-memory transcript-page-index content <doc-id-or-path> <node-id-or-range>` prints exact turn text.
- [x] `claude-memory transcript-page-index query <query>` uses deterministic lexical scoring and returns traceable transcript references plus a follow-up content command.
- [x] Keep Transcript PageIndex query free of model calls and alternate query modes.
- [x] Keep KB query deterministic; KB agentic mode is retired.

### Enrich integration

- [x] `claude-memory enrich` may include KB PageIndex output alongside unified prompt/answer history when it fits hook latency and output budgets.
- [x] `claude-memory enrich` omits KB output when the text index is missing or stale until an explicit rebuild.
- [x] `claude-memory enrich` must label KB PageIndex context as `KB PageIndex`.
- [x] `claude-memory enrich` must not inject Transcript PageIndex results by default.

### Non-goals and bounded parity

- PDF parsing parity is deferred until there is a concrete need for document formats beyond Markdown and transcripts.
- OCR and PageIndex cloud/API parity are out of scope for this local Rust implementation.
- FinanceBench and other reference benchmark parity are not claimed.
- Transcript PageIndex integrations beyond the CLI are deferred until query-quality and corpus-scale build benchmarks improve.
- Transcript PageIndex does not replace friction-driven memory creation.

## How it works

- `src/kb_search.rs` implements the KB TSV text index and exact source line-range retrieval.
- `src/page_index.rs` implements the Transcript PageIndex document model, Claude/Codex parsing, build, and deterministic lexical query.
- [kb-page-index.md](kb-page-index.md) describes the current KB PageIndex implementation.
- [friction-memory-creation.md](friction-memory-creation.md) records the retired transcript-mining boundary.

## Implementation inventory

- `src/kb_search.rs` — KB TSV text-index builder/search path with exact source line-range content fetch.
- `src/page_index.rs` — transcript outline builder for Claude/Codex sessions using the nested PageIndex document model.
- `src/indexing_cmds.rs` — CLI command handlers for KB build/query/content and Transcript PageIndex commands.
- `src/enrich_cmd.rs` — prompt hook integration for KB PageIndex context.
- `src/main.rs` — CLI declaration and dispatch for KB PageIndex and Transcript PageIndex commands.
- `src/main_tests.rs` — CLI parsing tests for PageIndex commands.
- `src/extract.rs` — Claude transcript parsing used by transcript PageIndex and session-history indexing.

## Tests asserting this spec

- Current tests assert the checked behavior above:
  - `tests/kb_page_index_cli.rs`
    - `explicit_build_writes_only_text_index_files`
    - `query_reads_explicit_text_index`
    - `stale_query_fails_without_rebuilding`
    - `content_fetch_reads_exact_markdown_line_range`
    - `content_fetch_preserves_exact_line_endings`
    - `json_only_kb_commands_are_retired`
  - `src/kb_search.rs`
    - `text_index_files_round_trip_generated_sections`
    - `stale_text_search_rejects_without_automatic_rebuild`
    - `text_search_weights_heading_over_path_over_body`
    - `text_search_rewards_exact_phrase`
    - `frontend_quality_gate_excludes_archive_noise_from_top_three`
    - `bash_hook_quality_gate_returns_distinct_documents`
    - `quality_gate_preserves_amdgpu_first_and_absent_query_behavior`
  - `src/page_index.rs`
    - `fixture_transcripts_prove_structure_content_query_and_no_memory_units`
    - `session_index_groups_prompt_and_answer_in_one_node`
    - `outline_exposes_node_ids_and_titles`
    - `node_text_returns_prompt_and_answer`
    - `codex_parser_keeps_only_user_and_assistant_messages`
    - `page_index_sources_collect_claude_archive_and_codex_sessions`
  - `src/main_tests.rs`
    - `transcript_page_index_accepts_projects_archive_output_and_limit`
    - `transcript_page_index_accepts_document_structure_content_and_query_commands`
    - `kb_page_index_accepts_build_paths`
    - `kb_page_index_accepts_query_paths_and_limit`
    - `kb_page_index_accepts_content_command`

## Known gaps (current cycle)

- [x] Add tests for the shared nested PageIndex document model.
- [x] Add tests for structure output that omits full node text.
- [x] Add tests for exact content fetch by node id and range.
- [x] Add tests for KB stale-index rejection when files are added, changed, or deleted.
- [x] Add tests for transcript query returning traceable document/node references.
- [x] Add tests proving Transcript PageIndex does not create memory units.
- [ ] Re-run build-time, output-size, and query-quality benchmarks for the current TSV text index; the historical PageIndex benchmark no longer measures this runtime.
- [ ] Evaluate KB result quality beyond the targeted frontend, bash-hook, AMDGPU-first, and absent-query gates.

## Out of scope

- Full PDF PageIndex parity; no PDF parser or OCR work until Markdown and transcript parity are complete.
- KB document/structure commands and KB agentic query mode.
- Cloud PageIndex API compatibility.
- Transcript PageIndex integrations beyond the CLI before query quality is proven.
- Corpus-scale PageIndex file-system routing across millions of documents before single-document query quality is proven.
- Replacing prompt/answer vector history.
- Reintroducing friction-driven memory creation or memory-unit runtime retrieval.
