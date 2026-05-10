PageIndex parity defines the contract for matching the useful architecture of `VectifyAI/PageIndex` reference commit `f50e529` inside `claude-memory`. The target is not a line-for-line port; it is a Rust-native PageIndex surface for local KB Markdown and Claude/Codex transcript history with the same core retrieval model: document metadata, structure without full text, exact content fetch, and optional agentic tree-walk retrieval. Implementation details belong in [docs/wiki/systems/page-index-parity.md](../wiki/systems/page-index-parity.md).

## What it must do

### Reference baseline

- [ ] Record the reference implementation and commit being matched: `VectifyAI/PageIndex` at `f50e529`.
- [x] Match the reference architectural flow: index a document, inspect metadata, inspect structure without full text, fetch tight content ranges, answer with traceable references.
- [x] Keep the reference distinction between document structure and document content: structure output must not require dumping full node text.
- [x] Use stable node identifiers that are suitable for follow-up content fetches.

### Supported source families

- [x] Support KB Markdown documents from `/syncthing/Sync/KB` as PageIndex documents.
- [x] Support live Claude transcripts from `~/.claude/projects` as PageIndex documents.
- [x] Support archived Claude transcripts from `~/.claude/archive` as PageIndex documents.
- [x] Support live Codex transcripts from `~/.codex/sessions` as PageIndex documents.
- [x] Support archived Codex transcripts from `~/.codex/archived_sessions` as PageIndex documents.
- [x] Keep KB PageIndex and Transcript PageIndex as separate surfaces with separate default output directories.

### Document model

- [x] Store document metadata: `doc_id`, source path, source family, document name/title, document description when available, and line/turn count.
- [x] Store nested `nodes` with stable zero-padded `node_id`, title, source locator, optional summary, optional children, and internal text/content references.
- [x] Preserve exact content internally so later fetch commands can return source text without re-reading the original file when the index is fresh.
- [x] Provide structure serialization without node text for low-token inspection.
- [x] Provide content serialization for specific node ids, line ranges, or turn ranges.

### KB Markdown behavior

- [x] Build nested KB nodes from Markdown heading hierarchy.
- [ ] Ignore Markdown headings inside fenced code blocks.
- [x] Preserve heading source line numbers.
- [x] Preserve section text for exact content fetch.
- [x] Refresh or rebuild the persistent KB PageIndex when Markdown files are added, changed, or deleted.

### Transcript behavior

- [x] Build transcript documents from Claude and Codex session JSONL formats.
- [x] Group turns into navigable exchange nodes with exact turn ranges.
- [x] Preserve assistant tool-call counts in transcript node summaries.
- [x] Fetch exact turn text for a node id or turn range.
- [x] Transcript PageIndex must not write memory units and must not replace friction-driven memory creation.

### CLI and retrieval surfaces

- [x] `claude-memory kb-page-index build` builds the KB PageIndex.
- [x] `claude-memory kb-page-index document <doc-id-or-path>` prints document metadata.
- [x] `claude-memory kb-page-index structure <doc-id-or-path>` prints nested structure without node text.
- [x] `claude-memory kb-page-index content <doc-id-or-path> <node-id-or-range>` prints exact KB source text.
- [x] `claude-memory kb-page-index query <query>` returns traceable KB references and uses structure/content retrieval rather than flat snippet-only search.
- [x] `claude-memory transcript-page-index build` builds the transcript PageIndex.
- [x] `claude-memory transcript-page-index document <doc-id-or-path>` prints transcript metadata.
- [x] `claude-memory transcript-page-index structure <doc-id-or-path>` prints transcript outline without full turn text.
- [x] `claude-memory transcript-page-index content <doc-id-or-path> <node-id-or-range>` prints exact turn text.
- [x] `claude-memory transcript-page-index query <query>` returns traceable transcript references and a follow-up content command.

### Agentic tree-walk retrieval

- [x] Provide a query mode that mirrors the reference tool loop: inspect document metadata, inspect structure, choose tight node/range targets, fetch content, answer from fetched content only.
- [x] Use the project `llm` backend abstraction for any model calls; do not add direct external API calls.
- [x] Include the retrieval path in query output or logs so the selected document, node ids, and fetched ranges are auditable.
- [x] Keep deterministic lexical search available as a debug/fallback mode and label it clearly when used.

### Enrich integration

- [x] `claude-memory enrich` may include KB PageIndex output when it fits hook latency and output budgets.
- [x] `claude-memory enrich` must label KB PageIndex context as `KB PageIndex`.
- [x] `claude-memory enrich` must not inject Transcript PageIndex results by default.

### Non-goals and bounded parity

- [ ] Do not implement PDF parsing parity in this cycle.
- [ ] Do not implement OCR or PageIndex cloud/API parity in this cycle.
- [ ] Do not claim FinanceBench or other reference benchmark parity.
- [ ] Do not replace friction-driven memory creation with Transcript PageIndex.

## How it works

- `src/kb_search.rs`, `src/page_index.rs`, and `src/page_index_agentic.rs` implement the Rust document model, persistent index layout, CLI/tool loop, and retrieval modes.
- [kb-page-index.md](kb-page-index.md) describes the current KB PageIndex implementation.
- [friction-memory-creation.md](friction-memory-creation.md) describes transcript mining for durable memory creation, which remains separate.

## Implementation inventory

- `src/kb_search.rs` — KB PageIndex builder/search path with nested document structure and content fetch.
- `src/page_index.rs` — transcript outline builder for Claude/Codex sessions using the nested PageIndex document model.
- `src/page_index_agentic.rs` — shared agentic/tree-walk retrieval mode and lexical fallback.
- `src/indexing_cmds.rs` — CLI command handlers for PageIndex build/query/document/structure/content commands.
- `src/enrich_cmd.rs` — prompt hook integration for KB PageIndex context.
- `src/main.rs` — CLI declaration and dispatch for KB PageIndex and Transcript PageIndex commands.
- `src/main_tests.rs` — CLI parsing tests for PageIndex commands.
- `src/extract.rs` — Claude transcript parsing used by transcript PageIndex and friction analysis.

## Tests asserting this spec

- Current tests assert the checked behavior above:
  - `src/kb_search.rs`
    - `build_and_search_persisted_kb_index`
    - `search_or_build_refreshes_stale_index`
    - `fixture_markdown_proves_nested_structure_content_and_query`
    - `search_or_build_refreshes_added_and_deleted_markdown_files`
    - `long_queries_require_three_distinct_terms`
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
    - `kb_page_index_accepts_document_structure_and_content_commands`
  - `src/page_index_agentic.rs`
    - `tree_walk_inspects_metadata_structure_then_fetches_content`
    - `empty_agentic_plan_uses_labeled_lexical_fallback`

## Known gaps (current cycle)

- [x] Add tests for the shared nested PageIndex document model.
- [x] Add tests for structure output that omits full node text.
- [x] Add tests for exact content fetch by node id and range.
- [x] Add tests for KB new-file, changed-file, and deleted-file refresh.
- [x] Add tests for transcript query returning traceable document/node references.
- [x] Add tests proving Transcript PageIndex does not create memory units.
- [x] Add an agentic tree-walk test with a fake LLM/tool transcript before using a live model backend.
- [x] Benchmark build time, output size, and query quality against the current flat implementation and `rg`.

## Out of scope

- Full PDF PageIndex parity; no PDF parser or OCR work until Markdown and transcript parity are complete.
- Cloud PageIndex API/MCP compatibility.
- Corpus-scale PageIndex file-system routing across millions of documents.
- Replacing prompt/answer vector history.
- Replacing friction-driven memory creation.
