Prompt and answer history indexing stores raw user prompts and assistant responses from active and archived Claude transcript files in one searchable session-history index. `prompt_search` and `answer_search` are filtered views over that index; implementation details belong in [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md).

## What it must do

### Storage and filtering

- [x] Store prompt and answer chunks in the `claude-session-history` collection.
- [x] Persist each chunk with its text, `type` (`prompt` or `answer`), `source` (`session` or `archive`), path, session id, and persisted hash.
- [x] Make prompt search a `type=prompt` view and answer search a `type=answer` view.
- [x] Allow prompt and answer searches to restrict results by `source`.
- [x] Include history type in persisted chunk identity so identical text can exist once as a prompt and once as an answer.
- [x] Return text, source, path, session id, and score without panicking on missing payload fields.

### Indexing lifecycle

- [x] Index user messages from active Claude session JSONL files.
- [x] Index assistant messages from active Claude session JSONL files.
- [x] Index user messages from archived Claude `.jsonl.zst` files.
- [x] Index assistant messages from archived Claude `.jsonl.zst` files.
- [x] Deduplicate existing chunks and repeated chunks within one indexing input.
- [x] Index a single supported conversation file into the same prompt and answer views.
- [x] Leave project summaries, KB Markdown, manual memories, and the `claude-memory`, `claude-session-prompts`, and `claude-answers` stores outside this index.

### CLI and MCP search

- [x] Default `claude-memory search <query>` targets memories, not session history.
- [x] Accept `claude-memory search --type prompts <query>`.
- [x] Accept `claude-memory search --type answers <query>`.
- [x] Expose `prompt_search` for user prompts and questions from session history.
- [x] Expose `answer_search` for assistant responses and solutions from session history.
- [x] Return no session-history results when semantic search is disabled.
- [x] Keep KB PageIndex, transcript PageIndex, memory-unit search, and memory ingestion as separate surfaces.

## How it works

- [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md) describes the shared collection, payload fields, extraction, hashing, deduplication, hybrid search, and source filters.

## Implementation inventory

- `src/index.rs` — owns session-history collection setup, active/archive traversal, single-file indexing, and stats.
- `src/extract.rs` — extracts user and assistant transcript text and attaches history/source metadata.
- `src/index_writer.rs` — computes persisted identity, deduplicates chunks, builds payloads, and writes points.
- `src/index_search.rs` — applies history-type and source filters for CLI search.
- `src/bin/mcp.rs` — exposes filtered `prompt_search` and `answer_search` tools.
- `src/main.rs` — declares and dispatches session-history CLI search targets.
- `src/indexing_cmds.rs` — dispatches index and single-file commands.
- `src/qdrant_hybrid.rs` — creates the shared dense+sparse collection and query vectors.

## Tests asserting this spec

- `src/index_tests.rs`
  - `filter_new_keeps_new_items`
  - `filter_new_removes_existing_hashes`
  - `filter_new_empty_input_returns_empty`
  - `filter_new_all_duplicates_returns_empty`
  - `filter_new_deduplicates_within_input`
  - `identical_prompt_and_answer_text_have_distinct_history_hashes`
  - `qdrant_history_filters_isolate_type_and_source`
  - `build_search_results_extracts_fields`
  - `build_search_results_empty_payload_graceful`
  - `build_search_results_empty_input`
- `src/main_tests.rs`
  - `search_defaults_to_memories`
  - `search_accepts_prompt_type`
  - `search_accepts_answer_type`
- `src/extract.rs`
  - user-message and assistant-message extraction tests in the module test suite.

## Known gaps (current cycle)

- [ ] Add an integration test proving one fixture session writes extracted prompt and answer chunks to the shared collection with distinct `type` values.

## Out of scope

- Memory-unit storage and retrieval; see [memory-units.md](memory-units.md).
- Friction-driven memory creation and notable-fact handling; see [friction-memory-creation.md](friction-memory-creation.md).
- KB Markdown retrieval and ingestion; see [kb-page-index.md](kb-page-index.md).
- Transcript outline PageIndex retrieval; it is a separate local navigation surface.
