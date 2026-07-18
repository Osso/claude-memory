Prompt and answer history indexing stores raw user prompts and assistant responses from Claude, Codex, and Pi transcript files in one searchable session-history index. Typed prompt and answer searches are CLI-only; implementation details belong in [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md).

## What it must do

### Storage and filtering

- [x] Store prompt and answer chunks in the `claude-session-history` collection.
- [x] Persist each chunk with its text, `type` (`prompt` or `answer`), `source` (`session` or `archive`), path, session id, and persisted hash.
- [x] Make prompt search a `type=prompt` view and answer search a `type=answer` view.
- [x] Allow prompt and answer searches to restrict results by `source`.
- [x] Derive persisted identity from history type, source, and content hash: identical text remains distinct across prompt/answer and session/archive views, while identical chunks within the same type/source collapse to one point.
- [x] Do not add message, turn, or chunk ordinals solely to preserve repeated identical chunks; duplicate collapse is intentional.
- [x] Return text, source, path, session id, and score without panicking on missing payload fields.

### Indexing lifecycle

- [x] Index user messages from active Claude session JSONL files.
- [x] Index assistant messages from active Claude session JSONL files.
- [x] Index user messages from archived Claude `.jsonl.zst` files.
- [x] Index assistant messages from archived Claude `.jsonl.zst` files.
- [x] Index user and assistant messages from active and archived Codex JSONL files.
- [x] Index user and assistant messages from Pi session JSONL files, including archived sessions that remain in the session tree.
- [x] Exclude Pi detached-job and runtime JSONL artifacts that lack a Pi session header.
- [x] Deduplicate existing chunks and repeated identical chunks within the same type/source.
- [x] Index a single supported conversation file into the same prompt and answer views.
- [x] Run `claude-memory index-file <transcript_path>` automatically from Claude Code, Codex, and Pi session shutdown integration.
- [x] Auto-detect Claude, Codex, and Pi JSONL formats for `index-file`.
- [x] Reserve manual `claude-memory index` for incremental backfill and recovery; skip existing hashes unless `--fresh` is supplied.
- [x] Keep UserPromptSubmit `enrich` retrieval-only; it does not index transcripts.
- [x] Leave project summaries, KB Markdown, manual memories, and the `claude-memory`, `claude-session-prompts`, and `claude-answers` stores outside this index.

### CLI search

- [x] Require `--type prompts|answers` for `claude-memory search <query>`.
- [x] Accept `claude-memory search --type prompts <query>`.
- [x] Accept `claude-memory search --type answers <query>`.
- [x] Keep typed prompt and answer search CLI-only.
- [x] Return no session-history results when semantic search is disabled.
- [x] Keep KB PageIndex and Transcript PageIndex as separate surfaces; legacy memory-unit data is compatibility-only and KB-to-memory-unit ingestion is retired.

## How it works

- [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md) describes the shared collection, payload fields, extraction, hashing, deduplication, hybrid search, and source filters.

## Implementation inventory

- `src/index.rs` — owns session-history collection setup, active/archive traversal, single-file indexing, and stats.
- `src/extract.rs` — extracts user and assistant transcript text and attaches history/source metadata.
- `src/index_writer.rs` — computes persisted identity, deduplicates chunks, builds payloads, and writes points.
- `src/index_search.rs` — applies history-type and source filters for CLI search.
- `src/main.rs` — declares and dispatches session-history CLI search targets.
- `src/indexing_cmds.rs` — dispatches index and single-file commands.
- `src/qdrant_hybrid.rs` — creates the shared dense+sparse collection and query vectors.

## Tests asserting this spec

- `src/index_tests.rs`
  - `index_sources_discover_claude_codex_and_pi_sessions`
  - `index_file_extracts_claude_codex_and_pi_prompt_answer_records`
  - `filter_new_keeps_new_items`
  - `filter_new_removes_existing_hashes`
  - `filter_new_empty_input_returns_empty`
  - `filter_new_all_duplicates_returns_empty`
  - `filter_new_deduplicates_within_input`
  - `identical_prompt_and_answer_text_have_distinct_history_hashes`
  - `identical_prompt_text_from_session_and_archive_has_distinct_history_hashes`
  - `filter_new_deduplicates_within_input`
  - `qdrant_history_filters_isolate_type_and_source`
  - `build_search_results_extracts_fields`
  - `build_search_results_empty_payload_graceful`
  - `build_search_results_empty_input`
- `src/main_tests.rs`
  - `search_requires_prompt_or_answer_type`
  - `search_accepts_prompt_type`
  - `search_accepts_answer_type`
- `src/extract.rs`
  - Claude/Pi user-message and assistant-message extraction tests.
  - Codex prompt/answer extraction and context-prelude filtering tests.
- `/syncthing/Sync/Provisioning/tests/pi-claude-memory-extension-qdrant.py`
  - Pi `session_shutdown` exact-path, failure-propagation, and Qdrant integration proof.

## Out of scope

- Legacy memory-unit runtime storage and retrieval; those paths are retired in [memory-units.md](memory-units.md).
- Friction-driven memory creation and notable-fact handling; see [friction-memory-creation.md](friction-memory-creation.md).
- KB Markdown retrieval through PageIndex; see [kb-page-index.md](kb-page-index.md). The former KB-to-memory-unit ingestion path is retired.
- Transcript outline PageIndex retrieval; it is a separate local navigation surface.
