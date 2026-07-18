Prompt and answer history indexing stores raw user prompts and assistant responses from Claude, Codex, and Pi transcript files in one searchable session-history index. `claude-memory search <query>` runs one globally ranked prompt+answer query; `--type prompts|answers` is optional filtering. Implementation details belong in [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md).

## What it must do

### Storage and filtering

- [x] Store prompt and answer chunks in the `claude-session-history` collection.
- [x] Persist each chunk with its text, `type` (`prompt` or `answer`), `source` (`session` or `archive`), path, session id, and persisted hash.
- [x] Make `--type prompts` a `type=prompt` view and `--type answers` a `type=answer` view.
- [x] Allow typed searches to restrict results by `source`.
- [x] Derive persisted identity from history type, source, and content hash: identical text remains distinct across prompt/answer and session/archive views, while identical chunks within the same type/source collapse to one point.
- [x] Do not add message, turn, or chunk ordinals solely to preserve repeated identical chunks; duplicate collapse is intentional.
- [x] Return type, text, source, path, session id, and score without panicking on missing payload fields.

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

- [x] Run `claude-memory search <query>` as one globally ranked prompt+answer Qdrant query.
- [x] Accept optional `claude-memory search --type prompts <query>` filtering.
- [x] Accept optional `claude-memory search --type answers <query>` filtering.
- [x] Apply `--limit` globally to the selected query, including combined prompt+answer search.
- [x] Emit stable NDJSON with fields `type`, `text`, `source`, `path`, `session_id`, and `score` when `--json` is set.
- [x] Keep prompt and answer search CLI-only.
- [x] Return no session-history results when semantic search is disabled.
- [x] Keep KB PageIndex and Transcript PageIndex as separate surfaces; legacy memory-unit data is compatibility-only and KB-to-memory-unit ingestion is retired.

## How it works

- [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md) describes the shared collection, payload fields, extraction, hashing, deduplication, hybrid search, global search, and optional type/source filters.

## Implementation inventory

- `src/index.rs` — owns session-history collection setup, active/archive traversal, single-file indexing, and stats.
- `src/extract.rs` — extracts user and assistant transcript text and attaches history/source metadata.
- `src/index_writer.rs` — computes persisted identity, deduplicates chunks, builds payloads, and writes points.
- `src/index_search.rs` — applies optional history-type and source filters for CLI search.
- `src/main.rs` — declares and dispatches global or typed session-history search and NDJSON output.
- `src/indexing_cmds.rs` — dispatches index and single-file commands.
- `src/qdrant_hybrid.rs` — creates the shared dense+sparse collection and query vectors.

## Tests asserting this spec

- `src/index_tests.rs`
  - [x] `index_sources_discover_claude_codex_and_pi_sessions`
  - [x] `index_file_extracts_claude_codex_and_pi_prompt_answer_records`
  - [x] `filter_new_keeps_new_items`
  - [x] `filter_new_removes_existing_hashes`
  - [x] `filter_new_empty_input_returns_empty`
  - [x] `filter_new_all_duplicates_returns_empty`
  - [x] `filter_new_deduplicates_within_input`
  - [x] `identical_prompt_and_answer_text_have_distinct_history_hashes`
  - [x] `identical_prompt_text_from_session_and_archive_has_distinct_history_hashes`
  - [x] `qdrant_history_filters_isolate_type_and_source`
  - [x] `get_string_returns_value_for_known_key`
  - [x] `get_string_returns_empty_for_missing_key`
  - [x] `get_string_returns_empty_for_non_string_value`
  - [x] `get_string_returns_empty_for_null_kind`
  - [x] `build_search_results_extracts_fields`
  - [x] `build_search_results_empty_payload_graceful`
  - [x] `build_search_results_empty_input`
- `src/main_tests.rs`
  - [x] `search_defaults_to_combined_prompt_and_answer_history`
  - [x] `search_accepts_prompt_type`
  - [x] `search_accepts_answer_type`
  - [x] `search_accepts_json_output`
  - [x] `search_json_output_is_stable_ndjson_in_rank_order`
- `src/extract.rs`
  - [x] Claude/Pi user-message and assistant-message extraction tests.
  - [x] Codex prompt/answer extraction and context-prelude filtering tests.
- `/syncthing/Sync/Provisioning/tests/pi-claude-memory-extension-qdrant.py`
  - [x] Pi `session_shutdown` exact-path, failure-propagation, and Qdrant integration proof.

## Known gaps (current cycle)

- None.

## Out of scope

- Legacy memory-unit runtime storage and retrieval; those paths are retired in [memory-units.md](memory-units.md).
- Friction-driven memory creation and notable-fact handling; see [friction-memory-creation.md](friction-memory-creation.md).
- KB Markdown retrieval through PageIndex; see [kb-page-index.md](kb-page-index.md). The former KB-to-memory-unit ingestion path is retired.
- Transcript outline PageIndex retrieval; it is a separate local navigation surface.
