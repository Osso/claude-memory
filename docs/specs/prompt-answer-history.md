Prompt and answer history indexing stores raw user prompts separately from assistant responses so each can be searched for its own purpose. Prompt history answers "what did I ask or discuss?", while answer history answers "how did the assistant solve or explain something?"; implementation details belong in [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md).

## What it must do

### Data separation

- [ ] Store user prompts in a prompt/search collection separate from assistant answers.
- [ ] Store assistant responses in an answer/search collection separate from user prompts.
- [ ] Preserve source metadata for each indexed chunk, including source kind and source path.
- [ ] Preserve session identity when it is available.
- [ ] Keep KB Markdown prompt-search indexing distinct from KB PageIndex retrieval.

### Indexing lifecycle

- [x] Skip chunks whose content hash already exists in the target collection.
- [x] Deduplicate repeated chunks within the same indexing input.
- [ ] Index active Claude session prompts.
- [ ] Index archived Claude session prompts.
- [ ] Index active Claude session answers.
- [ ] Index archived Claude session answers.
- [ ] Index project summaries into prompt search.
- [ ] Index Markdown KB chunks into prompt search as legacy/vector-search fallback.
- [ ] Support indexing a single conversation file into both prompt and answer collections.

### CLI and MCP search

- [x] Default `claude-memory search <query>` targets memories, not prompt or answer history.
- [x] Accept `claude-memory search --type prompts <query>`.
- [x] Accept `claude-memory search --type answers <query>`.
- [ ] MCP `prompt_search` searches prompt/question/legacy-KB history.
- [ ] MCP `answer_search` searches assistant response history.
- [x] Search results include text, source, path, and score.
- [x] Empty or missing payload fields are handled without panics.

## How it works

- [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md) describes collection layout, extraction, chunking, hybrid search, and MCP filtering.

## Implementation inventory

- `src/index.rs` — owns prompt and answer collection setup, indexing, dedup-by-hash, search, stats, and single-file indexing.
- `src/extract.rs` — extracts user-message chunks, assistant-message chunks, summaries, archives, and Markdown chunks.
- `src/index/search_results.rs` — converts Qdrant scored points into public search results.
- `src/index_tests.rs` — tests chunk deduplication and search-result payload conversion.
- `src/main.rs` — declares and dispatches CLI search targets and indexing commands.
- `src/main_tests.rs` — tests CLI parsing for prompt and answer search targets.
- `src/bin/mcp.rs` — exposes `prompt_search` and `answer_search` MCP tools.
- `src/qdrant_hybrid.rs` — creates dense+sparse hybrid Qdrant collections and vectors.
- `README.md` — documents prompt and answer search commands.

## Tests asserting this spec

- `src/main_tests.rs`
  - `search_defaults_to_memories`
  - `search_accepts_prompt_type`
  - `search_accepts_answer_type`
- `src/index_tests.rs`
  - `filter_new_keeps_new_items`
  - `filter_new_removes_existing_hashes`
  - `filter_new_empty_input_returns_empty`
  - `filter_new_all_duplicates_returns_empty`
  - `filter_new_deduplicates_within_input`
  - `build_search_results_maps_payload_fields`
  - `build_search_results_empty_payload_graceful`
  - `build_search_results_empty_input`
- `src/extract.rs`
  - user-message and assistant-message extraction tests in the module test suite.

## Known gaps (current cycle)

- [ ] Add integration tests proving prompt and answer collections receive different content from the same fixture session.
- [ ] Add CLI or MCP tests proving prompt and answer searches target the intended collection.
- [ ] Decide whether legacy KB chunks should remain in prompt search now that KB PageIndex exists.

## Out of scope

- Memory-unit storage and retrieval; see [memory-units.md](memory-units.md).
- Friction-driven memory creation; see [friction-memory-creation.md](friction-memory-creation.md).
- KB PageIndex retrieval; see [kb-page-index.md](kb-page-index.md).
