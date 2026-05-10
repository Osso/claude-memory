Friction-driven memory creation analyzes completed transcripts to create memory units only when a past assistant struggle reveals a durable preload that would have avoided the struggle. It is the automatic memory creation path for transcript mining; implementation details belong in [docs/wiki/systems/friction-memory-creation.md](../wiki/systems/friction-memory-creation.md).

## What it must do

### Friction detection

- [ ] Read Claude session turns from JSONL or compressed archive files.
- [ ] Classify candidate turns for assistant friction before attempting memory extraction.
- [ ] Treat no-friction turns as `NoFriction` outcomes.
- [ ] Preserve enough transcript context for the classifier to judge the target turn.

### Candidate extraction and validation

- [ ] Extract a 1-3 sentence preload candidate for flagged friction turns.
- [ ] Allow the extractor to return null when no useful preload exists.
- [ ] Replay the original user prompt with the candidate preload as background context.
- [ ] Judge replay correctness against the eventual session resolution.
- [ ] Retry extraction with judge feedback up to three attempts before discarding.
- [ ] Discard candidates that fail validation instead of storing provisional memories.

### Storage outcomes

- [x] Classify live JSONL sessions as memory-unit source `session`.
- [x] Classify `.jsonl.zst` archives as memory-unit source `archive`.
- [ ] Store validated candidates as memory units with source session and source turn metadata.
- [ ] Mark stored outcomes as deduped when write-time dedup merges with an existing unit.
- [ ] Report discarded outcomes with a reason.

### Backfill and CLI operation

- [ ] `claude-memory analyze <session_jsonl>` runs the analyzer over one session file.
- [ ] Backfill walks sessions, skips sessions below the minimum user-turn threshold, and records processed session IDs.
- [ ] Backfill can include compressed archives when an archive directory is provided.
- [ ] Analyzer failures during backfill do not erase the processed-state file.

## How it works

- [docs/wiki/systems/friction-memory-creation.md](../wiki/systems/friction-memory-creation.md) describes the classifier, extractor, replay, judge, retry loop, storage, and backfill flow.

## Implementation inventory

- `src/analyze.rs` — orchestrates friction classification, extraction, replay, correctness judging, retry, and memory-unit storage.
- `src/backfill.rs` — walks session files, applies minimum-turn filtering, and persists processed-session state.
- `src/extract.rs` — reads session turns used by the analyzer and backfill.
- `src/llm.rs` — provides LLM completion and JSON/index helper behavior used by analyzer stages.
- `src/memory_unit.rs` — stores validated preload candidates through dedup-at-write.
- `src/main.rs` — exposes `analyze` and `backfill` CLI commands and prints analyzer outcomes.

## Tests asserting this spec

- `src/analyze.rs`
  - `memory_unit_source_uses_session_for_live_jsonl`
  - `memory_unit_source_uses_archive_for_zst_jsonl_archive`

## Known gaps (current cycle)

- [ ] Add deterministic tests for the classifier/extractor/replay/judge orchestration with fake LLM responses.
- [ ] Add tests for the three-attempt retry and discard behavior.
- [ ] Add tests proving validated candidates are written with correct source session and source turn metadata.
- [ ] Add backfill tests for processed-state persistence and minimum-turn filtering.
- [ ] Reconcile the implementation with the original locked design if efficiency judging remains disabled.

## Out of scope

- Manual memory writes; see [memory-units.md](memory-units.md).
- Raw prompt and answer indexing; see [prompt-answer-history.md](prompt-answer-history.md).
- KB PageIndex retrieval; see [kb-page-index.md](kb-page-index.md).
- Transcript outline PageIndex retrieval; it may help locate source sessions but does not create durable memory units.
