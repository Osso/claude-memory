The summary and duplicate-KB-vector retirement slice removes the obsolete KB-to-memory-unit ingestion surface while preserving the active PageIndex and history contracts. The current state and non-actions are documented in [the retirement wiki note](../wiki/systems/kb-summary-and-vector-retirement.md).

## What it must do

### Retired surface

- [x] Reject `claude-memory ingest-kb`; the CLI command is no longer part of the public surface.
- [x] Expose no replacement KB-to-memory-unit writer through the public CLI.
- [x] Leave manual memory, memory-unit, and notable-fact API behavior unchanged.

### Preserved retrieval and history

- [x] Preserve KB PageIndex build, query, document, structure, and exact content retrieval.
- [x] Preserve KB PageIndex refresh behavior for changed, added, and deleted Markdown files.
- [x] Preserve Transcript PageIndex node summaries; `PageIndexNode.summary` remains part of the node model.
- [x] Preserve prompt/answer history indexing and search as a separate surface.
- [x] Preserve all legacy Qdrant data; live before/after counts remain unchanged.

### Compatibility recognition

- [x] Keep legacy `source=summary` and `source=kb` recognition in export/migration classification for parity, even though neither is an active search or ingestion path.

No active summary producer or summary search path existed before this retirement. PageIndex node summaries are unrelated and remain active.

## How it works

- [docs/wiki/systems/kb-summary-and-vector-retirement.md](../wiki/systems/kb-summary-and-vector-retirement.md) describes the retired and preserved paths.
- [kb-page-index.md](kb-page-index.md) defines the KB retrieval contract.
- [memory-units.md](memory-units.md) defines the separate memory-unit contract.
- [prompt-answer-history.md](prompt-answer-history.md) defines transcript history search.
- [storage-migration.md](storage-migration.md) defines legacy source recognition during migration.

## Implementation inventory

- `src/main.rs` — no longer declares or dispatches `ingest-kb`.
- `src/indexing_cmds.rs` — no longer contains the KB ingestion handler.
- `src/lib.rs` — no longer exports `kb_ingest`.
- `src/kb_ingest.rs` — deleted obsolete KB-to-memory-unit implementation.
- `src/kb_search.rs` — preserved KB PageIndex implementation.
- `src/page_index.rs` — preserved transcript PageIndex node summaries.

## Tests asserting this spec

- `src/main_tests.rs::ingest_kb_command_is_retired`
- Existing KB PageIndex tests in `src/kb_search_tests.rs`, including persisted search and added/changed/deleted Markdown refresh.
- `src/page_index_tests.rs::summary_separates_text_turns_from_tool_calls`

## Known gaps (current cycle)

None for this slice.

## Out of scope

- Reintroducing KB Markdown facts as memory units.
- Removing or migrating legacy Qdrant collections or points.
- Changing manual memory, memory-unit, notable-fact, prompt/answer history, or PageIndex APIs.
- Treating `PageIndexNode.summary` as the retired summary-vector feature.
- Touching unrelated dirty changes in `src/analyze.rs` or `src/notable_fact.rs`.
