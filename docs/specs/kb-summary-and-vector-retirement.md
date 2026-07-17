The summary and duplicate-KB-vector retirement slice removes the obsolete KB-to-memory-unit ingestion surface while preserving the active PageIndex and history contracts. The current state and non-actions are documented in [the retirement wiki note](../wiki/systems/kb-summary-and-vector-retirement.md).

## What it must do

### Retired surface

- [x] Reject `claude-memory ingest-kb`; the CLI command is no longer part of the public surface.
- [x] Expose no replacement KB-to-memory-unit writer through the public CLI.
- [x] Leave no memory-unit, graph, migration, or export runtime reader/dedup/enrich path available.

### Preserved retrieval and history

- [x] Preserve KB PageIndex build, query, document, structure, and exact content retrieval.
- [x] Preserve KB PageIndex refresh behavior for changed, added, and deleted Markdown files.
- [x] Preserve Transcript PageIndex node summaries; `PageIndexNode.summary` remains part of the node model.
- [x] Preserve prompt/answer history indexing and search as a separate surface.
- [x] Preserve all legacy Qdrant data; live before/after counts remain unchanged.

### Historical export and current storage

- [x] Complete the canonical durable-memory KB Markdown export before removing compatibility code.
- [x] Preserve the exported Markdown, manifest, and migration backups.
- [x] Keep Qdrant limited to `claude-session-history` after legacy collection retirement.

No active summary producer or summary search path existed before this retirement. PageIndex node summaries are unrelated and remain active. Memory-unit, migration, and export compatibility code is deleted.

## How it works

- [docs/wiki/systems/kb-summary-and-vector-retirement.md](../wiki/systems/kb-summary-and-vector-retirement.md) describes the retired and preserved paths.
- [kb-page-index.md](kb-page-index.md) defines the KB retrieval contract.
- [memory-units.md](memory-units.md) defines the legacy memory-unit compatibility boundary.
- [prompt-answer-history.md](prompt-answer-history.md) defines transcript history search.

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
- Reintroducing the deleted migration/export compatibility surface.
- Reintroducing manual-memory, memory-unit, graph, prompt/answer history, or PageIndex runtime paths.
- Treating `PageIndexNode.summary` as the retired summary-vector feature.
- Treating the completed export as an active runtime command.
