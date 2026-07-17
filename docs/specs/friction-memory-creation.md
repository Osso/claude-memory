# Friction-driven memory creation

The transcript analyzer and friction-memory creation pipeline are retired.
This specification records the removal boundary; implementation notes live in
[the retirement wiki note](../wiki/systems/friction-memory-creation.md).

## What it must do

### Retired pipeline

- [x] Reject the retired `claude-memory analyze` and `claude-memory backfill` command paths.
- [x] Do not classify completed transcripts or write new memory-unit records automatically.
- [x] Remove the analyzer, backfill, and memory-unit runtime retrieval/deduplication paths.

### Historical export and current storage

- [x] Complete the canonical durable-memory KB Markdown export before removing compatibility code.
- [x] Preserve the exported Markdown, manifest, and migration backups.
- [x] Keep Qdrant limited to `claude-session-history` after legacy collection retirement.

### Current retrieval

- [x] Use unified prompt/answer history plus KB PageIndex for prompt enrichment.
- [x] Keep Transcript PageIndex as a separate CLI navigation surface.

## How it works

- [docs/wiki/systems/friction-memory-creation.md](../wiki/systems/friction-memory-creation.md) describes the retired boundary.
- [memory-units.md](memory-units.md) describes the retired memory-unit boundary.
- [prompt-answer-history.md](prompt-answer-history.md) describes active history retrieval.
- [kb-page-index.md](kb-page-index.md) describes active KB retrieval.

## Implementation inventory

No analyzer, backfill, memory-unit runtime, deduplication, migration, or export
compatibility module remains.

## Tests asserting this spec

- `src/main_tests.rs` — retired command surface.

## Known gaps (current cycle)

None for the retirement boundary.

## Out of scope

- Reintroducing automatic friction analysis or memory-unit writes.
- Reintroducing memory-unit or graph runtime retrieval.
- Reintroducing the deleted migration/export compatibility surface.
