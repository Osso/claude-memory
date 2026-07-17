# Notable-fact ingestion

Notable-fact ingestion is retired. This specification records the removal
boundary and the compatibility reader that remains for KB export; implementation
notes live in [the retirement wiki note](../wiki/systems/notable-fact-ingestion.md).

## What it must do

### Retired pipeline

- [x] Keep `analyze` and `backfill` out of the public CLI.
- [x] Do not write or retrieve notable-fact records at runtime.
- [x] Remove the notable-fact analyzer/writer module and graph-related runtime paths.

### Preserved compatibility

- [x] Allow completed KB export to read existing notable-fact records.
- [x] Preserve legacy source recognition and parity accounting for migration/export readers.
- [x] Make no claim that notable-fact Qdrant points or collections were deleted.

### Current retrieval

- [x] Use unified prompt/answer history plus KB PageIndex for prompt enrichment.
- [x] Keep Transcript PageIndex as a separate CLI navigation surface.

## How it works

- [docs/wiki/systems/notable-fact-ingestion.md](../wiki/systems/notable-fact-ingestion.md) describes the retired boundary.
- [kb-markdown-export.md](kb-markdown-export.md) describes the compatibility export.
- [storage-migration.md](storage-migration.md) describes legacy migration readers.
- [prompt-answer-history.md](prompt-answer-history.md) describes active history retrieval.

## Implementation inventory

- `src/kb_export.rs` — compatibility classification for legacy notable-fact records.
- `src/bin/claude-memory-export-kb.rs` — guarded export reader.

Deleted analyzer/writer and graph runtime modules are intentionally absent.

## Tests asserting this spec

- `tests/kb_export.rs` — legacy notable-fact classification and export behavior.
- Migration tests — legacy source recognition and parity behavior.
- `src/main_tests.rs` — retired command surface.

## Known gaps (current cycle)

None for the retirement boundary.

## Out of scope

- Reintroducing notable-fact extraction, writes, or runtime retrieval.
- Reintroducing graph commands or graph enrichment.
- Deleting legacy Qdrant collections or points.
