# Notable-fact ingestion

Notable-fact ingestion is retired. This specification records the removal
boundary after the KB export compatibility reader was removed; implementation
notes live in [the retirement wiki note](../wiki/systems/notable-fact-ingestion.md).

## What it must do

### Retired pipeline

- [x] Keep `analyze` and `backfill` out of the public CLI.
- [x] Do not write or retrieve notable-fact records at runtime.
- [x] Remove the notable-fact analyzer/writer module and graph-related runtime paths.

### Historical export and current storage

- [x] Complete the canonical durable-memory KB Markdown export before removing compatibility code.
- [x] Preserve the exported Markdown, manifest, and migration backups.
- [x] Keep Qdrant limited to `claude-session-history` after legacy collection retirement.

### Current retrieval

- [x] Use unified prompt/answer history plus KB PageIndex for prompt enrichment.
- [x] Keep Transcript PageIndex as a separate CLI navigation surface.

## How it works

- [docs/wiki/systems/notable-fact-ingestion.md](../wiki/systems/notable-fact-ingestion.md) describes the retired boundary.
- [prompt-answer-history.md](prompt-answer-history.md) describes active history retrieval.

## Implementation inventory


Deleted analyzer/writer and graph runtime modules are intentionally absent.

## Tests asserting this spec

- `src/main_tests.rs` — retired command surface.
## Known gaps (current cycle)

None for the retirement boundary.

## Out of scope

- Reintroducing notable-fact extraction, writes, or runtime retrieval.
- Reintroducing graph commands or graph enrichment.
- Reintroducing the deleted migration/export compatibility surface.
