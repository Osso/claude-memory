# Legacy memory-unit retirement

The former memory-unit runtime feature and its migration/export compatibility
readers are retired. Current prompt enrichment uses unified prompt/answer
history and KB PageIndex only. Runtime details belong in [the memory-unit
retirement wiki note](../wiki/systems/memory-units.md).

## What it must do

### Retired runtime

- [x] Expose no memory-unit search, listing, deletion, deduplication, or enrich runtime path.
- [x] Keep `deduplicate` out of the public CLI.
- [x] Keep the deleted `src/memory_unit.rs` and `src/dedup.rs` modules out of the runtime inventory.

### Historical export and current storage

- [x] Complete the canonical durable-memory KB Markdown export before removing compatibility code.
- [x] Preserve the exported Markdown, manifest, and migration backups.
- [x] Keep Qdrant limited to `claude-session-history` after legacy collection retirement.

### Current retrieval

- [x] Use unified prompt/answer history plus KB PageIndex for `enrich`.
- [x] Keep Transcript PageIndex as a separate CLI navigation surface.

## How it works

- [Memory-unit retirement](../wiki/systems/memory-units.md) describes the deleted runtime boundary.
- [Prompt and answer history](prompt-answer-history.md) describes the active history retrieval surface.

## Implementation inventory

Deleted runtime modules are intentionally absent: `src/memory_unit.rs`,
`src/dedup.rs`, `src/graph.rs`, `src/graph/`, and `src/graph_cmds.rs`.

## Tests asserting this spec

- `src/main_tests.rs` — retired command surface.

## Known gaps (current cycle)

None for the retirement boundary.

## Out of scope

- Reintroducing memory-unit runtime search, listing, deletion, deduplication, or enrich.
- Reintroducing graph runtime modules or graph commands.
- Reintroducing the deleted migration/export compatibility surface.
