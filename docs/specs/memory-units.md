# Legacy memory-unit compatibility

The former memory-unit runtime feature is retired. This specification records
the compatibility boundary for legacy memory-unit data still read by migration
and KB export; current prompt enrichment uses unified prompt/answer history and
KB PageIndex only. Runtime details belong in
[the memory-unit retirement wiki note](../wiki/systems/memory-units.md).

## What it must do

### Retired runtime

- [x] Expose no memory-unit search, listing, deletion, deduplication, or enrich runtime path.
- [x] Keep `deduplicate` out of the public CLI.
- [x] Keep the deleted `src/memory_unit.rs` and `src/dedup.rs` modules out of the runtime inventory.

### Compatibility readers

- [x] Allow KB export classification to read legacy `claude-memory-units` records.
- [x] Preserve legacy source, provenance, project, and category metadata needed for export/parity.
- [x] Allow migration/export workflows to retain legacy collection recognition without using it for prompt retrieval.
- [x] Make no collection- or point-deletion claim as part of this retirement.

### Current retrieval

- [x] Use unified prompt/answer history plus KB PageIndex for `enrich`.
- [x] Keep Transcript PageIndex as a separate CLI navigation surface.

## How it works

- [Memory-unit retirement](../wiki/systems/memory-units.md) describes the deleted runtime boundary.
- [KB Markdown export](kb-markdown-export.md) describes legacy record classification and canonical Markdown output.
- [Storage migration](storage-migration.md) describes legacy collection backup and parity behavior.
- [Prompt and answer history](prompt-answer-history.md) describes the active history retrieval surface.

## Implementation inventory

- `src/kb_export.rs` — compatibility classification and Markdown rendering for legacy memory-unit records.
- `src/bin/claude-memory-export-kb.rs` — guarded live export reader.
- `src/migration.rs` — legacy source classification and migration parity logic.
- `src/bin/claude-memory-migrate.rs` — guarded migration reader and backup workflow.

Deleted runtime modules are intentionally absent: `src/memory_unit.rs`,
`src/dedup.rs`, `src/graph.rs`, `src/graph/`, and `src/graph_cmds.rs`.

## Tests asserting this spec

- `tests/kb_export.rs` — legacy memory-unit classification, provenance, deduplication, quarantine, and safe export behavior.
- `src/migration.rs` — legacy source classification and parity behavior.
- `src/main_tests.rs` — retired command surface.

## Known gaps (current cycle)

None for the retirement boundary.

## Out of scope

- Reintroducing memory-unit runtime search, listing, deletion, deduplication, or enrich.
- Reintroducing graph runtime modules or graph commands.
- Deleting or migrating legacy Qdrant collections as part of this change.
