# Memory units

Memory units are compact durable preloads stored separately from raw prompt and
answer history. The former friction analyzer is retired; this spec covers the
remaining storage, read, deduplication, and enrich paths. Manual project
memories remain editable Markdown under `docs/local/` and are not inserted into
memory units.

## Current contract

- Memory units remain in the dedicated `claude-memory-units` collection.
- Stored units retain text, creation time, source, source session, source turn,
  optional category/project, and `seen_in_sessions` metadata.
- Existing units can be searched, listed, filtered, deleted, and merged through
  the remaining deduplication path.
- `enrich` retrieves relevant memory units when semantic search is enabled and
  labels them as possibly useful hints, not authoritative facts.
- No active transcript analyzer populates memory units automatically.
- This retirement does not delete the collection or its records.

## Manual memory location

- Project-specific durable context: `docs/local/memory.md`.
- Cross-project agent behavior: `/home/osso/AgentConfig/rules`.
- Manual Qdrant memory writes are disabled by the runtime guidance.

## Retrieval and related surfaces

Memory-unit search is gated by `[search].enabled`. Prompt/answer history remains
in `claude-session-history`. KB PageIndex retrieves canonical Markdown, and
transcript PageIndex navigates raw sessions. The completed KB export and
migration/export compatibility readers are separate from memory-unit enrich.

## Implementation inventory

- `src/memory_unit.rs` — schema, collection access, search, listing, deletion,
  payload conversion, and deduplication support.
- `src/enrich_cmd.rs` — retrieves memory units for prompt enrichment.
- `src/dedup.rs` — maintains the remaining deduplication command path.
- `src/kb_search.rs` — KB PageIndex retrieval; it does not write memory units.
- `src/qdrant_hybrid.rs` — hybrid collection layout.

## Out of scope

- Retired friction-driven memory creation; see
  [friction-memory-creation.md](friction-memory-creation.md).
- Retired notable-fact ingestion; see
  [notable-fact-ingestion.md](notable-fact-ingestion.md).
- Raw prompt and answer history; see
  [prompt-answer-history.md](prompt-answer-history.md).
- KB-to-memory-unit ingestion; see
  [kb-summary-and-vector-retirement.md](kb-summary-and-vector-retirement.md).
