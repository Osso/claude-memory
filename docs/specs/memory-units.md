Memory units are compact, durable preloads stored separately from raw prompt and answer history. They represent manual memories and analyzer-created preloads that should be retrieved for prompt enrichment; implementation details belong in [docs/wiki/systems/memory-units.md](../wiki/systems/memory-units.md).

## What it must do

### Storage model

- [x] Carry a source vocabulary field for each memory unit.
- [ ] Store memory units in a dedicated collection separate from prompt and answer history.
- [ ] Store memory text, creation time, source, source session, source turn, optional category, project scope, and seen-in-sessions metadata.
- [ ] Use the same hybrid vector format as other searchable collections.
- [ ] Deduplicate similar memory units at write time.
- [ ] Merge duplicate sightings by appending to `seen_in_sessions` instead of inserting another point.

### Manual memory writes

- [ ] MCP `memory_write` writes manual memories into memory units, not prompt history.
- [ ] CLI `memory-write` writes manual memories into memory units.
- [ ] Manual memories use source `memory`, source session `manual`, and source turn `0`.
- [ ] Manual memory writes accept optional category metadata and require explicit project scope through MCP and CLI.
- [ ] Manual memory writes use `__global__` as the explicit sentinel for global memories.
- [ ] Empty manual memory text is rejected.

### Listing, filtering, and deletion

- [x] Stored memory listing reads the source from payload.
- [x] Manual memory filters match exact category and project values.
- [ ] Memory listing supports category and project filters.
- [ ] Memory deletion removes a memory unit by numeric point ID.
- [ ] Listed memory units include seen count and source metadata.

### Retrieval and enrich

- [x] Search-result conversion uses payload source and source session.
- [ ] Semantic memory-unit search is gated by search configuration.
- [x] When semantic search is disabled, CLI memory search falls back to substring listing.
- [ ] Enrich includes memory-unit results above the configured relevance floor.
- [ ] Enrich labels memory-unit results as possibly useful hints, not authoritative facts.

## How it works

- [docs/wiki/systems/memory-units.md](../wiki/systems/memory-units.md) describes schema, collection lifecycle, dedup-at-write, manual writes, listing, and enrich retrieval.

## Implementation inventory

- `src/memory_unit.rs` — defines the memory-unit schema, collection, write deduplication, search, listing, filtering, and deletion.
- `src/main.rs` — exposes CLI memory search, memory write, and memory delete commands.
- `src/bin/mcp.rs` — exposes MCP `memory_write` and `memory_list`, and stores manual memories as units.
- `src/analyze.rs` — writes validated friction-derived memory units.
- `src/kb_ingest.rs` — writes KB-derived facts as memory units when the experimental ingester is used.
- `src/enrich_cmd.rs` — retrieves memory units for prompt enrichment.
- `src/qdrant_hybrid.rs` — creates the hybrid collection layout.

## Tests asserting this spec

- `src/memory_unit.rs`
  - `memory_unit_has_existing_source_vocabulary_field`
  - `stored_memory_reads_source_from_payload`
  - `search_result_uses_payload_source`
  - `global_project_scope_normalizes_to_unscoped_memory`
  - `blank_project_scope_is_rejected`
  - `manual_memory_filter_matches_category_and_project`
- `src/main_tests.rs`
  - `memory_write_requires_project_scope`
  - `memory_write_accepts_explicit_project_scope`
  - `memory_search_uses_semantic_query_when_enabled`
  - `memory_search_falls_back_to_substring_when_disabled`

## Known gaps (current cycle)

- [ ] Add tests proving MCP `memory_write` and CLI `memory-write` create memory units with the expected payload.
- [ ] Add tests for dedup-at-write merge behavior and `seen_in_sessions` updates.
- [ ] Add tests for memory deletion by numeric point ID.
- [ ] Decide whether KB-derived facts should remain in memory units now that KB PageIndex exists.

## Out of scope

- Raw prompt and answer history search; see [prompt-answer-history.md](prompt-answer-history.md).
- The friction pipeline that creates automatic memory units; see [friction-memory-creation.md](friction-memory-creation.md).
- KB PageIndex retrieval; see [kb-page-index.md](kb-page-index.md).
