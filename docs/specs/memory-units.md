Memory units are compact, durable preloads stored separately from raw prompt and answer history. They represent analyzer-created preloads that should be retrieved for prompt enrichment; implementation details belong in [docs/wiki/systems/memory-units.md](../wiki/systems/memory-units.md). Manual project memories are kept as editable local docs under `docs/local/`, not inserted into memory units.

## What it must do

### Storage model

- [x] Carry a source vocabulary field for each memory unit.
- [ ] Store memory units in a dedicated collection separate from prompt and answer history.
- [ ] Store memory text, creation time, source, source session, source turn, optional category, project scope, and seen-in-sessions metadata.
- [ ] Use the same hybrid vector format as other searchable collections.
- [ ] Deduplicate similar memory units at write time.
- [ ] Merge duplicate sightings by appending to `seen_in_sessions` instead of inserting another point.

### Manual memory location

- [x] Manual project memories are written to `docs/local/memory.md`.
- [x] Cross-project agent behavior is written to `/home/osso/AgentConfig/rules`.

### Stored memory metadata

- [x] Stored memory listing reads the source from payload.
- [ ] Listed memory units include seen count and source metadata.

### Retrieval and enrich

- [x] Search-result conversion uses payload source and source session.
- [ ] Semantic memory-unit search is gated by search configuration.
- [ ] Enrich includes memory-unit results above the configured relevance floor.
- [ ] Enrich labels memory-unit results as possibly useful hints, not authoritative facts.

## How it works

- [docs/wiki/systems/memory-units.md](../wiki/systems/memory-units.md) describes schema, collection lifecycle, dedup-at-write, manual memory locations, listing, and enrich retrieval.

## Implementation inventory

- `src/memory_unit.rs` — defines the memory-unit schema, collection, write deduplication, search, listing, filtering, and deletion.
- `src/main.rs` — dispatches analysis, backfill, and session-history CLI search commands.
- `src/bin/mcp.rs` — exposes MCP `prompt_search` and `answer_search` tools.
- `src/analyze.rs` — writes validated friction-derived memory units.
- `src/kb_search.rs` — remains the KB PageIndex retrieval path; it does not write memory units.
- `src/enrich_cmd.rs` — retrieves memory units for prompt enrichment.
- `src/qdrant_hybrid.rs` — creates the hybrid collection layout.

## Tests asserting this spec

- `src/memory_unit.rs`
  - `memory_unit_has_existing_source_vocabulary_field`
  - `stored_memory_reads_source_from_payload`
  - `search_result_uses_payload_source`
  - `manual_memory_filter_matches_category_and_project`
- `src/bin/mcp.rs`
- `src/main_tests.rs`

## Known gaps (current cycle)

- [ ] Add tests for dedup-at-write merge behavior and `seen_in_sessions` updates.
- [ ] Add tests for memory deletion by numeric point ID.
- [ ] Add a dedicated regression test for the retired KB-to-memory-unit path; CLI retirement coverage is listed in [kb-summary-and-vector-retirement.md](kb-summary-and-vector-retirement.md).

## Out of scope

- Raw prompt and answer history search; see [prompt-answer-history.md](prompt-answer-history.md).
- The friction pipeline that creates automatic memory units; see [friction-memory-creation.md](friction-memory-creation.md).
- KB PageIndex retrieval; see [kb-page-index.md](kb-page-index.md). KB Markdown-to-memory-unit ingestion is retired and documented in [kb-summary-and-vector-retirement.md](kb-summary-and-vector-retirement.md).
