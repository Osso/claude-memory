Notable fact ingestion extracts durable project understanding from existing session traversal and stores it separately from operational memory-unit preloads. It adapts codealmanac's absorb model for claude-memory: sessions are raw material, and the output is mergeable project memory facts rather than wiki pages.

## What it must do

### Extraction contract

- [ ] Reuse existing session/backfill traversal instead of adding new transcript discovery.
- [x] Use an almanac-style prompt frame based on purpose, notability, and absorb operation rules.
- [x] Treat sessions as raw material to distill, not logs to summarize.
- [ ] Output zero or more durable notable facts.
- [ ] Reject progress logs, file summaries, generic docs, unsupported guesses, and obvious one-file facts.

### Storage model

- [x] Store notable facts in a dedicated collection separate from memory-unit preloads.
- [x] Store fact text, created time, source, source session, optional project, topic tags, and seen-in-sessions metadata.
- [ ] Deduplicate similar notable facts at write time.
- [ ] Merge duplicate sightings by appending to `seen_in_sessions`.

### Pipeline behavior

- [ ] Run notable fact ingestion as a separate analyzer stage from friction-memory creation.
- [ ] Keep memory-unit extraction focused on operational shortcuts.
- [ ] Do not require friction to be present before notable facts can be extracted.

## How it works

- [docs/wiki/systems/notable-fact-ingestion.md](../wiki/systems/notable-fact-ingestion.md)

## Implementation inventory

- `src/notable_fact.rs` — notable fact schema, prompt contract, JSON parsing, collection lifecycle, and write deduplication.
- `src/analyze.rs` — invokes notable fact ingestion over the already parsed session turns.
- `src/backfill.rs` — indirectly reuses notable fact ingestion through `analyze_session`.

## Tests asserting this spec

- `src/notable_fact.rs`
  - `absorb_prompt_treats_sessions_as_raw_material`
  - `parse_notable_fact_json_keeps_project_and_topics`
  - `notable_fact_collection_is_separate_from_memory_units`
  - `notable_fact_payload_records_merge_metadata`

## Known gaps (current cycle)

- [ ] Wire notable fact extraction into analyzer session processing.
- [ ] Add retrieval/enrich support for notable facts after the storage path is proven.

## Out of scope

- Replacing memory units; notable facts are a separate durable memory surface.
- Markdown wiki page generation; claude-memory stores facts, not `.almanac/` pages.
