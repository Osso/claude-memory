# Notable-fact ingestion

## Status

Notable-fact ingestion is retired. The analyzer and writer module were removed
with the transcript analyzer deletion. `analyze` and `backfill` no longer exist,
and no active notable-fact writer remains.

Existing durable notable-fact records remain readable by the completed
`claude-memory-export-kb` compatibility path. The export writes canonical
Markdown under `/syncthing/Sync/KB/memory/notable-facts/`. Migration/export
readers retain legacy source recognition for parity. This change does not delete
notable-fact points or collections.

## Retired contract

The former contract extracted project facts from session traversal, stored them
in `claude-notable-facts`, and deduplicated sightings. That pipeline is no
longer an active feature. These are historical design requirements, not current
runtime behavior:

- session traversal for notable-fact extraction;
- almanac-style fact prompting and JSON parsing;
- notable-fact collection writes and write-time merge handling;
- analyzer and backfill integration;
- notable-fact retrieval or prompt enrichment.

## Remaining surfaces

- Completed KB Markdown export can read existing notable-fact records.
- Legacy migration/export compatibility readers remain available.
- Prompt/answer history remains in `claude-session-history`.
- KB PageIndex and transcript PageIndex remain separate retrieval surfaces.
- Memory-unit read, deduplication, and enrich paths remain separate from this
  retired pipeline.

See [retrieval flows](../wiki/systems/retrieval-flows.md),
[KB Markdown export](kb-markdown-export.md), and
[storage migration](storage-migration.md).
