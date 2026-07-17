# KB Summary and Duplicate-Vector Retirement

This slice retires the obsolete KB-to-memory-unit ingestion path and removes the stale `ingest-kb` CLI surface. It does not replace KB retrieval: KB Markdown remains available through KB PageIndex.

## Current state

- `src/kb_ingest.rs`, the `kb_ingest` library export, `run_ingest_kb`, and `claude-memory ingest-kb` are retired.
- No active summary producer or summary search path was present before this slice. This is retirement of an obsolete/duplicate surface, not a change to a running summary pipeline.
- Transcript PageIndex `PageIndexNode.summary` remains. It is node metadata used for transcript outlines and retrieval, not the retired summary-vector path.
- KB PageIndex remains the structured, exact-content KB retrieval path and remains available to `enrich`.
- Prompt/answer history remains a separate session-history surface.

## Compatibility and preservation

Legacy readers retain source recognition for parity:

- migration recognizes `source=summary` and `source=kb` as unsupported history sources and skips them rather than migrating them;
- export recognizes legacy `source=summary` as non-durable and `source=kb` as an excluded KB vector, with manifest accounting.

This slice does not write live data, delete points or collections, or alter legacy Qdrant contents. Manual memory, memory-unit, notable-fact, prompt/answer history, and collection APIs remain outside the retirement.

## Related contracts

- [KB PageIndex](../../specs/kb-page-index.md)
- [Memory units](../../specs/memory-units.md)
- [Prompt and answer history](../../specs/prompt-answer-history.md)
- [Storage migration](../../specs/storage-migration.md)
- [KB Markdown export](../../specs/kb-markdown-export.md)
