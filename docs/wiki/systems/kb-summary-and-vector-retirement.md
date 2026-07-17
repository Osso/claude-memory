# KB Summary and Duplicate-Vector Retirement

This slice retires the obsolete KB-to-memory-unit ingestion path and the stale
`ingest-kb` CLI surface. KB Markdown remains available through KB PageIndex.

## Current state

- `src/kb_ingest.rs`, the `kb_ingest` library export, `run_ingest_kb`, and
  `claude-memory ingest-kb` are retired.
- No active summary producer or summary search path is part of retrieval.
- Transcript PageIndex `PageIndexNode.summary` remains node metadata, not a
  summary-vector path.
- KB PageIndex remains the structured exact-content KB retrieval path.
- Prompt/answer history remains a separate session-history surface.
- The transcript analyzer and notable-fact analyzer/writer are also retired;
  `analyze` and `backfill` are not available.

## Compatibility and preservation

Legacy readers retain source recognition for parity:

- migration recognizes `source=summary` and `source=kb` as unsupported history
  sources and skips them rather than migrating them;
- export recognizes legacy `source=summary` as non-durable and `source=kb` as an
  excluded KB vector, with manifest accounting.

The durable-memory KB export is completed. Existing exported Markdown remains
available through KB PageIndex. This slice does not write live notable facts,
delete points, or delete collections. Legacy Qdrant contents remain outside the
retirement boundary.

## Related contracts

- [KB PageIndex](../../specs/kb-page-index.md)
- [Memory units](../../specs/memory-units.md)
- [Prompt and answer history](../../specs/prompt-answer-history.md)
- [Storage migration](../../specs/storage-migration.md)
- [KB Markdown export](../../specs/kb-markdown-export.md)
