# KB Summary and Duplicate-Vector Retirement

This slice retires the obsolete KB-to-memory-unit ingestion path and the stale
`ingest-kb` CLI surface. KB Markdown remains available through KB PageIndex.

## Current state

- `src/kb_ingest.rs`, the `kb_ingest` library export, `run_ingest_kb`, and
  `claude-memory ingest-kb` are retired.
- No active summary producer or summary search path is part of retrieval.
- Transcript PageIndex `PageIndexNode.summary` remains node metadata, not a
  summary-vector path.
- KB PageIndex remains the structured exact-content KB retrieval path and is the second `enrich` source.
- Prompt/answer history remains a separate session-history surface and is the history source used by `enrich`.
- The transcript analyzer and notable-fact analyzer/writer are also retired;
  `analyze` and `backfill` are not available.

## Historical export and current storage

The canonical durable-memory KB Markdown export completed before the
compatibility code was removed. Existing Markdown and its manifest remain
available through KB PageIndex, and migration backups exist.

The legacy collection migration/export binaries, modules, and tests were deleted
after the legacy collections were retired. Qdrant now contains only
`claude-session-history`.

## Related contracts

- [KB PageIndex](../../specs/kb-page-index.md)
- [Memory units](../../specs/memory-units.md)
- [Prompt and answer history](../../specs/prompt-answer-history.md)
