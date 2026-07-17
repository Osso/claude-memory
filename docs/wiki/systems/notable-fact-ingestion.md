# Notable-fact ingestion

Notable-fact ingestion is retired. The analyzer and writer module were removed
with the transcript analyzer deletion. `analyze` and `backfill` no longer exist,
and no active notable-fact writer or runtime retrieval path remains.

The canonical durable-memory KB Markdown export completed before the
compatibility code was removed. Its Markdown and manifest remain available
through KB PageIndex, and migration backups exist. The former migration/export
readers and legacy collections were retired. Qdrant now contains only
`claude-session-history`.

Prompt/answer history plus KB PageIndex are the active prompt-enrichment
sources. Transcript PageIndex remains a separate CLI navigation surface.

This page records the retirement boundary; it is not an active extraction
contract.
