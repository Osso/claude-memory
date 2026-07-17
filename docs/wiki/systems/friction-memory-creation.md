# Friction-driven memory creation

Friction-driven transcript analysis is retired. `claude-memory analyze` and
`claude-memory backfill` are unavailable, and the analyzer modules were removed.
Completed transcripts are not automatically classified, validated, or written
as memory units by this path.

The memory-unit runtime search, listing/deletion, enrich, and deduplication
paths are also retired. The former migration/export compatibility readers and
legacy collections were retired after the canonical KB Markdown export
completed. Qdrant now contains only `claude-session-history`.

Prompt/answer history plus KB PageIndex are the active prompt-enrichment
sources. Transcript PageIndex remains a separate CLI navigation surface.

This page records the retired pipeline boundary; it is not a current analyzer
contract.
