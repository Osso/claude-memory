# Memory units

Memory-unit runtime retrieval is retired. `memory_unit.rs`, the dedicated
runtime search/listing/deletion path, and `dedup.rs` were deleted. `enrich` no
longer searches or injects memory-unit records.

The canonical durable-memory KB Markdown export completed before the legacy
compatibility code was removed. Its Markdown and manifest remain available
through KB PageIndex, and migration backups exist. The former migration/export
compatibility readers and legacy collections were retired; Qdrant now contains
only `claude-session-history`.

The current runtime retrieval path is unified prompt/answer history plus KB
PageIndex. Transcript PageIndex remains a separate CLI navigation surface.

This page records the retired runtime boundary, not an active memory-unit API.
