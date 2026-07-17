# Memory units

Memory-unit runtime retrieval is retired. `memory_unit.rs`, the dedicated
runtime search/listing/deletion path, and `dedup.rs` were deleted. `enrich` no
longer searches or injects memory-unit records.

Legacy memory-unit records remain relevant only to compatibility readers used by
KB export and migration/parity workflows. Those readers preserve source
classification and export behavior where required; this retirement does not
claim deletion of the `claude-memory-units` collection or its records.

The current runtime retrieval path is unified prompt/answer history plus KB
PageIndex. Transcript PageIndex remains a separate CLI navigation surface.

This page records the retired runtime boundary, not an active memory-unit API.
