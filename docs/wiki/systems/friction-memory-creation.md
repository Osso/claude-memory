# Friction-driven memory creation

Friction-driven transcript analysis is retired. `claude-memory analyze` and
`claude-memory backfill` are unavailable, and the analyzer modules were removed.
Completed transcripts are not automatically classified, validated, or written
as memory units by this path.

The memory-unit runtime search, listing/deletion, enrich, and deduplication
paths are also retired. Existing legacy records remain readable only where KB
export or migration/parity compatibility requires them. This retirement does
not claim deletion of the `claude-memory-units` collection or its records.

Prompt/answer history plus KB PageIndex are the active prompt-enrichment
sources. Transcript PageIndex remains a separate CLI navigation surface.

This page records the retired pipeline boundary; it is not a current analyzer
contract.
