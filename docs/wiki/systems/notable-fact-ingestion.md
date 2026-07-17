# Notable-fact ingestion

Notable-fact ingestion is retired. The analyzer and writer module were removed
with the transcript analyzer deletion. `analyze` and `backfill` no longer exist,
and no active notable-fact writer or runtime retrieval path remains.

Existing notable-fact records remain readable by the completed
`claude-memory-export-kb` compatibility path. Migration/export readers retain
legacy source recognition for parity. This change does not claim deletion of
notable-fact points or collections.

Prompt/answer history plus KB PageIndex are the active prompt-enrichment
sources. Transcript PageIndex remains a separate CLI navigation surface.

This page records the retirement boundary; it is not an active extraction
contract.
