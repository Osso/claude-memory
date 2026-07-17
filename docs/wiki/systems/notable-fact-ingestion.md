# Notable-fact ingestion

Notable-fact ingestion is retired. `src/notable_fact.rs`, the transcript
analyzer integration, and the `analyze`/`backfill` command path were removed.
No active notable-fact writer remains.

Existing records are not deleted. The completed KB Markdown export can read
legacy notable-fact records for canonical export and parity checks. Migration
and export compatibility readers retain legacy source recognition. Prompt/answer
history, KB PageIndex, transcript PageIndex, and memory-unit retrieval remain
separate active surfaces.

This page records the retirement boundary; it is not an active extraction
contract.
