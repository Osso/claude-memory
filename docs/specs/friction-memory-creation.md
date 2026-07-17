# Friction-driven memory creation

## Status

The transcript analyzer and friction-memory creation pipeline are retired.
`claude-memory analyze <session_jsonl>` and `claude-memory backfill` are no
longer CLI commands. `src/analyze.rs` and `src/backfill.rs` were removed, so
completed transcripts are not automatically classified, validated, or written
as memory units by this pipeline.

This retirement does not delete the `claude-memory-units` collection or its
records. Existing memory-unit read, listing/deletion, deduplication, and enrich
paths remain. Prompt/answer history and both PageIndex surfaces remain active.

## Retired contract

The former contract classified assistant friction, extracted a short preload,
validated it against the eventual session resolution, retried failed candidates,
and stored accepted candidates with source metadata. Backfill walked live and
archived sessions with processed-session state. Those behaviors are historical
only; no active analyzer writer remains for them.

## Remaining surfaces

- `memory_unit.rs` retains memory-unit schema, collection access, search,
  listing/deletion, and deduplication support.
- `enrich` can retrieve memory units as possibly useful hints.
- Prompt/answer history remains a filtered view over
  `claude-session-history`.
- KB PageIndex remains the exact Markdown retrieval path.
- Transcript PageIndex remains CLI-only transcript navigation.
- The completed KB export and migration compatibility readers remain separate
  from transcript analysis.

See [retrieval flows](../wiki/systems/retrieval-flows.md),
[memory units](memory-units.md), and
[prompt/answer history](prompt-answer-history.md).
