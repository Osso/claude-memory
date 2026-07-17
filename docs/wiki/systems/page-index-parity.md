# PageIndex Parity

This project matches the useful PageIndex retrieval architecture from
`VectifyAI/PageIndex` commit `f50e529`, not the full Python implementation. The
local implementation is scoped to Markdown KB notes and Claude/Codex transcript
history.

## Document Model

KB PageIndex and Transcript PageIndex use the same nested document shape:

- document metadata identifies the source, family, title, description, and size
- structure output returns node ids, titles, summaries, locators, and children
  without full text
- content output fetches exact text by node id or source range
- query output returns traceable document/node references plus a follow-up
  content command

The persistent KB model lives in `src/kb_search.rs`. The transcript model lives
in `src/page_index.rs`.

## Retrieval Flow

The intended retrieval loop mirrors PageIndex:

1. Search for candidate documents or nodes.
2. Inspect document metadata.
3. Inspect document structure without dumping all content.
4. Fetch tight node or range content.
5. Answer from fetched content with auditable references.

`src/page_index_agentic.rs` contains the shared tree-walk abstraction. Lexical
query mode remains available as a deterministic fallback and debugging path.

## Surfaces

KB PageIndex is exposed through `claude-memory kb-page-index` and may be used by
`claude-memory enrich` under tight output caps. Transcript PageIndex is exposed
through `claude-memory transcript-page-index` and remains CLI-only.

The two surfaces intentionally stay separate:

- KB PageIndex is a structured KB retrieval surface.
- Transcript PageIndex is a source-inspection surface for Claude/Codex sessions.
- No active runtime path writes durable transcript-derived memory units;
  legacy records are compatibility-only.

## Bounded Parity

The local implementation does not include PDF parsing, OCR, PageIndex cloud/API
compatibility, FinanceBench claims, or corpus-level filesystem routing. Full
transcript corpus indexing is still too large for routine use; the benchmark in
`docs/benchmarks/page-index-2026-05-10.md` records the current cost and quality
tradeoffs.
