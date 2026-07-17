# PageIndex Parity

This project matches the useful PageIndex retrieval architecture from
`VectifyAI/PageIndex` commit `f50e529`, not the full Python implementation. The
local implementation is scoped to Markdown KB notes and Claude/Codex transcript
history.

## Document Model

KB PageIndex and Transcript PageIndex now have different user-facing surfaces:

- KB build writes only `nodes.tsv` and `manifest.tsv`
- KB query reads those files and rejects stale indexes without rebuilding
- KB content requires the source KB and an exact inclusive line range
- Transcript PageIndex retains document metadata, structure, exact content fetch,
  and traceable query references

The persistent KB model lives in `src/kb_search.rs`. The transcript model lives
in `src/page_index.rs`.

## Retrieval Flow

KB retrieval is deterministic text search over the persisted TSV index. Its
`document`, `structure`, and agentic query commands are retired. Rebuild the KB
index explicitly after source changes, then query or fetch exact source lines.

Transcript PageIndex query is deterministic lexical scoring over persisted
transcript nodes. It returns traceable document/node hits and a follow-up
content command. Metadata, structure, and exact content remain available as
explicit CLI source-inspection commands; query does not perform an agentic,
tree-walk, or LLM retrieval loop.

## Surfaces

KB PageIndex is exposed through `claude-memory kb-page-index` and may be used by
`claude-memory enrich` under tight output caps. Transcript PageIndex is exposed
through `claude-memory transcript-page-index` and remains CLI-only.

The two surfaces intentionally stay separate:

- KB PageIndex is a deterministic TSV text-retrieval surface; query follow-ups use exact source line ranges and explicit `--kb`/`--index` paths.
- Transcript PageIndex is a source-inspection surface for Claude/Codex sessions and retains its document/structure/content flow.
- No active runtime path writes durable transcript-derived memory units;
  legacy records are compatibility-only.

## Bounded Parity

The local implementation does not include PDF parsing, OCR, PageIndex cloud/API
compatibility, FinanceBench claims, or corpus-level filesystem routing. Full
transcript corpus indexing is still too large for routine use; the benchmark in
`docs/benchmarks/page-index-2026-05-10.md` records the current cost and quality
tradeoffs.
