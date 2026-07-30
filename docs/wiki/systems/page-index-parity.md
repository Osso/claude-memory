# PageIndex Parity

This project adopts the useful retrieval shape from `VectifyAI/PageIndex` without reproducing its full Python implementation. Local behavior is scoped to Markdown KB notes and Claude/Codex transcript history.

## Document Model

KB PageIndex and Transcript PageIndex have separate surfaces:

- KB build writes `nodes.tsv` and `manifest.tsv`.
- KB query validates and synchronously rebuilds a missing or stale index, then prints matching excerpts directly.
- KB `document`, `structure`, `content`, and agentic query commands are retired.
- Transcript PageIndex retains document metadata, structure, exact content fetch, and traceable query references.

The persistent KB model lives in `src/kb_search.rs`. The transcript model lives in `src/page_index.rs`.

## KB Retrieval Flow

Query and enrich acquire the index lock, validate `manifest.tsv`, and rebuild when the Markdown file set or metadata changed. Rebuild writes a staging index and activates it only after both TSV files are complete. Failed rebuilding preserves the previous index.

Direct query fails if rebuilding fails. Enrich instead uses the previous index when available and adds a warning to agent context; without a usable index it warns and continues with other enrichment sources.

The explicit `kb-page-index build` command remains for prewarming and diagnostics. Routine KB edits require no manual rebuild.

## Transcript Retrieval Flow

Transcript PageIndex query remains deterministic lexical scoring over persisted transcript nodes. It returns traceable document/node hits and follow-up content commands. Metadata, structure, and exact content remain explicit CLI source-inspection commands.

## Surfaces

KB PageIndex is exposed through `claude-memory kb-page-index` and used by `claude-memory enrich` under tight output caps. Transcript PageIndex remains CLI-only.

Neither PageIndex path writes durable memory units or graph records.

## Bounded Parity

PDF parsing, OCR, PageIndex cloud/API compatibility, FinanceBench claims, and corpus-level filesystem routing remain out of scope. The benchmark in `docs/benchmarks/page-index-2026-05-10.md` records transcript corpus cost and quality tradeoffs.
