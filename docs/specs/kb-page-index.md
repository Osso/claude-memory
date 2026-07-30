The KB PageIndex feature provides persistent, heading-aware retrieval over the local Markdown knowledge base at `/syncthing/Sync/KB`. It supplies KB context to prompt enrichment without duplicating KB facts into vector history. Implementation details live in [the PageIndex wiki](../wiki/systems/page-index-parity.md).

## What it must do

### Index lifecycle

- [x] Build a persistent KB text index from Markdown into exactly `nodes.tsv` and `manifest.tsv`.
- [x] Preserve Markdown heading paths and source line ranges in indexed results.
- [x] Detect missing and stale indexes when files are added, changed, or deleted.
- [x] Rebuild missing or stale indexes synchronously when query or enrich accesses them.
- [x] Serialize concurrent access and replace completed indexes atomically.
- [x] Preserve the previous index when rebuilding fails.

### CLI surface

- [x] Accept `claude-memory kb-page-index build --kb <dir> --output <dir>` for explicit prewarming and diagnostics.
- [x] Accept `claude-memory kb-page-index query <query> --limit <n> --kb <dir> --index <dir>`.
- [x] Print source path, inclusive line range, heading path, score, and matched excerpt directly.
- [x] Print a clear no-results message when no KB section matches.
- [x] Retire the KB `document`, `structure`, `content`, and agentic query commands.

### Retrieval behavior

- [x] Keep deterministic lexical scoring over persisted heading nodes.
- [x] Rebuild before direct query when the persisted index is missing or stale.
- [x] Fail direct query explicitly when rebuilding fails.
- [x] Keep only results in the best distinct query-term coverage tier up to the requested limit.
- [x] Exclude archive results when nonarchive matches exist in that tier.
- [x] Keep at most one matching section per source document before applying the limit.
- [x] Preserve the real-KB quality gates for frontend-design, Claude Bash hook, and AMDGPU queries.

### Enrich integration

- [x] Format KB results under a distinct `Relevant KB notes (KB PageIndex)` section.
- [x] Include KB PageIndex results alongside unified prompt/answer history when both are relevant.
- [x] Rebuild a missing or stale index before enrichment search.
- [x] If rebuilding fails and a prior index exists, use it and inject a warning into agent context.
- [x] If rebuilding fails without a prior index, inject a warning and continue other enrichment without KB results.
- [x] Do not write KB rebuild warnings to stderr.
- [x] Cap enrich KB output to a small number of results.

## How it works

- [PageIndex parity and retrieval flow](../wiki/systems/page-index-parity.md)
- [Runtime retrieval flows](../wiki/systems/retrieval-flows.md)

## Implementation inventory

- `src/kb_search.rs` — validates, locks, rebuilds, atomically replaces, scores, and queries the KB index.
- `src/enrich_cmd.rs` — injects KB results and rebuild warnings into agent context.
- `src/indexing_cmds.rs` — implements explicit build and self-healing query handlers.
- `src/kb_page_index_cli.rs` — declares the public KB PageIndex CLI.
- `src/main.rs` — dispatches KB PageIndex commands.
- `README.md` — documents installation and user-facing behavior.

## Tests asserting this spec

- `tests/kb_page_index_cli.rs`
- `src/kb_search_tests.rs`
- `src/enrich_cmd.rs`
- `src/main_tests.rs`

## Known gaps (current cycle)

None.

## Out of scope

- Filesystem watchers, cron jobs, and editor/write hooks.
- New user configuration for index maintenance.
- Stale fallback for direct CLI query.
- Vector embeddings or LLM-guided traversal for KB sections.
- Transcript PageIndex behavior.
