`claude-memory-export-kb` exports durable knowledge from legacy Qdrant storage into canonical Markdown under `/syncthing/Sync/KB/memory`. The contract is implemented by `src/kb_export.rs` and `src/bin/claude-memory-export-kb.rs`; mechanics and layout live in [docs/wiki/systems/kb-markdown-export.md](../wiki/systems/kb-markdown-export.md). This specification does not claim that a live export has run.

## What it must do

### Source classification

- [x] Route memory-unit points with `source=session` to friction.
- [x] Route notable-fact points with `source=session` to notable facts and reject unknown/missing notable-fact sources.
- [x] Route `source=memory` records from memory units and legacy `claude-memory` to manual memories.
- [x] Quarantine memory-unit points with missing or unknown source values.
- [x] Exclude `source=kb` vectors from Markdown while retaining manifest accounting.
- [x] Validate every legacy `source` before filtering non-manual history, summary, or KB records.
- [x] Fail malformed source/text fields and require non-empty legacy manual path/hash provenance.
- [x] Require the authorized live unclassified count of 222 before apply or verify.

### Deduplication and provenance

- [x] Deduplicate exact text only within the same destination and original project scope.
- [x] Preserve distinct original projects even when safe filenames collide, including empty global scope versus a literal `global` project.
- [x] Merge duplicate source point IDs and available source paths, legacy hashes, sessions, turns, topics, categories, and creation metadata.
- [x] Emit one manifest entry for every admitted source point, including its original source value, duplicates, quarantine, and KB-vector exclusions.

### Markdown and PageIndex

- [x] Write deterministic project-scoped files below `memory/friction`, `memory/notable-facts`, and `memory/manual-memories`.
- [x] Write unclassified records to `memory/quarantine/unclassified-memory-units.md`.
- [x] Encode exact source text reversibly so headings and unmatched code fences cannot corrupt later Markdown record boundaries.
- [x] Emit stable anchors scoped by destination and original project plus exact SHA-256 content hashes.
- [x] Produce Markdown discoverable through the real KB PageIndex builder and search path.

### Safety and parity

- [x] Preflight every output and refuse conflicting files or non-file targets before writing.
- [x] Write deterministic files directly after source stability is confirmed.
- [x] Treat identical existing output as an idempotent no-op.
- [x] Persist deterministic source/count/path/anchor/hash accounting in `memory/export-manifest.json`.
- [x] Compare every persisted document byte-for-byte and independently decode source text to verify its SHA-256 hash.
- [x] Reload live sources before and after writing and fail if counts, sources, dispositions, IDs, paths, anchors, content hashes, or rendered provenance changed.
- [x] Preserve prompt/answer history in `claude-session-history` and provide no deletion operation.

### Command surface

- [ ] `claude-memory-export-kb plan` reads and reports classification without writing (implemented; live command proof remains pending).
- [ ] `claude-memory-export-kb apply --kb-root <path>` validates, writes, reloads sources, verifies parity, and rebuilds PageIndex (implemented; authorized live proof remains pending).
- [ ] `claude-memory-export-kb verify --kb-root <path>` rebuilds a live plan and checks the written export (implemented; live post-apply proof remains pending).

## How it works

- [docs/wiki/systems/kb-markdown-export.md](../wiki/systems/kb-markdown-export.md) describes source reads, classification, publication, parity, and PageIndex refresh.
- [memory-units.md](memory-units.md) defines memory-unit payloads.
- [page-index-parity.md](page-index-parity.md) defines the KB retrieval surface.
- [prompt-answer-history.md](prompt-answer-history.md) defines the history collection this export does not modify.

## Implementation inventory

- `src/kb_export.rs` — classification, deduplication, provenance, injective paths, Markdown/manifest rendering, direct conflict-refusing writes, and parity verification.
- `src/bin/claude-memory-export-kb.rs` — Qdrant reads, live guardrails, `plan`/`apply`/`verify`, fresh-source comparison, and PageIndex rebuild.
- `src/lib.rs` — exports the `kb_export` module.

## Tests asserting this spec

- `tests/kb_export.rs` — classification, deduplication, quarantine, malformed records, source drift, project collisions, hostile Markdown, path conflicts, idempotence, independent content parity, and real PageIndex discovery.
- `src/bin/claude-memory-export-kb.rs` — point-ID preservation and validate-before-filter legacy selection.

## Known gaps (current cycle)

- [ ] Obtain passing independent review and final verifier proof.
- [ ] Execute and record the authorized live export after the implementation commit.

## Out of scope

- Summary retirement, duplicate KB-vector ingestion removal, or legacy collection deletion.
- Changes to prompt/answer indexing or retrieval.
- Treating a plan or fixture test as evidence that the live export occurred.
