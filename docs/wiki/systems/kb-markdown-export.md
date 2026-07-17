# KB Markdown Export

`claude-memory-export-kb` moves durable knowledge from legacy Qdrant surfaces into editable KB Markdown. The test-backed contract is [docs/specs/kb-markdown-export.md](../../specs/kb-markdown-export.md). No live export is recorded here yet.

## Classification

The planner reads all memory units and notable facts. It also reads every legacy `claude-memory` payload long enough to validate `source`, then admits only `source=memory` records.

| Source | Destination |
|---|---|
| memory unit, `source=session` | `memory/friction/` |
| notable fact, `source=session` | `memory/notable-facts/` |
| notable fact with missing/unknown source | rejected for review |
| memory unit or legacy memory, `source=memory` | `memory/manual-memories/` |
| memory unit with missing/unknown source | `memory/quarantine/unclassified-memory-units.md` |
| any `source=kb` vector | excluded, manifest-accounted |
| legacy prompt/archive/summary record | validated, then excluded before the export plan |

Apply and verify require exactly 222 raw unclassified memory-unit source points, counted before content deduplication. A changed count fails closed for human review.

## Identity and deduplication

SHA-256 of exact source text supplies content identity. Deduplication key is destination, original project value, and content hash. Duplicate points merge provenance but retain separate manifest entries with their original source values. The original project remains metadata and key material. Non-empty project names use an injective `project-` filename encoding: ASCII letters, digits, `-`, and `_` remain literal; every other byte is percent-encoded. Empty global scope alone uses `__global__.md`, remaining distinct from every named project.

## Markdown representation

```text
KB/memory/friction/<project>.md
KB/memory/notable-facts/<project>.md
KB/memory/manual-memories/<project>.md
KB/memory/quarantine/unclassified-memory-units.md
KB/memory/export-manifest.json
```

Each record has a destination/project-scoped stable anchor, exact content hash, original project, source point IDs, and available source-path/legacy-hash/session/turn/topic/category/timestamp provenance. Provenance values are Markdown-encoded so embedded newlines, headings, fences, or marker text cannot alter later record boundaries. Source text is line-prefixed as a Markdown blockquote between explicit markers. This representation is reversible and prevents embedded headings or unmatched code fences from changing later record boundaries.

## Publication safety

Before writing, every relative path is validated and every target is checked. Identical files are idempotent; different files, non-UTF-8 files, directories, or other target types abort before any write. New files are then written directly with `std::fs`.

## Manifest and parity

`memory/export-manifest.json` records counts and one entry per admitted source point: collection, point ID, original source, legacy path/hash, disposition, destination path, anchor, and source-text hash.

Verification compares persisted document bytes with the planned document, then independently locates each manifest anchor, reverses the blockquote encoding, hashes persisted source text, and compares that hash with the manifest. It also proves source-point count equals manifest count.

Apply reads sources twice before writing and once afterward. Counts, every manifest entry, and rendered provenance documents must match across those reads. It writes directly, verifies persisted parity, and rebuilds PageIndex. A later destructive cleanup requires another fresh verify while writers remain quiesced.

## Commands

```text
claude-memory-export-kb plan
claude-memory-export-kb apply --kb-root /syncthing/Sync/KB
claude-memory-export-kb verify --kb-root /syncthing/Sync/KB
```

`plan` is read-only. `apply` writes only KB output, verifies a fresh source plan, then rebuilds the existing KB PageIndex. `verify` is read-only. The binary has no source-deletion or collection-deletion command.

## Separate later work

Prompt/answer history remains in `claude-session-history`. Summary retirement, duplicate KB-vector ingestion removal, legacy reader/writer removal, and collection deletion are separate commits with separate verification gates.
