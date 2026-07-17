# Storage Migration

`claude-memory-migrate` is a one-time command-line migration tool for moving eligible history points from the legacy `claude-memory` and `claude-answers` Qdrant collections into `claude-session-history`. It is separate from normal indexing, search, KB PageIndex, and transcript PageIndex flows.

## Commands

```text
claude-memory-migrate plan
claude-memory-migrate apply --backup-dir <directory>
claude-memory-migrate verify
```

- `plan` reads the two legacy history collections and prints deterministic classification counts. It does not write Qdrant data or files.
- `apply` backs up all four legacy collections, atomically creates a new `claude-session-history`, copies deterministic points, then verifies full parity. It fails if the destination already exists.
- `verify` reads the legacy sources and destination and compares exact IDs, counts, vectors, payloads, keys, and grouped counts. It does not write.

There is no delete command and no fallback migration path.

## Classification contract

Only payloads with `source=session` or `source=archive` are eligible. Other source values, including summary, KB, and manual-memory records, are skipped and counted. Required payload fields must be valid; malformed eligible points fail the operation. A missing `session_id` becomes the empty string.

The source collection selects history type:

- `claude-memory` → `type=prompt`
- `claude-answers` → `type=answer`

Destination identity is the canonical string `type:source:legacy_hash`, using the legacy content hash. Duplicate identities within the same type/source intentionally collapse to one point; migration does not add message, turn, or chunk ordinals to preserve repeated identical occurrences. Raw, eligible, unique, duplicate, skipped, and grouped counts are reported in stable order.

Vectors are copied exactly. Payload fields are retained, with canonical destination `type`, `hash`, and normalized `session_id` values applied as required by the destination contract.

## Apply safety

`apply` records the source watermark, then creates a unique backup directory below `--backup-dir` with mode `0700`. It snapshots and downloads these four collections before any destination write:

- `claude-memory`
- `claude-answers`
- `claude-memory-units`
- `claude-notable-facts`

Qdrant-provided snapshot names must be safe single filenames. Each downloaded file is collection-prefixed, written with mode `0600`, and requires an advertised checksum. The command verifies reported size, checksum, exact bytes reread from disk, SHA-256, and the final four-file backup set before any destination write occurs.

After backup, the command rereads sources and requires raw counts, identity keys, and transformed payloads to match the pre-backup watermark, then performs one final stability read immediately before destination creation or resume. Any source change aborts before destination writes.

`apply` atomically creates the hybrid `claude-session-history` collection and deterministically upserts classified unique points. A destination created by the current run is removed after a handled copy or parity failure. If a process crash leaves a destination behind, retry is blocked until that collection is explicitly inspected; the tool never guesses ownership or silently replaces it. Parity requires exact point IDs, raw point count, vectors, full payloads, `(type, source, hash)` keys, and grouped counts.

## Normal operation versus migration

Normal session-history indexing still reads active and archived transcript files directly. The legacy collections are not normal indexing targets or alternate search paths; they are read here only as migration inputs. Search and MCP tools continue to use the shared `claude-session-history` collection described in [prompt-answer-history.md](prompt-answer-history.md).

This page documents the command contract. It does not record a completed live migration.
