The `claude-memory-migrate` binary defines a one-time, guarded migration from the legacy Qdrant history stores into `claude-session-history`. This specification is the contract; implementation notes live in [docs/wiki/systems/storage-migration.md](../wiki/systems/storage-migration.md). This documents the command behavior only and does not claim that a live migration has been run.

## What it must do

### Planning and classification

- [x] `plan` is read-only.
- [x] `plan` reads and classifies only `claude-memory` and `claude-answers`.
- [x] Only points whose payload `source` is `session` or `archive` are eligible.
- [x] Unsupported sources, including summary, KB, and manual-memory records, are counted as skipped rather than migrated.
- [x] Missing required payload fields fail classification instead of being silently repaired.
- [x] A missing `session_id` is normalized to the empty string.
- [x] Planning reports raw, eligible, unique, duplicate, skipped, and deterministic per-type/source counts.

### Identity and data preservation

- [x] Destination history identity is `type:source:legacy_hash`, where the legacy hash is the content hash.
- [x] Prompt records from `claude-memory` become `type=prompt`; answer records from `claude-answers` become `type=answer`.
- [x] Duplicate destination identities within the same type/source are intentionally retained once with deterministic accounting; migration does not preserve repeated identical occurrences using message, turn, or chunk ordinals.
- [x] Source vectors are preserved exactly.
- [x] Every source payload field is retained; canonical destination `type`, `hash`, and normalized `session_id` fields are written according to the destination contract.

### Apply safety and migration

- [x] The complete apply ordering gate is enforced: an initial source watermark and complete backup-set verification precede classification, source-stability checks, and destination creation; backup-set validation requires each named legacy collection, source changes before destination creation abort before writes, source changes after copying trigger rollback, including post-copy watermark-read failures, existing destinations are refused before copying, and handled copy or parity failures enter rollback handling; if rollback fails, the returned error preserves both migration and rollback failures.
- [ ] `apply --backup-dir <dir>` atomically creates a new destination and fails when `claude-session-history` already exists; deterministic IDs prevent duplicates within one run.
- [ ] Before writing the destination, `apply` creates/downloads snapshots for all four legacy collections: `claude-memory`, `claude-answers`, `claude-memory-units`, and `claude-notable-facts`.
- [x] Each backup run uses a unique directory with mode `0700`.
- [x] Snapshot names are validated as safe single filenames; downloaded files are collection-prefixed, mode `0600`, and require advertised checksum, expected size, exact written bytes, and final four-file backup-set revalidation before destination writes.
- [x] Source raw point IDs, vectors, and payloads remain stable across pre-backup, post-backup, and immediately pre-write reads; changes detected before destination creation abort before destination writes, and changes detected after copying trigger rollback.
- [x] `apply` creates a new `claude-session-history`, copies eligible unique points, and performs exact ID, point-count, vector, full-payload, key, and grouped-count parity verification.
- [x] A destination created by the current run is removed after a handled copy failure, post-copy watermark-read failure, or parity failure after the destination has been populated.
- [ ] Crash recovery requires explicit inspection before retrying and is never automatic.

### Verification and command surface

- [x] `verify` is read-only and checks exact destination IDs, point count, vectors, full payloads, `(type, source, hash)` keys, and grouped counts.
- [x] The binary exposes only `plan`, `apply`, and `verify`.
- [x] The migration surface provides no deletion command and no fallback migration path.

## How it works

- [docs/wiki/systems/storage-migration.md](../wiki/systems/storage-migration.md) describes source classification, backup verification, destination writes, and parity checks.
- [docs/wiki/systems/prompt-answer-history.md](../wiki/systems/prompt-answer-history.md) describes the destination collection and its normal indexing/search surfaces.

## Implementation inventory

- `src/bin/claude-memory-migrate.rs` — CLI commands, Qdrant reads/writes, four-collection snapshots, backup integrity checks, and live parity verification.
- `src/migration.rs` — payload validation, source classification, canonical identity construction, deterministic deduplication, and parity key comparison.
- `src/index.rs` — declares the `claude-session-history` destination collection name.
- `src/qdrant_hybrid.rs` — creates the destination hybrid collection.

## Tests asserting this spec

- `src/migration.rs`
  - `prompt_and_answer_records_map_to_typed_history`
  - `identical_type_and_text_from_different_sources_have_distinct_identity`
  - `non_session_sources_are_rejected`
  - `destination_point_preserves_vectors_and_extra_payload`
  - `dry_run_classification_accounts_for_deduplication_and_skips`
  - `missing_optional_session_id_normalizes_to_empty_string`
  - `malformed_eligible_point_fails_fast`
  - `parity_requires_exact_type_source_and_hash_set`
- `src/bin/claude-memory-migrate.rs`
  - `backup_directories_are_unique_and_permission_locked`
  - `snapshot_metadata_rejects_size_and_checksum_mismatch`
  - `snapshot_metadata_requires_advertised_checksum`
  - `unsafe_snapshot_names_are_rejected`
  - `missing_requested_vectors_fail_closed`
  - `written_snapshot_is_byte_and_hash_verified`
  - `backup_set_requires_each_named_legacy_collection`
  - `apply_stops_before_destination_when_backup_set_is_incomplete`
  - `apply_stops_before_destination_when_source_changes`
  - `apply_refuses_existing_destination_without_copy_or_rollback`
  - `apply_rolls_back_destination_created_before_copy_failure`
  - `apply_rolls_back_when_source_changes_during_copy`
  - `apply_rolls_back_when_post_copy_watermark_read_fails`
  - `apply_rolls_back_when_parity_verification_fails`
  - `apply_reports_copy_and_rollback_failures`
  - `full_parity_rejects_vector_payload_id_and_count_changes`
  - `parity_rejects_equal_counts_with_different_keys`

## Known gaps (current cycle)

- [ ] Add an integration test proving live destination point/vector/payload parity after a migration.
- [ ] Execute and record an authorized live migration; this documentation does not claim that one has occurred.

## Out of scope

- Deleting legacy collections or source points.
- Fallback migration commands or alternate destination paths.
- Migrating unsupported summary, KB, or manual-memory source records.
- Treating a successful dry run or test suite as evidence that production/live migration is complete.
- Preserving repeated identical chunks as separate points within the same type/source; intentional content deduplication is part of the destination contract.
