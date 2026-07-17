# KB Markdown Export

`claude-memory-export-kb` moves durable knowledge from legacy Qdrant surfaces into
editable Markdown under `/syncthing/Sync/KB/memory`. The export is completed for
this migration slice. The Markdown files and `memory/export-manifest.json` are
the canonical exported representation, and KB PageIndex remains the retrieval
surface.

## Readers and compatibility

The planner reads memory units, existing notable-fact records, and eligible
legacy manual records. It preserves source validation for legacy values:
`source=summary` remains non-durable and `source=kb` vectors remain excluded
with manifest accounting. The migration reader retains the same legacy source
compatibility rules for prompt/answer history.

The notable-fact analyzer/writer is retired. Existing notable-fact records are
read only for export/parity compatibility; no active notable-fact writer
remains.

## Output

```text
KB/memory/friction/<project>.md
KB/memory/notable-facts/<project>.md
KB/memory/manual-memories/<project>.md
KB/memory/quarantine/unclassified-memory-units.md
KB/memory/export-manifest.json
```

The export preserves source text, provenance, project scope, source IDs, and
content hashes. PageIndex can discover the resulting Markdown.

## Commands

```text
claude-memory-export-kb plan
claude-memory-export-kb apply --kb-root /syncthing/Sync/KB
claude-memory-export-kb verify --kb-root /syncthing/Sync/KB
```

`plan` and `verify` are read-only. `apply` writes only KB output, checks source
and document parity, and rebuilds KB PageIndex. No command here deletes source
points or collections. Completion of the export is not collection deletion.

## Separate work

Prompt/answer history remains in `claude-session-history`. The former KB-to-
memory-unit ingestion path is retired. Memory-unit and graph runtime search,
deduplication, and enrich paths are also retired. Export and migration readers
remain available for legacy compatibility; they do not claim deletion of source
points or collections. Transcript analyzer and notable-fact writer removal are
documented in the related specs.
