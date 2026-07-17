# KB Markdown export

`claude-memory-export-kb` exports durable knowledge from legacy Qdrant storage
into canonical Markdown under `/syncthing/Sync/KB/memory`. The export is
completed for this migration slice; its Markdown and manifest are now the
editable KB representation. The export reader remains available for parity and
compatibility checks.

## Preserved behavior

- Memory-unit and existing notable-fact records can be classified and exported.
- Legacy manual records remain supported through source validation.
- `source=summary` and `source=kb` recognition remains for compatibility;
  summary records are non-durable and KB vectors are excluded with manifest
  accounting.
- Prompt/answer history remains in `claude-session-history`.
- KB PageIndex discovers the exported Markdown.
- Export and migration readers do not delete source points or collections.

## Command surface

```text
claude-memory-export-kb plan
claude-memory-export-kb apply --kb-root /syncthing/Sync/KB
claude-memory-export-kb verify --kb-root /syncthing/Sync/KB
```

`plan` and `verify` remain read-only. `apply` writes KB output, verifies source
and content parity, and rebuilds the existing KB PageIndex. Completion of the
export does not imply deletion of any legacy collection.

## Related contracts

- [Memory units](memory-units.md)
- [KB PageIndex](kb-page-index.md)
- [Prompt and answer history](prompt-answer-history.md)
- [Storage migration](storage-migration.md)
