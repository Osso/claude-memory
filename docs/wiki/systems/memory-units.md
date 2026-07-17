# Memory units

Memory units are compact durable preloads stored separately from raw prompt and
answer history. The former friction analyzer is retired. Remaining paths cover
collection access, search, listing/deletion, deduplication, and prompt enrich.

`enrich` retrieves relevant units when semantic search is enabled and labels
them as possibly useful hints, not authoritative facts. Manual project memory
lives in `docs/local/memory.md`; cross-project rules live in
`/home/osso/AgentConfig/rules`.

No active transcript analyzer populates memory units automatically. This change
does not delete the `claude-memory-units` collection or its records. Prompt /
answer history, KB PageIndex, transcript PageIndex, completed KB export, and
migration/export compatibility readers remain separate surfaces.
