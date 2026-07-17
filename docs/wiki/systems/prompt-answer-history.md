# Prompt and Answer History

Session-history search stores raw transcript text for two questions:

- `prompt_search`: what the user asked or discussed
- `answer_search`: how the assistant responded or solved a problem

Both tools read one Qdrant collection. The search surface selects the history
kind with payload filters.

## Collection and payload

The collection is `claude-session-history`, using the project hybrid dense and
BM25 vector layout. Each point stores:

- `text` — embedded transcript chunk
- `type` — `prompt` or `answer`
- `source` — `session` or `archive`
- `path` — source-relative path or archive filename
- `session_id` — session identifier when available
- `hash` — persisted history identity

The persisted hash is `type:source:chunk_hash`. Identical text therefore remains
distinct across prompt/answer and session/archive views. Point identity is
derived from the same history hash.

## Index inputs

`claude-memory index` reads only Claude transcript files:

- active sessions: `~/.claude/projects/**/*.jsonl`, with `source=session`
- archives: `~/.claude/archive/*.jsonl.zst`, with `source=archive`

User text becomes `type=prompt`; assistant text becomes `type=answer`. Text is
joined and split into overlapping embedding chunks. `claude-memory index-file`
accepts one `.jsonl` or `.jsonl.zst` transcript and writes both history types
to the same collection.

Session-history indexing does not read project summaries or KB Markdown. Manual
memories and the `claude-memory`, `claude-session-prompts`, and
`claude-answers` stores are not targets or alternate search paths for this
surface. KB ingestion, KB PageIndex, memory-unit search, and transcript
PageIndex remain separate features.

## Deduplication and writes

Index startup scrolls existing `hash` payloads from `claude-session-history`.
Each input is filtered against those hashes and against hashes already seen in
the same input. New chunks are embedded in batches and upserted with their
payload metadata.

The `--fresh` flag ignores loaded hashes for a complete re-index. It does not
change the collection or payload model.

## Search paths

CLI search defaults to memory units:

```text
claude-memory search <query>
claude-memory search --type prompts <query>
claude-memory search --type answers <query>
```

The prompt and answer targets query `claude-session-history` with a required
`type` filter. An optional source filter matches `session` or `archive`.

The MCP tools use the same collection and filters:

- `prompt_search` requires `type=prompt`
- `answer_search` requires `type=answer`

Both paths use hybrid dense/BM25 retrieval when `[search].enabled = true`.
When semantic search is disabled, these history paths return no results.
Search result formatting reads `text`, `source`, `path`, `session_id`, and
score; absent string payloads become empty fields.

## Separate surfaces

- Memory-unit search is the default CLI search target.
- KB PageIndex provides exact Markdown retrieval and prompt-enrichment context.
- Transcript PageIndex provides local transcript navigation.
- `memory_write` does not store manual memories; project-local durable context
  belongs in `docs/local/memory.md`.
