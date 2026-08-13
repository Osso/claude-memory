# Prompt and Answer History

CLI session-history search exposes two typed views:

- `claude-memory search --type prompts`: what the user asked or discussed
- `claude-memory search --type answers`: how the assistant responded or solved a problem

Both views use the configured Qdrant collection. The search surface selects the
history kind with payload filters.

## Embedding profiles

The default profile is Ollama `qwen3-embedding:0.6b-ctx2048` at 1024 dense
dimensions in `claude-session-history`. The backend, model, vector size,
collection, and optional query instruction are selected with:

- `CLAUDE_MEMORY_EMBEDDING_BACKEND` (`ollama` or `openrouter`)
- `CLAUDE_MEMORY_EMBEDDING_MODEL`
- `CLAUDE_MEMORY_VECTOR_SIZE` (positive integer)
- `CLAUDE_MEMORY_COLLECTION`
- `CLAUDE_MEMORY_QUERY_INSTRUCTION`

The query instruction is added only to query embeddings, not document text.
OpenRouter uses `qwen/qwen3-embedding-8b` when selected and reads `api_key`
only from `~/.config/openrouter/config.toml`. Document requests are batched and
transient failures are retried. A dense-vector dimension mismatch fails without
replacing the collection. An embedding failure stops indexing rather than
continuing with a partial write.

## Collection and payload

The configured collection uses the project hybrid dense and BM25 vector layout.
Each point stores:

- `text` — embedded transcript chunk
- `type` — `prompt` or `answer`
- `source` — `session` or `archive`
- `path` — source-relative path or archive filename
- `session_id` — session identifier when available
- `hash` — persisted history identity

The persisted hash is `type:source:chunk_hash`. Identical text therefore remains
distinct across prompt/answer and session/archive views, while repeated identical
chunks within the same type/source intentionally collapse. Point identity is
derived from the same history hash; message, turn, and chunk ordinals are not
added merely to preserve duplicate occurrences.

## Index inputs

`claude-memory index` reads these transcript sources:

- Claude active: `~/.claude/projects/**/*.jsonl`, with `source=session`
- Claude archive: `~/.claude/archive/**/*.jsonl.zst`, with `source=archive`
- Codex active: `~/.codex/sessions/**/*.jsonl`, with `source=session`
- Codex archive: `~/.codex/archived_sessions/**/*.jsonl`, with `source=archive`
- Pi: `~/.config/pi/agent/sessions/**/*.jsonl`, with `source=session`

Pi archive status is metadata; archived sessions remain in the same session
tree. Discovery requires the Pi session header and therefore excludes detached
job/runtime JSONL artifacts. User text becomes `type=prompt`; assistant text
becomes `type=answer`. Text is joined and split into overlapping embedding
chunks. `claude-memory index-file` auto-detects Claude, Codex, and Pi JSONL and
also accepts Claude `.jsonl.zst` archives, writing both history types to the
same collection.

Session-history indexing does not read project summaries or KB Markdown.
Manual memories and the former `claude-memory`, `claude-session-prompts`, and
`claude-answers` stores are not normal indexing targets or alternate search
paths for this surface. The legacy memory-unit, migration, and export paths are
retired. KB PageIndex and transcript PageIndex remain separate features.

## Deduplication and writes

Index startup scrolls existing `hash` payloads from the configured collection.
Each input is filtered against those hashes and against hashes already seen in
the same input. New chunks are embedded in batches and upserted with their
payload metadata. A separate collection is required when the selected model or
vector size differs from an existing collection; the mismatch check never
replaces that collection.

The `--fresh` flag ignores loaded hashes for a complete re-index. It does not
change the collection or payload model.

## Search paths

CLI search defaults to one combined prompt-and-answer query. Optional filters
narrow that same ranked query:

```text
claude-memory search <query>
claude-memory search --type prompts <query>
claude-memory search --type answers <query>
claude-memory search --session <id-substring> <query>
```

The type filter matches `prompt` or `answer`; an internal source filter can
match `session` or `archive`. The optional session filter restricts results to
persisted session IDs containing the supplied substring. Session substring
matching is case-sensitive and applies to the persisted `session_id`. Qdrant
exact-match filters do not support arbitrary substrings, so search first scrolls
the relevant payloads to collect
exact session IDs containing the substring, then applies those IDs to the
ranked vector query. `--limit` therefore applies after session filtering rather
than truncating a global result set first.

Search uses the named dense vector when `[search].enabled = true`; the collection also stores Qdrant BM25 sparse vectors, but this search path does not fuse them. The `enrich` path embeds its query once, applying the optional query instruction there, then applies separate prompt and answer filters to Qdrant searches using that shared vector. After the shared embedding and collection setup succeed,
prompt and answer search errors remain independent, so one failed group does
not discard the other group's results. When semantic search is disabled, these
history paths return no results. Search result formatting reads `text`, `source`,
`path`, `session_id`, and score; absent string payloads become empty fields.

## Separate surfaces

- KB PageIndex provides exact Markdown retrieval and prompt-enrichment context.
- Transcript PageIndex provides local transcript navigation.
- Legacy memory-unit records are not a runtime enrichment surface; their former
  compatibility readers are retired.
