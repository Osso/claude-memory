---
name: claude-memory
description: Install, validate, and operate claude-memory with the Markdown KB PageIndex. Use when setting up claude-memory, checking its service dependencies, querying the KB, or deciding how memory and KB rules apply.
---

# Claude Memory setup and KB rules

Run setup in order. Report each command and result. Do not enable, restart, install, or actuate an OS service without explicit user approval.

## 1. Inspect prerequisites

```bash
command -v cargo
command -v ollama
command -v curl
command -v claude-memory || true
curl -fsS http://localhost:6333/collections
ollama list
```

Required services:

- Qdrant HTTP API: `http://localhost:6333`
- Qdrant gRPC: `localhost:6334`
- Ollama API: `http://localhost:11434` for the local rollback profile
- Markdown KB: `/syncthing/Sync/KB`

Production uses OpenRouter. Its `api_key` is read only from
`~/.config/openrouter/config.toml`; Ollama remains available for rollback.

Automatic enrichment currently uses `/syncthing/Sync/KB` as its fixed KB root. Direct KB commands can use another path with `--kb`.

If Qdrant or the selected embedding backend is unavailable, stop and report the
missing service. Ask before proposing package installation, service enablement,
or service restart.

## 2. Install the CLI

```bash
cargo install --git https://github.com/Osso/claude-memory
export PATH="$HOME/.cargo/bin:$PATH"
claude-memory --help
```

For a local checkout, run its deployment script instead:

```bash
./deploy.sh
```

## 3. Configure embeddings

Production reads the persistent profile from
`~/.config/claude-memory/config.toml`:

```toml
[embedding]
backend = "openrouter"
model = "qwen/qwen3-embedding-8b"
vector_size = 4096
collection = "claude-session-history-qwen3-8b"
query_instruction = "Represent this query for retrieval"
```

`CLAUDE_MEMORY_EMBEDDING_BACKEND`, `CLAUDE_MEMORY_EMBEDDING_MODEL`,
`CLAUDE_MEMORY_VECTOR_SIZE`, `CLAUDE_MEMORY_COLLECTION`, and
`CLAUDE_MEMORY_QUERY_INSTRUCTION` override corresponding file values. Built-in
local Ollama defaults apply only when neither source is configured:
`qwen3-embedding:0.6b-ctx2048`, 1024 dimensions, and
`claude-session-history`. Keep that collection intact for rollback.

OpenRouter requests enforce zero data retention; document requests are batched
and transient failures are retried. Dense-vector dimension mismatches fail
without replacing a collection, and an embedding failure stops indexing.

## 4. Initialize and validate

Backfill missing transcript chunks:

```bash
claude-memory index
```

Build the KB PageIndex only when prewarming or diagnosing. Normal query and enrichment are self-healing:

```bash
claude-memory kb-page-index build --kb /syncthing/Sync/KB
claude-memory kb-page-index query "frontend design skill" --kb /syncthing/Sync/KB
claude-memory enrich "frontend design skill"
printf '%s\n' '{"prompt":"frontend design skill"}' | claude-memory enrich
```

`enrich "prompt"` is for manual testing. With no positional argument, `enrich` reads UserPromptSubmit JSON from stdin. Enrichment retrieves existing history and KB context; it does not index the current transcript.

KB `query` and `enrich` synchronously build missing indexes and rebuild stale indexes. The index contains `nodes.tsv` and `manifest.tsv`; completed rebuilds replace the prior index atomically under a lock.

## 5. Operating rules

Apply these rules when using the memory system:

- Search `claude-memory` when starting a task, when a recorded preference or correction may affect an action, or when prior project decisions may resolve uncertainty.
- Use `claude-memory search "query"` for combined prompt/answer history.
- Use `claude-memory search --type prompts "query"` for prior user prompts.
- Use `claude-memory search --type answers "query"` for prior assistant answers.
- Use `claude-memory search --session <id-substring> "query"` to restrict results to a session ID substring.
- Use `claude-memory kb-page-index query "query"` for Markdown KB facts.
- If relevant enrichment is already present, do not repeat the same search merely to satisfy the search rule.
- Treat the KB as a linked wiki, not duplicated notes.
- Keep each fact in one source-of-truth document; link to it from other pages.
- Add `verified: YYYY-MM-DD` to volatile state and inventory facts.
- Treat unstamped or old volatile state as potentially stale.
- Before changing a state fact, search the KB for duplicate claims and reconcile them.
- Cross-link related KB pages when writing.
- Put project behavior and architecture in tracked project documentation.
- Put cross-project agent behavior in persistent global rules.
- Do not store routine task details or temporary state as durable memory.

## Command boundaries

Supported KB PageIndex commands:

```text
claude-memory kb-page-index build
claude-memory kb-page-index query
```

KB `document`, `structure`, `content`, and agentic query modes are retired. Transcript PageIndex is a separate CLI navigation surface. Legacy memory-unit, graph, migration, and export runtime commands are retired.
