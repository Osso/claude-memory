# Claude Memory: install and agent rules

This file is a portable setup reference for `claude-memory` and a Markdown knowledge base. Copy the rules section into the persistent rules system used by your agent.

## Requirements

- Rust and Cargo
- Qdrant at `http://localhost:6334`
- Ollama at `http://localhost:11434`
- Markdown knowledge base at `/syncthing/Sync/KB`

`claude-memory enrich` currently uses `/syncthing/Sync/KB` as its fixed KB root. Direct KB commands accept a different path through `--kb`, but automatic prompt enrichment does not.

## Install

```bash
cargo install --git https://github.com/Osso/claude-memory
```

Ensure Cargo binaries are available:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

For a local repository checkout, use its deployment script:

```bash
./deploy.sh
```

## Configure embeddings

Pull the embedding model and create the required context-limited variant:

```bash
ollama pull qwen3-embedding:0.6b
printf 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048\n' \
  | ollama create qwen3-embedding:0.6b-ctx2048 -f -
```

Start Qdrant and Ollama through your operating system or container manager. Verify the installed interface:

```bash
claude-memory --help
ollama list
curl -fsS http://localhost:6334/collections
```

## Initialize and use

Backfill missing Claude, Codex, and Pi transcript history:

```bash
claude-memory index
```

Search globally ranked prompt and answer history:

```bash
claude-memory search "query"
claude-memory search --type prompts "query"
claude-memory search --type answers "query"
claude-memory search --limit 10 --json "query"
```

Query the Markdown KB:

```bash
claude-memory kb-page-index query "query" --kb /syncthing/Sync/KB
```

The KB PageIndex is self-healing. `query` and `enrich` synchronously build a missing index and rebuild a stale index when Markdown files are added, changed, or deleted. Manual builds are only for prewarming or diagnostics:

```bash
claude-memory kb-page-index build --kb /syncthing/Sync/KB
```

Prompt-hook integrations call `claude-memory enrich` with JSON on standard input:

```json
{"prompt":"current user prompt"}
```

Session-shutdown integrations should run:

```bash
claude-memory index-file <transcript_path>
```

`enrich` retrieves existing history and KB context; it does not index the current transcript.

## Agent rules

```markdown
# Memory and knowledge base

Use tracked project documentation for project-specific context. Do not create a separate project-local memory file when the project already has an appropriate tracked document.

Search `claude-memory` when:
- starting a task without relevant injected memory or KB context;
- a recorded preference or correction may affect the action;
- prior project decisions or sessions may resolve uncertainty.

Use:
- `claude-memory search "query"` for combined prompt and answer history;
- `claude-memory search --type prompts "query"` for prior user prompts;
- `claude-memory search --type answers "query"` for prior assistant answers;
- `claude-memory kb-page-index query "query"` for Markdown KB facts.

If the prompt already contains relevant enrichment results, do not repeat the same search merely to satisfy the search rule.

Treat the KB as a linked wiki:
- Keep each fact in one source-of-truth document.
- Link to that document instead of duplicating its prose.
- Add `verified: YYYY-MM-DD` to volatile state and inventory facts.
- Treat unstamped or old volatile state as potentially stale.
- Before changing a fact, search the KB for duplicate claims and reconcile them.
- Cross-link related KB pages when writing.
- Store durable project behavior and architecture in tracked project documentation.
- Store cross-project agent behavior in the agent's persistent global rules.
- Do not store routine task details or temporary state as durable memory.
```

## Current command boundaries

Supported KB PageIndex commands:

```text
claude-memory kb-page-index build
claude-memory kb-page-index query
```

KB `document`, `structure`, `content`, and agentic query modes are retired. Transcript PageIndex remains a separate CLI navigation surface.

Qdrant stores only the unified `claude-session-history` collection. Legacy memory-unit, graph, migration, and export runtime commands are retired.
