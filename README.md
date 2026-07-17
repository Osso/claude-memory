# claude-memory

Semantic memory for Claude Code. Indexes active and archived Claude transcript chunks into one session-history collection, then exposes separate memory, history, and PageIndex surfaces through MCP tools and the CLI.

## What it does

- **Memory-unit search** over durable distilled memories, with substring search when semantic search is disabled
- **Prompt/answer history search** over active and archived transcript chunks using Ollama embeddings and Qdrant
- **KB PageIndex** for traceable Markdown knowledge-base retrieval with exact content fetches
- **Prompt enrichment hook** injects small labeled memory and KB context into Claude Code prompts automatically
- **Optional graph context** reads CozoDB entity relationships only when `[graph].enabled = true`

## Architecture

```
Claude Code ──MCP──► claude-memory-mcp ──► Qdrant (vectors + BM25)
                                       ──► Ollama (embeddings + default LLM)
                                       ──► KB/Transcript PageIndex files
                                       ──► CozoDB (optional graph)
                                       ──► configured LLM backend (optional)
```

- **Vector store**: Qdrant with hybrid collections (dense + BM25 sparse vectors)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (1024 dimensions)
- **Graph**: optional CozoDB SQLite backend for entity-relationship storage
- **LLM**: configurable via `CLAUDE_MEMORY_LLM_BACKEND`; defaults to local Ollama

## Data sources

Session-history indexing reads only:

| Source | Path | Payload source |
|--------|------|---------------|
| Active sessions | `~/.claude/projects/**/*.jsonl` | `source=session` |
| Session archives | `~/.claude/archive/*.jsonl.zst` | `source=archive` |

Project summaries, KB Markdown, manual memories, and the
`claude-memory`, `claude-session-prompts`, and `claude-answers` stores are not
normal session-history indexing targets or alternate search paths. The separate
`claude-memory-migrate` tool reads `claude-memory` and `claude-answers` only as
one-time migration inputs. KB PageIndex, memory-unit search, and transcript
PageIndex use separate surfaces; the former `ingest-kb` surface is retired.
Local project memories remain editable Markdown under `docs/local/`.

## Setup

### Dependencies

1. **Qdrant** (localhost:6334):
   ```bash
   authsudo arch install /syncthing/Sync/Projects/system/arch-pkgbuilds/qdrant-bin
   authsudo systemctl enable --now qdrant.service
   ```
   The local package installs Qdrant as a host systemd service bound to localhost.

2. **Ollama** (localhost:11434):
   ```bash
   ollama serve
   ollama pull qwen3-embedding:0.6b
   echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | \
     ollama create qwen3-embedding:0.6b-ctx2048 -f -
   ```

3. **Optional LLM backend environment**:
   - unset `CLAUDE_MEMORY_LLM_BACKEND` uses local Ollama
   - supported backends: `ollama`, `anthropic`, `openrouter`, `claude`, `codex`
   - `CLAUDE_MEMORY_LLM_MODEL` overrides the backend default model
   - legacy caveat: `claude-memory deduplicate` still preflights `ANTHROPIC_API_KEY`

### Build

```bash
cargo build --release
```

Produces four binaries:
- `claude-memory` — CLI for indexing and searching
- `claude-memory-mcp` — MCP server for Claude Code integration
- `claude-memory-migrate` — guarded legacy Qdrant storage migration CLI
- `claude-memory-export-kb` — guarded durable-memory to KB Markdown export CLI

## CLI commands

```
claude-memory index              # Index active and archived transcript chunks
claude-memory index --fresh      # Re-index transcript chunks from scratch
claude-memory index-file <path>  # Index one conversation file (prompts + answers)
claude-memory search --type prompts <q>  # Search the prompt history view
claude-memory search --type answers <q>  # Search the answer history view
# --type is required; valid values are prompts and answers
claude-memory kb-page-index build        # Build the persistent KB PageIndex
claude-memory kb-page-index query <q>    # Query KB notes through PageIndex
claude-memory transcript-page-index build     # Build transcript PageIndex trees
claude-memory transcript-page-index query <q>  # Query transcript outlines
claude-memory build-graph        # Extract entities/relationships from memories
claude-memory build-graph --kb   # Also extract from KB files
claude-memory graph-dump         # Show graph contents
claude-memory deduplicate        # Merge similar memories via LLM
claude-memory enrich             # Enrich prompt with context (hook use)
claude-memory stats              # Show collection statistics

# Durable-memory to KB Markdown export (plan/apply/verify)
claude-memory-export-kb plan
claude-memory-export-kb apply --kb-root /syncthing/Sync/KB
claude-memory-export-kb verify --kb-root /syncthing/Sync/KB

# One-time legacy storage migration (read-only plan/verify; apply requires a backup root)
claude-memory-migrate plan
claude-memory-migrate apply --backup-dir <directory>
claude-memory-migrate verify
```

## PageIndex commands

PageIndex is the raw, traceable retrieval surface. It is separate from memory
units and vector prompt/answer history.

### KB PageIndex

Use KB PageIndex for exact context from `/syncthing/Sync/KB` Markdown notes.
KB Markdown is not ingested into memory units by an active CLI path.

```bash
claude-memory kb-page-index build \
  --kb /syncthing/Sync/KB \
  --output ~/.cache/claude-memory/kb-page-index

claude-memory kb-page-index query "rust graphics toolkit" --limit 3
claude-memory kb-page-index query "rust graphics toolkit" --mode agentic
claude-memory kb-page-index document research/rust-ui-libraries.md
claude-memory kb-page-index structure research/rust-ui-libraries.md
claude-memory kb-page-index content research/rust-ui-libraries.md 000002
claude-memory kb-page-index content research/rust-ui-libraries.md 12-24
```

The default query mode is deterministic lexical search. `--mode agentic` uses
the project `llm` backend to inspect metadata and structure, choose tight
content fetches, and answer from fetched content. If the model call fails or
returns no usable fetch plan, it falls back to the labeled lexical mode.

### Transcript PageIndex

Use Transcript PageIndex for local navigation of Claude and Codex session
history. It does not create durable memory units and is not injected into prompts
by default. It is intentionally CLI-only until query quality and full-corpus
build cost justify adding an MCP tool:

```bash
claude-memory transcript-page-index build \
  --projects ~/.claude/projects \
  --archive ~/.claude/archive \
  --codex-sessions ~/.codex/sessions \
  --codex-archive ~/.codex/archived_sessions \
  --output ~/.cache/claude-memory/transcript-page-index

claude-memory transcript-page-index query "deploy script" --limit 3
claude-memory transcript-page-index query "deploy script" --mode agentic
claude-memory transcript-page-index document <session-id-or-path>
claude-memory transcript-page-index structure <session-id-or-path>
claude-memory transcript-page-index content <session-id-or-path> 000001
claude-memory transcript-page-index content <session-id-or-path> turns:4-8
```

Benchmarks for the current implementation are recorded in
`docs/benchmarks/page-index-2026-05-10.md`.

## MCP tools

The MCP server exposes two tools:

| Tool | Description |
|------|-------------|
| `prompt_search` | Search user prompts and questions from session history |
| `answer_search` | Search assistant responses and solutions from session history |

### Claude Code integration

Add to your MCP config:

```json
{
  "mcpServers": {
    "claude-memory": {
      "command": "claude-memory-mcp"
    }
  }
}
```

For prompt enrichment, add a `UserPromptSubmit` hook in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "claude-memory enrich",
            "timeout": 15
          }
        ]
      }
    ]
  }
}
```

This injects relevant memory context into every prompt automatically.

### Optional Semantic Search

Semantic search and graph context are disabled by default. Enable semantic
search in `~/.config/claude-memory/config.toml` when Ollama/Qdrant are
available:

```toml
[search]
enabled = true
```

To enable optional graph context as well:

```toml
[graph]
enabled = true
```

## How retrieval works

The runtime flows are documented in
[`docs/wiki/systems/retrieval-flows.md`](docs/wiki/systems/retrieval-flows.md).
Short version:

- `claude-memory enrich` combines memory-unit hints, deterministic KB PageIndex
  context, and optional graph reads.
- CLI `search` requires `--type prompts|answers` and queries the corresponding
  filtered view of the single `claude-session-history` collection when semantic
  search is enabled.
- MCP exposes only `prompt_search` and `answer_search`, which query the same
  filtered views.
- KB PageIndex is the exact Markdown retrieval surface.
- Transcript PageIndex is CLI-only source inspection and is not injected into
  prompts by default.

## License

MIT
