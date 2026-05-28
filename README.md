# claude-memory

Semantic memory for Claude Code. Indexes conversation history and knowledge base files into a hybrid vector store, then exposes search via MCP tools and CLI.

## What it does

- **Hybrid search** over past conversations, project summaries, and markdown knowledge bases using dense vectors (Qwen3 embeddings) + BM25 sparse search with Reciprocal Rank Fusion
- **LLM filtering** validates search results are genuinely relevant, not just keyword matches
- **Graph memory** extracts entities and relationships from memories using Claude Haiku, enriches search results with related context
- **Automatic deduplication** merges similar memories via LLM on write
- **Prompt enrichment hook** injects relevant memory context into Claude Code prompts automatically

## Architecture

```
Claude Code ──MCP──► claude-memory-mcp ──► Qdrant (vectors + BM25)
                                       ──► Ollama (embeddings)
                                       ──► CozoDB (graph)
                                       ──► Anthropic API (filtering/merging/extraction)
```

- **Vector store**: Qdrant with hybrid collections (dense + BM25 sparse vectors)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (1024 dimensions)
- **Graph**: CozoDB with SQLite backend for entity-relationship storage
- **LLM**: Claude Haiku for relevance filtering, memory merging, and graph extraction

## Data sources

| Source | Path | Description |
|--------|------|-------------|
| Session archives | `~/.claude/archive/*.jsonl.zst` | Compressed past conversations |
| Active sessions | `~/.claude/projects/**/sessions/*.jsonl` | Current project conversations |
| Project summaries | `~/.claude/projects/**/summary.md` | Project summary files |
| Knowledge base | `/syncthing/Sync/KB/**/*.md` | Markdown knowledge base |
| Local project memories | `docs/local/` in each project | Explicitly recorded project-local context |

## Setup

### Dependencies

1. **Qdrant** (localhost:6334):
   ```bash
   docker run -p 6334:6334 qdrant/qdrant
   ```

2. **Ollama** (localhost:11434):
   ```bash
   ollama serve
   ollama pull qwen3-embedding:0.6b
   echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | \
     ollama create qwen3-embedding:0.6b-ctx2048 -f -
   ```

3. **Environment**: `ANTHROPIC_API_KEY` for LLM features (filtering, merging, graph extraction)

### Build

```bash
cargo build --release
```

Produces two binaries:
- `claude-memory` — CLI for indexing and searching
- `claude-memory-mcp` — MCP server for Claude Code integration

## CLI commands

```
claude-memory index              # Index all sources
claude-memory index --fresh      # Re-index everything from scratch
claude-memory index-file <path>  # Index a single conversation file
claude-memory ingest-kb          # Extract KB Markdown facts into memory units
claude-memory search <q>                 # Search memories by default
claude-memory search --type prompts <q>  # Search user prompts and KB
claude-memory search --type answers <q>  # Search assistant responses
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
```

## PageIndex commands

PageIndex is the raw, traceable retrieval surface. It is separate from memory
units and vector prompt/answer history.

### KB PageIndex

Use KB PageIndex for exact context from `/syncthing/Sync/KB` Markdown notes:

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

The MCP server exposes four tools:

| Tool | Description |
|------|-------------|
| `memory_write` | Disabled for storage; returns guidance to write project memories under `docs/local/` |
| `prompt_search` | Search prompts, questions, and legacy KB vector chunks |
| `answer_search` | Search assistant responses and solutions |
| `memory_list` | List all memories matching category/project filters |

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

Semantic search depends on Ollama embeddings and is disabled by default. Enable
it in `~/.config/claude-memory/config.toml` when Ollama is available:

```toml
[search]
enabled = true
```

## How search works

1. Query embedded via Ollama
2. Parallel dense vector + BM25 sparse search in Qdrant (4x over-fetch)
3. Results merged via Reciprocal Rank Fusion
4. LLM filters for genuine relevance (falls back to top-N on failure)
5. Graph context appended from related entities

## License

MIT
