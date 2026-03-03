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
| Manual memories | via MCP `memory_write` | Explicitly recorded learnings |

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
claude-memory search-prompts <q> # Search user prompts and KB
claude-memory search-answers <q> # Search assistant responses
claude-memory build-graph        # Extract entities/relationships from memories
claude-memory build-graph --kb   # Also extract from KB files
claude-memory graph-dump         # Show graph contents
claude-memory deduplicate        # Merge similar memories via LLM
claude-memory enrich             # Enrich prompt with context (hook use)
claude-memory stats              # Show collection statistics
```

## MCP tools

The MCP server exposes four tools:

| Tool | Description |
|------|-------------|
| `memory_write` | Store a memory with optional category and project |
| `prompt_search` | Search prompts, questions, and knowledge base |
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

For prompt enrichment, add a `UserPromptSubmit` hook that calls `claude-memory enrich`.

## How search works

1. Query embedded via Ollama
2. Parallel dense vector + BM25 sparse search in Qdrant (4x over-fetch)
3. Results merged via Reciprocal Rank Fusion
4. LLM filters for genuine relevance (falls back to top-N on failure)
5. Graph context appended from related entities

## License

MIT
