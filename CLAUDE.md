# claude-memory

Semantic memory search for Claude Code sessions and knowledge base.

## Architecture

- **Vector store**: Qdrant (localhost:6334)
- **Embeddings**: Ollama `qwen3-embedding:0.6b-ctx2048` (localhost:11434, 1024 dimensions)
- **Integration**: MCP server for Claude Code

## Usage

Manual CLI invocation + Claude Code hooks:

```bash
# Index all sources
claude-memory index

# Search
claude-memory search "query"

# Stats
claude-memory stats
```

## Data Sources

- Session archives: `~/.claude/archive/*.jsonl.zst`
- Active sessions: `~/.claude/projects/**/sessions/*.jsonl`
- Project summaries: `~/.claude/projects/**/summary.md`
- Knowledge base: `/syncthing/Sync/KB/**/*.md`

## Dependencies

Requires running services:
- Qdrant: `docker run -p 6334:6334 qdrant/qdrant`
- Ollama: `ollama serve` with `ollama pull qwen3-embedding:0.6b` then create ctx-limited variant:
  ```bash
  echo -e 'FROM qwen3-embedding:0.6b\nPARAMETER num_ctx 2048' | ollama create qwen3-embedding:0.6b-ctx2048 -f -
  ```
