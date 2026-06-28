# Retrieval Flow Lessons

These notes capture design lessons from maintaining the retrieval surfaces.

## Keep Runtime Flows Separate

`enrich`, CLI search, MCP search, KB PageIndex, Transcript PageIndex, and graph
lookup are separate products. A fix in one path does not prove the others are
covered. When changing retrieval behavior, check each caller explicitly.

## Put Optional Context Behind Gates

Graph context is useful only when intentionally enabled. It must be gated at both
write/extraction time and read/enrichment time:

- extraction: `build-graph`
- hook output: `claude-memory enrich`
- MCP output: `prompt_search` / `answer_search`

A disabled subsystem can still leak stale context if only the writer is gated.

## Prefer Local Defaults, But Document Overrides

The default LLM backend is local Ollama. Non-local backends are still available
for quality or compatibility, but they should be explicit through
`CLAUDE_MEMORY_LLM_BACKEND`. Documentation should describe the default and the
override path, not assume a single hosted provider.

## Search Is Not One Thing

There are three main lookup families:

- memory units: durable distilled facts and preloads
- prompt/answer history: raw conversation chunks in Qdrant
- PageIndex: traceable document/structure/content retrieval

Naming matters. Broad words like "search" hide different storage, gates,
ranking, and output contracts.

## Traceability Beats Magic Injection

KB PageIndex and Transcript PageIndex should stay traceable: document id, node
id, source path, and content command. Prompt enrichment should stay small and
labeled because injected context is easy to over-trust.
