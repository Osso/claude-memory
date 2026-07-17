# Retrieval Flow Lessons

These notes capture design lessons from maintaining the retrieval surfaces.
Some entries describe the retired graph and memory-unit paths and are preserved
as historical context only.

## Keep Runtime Flows Separate

`enrich`, CLI search, MCP search, KB PageIndex, and Transcript PageIndex are
separate products. A fix in one path does not prove the others are covered.
When changing retrieval behavior, check each caller explicitly.

## Retired Optional Context Paths

The former graph context path required coordinated write and read gates. That
lesson is historical: `build-graph`, `graph-clean`, `graph-dump`, graph
enrichment, and the graph runtime modules are retired.

The former memory-unit enrichment path is also retired. Current `enrich` uses
unified prompt/answer history plus KB PageIndex only.

## Prefer Local Defaults, But Document Overrides

The default LLM backend is local Ollama. Non-local backends are still available
for quality or compatibility, but they should be explicit through
`CLAUDE_MEMORY_LLM_BACKEND`. Documentation should describe the default and the
override path, not assume a single hosted provider.

## Search Is Not One Thing

Current runtime lookup has two source families:

- prompt/answer history: raw conversation chunks in Qdrant
- PageIndex: traceable document/structure/content retrieval

Legacy memory-unit and graph collections may still be named by compatibility
readers, but they are not runtime search surfaces.

## Traceability Beats Magic Injection

KB PageIndex and Transcript PageIndex should stay traceable: document id, node
id, source path, and content command. Prompt enrichment should stay small and
labeled because injected context is easy to over-trust.
