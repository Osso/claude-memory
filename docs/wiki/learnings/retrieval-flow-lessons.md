# Retrieval Flow Lessons

These notes capture design lessons from maintaining the retrieval surfaces.
Some entries describe the retired graph and memory-unit paths and are preserved
as historical context only.

## Keep Runtime Flows Separate

`enrich`, CLI search, KB PageIndex, and Transcript PageIndex are separate
products. A fix in one path does not prove the others are covered.
When changing retrieval behavior, check each caller explicitly.

## Retired Optional Context Paths

The former graph context path required coordinated write and read gates. That
lesson is historical: `build-graph`, `graph-clean`, `graph-dump`, graph
enrichment, and the graph runtime modules are retired.

The former memory-unit enrichment path is also retired. Current `enrich` uses
unified prompt/answer history plus KB PageIndex only.

## Search Is Not One Thing

Current runtime lookup has three source families:

- prompt/answer history: raw conversation chunks in Qdrant
- KB PageIndex: deterministic TSV search plus exact source line-range retrieval
- Transcript PageIndex: deterministic lexical node search plus traceable document/structure/content retrieval

Legacy memory-unit, migration, and export collections are no longer runtime
search surfaces; Qdrant now contains only `claude-session-history`.

## Traceability Beats Magic Injection

KB PageIndex and Transcript PageIndex should stay traceable. KB results expose
source path, line range, and a custom content command with explicit `--kb` and
`--index` paths; Transcript results expose document/node references and content
commands. Prompt enrichment should stay small and labeled because injected
context is easy to over-trust.
