# Memory Units: Notability Frame

Source: adapted from codealmanac `prompts/base/notability.md` and `prompts/base/purpose.md`
Purpose: Foundation for the extractor prompt (step [2] in the friction-memory pipeline).

The extractor should apply this frame when deciding whether a friction moment yields
a storable preload. The output is a 1-3 sentence preload, not a wiki page — adapt
accordingly (no frontmatter, no wikilinks, just the fact).

---

## Core Test

A memory unit is worth storing when it preserves durable, reusable understanding
that would be **costly, useful, or risky to reconstruct** on the next session.

## What Usually Deserves a Memory Unit

- **Entity**: a named thing that the project reasons about (a CLI flag, a Qdrant
  collection, an API field, a config key) and its meaning in this project.
- **Flow**: behavior that crosses files, commands, or systems in a non-obvious way.
- **Contract**: obligations between callers, providers, schemas, or external services
  that are not visible from a single file.
- **Decision or rationale**: why the project chose one path over plausible alternatives.
- **Risk or invariant**: a rule, assumption, or fragile behavior that future work must
  preserve — especially one that caused a friction moment.
- **Research synthesis**: a conclusion reached after reading docs, papers, or source code
  that is not obvious from the surface.
- **Correction**: a mistake that was made and resolved during the session.

## What Does Not Deserve a Memory Unit

Reject candidates that are only:

- file-by-file summaries ("src/index.rs handles indexing")
- folder trees in prose
- one-off facts obvious from reading a single nearby file
- guesses about intent not supported by code or explicit user statement
- generic API documentation that could be found by reading the SDK docs
- task progress logs ("we fixed the bug by doing X")
- conclusions that do not connect to future work

If the candidate would not change assistant behavior on the next similar prompt,
discard it. Silence is a valid outcome.

## Scope Classification

After extracting a candidate, classify its scope:

- **Global**: a cross-project pattern — a model behavior, a tool preference, a
  workflow habit that applies regardless of which repo is open.
- **Project-local**: a fact about a specific project — a schema detail, a naming
  convention, a local config choice, a per-repo architectural decision.

Automatic extraction stores global memories as `project = None` and local memories as
`project = <cwd-slug>`. Manual writes must pass an explicit project scope; use
`__global__` when the memory should be stored globally. Enrich only surfaces
project-local memories when running in that project.

## Adaptation Notes

Codealmanac's notability frame targets wiki pages (markdown files, 100-500 words,
with frontmatter and wikilinks). Memory units are 1-3 sentences. Apply the same
judgment — notability, anti-patterns, core test — but compress the output to the
single most reusable sentence or two. No preamble, no "this session showed that",
just the fact a future session needs.
