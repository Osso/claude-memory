# Transcript-Mining Analyzer + Memory Units

Status: design locked, implementation pending
Date: 2026-04-28

## Goal

Replace the LLM-extracted triplet graph with **operational shortcut memory units** — atomic 1-3 sentence preloads derived from analyzing past transcripts. Each unit answers the question:

> "What 1-3 sentence preload would have let the assistant skip the most exploration on this prompt?"

The graph code is gated off behind config; this plan introduces the new pipeline alongside.

## Core insight

A memory is useful only if it would have *changed assistant behavior* on a real prompt. Extraction therefore runs only on transcript moments where the assistant **actually struggled** (friction-driven). Validation runs the candidate preload through a replay test: would it have short-circuited the exploration?

## Pipeline

```
session transcript
   │
   ▼
[1] friction classifier (per prompt-response pair, LLM call)
   │   → flagged turns
   ▼
[2] candidate extractor (full session as context, LLM call)
   │   → 1-3 sentence preload candidate
   ▼
[3] replay validator (LLM call: simulate response with preload)
   │
   ▼
[4] dual judge (correctness + efficiency, 2 LLM calls)
   │   ├─ both pass → store
   │   ├─ either fails → feedback to [2], iterate (max 3 tries)
   │   └─ all 3 fail → discard friction moment
   ▼
[5] dedup-at-write (cosine sim > 0.92 → append to seen_in_sessions)
   │
   ▼
   memories collection (Qdrant)
```

### Stage details

**[1] Friction classifier.** One LLM call per prompt-response pair in the transcript. Returns binary friction flag + brief reason. Looks for: long tool-call sequences, user corrections ("no, not that"), backtracks, repeated similar searches, "I couldn't find" patterns.

**[2] Candidate extractor.** Sees the entire session transcript up to (and including) the flagged turn. Prompt frames the task as: *"The assistant struggled at turn N. Identify the 1-3 sentence preload — ideally drawn from earlier in this transcript — that would have let the assistant skip the exploration. Return null if no such preload exists."*

Full-session context is intentional: the "would-have-helped" content often lives in the user's earliest statements, far from the friction moment.

**[3] Replay validator.** Re-runs the assistant on the original turn-N user prompt with only the candidate preload as context. Produces a simulated response.

**[4] Dual judge.**
- **Correctness judge:** Does the simulated response reach the same conclusion as the eventual resolution from the original transcript?
- **Efficiency judge:** Does the simulated response *use* the preload (cite it, build on it) instead of re-exploring? Or does it ignore the preload and explore anyway?

Both must pass. The failure reason from a failing judge is fed back to [2] for the next iteration. After 3 failed candidates, the friction moment is discarded — no memory is stored.

**[5] Dedup at write.** Before insertion, search the `memories` collection for near-duplicates (cosine similarity > 0.92). If found, append the new `source_session` to that memory's `seen_in_sessions` and skip the write. Otherwise insert.

## Memory unit schema

Stored in a new Qdrant collection `memories` (alongside existing `chunks`).

```rust
struct MemoryUnit {
    text: String,              // the 1-3 sentence preload
    created_at: DateTime,      // first extraction time
    source_session: String,    // session ID that originally produced it
    source_prompt: String,     // turn identifier (session_id + turn_index)
    seen_in_sessions: Vec<String>, // session IDs where dedup matched (incl. source)
}
```

Notes:
- No `kind` taxonomy yet (free-form text). Add tags later when patterns emerge in real data.
- No `usefulness_score` field. Static ranking — semantic similarity + recency at retrieval time. `seen_in_sessions.len()` is observable but not used for ranking yet.
- All stored memories pass both validation gates by construction; no `validated` flag needed.

## Storage layout

- **Existing `chunks` collection:** unchanged. Continues to serve the chunked-transcript retrieval path.
- **New `memories` collection:** stores memory units. Embedding model = same one used for chunks (qwen3-embedding via Ollama).
- Coexistence is intentional. The chunked path is a fallback; memory units are the operational shortcut path.

## Cold start

**Full backfill + forward.**

- **Backfill:** new CLI command (`claude-memory backfill` or similar) walks `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`, runs the full pipeline over each session. One-shot, resumable (track processed session IDs).
- **Forward:** existing session-end hook invokes the analyzer on the just-completed session. Runs the same pipeline.

Backfill seeds the corpus on day 1 so retrieval can be evaluated immediately. Forward keeps it growing.

## Retrieval / injection

- **Trigger: top-K with similarity threshold floor.** Per prompt, return up to K=3 memories above similarity threshold T (TBD; tune empirically). 0 if nothing matches.
- **Framing: tagged as hints.** Inject under a header like:

  ```
  ## Possibly-useful preloads (from prior sessions, may be stale or wrong; treat as hints, not facts)
  ```

  This avoids the prior `enrich` failure mode where graph results were treated as authoritative.
- **Hook:** the existing `enrich` hook, currently gated off, gets re-enabled in memory-units mode. Graph mode stays disabled.

## Reinforcement (deferred)

Static scoring for v1. No score evolution post-storage. Revisit if corpus quality degrades — at that point we have data to inform the right loop (implicit transcript-driven, explicit tool, or hybrid).

## Cost profile

Per friction moment, worst case (3 failed iterations):
- 1 friction classifier call (per prompt-response pair, runs upstream)
- 3 extractor calls
- 3 replay simulation calls
- 6 judge calls (2 per replay)
- = 12 LLM calls

Per friction moment, validated on first try:
- 1 extractor + 1 replay + 2 judges = 4 LLM calls

All via Codex CLI / `gpt-5.3-codex-spark` — zero marginal token cost on the Pro plan. Limiting factor is wall-clock latency, not money.

Backfill estimate (rough): 100 sessions × ~5 friction moments × ~6 calls average = ~3000 calls. At ~30s/call → ~25 hours. Run as background batch job; resume on interruption.

## What this replaces

- LLM-extracted triplet graph (already gated off)
- Chunked retrieval as primary path (chunks remain as fallback)

## Open implementation questions

These are the **how**, not the **what**. Resolve during implementation:

- Friction-classifier prompt design — what signals to weight, output format
- Extractor prompt phrasing — how to nudge toward sharp facts vs. platitudes
- Judge prompt design — rubric for correctness; rubric for "uses preload" vs "ignored preload"
- Similarity threshold T at retrieval and at dedup (likely different values)
- Backfill batching and resume mechanism
- Session-hook integration point (where in current code, what state to pass)

## Locked design decisions (Q&A audit trail)

| # | Decision |
|---|---|
| Q1 | Extraction is friction-driven, not per-prompt or per-session |
| Q2 | Friction detected by LLM classifier per prompt-response pair |
| Q3 | Memory unit = free-form text; taxonomy added later when data justifies it |
| Q4 | Extractor sees full session transcript, not a sliding window |
| Q5 | Replay validation with iteration (regenerate on failure with feedback) |
| Q6 | Dual judge: both correctness AND efficiency must pass |
| Q7 | Max 3 iterations; fail closed (discard moment, no provisional memories) |
| Q8 | Full backfill on day 1 + forward via session hook ongoing |
| Q9 | Static scoring; no reinforcement loop in v1 |
| Q10 | Trigger: top-K with similarity threshold; Framing: tagged as hints |
| Q11 | Dedup at write (cosine > 0.92 → append to seen_in_sessions) |
