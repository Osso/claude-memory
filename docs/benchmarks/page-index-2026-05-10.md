# PageIndex Benchmark - 2026-05-10

Historical benchmark captured before the memory-unit and graph runtime
retirement. The `claude-memory enrich` row measures the pre-retirement
composition and is not a current latency or source-composition claim.

This benchmark compares the PageIndex implementation against two
baselines available on this machine:

- `rg` over `/syncthing/Sync/KB/**/*.md`
- legacy `claude-memory search --type prompts`

The goal is not a synthetic throughput contest. The useful question is whether
PageIndex gives traceable document structure and exact content fetches cheaply
enough for CLI and hook use.

## Corpus

| Corpus | Size | Notes |
| --- | ---: | --- |
| KB Markdown | 116 Markdown files, 2.0 MiB | `/syncthing/Sync/KB` |
| Transcript sources | 20,007 JSONL/ZST files | `~/.claude/projects`, `~/.claude/archive`, `~/.codex/sessions`, `~/.codex/archived_sessions` |

Commands were run from this repo using `target/debug/claude-memory` after
`cargo build -q --bin claude-memory`.

## Build Time And Output Size

| Build | Command | Result |
| --- | --- | --- |
| KB PageIndex | `claude-memory kb-page-index build --kb /syncthing/Sync/KB --output <tmp>` | 0.48s elapsed, 116 files, 2,130 nodes, 5.9 MiB output |
| Transcript PageIndex, bounded | `claude-memory transcript-page-index build --max-sessions 500 --output <tmp>` | 2.13s elapsed, 424 sessions, 1,749 nodes, 9.7 MiB output |
| Transcript PageIndex, full corpus | `claude-memory transcript-page-index build --output <tmp>` | stopped after 329.57s; partial output had already reached 1.1 GiB |

Full transcript indexing is not yet acceptable as an always-run benchmark or
hook-adjacent workflow. It needs batching/resume, output compaction, and a
progress indicator before treating full-corpus builds as routine.

## Query Latency

Ten prompts were used for the query benchmark:

1. `local network router`
2. `rust graphics toolkit`
3. `pageindex reference commit`
4. `claude transcripts archive`
5. `cargo install deploy`
6. `home assistant zwave`
7. `docker uv python packages`
8. `codex sessions friction memory`
9. `ynab amex`
10. `solar smart home`

| Query path | Corpus | Time for 10 prompts | Mean per prompt | Notes |
| --- | --- | ---: | ---: | --- |
| `rg -i` OR-term scan | KB Markdown | 0.07s | 0.007s | Fastest, but output was 129 KiB and noisy because every term match is returned without document structure. |
| `claude-memory search --type prompts --limit 3` | legacy prompt/vector search | 1.24s | 0.124s | Fast, but results are transcript chunks and often not KB-specific. |
| `kb-page-index query --mode lexical --limit 3` | KB PageIndex | 5.43s | 0.543s | Returns document id, node id, score/reason, and exact `content` command. Cost is dominated by loading/parsing the JSON index per CLI process. |
| `transcript-page-index query --mode lexical --limit 3` | 424-session bounded transcript index | 1.46s | 0.146s | Faster than KB PageIndex on this smaller index, but lexical transcript quality is noisy on broad prompts. |
| `claude-memory enrich` (pre-retirement) | memory units + KB PageIndex | 9.17s | 0.917s | Historical measurement only; not the current enrich source composition. |

The default KB PageIndex cache initially failed hook lookup with `missing field
title`, meaning it was still in the old schema. Rebuilding the default KB index
fixed the benchmark path.

## Query Quality Notes

| Prompt | KB PageIndex observation | `rg` observation | Legacy prompt search observation |
| --- | --- | --- | --- |
| `local network router` | Found a Firefox tab archive node matching all terms, not a high-quality router note. | Returned many broad networking/router/bookmark hits. | Found router-related transcript chunks, including a Calix gateway discussion. |
| `rust graphics toolkit` | Strong result: `research/rust-ui-libraries.md` and exact graphics-stack nodes. | Found relevant Rust files but with many unrelated Rust/package hits. | Returned graphics-ish transcript chunks, less directly useful than KB PageIndex. |
| `pageindex reference commit` | No KB note. | Found generic `reference` matches, not PageIndex-specific. | Returned unrelated transcript chunks. |
| `claude transcripts archive` | No KB note. | Broad Claude/archive matches. | Transcript search is the better surface for this category. |
| `cargo install deploy` | No KB note. | Broad deployment/package matches. | Mixed transcript hits. |
| `home assistant zwave` | Matched bookmark archive, likely useful but coarse. | Found smart-home/Zigbee/Z-Wave references with noisy surrounding hits. | Returned smart-home/Z-Wave transcript chunks. |
| `docker uv python packages` | Useful package inventory hits. | Useful but verbose package hits. | Found package inventory chunks inside transcript search. |
| `codex sessions friction memory` | Found memory decision notes; useful but not transcript-specific. | Broad memory/codex hits. | Found analyzer-memory transcript chunks. |
| `ynab amex` | No KB note. | Broad finance/payment/bookmark matches. | Mostly unrelated payment/geospatial transcript chunks. |
| `solar smart home` | Matched bookmark archive, likely useful but coarse. | Broad smart-home/solar matches. | Returned smart-home transcript chunks. |

## Conclusion

KB PageIndex is worth keeping for KB retrieval because it gives structured,
traceable hits and exact content fetches that `rg` and legacy prompt search do
not provide. It is not a raw-speed replacement for `rg`; for debugging literal
matches, `rg` remains the baseline.

Transcript PageIndex should stay separate and should not be injected by default.
The bounded query path is fast enough for CLI exploration, but full-corpus build
size and runtime are the next bottleneck.

Hook-time KB enrichment is viable at current caps (`MAX_KB_RESULTS = 3`,
500-character previews), but the JSON index load cost makes repeated CLI
invocations slower than necessary. A future optimization should cache the parsed
index for daemon/MCP use or move the index to a more compact read format.
