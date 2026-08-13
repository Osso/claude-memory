# Embedding Profile Benchmark — 2026-08-13

Comparison of the local rollback profile and the hosted production profile. Configuration and runtime behavior are specified in [prompt-answer-history](../specs/prompt-answer-history.md); implementation details are in the [session-history wiki](../wiki/systems/prompt-answer-history.md).

## Profiles and corpus

| Profile | Collection | Provider/model | Dense dimensions | Points |
| --- | --- | --- | ---: | ---: |
| Rollback | `claude-session-history` | Ollama `qwen3-embedding:0.6b-ctx2048` | 1024 | 142,715 |
| Production | `claude-session-history-qwen3-8b` | OpenRouter `qwen/qwen3-embedding-8b` | 4096 | 142,715 |

Both collections were `green` with `optimizer_status=ok` at final verification.

Full-scroll integrity checks matched:

- Identity + payload SHA-256: `43a2e31dbf7796e5d55fd09388ad58982faee727041958924994a8d1230fd0e2`
- BM25 vector SHA-256: `ca54775667366aad5a209e0d9f108860dc11bdd7b55aea48a7cf2aa9d7c8d1bc`

Storage usage was 924 MiB for the source and 2.6 GiB for the target. Reported migration embedding usage was 45,531,865 tokens at approximately $0.525, excluding smoke and evaluation queries.

## Safety and operations

Per-request zero-data-retention routing was added and verified before the final 24,015-point resume. The source briefly entered `red` during bulk writes with Qdrant reporting `ENOSPC`; it recovered to `green` after a no-op optimizer update. No restart or collection deletion was performed.

## Retrieval comparison

Eight representative technical queries produced this qualitative result:

- 8B won: 5
- 0.6B won: 1
- Ties: 2
- Same top-1 result: 2/8
- Mean top-5 session overlap: 1.0/5

Latency:

- Rollback 0.6B median: 0.134 s
- Production 8B median: 0.602 s
- Production 8B mean: 1.686 s
- Production 8B maximum: 8.916 s

## Conclusion

The eight-query qualitative comparison favored 8B (5 wins, 1 loss, 2 ties), but is not sufficient alone to establish a general retrieval-quality advantage. The later OpenRouter 8B cutover was approved and performed after the comparison and operational validation. The 0.6B Ollama collection remains intact as the private, low-latency rollback profile.

## Limitations

This was a small qualitative set, not a statistically powered benchmark. Query instructions differed between profiles; no NDCG or other labeled relevance scores were collected; and raw cosine scores are not comparable across embedding models.
