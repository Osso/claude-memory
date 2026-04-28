//! LLM-based filtering and merging via llm-sdk (configurable backend).

use std::time::Duration;

use llm_sdk::Backend;

const FILTER_SYSTEM: &str = "\
You are a relevance judge for a semantic memory system.
Given a search query and numbered candidate results, identify which results are genuinely relevant.
A result is relevant if it directly addresses the query topic or contains information useful for that query.
Ignore results that merely share keywords but are about different topics.
Respond with ONLY a JSON array of 1-based indices of relevant results. Example: [1, 3, 5]
If none are relevant, respond with: []";

const MERGE_SYSTEM: &str = "\
Merge two related memory entries into one concise entry. Preserve all distinct facts from both. \
Do not add information not present in either entry. Output only the merged text.";

/// A raw search result passed to `filter_relevant`.
pub struct RawResult {
    pub text: String,
    pub score: f32,
}

fn log(msg: &str) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/claude/memory-mcp.log")
    else {
        return;
    };
    let now = chrono::Local::now().format("%H:%M:%S%.3f");
    let _ = writeln!(f, "[{now} llm] {msg}");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlmBackend {
    Anthropic,
    OpenRouter,
    Claude,
    Codex,
}

fn parse_backend() -> LlmBackend {
    match std::env::var("CLAUDE_MEMORY_LLM_BACKEND")
        .unwrap_or_default()
        .as_str()
    {
        "anthropic" => LlmBackend::Anthropic,
        "claude" => LlmBackend::Claude,
        "openrouter" => LlmBackend::OpenRouter,
        "codex" | "" => LlmBackend::Codex,
        other => {
            log(&format!(
                "unknown CLAUDE_MEMORY_LLM_BACKEND={other:?}, falling back to codex"
            ));
            LlmBackend::Codex
        }
    }
}

fn default_model_for_backend(backend: LlmBackend) -> &'static str {
    match backend {
        LlmBackend::Anthropic => "claude-haiku-4-5-20251001",
        LlmBackend::OpenRouter => "google/gemini-2.5-flash-lite",
        LlmBackend::Claude => "haiku",
        LlmBackend::Codex => "gpt-5.3-codex-spark",
    }
}

/// Build the appropriate backend and call `.complete(user)`. Returns `Some(text)` on success.
async fn complete(system: &str, user: &str, max_tokens: u32, timeout_secs: u64) -> Option<String> {
    let backend = parse_backend();
    let model = std::env::var("CLAUDE_MEMORY_LLM_MODEL")
        .unwrap_or_else(|_| default_model_for_backend(backend).to_owned());
    let timeout = Duration::from_secs(timeout_secs);

    let result: Result<llm_sdk::Output, llm_sdk::Error> = match backend {
        LlmBackend::Anthropic => {
            let b = llm_sdk::anthropic::Anthropic::new(&model)
                .api_key_env("ANTHROPIC_API_KEY")
                .system_prompt(system)
                .max_tokens(max_tokens)
                .timeout(timeout);
            b.complete(user).await
        }
        LlmBackend::OpenRouter => {
            let b = llm_sdk::openrouter::OpenRouter::new(&model)
                .api_key_env("OPENROUTER_API_KEY")
                .system_prompt(system)
                .timeout(timeout);
            b.complete(user).await
        }
        LlmBackend::Claude => {
            let b = llm_sdk::claude::Claude::new()
                .map_err(|e| {
                    log(&format!("claude backend init error: {e}"));
                })
                .ok()?
                .model(&model)
                .system_prompt(system)
                .timeout(timeout);
            b.complete(user).await
        }
        LlmBackend::Codex => {
            let b = llm_sdk::codex_cli::CodexCli::new()
                .map_err(|e| {
                    log(&format!("codex backend init error: {e}"));
                })
                .ok()?
                .model(&model)
                .system_prompt(system)
                .timeout(timeout);
            b.complete(user).await
        }
    };

    result
        .map(|o| o.text)
        .map_err(|e| log(&format!("llm error ({backend:?}): {e}")))
        .ok()
}

/// Parse a JSON array of `usize` from the LLM's raw text. Handles markdown fences.
fn parse_index_array(text: &str) -> Option<Vec<usize>> {
    if let Ok(v) = serde_json::from_str::<Vec<usize>>(text.trim()) {
        return Some(v);
    }

    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end < start {
        return None;
    }
    serde_json::from_str::<Vec<usize>>(&text[start..=end])
        .map_err(|e| log(&format!("index array parse failed: {e}")))
        .ok()
}

fn build_filter_user_message(query: &str, results: &[RawResult]) -> String {
    let mut user = format!("Query: {query}\n\nResults:\n");
    for (i, r) in results.iter().enumerate() {
        user.push_str(&format!("{}. {}\n\n", i + 1, r.text));
    }
    user
}

/// Filter a list of raw results to only those genuinely relevant to `query`.
///
/// Returns `None` if no backend is configured or if any error occurs.
pub async fn filter_relevant(query: &str, results: &[RawResult]) -> Option<Vec<usize>> {
    let user = build_filter_user_message(query, results);
    let raw = complete(FILTER_SYSTEM, &user, 200, 15).await?;
    parse_index_array(&raw).or_else(|| {
        log(&format!("could not parse indices from: {raw}"));
        None
    })
}

/// Merge two related memory entries into one concise entry.
///
/// Returns `None` if no backend is configured or if any error occurs.
pub async fn merge_memories(existing: &str, new: &str) -> Option<String> {
    let user = format!("Existing:\n{existing}\n\nNew:\n{new}");
    complete(MERGE_SYSTEM, &user, 1000, 20).await
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_index_array ---

    #[test]
    fn parse_index_array_valid() {
        let result = parse_index_array("[1, 3, 5]");
        assert_eq!(result, Some(vec![1, 3, 5]));
    }

    #[test]
    fn parse_index_array_empty() {
        let result = parse_index_array("[]");
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn parse_index_array_malformed() {
        let result = parse_index_array("not json");
        assert_eq!(result, None);
    }

    #[test]
    fn parse_index_array_with_surrounding_text() {
        let result = parse_index_array("keep these: [1, 2]");
        assert_eq!(result, Some(vec![1, 2]));
    }

    #[test]
    fn parse_index_array_markdown_fence() {
        let result = parse_index_array("```\n[2, 4]\n```");
        assert_eq!(result, Some(vec![2, 4]));
    }

    // --- build_filter_user_message ---

    #[test]
    fn build_filter_user_message_format() {
        let results = vec![
            RawResult {
                text: "alpha".to_string(),
                score: 0.9,
            },
            RawResult {
                text: "beta".to_string(),
                score: 0.7,
            },
        ];
        let msg = build_filter_user_message("my query", &results);
        assert!(msg.starts_with("Query: my query\n\nResults:\n"));
        assert!(msg.contains("1. alpha\n\n"));
        assert!(msg.contains("2. beta\n\n"));
    }

    #[test]
    fn build_filter_user_message_empty_results() {
        let msg = build_filter_user_message("anything", &[]);
        assert_eq!(msg, "Query: anything\n\nResults:\n");
    }

    // --- parse_backend ---

    #[test]
    fn parse_backend_explicit_values() {
        fn parse_str(s: &str) -> LlmBackend {
            match s {
                "anthropic" => LlmBackend::Anthropic,
                "claude" => LlmBackend::Claude,
                "openrouter" => LlmBackend::OpenRouter,
                "codex" | "" => LlmBackend::Codex,
                _ => LlmBackend::Codex,
            }
        }
        assert_eq!(parse_str("anthropic"), LlmBackend::Anthropic);
        assert_eq!(parse_str("openrouter"), LlmBackend::OpenRouter);
        assert_eq!(parse_str("claude"), LlmBackend::Claude);
        assert_eq!(parse_str("codex"), LlmBackend::Codex);
        assert_eq!(parse_str(""), LlmBackend::Codex);
        assert_eq!(parse_str("unknown"), LlmBackend::Codex);
    }

    // --- default_model_for_backend ---

    #[test]
    fn default_model_for_each_backend() {
        assert_eq!(
            default_model_for_backend(LlmBackend::Anthropic),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(
            default_model_for_backend(LlmBackend::OpenRouter),
            "google/gemini-2.5-flash-lite"
        );
        assert_eq!(default_model_for_backend(LlmBackend::Claude), "haiku");
        assert_eq!(
            default_model_for_backend(LlmBackend::Codex),
            "gpt-5.3-codex-spark"
        );
    }
}
