//! LLM-based filtering and merging via the Anthropic Claude Haiku API.

use std::time::Duration;

const MODEL: &str = "claude-haiku-4-5-20251001";

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

fn build_request_body(system: &str, user: &str, max_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}]
    })
}

async fn send_request(
    body: serde_json::Value,
    key: &str,
    timeout_secs: u64,
) -> Option<serde_json::Value> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .map_err(|e| log(&format!("request error: {e}")))
        .ok()?;

    let status = response.status();
    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| log(&format!("response parse error: {e}")))
        .ok()?;

    if !status.is_success() {
        log(&format!("API error {status}: {json}"));
        return None;
    }

    Some(json)
}

fn extract_text_content(json: &serde_json::Value) -> Option<String> {
    let text = json
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|b| b.get("text"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_owned());

    if text.is_none() {
        log(&format!("unexpected response shape: {json}"));
    }

    text
}

/// Call Claude Haiku with a system prompt and user message. Returns the text of the first
/// content block, or `None` on any error.
async fn call_haiku(
    system: &str,
    user: &str,
    max_tokens: u32,
    timeout_secs: u64,
) -> Option<String> {
    let key = std::env::var("ANTHROPIC_API_KEY").ok()?;
    let body = build_request_body(system, user, max_tokens);
    let json = send_request(body, &key, timeout_secs).await?;
    extract_text_content(&json)
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
/// Returns `None` if the `ANTHROPIC_API_KEY` environment variable is unset or if
/// any error occurs (network, parsing, etc.).
pub async fn filter_relevant(query: &str, results: &[RawResult]) -> Option<Vec<usize>> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        return None;
    }

    let user = build_filter_user_message(query, results);
    let raw = call_haiku(FILTER_SYSTEM, &user, 200, 15).await?;
    parse_index_array(&raw).or_else(|| {
        log(&format!("could not parse indices from: {raw}"));
        None
    })
}

/// Merge two related memory entries into one concise entry.
///
/// Returns `None` if the `ANTHROPIC_API_KEY` environment variable is unset or if
/// any error occurs.
pub async fn merge_memories(existing: &str, new: &str) -> Option<String> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        return None;
    }

    let user = format!("Existing:\n{existing}\n\nNew:\n{new}");
    call_haiku(MERGE_SYSTEM, &user, 1000, 20).await
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

    // --- extract_text_content ---

    #[test]
    fn extract_text_content_simple() {
        let json = serde_json::json!({
            "content": [{"type": "text", "text": "hello world"}]
        });
        assert_eq!(extract_text_content(&json), Some("hello world".to_string()));
    }

    #[test]
    fn extract_text_content_missing_content_key() {
        let json = serde_json::json!({"id": "msg_123"});
        assert_eq!(extract_text_content(&json), None);
    }

    #[test]
    fn extract_text_content_empty_content_array() {
        let json = serde_json::json!({"content": []});
        assert_eq!(extract_text_content(&json), None);
    }

    #[test]
    fn extract_text_content_missing_text_field() {
        let json = serde_json::json!({
            "content": [{"type": "tool_use", "id": "tool_1"}]
        });
        assert_eq!(extract_text_content(&json), None);
    }
}
