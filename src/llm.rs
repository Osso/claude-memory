//! LLM-based filtering and merging via llm-sdk (configurable backend).

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Result, anyhow};
use chrono::{DateTime, Local, NaiveTime};
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

const CODEX_FALLBACK_MODEL: &str = "gpt-5.4-mini";

static CODEX_COOLDOWN_UNTIL: OnceLock<Mutex<Option<DateTime<Local>>>> = OnceLock::new();

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

/// Build the appropriate backend and call `.complete(user)`.
/// Returns `Err` on auth/connection/timeout failures.
pub async fn complete(
    system: &str,
    user: &str,
    max_tokens: u32,
    timeout_secs: u64,
) -> Result<String> {
    let backend = parse_backend();
    let model = std::env::var("CLAUDE_MEMORY_LLM_MODEL")
        .unwrap_or_else(|_| default_model_for_backend(backend).to_owned());
    let timeout = Duration::from_secs(timeout_secs);

    let result = complete_with_backend(backend, &model, system, user, max_tokens, timeout).await;

    match result {
        Ok(o) => Ok(o.text),
        Err(e) => {
            let msg = format!("llm error ({backend:?}): {e}");
            log(&msg);
            Err(anyhow!(msg))
        }
    }
}

async fn complete_with_backend(
    backend: LlmBackend,
    model: &str,
    system: &str,
    user: &str,
    max_tokens: u32,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    match backend {
        LlmBackend::Anthropic => {
            complete_with_anthropic(model, system, user, max_tokens, timeout).await
        }
        LlmBackend::OpenRouter => complete_with_openrouter(model, system, user, timeout).await,
        LlmBackend::Claude => complete_with_claude(model, system, user, timeout).await,
        LlmBackend::Codex => complete_with_codex_fallback(model, system, user, timeout).await,
    }
}

async fn complete_with_anthropic(
    model: &str,
    system: &str,
    user: &str,
    max_tokens: u32,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    let backend = llm_sdk::anthropic::Anthropic::new(model)
        .api_key_env("ANTHROPIC_API_KEY")
        .system_prompt(system)
        .max_tokens(max_tokens)
        .timeout(timeout);
    backend.complete(user).await
}

async fn complete_with_openrouter(
    model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    let backend = llm_sdk::openrouter::OpenRouter::new(model)
        .api_key_env("OPENROUTER_API_KEY")
        .system_prompt(system)
        .timeout(timeout);
    backend.complete(user).await
}

async fn complete_with_claude(
    model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    let backend = llm_sdk::claude::Claude::new()
        .map_err(|error| llm_sdk::Error::Parse(format!("claude backend init failed: {error}")))?
        .model(model)
        .system_prompt(system)
        .timeout(timeout)
        .stdin_prompt()
        .extra_arg("--tools")
        .extra_arg("");
    backend.complete(user).await
}

async fn complete_with_codex_fallback(
    primary_model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    let cooldown_until = codex_cooldown_until();
    let selected_model = codex_model_for_time(primary_model, cooldown_until, Local::now());
    let first_result = complete_with_codex(&selected_model, system, user, timeout).await;
    match first_result {
        Ok(output) => Ok(output),
        Err(error) => {
            retry_codex_after_usage_limit(error, primary_model, system, user, timeout).await
        }
    }
}

async fn complete_with_codex(
    model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    // Disable all tool features so codex acts as a pure completion API.
    // With tools enabled, the agent calls shell/apply_patch/etc on technical
    // prompts, ballooning input tokens to 200K+ and timing out.
    let backend = llm_sdk::codex_cli::CodexCli::new()?
        .model(model)
        .system_prompt(system)
        .timeout(timeout)
        .extra_config("model_reasoning_effort=\"low\"")
        .extra_config("web_search=\"disabled\"")
        .extra_config("features.shell_tool=false")
        .extra_config("features.include_apply_patch_tool=false")
        .extra_config("features.tool_search=false")
        .extra_config("features.tool_suggest=false")
        .extra_config("features.memory_tool=false")
        .extra_config("features.request_permissions_tool=false");
    backend.complete(user).await
}

async fn retry_codex_after_usage_limit(
    error: llm_sdk::Error,
    primary_model: &str,
    system: &str,
    user: &str,
    timeout: Duration,
) -> std::result::Result<llm_sdk::Output, llm_sdk::Error> {
    let message = error.to_string();
    let Some(retry_at) = parse_codex_retry_time(&message, Local::now()) else {
        return Err(error);
    };

    set_codex_cooldown_until(retry_at);
    log(&format!(
        "codex model {primary_model} limited until {}; using {CODEX_FALLBACK_MODEL}",
        retry_at.format("%Y-%m-%d %H:%M:%S %Z")
    ));
    complete_with_codex(CODEX_FALLBACK_MODEL, system, user, timeout).await
}

fn codex_model_for_time(
    primary_model: &str,
    cooldown_until: Option<DateTime<Local>>,
    now: DateTime<Local>,
) -> String {
    match cooldown_until {
        Some(until) if now < until => CODEX_FALLBACK_MODEL.to_string(),
        _ => primary_model.to_string(),
    }
}

fn codex_cooldown_until() -> Option<DateTime<Local>> {
    let lock = CODEX_COOLDOWN_UNTIL.get_or_init(|| Mutex::new(None));
    lock.lock().ok().and_then(|guard| *guard)
}

fn set_codex_cooldown_until(retry_at: DateTime<Local>) {
    let lock = CODEX_COOLDOWN_UNTIL.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = lock.lock() {
        *guard = Some(retry_at);
    }
}

fn parse_codex_retry_time(message: &str, now: DateTime<Local>) -> Option<DateTime<Local>> {
    let marker = "try again at ";
    let start = message.find(marker)? + marker.len();
    let time_text = message[start..].split(['.', '\n', '\r']).next()?.trim();
    let retry_time = NaiveTime::parse_from_str(time_text, "%I:%M %p").ok()?;
    let mut retry_at = now
        .date_naive()
        .and_time(retry_time)
        .and_local_timezone(Local)
        .single()?;
    if retry_at <= now {
        retry_at += chrono::Duration::days(1);
    }
    Some(retry_at)
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
    let raw = complete(FILTER_SYSTEM, &user, 200, 15).await.ok()?;
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
    complete(MERGE_SYSTEM, &user, 1000, 20).await.ok()
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

    #[test]
    fn parse_codex_usage_limit_time_extracts_today_time() {
        let message = "You've hit your usage limit for GPT-5.3-Codex-Spark. Switch to another model now, or try again at 10:36 PM.";
        let now = chrono::Local::now();

        let retry_at = parse_codex_retry_time(message, now).unwrap();

        assert_eq!(retry_at.format("%I:%M %p").to_string(), "10:36 PM");
    }

    #[test]
    fn parse_codex_usage_limit_time_rolls_past_time_to_tomorrow() {
        let message = "try again at 12:01 AM";
        let today = chrono::Local::now().date_naive();
        let now = today
            .and_hms_opt(23, 0, 0)
            .unwrap()
            .and_local_timezone(chrono::Local)
            .single()
            .unwrap();

        let retry_at = parse_codex_retry_time(message, now).unwrap();

        assert!(retry_at > now);
    }

    #[test]
    fn codex_model_uses_fallback_during_cooldown() {
        let now = chrono::Local::now();
        let cooldown_until = now + chrono::Duration::minutes(5);

        let model = codex_model_for_time("gpt-5.3-codex-spark", Some(cooldown_until), now);

        assert_eq!(model, "gpt-5.4-mini");
    }

    #[test]
    fn codex_model_uses_primary_after_cooldown() {
        let now = chrono::Local::now();
        let cooldown_until = now - chrono::Duration::minutes(5);

        let model = codex_model_for_time("gpt-5.3-codex-spark", Some(cooldown_until), now);

        assert_eq!(model, "gpt-5.3-codex-spark");
    }
}
