//! Prompt enrichment hook output.

use anyhow::{Context, Result};

use crate::{index, kb_search};

const MAX_KB_RESULTS: usize = 3;
const MAX_KB_RESULT_CHARS: usize = 500;
const MAX_SESSION_CHUNKS_PER_GROUP: usize = 3;
const RAW_SESSION_SOURCES: &[&str] = &["session", "archive"];

pub async fn run_enrich(limit: usize) -> Result<()> {
    let prompt = read_prompt_from_hook()?;
    if prompt.is_empty() {
        print_hook_output("");
        return Ok(());
    }

    let mut sections = Vec::new();
    add_session_chunk_section(&prompt, limit, &mut sections).await;
    add_kb_section(&prompt, limit, &mut sections);

    if sections.is_empty() {
        print_hook_output("");
    } else {
        print_hook_output(&sections.join("\n\n"));
    }
    Ok(())
}

fn read_prompt_from_hook() -> Result<String> {
    let input = read_hook_stdin()?;
    Ok(input["prompt"].as_str().unwrap_or("").to_string())
}

fn read_hook_stdin() -> Result<serde_json::Value> {
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
    serde_json::from_str(&buf).context("failed to parse hook input")
}

async fn add_session_chunk_section(prompt: &str, limit: usize, sections: &mut Vec<String>) {
    let session_limit = limit.min(MAX_SESSION_CHUNKS_PER_GROUP);
    let prompt_chunks = search_session_chunks(prompt, session_limit, ChunkKind::Prompt).await;
    let answer_chunks = search_session_chunks(prompt, session_limit, ChunkKind::Answer).await;

    if prompt_chunks.is_empty() && answer_chunks.is_empty() {
        return;
    }

    sections.push(format_session_chunk_results(&prompt_chunks, &answer_chunks));
}

#[derive(Clone, Copy)]
enum ChunkKind {
    Prompt,
    Answer,
}

async fn search_session_chunks(
    prompt: &str,
    limit: usize,
    kind: ChunkKind,
) -> Vec<index::SearchResult> {
    let results = match kind {
        ChunkKind::Prompt => index::search_prompt_sources(prompt, limit, RAW_SESSION_SOURCES).await,
        ChunkKind::Answer => index::search_answer_sources(prompt, limit, RAW_SESSION_SOURCES).await,
    };

    match results {
        Ok(results) => results
            .into_iter()
            .filter(|result| !result.session_id.is_empty())
            .collect(),
        Err(error) => {
            let label = match kind {
                ChunkKind::Prompt => "prompt",
                ChunkKind::Answer => "answer",
            };
            eprintln!("enrich: {label} chunk search failed: {error:#}");
            Vec::new()
        }
    }
}

fn add_kb_section(prompt: &str, limit: usize, sections: &mut Vec<String>) {
    let kb_limit = limit.min(MAX_KB_RESULTS);
    match kb_search::search_default_kb_context(prompt, kb_limit) {
        Ok(results) if !results.is_empty() => sections.push(format_kb_results(&results)),
        Ok(_) => {}
        Err(error) => eprintln!("enrich: KB PageIndex search failed: {error:#}"),
    }
}

fn print_hook_output(context: &str) {
    if context.is_empty() {
        println!("{{}}");
        return;
    }

    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context
        }
    });
    println!("{output}");
}

fn format_session_chunk_results(
    prompt_results: &[index::SearchResult],
    answer_results: &[index::SearchResult],
) -> String {
    let mut out = String::from(
        "## Relevant past session chunks (raw prompt/answer history; use session_id to inspect source transcript)",
    );
    append_session_chunk_group(&mut out, "prompts", prompt_results);
    append_session_chunk_group(&mut out, "answers", answer_results);
    out
}

fn append_session_chunk_group(out: &mut String, label: &str, results: &[index::SearchResult]) {
    if results.is_empty() {
        return;
    }

    out.push_str(&format!("\n### {label}"));
    for result in results {
        let text = preview_text(&result.text, 300);
        let history_source = transcript_history_source(result);
        out.push_str(&format!(
            "\n- ({:.2}) source: {} source_type: {} session_id: {}\n  path: {}\n  text: {}",
            result.score, history_source, result.source, result.session_id, result.path, text
        ));
    }
}

fn transcript_history_source(result: &index::SearchResult) -> &'static str {
    if result.path.contains(".codex") {
        return "codex";
    }

    match result.source.as_str() {
        "session" | "archive" => "claude",
        _ => "unknown",
    }
}

fn format_kb_results(results: &[kb_search::KbSearchResult]) -> String {
    let mut out = String::from("## Relevant KB notes (KB PageIndex)");
    for result in results {
        let text = preview_text(&result.text, MAX_KB_RESULT_CHARS);
        out.push_str(&format!(
            "\n- ({}) {} > {} [{}]: {}\n  next: {}",
            result.score,
            result.path,
            result.heading,
            result.node_id,
            text,
            result.next_content_command
        ));
    }
    out
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let text = text.replace('\n', " ");
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_text_index_result_formats_for_enrich_with_explicit_content_roots() {
        let root = std::env::temp_dir().join(format!("enrich-kb-{}", uuid::Uuid::new_v4()));
        let kb_dir = root.join("knowledge base");
        let index_dir = root.join("text index");
        std::fs::create_dir_all(&kb_dir).unwrap();
        std::fs::write(
            kb_dir.join("rules.md"),
            "# Rules\n\n## Frontend\nLoad frontend design skill immediately.\n",
        )
        .unwrap();
        kb_search::build_text_index(&kb_dir, &index_dir).unwrap();

        let results = kb_search::search_kb_context(
            &kb_dir,
            &index_dir,
            "frontend design skill immediately",
            MAX_KB_RESULTS,
        )
        .unwrap();
        let formatted = format_kb_results(&results);

        assert!(formatted.contains("Relevant KB notes (KB PageIndex)"));
        assert!(formatted.contains("rules.md > Rules > Frontend"));
        assert!(formatted.contains(&format!("--kb '{}'", kb_dir.display())));
        assert!(formatted.contains(&format!("--index '{}'", index_dir.display())));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn kb_results_include_source_path_and_heading() {
        let results = vec![kb_search::KbSearchResult {
            doc_id: "memory/corrections.md".to_string(),
            path: "memory/corrections.md".to_string(),
            heading: "Corrections > Process".to_string(),
            text: "Load frontend design skill immediately.".to_string(),
            score: 44,
            node_id: "000002".to_string(),
            title: "Process".to_string(),
            reason: "matched 4 query terms".to_string(),
            content_command: "claude-memory kb-page-index content memory/corrections.md 000002"
                .to_string(),
            next_content_command:
                "claude-memory kb-page-index content memory/corrections.md 000002".to_string(),
        }];

        let formatted = format_kb_results(&results);

        assert!(formatted.contains("Relevant KB notes"));
        assert!(formatted.contains("memory/corrections.md"));
        assert!(formatted.contains("Corrections > Process"));
    }

    #[test]
    fn kb_results_are_capped_for_hook_output() {
        let long_text = format!("{} tail marker", "a".repeat(700));
        let results = vec![kb_search::KbSearchResult {
            doc_id: "memory/corrections.md".to_string(),
            path: "memory/corrections.md".to_string(),
            heading: "Corrections > Process".to_string(),
            text: long_text,
            score: 44,
            node_id: "000002".to_string(),
            title: "Process".to_string(),
            reason: "matched query terms: corrections".to_string(),
            content_command: "claude-memory kb-page-index content memory/corrections.md 000002"
                .to_string(),
            next_content_command:
                "claude-memory kb-page-index content memory/corrections.md 000002".to_string(),
        }];

        let formatted = format_kb_results(&results);

        assert!(formatted.contains("Relevant KB notes (KB PageIndex)"));
        assert!(!formatted.contains("tail marker"));
        assert!(formatted.contains("..."));
    }

    #[test]
    fn session_chunk_results_include_session_id() {
        let prompt_results = vec![index::SearchResult {
            record_type: "prompt".to_string(),
            text: "User asked about transcript page indexes.".to_string(),
            source: "session".to_string(),
            path: "project/session-1.jsonl".to_string(),
            session_id: "session-1".to_string(),
            score: 0.82,
        }];
        let answer_results = vec![index::SearchResult {
            record_type: "answer".to_string(),
            text: "Assistant explained how chunks are indexed.".to_string(),
            source: "session".to_string(),
            path: "project/session-2.jsonl".to_string(),
            session_id: "session-2".to_string(),
            score: 0.79,
        }];

        let formatted = format_session_chunk_results(&prompt_results, &answer_results);

        assert!(formatted.contains("Relevant past session chunks"));
        assert!(formatted.contains("prompts"));
        assert!(formatted.contains("answers"));
        assert!(formatted.contains("source: claude source_type: session session_id: session-1"));
        assert!(formatted.contains("source: claude source_type: session session_id: session-2"));
    }

    #[test]
    fn transcript_history_source_uses_codex_path_when_available() {
        let result = index::SearchResult {
            record_type: "answer".to_string(),
            text: "Codex answer".to_string(),
            source: "session".to_string(),
            path: ".codex/sessions/2026/06/session.jsonl".to_string(),
            session_id: "session".to_string(),
            score: 0.81,
        };

        assert_eq!(transcript_history_source(&result), "codex");
    }
}
