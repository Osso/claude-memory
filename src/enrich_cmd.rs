//! Prompt enrichment hook output.

use anyhow::{Context, Result};

use crate::{config, graph, index, kb_search, memory_unit};

const MIN_MEMORY_SCORE: f32 = 0.65;
const MAX_KB_RESULTS: usize = 3;

pub async fn run_enrich(limit: usize) -> Result<()> {
    let prompt = read_prompt_from_hook()?;
    if prompt.is_empty() {
        print_hook_output("");
        return Ok(());
    }

    let mut sections = Vec::new();
    add_memory_section(&prompt, limit, &mut sections).await;
    add_kb_section(&prompt, limit, &mut sections);
    add_graph_section(&prompt, &mut sections).await;

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

async fn add_memory_section(prompt: &str, limit: usize, sections: &mut Vec<String>) {
    let units = match memory_unit::search(prompt, limit).await {
        Ok(results) => results,
        Err(error) => {
            eprintln!("enrich: memory-units search failed: {error:#}");
            Vec::new()
        }
    };

    let relevant_units: Vec<&index::SearchResult> = units
        .iter()
        .filter(|result| result.score >= MIN_MEMORY_SCORE)
        .collect();
    if !relevant_units.is_empty() {
        sections.push(format_memory_unit_results(&relevant_units));
    }
}

fn add_kb_section(prompt: &str, limit: usize, sections: &mut Vec<String>) {
    let kb_limit = limit.min(MAX_KB_RESULTS);
    match kb_search::search_default_kb(prompt, kb_limit) {
        Ok(results) if !results.is_empty() => sections.push(format_kb_results(&results)),
        Ok(_) => {}
        Err(error) => eprintln!("enrich: KB search failed: {error:#}"),
    }
}

async fn add_graph_section(prompt: &str, sections: &mut Vec<String>) {
    if !config::graph_enabled() {
        return;
    }

    let entities = graph::find_concepts(prompt).await;
    if entities.is_empty() {
        return;
    }

    if let Ok(related) = graph::query_related(&entities)
        && !related.is_empty()
    {
        sections.push(format_graph_results(&related));
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

fn format_memory_unit_results(results: &[&index::SearchResult]) -> String {
    let mut out = String::from(
        "## Possibly-useful preloads (from prior sessions, may be stale or wrong; treat as hints, not facts)",
    );
    for result in results {
        let text = preview_text(&result.text, 300);
        out.push_str(&format!("\n- ({:.2}) {}", result.score, text));
    }
    out
}

fn format_kb_results(results: &[kb_search::KbSearchResult]) -> String {
    let mut out = String::from("## Relevant KB notes (KB PageIndex)");
    for result in results {
        let text = result.text.replace('\n', " ");
        out.push_str(&format!(
            "\n- ({}) {} > {}: {}",
            result.score, result.path, result.heading, text
        ));
    }
    out
}

fn format_graph_results(related: &[String]) -> String {
    let mut out = String::from("Graph context:");
    for result in related.iter().take(20) {
        out.push_str(&format!("\n- {result}"));
    }
    if related.len() > 20 {
        out.push_str(&format!("\n  ...and {} more", related.len() - 20));
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
}
