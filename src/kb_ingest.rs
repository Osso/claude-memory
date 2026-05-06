//! KB Markdown fact ingestion into memory units.

use anyhow::{Context, Result};
use chrono::{Local, Utc};
use qdrant_client::Qdrant;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

use crate::embed::Embedder;
use crate::llm;
use crate::memory_unit::{DedupOutcome, MemoryUnit, upsert_with_dedup};

const FACT_EXTRACT_SYSTEM: &str = "\
Extract durable operational facts from a Markdown knowledge-base section.
Return only facts that would help a future coding assistant skip investigation.
Keep each fact self-contained, concrete, and no more than 2 sentences.
Discard prose, status updates without reusable value, generic advice, and vague summaries.
Respond ONLY with JSON: {\"facts\": [\"...\"]}.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KbSection {
    pub path: String,
    pub heading: String,
    pub text: String,
}

#[derive(Debug)]
pub struct KbIngestSummary {
    pub files: usize,
    pub sections: usize,
    pub facts: usize,
    pub inserted: usize,
    pub merged: usize,
}

#[derive(Deserialize)]
struct FactsJson {
    facts: Vec<String>,
}

pub async fn ingest_kb_dir(
    kb_dir: &Path,
    max_files: Option<usize>,
    dry_run: bool,
) -> Result<KbIngestSummary> {
    let files = collect_markdown_files(kb_dir, max_files);
    let client = if dry_run {
        None
    } else {
        let client = Qdrant::from_url(crate::index::QDRANT_URL)
            .build()
            .context("failed to connect to Qdrant")?;
        crate::memory_unit::ensure_memory_units_collection(&client).await?;
        Some(client)
    };
    let embedder = Embedder::new();
    let mut summary = KbIngestSummary {
        files: files.len(),
        sections: 0,
        facts: 0,
        inserted: 0,
        merged: 0,
    };

    for (index, path) in files.iter().enumerate() {
        eprintln!(
            "KB file {}/{}: {}",
            index + 1,
            files.len(),
            relative_path(kb_dir, path).display()
        );
        ingest_kb_file(kb_dir, path, client.as_ref(), &embedder, &mut summary).await?;
    }

    Ok(summary)
}

async fn ingest_kb_file(
    kb_dir: &Path,
    path: &Path,
    client: Option<&Qdrant>,
    embedder: &Embedder,
    summary: &mut KbIngestSummary,
) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let sections = split_markdown_sections(&relative_path(kb_dir, path), &content);
    summary.sections += sections.len();

    let section_count = sections.len();
    for (index, section) in sections.into_iter().enumerate() {
        eprintln!(
            "  section {}/{}: {}",
            index + 1,
            section_count,
            section.heading
        );
        let facts = extract_section_facts(&section).await;
        eprintln!("    facts: {}", facts.len());
        summary.facts += facts.len();
        store_section_facts(client, embedder, &section, facts, summary).await?;
    }

    Ok(())
}

async fn store_section_facts(
    client: Option<&Qdrant>,
    embedder: &Embedder,
    section: &KbSection,
    facts: Vec<String>,
    summary: &mut KbIngestSummary,
) -> Result<()> {
    let Some(client) = client else {
        return Ok(());
    };

    for fact in facts {
        match upsert_kb_fact(client, embedder, section, fact).await? {
            DedupOutcome::Inserted(_) => summary.inserted += 1,
            DedupOutcome::Merged(_) => summary.merged += 1,
        }
    }

    Ok(())
}

pub fn split_markdown_sections(path: &Path, markdown: &str) -> Vec<KbSection> {
    let path = path.to_string_lossy().to_string();
    let mut headings: Vec<String> = Vec::new();
    let mut current_heading = fallback_heading(&path);
    let mut current_text = String::new();
    let mut sections = Vec::new();

    for line in markdown.lines() {
        if let Some((level, title)) = parse_heading(line) {
            push_section(&mut sections, &path, &current_heading, &current_text);
            headings.truncate(level.saturating_sub(1));
            headings.push(title);
            current_heading = headings.join(" > ");
            current_text.clear();
            continue;
        }

        current_text.push_str(line);
        current_text.push('\n');
    }

    push_section(&mut sections, &path, &current_heading, &current_text);
    sections
}

pub fn fallback_facts(section: &KbSection) -> Vec<String> {
    section
        .text
        .lines()
        .filter_map(strip_list_marker)
        .map(str::trim)
        .filter(|line| line.len() >= 12)
        .filter(|line| line.ends_with('.') || line.ends_with('`'))
        .map(ToOwned::to_owned)
        .collect()
}

pub fn parse_fact_json(raw: &str) -> Result<Vec<String>> {
    let json = extract_json_object(raw);
    let parsed: FactsJson = serde_json::from_str(json)
        .with_context(|| format!("fact JSON parse failed | raw: {raw}"))?;
    Ok(parsed
        .facts
        .into_iter()
        .map(|fact| fact.trim().to_string())
        .filter(|fact| !fact.is_empty())
        .collect())
}

async fn extract_section_facts(section: &KbSection) -> Vec<String> {
    if section.text.trim().is_empty() {
        return Vec::new();
    }

    let user = format!(
        "Path: {}\nHeading: {}\n\nMarkdown section:\n{}",
        section.path, section.heading, section.text
    );
    eprintln!("    [{}] LLM start", timestamp());
    let started = Instant::now();
    match llm::complete(FACT_EXTRACT_SYSTEM, &user, 700, 60).await {
        Ok(raw) => {
            eprintln!("    [{}] LLM done ({:.1?})", timestamp(), started.elapsed());
            parse_fact_json(&raw).unwrap_or_else(|_| fallback_facts(section))
        }
        Err(error) => {
            eprintln!(
                "    [{}] LLM failed ({:.1?}): {error}",
                timestamp(),
                started.elapsed()
            );
            fallback_facts(section)
        }
    }
}

async fn upsert_kb_fact(
    client: &Qdrant,
    embedder: &Embedder,
    section: &KbSection,
    fact: String,
) -> Result<DedupOutcome> {
    eprintln!("    [{}] embed/upsert start", timestamp());
    let started = Instant::now();
    let unit = MemoryUnit {
        text: format!("{}: {fact}", section.heading),
        created_at: Utc::now(),
        source: "kb".to_string(),
        source_session: section.path.clone(),
        source_turn: 0,
        category: Some("kb".to_string()),
        project: None,
        seen_in_sessions: vec![section.path.clone()],
    };
    let outcome = upsert_with_dedup(client, embedder, unit).await;
    eprintln!(
        "    [{}] embed/upsert done ({:.1?})",
        timestamp(),
        started.elapsed()
    );
    outcome
}

fn timestamp() -> String {
    Local::now().format("%H:%M:%S%.3f").to_string()
}

fn collect_markdown_files(kb_dir: &Path, max_files: Option<usize>) -> Vec<PathBuf> {
    if !kb_dir.exists() {
        return Vec::new();
    }

    let mut files: Vec<PathBuf> = WalkDir::new(kb_dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    files.sort();
    if let Some(limit) = max_files {
        files.truncate(limit);
    }
    files
}

fn relative_path(base: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(base).unwrap_or(path).to_path_buf()
}

fn fallback_heading(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if level == 0 || level > 6 {
        return None;
    }

    let title = trimmed.get(level..)?.trim();
    if title.is_empty() {
        return None;
    }
    Some((level, title.trim_matches('#').trim().to_string()))
}

fn push_section(sections: &mut Vec<KbSection>, path: &str, heading: &str, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }

    sections.push(KbSection {
        path: path.to_string(),
        heading: heading.to_string(),
        text: text.to_string(),
    });
}

fn strip_list_marker(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("- ") {
        return Some(rest);
    }
    if let Some(rest) = trimmed.strip_prefix("* ") {
        return Some(rest);
    }
    strip_numbered_marker(trimmed)
}

fn strip_numbered_marker(line: &str) -> Option<&str> {
    let dot = line.find(". ")?;
    if line[..dot].chars().all(|ch| ch.is_ascii_digit()) {
        Some(&line[dot + 2..])
    } else {
        None
    }
}

fn extract_json_object(raw: &str) -> &str {
    let text = raw.trim();
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .unwrap_or(text)
        .trim();
    let text = text.strip_suffix("```").unwrap_or(text).trim();
    let start = text.find('{').unwrap_or(0);
    let end = text.rfind('}').map(|index| index + 1).unwrap_or(text.len());
    &text[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn markdown_sections_keep_heading_path() {
        let markdown =
            "# Root\nIntro\n\n## Setup\nUse uv for Python packages.\n\n## Deploy\nRun cargo test.";

        let sections = split_markdown_sections(Path::new("guides/dev.md"), markdown);

        assert_eq!(sections.len(), 3);
        assert_eq!(sections[1].heading, "Root > Setup");
        assert_eq!(sections[1].text, "Use uv for Python packages.");
        assert_eq!(sections[1].path, "guides/dev.md");
    }

    #[test]
    fn fallback_facts_extract_concrete_bullets_only() {
        let section = KbSection {
            path: "guides/dev.md".to_string(),
            heading: "Setup".to_string(),
            text: "- Use uv instead of pip.\n- nice\nParagraph without marker.".to_string(),
        };

        let facts = fallback_facts(&section);

        assert_eq!(facts, vec!["Use uv instead of pip."]);
    }

    #[test]
    fn parse_fact_json_accepts_array_field() {
        let raw = r#"{"facts":["Use uv instead of pip.","Rust projects use edition 2024."]}"#;

        let facts = parse_fact_json(raw).unwrap();

        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0], "Use uv instead of pip.");
    }
}
