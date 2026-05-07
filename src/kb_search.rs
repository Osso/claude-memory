//! Heading-aware raw Markdown search for the local knowledge base.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const MIN_SCORE: usize = 12;
const SNIPPET_CHARS: usize = 420;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KbSearchResult {
    pub path: String,
    pub heading: String,
    pub text: String,
    pub score: usize,
}

#[derive(Debug, Clone)]
struct KbSection {
    path: String,
    heading: String,
    text: String,
}

pub fn search_default_kb(query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    search_kb(Path::new(DEFAULT_KB_DIR), query, limit)
}

pub fn search_kb(kb_dir: &Path, query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    if limit == 0 || !kb_dir.exists() {
        return Ok(Vec::new());
    }

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    for path in collect_markdown_files(kb_dir) {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        for section in split_markdown_sections(kb_dir, &path, &text) {
            if let Some(result) = score_section(&section, query, &query_tokens) {
                results.push(result);
            }
        }
    }

    results.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.heading.cmp(&right.heading))
    });
    results.truncate(limit);
    Ok(results)
}

fn score_section(
    section: &KbSection,
    query: &str,
    query_tokens: &[String],
) -> Option<KbSearchResult> {
    let haystack = format!("{}\n{}\n{}", section.path, section.heading, section.text);
    let haystack_lower = haystack.to_lowercase();
    let counts = token_counts(&haystack);
    let mut matched = 0usize;
    let mut occurrences = 0usize;

    for token in query_tokens {
        if let Some(count) = counts.get(token) {
            matched += 1;
            occurrences += (*count).min(4);
        }
    }

    if matched < required_match_count(query_tokens.len()) {
        return None;
    }

    let query_lower = query.to_lowercase();
    let mut score = matched * 10 + occurrences;
    if haystack_lower.contains(&query_lower) {
        score += 25;
    }
    for pair in query_tokens.windows(2) {
        let phrase = format!("{} {}", pair[0], pair[1]);
        if haystack_lower.contains(&phrase) {
            score += 6;
        }
    }

    if score < MIN_SCORE {
        return None;
    }

    Some(KbSearchResult {
        path: section.path.clone(),
        heading: section.heading.clone(),
        text: build_snippet(&section.text, query_tokens),
        score,
    })
}

fn collect_markdown_files(kb_dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = WalkDir::new(kb_dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    files.sort();
    files
}

fn split_markdown_sections(kb_dir: &Path, path: &Path, markdown: &str) -> Vec<KbSection> {
    let rel_path = path
        .strip_prefix(kb_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();
    let mut headings: Vec<String> = Vec::new();
    let mut current_heading = fallback_heading(&rel_path);
    let mut current_text = String::new();
    let mut sections = Vec::new();

    for line in markdown.lines() {
        if let Some((level, title)) = parse_heading(line) {
            push_section(&mut sections, &rel_path, &current_heading, &current_text);
            headings.truncate(level.saturating_sub(1));
            headings.push(title);
            current_heading = headings.join(" > ");
            current_text.clear();
            continue;
        }

        current_text.push_str(line);
        current_text.push('\n');
    }

    push_section(&mut sections, &rel_path, &current_heading, &current_text);
    sections
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

fn fallback_heading(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn token_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokenize(text) {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '/' || ch == '.' {
            current.push(ch);
            continue;
        }
        push_token(&mut tokens, &mut current);
    }
    push_token(&mut tokens, &mut current);
    tokens
}

fn push_token(tokens: &mut Vec<String>, current: &mut String) {
    if current.len() > 1 && !is_stopword(current) {
        tokens.push(std::mem::take(current));
    } else {
        current.clear();
    }
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "or"
            | "to"
            | "of"
            | "in"
            | "for"
            | "a"
            | "an"
            | "is"
            | "are"
            | "we"
            | "it"
            | "with"
            | "on"
            | "as"
            | "by"
            | "be"
            | "this"
            | "that"
            | "use"
    )
}

fn build_snippet(text: &str, query_tokens: &[String]) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower = compact.to_lowercase();
    let start = query_tokens
        .iter()
        .filter_map(|token| lower.find(token))
        .min()
        .unwrap_or(0)
        .saturating_sub(80);
    let start = previous_char_boundary(&compact, start);
    let start = previous_word_boundary(&compact, start);
    truncate_chars(&compact[start..], SNIPPET_CHARS)
}

fn required_match_count(query_token_count: usize) -> usize {
    match query_token_count {
        0 => 1,
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn previous_char_boundary(text: &str, index: usize) -> usize {
    let mut boundary = index.min(text.len());
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

fn previous_word_boundary(text: &str, index: usize) -> usize {
    let mut boundary = index;
    while boundary > 0 && !text[..boundary].ends_with(char::is_whitespace) {
        boundary = previous_char_boundary(text, boundary.saturating_sub(1));
    }
    boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_search_prefers_exact_heading_matches() {
        let markdown = "# Corrections\n\n## Process\nLoad frontend design skill immediately.\n";
        let sections =
            split_markdown_sections(Path::new("/kb"), Path::new("/kb/memory.md"), markdown);
        let query = "frontend design skill load immediately";
        let result =
            score_section(&sections[0], query, &tokenize(query)).expect("section should match");

        assert_eq!(result.heading, "Corrections > Process");
        assert!(result.score >= 40);
        assert!(result.text.contains("frontend design skill"));
    }

    #[test]
    fn section_search_requires_more_than_one_weak_token_for_multi_token_query() {
        let markdown = "# Notes\n\nSkill points reset video bookmark.\n";
        let sections =
            split_markdown_sections(Path::new("/kb"), Path::new("/kb/bookmarks.md"), markdown);

        let result = score_section(
            &sections[0],
            "frontend design skill load immediately",
            &tokenize("frontend design skill load immediately"),
        );

        assert!(result.is_none());
    }

    #[test]
    fn long_queries_require_three_distinct_terms() {
        let markdown = "# Notes\n\nDesign links and profession skill points.\n";
        let sections =
            split_markdown_sections(Path::new("/kb"), Path::new("/kb/bookmarks.md"), markdown);
        let query = "frontend design skill load immediately";

        let result = score_section(&sections[0], query, &tokenize(query));

        assert!(result.is_none());
    }
}
