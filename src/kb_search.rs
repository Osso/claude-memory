//! Persistent heading-aware PageIndex for the local Markdown knowledge base.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const INDEX_FILE_NAME: &str = "index.json";
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
pub struct KbBuildSummary {
    pub files: usize,
    pub nodes: usize,
    pub index_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KbPageIndex {
    source_dir: String,
    built_at: DateTime<Utc>,
    files: Vec<KbIndexedFile>,
    docs: Vec<KbIndexedDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KbIndexedFile {
    path: String,
    fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KbIndexedDoc {
    path: String,
    title: String,
    nodes: Vec<KbIndexedNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KbIndexedNode {
    node_id: String,
    heading: String,
    level: usize,
    parent: Option<String>,
    text: String,
    token_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct MarkdownSection {
    node_id: String,
    heading: String,
    level: usize,
    parent: Option<String>,
    text: String,
}

pub fn default_index_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("claude-memory/kb-page-index")
}

pub fn build_default_index() -> Result<KbBuildSummary> {
    build_index(Path::new(DEFAULT_KB_DIR), &default_index_dir())
}

pub fn build_index(kb_dir: &Path, index_dir: &Path) -> Result<KbBuildSummary> {
    let files = collect_markdown_files(kb_dir);
    let indexed_files = files
        .iter()
        .map(|path| build_file_fingerprint(kb_dir, path))
        .collect::<Result<Vec<_>>>()?;
    let docs = files
        .iter()
        .map(|path| build_doc(kb_dir, path))
        .collect::<Result<Vec<_>>>()?;
    let node_count = docs.iter().map(|doc| doc.nodes.len()).sum();
    let index = KbPageIndex {
        source_dir: kb_dir.to_string_lossy().to_string(),
        built_at: Utc::now(),
        files: indexed_files,
        docs,
    };

    std::fs::create_dir_all(index_dir)
        .with_context(|| format!("failed to create {}", index_dir.display()))?;
    let index_path = index_path(index_dir);
    let json = serde_json::to_string_pretty(&index).context("failed to serialize KB PageIndex")?;
    std::fs::write(&index_path, json)
        .with_context(|| format!("failed to write {}", index_path.display()))?;

    Ok(KbBuildSummary {
        files: index.files.len(),
        nodes: node_count,
        index_path,
    })
}

pub fn search_default_kb(query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    search_or_build(
        Path::new(DEFAULT_KB_DIR),
        &default_index_dir(),
        query,
        limit,
    )
}

pub fn search_or_build(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    ensure_fresh_index(kb_dir, index_dir)?;
    search_index(index_dir, query, limit)
}

pub fn search_index(index_dir: &Path, query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let query_tokens = tokenize(query);
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let index = load_index(index_dir)?;
    let mut results = Vec::new();
    for doc in &index.docs {
        for node in &doc.nodes {
            if let Some(result) = score_node(doc, node, query, &query_tokens) {
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

fn ensure_fresh_index(kb_dir: &Path, index_dir: &Path) -> Result<()> {
    if !index_path(index_dir).exists() {
        build_index(kb_dir, index_dir)?;
        return Ok(());
    }

    let index = load_index(index_dir)?;
    if index.source_dir != kb_dir.to_string_lossy() || index_is_stale(kb_dir, &index)? {
        build_index(kb_dir, index_dir)?;
    }
    Ok(())
}

fn load_index(index_dir: &Path) -> Result<KbPageIndex> {
    let path = index_path(index_dir);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn index_is_stale(kb_dir: &Path, index: &KbPageIndex) -> Result<bool> {
    let current_files = collect_markdown_files(kb_dir)
        .iter()
        .map(|path| build_file_fingerprint(kb_dir, path))
        .collect::<Result<Vec<_>>>()?;
    Ok(current_files.len() != index.files.len()
        || current_files
            .iter()
            .zip(index.files.iter())
            .any(|(left, right)| left.path != right.path || left.fingerprint != right.fingerprint))
}

fn index_path(index_dir: &Path) -> PathBuf {
    index_dir.join(INDEX_FILE_NAME)
}

fn build_doc(kb_dir: &Path, path: &Path) -> Result<KbIndexedDoc> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let rel_path = relative_path(kb_dir, path);
    let sections = split_markdown_sections(&rel_path, &text);
    let nodes = sections
        .into_iter()
        .map(|section| KbIndexedNode {
            node_id: section.node_id,
            heading: section.heading,
            level: section.level,
            parent: section.parent,
            token_counts: token_counts(&section.text),
            text: section.text,
        })
        .collect();

    Ok(KbIndexedDoc {
        title: fallback_heading(&rel_path),
        path: rel_path,
        nodes,
    })
}

fn build_file_fingerprint(kb_dir: &Path, path: &Path) -> Result<KbIndexedFile> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(KbIndexedFile {
        path: relative_path(kb_dir, path),
        fingerprint: format!("{:x}", hasher.finalize()),
    })
}

fn collect_markdown_files(kb_dir: &Path) -> Vec<PathBuf> {
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
    files
}

fn split_markdown_sections(path: &str, markdown: &str) -> Vec<MarkdownSection> {
    let mut heading_stack: Vec<String> = Vec::new();
    let mut id_stack: Vec<String> = Vec::new();
    let mut current = SectionBuilder::new("0", &fallback_heading(path), 1, None);
    let mut sections = Vec::new();
    let mut heading_count = 0usize;

    for line in markdown.lines() {
        if let Some((level, title)) = parse_heading(line) {
            current.push_if_not_empty(&mut sections);
            heading_count += 1;
            heading_stack.truncate(level.saturating_sub(1));
            id_stack.truncate(level.saturating_sub(1));
            heading_stack.push(title);
            let node_id = heading_count.to_string();
            let parent = id_stack.last().cloned();
            id_stack.push(node_id.clone());
            current = SectionBuilder::new(&node_id, &heading_stack.join(" > "), level, parent);
            continue;
        }

        current.push_line(line);
    }

    current.push_if_not_empty(&mut sections);
    sections
}

struct SectionBuilder {
    node_id: String,
    heading: String,
    level: usize,
    parent: Option<String>,
    text: String,
}

impl SectionBuilder {
    fn new(node_id: &str, heading: &str, level: usize, parent: Option<String>) -> Self {
        Self {
            node_id: node_id.to_string(),
            heading: heading.to_string(),
            level,
            parent,
            text: String::new(),
        }
    }

    fn push_line(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
    }

    fn push_if_not_empty(&self, sections: &mut Vec<MarkdownSection>) {
        let text = self.text.trim();
        if text.is_empty() {
            return;
        }

        sections.push(MarkdownSection {
            node_id: self.node_id.clone(),
            heading: self.heading.clone(),
            level: self.level,
            parent: self.parent.clone(),
            text: text.to_string(),
        });
    }
}

fn score_node(
    doc: &KbIndexedDoc,
    node: &KbIndexedNode,
    query: &str,
    query_tokens: &[String],
) -> Option<KbSearchResult> {
    let structural_text = format!("{}\n{}\n{}", doc.path, doc.title, node.heading);
    let structural_counts = token_counts(&structural_text);
    let mut matched_tokens = Vec::new();
    let mut occurrences = 0usize;
    let mut structural_matches = 0usize;

    for token in query_tokens {
        let body_count = node.token_counts.get(token).copied().unwrap_or(0);
        let structural_count = structural_counts.get(token).copied().unwrap_or(0);
        if body_count + structural_count > 0 {
            matched_tokens.push(token.as_str());
            occurrences += (body_count + structural_count).min(4);
        }
        if structural_count > 0 {
            structural_matches += 1;
        }
    }

    matched_tokens.sort_unstable();
    matched_tokens.dedup();
    let matched = matched_tokens.len();
    if matched < required_match_count(query_tokens.len()) {
        return None;
    }

    let combined_text = format!("{structural_text}\n{}", node.text);
    let mut score = matched * 10 + occurrences + structural_matches * 4;
    score += phrase_score(&combined_text, query, query_tokens);
    if score < MIN_SCORE {
        return None;
    }

    Some(KbSearchResult {
        path: doc.path.clone(),
        heading: node.heading.clone(),
        text: build_snippet(&node.text, query_tokens),
        score,
    })
}

fn phrase_score(text: &str, query: &str, query_tokens: &[String]) -> usize {
    let lower = text.to_lowercase();
    let mut score = 0usize;
    if lower.contains(&query.to_lowercase()) {
        score += 25;
    }
    for pair in query_tokens.windows(2) {
        let phrase = format!("{} {}", pair[0], pair[1]);
        if lower.contains(&phrase) {
            score += 6;
        }
    }
    score
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

fn relative_path(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
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
        let token = std::mem::take(current);
        tokens.push(token.clone());
        for part in token.split(['-', '/', '.']) {
            if part.len() > 1 && !is_stopword(part) {
                tokens.push(part.to_string());
            }
        }
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
        0 | 1 => 1,
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
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn build_and_search_persisted_kb_index() {
        let root = unique_temp_dir("kb-page-index-search");
        let kb_dir = root.join("kb");
        let index_dir = root.join("index");
        std::fs::create_dir_all(kb_dir.join("memory")).unwrap();
        std::fs::write(
            kb_dir.join("memory/corrections.md"),
            "# Corrections\n\n## Process\nLoad frontend design skill immediately.\n",
        )
        .unwrap();

        let summary = build_index(&kb_dir, &index_dir).unwrap();
        let results =
            search_index(&index_dir, "frontend design skill load immediately", 3).unwrap();

        assert_eq!(summary.files, 1);
        assert_eq!(summary.nodes, 1);
        assert_eq!(results[0].path, "memory/corrections.md");
        assert_eq!(results[0].heading, "Corrections > Process");
    }

    #[test]
    fn search_or_build_refreshes_stale_index() {
        let root = unique_temp_dir("kb-page-index-refresh");
        let kb_dir = root.join("kb");
        let index_dir = root.join("index");
        std::fs::create_dir_all(&kb_dir).unwrap();
        let path = kb_dir.join("notes.md");
        std::fs::write(&path, "# Notes\nOld content.\n").unwrap();
        build_index(&kb_dir, &index_dir).unwrap();

        std::fs::write(&path, "# Notes\nUse uv instead of pip.\n").unwrap();
        let results = search_or_build(&kb_dir, &index_dir, "uv instead pip", 3).unwrap();

        assert_eq!(results[0].path, "notes.md");
        assert!(results[0].text.contains("uv instead of pip"));
    }

    #[test]
    fn long_queries_require_three_distinct_terms() {
        let markdown = "# Notes\n\nDesign links and profession skill points.\n";
        let doc = build_doc_from_text("bookmarks.md", markdown);
        let query = "frontend design skill load immediately";

        let result = score_node(&doc, &doc.nodes[0], query, &tokenize(query));

        assert!(result.is_none());
    }

    fn build_doc_from_text(path: &str, markdown: &str) -> KbIndexedDoc {
        let nodes = split_markdown_sections(path, markdown)
            .into_iter()
            .map(|section| KbIndexedNode {
                node_id: section.node_id,
                heading: section.heading,
                level: section.level,
                parent: section.parent,
                token_counts: token_counts(&section.text),
                text: section.text,
            })
            .collect();
        KbIndexedDoc {
            path: path.to_string(),
            title: fallback_heading(path),
            nodes,
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }
}
