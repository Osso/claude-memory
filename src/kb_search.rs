//! Persistent heading-aware PageIndex for the local Markdown knowledge base.

use anyhow::{Context, Result, bail};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[path = "kb_search_model.rs"]
mod kb_search_model;
#[path = "kb_search_text.rs"]
mod kb_search_text;
pub use kb_search_model::{
    KbDocContent, KbDocMetadata, KbDocStructure, KbIndexedDoc, KbIndexedFile, KbIndexedNode,
    KbNodeStructure, KbPageIndex,
};
use kb_search_text::{build_snippet, token_counts, tokenize};

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const INDEX_FILE_NAME: &str = "index.json";
const MIN_SCORE: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KbSearchResult {
    pub doc_id: String,
    pub path: String,
    pub heading: String,
    pub text: String,
    pub score: usize,
    pub node_id: String,
    pub title: String,
    pub reason: String,
    pub content_command: String,
    pub next_content_command: String,
}

#[derive(Debug, Clone)]
pub struct KbBuildSummary {
    pub files: usize,
    pub nodes: usize,
    pub index_path: PathBuf,
}

#[derive(Debug, Clone)]
struct MarkdownSection {
    node_id: String,
    title: String,
    heading_path: String,
    level: usize,
    parent: Option<String>,
    source_line: usize,
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
    let node_count = docs.iter().map(|doc| count_nodes(&doc.nodes)).sum();
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
        for node in flatten_nodes(&doc.nodes) {
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

pub fn document_metadata(
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
) -> Result<KbDocMetadata> {
    let index = load_index(index_dir)?;
    let doc = find_doc(&index, doc_selector.as_ref())?;
    Ok(doc.metadata())
}

pub fn document_structure(
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
) -> Result<KbDocStructure> {
    let index = load_index(index_dir)?;
    let doc = find_doc(&index, doc_selector.as_ref())?;
    Ok(doc.structure_without_text())
}

pub fn document_content(
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
    locator: &str,
) -> Result<KbDocContent> {
    let index = load_index(index_dir)?;
    let doc = find_doc(&index, doc_selector.as_ref())?;
    let text = content_for_locator(doc, locator)?;
    Ok(KbDocContent {
        doc_id: doc.doc_id.clone(),
        source_path: doc.source_path.clone(),
        locator: locator.to_string(),
        text,
    })
}

pub fn query_index(index_dir: &Path, query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    search_index(index_dir, query, limit)
}

pub fn load_document(
    kb_dir: &Path,
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
) -> Result<KbIndexedDoc> {
    ensure_fresh_index(kb_dir, index_dir)?;
    let index = load_index(index_dir)?;
    Ok(find_doc(&index, doc_selector.as_ref())?.clone())
}

pub fn load_content(
    kb_dir: &Path,
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
    locator: &str,
) -> Result<String> {
    ensure_fresh_index(kb_dir, index_dir)?;
    Ok(document_content(index_dir, doc_selector, locator)?.text)
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

fn find_doc<'a>(index: &'a KbPageIndex, selector: &Path) -> Result<&'a KbIndexedDoc> {
    let normalized = normalize_doc_selector(&index.source_dir, selector);
    let with_extension = ensure_markdown_extension(&normalized);
    index
        .docs
        .iter()
        .find(|doc| doc.doc_id == normalized || doc.source_path == normalized)
        .or_else(|| {
            index
                .docs
                .iter()
                .find(|doc| doc.doc_id == with_extension || doc.source_path == with_extension)
        })
        .with_context(|| format!("document not found in KB PageIndex: {}", selector.display()))
}

fn normalize_doc_selector(source_dir: &str, selector: &Path) -> String {
    let source_root = Path::new(source_dir);
    let path = selector.strip_prefix(source_root).unwrap_or(selector);
    path.to_string_lossy().to_string()
}

fn ensure_markdown_extension(selector: &str) -> String {
    if selector.ends_with(".md") {
        selector.to_string()
    } else {
        format!("{selector}.md")
    }
}

fn content_for_locator(doc: &KbIndexedDoc, locator: &str) -> Result<String> {
    if let Some(node) = find_node(&doc.nodes, locator) {
        return Ok(format_content_text(&node.text));
    }

    if let Some((start, end)) = parse_line_range(locator) {
        return content_for_line_range(doc, start, end);
    }

    bail!(
        "locator must be a node id or inclusive line range like 4-8: {}",
        locator
    )
}

fn find_node<'a>(nodes: &'a [KbIndexedNode], node_id: &str) -> Option<&'a KbIndexedNode> {
    for node in nodes {
        if node.node_id == node_id {
            return Some(node);
        }
        if let Some(child) = find_node(&node.nodes, node_id) {
            return Some(child);
        }
    }
    None
}

fn parse_line_range(locator: &str) -> Option<(usize, usize)> {
    let (start, end) = locator.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    if start == 0 || end < start {
        return None;
    }
    Some((start, end))
}

fn content_for_line_range(doc: &KbIndexedDoc, start: usize, end: usize) -> Result<String> {
    let lines = doc.text.lines().collect::<Vec<_>>();
    if start > lines.len() {
        bail!(
            "line range starts after end of document: {} has {} lines",
            doc.doc_id,
            lines.len()
        );
    }

    let end = end.min(lines.len());
    let selected = lines[start - 1..end].join("\n");
    Ok(format_content_text(&selected))
}

fn format_content_text(text: &str) -> String {
    if text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{text}\n")
    }
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
    let doc_description = sections.first().map(|section| section.title.clone());
    let nodes = build_nested_nodes(&sections, None);

    Ok(KbIndexedDoc {
        doc_id: rel_path.clone(),
        doc_name: fallback_heading(&rel_path),
        doc_description,
        source_path: rel_path,
        line_count: text.lines().count(),
        text,
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
    let mut current: Option<SectionBuilder> = None;
    let mut sections = Vec::new();
    let mut heading_count = 0usize;
    let mut in_code_fence = false;

    for (line_index, line) in markdown.lines().enumerate() {
        let line_number = line_index + 1;
        if is_fence_line(line) {
            in_code_fence = !in_code_fence;
            push_markdown_line(path, &mut current, &mut heading_count, line_number, line);
            continue;
        }

        let heading_started = !in_code_fence
            && try_start_heading_section(
                line,
                line_number,
                &mut heading_stack,
                &mut id_stack,
                &mut heading_count,
                &mut current,
                &mut sections,
            );
        if heading_started {
            continue;
        }

        push_markdown_line(path, &mut current, &mut heading_count, line_number, line);
    }

    if let Some(section) = current {
        section.push(&mut sections);
    }
    sections
}

fn try_start_heading_section(
    line: &str,
    line_number: usize,
    heading_stack: &mut Vec<String>,
    id_stack: &mut Vec<String>,
    heading_count: &mut usize,
    current: &mut Option<SectionBuilder>,
    sections: &mut Vec<MarkdownSection>,
) -> bool {
    let Some((level, title)) = parse_heading(line) else {
        return false;
    };

    if let Some(section) = current.take() {
        section.push(sections);
    }
    *heading_count += 1;
    heading_stack.truncate(level.saturating_sub(1));
    id_stack.truncate(level.saturating_sub(1));
    heading_stack.push(title.clone());

    let node_id = format_node_id(*heading_count);
    let parent = id_stack.last().cloned();
    id_stack.push(node_id.clone());
    *current = Some(SectionBuilder::new(
        &node_id,
        &title,
        &heading_stack.join(" > "),
        level,
        parent,
        line_number,
    ));
    if let Some(section) = current {
        section.push_line(line);
    }
    true
}

struct SectionBuilder {
    node_id: String,
    title: String,
    heading_path: String,
    level: usize,
    parent: Option<String>,
    source_line: usize,
    text: String,
}

impl SectionBuilder {
    fn new(
        node_id: &str,
        title: &str,
        heading_path: &str,
        level: usize,
        parent: Option<String>,
        source_line: usize,
    ) -> Self {
        Self {
            node_id: node_id.to_string(),
            title: title.to_string(),
            heading_path: heading_path.to_string(),
            level,
            parent,
            source_line,
            text: String::new(),
        }
    }

    fn push_line(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
    }

    fn push(self, sections: &mut Vec<MarkdownSection>) {
        sections.push(MarkdownSection {
            node_id: self.node_id,
            title: self.title,
            heading_path: self.heading_path,
            level: self.level,
            parent: self.parent,
            source_line: self.source_line,
            text: self.text.trim().to_string(),
        });
    }
}

fn push_markdown_line(
    path: &str,
    current: &mut Option<SectionBuilder>,
    heading_count: &mut usize,
    line_number: usize,
    line: &str,
) {
    if current.is_none() && line.trim().is_empty() {
        return;
    }

    if current.is_none() {
        *heading_count += 1;
        let title = fallback_heading(path);
        let node_id = format_node_id(*heading_count);
        *current = Some(SectionBuilder::new(
            &node_id,
            &title,
            &title,
            1,
            None,
            line_number,
        ));
    }

    if let Some(section) = current {
        section.push_line(line);
    }
}

fn build_nested_nodes(sections: &[MarkdownSection], parent: Option<&str>) -> Vec<KbIndexedNode> {
    sections
        .iter()
        .filter(|section| section.parent.as_deref() == parent)
        .map(|section| KbIndexedNode {
            node_id: section.node_id.clone(),
            title: section.title.clone(),
            heading_path: section.heading_path.clone(),
            level: section.level,
            source_line: section.source_line,
            token_counts: token_counts(&section.text),
            text: section.text.clone(),
            nodes: build_nested_nodes(sections, Some(&section.node_id)),
        })
        .collect()
}

fn flatten_nodes(nodes: &[KbIndexedNode]) -> Vec<&KbIndexedNode> {
    let mut flattened = Vec::new();
    for node in nodes {
        flattened.push(node);
        flattened.extend(flatten_nodes(&node.nodes));
    }
    flattened
}

fn count_nodes(nodes: &[KbIndexedNode]) -> usize {
    nodes.len()
        + nodes
            .iter()
            .map(|node| count_nodes(&node.nodes))
            .sum::<usize>()
}

fn format_node_id(position: usize) -> String {
    format!("{position:06}")
}

struct NodeMatchStats {
    matched: usize,
    occurrences: usize,
    structural_matches: usize,
    terms: Vec<String>,
}

fn score_node(
    doc: &KbIndexedDoc,
    node: &KbIndexedNode,
    query: &str,
    query_tokens: &[String],
) -> Option<KbSearchResult> {
    let structural_text = format!(
        "{}\n{}\n{}",
        doc.source_path, doc.doc_name, node.heading_path
    );
    let stats = node_match_stats(node, query_tokens, &structural_text);
    if stats.matched < required_match_count(query_tokens.len()) {
        return None;
    }

    let combined_text = format!("{structural_text}\n{}", node.text);
    let mut score = stats.matched * 10 + stats.occurrences + stats.structural_matches * 4;
    score += phrase_score(&combined_text, query, query_tokens);
    if score < MIN_SCORE {
        return None;
    }

    Some(KbSearchResult {
        doc_id: doc.doc_id.clone(),
        path: doc.source_path.clone(),
        heading: node.heading_path.clone(),
        text: build_snippet(&node.text, query_tokens),
        score,
        node_id: node.node_id.clone(),
        title: node.title.clone(),
        reason: format!("matched query terms: {}", stats.terms.join(", ")),
        content_command: content_command(doc, node),
        next_content_command: content_command(doc, node),
    })
}

fn content_command(doc: &KbIndexedDoc, node: &KbIndexedNode) -> String {
    format!(
        "claude-memory kb-page-index content {} {}",
        doc.doc_id, node.node_id
    )
}

fn node_match_stats(
    node: &KbIndexedNode,
    query_tokens: &[String],
    structural_text: &str,
) -> NodeMatchStats {
    let structural_counts = token_counts(structural_text);
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
    let terms = matched_tokens
        .iter()
        .map(|token| (*token).to_string())
        .collect();
    NodeMatchStats {
        matched: matched_tokens.len(),
        occurrences,
        structural_matches,
        terms,
    }
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

fn is_fence_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
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

fn required_match_count(query_token_count: usize) -> usize {
    match query_token_count {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

#[cfg(test)]
#[path = "kb_search_tests.rs"]
mod tests;
