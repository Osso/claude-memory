//! Persistent heading-aware PageIndex for the local Markdown knowledge base.

use anyhow::{Context, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[path = "kb_search_model.rs"]
mod kb_search_model;
#[path = "kb_search_text.rs"]
mod kb_search_text;
pub use kb_search_model::{
    KbDocStructure, KbIndexedDoc, KbIndexedFile, KbIndexedNode, KbNodeStructure, KbPageIndex,
};
use kb_search_text::{build_snippet, token_counts, tokenize};

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const INDEX_FILE_NAME: &str = "index.json";
const MIN_SCORE: usize = 12;

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
    let doc_description = sections.first().map(|section| section.title.clone());
    let nodes = build_nested_nodes(&sections, None);

    Ok(KbIndexedDoc {
        doc_id: rel_path.clone(),
        doc_name: fallback_heading(&rel_path),
        doc_description,
        source_path: rel_path,
        line_count: text.lines().count(),
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
        path: doc.source_path.clone(),
        heading: node.heading_path.clone(),
        text: build_snippet(&node.text, query_tokens),
        score,
    })
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
    NodeMatchStats {
        matched: matched_tokens.len(),
        occurrences,
        structural_matches,
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
        assert_eq!(summary.nodes, 2);
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
    fn build_doc_uses_nested_page_index_document_model() {
        let markdown = "# Router\nLocal network router note.\n\n## DHCP\nLease reservations.\n\n### Static leases\nPin important devices.\n\n## Firewall\nWAN block rules.\n";
        let doc = build_doc_from_text("state/router.md", markdown);

        assert_eq!(doc.doc_id, "state/router.md");
        assert_eq!(doc.doc_name, "router");
        assert_eq!(doc.doc_description.as_deref(), Some("Router"));
        assert_eq!(doc.source_path, "state/router.md");
        assert_eq!(doc.line_count, 11);
        assert_eq!(doc.nodes.len(), 1);

        let root = &doc.nodes[0];
        assert_eq!(root.node_id, "000001");
        assert_eq!(root.title, "Router");
        assert_eq!(root.heading_path, "Router");
        assert_eq!(root.source_line, 1);
        assert_eq!(root.nodes.len(), 2);
        assert!(root.text.contains("Local network router note."));

        let dhcp = &root.nodes[0];
        assert_eq!(dhcp.node_id, "000002");
        assert_eq!(dhcp.title, "DHCP");
        assert_eq!(dhcp.heading_path, "Router > DHCP");
        assert_eq!(dhcp.nodes[0].node_id, "000003");
        assert_eq!(dhcp.nodes[0].heading_path, "Router > DHCP > Static leases");
        assert!(dhcp.nodes[0].text.contains("Pin important devices."));
    }

    #[test]
    fn structure_view_omits_internal_node_text() {
        let markdown = "# Router\nSecret body text.\n\n## DHCP\nLease reservations.\n";
        let doc = build_doc_from_text("state/router.md", markdown);

        let structure = doc.structure_without_text();
        let json = serde_json::to_string(&structure).unwrap();

        assert!(json.contains("Router"));
        assert!(json.contains("000002"));
        assert!(!json.contains("Secret body text"));
        assert!(!json.contains("Lease reservations"));
        assert!(!json.contains("token_counts"));
    }

    #[test]
    fn search_or_build_refreshes_added_and_deleted_markdown_files() {
        let root = unique_temp_dir("kb-page-index-add-delete-refresh");
        let kb_dir = root.join("kb");
        let index_dir = root.join("index");
        std::fs::create_dir_all(&kb_dir).unwrap();
        let old_path = kb_dir.join("old.md");
        let new_path = kb_dir.join("new.md");
        std::fs::write(&old_path, "# Old\nRemove me after first build.\n").unwrap();
        build_index(&kb_dir, &index_dir).unwrap();

        std::fs::remove_file(&old_path).unwrap();
        std::fs::write(&new_path, "# New\nFresh page index content.\n").unwrap();

        let fresh_results = search_or_build(&kb_dir, &index_dir, "fresh page index content", 3)
            .expect("new file should trigger refresh");
        let old_results = search_index(&index_dir, "remove me after first build", 3)
            .expect("refreshed index should remain readable");

        assert_eq!(fresh_results[0].path, "new.md");
        assert!(old_results.is_empty());
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
        let sections = split_markdown_sections(path, markdown);
        let doc_description = sections.first().map(|section| section.title.clone());
        KbIndexedDoc {
            doc_id: path.to_string(),
            doc_name: fallback_heading(path),
            doc_description,
            source_path: path.to_string(),
            line_count: markdown.lines().count(),
            nodes: build_nested_nodes(&sections, None),
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
