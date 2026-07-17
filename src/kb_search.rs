//! Persistent heading-aware PageIndex for the local Markdown knowledge base.

use anyhow::{Context, Result, bail};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

#[path = "kb_search_markdown.rs"]
mod kb_search_markdown;
#[path = "kb_search_model.rs"]
mod kb_search_model;
#[path = "kb_search_text.rs"]
mod kb_search_text;
use kb_search_markdown::{MarkdownSection, fallback_heading, split_markdown_sections};
pub use kb_search_model::{
    KbDocContent, KbDocMetadata, KbDocStructure, KbIndexedDoc, KbIndexedFile, KbIndexedNode,
    KbNodeStructure, KbPageIndex,
};
use kb_search_text::{build_snippet, token_counts, tokenize};

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const INDEX_FILE_NAME: &str = "index.json";
const NODES_FILE_NAME: &str = "nodes.tsv";
const MANIFEST_FILE_NAME: &str = "manifest.tsv";
const HEADING_WEIGHT: usize = 600;
const PATH_WEIGHT: usize = 400;
const BODY_WEIGHT: usize = 100;
const TERM_FREQUENCY_CAP: usize = 3;
const PHRASE_BONUS: usize = 1_000;
const SCORE_SCALE: usize = 100;
const MIN_SCORE: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIndexNode {
    pub path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub heading_path: String,
    pub normalized_body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextManifestEntry {
    pub path: String,
    pub mtime_ns: u128,
    pub size: u64,
}

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

pub fn build_text_index(kb_dir: &Path, index_dir: &Path) -> Result<KbBuildSummary> {
    let files = collect_markdown_files(kb_dir);
    let mut nodes = Vec::new();
    let mut manifest = Vec::new();

    for path in &files {
        let relative = relative_path(kb_dir, path);
        let (markdown, entry) = read_stable_markdown(kb_dir, path)?;
        let sections = split_markdown_sections(&relative, &markdown);
        let line_count = markdown.lines().count();
        for (index, section) in sections.iter().enumerate() {
            nodes.push(TextIndexNode {
                path: relative.clone(),
                line_start: section.source_line,
                line_end: sections
                    .get(index + 1)
                    .map_or(line_count, |next| next.source_line.saturating_sub(1)),
                heading_path: section.heading_path.clone(),
                normalized_body: normalized_section_body(section),
            });
        }
        manifest.push(entry);
    }

    std::fs::create_dir_all(index_dir)
        .with_context(|| format!("failed to create {}", index_dir.display()))?;
    let nodes_path = index_dir.join(NODES_FILE_NAME);
    std::fs::write(&nodes_path, render_text_nodes(&nodes))?;
    std::fs::write(
        index_dir.join(MANIFEST_FILE_NAME),
        render_text_manifest(&manifest),
    )?;

    Ok(KbBuildSummary {
        files: manifest.len(),
        nodes: nodes.len(),
        index_path: nodes_path,
    })
}

pub fn load_text_nodes(index_dir: &Path) -> Result<Vec<TextIndexNode>> {
    let path = index_dir.join(NODES_FILE_NAME);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_text_nodes(&text).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn load_text_manifest(index_dir: &Path) -> Result<Vec<TextManifestEntry>> {
    let path = index_dir.join(MANIFEST_FILE_NAME);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_text_manifest(&text).with_context(|| format!("failed to parse {}", path.display()))
}

pub fn validate_text_manifest(kb_dir: &Path, index_dir: &Path) -> Result<()> {
    let expected = load_text_manifest(index_dir)?;
    let files = collect_markdown_files(kb_dir);
    if files.len() != expected.len() {
        bail!("stale KB text index: Markdown file set changed");
    }
    for (path, expected_entry) in files.iter().zip(expected.iter()) {
        let actual = manifest_entry(kb_dir, path)?;
        if !manifest_entries_match(expected_entry, &actual) {
            bail!("stale KB text index: {} changed", actual.path);
        }
    }
    Ok(())
}

fn read_stable_markdown(kb_dir: &Path, path: &Path) -> Result<(String, TextManifestEntry)> {
    let before = manifest_entry(kb_dir, path)?;
    let markdown = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let after = manifest_entry(kb_dir, path)?;
    if !manifest_entries_match(&before, &after) {
        bail!(
            "Markdown changed while building text index: {}",
            before.path
        );
    }
    Ok((markdown, after))
}

pub fn search_text_index(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    validate_text_manifest(kb_dir, index_dir)?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let phrase_tokens = text_index_tokens(query);
    if phrase_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let scoring_terms = unique_tokens(phrase_tokens.clone());
    let mut results: Vec<KbSearchResult> = load_text_nodes(index_dir)?
        .into_iter()
        .filter_map(|node| score_text_node(node, &scoring_terms, &phrase_tokens))
        .collect();
    results.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.heading.cmp(&right.heading))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    results.truncate(limit);
    Ok(results)
}

fn score_text_node(
    node: TextIndexNode,
    scoring_terms: &[String],
    phrase_tokens: &[String],
) -> Option<KbSearchResult> {
    let heading_tokens = text_index_tokens(&node.heading_path);
    let path_tokens = text_index_tokens(&node.path);
    let body_tokens = text_index_tokens(&node.normalized_body);
    let structural_frequency =
        structural_term_frequency(scoring_terms, &heading_tokens, &path_tokens);
    let body_frequency = body_term_frequency(scoring_terms, &body_tokens);
    if structural_frequency == 0 && body_frequency == 0 {
        return None;
    }
    let length_divisor = integer_sqrt(body_tokens.len().max(1)).max(1);
    let phrase_bonus =
        usize::from(contains_token_sequence(&body_tokens, phrase_tokens)) * PHRASE_BONUS;
    let score = structural_frequency * SCORE_SCALE
        + body_frequency * SCORE_SCALE / length_divisor
        + phrase_bonus;
    let locator = format!("{}-{}", node.line_start, node.line_end);
    let title = node
        .heading_path
        .rsplit(" > ")
        .next()
        .unwrap_or(&node.heading_path)
        .to_string();
    let command = format!(
        "claude-memory kb-page-index content {} {}",
        node.path, locator
    );
    Some(KbSearchResult {
        doc_id: node.path.clone(),
        path: node.path,
        heading: node.heading_path,
        text: node.normalized_body,
        score,
        node_id: locator,
        title,
        reason: format!("matched deterministic text index; score {score}"),
        content_command: command.clone(),
        next_content_command: command,
    })
}

fn text_index_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn unique_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    tokens
        .into_iter()
        .filter(|token| seen.insert(token.clone()))
        .collect()
}

fn structural_term_frequency(
    query_tokens: &[String],
    heading_tokens: &[String],
    path_tokens: &[String],
) -> usize {
    query_tokens
        .iter()
        .map(|term| {
            capped_frequency(heading_tokens, term) * HEADING_WEIGHT
                + capped_frequency(path_tokens, term) * PATH_WEIGHT
        })
        .sum()
}

fn body_term_frequency(query_tokens: &[String], body_tokens: &[String]) -> usize {
    query_tokens
        .iter()
        .map(|term| capped_frequency(body_tokens, term) * BODY_WEIGHT)
        .sum()
}

fn contains_token_sequence(tokens: &[String], sequence: &[String]) -> bool {
    !sequence.is_empty()
        && tokens
            .windows(sequence.len())
            .any(|window| window == sequence)
}

fn capped_frequency(tokens: &[String], term: &str) -> usize {
    tokens
        .iter()
        .filter(|token| token.as_str() == term)
        .count()
        .min(TERM_FREQUENCY_CAP)
}

fn integer_sqrt(value: usize) -> usize {
    let mut low = 0;
    let mut high = value;
    let mut root = 0;
    while low <= high {
        let middle = low + (high - low) / 2;
        if middle == 0 || middle <= value / middle {
            root = middle;
            low = middle.saturating_add(1);
        } else {
            high = middle - 1;
        }
    }
    root
}

fn manifest_entry(kb_dir: &Path, path: &Path) -> Result<TextManifestEntry> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
    Ok(TextManifestEntry {
        path: relative_path(kb_dir, path),
        mtime_ns: modified.as_nanos(),
        size: metadata.len(),
    })
}

fn manifest_entries_match(left: &TextManifestEntry, right: &TextManifestEntry) -> bool {
    left.path == right.path && left.mtime_ns == right.mtime_ns && left.size == right.size
}

fn normalized_section_body(section: &MarkdownSection) -> String {
    let mut lines = section.text.lines();
    let first = lines.next().unwrap_or_default();
    let body = if is_markdown_heading(first) {
        lines.collect::<Vec<_>>().join(" ")
    } else {
        section.text.clone()
    };
    body.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_markdown_heading(line: &str) -> bool {
    let trimmed = line.trim_start();
    let hashes = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    (1..=6).contains(&hashes)
        && trimmed
            .get(hashes..)
            .is_some_and(|title| !title.trim().is_empty())
}

fn render_text_nodes(nodes: &[TextIndexNode]) -> String {
    nodes
        .iter()
        .map(|node| {
            format!(
                "{}\t{}\t{}\t{}\t{}\n",
                escape_tsv(&node.path),
                node.line_start,
                node.line_end,
                escape_tsv(&node.heading_path),
                escape_tsv(&node.normalized_body)
            )
        })
        .collect()
}

fn parse_text_nodes(text: &str) -> Result<Vec<TextIndexNode>> {
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            parse_text_node(line).with_context(|| format!("nodes.tsv row {}", index + 1))
        })
        .collect()
}

fn parse_text_node(line: &str) -> Result<TextIndexNode> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 5 {
        bail!("expected 5 fields");
    }
    let line_start: usize = fields[1].parse().context("invalid line_start")?;
    let line_end: usize = fields[2].parse().context("invalid line_end")?;
    if line_start == 0 || line_end < line_start {
        bail!("invalid line range {line_start}..{line_end}");
    }
    Ok(TextIndexNode {
        path: unescape_tsv(fields[0]).context("invalid path")?,
        line_start,
        line_end,
        heading_path: unescape_tsv(fields[3]).context("invalid heading_path")?,
        normalized_body: unescape_tsv(fields[4]).context("invalid normalized_body")?,
    })
}

fn render_text_manifest(entries: &[TextManifestEntry]) -> String {
    entries
        .iter()
        .map(|entry| {
            format!(
                "{}\t{}\t{}\n",
                escape_tsv(&entry.path),
                entry.mtime_ns,
                entry.size
            )
        })
        .collect()
}

fn parse_text_manifest(text: &str) -> Result<Vec<TextManifestEntry>> {
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            parse_text_manifest_entry(line)
                .with_context(|| format!("manifest.tsv row {}", index + 1))
        })
        .collect()
}

fn parse_text_manifest_entry(line: &str) -> Result<TextManifestEntry> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 3 {
        bail!("expected 3 fields");
    }
    Ok(TextManifestEntry {
        path: unescape_tsv(fields[0]).context("invalid path")?,
        mtime_ns: fields[1].parse().context("invalid mtime_ns")?,
        size: fields[2].parse().context("invalid size")?,
    })
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn unescape_tsv(value: &str) -> Result<String> {
    let mut output = String::new();
    let mut chars = value.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            output.push(character);
            continue;
        }
        match chars.next() {
            Some('\\') => output.push('\\'),
            Some('t') => output.push('\t'),
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            _ => bail!("invalid TSV escape"),
        }
    }
    Ok(output)
}

pub fn search_default_kb(query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    search_or_build(
        Path::new(DEFAULT_KB_DIR),
        &default_index_dir(),
        query,
        limit,
    )
}

pub fn search_default_kb_context(query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    search_or_build_context(
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

pub fn search_or_build_context(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    ensure_fresh_index(kb_dir, index_dir)?;
    search_index(index_dir, query, limit)?
        .into_iter()
        .map(|mut result| {
            let content = document_content(index_dir, Path::new(&result.doc_id), &result.node_id)?;
            result.text = content.text;
            Ok(result)
        })
        .collect()
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
    let structural_text = node_structural_text(doc, node);
    let stats = node_match_stats(node, query_tokens, &structural_text);
    if stats.matched < required_match_count(query_tokens.len()) {
        return None;
    }

    let combined_text = format!("{structural_text}\n{}", node.text);
    let phrase_score = phrase_score(&combined_text, query, query_tokens);
    if is_link_dump_node(node) && phrase_score == 0 {
        return None;
    }

    let score = score_node_match(&stats, phrase_score);
    if score < MIN_SCORE {
        return None;
    }

    Some(search_result_for_node(
        doc,
        node,
        query_tokens,
        score,
        stats,
    ))
}

fn node_structural_text(doc: &KbIndexedDoc, node: &KbIndexedNode) -> String {
    format!(
        "{}\n{}\n{}",
        doc.source_path, doc.doc_name, node.heading_path
    )
}

fn score_node_match(stats: &NodeMatchStats, phrase_score: usize) -> usize {
    let base_score = stats.matched * 10 + stats.occurrences + stats.structural_matches * 4;
    base_score + phrase_score
}

fn is_link_dump_node(node: &KbIndexedNode) -> bool {
    let mut non_empty_lines = 0usize;
    let mut markdown_link_lines = 0usize;

    for line in node.text.lines().map(str::trim) {
        if line.is_empty() {
            continue;
        }

        non_empty_lines += 1;
        if line.starts_with("- [") && line.contains("](") {
            markdown_link_lines += 1;
        }
    }

    markdown_link_lines >= 10 && markdown_link_lines * 2 >= non_empty_lines
}

fn search_result_for_node(
    doc: &KbIndexedDoc,
    node: &KbIndexedNode,
    query_tokens: &[String],
    score: usize,
    stats: NodeMatchStats,
) -> KbSearchResult {
    KbSearchResult {
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
    }
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

fn relative_path(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
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
