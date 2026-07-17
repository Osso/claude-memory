//! Persistent heading-aware PageIndex for the local Markdown knowledge base.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

#[path = "kb_search_markdown.rs"]
mod kb_search_markdown;
use kb_search_markdown::{MarkdownSection, split_markdown_sections};

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const NODES_FILE_NAME: &str = "nodes.tsv";
const MANIFEST_FILE_NAME: &str = "manifest.tsv";
const HEADING_WEIGHT: usize = 600;
const PATH_WEIGHT: usize = 400;
const BODY_WEIGHT: usize = 100;
const TERM_FREQUENCY_CAP: usize = 3;
const PHRASE_BONUS: usize = 1_000;
const SCORE_SCALE: usize = 100;

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
pub struct KbDocContent {
    pub doc_id: String,
    pub source_path: String,
    pub locator: String,
    pub text: String,
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

pub fn build_text_index(kb_dir: &Path, index_dir: &Path) -> Result<KbBuildSummary> {
    ensure_disjoint_directories(kb_dir, index_dir)?;
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

    if index_dir.exists() {
        std::fs::remove_dir_all(index_dir)
            .with_context(|| format!("failed to replace {}", index_dir.display()))?;
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

fn ensure_disjoint_directories(kb_dir: &Path, index_dir: &Path) -> Result<()> {
    let kb_dir = std::fs::canonicalize(kb_dir)
        .with_context(|| format!("failed to resolve {}", kb_dir.display()))?;
    let index_dir = resolve_future_path(index_dir)?;
    if kb_dir.starts_with(&index_dir) || index_dir.starts_with(&kb_dir) {
        bail!(
            "KB and index directories overlap: {} and {}",
            kb_dir.display(),
            index_dir.display()
        );
    }
    Ok(())
}

fn resolve_future_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to resolve current directory")?
            .join(path)
    };
    let mut existing = absolute.as_path();
    let mut missing = Vec::new();
    while !existing.exists() {
        missing.push(
            existing
                .file_name()
                .context("path has no existing ancestor")?,
        );
        existing = existing.parent().context("path has no existing ancestor")?;
    }
    let mut resolved = std::fs::canonicalize(existing)
        .with_context(|| format!("failed to resolve {}", existing.display()))?;
    for component in missing.into_iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn load_text_nodes(index_dir: &Path) -> Result<Vec<TextIndexNode>> {
    let path = index_dir.join(NODES_FILE_NAME);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_text_nodes(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn load_text_manifest(index_dir: &Path) -> Result<Vec<TextManifestEntry>> {
    let path = index_dir.join(MANIFEST_FILE_NAME);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_text_manifest(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn validate_text_manifest(kb_dir: &Path, index_dir: &Path) -> Result<()> {
    if !kb_dir.is_dir() {
        bail!("KB directory does not exist: {}", kb_dir.display());
    }
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
    for result in &mut results {
        let command = text_content_command(&result.path, &result.node_id, kb_dir, index_dir);
        result.content_command = command.clone();
        result.next_content_command = command;
    }
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
    Some(KbSearchResult {
        doc_id: node.path.clone(),
        path: node.path,
        heading: node.heading_path,
        text: node.normalized_body,
        score,
        node_id: locator,
        title,
        reason: format!("matched deterministic text index; score {score}"),
        content_command: String::new(),
        next_content_command: String::new(),
    })
}

fn text_content_command(path: &str, locator: &str, kb_dir: &Path, index_dir: &Path) -> String {
    format!(
        "claude-memory kb-page-index content {} {} --kb {} --index {}",
        shell_quote(path),
        shell_quote(locator),
        shell_quote(&kb_dir.to_string_lossy()),
        shell_quote(&index_dir.to_string_lossy())
    )
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
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
    search_kb(
        Path::new(DEFAULT_KB_DIR),
        &default_index_dir(),
        query,
        limit,
    )
}

pub fn search_default_kb_context(query: &str, limit: usize) -> Result<Vec<KbSearchResult>> {
    search_kb_context(
        Path::new(DEFAULT_KB_DIR),
        &default_index_dir(),
        query,
        limit,
    )
}

pub fn search_kb(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    search_text_index(kb_dir, index_dir, query, limit)
}

pub fn search_kb_context(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    search_text_index(kb_dir, index_dir, query, limit)?
        .into_iter()
        .map(|mut result| {
            let content = text_document_content(
                kb_dir,
                index_dir,
                Path::new(&result.doc_id),
                &result.node_id,
            )?;
            result.text = content.text;
            Ok(result)
        })
        .collect()
}

pub fn text_document_content(
    kb_dir: &Path,
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
    locator: &str,
) -> Result<KbDocContent> {
    validate_text_manifest(kb_dir, index_dir)?;
    let source_path = resolve_text_document_path(kb_dir, index_dir, doc_selector.as_ref())?;
    let (start, end) = parse_line_range(locator)
        .with_context(|| format!("locator must be an inclusive line range like 4-8: {locator}"))?;
    let text = std::fs::read_to_string(kb_dir.join(&source_path))
        .with_context(|| format!("failed to read {}", kb_dir.join(&source_path).display()))?;
    let lines = text.split_inclusive('\n').collect::<Vec<_>>();
    if start > lines.len() || end > lines.len() {
        bail!(
            "line range {start}-{end} exceeds document length: {} has {} lines",
            source_path,
            lines.len()
        );
    }
    Ok(KbDocContent {
        doc_id: source_path.clone(),
        source_path,
        locator: locator.to_string(),
        text: lines[start - 1..end].concat(),
    })
}

fn resolve_text_document_path(kb_dir: &Path, index_dir: &Path, selector: &Path) -> Result<String> {
    let normalized = selector
        .strip_prefix(kb_dir)
        .unwrap_or(selector)
        .to_string_lossy()
        .to_string();
    let with_extension = ensure_markdown_extension(&normalized);
    load_text_manifest(index_dir)?
        .into_iter()
        .find(|entry| entry.path == normalized || entry.path == with_extension)
        .map(|entry| entry.path)
        .with_context(|| {
            format!(
                "document not found in KB text index: {}",
                selector.display()
            )
        })
}

fn ensure_markdown_extension(selector: &str) -> String {
    if selector.ends_with(".md") {
        selector.to_string()
    } else {
        format!("{selector}.md")
    }
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

fn relative_path(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
#[path = "kb_search_tests.rs"]
mod tests;
