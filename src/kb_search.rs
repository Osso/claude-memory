//! Persistent heading-aware PageIndex for the local Markdown knowledge base.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use walkdir::WalkDir;

#[path = "kb_search_markdown.rs"]
mod kb_search_markdown;
#[path = "kb_search_scoring.rs"]
mod kb_search_scoring;
use kb_search_markdown::{MarkdownSection, split_markdown_sections};
#[cfg(test)]
use kb_search_scoring::integer_sqrt;
use kb_search_scoring::{
    coverage_score_stride, is_archive_path, score_text_node, text_index_tokens, unique_tokens,
};

pub const DEFAULT_KB_DIR: &str = "/syncthing/Sync/KB";

const NODES_FILE_NAME: &str = "nodes.tsv";
const MANIFEST_FILE_NAME: &str = "manifest.tsv";

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
}

#[derive(Debug, Clone)]
pub struct KbContextSearch {
    pub results: Vec<KbSearchResult>,
    pub warning: Option<String>,
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
    let _lock = lock_text_index(index_dir)?;
    build_text_index_locked(kb_dir, index_dir)
}

fn build_text_index_locked(kb_dir: &Path, index_dir: &Path) -> Result<KbBuildSummary> {
    if !kb_dir.is_dir() {
        bail!("KB directory does not exist: {}", kb_dir.display());
    }
    ensure_disjoint_directories(kb_dir, index_dir)?;
    let (nodes, manifest) = read_text_index_source(kb_dir)?;
    replace_text_index(index_dir, &nodes, &manifest)?;
    Ok(KbBuildSummary {
        files: manifest.len(),
        nodes: nodes.len(),
        index_path: index_dir.join(NODES_FILE_NAME),
    })
}

fn read_text_index_source(kb_dir: &Path) -> Result<(Vec<TextIndexNode>, Vec<TextManifestEntry>)> {
    let files = collect_markdown_files(kb_dir);
    let mut nodes = Vec::new();
    let mut manifest = Vec::new();
    for path in &files {
        let relative = relative_path(kb_dir, path);
        let (markdown, entry) = read_stable_markdown(kb_dir, path)?;
        nodes.extend(text_index_nodes(&relative, &markdown));
        manifest.push(entry);
    }
    validate_manifest_entries(kb_dir, &manifest)?;
    Ok((nodes, manifest))
}

fn text_index_nodes(relative_path: &str, markdown: &str) -> Vec<TextIndexNode> {
    let sections = split_markdown_sections(relative_path, markdown);
    let line_count = markdown.lines().count();
    sections
        .iter()
        .enumerate()
        .map(|(index, section)| TextIndexNode {
            path: relative_path.to_string(),
            line_start: section.source_line,
            line_end: sections
                .get(index + 1)
                .map_or(line_count, |next| next.source_line.saturating_sub(1)),
            heading_path: section.heading_path.clone(),
            normalized_body: normalized_section_body(section),
        })
        .collect()
}

fn lock_text_index(index_dir: &Path) -> Result<File> {
    let parent = index_dir.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let name = index_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("kb-page-index");
    let lock_path = parent.join(format!(".{name}.lock"));
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open {}", lock_path.display()))?;
    lock.lock()
        .with_context(|| format!("failed to lock {}", lock_path.display()))?;
    Ok(lock)
}

fn replace_text_index(
    index_dir: &Path,
    nodes: &[TextIndexNode],
    manifest: &[TextManifestEntry],
) -> Result<()> {
    let staging_dir = write_staged_text_index(index_dir, nodes, manifest)?;
    activate_staged_text_index(index_dir, &staging_dir)
}

fn write_staged_text_index(
    index_dir: &Path,
    nodes: &[TextIndexNode],
    manifest: &[TextManifestEntry],
) -> Result<PathBuf> {
    let staging_dir = sibling_index_path(index_dir, "staging");
    std::fs::create_dir(&staging_dir)
        .with_context(|| format!("failed to create {}", staging_dir.display()))?;
    std::fs::write(staging_dir.join(NODES_FILE_NAME), render_text_nodes(nodes))?;
    std::fs::write(
        staging_dir.join(MANIFEST_FILE_NAME),
        render_text_manifest(manifest),
    )?;
    Ok(staging_dir)
}

fn activate_staged_text_index(index_dir: &Path, staging_dir: &Path) -> Result<()> {
    let backup_dir = sibling_index_path(index_dir, "backup");
    if index_dir.exists() {
        preserve_active_text_index(index_dir, &backup_dir)?;
    }
    if let Err(activation_error) = std::fs::rename(staging_dir, index_dir) {
        restore_active_text_index(index_dir, &backup_dir, activation_error)?;
    }
    if backup_dir.exists() {
        std::fs::remove_dir_all(&backup_dir)
            .with_context(|| format!("failed to remove {}", backup_dir.display()))?;
    }
    Ok(())
}

fn preserve_active_text_index(index_dir: &Path, backup_dir: &Path) -> Result<()> {
    std::fs::rename(index_dir, backup_dir).with_context(|| {
        format!(
            "failed to preserve {} as {}",
            index_dir.display(),
            backup_dir.display()
        )
    })
}

fn restore_active_text_index(
    index_dir: &Path,
    backup_dir: &Path,
    activation_error: std::io::Error,
) -> Result<()> {
    if backup_dir.exists() {
        std::fs::rename(backup_dir, index_dir).with_context(|| {
            format!(
                "failed to restore {} after activation failed: {activation_error}",
                index_dir.display()
            )
        })?;
    }
    Err(activation_error).with_context(|| format!("failed to activate {}", index_dir.display()))
}

fn sibling_index_path(index_dir: &Path, label: &str) -> PathBuf {
    let parent = index_dir.parent().unwrap_or_else(|| Path::new("."));
    let name = index_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("kb-page-index");
    parent.join(format!(".{name}.{label}-{}", uuid::Uuid::new_v4()))
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
    let expected = load_text_manifest(index_dir)?;
    validate_manifest_entries(kb_dir, &expected)
}

fn validate_manifest_entries(kb_dir: &Path, expected: &[TextManifestEntry]) -> Result<()> {
    if !kb_dir.is_dir() {
        bail!("KB directory does not exist: {}", kb_dir.display());
    }
    let files = collect_markdown_files(kb_dir);
    if files.len() != expected.len() {
        bail!("stale KB text index: Markdown file set changed");
    }
    for (path, expected_entry) in files.iter().zip(expected) {
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
    let _lock = lock_text_index(index_dir)?;
    validate_text_manifest(kb_dir, index_dir)?;
    search_text_index_locked(index_dir, query, limit)
}

fn search_text_index_locked(
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let phrase_tokens = text_index_tokens(query);
    if phrase_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let scoring_terms = unique_tokens(phrase_tokens.clone());
    let coverage_stride = coverage_score_stride(scoring_terms.len());
    let results = load_text_nodes(index_dir)?
        .into_iter()
        .filter_map(|node| score_text_node(node, &scoring_terms, &phrase_tokens, coverage_stride))
        .collect();
    Ok(rank_text_results(results, coverage_stride, limit))
}

fn rank_text_results(
    mut results: Vec<KbSearchResult>,
    coverage_stride: usize,
    limit: usize,
) -> Vec<KbSearchResult> {
    let maximum_coverage = results
        .iter()
        .map(|result| result.score / coverage_stride)
        .max()
        .unwrap_or(0);
    results.retain(|result| result.score / coverage_stride == maximum_coverage);
    if results.iter().any(|result| !is_archive_path(&result.path)) {
        results.retain(|result| !is_archive_path(&result.path));
    }
    results.sort_by(|left, right| {
        is_archive_path(&left.path)
            .cmp(&is_archive_path(&right.path))
            .then_with(|| right.score.cmp(&left.score))
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.heading.cmp(&right.heading))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    let mut seen_paths = HashSet::new();
    results.retain(|result| seen_paths.insert(result.path.clone()));
    results.truncate(limit);
    results
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

pub fn search_default_kb_context_resilient(query: &str, limit: usize) -> KbContextSearch {
    search_kb_context_resilient(
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
    let _lock = lock_text_index(index_dir)?;
    ensure_text_index_locked(kb_dir, index_dir)?;
    search_text_index_locked(index_dir, query, limit)
}

pub fn search_kb_context(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<KbSearchResult>> {
    let _lock = lock_text_index(index_dir)?;
    ensure_text_index_locked(kb_dir, index_dir)?;
    let results = search_text_index_locked(index_dir, query, limit)?;
    read_kb_result_texts(kb_dir, index_dir, results)
}

pub fn search_kb_context_resilient(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> KbContextSearch {
    let _lock = match lock_text_index(index_dir) {
        Ok(lock) => lock,
        Err(error) => return unavailable_kb_context(error),
    };
    if validate_text_manifest(kb_dir, index_dir).is_ok() {
        return search_fresh_kb_context_locked(kb_dir, index_dir, query, limit);
    }

    match build_text_index_locked(kb_dir, index_dir) {
        Ok(_) => search_fresh_kb_context_locked(kb_dir, index_dir, query, limit),
        Err(error) => search_stale_kb_context_locked(index_dir, query, limit, error),
    }
}

fn ensure_text_index_locked(kb_dir: &Path, index_dir: &Path) -> Result<()> {
    if validate_text_manifest(kb_dir, index_dir).is_ok() {
        return Ok(());
    }
    build_text_index_locked(kb_dir, index_dir)
        .with_context(|| format!("failed to rebuild KB text index at {}", index_dir.display()))?;
    Ok(())
}

fn search_fresh_kb_context_locked(
    kb_dir: &Path,
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> KbContextSearch {
    let results = search_text_index_locked(index_dir, query, limit)
        .and_then(|results| read_kb_result_texts(kb_dir, index_dir, results));
    match results {
        Ok(results) => KbContextSearch {
            results,
            warning: None,
        },
        Err(error) => unavailable_kb_context(error),
    }
}

fn search_stale_kb_context_locked(
    index_dir: &Path,
    query: &str,
    limit: usize,
    rebuild_error: anyhow::Error,
) -> KbContextSearch {
    match search_text_index_locked(index_dir, query, limit) {
        Ok(results) => KbContextSearch {
            results,
            warning: Some(format!(
                "KB PageIndex rebuild failed; using stale index: {rebuild_error:#}"
            )),
        },
        Err(_) => unavailable_kb_context(rebuild_error),
    }
}

fn read_kb_result_texts(
    kb_dir: &Path,
    index_dir: &Path,
    results: Vec<KbSearchResult>,
) -> Result<Vec<KbSearchResult>> {
    results
        .into_iter()
        .map(|mut result| {
            result.text = read_indexed_source_text(
                kb_dir,
                index_dir,
                Path::new(&result.doc_id),
                &result.node_id,
            )?;
            Ok(result)
        })
        .collect()
}

fn read_indexed_source_text(
    kb_dir: &Path,
    index_dir: &Path,
    doc_selector: &Path,
    locator: &str,
) -> Result<String> {
    let source_path = resolve_text_document_path(kb_dir, index_dir, doc_selector)?;
    let (start, end) = parse_line_range(locator)
        .with_context(|| format!("invalid indexed line range: {locator}"))?;
    let path = kb_dir.join(&source_path);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let lines = text.split_inclusive('\n').collect::<Vec<_>>();
    if start > lines.len() || end > lines.len() {
        bail!(
            "indexed line range {start}-{end} exceeds document length: {} has {} lines",
            source_path,
            lines.len()
        );
    }
    Ok(lines[start - 1..end].concat())
}

fn resolve_text_document_path(kb_dir: &Path, index_dir: &Path, selector: &Path) -> Result<String> {
    let normalized = selector
        .strip_prefix(kb_dir)
        .unwrap_or(selector)
        .to_string_lossy()
        .to_string();
    load_text_manifest(index_dir)?
        .into_iter()
        .find(|entry| entry.path == normalized)
        .map(|entry| entry.path)
        .with_context(|| {
            format!(
                "document not found in KB text index: {}",
                selector.display()
            )
        })
}

fn parse_line_range(locator: &str) -> Option<(usize, usize)> {
    let (start, end) = locator.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    (start > 0 && end >= start).then_some((start, end))
}

fn unavailable_kb_context(error: anyhow::Error) -> KbContextSearch {
    KbContextSearch {
        results: Vec::new(),
        warning: Some(format!("KB PageIndex unavailable: {error:#}")),
    }
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
