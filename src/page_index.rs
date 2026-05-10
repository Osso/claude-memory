//! Local PageIndex-style outline tree for raw Claude/Codex transcript history.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::extract::{Role, Turn, read_session_turns};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexDoc {
    pub doc_id: String,
    pub doc_name: String,
    pub doc_description: Option<String>,
    pub source_family: String,
    pub source_path: String,
    pub turn_count: usize,
    pub text: String,
    pub nodes: Vec<PageIndexNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexNode {
    pub node_id: String,
    pub title: String,
    pub summary: String,
    pub source_locator: String,
    pub start_turn: u32,
    pub end_turn: u32,
    pub text: String,
    pub nodes: Vec<PageIndexNode>,
}

#[derive(Debug)]
pub struct BuildSummary {
    pub sessions: usize,
    pub nodes: usize,
    pub output_dir: PathBuf,
}

pub struct PageIndexSources<'a> {
    pub claude_projects_dir: &'a Path,
    pub claude_archive_dir: &'a Path,
    pub codex_sessions_dir: &'a Path,
    pub codex_archive_dir: &'a Path,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexDocMetadata {
    pub doc_id: String,
    pub doc_name: String,
    pub doc_description: Option<String>,
    pub source_family: String,
    pub source_path: String,
    pub turn_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexDocStructure {
    pub doc_id: String,
    pub doc_name: String,
    pub doc_description: Option<String>,
    pub source_family: String,
    pub source_path: String,
    pub turn_count: usize,
    pub nodes: Vec<PageIndexNodeStructure>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexNodeStructure {
    pub node_id: String,
    pub title: String,
    pub summary: String,
    pub source_locator: String,
    pub start_turn: u32,
    pub end_turn: u32,
    pub nodes: Vec<PageIndexNodeStructure>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexDocContent {
    pub doc_id: String,
    pub source_path: String,
    pub locator: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageIndexQueryResult {
    pub doc_id: String,
    pub source_path: String,
    pub node_id: String,
    pub title: String,
    pub score: usize,
    pub reason: String,
    pub next_content_command: String,
}

impl PageIndexDoc {
    pub fn outline(&self) -> String {
        let mut lines = vec![format!("{} — {}", self.doc_id, self.doc_name)];
        for node in &self.nodes {
            lines.push(format!(
                "{}. {} [{}-{}] — {}",
                node.node_id, node.title, node.start_turn, node.end_turn, node.summary
            ));
        }
        lines.join("\n")
    }

    pub fn node_text(&self, node_id: &str) -> Option<String> {
        find_node(&self.nodes, node_id).map(|node| node.text.clone())
    }

    pub fn metadata(&self) -> PageIndexDocMetadata {
        PageIndexDocMetadata {
            doc_id: self.doc_id.clone(),
            doc_name: self.doc_name.clone(),
            doc_description: self.doc_description.clone(),
            source_family: self.source_family.clone(),
            source_path: self.source_path.clone(),
            turn_count: self.turn_count,
        }
    }

    pub fn structure_without_text(&self) -> PageIndexDocStructure {
        PageIndexDocStructure {
            doc_id: self.doc_id.clone(),
            doc_name: self.doc_name.clone(),
            doc_description: self.doc_description.clone(),
            source_family: self.source_family.clone(),
            source_path: self.source_path.clone(),
            turn_count: self.turn_count,
            nodes: self
                .nodes
                .iter()
                .map(PageIndexNode::structure_without_text)
                .collect(),
        }
    }
}

impl PageIndexNode {
    pub fn structure_without_text(&self) -> PageIndexNodeStructure {
        PageIndexNodeStructure {
            node_id: self.node_id.clone(),
            title: self.title.clone(),
            summary: self.summary.clone(),
            source_locator: self.source_locator.clone(),
            start_turn: self.start_turn,
            end_turn: self.end_turn,
            nodes: self
                .nodes
                .iter()
                .map(PageIndexNode::structure_without_text)
                .collect(),
        }
    }
}

pub fn build_session_index(path: &Path, turns: &[Turn]) -> PageIndexDoc {
    let doc_id = session_id(path);
    let text = format_turns(turns);
    PageIndexDoc {
        doc_name: doc_id.clone(),
        doc_description: turns.first().map(node_title),
        source_family: "transcript".to_string(),
        source_path: path.to_string_lossy().to_string(),
        turn_count: turns.len(),
        text,
        doc_id,
        nodes: build_exchange_nodes(turns),
    }
}

pub fn write_session_index(output_dir: &Path, index: &PageIndexDoc) -> Result<PathBuf> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let path = output_dir.join(format!("{}.json", sanitize_doc_id(&index.doc_id)));
    let json = serde_json::to_string_pretty(index).context("failed to serialize page index")?;
    std::fs::write(&path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

pub fn collect_session_files(sources: &PageIndexSources<'_>) -> Vec<PathBuf> {
    let mut files = collect_live_sessions(sources.claude_projects_dir);
    files.extend(collect_claude_archive_sessions(sources.claude_archive_dir));
    files.extend(collect_codex_sessions(sources.codex_sessions_dir));
    files.extend(collect_codex_sessions(sources.codex_archive_dir));
    files.sort();
    files
}

pub fn default_output_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("claude-memory/transcript-page-index")
}

pub fn build_page_index(
    sources: &PageIndexSources<'_>,
    output_dir: &Path,
    max_sessions: Option<usize>,
) -> Result<BuildSummary> {
    let mut sessions = collect_session_files(sources);
    if let Some(limit) = max_sessions {
        sessions.truncate(limit);
    }

    let mut nodes = 0;
    let mut indexed = 0;
    for path in sessions {
        let turns = read_turns_for_page_index(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if turns.is_empty() {
            continue;
        }
        let index = build_session_index(&path, &turns);
        nodes += count_nodes(&index.nodes);
        write_session_index(output_dir, &index)?;
        indexed += 1;
    }

    Ok(BuildSummary {
        sessions: indexed,
        nodes,
        output_dir: output_dir.to_path_buf(),
    })
}

pub fn document_metadata(
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
) -> Result<PageIndexDocMetadata> {
    Ok(load_document(index_dir, doc_selector)?.metadata())
}

pub fn document_structure(
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
) -> Result<PageIndexDocStructure> {
    Ok(load_document(index_dir, doc_selector)?.structure_without_text())
}

pub fn document_content(
    index_dir: &Path,
    doc_selector: impl AsRef<Path>,
    locator: &str,
) -> Result<PageIndexDocContent> {
    let doc = load_document(index_dir, doc_selector)?;
    let text = content_for_locator(&doc, locator)?;
    Ok(PageIndexDocContent {
        doc_id: doc.doc_id,
        source_path: doc.source_path,
        locator: locator.to_string(),
        text,
    })
}

pub fn query_index(
    index_dir: &Path,
    query: &str,
    limit: usize,
) -> Result<Vec<PageIndexQueryResult>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let query_terms = query_terms(query);
    if query_terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    for doc in load_documents(index_dir)? {
        for node in flatten_nodes(&doc.nodes) {
            if let Some(result) = score_query_node(&doc, node, &query_terms) {
                results.push(result);
            }
        }
    }

    results.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    results.truncate(limit);
    Ok(results)
}

fn build_exchange_nodes(turns: &[Turn]) -> Vec<PageIndexNode> {
    let mut nodes = Vec::new();
    let mut start = 0;
    while start < turns.len() {
        let end = exchange_end(turns, start);
        nodes.push(build_node(nodes.len() + 1, &turns[start..=end]));
        start = end + 1;
    }
    nodes
}

fn exchange_end(turns: &[Turn], start: usize) -> usize {
    let mut end = start;
    for (index, turn) in turns.iter().enumerate().skip(start + 1) {
        if matches!(turn.role, Role::User) {
            break;
        }
        end = index;
    }
    end
}

fn build_node(position: usize, turns: &[Turn]) -> PageIndexNode {
    let first_turn = turns.first().expect("node must have at least one turn");
    let last_turn = turns.last().expect("node must have at least one turn");
    PageIndexNode {
        node_id: format_node_id(position),
        title: node_title(first_turn),
        summary: node_summary(turns),
        source_locator: format!("turns:{}-{}", first_turn.turn_index, last_turn.turn_index),
        start_turn: first_turn.turn_index,
        end_turn: last_turn.turn_index,
        text: format_turns(turns),
        nodes: Vec::new(),
    }
}

fn find_node<'a>(nodes: &'a [PageIndexNode], node_id: &str) -> Option<&'a PageIndexNode> {
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

fn count_nodes(nodes: &[PageIndexNode]) -> usize {
    nodes.len()
        + nodes
            .iter()
            .map(|node| count_nodes(&node.nodes))
            .sum::<usize>()
}

fn format_node_id(position: usize) -> String {
    format!("{position:06}")
}

fn load_document(index_dir: &Path, selector: impl AsRef<Path>) -> Result<PageIndexDoc> {
    let selector = selector.as_ref();
    let candidates = document_candidates(index_dir, selector);
    for path in candidates {
        if path.exists() {
            return read_document_file(&path);
        }
    }
    bail!(
        "document not found in transcript PageIndex: {}",
        selector.display()
    )
}

fn load_documents(index_dir: &Path) -> Result<Vec<PageIndexDoc>> {
    if !index_dir.exists() {
        return Ok(Vec::new());
    }

    let mut paths: Vec<PathBuf> = WalkDir::new(index_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths.iter().map(|path| read_document_file(path)).collect()
}

fn document_candidates(index_dir: &Path, selector: &Path) -> Vec<PathBuf> {
    let selector_text = selector.to_string_lossy();
    let doc_id = selector
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&selector_text);
    vec![
        index_dir.join(selector),
        index_dir.join(format!("{selector_text}.json")),
        index_dir.join(format!("{}.json", sanitize_doc_id(&selector_text))),
        index_dir.join(format!("{}.json", sanitize_doc_id(doc_id))),
    ]
}

fn read_document_file(path: &Path) -> Result<PageIndexDoc> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn content_for_locator(doc: &PageIndexDoc, locator: &str) -> Result<String> {
    if let Some(node) = find_node(&doc.nodes, locator) {
        return Ok(format_content_text(&node.text));
    }

    if let Some((start, end)) = parse_turn_range(locator) {
        return content_for_turn_range(doc, start, end);
    }

    bail!(
        "locator must be a node id or inclusive turn range like 4-8: {}",
        locator
    )
}

fn parse_turn_range(locator: &str) -> Option<(usize, usize)> {
    let locator = locator.strip_prefix("turns:").unwrap_or(locator);
    let (start, end) = locator.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    if end < start {
        return None;
    }
    Some((start, end))
}

fn content_for_turn_range(doc: &PageIndexDoc, start: usize, end: usize) -> Result<String> {
    let turns = doc.text.split("\n\n").collect::<Vec<_>>();
    if start >= turns.len() {
        bail!(
            "turn range starts after end of transcript: {} has {} turns",
            doc.doc_id,
            turns.len()
        );
    }

    let end = end.min(turns.len().saturating_sub(1));
    Ok(format_content_text(&turns[start..=end].join("\n\n")))
}

fn format_content_text(text: &str) -> String {
    if text.ends_with('\n') {
        text.to_string()
    } else {
        format!("{text}\n")
    }
}

fn flatten_nodes(nodes: &[PageIndexNode]) -> Vec<&PageIndexNode> {
    let mut flattened = Vec::new();
    for node in nodes {
        flattened.push(node);
        flattened.extend(flatten_nodes(&node.nodes));
    }
    flattened
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|term| term.len() > 1)
        .map(str::to_lowercase)
        .collect()
}

fn score_query_node(
    doc: &PageIndexDoc,
    node: &PageIndexNode,
    query_terms: &[String],
) -> Option<PageIndexQueryResult> {
    let searchable = format!(
        "{}\n{}\n{}\n{}",
        doc.doc_id, node.title, node.summary, node.text
    );
    let searchable = searchable.to_lowercase();
    let matched_terms: Vec<&str> = query_terms
        .iter()
        .map(String::as_str)
        .filter(|term| searchable.contains(term))
        .collect();
    if matched_terms.is_empty() {
        return None;
    }

    let score = matched_terms.len() * 10;
    Some(PageIndexQueryResult {
        doc_id: doc.doc_id.clone(),
        source_path: doc.source_path.clone(),
        node_id: node.node_id.clone(),
        title: node.title.clone(),
        score,
        reason: format!("matched query terms: {}", matched_terms.join(", ")),
        next_content_command: format!(
            "claude-memory transcript-page-index content {} {}",
            doc.doc_id, node.node_id
        ),
    })
}

fn node_title(turn: &Turn) -> String {
    let title = turn.text.lines().next().unwrap_or("<empty turn>").trim();
    truncate_chars(title, 80)
}

fn node_summary(turns: &[Turn]) -> String {
    let user_text_count = turns
        .iter()
        .filter(|turn| matches!(turn.role, Role::User) && !turn.text.trim().is_empty())
        .count();
    let assistant_text_count = turns
        .iter()
        .filter(|turn| matches!(turn.role, Role::Assistant) && !turn.text.trim().is_empty())
        .count();
    let tool_call_count: u32 = turns
        .iter()
        .filter(|turn| matches!(turn.role, Role::Assistant))
        .map(|turn| turn.tool_call_count)
        .sum();

    let mut parts = vec![
        format!("{user_text_count} user text turn(s)"),
        format!("{assistant_text_count} assistant text turn(s)"),
    ];
    if tool_call_count > 0 {
        parts.push(format!("{tool_call_count} tool call(s)"));
    }
    parts.join(", ")
}

fn format_turn(turn: &Turn) -> String {
    let role = match turn.role {
        Role::User => "User",
        Role::Assistant => "Assistant",
    };
    if turn.text.trim().is_empty() && turn.tool_call_count > 0 {
        return format!("{role}: <{} tool call(s)>", turn.tool_call_count);
    }
    format!("{role}: {}", turn.text)
}

fn format_turns(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(format_turn)
        .collect::<Vec<_>>()
        .join("\n\n")
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

fn session_id(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("session");
    name.strip_suffix(".jsonl.zst")
        .or_else(|| name.strip_suffix(".jsonl"))
        .unwrap_or(name)
        .to_string()
}

fn sanitize_doc_id(doc_id: &str) -> String {
    doc_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn collect_live_sessions(projects_dir: &Path) -> Vec<PathBuf> {
    collect_files(projects_dir, |path| {
        path.extension().is_some_and(|ext| ext == "jsonl")
    })
}

fn collect_claude_archive_sessions(archive_dir: &Path) -> Vec<PathBuf> {
    collect_files(archive_dir, |path| {
        path.to_string_lossy().ends_with(".jsonl.zst")
    })
}

fn collect_codex_sessions(dir: &Path) -> Vec<PathBuf> {
    collect_files(dir, |path| {
        path.extension().is_some_and(|ext| ext == "jsonl")
    })
}

fn collect_files<F>(dir: &Path, predicate: F) -> Vec<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    if !dir.exists() {
        return Vec::new();
    }

    WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| path.is_file())
        .filter(|path| predicate(path))
        .collect()
}

fn read_turns_for_page_index(path: &Path) -> Result<Vec<Turn>> {
    if is_codex_session(path) {
        read_codex_turns(path)
    } else {
        read_session_turns(path)
    }
}

fn is_codex_session(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == ".codex")
}

pub fn read_codex_turns(path: &Path) -> Result<Vec<Turn>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    read_codex_turns_from(BufReader::new(file))
}

fn read_codex_turns_from<R: BufRead>(reader: R) -> Result<Vec<Turn>> {
    let mut turns = Vec::new();
    for line in reader.lines() {
        let line = line.context("failed to read codex line")?;
        if let Some((role, text, tool_call_count)) = parse_codex_line(&line) {
            let turn_index = turns.len() as u32;
            turns.push(Turn {
                role,
                text,
                turn_index,
                has_tool_use: tool_call_count > 0,
                tool_call_count,
            });
        }
    }
    Ok(turns)
}

fn parse_codex_line(line: &str) -> Option<(Role, String, u32)> {
    let value: Value = serde_json::from_str(line).ok()?;
    if value.get("type")?.as_str()? != "response_item" {
        return None;
    }
    let payload = value.get("payload")?;
    match payload.get("type")?.as_str()? {
        "message" => parse_codex_message(payload),
        "function_call" => Some((Role::Assistant, String::new(), 1)),
        _ => None,
    }
}

fn parse_codex_message(payload: &Value) -> Option<(Role, String, u32)> {
    let role = match payload.get("role")?.as_str()? {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => return None,
    };
    let text = codex_content_text(payload.get("content")?);
    if text.trim().is_empty() {
        return None;
    }
    if matches!(role, Role::User) && is_codex_context_prelude(&text) {
        return None;
    }
    Some((role, text, 0))
}

fn is_codex_context_prelude(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("# AGENTS.md instructions")
        || trimmed.starts_with("<environment_context>")
        || trimmed.starts_with("<collaboration_mode>")
        || trimmed.starts_with("<skills_instructions>")
        || trimmed.starts_with("<permissions instructions>")
}

fn codex_content_text(content: &Value) -> String {
    let Some(blocks) = content.as_array() else {
        return content.as_str().unwrap_or_default().trim().to_string();
    };
    blocks
        .iter()
        .filter_map(codex_text_block)
        .collect::<Vec<_>>()
        .join("\n")
}

fn codex_text_block(block: &Value) -> Option<String> {
    let block_type = block.get("type")?.as_str()?;
    if block_type != "input_text" && block_type != "output_text" {
        return None;
    }
    let text = block.get("text")?.as_str()?.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

#[cfg(test)]
#[path = "page_index_tests.rs"]
mod tests;
