//! Local PageIndex-style outline tree for raw Claude/Codex transcript history.

use anyhow::{Context, Result};
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
mod tests {
    use super::*;
    use crate::extract::{Role, Turn};
    use std::path::Path;

    fn turn(role: Role, text: &str, turn_index: u32) -> Turn {
        Turn {
            role,
            text: text.to_string(),
            turn_index,
            has_tool_use: false,
            tool_call_count: 0,
        }
    }

    #[test]
    fn session_index_uses_nested_document_model() {
        let turns = vec![
            turn(Role::User, "How do we deploy?", 0),
            turn(Role::Assistant, "Run the deploy script.", 1),
            turn(Role::User, "How do we test?", 2),
        ];

        let index = build_session_index(Path::new("/tmp/session.jsonl"), &turns);

        assert_eq!(index.doc_id, "session");
        assert_eq!(index.doc_name, "session");
        assert_eq!(index.source_family, "transcript");
        assert_eq!(index.source_path, "/tmp/session.jsonl");
        assert_eq!(index.turn_count, 3);
        assert!(index.text.contains("User: How do we deploy?"));
        assert_eq!(index.nodes.len(), 2);

        let first_node = &index.nodes[0];
        assert_eq!(first_node.node_id, "000001");
        assert_eq!(first_node.source_locator, "turns:0-1");
        assert_eq!(first_node.nodes.len(), 0);
        assert!(
            first_node
                .text
                .contains("Assistant: Run the deploy script.")
        );
    }

    #[test]
    fn session_index_groups_prompt_and_answer_in_one_node() {
        let turns = vec![
            turn(Role::User, "How do we deploy?", 0),
            turn(Role::Assistant, "Run the deploy script.", 1),
            turn(Role::User, "How do we test?", 2),
        ];

        let index = build_session_index(Path::new("session.jsonl"), &turns);

        assert_eq!(index.nodes.len(), 2);
        assert_eq!(index.nodes[0].node_id, "000001");
        assert_eq!(index.nodes[0].start_turn, 0);
        assert_eq!(index.nodes[0].end_turn, 1);
    }

    #[test]
    fn outline_exposes_node_ids_and_titles() {
        let turns = vec![turn(Role::User, "How do we deploy safely?", 0)];
        let index = build_session_index(Path::new("session.jsonl"), &turns);

        let outline = index.outline();

        assert!(outline.contains("000001. How do we deploy safely?"));
    }

    #[test]
    fn node_text_returns_prompt_and_answer() {
        let turns = vec![
            turn(Role::User, "How do we deploy?", 0),
            turn(Role::Assistant, "Run the deploy script.", 1),
        ];
        let index = build_session_index(Path::new("session.jsonl"), &turns);

        let text = index.node_text("000001").unwrap();

        assert!(text.contains("User: How do we deploy?"));
        assert!(text.contains("Assistant: Run the deploy script."));
    }

    #[test]
    fn summary_separates_text_turns_from_tool_calls() {
        let turns = vec![
            turn(Role::User, "Please inspect the repo.", 0),
            Turn {
                role: Role::Assistant,
                text: String::new(),
                turn_index: 1,
                has_tool_use: true,
                tool_call_count: 3,
            },
            turn(Role::Assistant, "Done.", 2),
        ];

        let index = build_session_index(Path::new("session.jsonl"), &turns);

        assert_eq!(
            index.nodes[0].summary,
            "1 user text turn(s), 1 assistant text turn(s), 3 tool call(s)"
        );
    }

    #[test]
    fn codex_parser_keeps_only_user_and_assistant_messages() {
        let input = r##"{"type":"session_meta","payload":{"id":"ignored"}}
{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"rules"}]}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"# AGENTS.md instructions for /tmp/repo\nrules"}]}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"try it"}]}}
{"type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"{}"}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}
"##;

        let turns = read_codex_turns_from(BufReader::new(input.as_bytes())).unwrap();

        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].role, Role::User);
        assert_eq!(turns[0].text, "try it");
        assert_eq!(turns[1].tool_call_count, 1);
        assert_eq!(turns[2].role, Role::Assistant);
        assert_eq!(turns[2].text, "done");
    }

    #[test]
    fn page_index_sources_collect_claude_archive_and_codex_sessions() {
        let root = std::env::temp_dir().join(format!("page-index-sources-{}", std::process::id()));
        let claude_projects = root.join("claude/projects");
        let claude_archive = root.join("claude/archive");
        let codex_sessions = root.join("codex/sessions/2026/05/06");
        let codex_archive = root.join("codex/archived_sessions");
        std::fs::create_dir_all(&claude_projects).unwrap();
        std::fs::create_dir_all(&claude_archive).unwrap();
        std::fs::create_dir_all(&codex_sessions).unwrap();
        std::fs::create_dir_all(&codex_archive).unwrap();
        std::fs::write(claude_projects.join("live.jsonl"), "").unwrap();
        std::fs::write(claude_archive.join("archive.jsonl.zst"), "").unwrap();
        std::fs::write(codex_sessions.join("codex-live.jsonl"), "").unwrap();
        std::fs::write(codex_archive.join("codex-archive.jsonl"), "").unwrap();

        let sources = PageIndexSources {
            claude_projects_dir: &claude_projects,
            claude_archive_dir: &claude_archive,
            codex_sessions_dir: &root.join("codex/sessions"),
            codex_archive_dir: &codex_archive,
        };

        let files = collect_session_files(&sources);

        assert_eq!(files.len(), 4);
        let _ = std::fs::remove_dir_all(root);
    }
}
