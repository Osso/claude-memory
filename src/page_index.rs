//! Local PageIndex-style tree for raw prompt/answer transcript history.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::extract::{Role, Turn, read_session_turns};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexDoc {
    pub doc_id: String,
    pub source_path: String,
    pub title: String,
    pub nodes: Vec<PageIndexNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageIndexNode {
    pub node_id: String,
    pub title: String,
    pub summary: String,
    pub start_turn: u32,
    pub end_turn: u32,
    pub children: Vec<PageIndexNode>,
}

#[derive(Debug)]
pub struct BuildSummary {
    pub sessions: usize,
    pub nodes: usize,
    pub output_dir: PathBuf,
}

impl PageIndexDoc {
    pub fn outline(&self) -> String {
        let mut lines = vec![format!("{} — {}", self.doc_id, self.title)];
        for node in &self.nodes {
            lines.push(format!(
                "{}. {} [{}-{}] — {}",
                node.node_id, node.title, node.start_turn, node.end_turn, node.summary
            ));
        }
        lines.join("\n")
    }

    pub fn node_text(&self, node_id: &str, turns: &[Turn]) -> Option<String> {
        let node = self.nodes.iter().find(|node| node.node_id == node_id)?;
        let mut lines = Vec::new();
        for turn in turns {
            if turn.turn_index < node.start_turn || turn.turn_index > node.end_turn {
                continue;
            }
            lines.push(format_turn(turn));
        }
        Some(lines.join("\n\n"))
    }
}

pub fn build_session_index(path: &Path, turns: &[Turn]) -> PageIndexDoc {
    let doc_id = session_id(path);
    PageIndexDoc {
        title: doc_id.clone(),
        source_path: path.to_string_lossy().to_string(),
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

pub fn collect_session_files(projects_dir: &Path, archive_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut files = collect_live_sessions(projects_dir);
    if let Some(archive_dir) = archive_dir {
        files.extend(collect_archive_sessions(archive_dir));
    }
    files.sort();
    files
}

pub fn default_output_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("claude-memory/page-index")
}

pub fn build_page_index(
    projects_dir: &Path,
    archive_dir: Option<&Path>,
    output_dir: &Path,
    max_sessions: Option<usize>,
) -> Result<BuildSummary> {
    let mut sessions = collect_session_files(projects_dir, archive_dir);
    if let Some(limit) = max_sessions {
        sessions.truncate(limit);
    }

    let mut nodes = 0;
    let mut indexed = 0;
    for path in sessions {
        let turns = read_session_turns(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if turns.is_empty() {
            continue;
        }
        let index = build_session_index(&path, &turns);
        nodes += index.nodes.len();
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
        node_id: position.to_string(),
        title: node_title(first_turn),
        summary: node_summary(turns),
        start_turn: first_turn.turn_index,
        end_turn: last_turn.turn_index,
        children: Vec::new(),
    }
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

fn collect_archive_sessions(archive_dir: &Path) -> Vec<PathBuf> {
    collect_files(archive_dir, |path| {
        path.to_string_lossy().ends_with(".jsonl.zst")
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
    fn session_index_groups_prompt_and_answer_in_one_node() {
        let turns = vec![
            turn(Role::User, "How do we deploy?", 0),
            turn(Role::Assistant, "Run the deploy script.", 1),
            turn(Role::User, "How do we test?", 2),
        ];

        let index = build_session_index(Path::new("session.jsonl"), &turns);

        assert_eq!(index.nodes.len(), 2);
        assert_eq!(index.nodes[0].node_id, "1");
        assert_eq!(index.nodes[0].start_turn, 0);
        assert_eq!(index.nodes[0].end_turn, 1);
    }

    #[test]
    fn outline_exposes_node_ids_and_titles() {
        let turns = vec![turn(Role::User, "How do we deploy safely?", 0)];
        let index = build_session_index(Path::new("session.jsonl"), &turns);

        let outline = index.outline();

        assert!(outline.contains("1. How do we deploy safely?"));
    }

    #[test]
    fn node_text_returns_prompt_and_answer() {
        let turns = vec![
            turn(Role::User, "How do we deploy?", 0),
            turn(Role::Assistant, "Run the deploy script.", 1),
        ];
        let index = build_session_index(Path::new("session.jsonl"), &turns);

        let text = index.node_text("1", &turns).unwrap();

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
}
