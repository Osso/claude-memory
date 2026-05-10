use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::extract::{Role, Turn};

pub(super) fn read_codex_turns(path: &Path) -> Result<Vec<Turn>> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    read_codex_turns_from(BufReader::new(file))
}

pub(super) fn read_codex_turns_from<R: BufRead>(reader: R) -> Result<Vec<Turn>> {
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
