//! Extract text from Claude Code conversation files and markdown.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::chunk::{chunk_text, Chunk};

#[derive(Debug, Clone)]
pub struct IndexedChunk {
    pub chunk: Chunk,
    pub source: String,
    pub path: String,
    pub session_id: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    message: Option<MessageContent>,
}

#[derive(Deserialize)]
struct MessageContent {
    content: Option<serde_json::Value>,
}

/// Content block in assistant messages.
#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    text: Option<String>,
}

/// Extract chunks from a session summary markdown file.
pub fn extract_summary(path: &Path, base_path: &Path) -> Result<Vec<IndexedChunk>> {
    let text = std::fs::read_to_string(path).context("failed to read summary")?;
    let session_id = path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string());

    let rel_path = path
        .strip_prefix(base_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string());

    Ok(chunk_text(&text)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: "summary".to_string(),
            path: rel_path.clone(),
            session_id: session_id.clone(),
        })
        .collect())
}

/// Extract user messages from an uncompressed JSONL file.
pub fn extract_jsonl(path: &Path, base_path: &Path) -> Result<Vec<IndexedChunk>> {
    let session_id = path.file_stem().map(|s| s.to_string_lossy().to_string());
    let rel_path = path
        .strip_prefix(base_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string());

    let file = File::open(path).context("failed to open JSONL")?;
    let reader = BufReader::new(file);

    let texts = extract_user_messages(reader)?;
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let combined = texts.join("\n");
    Ok(chunk_text(&combined)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: "session".to_string(),
            path: rel_path.clone(),
            session_id: session_id.clone(),
        })
        .collect())
}

/// Extract user messages from a zstd-compressed JSONL file.
pub fn extract_zst(path: &Path) -> Result<Vec<IndexedChunk>> {
    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.trim_end_matches(".jsonl").to_string());

    let file = File::open(path).context("failed to open ZST")?;
    let decoder = zstd::Decoder::new(file).context("failed to create zstd decoder")?;
    let reader = BufReader::new(decoder);

    let texts = extract_user_messages(reader)?;
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let combined = texts.join("\n");
    let file_path = path.file_name().unwrap().to_string_lossy().to_string();
    Ok(chunk_text(&combined)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: "archive".to_string(),
            path: file_path.clone(),
            session_id: session_id.clone(),
        })
        .collect())
}

/// Extract assistant responses from an uncompressed JSONL file.
pub fn extract_jsonl_answers(path: &Path, base_path: &Path) -> Result<Vec<IndexedChunk>> {
    let session_id = path.file_stem().map(|s| s.to_string_lossy().to_string());
    let rel_path = path
        .strip_prefix(base_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string());

    let file = File::open(path).context("failed to open JSONL")?;
    let reader = BufReader::new(file);

    let texts = extract_assistant_messages(reader)?;
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let combined = texts.join("\n");
    Ok(chunk_text(&combined)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: "session".to_string(),
            path: rel_path.clone(),
            session_id: session_id.clone(),
        })
        .collect())
}

/// Extract assistant responses from a zstd-compressed JSONL file.
pub fn extract_zst_answers(path: &Path) -> Result<Vec<IndexedChunk>> {
    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.trim_end_matches(".jsonl").to_string());

    let file = File::open(path).context("failed to open ZST")?;
    let decoder = zstd::Decoder::new(file).context("failed to create zstd decoder")?;
    let reader = BufReader::new(decoder);

    let texts = extract_assistant_messages(reader)?;
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let combined = texts.join("\n");
    let file_path = path.file_name().unwrap().to_string_lossy().to_string();
    Ok(chunk_text(&combined)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: "archive".to_string(),
            path: file_path.clone(),
            session_id: session_id.clone(),
        })
        .collect())
}

fn extract_user_messages<R: Read>(reader: BufReader<R>) -> Result<Vec<String>> {
    let mut texts = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let msg: Message = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(_) => continue,
        };

        if msg.msg_type.as_deref() != Some("user") {
            continue;
        }

        if let Some(content) = msg.message.and_then(|m| m.content) {
            match content {
                serde_json::Value::String(s) if !s.trim().is_empty() => {
                    texts.push(format!("User: {}", s));
                }
                _ => {}
            }
        }
    }

    Ok(texts)
}

fn extract_assistant_messages<R: Read>(reader: BufReader<R>) -> Result<Vec<String>> {
    let mut texts = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let msg: Message = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(_) => continue,
        };

        if msg.msg_type.as_deref() != Some("assistant") {
            continue;
        }

        if let Some(content) = msg.message.and_then(|m| m.content) {
            // Assistant content is an array of blocks
            if let serde_json::Value::Array(blocks) = content {
                for block in blocks {
                    let cb: ContentBlock = match serde_json::from_value(block) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };

                    // Only extract text blocks, skip tool_use and thinking
                    if cb.block_type.as_deref() == Some("text") {
                        if let Some(text) = cb.text {
                            if !text.trim().is_empty() {
                                texts.push(format!("Assistant: {}", text));
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(texts)
}

/// Extract chunks from a markdown file.
pub fn extract_markdown(path: &Path, base_path: &Path) -> Result<Vec<IndexedChunk>> {
    let text = std::fs::read_to_string(path).context("failed to read markdown")?;

    let rel_path = path
        .strip_prefix(base_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string());

    Ok(chunk_text(&text)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: "kb".to_string(),
            path: rel_path.clone(),
            session_id: None,
        })
        .collect())
}
