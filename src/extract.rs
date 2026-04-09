//! Extract text from Claude Code conversation files and markdown.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::chunk::{Chunk, chunk_text};

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
    let file = File::open(path).context("failed to open JSONL")?;
    let reader = BufReader::new(file);
    let metadata = session_metadata(path, base_path);
    extract_chunks_from_reader(reader, extract_user_message, metadata, "session")
}

/// Extract user messages from a zstd-compressed JSONL file.
pub fn extract_zst(path: &Path) -> Result<Vec<IndexedChunk>> {
    let file = File::open(path).context("failed to open ZST")?;
    let decoder = zstd::Decoder::new(file).context("failed to create zstd decoder")?;
    let reader = BufReader::new(decoder);
    let metadata = archive_metadata(path);
    extract_chunks_from_reader(reader, extract_user_message, metadata, "archive")
}

/// Extract assistant responses from an uncompressed JSONL file.
pub fn extract_jsonl_answers(path: &Path, base_path: &Path) -> Result<Vec<IndexedChunk>> {
    let file = File::open(path).context("failed to open JSONL")?;
    let reader = BufReader::new(file);
    let metadata = session_metadata(path, base_path);
    extract_chunks_from_reader(reader, extract_assistant_message, metadata, "session")
}

/// Extract assistant responses from a zstd-compressed JSONL file.
pub fn extract_zst_answers(path: &Path) -> Result<Vec<IndexedChunk>> {
    let file = File::open(path).context("failed to open ZST")?;
    let decoder = zstd::Decoder::new(file).context("failed to create zstd decoder")?;
    let reader = BufReader::new(decoder);
    let metadata = archive_metadata(path);
    extract_chunks_from_reader(reader, extract_assistant_message, metadata, "archive")
}

#[cfg(test)]
fn extract_user_messages<R: Read>(reader: BufReader<R>) -> Result<Vec<String>> {
    collect_messages(reader, extract_user_message)
}

#[cfg(test)]
fn extract_assistant_messages<R: Read>(reader: BufReader<R>) -> Result<Vec<String>> {
    collect_messages(reader, extract_assistant_message)
}

fn extract_chunks_from_reader<R, F>(
    reader: BufReader<R>,
    extractor: F,
    metadata: ChunkMetadata,
    source: &str,
) -> Result<Vec<IndexedChunk>>
where
    R: Read,
    F: Fn(Message) -> Vec<String>,
{
    let texts = collect_messages(reader, extractor)?;
    Ok(build_indexed_chunks(texts, metadata, source))
}

fn collect_messages<R, F>(reader: BufReader<R>, extractor: F) -> Result<Vec<String>>
where
    R: Read,
    F: Fn(Message) -> Vec<String>,
{
    let mut texts = Vec::new();
    for message in reader
        .lines()
        .filter_map(Result::ok)
        .filter_map(parse_message)
    {
        texts.extend(extractor(message));
    }
    Ok(texts)
}

fn parse_message(line: String) -> Option<Message> {
    serde_json::from_str(&line).ok()
}

fn extract_user_message(message: Message) -> Vec<String> {
    if message.msg_type.as_deref() != Some("user") {
        return vec![];
    }

    match message_content(message) {
        Some(serde_json::Value::String(text)) if !text.trim().is_empty() => {
            vec![format!("User: {text}")]
        }
        _ => vec![],
    }
}

fn extract_assistant_message(message: Message) -> Vec<String> {
    if message.msg_type.as_deref() != Some("assistant") {
        return vec![];
    }

    let Some(serde_json::Value::Array(blocks)) = message_content(message) else {
        return vec![];
    };

    blocks
        .into_iter()
        .filter_map(parse_content_block)
        .filter_map(extract_block_text)
        .collect()
}

fn message_content(message: Message) -> Option<serde_json::Value> {
    message.message.and_then(|message| message.content)
}

fn parse_content_block(block: serde_json::Value) -> Option<ContentBlock> {
    serde_json::from_value(block).ok()
}

fn extract_block_text(block: ContentBlock) -> Option<String> {
    if block.block_type.as_deref() != Some("text") {
        return None;
    }

    let text = block.text?;
    if text.trim().is_empty() {
        return None;
    }

    Some(format!("Assistant: {text}"))
}

fn build_indexed_chunks(
    texts: Vec<String>,
    metadata: ChunkMetadata,
    source: &str,
) -> Vec<IndexedChunk> {
    if texts.is_empty() {
        return vec![];
    }

    let combined = texts.join("\n");
    chunk_text(&combined)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: source.to_string(),
            path: metadata.path.clone(),
            session_id: metadata.session_id.clone(),
        })
        .collect()
}

struct ChunkMetadata {
    path: String,
    session_id: Option<String>,
}

fn session_metadata(path: &Path, base_path: &Path) -> ChunkMetadata {
    ChunkMetadata {
        path: path
            .strip_prefix(base_path)
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string()),
        session_id: path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string()),
    }
}

fn archive_metadata(path: &Path) -> ChunkMetadata {
    ChunkMetadata {
        path: path.file_name().unwrap().to_string_lossy().to_string(),
        session_id: path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(|stem| stem.trim_end_matches(".jsonl").to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};

    fn make_reader(data: &str) -> BufReader<Cursor<Vec<u8>>> {
        BufReader::new(Cursor::new(data.as_bytes().to_vec()))
    }

    // ── extract_user_messages ──────────────────────────────────────────────

    #[test]
    fn user_messages_extracts_valid_human_turns() {
        let data = r#"{"type":"user","message":{"content":"Hello world"}}
{"type":"user","message":{"content":"Second message"}}
"#;
        let result = extract_user_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["User: Hello world", "User: Second message"]);
    }

    #[test]
    fn user_messages_skips_malformed_json() {
        let data = r#"{"type":"user","message":{"content":"Good line"}}
not json at all
{"type":"user","message":{"content":"Also good"}}
"#;
        let result = extract_user_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["User: Good line", "User: Also good"]);
    }

    #[test]
    fn user_messages_empty_input_returns_empty() {
        let result = extract_user_messages(make_reader("")).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn user_messages_skips_non_human_roles() {
        let data = r#"{"type":"assistant","message":{"content":"[{\"type\":\"text\",\"text\":\"Hi\"}]"}}
{"type":"system","message":{"content":"System prompt"}}
{"type":"user","message":{"content":"User only"}}
"#;
        let result = extract_user_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["User: User only"]);
    }

    #[test]
    fn user_messages_skips_whitespace_only_content() {
        let data = "{\"type\":\"user\",\"message\":{\"content\":\"   \"}}\n\
                    {\"type\":\"user\",\"message\":{\"content\":\"real\"}}\n";
        let result = extract_user_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["User: real"]);
    }

    // ── extract_assistant_messages ─────────────────────────────────────────

    #[test]
    fn assistant_messages_extracts_text_blocks() {
        let data = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Here is my answer."}]}}
"#;
        let result = extract_assistant_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["Assistant: Here is my answer."]);
    }

    #[test]
    fn assistant_messages_skips_tool_use_blocks() {
        let data = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{}},{"type":"text","text":"Done."}]}}
"#;
        let result = extract_assistant_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["Assistant: Done."]);
    }

    #[test]
    fn assistant_messages_mixed_valid_invalid_lines() {
        let data = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"First."}]}}
{bad json
{"type":"assistant","message":{"content":[{"type":"text","text":"Third."}]}}
"#;
        let result = extract_assistant_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["Assistant: First.", "Assistant: Third."]);
    }

    #[test]
    fn assistant_messages_skips_non_assistant_roles() {
        let data = r#"{"type":"user","message":{"content":"Human text"}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Assistant text."}]}}
"#;
        let result = extract_assistant_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["Assistant: Assistant text."]);
    }

    #[test]
    fn assistant_messages_skips_whitespace_only_text_blocks() {
        let data = "{\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"  \"}]}}\n\
                    {\"type\":\"assistant\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Real answer.\"}]}}\n";
        let result = extract_assistant_messages(make_reader(data)).unwrap();
        assert_eq!(result, vec!["Assistant: Real answer."]);
    }

    #[test]
    fn assistant_messages_empty_input_returns_empty() {
        let result = extract_assistant_messages(make_reader("")).unwrap();
        assert!(result.is_empty());
    }
}

/// Extract chunks from a markdown file.
pub fn extract_markdown(path: &Path, base_path: &Path) -> Result<Vec<IndexedChunk>> {
    let text = std::fs::read_to_string(path).context("failed to read markdown")?;

    let rel_path = path
        .strip_prefix(base_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.to_string_lossy().to_string());

    // Files under memory/ subdir are tagged as source=memory
    let source = if rel_path.starts_with("memory/") {
        "memory"
    } else {
        "kb"
    };

    Ok(chunk_text(&text)
        .into_iter()
        .map(|chunk| IndexedChunk {
            chunk,
            source: source.to_string(),
            path: rel_path.clone(),
            session_id: None,
        })
        .collect())
}
