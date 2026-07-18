//! Qdrant indexing and search.

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use serde::Serialize;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tokio::sync::Mutex;
use walkdir::WalkDir;

use crate::embed::Embedder;
use crate::extract::{
    HistoryType, IndexedChunk, extract_codex_jsonl, extract_codex_jsonl_answers, extract_jsonl,
    extract_jsonl_answers, extract_zst, extract_zst_answers,
};
use crate::qdrant_hybrid::ensure_hybrid_collection;

#[path = "index_search.rs"]
mod index_search;
#[path = "index_writer.rs"]
mod index_writer;
mod search_results;
pub use index_search::{global_history_filter, history_filter};
pub use index_search::{
    search_all, search_answer_sources, search_answers, search_prompt_sources, search_prompts,
};
pub(crate) use index_writer::filter_new;
#[cfg(test)]
pub(crate) use index_writer::history_hash;
use index_writer::{get_existing_hashes, index_chunks};
#[cfg(test)]
pub(crate) use search_results::{build_search_results, get_string};

pub const QDRANT_URL: &str = "http://localhost:6334";
pub const COLLECTION_SESSION_HISTORY: &str = "claude-session-history";

#[derive(Serialize)]
pub struct SearchResult {
    #[serde(rename = "type")]
    pub record_type: String,
    pub text: String,
    pub source: String,
    pub path: String,
    pub session_id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum IndexFileFormat {
    ClaudePi,
    ClaudeZst,
    Codex,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum IndexFileSource {
    Session,
    Archive,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct IndexFile {
    pub path: PathBuf,
    pub format: IndexFileFormat,
    pub source: IndexFileSource,
}

pub struct IndexSources<'a> {
    pub claude_projects_dir: &'a Path,
    pub claude_archive_dir: &'a Path,
    pub codex_sessions_dir: &'a Path,
    pub codex_archive_dir: &'a Path,
    pub pi_sessions_dir: &'a Path,
}

/// Shared state for indexing operations.
struct IndexState {
    client: Qdrant,
    embedder: Embedder,
    hashes: Mutex<HashSet<String>>,
    batch_size: usize,
    delay_ms: u64,
}

/// Run the indexing process with streaming (low memory).
pub async fn run_index(
    archive_dir: &Path,
    projects_dir: &Path,
    batch_size: usize,
    fresh: bool,
    delay_ms: u64,
) -> Result<()> {
    let state = init_index_state(batch_size, fresh, delay_ms).await?;

    let jsonls = collect_jsonl_files(projects_dir);
    let archives = collect_archive_files(archive_dir);

    eprintln!("\n=== Indexing session prompts (user messages) ===");
    let prompts_indexed = index_all_prompts(&state, projects_dir, &jsonls, &archives).await?;

    eprintln!("\n=== Indexing answers (assistant responses) ===");
    let answers_indexed = index_all_answers(&state, projects_dir, &jsonls, &archives).await?;

    eprintln!(
        "\nDone! Prompts indexed: {}, Answers indexed: {}",
        prompts_indexed, answers_indexed
    );
    Ok(())
}

/// Index all configured Claude, Codex, and Pi session sources.
pub async fn run_index_sources(
    sources: &IndexSources<'_>,
    batch_size: usize,
    fresh: bool,
    delay_ms: u64,
) -> Result<()> {
    let state = init_index_state(batch_size, fresh, delay_ms).await?;
    let files = collect_index_files(sources);

    eprintln!("\n=== Indexing session prompts (user messages) ===");
    let prompts_indexed =
        index_discovered_files(&state, sources, &files, HistoryType::Prompt).await?;

    eprintln!("\n=== Indexing answers (assistant responses) ===");
    let answers_indexed =
        index_discovered_files(&state, sources, &files, HistoryType::Answer).await?;

    eprintln!(
        "\nDone! Prompts indexed: {}, Answers indexed: {}",
        prompts_indexed, answers_indexed
    );
    Ok(())
}

/// Recursively discover supported files from every configured source root.
pub fn collect_index_files(sources: &IndexSources<'_>) -> Vec<IndexFile> {
    let mut files = collect_index_files_from(
        sources.claude_projects_dir,
        IndexFileFormat::ClaudePi,
        IndexFileSource::Session,
        is_jsonl,
    );
    files.extend(collect_index_files_from(
        sources.claude_archive_dir,
        IndexFileFormat::ClaudeZst,
        IndexFileSource::Archive,
        is_jsonl_zst,
    ));
    files.extend(collect_index_files_from(
        sources.codex_sessions_dir,
        IndexFileFormat::Codex,
        IndexFileSource::Session,
        is_jsonl,
    ));
    files.extend(collect_index_files_from(
        sources.codex_archive_dir,
        IndexFileFormat::Codex,
        IndexFileSource::Archive,
        is_jsonl,
    ));
    files.extend(collect_index_files_from(
        sources.pi_sessions_dir,
        IndexFileFormat::ClaudePi,
        IndexFileSource::Session,
        is_pi_session_jsonl,
    ));
    files.sort_by(|left, right| left.path.cmp(&right.path));
    files
}

fn collect_index_files_from(
    root: &Path,
    format: IndexFileFormat,
    source: IndexFileSource,
    matches: fn(&Path) -> bool,
) -> Vec<IndexFile> {
    if !root.exists() {
        return vec![];
    }

    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file() && matches(entry.path()))
        .map(|entry| IndexFile {
            path: entry.path().to_path_buf(),
            format,
            source,
        })
        .collect()
}

fn is_jsonl(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "jsonl")
}

fn is_jsonl_zst(path: &Path) -> bool {
    path.to_string_lossy().ends_with(".jsonl.zst")
}

fn is_pi_session_jsonl(path: &Path) -> bool {
    if !is_jsonl(path) {
        return false;
    }
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
        .next()
        .is_some_and(|value| {
            value.get("type").and_then(serde_json::Value::as_str) == Some("session")
        })
}

async fn init_index_state(batch_size: usize, fresh: bool, delay_ms: u64) -> Result<IndexState> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    ensure_hybrid_collection(&client, COLLECTION_SESSION_HISTORY).await?;
    let hashes = load_hashes(&client, fresh).await?;
    eprintln!("Found {} session-history chunks", hashes.len());
    Ok(build_index_state(client, hashes, batch_size, delay_ms))
}

async fn load_hashes(client: &Qdrant, fresh: bool) -> Result<HashSet<String>> {
    if fresh {
        eprintln!("Fresh index requested, ignoring existing data");
        return Ok(HashSet::new());
    }
    eprintln!("Loading existing hashes for resume...");
    get_existing_hashes(client, COLLECTION_SESSION_HISTORY).await
}

fn build_index_state(
    client: Qdrant,
    hashes: HashSet<String>,
    batch_size: usize,
    delay_ms: u64,
) -> IndexState {
    IndexState {
        client,
        embedder: Embedder::new(),
        hashes: Mutex::new(hashes),
        batch_size,
        delay_ms,
    }
}

fn collect_jsonl_files(projects_dir: &Path) -> Vec<walkdir::DirEntry> {
    if !projects_dir.exists() {
        return vec![];
    }
    WalkDir::new(projects_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "jsonl")
                .unwrap_or(false)
                && e.path()
                    .file_name()
                    .map(|f| f != "sessions-index.json")
                    .unwrap_or(true)
        })
        .collect()
}

fn collect_archive_files(archive_dir: &Path) -> Vec<std::fs::DirEntry> {
    if !archive_dir.exists() {
        return vec![];
    }
    std::fs::read_dir(archive_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .map(|f| f.to_string_lossy().ends_with(".jsonl.zst"))
                .unwrap_or(false)
        })
        .collect()
}

async fn index_discovered_files(
    state: &IndexState,
    sources: &IndexSources<'_>,
    files: &[IndexFile],
    history_type: HistoryType,
) -> Result<usize> {
    let kind = history_type.as_str();
    let mut indexed = 0;
    eprintln!("Processing {} discovered files ({kind})...", files.len());

    for (position, file) in files.iter().enumerate() {
        match extract_index_file(file, sources, history_type) {
            Ok(chunks) => indexed += index_new_chunks(state, &chunks).await?,
            Err(error) => tracing::warn!("Failed to extract {}: {}", file.path.display(), error),
        }
        report_index_progress("Sources", position, files.len(), indexed);
    }
    eprintln!();
    Ok(indexed)
}

fn extract_index_file(
    file: &IndexFile,
    sources: &IndexSources<'_>,
    history_type: HistoryType,
) -> Result<Vec<IndexedChunk>> {
    let mut chunks = match file.format {
        IndexFileFormat::ClaudePi => {
            let base_path = if file.path.starts_with(sources.pi_sessions_dir) {
                sources.pi_sessions_dir
            } else {
                sources.claude_projects_dir
            };
            extract_claude_jsonl(&file.path, base_path, history_type)?
        }
        IndexFileFormat::ClaudeZst => extract_claude_zst(&file.path, history_type)?,
        IndexFileFormat::Codex => {
            let base_path = match file.source {
                IndexFileSource::Session => sources.codex_sessions_dir,
                IndexFileSource::Archive => sources.codex_archive_dir,
            };
            extract_codex_jsonl_history(&file.path, base_path, history_type)?
        }
    };

    if file.source == IndexFileSource::Archive {
        for chunk in &mut chunks {
            chunk.source = "archive".to_string();
        }
    }
    Ok(chunks)
}

fn extract_claude_jsonl(
    path: &Path,
    base_path: &Path,
    history_type: HistoryType,
) -> Result<Vec<IndexedChunk>> {
    match history_type {
        HistoryType::Prompt => extract_jsonl(path, base_path),
        HistoryType::Answer => extract_jsonl_answers(path, base_path),
    }
}

fn extract_claude_zst(path: &Path, history_type: HistoryType) -> Result<Vec<IndexedChunk>> {
    match history_type {
        HistoryType::Prompt => extract_zst(path),
        HistoryType::Answer => extract_zst_answers(path),
    }
}

fn extract_codex_jsonl_history(
    path: &Path,
    base_path: &Path,
    history_type: HistoryType,
) -> Result<Vec<IndexedChunk>> {
    match history_type {
        HistoryType::Prompt => extract_codex_jsonl(path, base_path),
        HistoryType::Answer => extract_codex_jsonl_answers(path, base_path),
    }
}

async fn index_all_prompts(
    state: &IndexState,
    projects_dir: &Path,
    jsonls: &[walkdir::DirEntry],
    archives: &[std::fs::DirEntry],
) -> Result<usize> {
    index_history(state, projects_dir, jsonls, archives, HistoryType::Prompt).await
}

async fn index_all_answers(
    state: &IndexState,
    projects_dir: &Path,
    jsonls: &[walkdir::DirEntry],
    archives: &[std::fs::DirEntry],
) -> Result<usize> {
    index_history(state, projects_dir, jsonls, archives, HistoryType::Answer).await
}

async fn index_history(
    state: &IndexState,
    projects_dir: &Path,
    jsonls: &[walkdir::DirEntry],
    archives: &[std::fs::DirEntry],
    history_type: HistoryType,
) -> Result<usize> {
    let session_paths: Vec<_> = jsonls
        .iter()
        .map(|entry| entry.path().to_path_buf())
        .collect();
    let archive_paths: Vec<_> = archives.iter().map(std::fs::DirEntry::path).collect();
    let kind = history_type.as_str();

    let indexed_sessions =
        index_history_paths(state, &session_paths, "Sessions", kind, 0, |path| {
            extract_session_history(path, projects_dir, history_type)
        })
        .await?;
    let indexed_archives = index_history_paths(
        state,
        &archive_paths,
        "Archives",
        kind,
        indexed_sessions,
        |path| extract_archive_history(path, history_type),
    )
    .await?;
    Ok(indexed_sessions + indexed_archives)
}

async fn index_history_paths<F>(
    state: &IndexState,
    paths: &[std::path::PathBuf],
    label: &str,
    kind: &str,
    base_indexed: usize,
    extract: F,
) -> Result<usize>
where
    F: Fn(&Path) -> Result<Vec<IndexedChunk>>,
{
    let mut indexed = 0;
    eprintln!(
        "Processing {} {} ({kind})...",
        paths.len(),
        label.to_lowercase()
    );
    for (position, path) in paths.iter().enumerate() {
        match extract(path) {
            Ok(chunks) => indexed += index_new_chunks(state, &chunks).await?,
            Err(error) => tracing::warn!("Failed to extract {}: {}", path.display(), error),
        }
        report_index_progress(label, position, paths.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

fn report_index_progress(label: &str, position: usize, total: usize, indexed: usize) {
    eprint!("\r  {label}: {}/{total} (indexed: {indexed})", position + 1);
}

fn extract_session_history(
    path: &Path,
    projects_dir: &Path,
    history_type: HistoryType,
) -> Result<Vec<IndexedChunk>> {
    match history_type {
        HistoryType::Prompt => extract_jsonl(path, projects_dir),
        HistoryType::Answer => extract_jsonl_answers(path, projects_dir),
    }
}

fn extract_archive_history(path: &Path, history_type: HistoryType) -> Result<Vec<IndexedChunk>> {
    match history_type {
        HistoryType::Prompt => extract_zst(path),
        HistoryType::Answer => extract_zst_answers(path),
    }
}

/// Filter and index new chunks, returning the count indexed.
async fn index_new_chunks(state: &IndexState, chunks: &[IndexedChunk]) -> Result<usize> {
    let new_chunks = {
        let hashes = state.hashes.lock().await;
        filter_new(chunks, &hashes)
    };
    if new_chunks.is_empty() {
        return Ok(0);
    }

    let indexed = index_chunks(
        &state.client,
        &state.embedder,
        &new_chunks,
        state.batch_size,
        COLLECTION_SESSION_HISTORY,
        state.delay_ms,
    )
    .await?;

    let mut hashes = state.hashes.lock().await;
    hashes.extend(new_chunks.iter().map(index_writer::history_hash));
    Ok(indexed)
}

fn read_index_file_format(path: &Path) -> Result<IndexFileFormat> {
    if is_jsonl_zst(path) {
        return Ok(IndexFileFormat::ClaudeZst);
    }
    if !is_jsonl(path) {
        anyhow::bail!("Unsupported file type: {}", path.display());
    }

    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("response_item") => return Ok(IndexFileFormat::Codex),
            Some("session" | "user" | "assistant") => return Ok(IndexFileFormat::ClaudePi),
            _ => {}
        }
    }
    anyhow::bail!("Unsupported JSONL session format: {}", path.display())
}

pub(crate) fn extract_single_file_history(
    path: &Path,
    history_type: HistoryType,
) -> Result<Vec<IndexedChunk>> {
    let base_path = path.parent().unwrap_or(path);
    let mut chunks = match read_index_file_format(path)? {
        IndexFileFormat::ClaudePi => extract_claude_jsonl(path, base_path, history_type)?,
        IndexFileFormat::ClaudeZst => extract_claude_zst(path, history_type)?,
        IndexFileFormat::Codex => extract_codex_jsonl_history(path, base_path, history_type)?,
    };
    if path
        .components()
        .any(|component| component.as_os_str() == "archived_sessions")
    {
        for chunk in &mut chunks {
            chunk.source = "archive".to_string();
        }
    }
    Ok(chunks)
}

/// Index a single conversation file (both prompts and answers).
pub async fn index_file(path: &Path, batch_size: usize) -> Result<usize> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    ensure_hybrid_collection(&client, COLLECTION_SESSION_HISTORY).await?;

    let hashes = get_existing_hashes(&client, COLLECTION_SESSION_HISTORY).await?;

    let embedder = Embedder::new();
    let mut total = 0;

    total += index_file_prompts(path, &client, &embedder, batch_size, &hashes).await?;
    total += index_file_answers(path, &client, &embedder, batch_size, &hashes).await?;

    Ok(total)
}

async fn index_file_prompts(
    path: &Path,
    client: &Qdrant,
    embedder: &Embedder,
    batch_size: usize,
    hashes: &HashSet<String>,
) -> Result<usize> {
    let chunks = extract_single_file_history(path, HistoryType::Prompt)?;

    let new_chunks = filter_new(&chunks, hashes);
    if new_chunks.is_empty() {
        return Ok(0);
    }
    index_chunks(
        client,
        embedder,
        &new_chunks,
        batch_size,
        COLLECTION_SESSION_HISTORY,
        0,
    )
    .await
}

async fn index_file_answers(
    path: &Path,
    client: &Qdrant,
    embedder: &Embedder,
    batch_size: usize,
    hashes: &HashSet<String>,
) -> Result<usize> {
    let chunks = extract_single_file_history(path, HistoryType::Answer)?;

    let new_chunks = filter_new(&chunks, hashes);
    if new_chunks.is_empty() {
        return Ok(0);
    }
    index_chunks(
        client,
        embedder,
        &new_chunks,
        batch_size,
        COLLECTION_SESSION_HISTORY,
        0,
    )
    .await
}

/// Show collection statistics.
pub async fn show_stats() -> Result<()> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    if let Ok(info) = client.collection_info(COLLECTION_SESSION_HISTORY).await {
        let points = info
            .result
            .and_then(|result| result.points_count)
            .unwrap_or(0);
        println!(
            "Session history ({}): {} points",
            COLLECTION_SESSION_HISTORY, points
        );
    } else {
        println!(
            "Session history ({}): not found",
            COLLECTION_SESSION_HISTORY
        );
    }

    Ok(())
}

#[cfg(test)]
#[path = "index_tests.rs"]
mod index_tests;
