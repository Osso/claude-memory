//! Qdrant indexing and search.

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use std::collections::HashSet;
use std::path::Path;
use tokio::sync::Mutex;
use walkdir::WalkDir;

use crate::embed::Embedder;
use crate::extract::{
    HistoryType, IndexedChunk, extract_jsonl, extract_jsonl_answers, extract_zst,
    extract_zst_answers,
};
use crate::qdrant_hybrid::ensure_hybrid_collection;

#[path = "index_search.rs"]
mod index_search;
#[path = "index_writer.rs"]
mod index_writer;
mod search_results;
pub use index_search::history_filter;
pub use index_search::{
    search_answer_sources, search_answers, search_prompt_sources, search_prompts,
};
pub(crate) use index_writer::filter_new;
#[cfg(test)]
pub(crate) use index_writer::history_hash;
use index_writer::{get_existing_hashes, index_chunks};
#[cfg(test)]
pub(crate) use search_results::{build_search_results, get_string};

pub const QDRANT_URL: &str = "http://localhost:6334";
pub const COLLECTION_SESSION_HISTORY: &str = "claude-session-history";

pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub path: String,
    pub session_id: String,
    pub score: f32,
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
    let chunks = if path.to_string_lossy().ends_with(".jsonl.zst") {
        extract_zst(path)?
    } else if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
        let base = path.parent().unwrap_or(path);
        extract_jsonl(path, base)?
    } else {
        anyhow::bail!("Unsupported file type: {}", path.display());
    };

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
    let chunks = if path.to_string_lossy().ends_with(".jsonl.zst") {
        extract_zst_answers(path)?
    } else if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
        let base = path.parent().unwrap_or(path);
        extract_jsonl_answers(path, base)?
    } else {
        return Ok(0);
    };

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
