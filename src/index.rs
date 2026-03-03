//! Qdrant indexing and search.

use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    Condition, Filter, PointStruct, ScrollPointsBuilder, SearchPointsBuilder,
    UpsertPointsBuilder, Value,
};
use qdrant_client::Qdrant;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use walkdir::WalkDir;

use crate::embed::Embedder;
use crate::extract::{
    extract_jsonl, extract_jsonl_answers, extract_markdown, extract_summary, extract_zst,
    extract_zst_answers, IndexedChunk,
};
use crate::qdrant_hybrid::{build_named_vectors, ensure_hybrid_collection};

const QDRANT_URL: &str = "http://localhost:6334";
const COLLECTION_PROMPTS: &str = "claude-memory";
const COLLECTION_ANSWERS: &str = "claude-answers";

pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub path: String,
    pub score: f32,
}

/// Shared state for indexing operations.
struct IndexState {
    client: Qdrant,
    embedder: Embedder,
    prompt_next_id: AtomicU64,
    answer_next_id: AtomicU64,
    prompt_hashes: HashSet<String>,
    answer_hashes: HashSet<String>,
    batch_size: usize,
    delay_ms: u64,
}

/// Run the indexing process with streaming (low memory).
pub async fn run_index(
    archive_dir: &Path,
    projects_dir: &Path,
    kb_dir: &Path,
    batch_size: usize,
    fresh: bool,
    delay_ms: u64,
) -> Result<()> {
    let state = init_index_state(batch_size, fresh, delay_ms).await?;

    let jsonls = collect_jsonl_files(projects_dir);
    let archives = collect_archive_files(archive_dir);

    eprintln!("\n=== Indexing prompts (user messages, summaries, KB) ===");
    let prompts_indexed =
        index_all_prompts(&state, projects_dir, &jsonls, &archives, kb_dir).await?;

    eprintln!("\n=== Indexing answers (assistant responses) ===");
    let answers_indexed =
        index_all_answers(&state, projects_dir, &jsonls, &archives).await?;

    eprintln!(
        "\nDone! Prompts indexed: {}, Answers indexed: {}",
        prompts_indexed, answers_indexed
    );
    Ok(())
}

async fn init_index_state(
    batch_size: usize,
    fresh: bool,
    delay_ms: u64,
) -> Result<IndexState> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    ensure_hybrid_collection(&client, COLLECTION_PROMPTS).await?;
    ensure_hybrid_collection(&client, COLLECTION_ANSWERS).await?;
    let (prompt_hashes, answer_hashes) = load_hashes(&client, fresh).await?;
    eprintln!(
        "Found {} prompt chunks, {} answer chunks",
        prompt_hashes.len(),
        answer_hashes.len()
    );
    Ok(build_index_state(
        client,
        prompt_hashes,
        answer_hashes,
        batch_size,
        delay_ms,
    ))
}

async fn load_hashes(client: &Qdrant, fresh: bool) -> Result<(HashSet<String>, HashSet<String>)> {
    if fresh {
        eprintln!("Fresh index requested, ignoring existing data");
        return Ok((HashSet::new(), HashSet::new()));
    }
    eprintln!("Loading existing hashes for resume...");
    let prompt_hashes = get_existing_hashes(client, COLLECTION_PROMPTS).await?;
    let answer_hashes = get_existing_hashes(client, COLLECTION_ANSWERS).await?;
    Ok((prompt_hashes, answer_hashes))
}

fn build_index_state(
    client: Qdrant,
    prompt_hashes: HashSet<String>,
    answer_hashes: HashSet<String>,
    batch_size: usize,
    delay_ms: u64,
) -> IndexState {
    let prompt_next_id = AtomicU64::new(prompt_hashes.len() as u64);
    let answer_next_id = AtomicU64::new(answer_hashes.len() as u64);
    IndexState {
        client,
        embedder: Embedder::new(),
        prompt_next_id,
        answer_next_id,
        prompt_hashes,
        answer_hashes,
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
            e.path().extension().map(|ext| ext == "jsonl").unwrap_or(false)
                && e.path().file_name().map(|f| f != "sessions-index.json").unwrap_or(true)
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
    kb_dir: &Path,
) -> Result<usize> {
    let mut indexed = 0;

    indexed += index_summaries(state, projects_dir, indexed).await?;
    indexed += index_session_prompts(state, jsonls, projects_dir, indexed).await?;
    indexed += index_archive_prompts(state, archives, indexed).await?;
    indexed += index_kb_files(state, kb_dir, indexed).await?;

    Ok(indexed)
}

async fn index_all_answers(
    state: &IndexState,
    projects_dir: &Path,
    jsonls: &[walkdir::DirEntry],
    archives: &[std::fs::DirEntry],
) -> Result<usize> {
    let mut indexed = 0;

    indexed += index_session_answers(state, jsonls, projects_dir, indexed).await?;
    indexed += index_archive_answers(state, archives, indexed).await?;

    Ok(indexed)
}

async fn index_summaries(
    state: &IndexState,
    projects_dir: &Path,
    base_indexed: usize,
) -> Result<usize> {
    if !projects_dir.exists() {
        return Ok(0);
    }
    let summaries: Vec<_> = WalkDir::new(projects_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().file_name().map(|f| f == "summary.md").unwrap_or(false))
        .collect();

    let mut indexed = 0;
    eprintln!("Processing {} summaries...", summaries.len());
    for (i, entry) in summaries.iter().enumerate() {
        let path = entry.path();
        match extract_summary(path, projects_dir) {
            Ok(chunks) => {
                indexed += index_new_chunks(state, &chunks, &state.prompt_hashes, &state.prompt_next_id, COLLECTION_PROMPTS).await?;
            }
            Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
        }
        eprint!("\r  Summaries: {}/{} (indexed: {})", i + 1, summaries.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

async fn index_session_prompts(
    state: &IndexState,
    jsonls: &[walkdir::DirEntry],
    projects_dir: &Path,
    base_indexed: usize,
) -> Result<usize> {
    let mut indexed = 0;
    eprintln!("Processing {} active sessions (prompts)...", jsonls.len());
    for (i, entry) in jsonls.iter().enumerate() {
        let path = entry.path();
        match extract_jsonl(path, projects_dir) {
            Ok(chunks) => {
                indexed += index_new_chunks(state, &chunks, &state.prompt_hashes, &state.prompt_next_id, COLLECTION_PROMPTS).await?;
            }
            Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
        }
        eprint!("\r  Sessions: {}/{} (indexed: {})", i + 1, jsonls.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

async fn index_archive_prompts(
    state: &IndexState,
    archives: &[std::fs::DirEntry],
    base_indexed: usize,
) -> Result<usize> {
    let mut indexed = 0;
    eprintln!("Processing {} archives (prompts)...", archives.len());
    for (i, entry) in archives.iter().enumerate() {
        let path = entry.path();
        match extract_zst(&path) {
            Ok(chunks) => {
                indexed += index_new_chunks(state, &chunks, &state.prompt_hashes, &state.prompt_next_id, COLLECTION_PROMPTS).await?;
            }
            Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
        }
        eprint!("\r  Archives: {}/{} (indexed: {})", i + 1, archives.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

async fn index_kb_files(
    state: &IndexState,
    kb_dir: &Path,
    base_indexed: usize,
) -> Result<usize> {
    if !kb_dir.exists() {
        return Ok(0);
    }
    let markdowns: Vec<_> = WalkDir::new(kb_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "md").unwrap_or(false))
        .collect();

    let mut indexed = 0;
    eprintln!("Processing {} KB files...", markdowns.len());
    for (i, entry) in markdowns.iter().enumerate() {
        let path = entry.path();
        match extract_markdown(path, kb_dir) {
            Ok(chunks) => {
                indexed += index_new_chunks(state, &chunks, &state.prompt_hashes, &state.prompt_next_id, COLLECTION_PROMPTS).await?;
            }
            Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
        }
        eprint!("\r  KB: {}/{} (indexed: {})", i + 1, markdowns.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

async fn index_session_answers(
    state: &IndexState,
    jsonls: &[walkdir::DirEntry],
    projects_dir: &Path,
    base_indexed: usize,
) -> Result<usize> {
    let mut indexed = 0;
    eprintln!("Processing {} active sessions (answers)...", jsonls.len());
    for (i, entry) in jsonls.iter().enumerate() {
        let path = entry.path();
        match extract_jsonl_answers(path, projects_dir) {
            Ok(chunks) => {
                indexed += index_new_chunks(state, &chunks, &state.answer_hashes, &state.answer_next_id, COLLECTION_ANSWERS).await?;
            }
            Err(e) => tracing::warn!("Failed to extract answers {}: {}", path.display(), e),
        }
        eprint!("\r  Sessions: {}/{} (indexed: {})", i + 1, jsonls.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

async fn index_archive_answers(
    state: &IndexState,
    archives: &[std::fs::DirEntry],
    base_indexed: usize,
) -> Result<usize> {
    let mut indexed = 0;
    eprintln!("Processing {} archives (answers)...", archives.len());
    for (i, entry) in archives.iter().enumerate() {
        let path = entry.path();
        match extract_zst_answers(&path) {
            Ok(chunks) => {
                indexed += index_new_chunks(state, &chunks, &state.answer_hashes, &state.answer_next_id, COLLECTION_ANSWERS).await?;
            }
            Err(e) => tracing::warn!("Failed to extract answers {}: {}", path.display(), e),
        }
        eprint!("\r  Archives: {}/{} (indexed: {})", i + 1, archives.len(), base_indexed + indexed);
    }
    eprintln!();
    Ok(indexed)
}

/// Filter and index new chunks, returning the count indexed.
async fn index_new_chunks(
    state: &IndexState,
    chunks: &[IndexedChunk],
    existing: &HashSet<String>,
    next_id: &AtomicU64,
    collection: &str,
) -> Result<usize> {
    let new_chunks = filter_new(chunks, existing);
    if new_chunks.is_empty() {
        return Ok(0);
    }
    index_chunks(
        &state.client,
        &state.embedder,
        &new_chunks,
        state.batch_size,
        next_id,
        collection,
        state.delay_ms,
    )
    .await
}

fn filter_new(chunks: &[IndexedChunk], existing: &HashSet<String>) -> Vec<IndexedChunk> {
    let mut seen = HashSet::new();
    chunks
        .iter()
        .filter(|c| !existing.contains(&c.chunk.hash) && seen.insert(c.chunk.hash.clone()))
        .cloned()
        .collect()
}

/// Get all existing chunk hashes from Qdrant.
async fn get_existing_hashes(client: &Qdrant, collection: &str) -> Result<HashSet<String>> {
    let mut hashes = HashSet::new();
    let mut offset: Option<qdrant_client::qdrant::PointId> = None;

    loop {
        let mut scroll = ScrollPointsBuilder::new(collection)
            .limit(1000)
            .with_payload(true);

        if let Some(off) = offset {
            scroll = scroll.offset(off);
        }

        let result = client.scroll(scroll).await.context("failed to scroll")?;

        for point in &result.result {
            if let Some(hash) = point.payload.get("hash") {
                if let Some(qdrant_client::qdrant::value::Kind::StringValue(s)) = &hash.kind {
                    hashes.insert(s.clone());
                }
            }
        }

        offset = result.next_page_offset;
        if offset.is_none() {
            break;
        }
    }

    Ok(hashes)
}

/// Search prompts (user messages, KB).
pub async fn search_prompts(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, COLLECTION_PROMPTS).await
}

/// Search answers (assistant responses).
pub async fn search_answers(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, COLLECTION_ANSWERS).await
}

async fn search_collection(
    query: &str,
    limit: usize,
    source: Option<&str>,
    collection: &str,
) -> Result<Vec<SearchResult>> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, collection).await?;

    let embedder = Embedder::new();
    let query_vec = embedder.embed(query).await?;

    let mut search = SearchPointsBuilder::new(collection, query_vec, limit as u64)
        .vector_name("dense")
        .with_payload(true);

    if let Some(src) = source {
        search = search.filter(Filter::must([Condition::matches("source", src.to_string())]));
    }

    let results = client
        .search_points(search)
        .await
        .context("search failed")?;
    Ok(build_search_results(results.result))
}

fn build_search_results(points: Vec<qdrant_client::qdrant::ScoredPoint>) -> Vec<SearchResult> {
    points
        .into_iter()
        .map(|point| {
            let payload = point.payload;
            SearchResult {
                text: get_string(&payload, "text"),
                source: get_string(&payload, "source"),
                path: get_string(&payload, "path"),
                score: point.score,
            }
        })
        .collect()
}

fn get_string(payload: &HashMap<String, Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Index a single conversation file (both prompts and answers).
pub async fn index_file(path: &Path, batch_size: usize) -> Result<usize> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    ensure_hybrid_collection(&client, COLLECTION_PROMPTS).await?;
    ensure_hybrid_collection(&client, COLLECTION_ANSWERS).await?;

    let prompt_hashes = get_existing_hashes(&client, COLLECTION_PROMPTS).await?;
    let answer_hashes = get_existing_hashes(&client, COLLECTION_ANSWERS).await?;
    let prompt_next_id = AtomicU64::new(prompt_hashes.len() as u64);
    let answer_next_id = AtomicU64::new(answer_hashes.len() as u64);

    let embedder = Embedder::new();
    let mut total = 0;

    total += index_file_prompts(
        path, &client, &embedder, batch_size, &prompt_hashes, &prompt_next_id,
    ).await?;
    total += index_file_answers(
        path, &client, &embedder, batch_size, &answer_hashes, &answer_next_id,
    ).await?;

    Ok(total)
}

async fn index_file_prompts(
    path: &Path,
    client: &Qdrant,
    embedder: &Embedder,
    batch_size: usize,
    hashes: &HashSet<String>,
    next_id: &AtomicU64,
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
    index_chunks(client, embedder, &new_chunks, batch_size, next_id, COLLECTION_PROMPTS, 0).await
}

async fn index_file_answers(
    path: &Path,
    client: &Qdrant,
    embedder: &Embedder,
    batch_size: usize,
    hashes: &HashSet<String>,
    next_id: &AtomicU64,
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
    index_chunks(client, embedder, &new_chunks, batch_size, next_id, COLLECTION_ANSWERS, 0).await
}

/// Show collection statistics.
pub async fn show_stats() -> Result<()> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    for (label, name) in [("Prompts", COLLECTION_PROMPTS), ("Answers", COLLECTION_ANSWERS)] {
        if let Ok(info) = client.collection_info(name).await {
            let points = info.result.and_then(|r| r.points_count).unwrap_or(0);
            println!("{} ({}): {} points", label, name, points);
        } else {
            println!("{} ({}): not found", label, name);
        }
    }

    Ok(())
}

async fn index_chunks(
    client: &Qdrant,
    embedder: &Embedder,
    chunks: &[IndexedChunk],
    batch_size: usize,
    next_id: &AtomicU64,
    collection: &str,
    delay_ms: u64,
) -> Result<usize> {
    let mut indexed = 0;
    let delay = std::time::Duration::from_millis(delay_ms);

    for batch in chunks.chunks(batch_size) {
        if delay_ms > 0 && indexed > 0 {
            tokio::time::sleep(delay).await;
        }

        let texts: Vec<&str> = batch.iter().map(|c| c.chunk.text.as_str()).collect();

        let embeddings = match embedder.embed_batch(&texts).await {
            Ok(e) => e,
            Err(e) => {
                eprintln!("\nEmbedding error: {}", e);
                continue;
            }
        };

        let points: Vec<PointStruct> = build_points(batch, embeddings, next_id);

        client
            .upsert_points(UpsertPointsBuilder::new(collection, points))
            .await
            .context("failed to upsert points")?;

        indexed += batch.len();
    }

    Ok(indexed)
}

fn build_points(
    batch: &[IndexedChunk],
    embeddings: Vec<Vec<f32>>,
    next_id: &AtomicU64,
) -> Vec<PointStruct> {
    batch
        .iter()
        .zip(embeddings)
        .map(|(chunk, embedding)| build_single_point(chunk, embedding, next_id))
        .collect()
}

fn build_single_point(chunk: &IndexedChunk, embedding: Vec<f32>, next_id: &AtomicU64) -> PointStruct {
    let id = next_id.fetch_add(1, Ordering::SeqCst);
    let named = build_named_vectors(embedding, &chunk.chunk.text);
    let payload: HashMap<String, Value> = [
        ("text", chunk.chunk.text.clone().into()),
        ("source", chunk.source.clone().into()),
        ("path", chunk.path.clone().into()),
        ("session_id", chunk.session_id.clone().unwrap_or_default().into()),
        ("hash", chunk.chunk.hash.clone().into()),
    ]
    .into_iter()
    .map(|(k, v): (&str, Value)| (k.to_string(), v))
    .collect();
    PointStruct::new(id, named, payload)
}
