//! Qdrant indexing and search.

use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, ScrollPointsBuilder,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder, Value,
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

const QDRANT_URL: &str = "http://localhost:6334";
const COLLECTION_PROMPTS: &str = "claude-memory";
const COLLECTION_ANSWERS: &str = "claude-answers";
const VECTOR_SIZE: u64 = 4096;

pub struct SearchResult {
    pub text: String,
    pub source: String,
    pub path: String,
    pub score: f32,
}

/// Run the indexing process with streaming (low memory).
pub async fn run_index(
    archive_dir: &Path,
    projects_dir: &Path,
    kb_dir: &Path,
    batch_size: usize,
    fresh: bool,
) -> Result<()> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    ensure_collection(&client, COLLECTION_PROMPTS).await?;
    ensure_collection(&client, COLLECTION_ANSWERS).await?;

    // Get existing hashes for resume capability
    let (prompt_hashes, answer_hashes) = if fresh {
        eprintln!("Fresh index requested, ignoring existing data");
        (HashSet::new(), HashSet::new())
    } else {
        eprintln!("Loading existing hashes for resume...");
        let p = get_existing_hashes(&client, COLLECTION_PROMPTS).await?;
        let a = get_existing_hashes(&client, COLLECTION_ANSWERS).await?;
        (p, a)
    };
    eprintln!(
        "Found {} prompt chunks, {} answer chunks",
        prompt_hashes.len(),
        answer_hashes.len()
    );

    let embedder = Embedder::new();
    let prompt_next_id = AtomicU64::new(prompt_hashes.len() as u64);
    let answer_next_id = AtomicU64::new(answer_hashes.len() as u64);
    let mut prompts_indexed = 0usize;
    let mut answers_indexed = 0usize;

    // Collect session files
    let jsonls: Vec<_> = if projects_dir.exists() {
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
    } else {
        vec![]
    };

    // Collect archive files
    let archives: Vec<_> = if archive_dir.exists() {
        std::fs::read_dir(archive_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .map(|f| f.to_string_lossy().ends_with(".jsonl.zst"))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        vec![]
    };

    // ===== INDEX PROMPTS COLLECTION =====
    eprintln!("\n=== Indexing prompts (user messages, summaries, KB) ===");

    // Process summaries → prompts
    if projects_dir.exists() {
        let summaries: Vec<_> = WalkDir::new(projects_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .map(|f| f == "summary.md")
                    .unwrap_or(false)
            })
            .collect();

        eprintln!("Processing {} summaries...", summaries.len());
        for (i, entry) in summaries.iter().enumerate() {
            let path = entry.path();
            match extract_summary(path, projects_dir) {
                Ok(chunks) => {
                    let new_chunks = filter_new(&chunks, &prompt_hashes);
                    if !new_chunks.is_empty() {
                        let count = index_chunks(
                            &client,
                            &embedder,
                            &new_chunks,
                            batch_size,
                            &prompt_next_id,
                            COLLECTION_PROMPTS,
                        )
                        .await?;
                        prompts_indexed += count;
                    }
                }
                Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
            }
            eprint!(
                "\r  Summaries: {}/{} (indexed: {})",
                i + 1,
                summaries.len(),
                prompts_indexed
            );
        }
        eprintln!();
    }

    // Process active sessions → prompts (user messages)
    eprintln!("Processing {} active sessions (prompts)...", jsonls.len());
    for (i, entry) in jsonls.iter().enumerate() {
        let path = entry.path();
        match extract_jsonl(path, projects_dir) {
            Ok(chunks) => {
                let new_chunks = filter_new(&chunks, &prompt_hashes);
                if !new_chunks.is_empty() {
                    let count = index_chunks(
                        &client,
                        &embedder,
                        &new_chunks,
                        batch_size,
                        &prompt_next_id,
                        COLLECTION_PROMPTS,
                    )
                    .await?;
                    prompts_indexed += count;
                }
            }
            Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
        }
        eprint!(
            "\r  Sessions: {}/{} (indexed: {})",
            i + 1,
            jsonls.len(),
            prompts_indexed
        );
    }
    eprintln!();

    // Process archives → prompts (user messages)
    eprintln!("Processing {} archives (prompts)...", archives.len());
    for (i, entry) in archives.iter().enumerate() {
        let path = entry.path();
        match extract_zst(&path) {
            Ok(chunks) => {
                let new_chunks = filter_new(&chunks, &prompt_hashes);
                if !new_chunks.is_empty() {
                    let count = index_chunks(
                        &client,
                        &embedder,
                        &new_chunks,
                        batch_size,
                        &prompt_next_id,
                        COLLECTION_PROMPTS,
                    )
                    .await?;
                    prompts_indexed += count;
                }
            }
            Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
        }
        eprint!(
            "\r  Archives: {}/{} (indexed: {})",
            i + 1,
            archives.len(),
            prompts_indexed
        );
    }
    eprintln!();

    // Process knowledge base → prompts
    if kb_dir.exists() {
        let markdowns: Vec<_> = WalkDir::new(kb_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "md")
                    .unwrap_or(false)
            })
            .collect();

        eprintln!("Processing {} KB files...", markdowns.len());
        for (i, entry) in markdowns.iter().enumerate() {
            let path = entry.path();
            match extract_markdown(path, kb_dir) {
                Ok(chunks) => {
                    let new_chunks = filter_new(&chunks, &prompt_hashes);
                    if !new_chunks.is_empty() {
                        let count = index_chunks(
                            &client,
                            &embedder,
                            &new_chunks,
                            batch_size,
                            &prompt_next_id,
                            COLLECTION_PROMPTS,
                        )
                        .await?;
                        prompts_indexed += count;
                    }
                }
                Err(e) => tracing::warn!("Failed to extract {}: {}", path.display(), e),
            }
            eprint!(
                "\r  KB: {}/{} (indexed: {})",
                i + 1,
                markdowns.len(),
                prompts_indexed
            );
        }
        eprintln!();
    }

    // ===== INDEX ANSWERS COLLECTION =====
    eprintln!("\n=== Indexing answers (assistant responses) ===");

    // Process active sessions → answers (assistant messages)
    eprintln!("Processing {} active sessions (answers)...", jsonls.len());
    for (i, entry) in jsonls.iter().enumerate() {
        let path = entry.path();
        match extract_jsonl_answers(path, projects_dir) {
            Ok(chunks) => {
                let new_chunks = filter_new(&chunks, &answer_hashes);
                if !new_chunks.is_empty() {
                    let count = index_chunks(
                        &client,
                        &embedder,
                        &new_chunks,
                        batch_size,
                        &answer_next_id,
                        COLLECTION_ANSWERS,
                    )
                    .await?;
                    answers_indexed += count;
                }
            }
            Err(e) => tracing::warn!("Failed to extract answers {}: {}", path.display(), e),
        }
        eprint!(
            "\r  Sessions: {}/{} (indexed: {})",
            i + 1,
            jsonls.len(),
            answers_indexed
        );
    }
    eprintln!();

    // Process archives → answers (assistant messages)
    eprintln!("Processing {} archives (answers)...", archives.len());
    for (i, entry) in archives.iter().enumerate() {
        let path = entry.path();
        match extract_zst_answers(&path) {
            Ok(chunks) => {
                let new_chunks = filter_new(&chunks, &answer_hashes);
                if !new_chunks.is_empty() {
                    let count = index_chunks(
                        &client,
                        &embedder,
                        &new_chunks,
                        batch_size,
                        &answer_next_id,
                        COLLECTION_ANSWERS,
                    )
                    .await?;
                    answers_indexed += count;
                }
            }
            Err(e) => tracing::warn!("Failed to extract answers {}: {}", path.display(), e),
        }
        eprint!(
            "\r  Archives: {}/{} (indexed: {})",
            i + 1,
            archives.len(),
            answers_indexed
        );
    }
    eprintln!();

    eprintln!(
        "\nDone! Prompts indexed: {}, Answers indexed: {}",
        prompts_indexed, answers_indexed
    );
    Ok(())
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

    let embedder = Embedder::new();
    let query_vec = embedder.embed(query).await?;

    let mut search =
        SearchPointsBuilder::new(collection, query_vec, limit as u64).with_payload(true);

    if let Some(src) = source {
        search = search.filter(Filter::must([Condition::matches("source", src.to_string())]));
    }

    let results = client
        .search_points(search)
        .await
        .context("search failed")?;

    Ok(results
        .result
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
        .collect())
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

    ensure_collection(&client, COLLECTION_PROMPTS).await?;
    ensure_collection(&client, COLLECTION_ANSWERS).await?;

    let prompt_hashes = get_existing_hashes(&client, COLLECTION_PROMPTS).await?;
    let answer_hashes = get_existing_hashes(&client, COLLECTION_ANSWERS).await?;
    let prompt_next_id = AtomicU64::new(prompt_hashes.len() as u64);
    let answer_next_id = AtomicU64::new(answer_hashes.len() as u64);

    let embedder = Embedder::new();
    let mut total = 0;

    // Extract and index prompts
    let prompt_chunks = if path.to_string_lossy().ends_with(".jsonl.zst") {
        extract_zst(path)?
    } else if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
        let base = path.parent().unwrap_or(path);
        extract_jsonl(path, base)?
    } else {
        anyhow::bail!("Unsupported file type: {}", path.display());
    };

    let new_prompts = filter_new(&prompt_chunks, &prompt_hashes);
    if !new_prompts.is_empty() {
        total += index_chunks(
            &client,
            &embedder,
            &new_prompts,
            batch_size,
            &prompt_next_id,
            COLLECTION_PROMPTS,
        )
        .await?;
    }

    // Extract and index answers
    let answer_chunks = if path.to_string_lossy().ends_with(".jsonl.zst") {
        extract_zst_answers(path)?
    } else if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
        let base = path.parent().unwrap_or(path);
        extract_jsonl_answers(path, base)?
    } else {
        vec![]
    };

    let new_answers = filter_new(&answer_chunks, &answer_hashes);
    if !new_answers.is_empty() {
        total += index_chunks(
            &client,
            &embedder,
            &new_answers,
            batch_size,
            &answer_next_id,
            COLLECTION_ANSWERS,
        )
        .await?;
    }

    Ok(total)
}

/// Show collection statistics.
pub async fn show_stats() -> Result<()> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;

    // Prompts collection
    if let Ok(info) = client.collection_info(COLLECTION_PROMPTS).await {
        let points = info.result.and_then(|r| r.points_count).unwrap_or(0);
        println!("Prompts ({}): {} points", COLLECTION_PROMPTS, points);
    } else {
        println!("Prompts ({}): not found", COLLECTION_PROMPTS);
    }

    // Answers collection
    if let Ok(info) = client.collection_info(COLLECTION_ANSWERS).await {
        let points = info.result.and_then(|r| r.points_count).unwrap_or(0);
        println!("Answers ({}): {} points", COLLECTION_ANSWERS, points);
    } else {
        println!("Answers ({}): not found", COLLECTION_ANSWERS);
    }

    Ok(())
}

async fn ensure_collection(client: &Qdrant, name: &str) -> Result<()> {
    let collections = client.list_collections().await?;

    if !collections.collections.iter().any(|c| c.name == name) {
        client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine)),
            )
            .await
            .context("failed to create collection")?;
        eprintln!("Created collection: {}", name);
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
) -> Result<usize> {
    let mut indexed = 0;

    for batch in chunks.chunks(batch_size) {
        let texts: Vec<&str> = batch.iter().map(|c| c.chunk.text.as_str()).collect();

        let embeddings = match embedder.embed_batch(&texts).await {
            Ok(e) => e,
            Err(e) => {
                eprintln!("\nEmbedding error: {}", e);
                continue;
            }
        };

        let points: Vec<PointStruct> = batch
            .iter()
            .zip(embeddings)
            .map(|(chunk, embedding)| {
                let id = next_id.fetch_add(1, Ordering::SeqCst);
                PointStruct::new(
                    id,
                    embedding,
                    [
                        ("text", chunk.chunk.text.clone().into()),
                        ("source", chunk.source.clone().into()),
                        ("path", chunk.path.clone().into()),
                        (
                            "session_id",
                            chunk.session_id.clone().unwrap_or_default().into(),
                        ),
                        ("hash", chunk.chunk.hash.clone().into()),
                    ],
                )
            })
            .collect();

        client
            .upsert_points(UpsertPointsBuilder::new(collection, points))
            .await
            .context("failed to upsert points")?;

        indexed += batch.len();
    }

    Ok(indexed)
}
