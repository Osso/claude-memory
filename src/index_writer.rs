use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{PointStruct, ScrollPointsBuilder, UpsertPointsBuilder, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

use crate::embed::Embedder;
use crate::extract::IndexedChunk;
use crate::qdrant_hybrid::build_named_vectors;

pub(crate) fn filter_new(chunks: &[IndexedChunk], existing: &HashSet<String>) -> Vec<IndexedChunk> {
    let mut seen = HashSet::new();
    chunks
        .iter()
        .filter(|chunk| is_unseen_chunk(chunk, existing, &mut seen))
        .cloned()
        .collect()
}

fn is_unseen_chunk(
    chunk: &IndexedChunk,
    existing: &HashSet<String>,
    seen: &mut HashSet<String>,
) -> bool {
    let hash = history_hash(chunk);
    !existing.contains(&hash) && seen.insert(hash)
}

/// Get all existing chunk hashes from Qdrant.
pub(crate) async fn get_existing_hashes(
    client: &Qdrant,
    collection: &str,
) -> Result<HashSet<String>> {
    let mut hashes = HashSet::new();
    let mut offset: Option<qdrant_client::qdrant::PointId> = None;

    loop {
        let result = scroll_hash_page(client, collection, offset).await?;
        collect_hashes_from_page(&mut hashes, &result.result);

        offset = result.next_page_offset;
        if offset.is_none() {
            break;
        }
    }

    Ok(hashes)
}

async fn scroll_hash_page(
    client: &Qdrant,
    collection: &str,
    offset: Option<qdrant_client::qdrant::PointId>,
) -> Result<qdrant_client::qdrant::ScrollResponse> {
    let mut scroll = ScrollPointsBuilder::new(collection)
        .limit(1000)
        .with_payload(true);

    if let Some(offset) = offset {
        scroll = scroll.offset(offset);
    }

    client.scroll(scroll).await.context("failed to scroll")
}

fn collect_hashes_from_page(
    hashes: &mut HashSet<String>,
    points: &[qdrant_client::qdrant::RetrievedPoint],
) {
    for point in points {
        if let Some(hash) = point.payload.get("hash")
            && let Some(qdrant_client::qdrant::value::Kind::StringValue(value)) = &hash.kind
        {
            hashes.insert(value.clone());
        }
    }
}

pub(crate) async fn index_chunks(
    client: &Qdrant,
    embedder: &Embedder,
    chunks: &[IndexedChunk],
    batch_size: usize,
    collection: &str,
    delay_ms: u64,
) -> Result<usize> {
    let mut indexed = 0;
    let delay = std::time::Duration::from_millis(delay_ms);

    for batch in chunks.chunks(batch_size) {
        wait_before_next_batch(delay, delay_ms, indexed).await;

        let Some(points) = embed_batch_points(embedder, batch).await else {
            continue;
        };

        client
            .upsert_points(UpsertPointsBuilder::new(collection, points))
            .await
            .context("failed to upsert points")?;

        indexed += batch.len();
    }

    Ok(indexed)
}

async fn wait_before_next_batch(delay: std::time::Duration, delay_ms: u64, indexed: usize) {
    if delay_ms > 0 && indexed > 0 {
        tokio::time::sleep(delay).await;
    }
}

async fn embed_batch_points(
    embedder: &Embedder,
    batch: &[IndexedChunk],
) -> Option<Vec<PointStruct>> {
    let texts: Vec<&str> = batch
        .iter()
        .map(|chunk| chunk.chunk.text.as_str())
        .collect();
    let embeddings = match embedder.embed_batch(&texts).await {
        Ok(embeddings) => embeddings,
        Err(error) => {
            eprintln!("\nEmbedding error: {}", error);
            return None;
        }
    };

    Some(build_points(batch, embeddings))
}

fn build_points(batch: &[IndexedChunk], embeddings: Vec<Vec<f32>>) -> Vec<PointStruct> {
    batch
        .iter()
        .zip(embeddings)
        .map(|(chunk, embedding)| build_single_point(chunk, embedding))
        .collect()
}

fn build_single_point(chunk: &IndexedChunk, embedding: Vec<f32>) -> PointStruct {
    let id = point_id_for_chunk(chunk);
    let named = build_named_vectors(embedding, &chunk.chunk.text);
    PointStruct::new(id, named, point_payload(chunk))
}

fn point_id_for_chunk(chunk: &IndexedChunk) -> String {
    let mut hasher = Sha256::new();
    hasher.update(history_hash(chunk).as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    Uuid::from_bytes(bytes).to_string()
}

pub(crate) fn history_hash(chunk: &IndexedChunk) -> String {
    format!(
        "{}:{}:{}",
        chunk.history_type.as_str(),
        chunk.source,
        chunk.chunk.hash
    )
}

fn point_payload(chunk: &IndexedChunk) -> HashMap<String, Value> {
    [
        ("text", chunk.chunk.text.clone().into()),
        ("type", chunk.history_type.as_str().into()),
        ("source", chunk.source.clone().into()),
        ("path", chunk.path.clone().into()),
        (
            "session_id",
            chunk.session_id.clone().unwrap_or_default().into(),
        ),
        ("hash", history_hash(chunk).into()),
    ]
    .into_iter()
    .map(|(key, value): (&str, Value)| (key.to_string(), value))
    .collect()
}
