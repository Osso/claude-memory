//! Memory unit schema, collection lifecycle, and dedup-at-write logic.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    DeletePointsBuilder, PointId, PointStruct, ScrollPointsBuilder, SearchPointsBuilder,
    SetPayloadPointsBuilder, UpsertPointsBuilder, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::embed::Embedder;
use crate::index::{QDRANT_URL, SearchResult};
use crate::qdrant_hybrid::ensure_hybrid_collection;

pub const COLLECTION_MEMORY_UNITS: &str = "claude-memory-units";

const DEDUP_THRESHOLD: f32 = 0.85;
const DEDUP_TOP_K: u64 = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub source_session: String,
    pub source_turn: u32,
    pub seen_in_sessions: Vec<String>,
}

#[derive(Debug)]
pub enum DedupOutcome {
    Inserted(Uuid),
    Merged(PointId),
}

/// Ensure the memory-units collection exists with the hybrid format.
pub async fn ensure_memory_units_collection(client: &Qdrant) -> Result<()> {
    ensure_hybrid_collection(client, COLLECTION_MEMORY_UNITS).await
}

/// Upsert a memory unit with dedup: if a near-duplicate exists (cosine > 0.92),
/// append the source session to seen_in_sessions and return Merged. Otherwise insert fresh.
pub async fn upsert_with_dedup(
    client: &Qdrant,
    embedder: &Embedder,
    unit: MemoryUnit,
) -> Result<DedupOutcome> {
    let embedding = embedder
        .embed(&unit.text)
        .await
        .context("failed to embed memory unit")?;

    let search = SearchPointsBuilder::new(COLLECTION_MEMORY_UNITS, embedding.clone(), DEDUP_TOP_K)
        .vector_name("dense")
        .with_payload(true)
        .score_threshold(DEDUP_THRESHOLD);

    let results = client
        .search_points(search)
        .await
        .context("dedup search failed")?;

    if let Some(hit) = results.result.into_iter().next() {
        let point_id = hit.id.clone().context("hit has no point id")?;

        // Fetch existing seen_in_sessions and append if not present
        let mut seen: Vec<String> = hit
            .payload
            .get("seen_in_sessions")
            .and_then(|v| match &v.kind {
                Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => Some(
                    list.values
                        .iter()
                        .filter_map(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => {
                                Some(s.clone())
                            }
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        if !seen.contains(&unit.source_session) {
            seen.push(unit.source_session.clone());
        }

        let updated_list = Value {
            kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                qdrant_client::qdrant::ListValue {
                    values: seen
                        .into_iter()
                        .map(|s| Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s)),
                        })
                        .collect(),
                },
            )),
        };

        let mut new_payload: HashMap<String, Value> = HashMap::new();
        new_payload.insert("seen_in_sessions".to_string(), updated_list);

        client
            .set_payload(
                SetPayloadPointsBuilder::new(COLLECTION_MEMORY_UNITS, new_payload)
                    .points_selector(vec![point_id.clone()]),
            )
            .await
            .context("failed to update seen_in_sessions")?;

        return Ok(DedupOutcome::Merged(point_id));
    }

    // No duplicate — insert fresh point
    let new_uuid = Uuid::new_v4();
    let id_u128 = new_uuid.as_u128();
    // Qdrant numeric IDs are u64; take the lower 64 bits (collisions negligible)
    let numeric_id = id_u128 as u64;

    let named = crate::qdrant_hybrid::build_named_vectors(embedding, &unit.text);

    let sessions_value = Value {
        kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
            qdrant_client::qdrant::ListValue {
                values: vec![Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                        unit.source_session.clone(),
                    )),
                }],
            },
        )),
    };

    let payload: HashMap<String, Value> = [
        ("text", unit.text.clone().into()),
        ("created_at", unit.created_at.to_rfc3339().into()),
        ("source_session", unit.source_session.clone().into()),
        (
            "source_turn",
            Value {
                kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(
                    unit.source_turn as i64,
                )),
            },
        ),
        ("seen_in_sessions", sessions_value),
    ]
    .into_iter()
    .map(|(k, v): (&str, Value)| (k.to_string(), v))
    .collect();

    let point = PointStruct::new(numeric_id, named, payload);
    client
        .upsert_points(UpsertPointsBuilder::new(
            COLLECTION_MEMORY_UNITS,
            vec![point],
        ))
        .await
        .context("failed to upsert memory unit")?;

    Ok(DedupOutcome::Inserted(new_uuid))
}

/// Search the memory-units collection by semantic similarity.
pub async fn search(query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_memory_units_collection(&client).await?;

    let embedder = Embedder::new();
    let query_vec = embedder.embed(query).await?;

    let search = SearchPointsBuilder::new(COLLECTION_MEMORY_UNITS, query_vec, limit as u64)
        .vector_name("dense")
        .with_payload(true);

    let results = client
        .search_points(search)
        .await
        .context("memory-units search failed")?;

    Ok(results
        .result
        .into_iter()
        .map(|p| {
            let text = string_field(&p.payload, "text");
            let session = string_field(&p.payload, "source_session");
            SearchResult {
                text,
                source: "memory-unit".to_string(),
                path: session,
                score: p.score,
            }
        })
        .collect())
}

/// A stored memory unit with its Qdrant point ID, for listing/deletion.
pub struct StoredMemory {
    pub id: u64,
    pub text: String,
    pub source_session: String,
    pub source_turn: u32,
    pub created_at: String,
    pub seen_count: usize,
}

/// List stored memory units. Either paged (offset) or filtered by substring/semantic query.
pub async fn list(
    limit: usize,
    offset: Option<u64>,
    substring: Option<&str>,
    query: Option<&str>,
) -> Result<Vec<StoredMemory>> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_memory_units_collection(&client).await?;

    if let Some(q) = query {
        let embedder = Embedder::new();
        let vec = embedder.embed(q).await?;
        let search = SearchPointsBuilder::new(COLLECTION_MEMORY_UNITS, vec, limit as u64)
            .vector_name("dense")
            .with_payload(true);
        let results = client.search_points(search).await?;
        return Ok(results
            .result
            .into_iter()
            .map(|p| point_to_stored_memory(p.id.as_ref(), &p.payload))
            .collect());
    }

    // Scroll all (optionally filter by substring client-side)
    let scroll_limit = if substring.is_some() {
        // pull a larger window so substring filtering still yields up to `limit`
        (limit * 10).max(200) as u32
    } else {
        limit as u32
    };
    let mut builder = ScrollPointsBuilder::new(COLLECTION_MEMORY_UNITS)
        .limit(scroll_limit)
        .with_payload(true);
    if let Some(o) = offset {
        builder = builder.offset(o);
    }

    let scrolled = client
        .scroll(builder)
        .await
        .context("memory-units scroll failed")?;

    let needle = substring.map(|s| s.to_lowercase());
    Ok(scrolled
        .result
        .into_iter()
        .map(|p| point_to_stored_memory(p.id.as_ref(), &p.payload))
        .filter(|m| match &needle {
            Some(n) => m.text.to_lowercase().contains(n),
            None => true,
        })
        .take(limit)
        .collect())
}

fn point_to_stored_memory(
    id: Option<&PointId>,
    payload: &HashMap<String, Value>,
) -> StoredMemory {
    let id_num = match id.and_then(|i| i.point_id_options.as_ref()) {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => *n,
        _ => 0,
    };
    let seen_count = payload
        .get("seen_in_sessions")
        .and_then(|v| match &v.kind {
            Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => Some(list.values.len()),
            _ => None,
        })
        .unwrap_or(0);
    StoredMemory {
        id: id_num,
        text: string_field(payload, "text"),
        source_session: string_field(payload, "source_session"),
        source_turn: payload
            .get("source_turn")
            .and_then(|v| match &v.kind {
                Some(qdrant_client::qdrant::value::Kind::IntegerValue(n)) => Some(*n as u32),
                _ => None,
            })
            .unwrap_or(0),
        created_at: string_field(payload, "created_at"),
        seen_count,
    }
}

/// Delete a memory unit by its numeric Qdrant point ID.
pub async fn delete(id: u64) -> Result<()> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    client
        .delete_points(
            DeletePointsBuilder::new(COLLECTION_MEMORY_UNITS).points(vec![PointId::from(id)]),
        )
        .await
        .with_context(|| format!("failed to delete memory unit {id}"))?;
    Ok(())
}

fn string_field(payload: &HashMap<String, Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| match &v.kind {
            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}
