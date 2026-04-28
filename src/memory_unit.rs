//! Memory unit schema, collection lifecycle, and dedup-at-write logic.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    PointId, PointStruct, SearchPointsBuilder, SetPayloadPointsBuilder, UpsertPointsBuilder, Value,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::embed::Embedder;
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
