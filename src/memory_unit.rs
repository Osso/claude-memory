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

use crate::config;
use crate::embed::Embedder;
use crate::index::{QDRANT_URL, SearchResult};
use crate::qdrant_hybrid::ensure_hybrid_collection;

pub const COLLECTION_MEMORY_UNITS: &str = "claude-memory-units";
pub const GLOBAL_PROJECT_SCOPE: &str = "__global__";

const DEDUP_THRESHOLD: f32 = 0.85;
const DEDUP_TOP_K: u64 = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub source: String,
    pub source_session: String,
    pub source_turn: u32,
    pub category: Option<String>,
    pub project: Option<String>,
    pub seen_in_sessions: Vec<String>,
}

pub fn normalize_manual_project_scope(project: &str) -> Result<Option<String>> {
    let project = project.trim();
    if project.is_empty() {
        anyhow::bail!(
            "project is required; pass a project slug or {GLOBAL_PROJECT_SCOPE} for global memories"
        );
    }
    if project == GLOBAL_PROJECT_SCOPE {
        return Ok(None);
    }

    Ok(Some(project.to_string()))
}

pub fn manual_memory_write_guidance() -> &'static str {
    concat!(
        "Manual memory writes are disabled. Do not store this in Qdrant. ",
        "For project-specific durable context, create or update docs/local/memory.md ",
        "in the current project. For cross-project behavior and long-term agent rules, ",
        "update /home/osso/AgentConfig/rules."
    )
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
        ("source", unit.source.clone().into()),
        ("created_at", unit.created_at.to_rfc3339().into()),
        ("source_session", unit.source_session.clone().into()),
        ("category", unit.category.clone().unwrap_or_default().into()),
        ("project", unit.project.clone().unwrap_or_default().into()),
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
    if !config::search_enabled() {
        return Ok(Vec::new());
    }

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
        .map(|p| payload_to_search_result(&p.payload, p.score))
        .collect())
}

/// A stored memory unit with its Qdrant point ID, for listing/deletion.
pub struct StoredMemory {
    pub id: u64,
    pub text: String,
    pub source: String,
    pub source_session: String,
    pub source_turn: u32,
    pub category: String,
    pub project: String,
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
        if !config::search_enabled() {
            return Ok(Vec::new());
        }

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

fn point_to_stored_memory(id: Option<&PointId>, payload: &HashMap<String, Value>) -> StoredMemory {
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
        source: source_field(payload),
        source_session: string_field(payload, "source_session"),
        category: string_field(payload, "category"),
        project: string_field(payload, "project"),
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

pub async fn list_manual_entries(
    category: Option<&str>,
    project: Option<&str>,
) -> Result<Vec<(String, String, String)>> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_memory_units_collection(&client).await?;

    let mut entries = Vec::new();
    let mut offset: Option<PointId> = None;
    loop {
        let mut scroll = ScrollPointsBuilder::new(COLLECTION_MEMORY_UNITS)
            .limit(100)
            .with_payload(true);
        if let Some(point_id) = offset {
            scroll = scroll.offset(point_id);
        }

        let result = client
            .scroll(scroll)
            .await
            .context("memory-units scroll failed")?;
        for point in &result.result {
            let memory = point_to_stored_memory(point.id.as_ref(), &point.payload);
            if is_matching_manual_memory(&memory, category, project) {
                entries.push((memory.category, memory.project, memory.text));
            }
        }

        offset = result.next_page_offset;
        if offset.is_none() {
            break;
        }
    }
    Ok(entries)
}

fn is_matching_manual_memory(
    memory: &StoredMemory,
    category: Option<&str>,
    project: Option<&str>,
) -> bool {
    memory.source == "memory"
        && category.is_none_or(|expected| memory.category == expected)
        && project.is_none_or(|expected| memory.project == expected)
}

fn payload_to_search_result(payload: &HashMap<String, Value>, score: f32) -> SearchResult {
    SearchResult {
        text: string_field(payload, "text"),
        source: source_field(payload),
        path: string_field(payload, "source_session"),
        score,
    }
}

fn source_field(payload: &HashMap<String, Value>) -> String {
    let source = string_field(payload, "source");
    if !source.is_empty() {
        return source;
    }

    match string_field(payload, "source_session").as_str() {
        "manual" => "memory".to_string(),
        _ => "session".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use qdrant_client::qdrant::value::Kind;

    fn string_value(text: &str) -> Value {
        Value {
            kind: Some(Kind::StringValue(text.to_string())),
        }
    }

    #[test]
    fn memory_unit_has_existing_source_vocabulary_field() {
        let unit = MemoryUnit {
            text: "Manual preload".to_string(),
            created_at: Utc::now(),
            source: "memory".to_string(),
            source_session: "manual".to_string(),
            source_turn: 0,
            category: None,
            project: None,
            seen_in_sessions: vec!["manual".to_string()],
        };

        assert_eq!(unit.source, "memory");
    }

    #[test]
    fn stored_memory_reads_source_from_payload() {
        let payload = [
            ("text".to_string(), string_value("Analyzer preload")),
            ("source".to_string(), string_value("archive")),
            (
                "source_session".to_string(),
                string_value("session.jsonl.zst"),
            ),
            (
                "created_at".to_string(),
                string_value("2026-05-06T00:00:00Z"),
            ),
        ]
        .into();

        let memory = point_to_stored_memory(None, &payload);

        assert_eq!(memory.source, "archive");
    }

    #[test]
    fn search_result_uses_payload_source() {
        let payload = [
            ("text".to_string(), string_value("Analyzer preload")),
            ("source".to_string(), string_value("session")),
            ("source_session".to_string(), string_value("session.jsonl")),
        ]
        .into();

        let result = payload_to_search_result(&payload, 0.7);

        assert_eq!(result.source, "session");
        assert_eq!(result.path, "session.jsonl");
    }

    #[test]
    fn manual_memory_write_guidance_disables_qdrant_path() {
        let guidance = manual_memory_write_guidance();

        assert!(guidance.contains("Manual memory writes are disabled"));
        assert!(guidance.contains("Do not store this in Qdrant"));
        assert!(guidance.contains("docs/local/memory.md"));
    }

    #[test]
    fn global_project_scope_normalizes_to_unscoped_memory() {
        let project = normalize_manual_project_scope("__global__").unwrap();

        assert_eq!(project, None);
    }

    #[test]
    fn blank_project_scope_is_rejected() {
        let error = normalize_manual_project_scope("   ").unwrap_err();

        assert!(error.to_string().contains("project is required"));
    }

    #[test]
    fn manual_memory_filter_matches_category_and_project() {
        let memory = StoredMemory {
            id: 0,
            text: "Manual preload".to_string(),
            source: "memory".to_string(),
            source_session: "manual".to_string(),
            source_turn: 0,
            category: "preference".to_string(),
            project: "claude-memory".to_string(),
            created_at: "2026-05-06T00:00:00Z".to_string(),
            seen_count: 1,
        };

        assert!(is_matching_manual_memory(
            &memory,
            Some("preference"),
            Some("claude-memory")
        ));
        assert!(!is_matching_manual_memory(
            &memory,
            Some("decision"),
            Some("claude-memory")
        ));
    }
}
