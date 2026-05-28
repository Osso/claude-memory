//! Almanac-style notable fact ingestion for durable project memory.

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
use crate::extract::Turn;
use crate::memory_unit::DedupOutcome;
use crate::qdrant_hybrid::{build_named_vectors, ensure_hybrid_collection};

pub const COLLECTION_NOTABLE_FACTS: &str = "claude-notable-facts";

const DEDUP_THRESHOLD: f32 = 0.85;
const DEDUP_TOP_K: u64 = 5;

const NOTABLE_FACT_SYSTEM: &str = r#"You extract durable project memory from assistant coding sessions.

Almanac frame:
- The session is raw material, not the output.
- Preserve durable, reusable project understanding that would be costly, useful, or risky to reconstruct later.
- Good facts include entities, flows, contracts, data models, operations, decisions, risks, invariants, incidents, corrections, and research synthesis.
- Prefer updating an existing evolving idea over creating date-stamped logs.
- Do not summarize sessions, files, docs, market reads, or conversations. Distill their reusable project meaning.
- Reject file-by-file summaries, folder trees, task progress logs, generic API docs, unsupported guesses, obvious one-file facts, and facts that would not change future assistant behavior.

Scope:
- Set project to null for cross-project patterns.
- Set project to the provided current project slug for project-local facts.

Respond ONLY with JSON:
{"facts":[{"text":"durable fact, 1-3 sentences","project":null|"project-slug","topics":["topic"]}]}"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotableFact {
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub source: String,
    pub source_session: String,
    pub project: Option<String>,
    pub topics: Vec<String>,
    pub seen_in_sessions: Vec<String>,
}

#[derive(Debug, Default)]
pub struct NotableFactIngestSummary {
    pub facts: usize,
    pub inserted: usize,
    pub merged: usize,
}

#[derive(Debug, Deserialize)]
struct FactListJson {
    facts: Vec<FactJson>,
}

#[derive(Debug, Deserialize)]
struct FactJson {
    text: String,
    project: Option<String>,
    #[serde(default)]
    topics: Vec<String>,
}

pub fn notable_fact_system_prompt() -> &'static str {
    NOTABLE_FACT_SYSTEM
}

pub async fn ensure_notable_facts_collection(client: &Qdrant) -> Result<()> {
    ensure_hybrid_collection(client, COLLECTION_NOTABLE_FACTS).await
}

pub async fn ingest_session_notable_facts(
    client: &Qdrant,
    embedder: &Embedder,
    turns: &[Turn],
    source: &str,
    source_session: &str,
    project_slug: Option<&str>,
) -> Result<NotableFactIngestSummary> {
    let extracted = extract_session_notable_facts(turns, project_slug).await?;
    let mut summary = NotableFactIngestSummary {
        facts: extracted.len(),
        inserted: 0,
        merged: 0,
    };

    for fact in extracted {
        let fact = NotableFact {
            text: fact.text,
            created_at: Utc::now(),
            source: source.to_string(),
            source_session: source_session.to_string(),
            project: fact.project,
            topics: fact.topics,
            seen_in_sessions: vec![source_session.to_string()],
        };
        match upsert_with_dedup(client, embedder, fact).await? {
            DedupOutcome::Inserted(_) => summary.inserted += 1,
            DedupOutcome::Merged(_) => summary.merged += 1,
        }
    }

    Ok(summary)
}

async fn extract_session_notable_facts(
    turns: &[Turn],
    project_slug: Option<&str>,
) -> Result<Vec<FactJson>> {
    if turns.is_empty() {
        return Ok(Vec::new());
    }

    let user_msg = notable_fact_user_prompt(turns, project_slug);
    let raw = crate::llm::complete(NOTABLE_FACT_SYSTEM, &user_msg, 900, 180)
        .await
        .context("notable fact extractor LLM call failed")?;
    parse_notable_fact_json(&raw)
}

pub fn notable_fact_user_prompt(turns: &[Turn], project_slug: Option<&str>) -> String {
    let project = project_slug.unwrap_or("unknown");
    let transcript = turns
        .iter()
        .map(format_notable_turn)
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "Current project slug: {project}\n\nSession transcript:\n\n{transcript}\n\nExtract durable notable facts. Return an empty facts array when the session has no durable project memory."
    )
}

fn format_notable_turn(turn: &Turn) -> String {
    format!(
        "[{} turn {}]\n{}",
        turn_role(turn),
        turn.turn_index,
        turn.text.trim()
    )
}

fn turn_role(turn: &Turn) -> &'static str {
    match turn.role {
        crate::extract::Role::User => "user",
        crate::extract::Role::Assistant => "assistant",
    }
}

fn parse_notable_fact_json(raw: &str) -> Result<Vec<FactJson>> {
    let json = extract_json_object(raw);
    let parsed: FactListJson = serde_json::from_str(json)
        .with_context(|| format!("notable fact JSON parse failed | raw: {raw}"))?;
    Ok(parsed
        .facts
        .into_iter()
        .filter_map(normalize_fact)
        .collect())
}

fn normalize_fact(fact: FactJson) -> Option<FactJson> {
    let text = fact.text.trim().to_string();
    if text.is_empty() {
        return None;
    }

    Some(FactJson {
        text,
        project: normalize_optional_string(fact.project),
        topics: normalize_topics(fact.topics),
    })
}

fn normalize_topics(topics: Vec<String>) -> Vec<String> {
    topics
        .into_iter()
        .map(|topic| topic.trim().to_string())
        .filter(|topic| !topic.is_empty())
        .collect()
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn extract_json_object(raw: &str) -> &str {
    let text = raw.trim();
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .unwrap_or(text)
        .trim();
    let text = text.strip_suffix("```").unwrap_or(text).trim();
    let start = text.find('{').unwrap_or(0);
    let end = text.rfind('}').map(|index| index + 1).unwrap_or(text.len());
    &text[start..end]
}

pub async fn upsert_with_dedup(
    client: &Qdrant,
    embedder: &Embedder,
    fact: NotableFact,
) -> Result<DedupOutcome> {
    let embedding = embedder
        .embed(&fact.text)
        .await
        .context("failed to embed notable fact")?;

    let search = SearchPointsBuilder::new(COLLECTION_NOTABLE_FACTS, embedding.clone(), DEDUP_TOP_K)
        .vector_name("dense")
        .with_payload(true)
        .score_threshold(DEDUP_THRESHOLD);

    let results = client
        .search_points(search)
        .await
        .context("notable fact dedup search failed")?;

    if let Some(hit) = results.result.into_iter().next() {
        let point_id = hit.id.clone().context("hit has no point id")?;
        merge_seen_session(client, point_id.clone(), &hit.payload, &fact.source_session).await?;
        return Ok(DedupOutcome::Merged(point_id));
    }

    let new_uuid = Uuid::new_v4();
    let point = PointStruct::new(
        new_uuid.as_u128() as u64,
        build_named_vectors(embedding, &fact.text),
        notable_fact_payload(&fact),
    );
    client
        .upsert_points(UpsertPointsBuilder::new(
            COLLECTION_NOTABLE_FACTS,
            vec![point],
        ))
        .await
        .context("failed to upsert notable fact")?;

    Ok(DedupOutcome::Inserted(new_uuid))
}

async fn merge_seen_session(
    client: &Qdrant,
    point_id: PointId,
    payload: &HashMap<String, Value>,
    source_session: &str,
) -> Result<()> {
    let mut seen = string_list_field(payload, "seen_in_sessions");
    if !seen.iter().any(|session| session == source_session) {
        seen.push(source_session.to_string());
    }

    let mut new_payload = HashMap::new();
    new_payload.insert("seen_in_sessions".to_string(), string_list_value(seen));
    client
        .set_payload(
            SetPayloadPointsBuilder::new(COLLECTION_NOTABLE_FACTS, new_payload)
                .points_selector(vec![point_id]),
        )
        .await
        .context("failed to update notable fact seen_in_sessions")?;
    Ok(())
}

fn notable_fact_payload(fact: &NotableFact) -> HashMap<String, Value> {
    [
        ("text".to_string(), fact.text.clone().into()),
        (
            "created_at".to_string(),
            fact.created_at.to_rfc3339().into(),
        ),
        ("source".to_string(), fact.source.clone().into()),
        (
            "source_session".to_string(),
            fact.source_session.clone().into(),
        ),
        (
            "project".to_string(),
            fact.project.clone().unwrap_or_default().into(),
        ),
        ("topics".to_string(), string_list_value(fact.topics.clone())),
        (
            "seen_in_sessions".to_string(),
            string_list_value(fact.seen_in_sessions.clone()),
        ),
    ]
    .into()
}

fn string_list_value(values: Vec<String>) -> Value {
    Value {
        kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
            qdrant_client::qdrant::ListValue {
                values: values
                    .into_iter()
                    .map(|value| Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(value)),
                    })
                    .collect(),
            },
        )),
    }
}

#[cfg(test)]
fn string_field(payload: &HashMap<String, Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|value| match &value.kind {
            Some(qdrant_client::qdrant::value::Kind::StringValue(text)) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn string_list_field(payload: &HashMap<String, Value>, key: &str) -> Vec<String> {
    payload
        .get(key)
        .and_then(|value| match &value.kind {
            Some(qdrant_client::qdrant::value::Kind::ListValue(list)) => Some(
                list.values
                    .iter()
                    .filter_map(|value| match &value.kind {
                        Some(qdrant_client::qdrant::value::Kind::StringValue(text)) => {
                            Some(text.clone())
                        }
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn absorb_prompt_treats_sessions_as_raw_material() {
        let prompt = notable_fact_system_prompt();

        assert!(prompt.contains("raw material"));
        assert!(prompt.contains("durable project memory"));
        assert!(prompt.contains("Do not summarize sessions"));
        assert!(prompt.contains("existing evolving"));
    }

    #[test]
    fn parse_notable_fact_json_keeps_project_and_topics() {
        let raw = r#"{
            "facts": [
                {
                    "text": "Transcript PageIndex is CLI-only until query quality and corpus build costs are proven.",
                    "project": "claude-memory",
                    "topics": ["page-index", "decision"]
                }
            ]
        }"#;

        let facts = parse_notable_fact_json(raw).unwrap();

        assert_eq!(facts.len(), 1);
        assert_eq!(
            facts[0].text,
            "Transcript PageIndex is CLI-only until query quality and corpus build costs are proven."
        );
        assert_eq!(facts[0].project.as_deref(), Some("claude-memory"));
        assert_eq!(facts[0].topics, vec!["page-index", "decision"]);
    }

    #[test]
    fn notable_fact_collection_is_separate_from_memory_units() {
        assert_eq!(COLLECTION_NOTABLE_FACTS, "claude-notable-facts");
        assert_ne!(
            COLLECTION_NOTABLE_FACTS,
            crate::memory_unit::COLLECTION_MEMORY_UNITS
        );
    }

    #[test]
    fn notable_fact_payload_records_merge_metadata() {
        let fact = NotableFact {
            text: "Memory units remain operational preloads; notable facts are durable project memory.".to_string(),
            created_at: Utc::now(),
            source: "session".to_string(),
            source_session: "session-1.jsonl".to_string(),
            project: Some("claude-memory".to_string()),
            topics: vec!["memory".to_string()],
            seen_in_sessions: vec!["session-1.jsonl".to_string()],
        };

        let payload = notable_fact_payload(&fact);

        assert_eq!(string_field(&payload, "text"), fact.text);
        assert_eq!(string_field(&payload, "project"), "claude-memory");
        assert_eq!(string_list_field(&payload, "topics"), vec!["memory"]);
        assert_eq!(
            string_list_field(&payload, "seen_in_sessions"),
            vec!["session-1.jsonl"]
        );
    }
}
