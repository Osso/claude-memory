use std::collections::HashMap;

use qdrant_client::qdrant::{ScoredPoint, Value, value::Kind::StringValue};

use super::SearchResult;

pub(crate) fn build_search_results(points: Vec<ScoredPoint>) -> Vec<SearchResult> {
    points
        .into_iter()
        .map(|point| {
            let payload = point.payload;
            SearchResult {
                record_type: get_string(&payload, "type"),
                text: get_string(&payload, "text"),
                source: get_string(&payload, "source"),
                path: get_string(&payload, "path"),
                session_id: get_string(&payload, "session_id"),
                score: point.score,
            }
        })
        .collect()
}

pub(crate) fn get_string(payload: &HashMap<String, Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|value| value.kind.as_ref())
        .and_then(|kind| match kind {
            StringValue(text) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}
