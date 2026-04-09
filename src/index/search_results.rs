use std::collections::HashMap;

use qdrant_client::qdrant::{ScoredPoint, Value, value::Kind::StringValue};

use super::SearchResult;

pub(super) fn build_search_results(points: Vec<ScoredPoint>) -> Vec<SearchResult> {
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

pub(super) fn get_string(payload: &HashMap<String, Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|value| value.kind.as_ref())
        .and_then(|kind| match kind {
            StringValue(text) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}
