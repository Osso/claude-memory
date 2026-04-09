use std::collections::{HashMap, HashSet};

use qdrant_client::qdrant::{ScoredPoint, Value};

use crate::chunk::Chunk;
use crate::extract::IndexedChunk;
use crate::index::{build_search_results, filter_new, get_string};

fn make_chunk(hash: &str) -> IndexedChunk {
    IndexedChunk {
        chunk: Chunk {
            text: format!("text for {}", hash),
            hash: hash.to_string(),
        },
        source: "session".to_string(),
        path: "/some/path".to_string(),
        session_id: None,
    }
}

fn str_value(s: &str) -> Value {
    Value {
        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
            s.to_string(),
        )),
    }
}

fn make_scored_point(text: &str, source: &str, path: &str, score: f32) -> ScoredPoint {
    let payload = [
        ("text".to_string(), str_value(text)),
        ("source".to_string(), str_value(source)),
        ("path".to_string(), str_value(path)),
    ]
    .into();
    ScoredPoint {
        id: None,
        payload,
        score,
        version: 0,
        vectors: None,
        shard_key: None,
        order_value: None,
    }
}

// --- filter_new ---

#[test]
fn filter_new_keeps_new_items() {
    let chunks = vec![make_chunk("aaa"), make_chunk("bbb")];
    let result = filter_new(&chunks, &HashSet::new());
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].chunk.hash, "aaa");
    assert_eq!(result[1].chunk.hash, "bbb");
}

#[test]
fn filter_new_removes_existing_hashes() {
    let chunks = vec![make_chunk("aaa"), make_chunk("bbb")];
    let existing: HashSet<String> = ["aaa".to_string()].into();
    let result = filter_new(&chunks, &existing);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].chunk.hash, "bbb");
}

#[test]
fn filter_new_empty_input_returns_empty() {
    let existing: HashSet<String> = ["aaa".to_string()].into();
    assert!(filter_new(&[], &existing).is_empty());
}

#[test]
fn filter_new_all_duplicates_returns_empty() {
    let chunks = vec![make_chunk("aaa"), make_chunk("bbb")];
    let existing: HashSet<String> = ["aaa".to_string(), "bbb".to_string()].into();
    assert!(filter_new(&chunks, &existing).is_empty());
}

#[test]
fn filter_new_deduplicates_within_input() {
    let chunks = vec![make_chunk("aaa"), make_chunk("aaa"), make_chunk("bbb")];
    let result = filter_new(&chunks, &HashSet::new());
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].chunk.hash, "aaa");
    assert_eq!(result[1].chunk.hash, "bbb");
}

// --- get_string ---

#[test]
fn get_string_returns_value_for_known_key() {
    let payload: HashMap<String, Value> = [("text".to_string(), str_value("hello"))].into();
    assert_eq!(get_string(&payload, "text"), "hello");
}

#[test]
fn get_string_returns_empty_for_missing_key() {
    assert_eq!(get_string(&HashMap::new(), "text"), "");
}

#[test]
fn get_string_returns_empty_for_non_string_value() {
    let payload: HashMap<String, Value> = [(
        "score".to_string(),
        Value {
            kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(1.5)),
        },
    )]
    .into();
    assert_eq!(get_string(&payload, "score"), "");
}

#[test]
fn get_string_returns_empty_for_null_kind() {
    let payload: HashMap<String, Value> = [("empty".to_string(), Value { kind: None })].into();
    assert_eq!(get_string(&payload, "empty"), "");
}

// --- build_search_results ---

#[test]
fn build_search_results_extracts_fields() {
    let point = make_scored_point("some text", "session", "/some/path", 0.95);
    let results = build_search_results(vec![point]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "some text");
    assert_eq!(results[0].source, "session");
    assert_eq!(results[0].path, "/some/path");
    assert!((results[0].score - 0.95).abs() < 1e-6);
}

#[test]
fn build_search_results_empty_payload_graceful() {
    let point = ScoredPoint {
        id: None,
        payload: HashMap::new(),
        score: 0.5,
        version: 0,
        vectors: None,
        shard_key: None,
        order_value: None,
    };
    let results = build_search_results(vec![point]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "");
    assert_eq!(results[0].source, "");
    assert_eq!(results[0].path, "");
}

#[test]
fn build_search_results_empty_input() {
    assert!(build_search_results(vec![]).is_empty());
}
