use std::collections::{HashMap, HashSet};

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    PointStruct, ScoredPoint, SearchPointsBuilder, UpsertPointsBuilder, Value,
};

use crate::chunk::Chunk;
use crate::extract::{HistoryType, IndexedChunk};
use crate::index::{
    IndexFileFormat, IndexFileSource, IndexSources, QDRANT_URL, build_search_results,
    collect_index_files, extract_single_file_history, filter_new, get_string, history_filter,
    history_hash,
};
use crate::qdrant_hybrid::{build_named_vectors, ensure_hybrid_collection};

fn make_chunk(hash: &str) -> IndexedChunk {
    IndexedChunk {
        chunk: Chunk {
            text: format!("text for {}", hash),
            hash: hash.to_string(),
        },
        history_type: HistoryType::Prompt,
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

fn make_scored_point(
    text: &str,
    source: &str,
    path: &str,
    session_id: &str,
    score: f32,
) -> ScoredPoint {
    let payload = [
        ("text".to_string(), str_value(text)),
        ("source".to_string(), str_value(source)),
        ("path".to_string(), str_value(path)),
        ("session_id".to_string(), str_value(session_id)),
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

#[test]
fn index_sources_discover_claude_codex_and_pi_sessions() {
    let root = std::env::temp_dir().join(format!("index-sources-{}", uuid::Uuid::new_v4()));
    let claude_projects = root.join("claude/projects/project");
    let claude_archive = root.join("claude/archive");
    let codex_sessions = root.join("codex/sessions/2026/07/18");
    let codex_archive = root.join("codex/archived_sessions");
    let pi_sessions = root.join("pi/sessions/project");
    for directory in [
        &claude_projects,
        &claude_archive,
        &codex_sessions,
        &codex_archive,
        &pi_sessions,
    ] {
        std::fs::create_dir_all(directory).unwrap();
    }
    std::fs::write(claude_projects.join("claude.jsonl"), "").unwrap();
    std::fs::write(claude_archive.join("claude.jsonl.zst"), "").unwrap();
    std::fs::write(codex_sessions.join("codex.jsonl"), "").unwrap();
    std::fs::write(codex_archive.join("codex-archive.jsonl"), "").unwrap();
    std::fs::write(pi_sessions.join("pi-active.jsonl"), "").unwrap();
    let archived_pi_dir = pi_sessions.join("archived-project");
    std::fs::create_dir_all(&archived_pi_dir).unwrap();
    std::fs::write(archived_pi_dir.join("pi-archived.jsonl"), "").unwrap();

    let sources = IndexSources {
        claude_projects_dir: &root.join("claude/projects"),
        claude_archive_dir: &claude_archive,
        codex_sessions_dir: &root.join("codex/sessions"),
        codex_archive_dir: &codex_archive,
        pi_sessions_dir: &root.join("pi/sessions"),
    };
    let files = collect_index_files(&sources);
    let observed: HashSet<_> = files
        .iter()
        .map(|file| (file.format, file.source))
        .collect();

    let pi_paths: HashSet<_> = files
        .iter()
        .filter(|file| file.path.starts_with(root.join("pi/sessions")))
        .map(|file| file.path.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    assert_eq!(files.len(), 6);
    assert_eq!(
        pi_paths,
        HashSet::from([
            "pi-active.jsonl".to_string(),
            "pi-archived.jsonl".to_string(),
        ])
    );
    assert_eq!(
        observed,
        HashSet::from([
            (IndexFileFormat::ClaudePi, IndexFileSource::Session),
            (IndexFileFormat::ClaudeZst, IndexFileSource::Archive),
            (IndexFileFormat::Codex, IndexFileSource::Session),
            (IndexFileFormat::Codex, IndexFileSource::Archive),
        ])
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn index_file_extracts_claude_codex_and_pi_prompt_answer_records() {
    let root = std::env::temp_dir().join(format!("index-file-formats-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let claude = root.join("claude.jsonl");
    let codex = root.join("codex.jsonl");
    let pi = root.join("pi.jsonl");
    std::fs::write(
        &claude,
        r#"{"type":"user","message":{"content":"Claude prompt"}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Claude answer"}]}}
"#,
    )
    .unwrap();
    std::fs::write(
        &codex,
        r#"{"timestamp":"2026-07-18T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Codex prompt"}]}}
{"timestamp":"2026-07-18T00:00:01Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Codex answer"}]}}
"#,
    )
    .unwrap();
    std::fs::write(
        &pi,
        r#"{"type":"session","version":3,"id":"pi-fixture","timestamp":"2026-07-18T00:00:00Z","cwd":"/tmp"}
{"type":"message","id":"user0001","parentId":null,"timestamp":"2026-07-18T00:00:01Z","message":{"role":"user","content":[{"type":"text","text":"Pi prompt"}]}}
{"type":"message","id":"asst0001","parentId":"user0001","timestamp":"2026-07-18T00:00:02Z","message":{"role":"assistant","content":[{"type":"thinking","thinking":"skip"},{"type":"text","text":"Pi answer"},{"type":"toolCall","id":"call-1","name":"read","arguments":{}}]}}
"#,
    )
    .unwrap();

    for (path, prompt, answer) in [
        (&claude, "User: Claude prompt", "Assistant: Claude answer"),
        (&codex, "User: Codex prompt", "Assistant: Codex answer"),
        (&pi, "User: Pi prompt", "Assistant: Pi answer"),
    ] {
        let prompts = extract_single_file_history(path, HistoryType::Prompt).unwrap();
        let answers = extract_single_file_history(path, HistoryType::Answer).unwrap();
        assert_eq!(prompts.len(), 1, "prompt count for {}", path.display());
        assert_eq!(answers.len(), 1, "answer count for {}", path.display());
        assert_eq!(prompts[0].history_type, HistoryType::Prompt);
        assert_eq!(answers[0].history_type, HistoryType::Answer);
        assert_eq!(prompts[0].chunk.text, prompt);
        assert_eq!(answers[0].chunk.text, answer);
    }
    let _ = std::fs::remove_dir_all(root);
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
    let existing: HashSet<String> = [history_hash(&chunks[0])].into();
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
    let existing: HashSet<String> = chunks.iter().map(history_hash).collect();
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

#[test]
fn identical_prompt_and_answer_text_have_distinct_history_hashes() {
    let prompt = make_chunk("same-content");
    let mut answer = prompt.clone();
    answer.history_type = HistoryType::Answer;

    assert_ne!(history_hash(&prompt), history_hash(&answer));
}

#[test]
fn identical_prompt_text_from_session_and_archive_has_distinct_history_hashes() {
    let session = make_chunk("same-content");
    let mut archive = session.clone();
    archive.source = "archive".to_string();

    assert_ne!(history_hash(&session), history_hash(&archive));
}

#[tokio::test]
async fn qdrant_history_filters_isolate_type_and_source() {
    let collection = format!("test-session-history-{}", uuid::Uuid::new_v4());
    let client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    ensure_hybrid_collection(&client, &collection)
        .await
        .unwrap();

    let points = vec![
        history_point(1, HistoryType::Prompt, "session"),
        history_point(2, HistoryType::Prompt, "archive"),
        history_point(3, HistoryType::Answer, "session"),
        history_point(4, HistoryType::Answer, "archive"),
    ];
    client
        .upsert_points(UpsertPointsBuilder::new(&collection, points))
        .await
        .unwrap();

    let prompt_sources =
        query_history_sources(&client, &collection, HistoryType::Prompt, &[]).await;
    let archived_answers =
        query_history_sources(&client, &collection, HistoryType::Answer, &["archive"]).await;
    client.delete_collection(&collection).await.unwrap();

    assert_eq!(prompt_sources, vec!["archive", "session"]);
    assert_eq!(archived_answers, vec!["archive"]);
}

fn history_point(id: u64, history_type: HistoryType, source: &str) -> PointStruct {
    let text = format!("{} {source}", history_type.as_str());
    let payload: HashMap<String, Value> = [
        ("text".to_string(), str_value(&text)),
        ("type".to_string(), str_value(history_type.as_str())),
        ("source".to_string(), str_value(source)),
        ("path".to_string(), str_value("fixture.jsonl")),
        ("session_id".to_string(), str_value("fixture")),
        (
            "hash".to_string(),
            str_value(&format!("{}:{id}", history_type.as_str())),
        ),
    ]
    .into();
    PointStruct::new(id, build_named_vectors(vec![1.0; 1024], &text), payload)
}

async fn query_history_sources(
    client: &Qdrant,
    collection: &str,
    history_type: HistoryType,
    sources: &[&str],
) -> Vec<String> {
    let search = SearchPointsBuilder::new(collection, vec![1.0; 1024], 10)
        .vector_name("dense")
        .with_payload(true)
        .filter(history_filter(history_type, sources));
    let mut matches: Vec<String> = client
        .search_points(search)
        .await
        .unwrap()
        .result
        .iter()
        .map(|point| get_string(&point.payload, "source"))
        .collect();
    matches.sort();
    matches
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
    let point = make_scored_point("some text", "session", "/some/path", "session-1", 0.95);
    let results = build_search_results(vec![point]);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "some text");
    assert_eq!(results[0].source, "session");
    assert_eq!(results[0].path, "/some/path");
    assert_eq!(results[0].session_id, "session-1");
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
    assert_eq!(results[0].session_id, "");
}

#[test]
fn build_search_results_empty_input() {
    assert!(build_search_results(vec![]).is_empty());
}
