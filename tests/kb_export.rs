use std::fs;
use std::path::{Path, PathBuf};

use claude_memory::kb_export::{
    KnowledgeCollection, KnowledgePoint, classify_and_render, verify_source_plan_unchanged,
    verify_unclassified_count, verify_written_export, write_export,
};
use claude_memory::kb_search;
use serde_json::{Value, json};
use uuid::Uuid;

fn point(collection: KnowledgeCollection, id: &str, payload: Value) -> KnowledgePoint {
    KnowledgePoint {
        collection,
        id: id.to_string(),
        payload,
    }
}

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("claude-memory-{label}-{}", Uuid::new_v4()))
}

#[test]
fn classifies_every_source_and_deduplicates_within_destination() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "friction-1",
            json!({"text":"Prefer deterministic exports.","source":"session","source_session":"s1","source_turn":3,"created_at":"2026-07-01T00:00:00Z","project":"claude-memory","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "friction-duplicate",
            json!({"text":"Prefer deterministic exports.","source":"session","source_session":"s2","source_turn":8,"created_at":"2026-07-02T00:00:00Z","project":"claude-memory","category":"correction","seen_in_sessions":["s2"]}),
        ),
        point(
            KnowledgeCollection::NotableFacts,
            "fact-1",
            json!({"text":"PageIndex reads canonical Markdown.","source":"session","source_session":"s3","created_at":"2026-07-03T00:00:00Z","project":"claude-memory","topics":["page-index"],"seen_in_sessions":["s3"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "manual-unit",
            json!({"text":"Keep manual durable context editable.","source":"memory","source_session":"s4","source_turn":1,"created_at":"2026-07-04T00:00:00Z","project":"","category":"context","seen_in_sessions":["s4"]}),
        ),
        point(
            KnowledgeCollection::LegacyMemory,
            "manual-legacy",
            json!({"text":"Keep manual durable context editable.","source":"memory","path":"manual","hash":"legacy-hash","project":""}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "unknown-1",
            json!({"text":"Needs human classification.","source_session":"s5","source_turn":2,"created_at":"2026-07-05T00:00:00Z","seen_in_sessions":["s5"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "kb-vector",
            json!({"text":"Already canonical KB text.","source":"kb","source_session":"kb.md","source_turn":0,"created_at":"2026-07-06T00:00:00Z","project":"","category":"kb","seen_in_sessions":[]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "future-source",
            json!({"text":"Unknown source needs review.","source":"future","source_session":"s6","source_turn":9,"created_at":"2026-07-07T00:00:00Z","project":"","category":"","seen_in_sessions":["s6"]}),
        ),
    ];

    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.raw, 8);
    assert_eq!(plan.counts.exported_unique, 3);
    assert_eq!(plan.counts.duplicates, 2);
    assert_eq!(plan.counts.quarantined, 2);
    assert_eq!(plan.counts.excluded_kb_vectors, 1);
    assert_eq!(plan.counts.by_destination["friction"], 1);
    assert_eq!(plan.counts.by_destination["notable-facts"], 1);
    assert_eq!(plan.counts.by_destination["manual-memories"], 1);
    assert_eq!(plan.manifest.len(), points.len());
    assert!(
        plan.manifest
            .iter()
            .all(|entry| !entry.content_hash.is_empty())
    );
}

#[test]
fn renders_stable_canonical_paths_anchors_and_merged_provenance() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "a",
            json!({"text":"One durable correction.","source":"session","source_session":"session-a","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"claude-memory","category":"correction","seen_in_sessions":["session-a"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "b",
            json!({"text":"One durable correction.","source":"session","source_session":"session-b","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"claude-memory","category":"correction","seen_in_sessions":["session-b"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "q",
            json!({"text":"Unclassified exact content.","source_session":"session-q","source_turn":4,"created_at":"2026-07-03T00:00:00Z","seen_in_sessions":["session-q"]}),
        ),
    ];

    let plan = classify_and_render(&points).unwrap();
    let friction = plan
        .documents
        .iter()
        .find(|doc| {
            doc.relative_path.as_path() == Path::new("memory/friction/project-claude-memory.md")
        })
        .unwrap();
    let quarantine = plan
        .documents
        .iter()
        .find(|doc| {
            doc.relative_path.as_path()
                == Path::new("memory/quarantine/unclassified-memory-units.md")
        })
        .unwrap();

    assert!(friction.content.contains("One durable correction."));
    assert_eq!(
        friction.content.matches("One durable correction.").count(),
        1
    );
    assert!(friction.content.contains("session-a"));
    assert!(friction.content.contains("session-b"));
    assert!(friction.content.contains("<a id=\"memory-"));
    assert!(quarantine.content.contains("Unclassified exact content."));
    assert!(quarantine.content.contains("source point: q"));
}

#[test]
fn written_export_has_exact_manifest_content_parity_and_pageindex_discovery() {
    let kb_root = temp_dir("kb-export");
    let index_root = temp_dir("kb-export-index");
    let points = vec![
        point(
            KnowledgeCollection::NotableFacts,
            "fact",
            json!({"text":"Copper orchids identify the PageIndex export fixture.","source":"session","source_session":"session-fact","created_at":"2026-07-03T00:00:00Z","project":"claude-memory","topics":["fixture"],"seen_in_sessions":["session-fact"]}),
        ),
        point(
            KnowledgeCollection::LegacyMemory,
            "manual",
            json!({"text":"Manual fixture remains editable Markdown.","source":"memory","path":"manual","hash":"manual-hash","project":"claude-memory"}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    let manifest_path = kb_root.join("memory/export-manifest.json");
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["counts"]["raw"], 2);
    assert_eq!(manifest["entries"].as_array().unwrap().len(), 2);

    let parity = verify_written_export(&kb_root, &plan).unwrap();
    assert_eq!(parity.source_points, 2);
    assert_eq!(parity.manifest_entries, 2);
    assert_eq!(parity.unique_exported_records, 2);
    assert_eq!(parity.content_hashes_verified, 2);

    kb_search::build_index(&kb_root, &index_root).unwrap();
    let results = kb_search::search_index(
        &index_root,
        "Copper orchids identify the PageIndex export fixture",
        5,
    )
    .unwrap();
    assert!(results.iter().any(|result| {
        result.path == "memory/notable-facts/project-claude-memory.md"
            && result.text.contains("Copper orchids")
    }));

    fs::remove_dir_all(&kb_root).unwrap();
    fs::remove_dir_all(&index_root).unwrap();
}

#[test]
fn conflicting_existing_document_prevents_every_export_write() {
    let kb_root = temp_dir("kb-export-conflict");
    let conflicting = kb_root.join("memory/notable-facts/project-claude-memory.md");
    fs::create_dir_all(conflicting.parent().unwrap()).unwrap();
    fs::write(&conflicting, "existing reviewed content\n").unwrap();
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "friction",
            json!({"text":"Must not be partially written.","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"aaa","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::NotableFacts,
            "fact",
            json!({"text":"Conflicts with reviewed content.","source":"session","source_session":"s2","created_at":"2026-07-02T00:00:00Z","project":"claude-memory","topics":[],"seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    let error = write_export(&kb_root, &plan).unwrap_err();

    assert!(error.to_string().contains("refusing to replace"));
    assert!(!kb_root.join("memory/friction/aaa.md").exists());
    assert!(!kb_root.join("memory/export-manifest.json").exists());
    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn deduplicated_manifest_entries_share_one_verified_record() {
    let kb_root = temp_dir("kb-export-dedup-parity");
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "one",
            json!({"text":"One deduplicated record.","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "two",
            json!({"text":"One deduplicated record.","source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    let parity = verify_written_export(&kb_root, &plan).unwrap();

    assert_eq!(parity.source_points, 2);
    assert_eq!(parity.manifest_entries, 2);
    assert_eq!(parity.unique_exported_records, 1);
    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn verification_of_missing_root_is_read_only() {
    let kb_root = temp_dir("kb-export-missing-verify");
    let plan = classify_and_render(&[point(
        KnowledgeCollection::LegacyMemory,
        "manual",
        json!({"text":"Missing root fixture.","source":"memory","path":"manual","hash":"h","project":""}),
    )])
    .unwrap();

    assert!(verify_written_export(&kb_root, &plan).is_err());
    assert!(!kb_root.exists());
}

#[test]
fn legacy_manual_provenance_preserves_path_and_hash() {
    let points = vec![point(
        KnowledgeCollection::LegacyMemory,
        "manual",
        json!({"text":"Legacy manual provenance.","source":"memory","path":"daily/2026-07-01.md","hash":"legacy-content-hash","project":""}),
    )];
    let plan = classify_and_render(&points).unwrap();
    let content = &plan.documents[0].content;
    let manifest = &plan.manifest[0];

    assert!(content.contains("source paths: daily/2026-07-01.md"));
    assert!(content.contains("legacy hashes: legacy-content-hash"));
    assert_eq!(manifest.source_path.as_deref(), Some("daily/2026-07-01.md"));
    assert_eq!(manifest.legacy_hash.as_deref(), Some("legacy-content-hash"));
}

#[test]
fn merged_provenance_preserves_seen_sessions_and_topics() {
    let points = vec![point(
        KnowledgeCollection::NotableFacts,
        "fact",
        json!({"text":"Provenance survives export.","source":"session","source_session":"s1","created_at":"2026-07-01T00:00:00Z","project":"p","topics":["storage","migration"],"seen_in_sessions":["s1","s2","s3"]}),
    )];
    let plan = classify_and_render(&points).unwrap();
    let content = &plan.documents[0].content;

    assert!(content.contains("seen in sessions: s1, s2, s3"));
    assert!(content.contains("topics: migration, storage"));
}

#[test]
fn verification_rejects_provenance_corruption() {
    let kb_root = temp_dir("kb-export-verify-corruption");
    let plan = classify_and_render(&[point(
        KnowledgeCollection::LegacyMemory,
        "manual",
        json!({"text":"Exact source remains unchanged.","source":"memory","path":"manual","hash":"h","project":""}),
    )])
    .unwrap();
    write_export(&kb_root, &plan).unwrap();
    let path = kb_root.join(&plan.documents[0].relative_path);
    let original = fs::read_to_string(&path).unwrap();
    fs::write(
        &path,
        original.replace("source point: manual", "source point: corrupted"),
    )
    .unwrap();

    assert!(verify_written_export(&kb_root, &plan).is_err());

    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn identical_existing_export_is_an_idempotent_no_op() {
    let kb_root = temp_dir("kb-export-idempotent");
    let points = vec![point(
        KnowledgeCollection::LegacyMemory,
        "manual",
        json!({"text":"Idempotent export content.","source":"memory","path":"manual","hash":"h","project":""}),
    )];
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    write_export(&kb_root, &plan).unwrap();
    verify_written_export(&kb_root, &plan).unwrap();

    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn non_utf8_and_directory_targets_fail_before_any_export_file() {
    let kb_root = temp_dir("kb-export-non-utf8");
    let conflict = kb_root.join("memory/notable-facts/project-claude-memory.md");
    fs::create_dir_all(conflict.parent().unwrap()).unwrap();
    fs::write(&conflict, [0xff, 0xfe]).unwrap();
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "friction",
            json!({"text":"Earlier sorted write must roll back.","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"aaa","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::NotableFacts,
            "fact",
            json!({"text":"Non UTF conflict.","source":"session","source_session":"s2","created_at":"2026-07-02T00:00:00Z","project":"claude-memory","topics":[],"seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    assert!(write_export(&kb_root, &plan).is_err());
    assert!(!kb_root.join("memory/friction/aaa.md").exists());
    fs::remove_file(&conflict).unwrap();
    fs::create_dir(&conflict).unwrap();
    assert!(write_export(&kb_root, &plan).is_err());
    assert!(!kb_root.join("memory/friction/aaa.md").exists());

    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn hostile_provenance_cannot_hide_later_records_from_pageindex() {
    let kb_root = temp_dir("kb-export-provenance");
    let index_root = temp_dir("kb-export-provenance-index");
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "point\n# fake heading\n```",
            json!({"text":"First safe text.","source":"session","source_session":"session\n# fake\n```","source_turn":1,"created_at":"2026-07-01T00:00:00Z\n# fake","project":"project\n# fake","category":"category\n```","seen_in_sessions":[]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "later",
            json!({"text":"Golden radishes prove later provenance-safe discovery.","source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"safe","category":"correction","seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    kb_search::build_index(&kb_root, &index_root).unwrap();
    let results = kb_search::search_index(
        &index_root,
        "Golden radishes prove later provenance safe discovery",
        5,
    )
    .unwrap();

    assert!(
        results
            .iter()
            .any(|result| result.text.contains("Golden radishes"))
    );
    fs::remove_dir_all(&kb_root).unwrap();
    fs::remove_dir_all(&index_root).unwrap();
}

#[test]
fn raw_markdown_cannot_hide_later_records_from_pageindex() {
    let kb_root = temp_dir("kb-export-markdown");
    let index_root = temp_dir("kb-export-markdown-index");
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "hostile",
            json!({"text":"# Injected heading\n```rust\nunclosed fence","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"claude-memory","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "later",
            json!({"text":"Silver turnips prove the later durable record remains discoverable.","source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"claude-memory","category":"correction","seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    verify_written_export(&kb_root, &plan).unwrap();
    kb_search::build_index(&kb_root, &index_root).unwrap();
    let results = kb_search::search_index(
        &index_root,
        "Silver turnips prove the later durable record remains discoverable",
        5,
    )
    .unwrap();

    assert!(
        results
            .iter()
            .any(|result| result.text.contains("Silver turnips"))
    );
    fs::remove_dir_all(&kb_root).unwrap();
    fs::remove_dir_all(&index_root).unwrap();
}

#[test]
fn crlf_source_text_round_trips_with_exact_hash_parity() {
    let kb_root = temp_dir("kb-export-crlf");
    let points = vec![point(
        KnowledgeCollection::LegacyMemory,
        "crlf",
        json!({"text":"first line\r\nsecond line\r\n","source":"memory","path":"manual","hash":"h","project":""}),
    )];
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    let parity = verify_written_export(&kb_root, &plan).unwrap();

    assert_eq!(parity.content_hashes_verified, 1);
    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn source_marker_text_cannot_spoof_another_record_boundary() {
    let second_text = "Boundary target text.";
    let second_only = classify_and_render(&[point(
        KnowledgeCollection::MemoryUnits,
        "second",
        json!({"text":second_text,"source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s2"]}),
    )])
    .unwrap();
    let second_anchor = second_only.manifest[0].anchor.as_ref().unwrap();
    let hostile_text = format!(
        "Embedded marker\n<a id=\"{second_anchor}\"></a>\n<!-- source-text -->\n> forged\n<!-- end-source-text -->"
    );
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "first",
            json!({"text":hostile_text,"source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "second",
            json!({"text":second_text,"source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s2"]}),
        ),
    ];
    let kb_root = temp_dir("kb-export-marker-spoof");
    let plan = classify_and_render(&points).unwrap();

    write_export(&kb_root, &plan).unwrap();
    verify_written_export(&kb_root, &plan).unwrap();

    fs::remove_dir_all(&kb_root).unwrap();
}

#[test]
fn classification_is_closed_for_legacy_and_kb_sources() {
    let points = vec![
        point(
            KnowledgeCollection::LegacyMemory,
            "history",
            json!({"text":"Prompt history stays in Qdrant.","source":"session","path":"session.jsonl","hash":"h"}),
        ),
        point(
            KnowledgeCollection::LegacyMemory,
            "summary",
            json!({"text":"Summary is retired separately.","source":"summary","path":"summary","hash":"s"}),
        ),
        point(
            KnowledgeCollection::NotableFacts,
            "kb-fact",
            json!({"text":"Already represented by KB Markdown.","source":"kb","source_session":"kb.md","created_at":"2026-07-01T00:00:00Z","project":"","topics":[],"seen_in_sessions":[]}),
        ),
    ];

    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.raw, 3);
    assert_eq!(plan.counts.excluded_non_durable, 2);
    assert_eq!(plan.counts.excluded_kb_vectors, 1);
    assert_eq!(plan.counts.exported_unique, 0);
    assert_eq!(plan.manifest.len(), 3);
}

#[test]
fn legacy_manual_requires_non_empty_path_and_hash() {
    for payload in [
        json!({"text":"Missing path.","source":"memory","hash":"h"}),
        json!({"text":"Missing hash.","source":"memory","path":"manual.md"}),
        json!({"text":"Empty provenance.","source":"memory","path":"","hash":""}),
    ] {
        let error = classify_and_render(&[point(
            KnowledgeCollection::LegacyMemory,
            "bad-provenance",
            payload,
        )])
        .unwrap_err();
        assert!(error.to_string().contains("path") || error.to_string().contains("hash"));
    }
}

#[test]
fn malformed_legacy_source_fails_instead_of_being_filtered_out() {
    let points = vec![point(
        KnowledgeCollection::LegacyMemory,
        "bad-source",
        json!({"text":"Malformed source.","source":42,"path":"manual","hash":"h"}),
    )];

    let error = classify_and_render(&points).unwrap_err();

    assert!(error.to_string().contains("non-string source"));
}

#[test]
fn unclassified_guardrail_requires_the_expected_live_count() {
    let points = vec![point(
        KnowledgeCollection::MemoryUnits,
        "unknown",
        json!({"text":"Review me.","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","seen_in_sessions":["s1"]}),
    )];
    let plan = classify_and_render(&points).unwrap();

    assert!(verify_unclassified_count(&plan, 1).is_ok());
    assert!(verify_unclassified_count(&plan, 222).is_err());
}

#[test]
fn source_plan_equality_detects_content_count_and_provenance_drift() {
    let before = classify_and_render(&[point(
        KnowledgeCollection::MemoryUnits,
        "one",
        json!({"text":"Stable text.","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s1"]}),
    )])
    .unwrap();
    let changed = classify_and_render(&[
        point(
            KnowledgeCollection::MemoryUnits,
            "one",
            json!({"text":"Changed text.","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "two",
            json!({"text":"Added text.","source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"p","category":"correction","seen_in_sessions":["s2"]}),
        ),
    ])
    .unwrap();

    assert!(verify_source_plan_unchanged(&before, &changed).is_err());

    let provenance_changed = classify_and_render(&[point(
        KnowledgeCollection::MemoryUnits,
        "one",
        json!({"text":"Stable text.","source":"session","source_session":"different-session","source_turn":99,"created_at":"2026-07-09T00:00:00Z","project":"p","category":"learning","seen_in_sessions":["different-session"]}),
    )])
    .unwrap();
    assert!(verify_source_plan_unchanged(&before, &provenance_changed).is_err());
}

#[test]
fn quarantine_guardrail_counts_only_memory_unit_sources() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "unit",
            json!({"text":"Memory unit review.","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::NotableFacts,
            "fact",
            json!({"text":"Known excluded KB notable record.","source":"kb","source_session":"kb.md","created_at":"2026-07-02T00:00:00Z","project":"","topics":[],"seen_in_sessions":[]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.quarantined, 1);
    assert_eq!(plan.counts.quarantined_source_points, 1);
}

#[test]
fn quarantine_preserves_each_original_source_value() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "missing",
            json!({"text":"Missing source text.","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "future",
            json!({"text":"Future source text.","source":"future","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();
    let quarantine = plan
        .documents
        .iter()
        .find(|document| {
            document.relative_path.as_path()
                == Path::new("memory/quarantine/unclassified-memory-units.md")
        })
        .unwrap();

    assert!(quarantine.content.contains("sources: &lt;missing&gt;"));
    assert!(quarantine.content.contains("sources: future"));
    assert!(
        plan.manifest
            .iter()
            .any(|entry| entry.source.as_deref() == Some("future"))
    );
    assert!(plan.manifest.iter().any(|entry| entry.source.is_none()));
}

#[test]
fn excluded_counts_are_raw_source_counts_not_deduplicated_records() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "kb-1",
            json!({"text":"Same excluded KB text.","source":"kb","source_session":"kb","source_turn":0,"created_at":"2026-07-01T00:00:00Z","seen_in_sessions":[]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "kb-2",
            json!({"text":"Same excluded KB text.","source":"kb","source_session":"kb","source_turn":0,"created_at":"2026-07-01T00:00:00Z","seen_in_sessions":[]}),
        ),
        point(
            KnowledgeCollection::LegacyMemory,
            "summary-1",
            json!({"text":"Same summary text.","source":"summary","path":"summary","hash":"h1"}),
        ),
        point(
            KnowledgeCollection::LegacyMemory,
            "summary-2",
            json!({"text":"Same summary text.","source":"summary","path":"summary","hash":"h2"}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.excluded_kb_vectors, 2);
    assert_eq!(plan.counts.excluded_non_durable, 2);
    assert_eq!(plan.counts.duplicates, 0);
}

#[test]
fn duplicate_unclassified_points_count_separately_for_live_guardrail() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "unknown-1",
            json!({"text":"Same unclassified text.","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "unknown-2",
            json!({"text":"Same unclassified text.","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.quarantined, 1);
    assert_eq!(plan.counts.quarantined_source_points, 2);
    assert!(verify_unclassified_count(&plan, 2).is_ok());
}

#[test]
fn same_quarantine_text_in_distinct_projects_has_distinct_anchors() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "one",
            json!({"text":"Same quarantine text.","source":"future","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"one","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "two",
            json!({"text":"Same quarantine text.","source":"future","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"two","seen_in_sessions":["s2"]}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();
    let anchors = plan
        .manifest
        .iter()
        .map(|entry| entry.anchor.clone().unwrap())
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(anchors.len(), 2);
}

#[test]
fn empty_global_scope_and_literal_global_project_use_distinct_files() {
    let points = vec![
        point(
            KnowledgeCollection::LegacyMemory,
            "empty",
            json!({"text":"Same scoped text.","source":"memory","path":"manual","hash":"h1","project":""}),
        ),
        point(
            KnowledgeCollection::LegacyMemory,
            "literal",
            json!({"text":"Same scoped text.","source":"memory","path":"manual","hash":"h2","project":"global"}),
        ),
    ];
    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.exported_unique, 2);
    let paths = plan
        .documents
        .iter()
        .map(|document| document.relative_path.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(paths.len(), 2);
    assert!(paths.contains(&PathBuf::from("memory/manual-memories/__global__.md")));
    assert!(paths.contains(&PathBuf::from("memory/manual-memories/project-global.md")));
}

#[test]
fn project_slug_collisions_remain_distinct_and_preserve_original_scope() {
    let points = vec![
        point(
            KnowledgeCollection::MemoryUnits,
            "slash",
            json!({"text":"Same text.","source":"session","source_session":"s1","source_turn":1,"created_at":"2026-07-01T00:00:00Z","project":"a/b","category":"correction","seen_in_sessions":["s1"]}),
        ),
        point(
            KnowledgeCollection::MemoryUnits,
            "dash",
            json!({"text":"Same text.","source":"session","source_session":"s2","source_turn":2,"created_at":"2026-07-02T00:00:00Z","project":"a-b","category":"correction","seen_in_sessions":["s2"]}),
        ),
    ];

    let plan = classify_and_render(&points).unwrap();

    assert_eq!(plan.counts.exported_unique, 2);
    let paths = plan
        .documents
        .iter()
        .map(|document| document.relative_path.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(paths.len(), 2);
    assert!(
        plan.documents
            .iter()
            .any(|document| document.content.contains("project: a/b"))
    );
    assert!(
        plan.documents
            .iter()
            .any(|document| document.content.contains("project: a-b"))
    );
}

#[test]
fn unknown_notable_fact_source_fails_closed() {
    let points = vec![point(
        KnowledgeCollection::NotableFacts,
        "future",
        json!({"text":"Unknown notable source.","source":"future","source_session":"s1","created_at":"2026-07-01T00:00:00Z","project":"","topics":[],"seen_in_sessions":["s1"]}),
    )];

    let error = classify_and_render(&points).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("unsupported notable-fact source")
    );
}

#[test]
fn malformed_in_scope_record_fails_instead_of_entering_quarantine() {
    let points = vec![point(
        KnowledgeCollection::NotableFacts,
        "bad",
        json!({"source":"session","source_session":"s1"}),
    )];

    let error = classify_and_render(&points).unwrap_err();

    assert!(error.to_string().contains("text"));
    assert!(error.to_string().contains("bad"));
}
