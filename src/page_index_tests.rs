use super::*;
use crate::extract::{Role, Turn};
use std::path::Path;

fn turn(role: Role, text: &str, turn_index: u32) -> Turn {
    Turn {
        role,
        text: text.to_string(),
        turn_index,
        has_tool_use: false,
        tool_call_count: 0,
    }
}

#[test]
fn session_index_uses_nested_document_model() {
    let turns = vec![
        turn(Role::User, "How do we deploy?", 0),
        turn(Role::Assistant, "Run the deploy script.", 1),
        turn(Role::User, "How do we test?", 2),
    ];

    let index = build_session_index(Path::new("/tmp/session.jsonl"), &turns);

    assert_eq!(index.doc_id, "session");
    assert_eq!(index.doc_name, "session");
    assert_eq!(index.source_family, "transcript");
    assert_eq!(index.source_path, "/tmp/session.jsonl");
    assert_eq!(index.turn_count, 3);
    assert!(index.text.contains("User: How do we deploy?"));
    assert_eq!(index.nodes.len(), 2);

    let first_node = &index.nodes[0];
    assert_eq!(first_node.node_id, "000001");
    assert_eq!(first_node.source_locator, "turns:0-1");
    assert_eq!(first_node.nodes.len(), 0);
    assert!(
        first_node
            .text
            .contains("Assistant: Run the deploy script.")
    );
}

#[test]
fn session_index_groups_prompt_and_answer_in_one_node() {
    let turns = vec![
        turn(Role::User, "How do we deploy?", 0),
        turn(Role::Assistant, "Run the deploy script.", 1),
        turn(Role::User, "How do we test?", 2),
    ];

    let index = build_session_index(Path::new("session.jsonl"), &turns);

    assert_eq!(index.nodes.len(), 2);
    assert_eq!(index.nodes[0].node_id, "000001");
    assert_eq!(index.nodes[0].start_turn, 0);
    assert_eq!(index.nodes[0].end_turn, 1);
}

#[test]
fn outline_exposes_node_ids_and_titles() {
    let turns = vec![turn(Role::User, "How do we deploy safely?", 0)];
    let index = build_session_index(Path::new("session.jsonl"), &turns);

    let outline = index.outline();

    assert!(outline.contains("000001. How do we deploy safely?"));
}

#[test]
fn node_text_returns_prompt_and_answer() {
    let turns = vec![
        turn(Role::User, "How do we deploy?", 0),
        turn(Role::Assistant, "Run the deploy script.", 1),
    ];
    let index = build_session_index(Path::new("session.jsonl"), &turns);

    let text = index.node_text("000001").unwrap();

    assert!(text.contains("User: How do we deploy?"));
    assert!(text.contains("Assistant: Run the deploy script."));
}

#[test]
fn document_structure_omits_text_and_content_fetch_returns_exact_turns() {
    let root = unique_temp_dir("transcript-page-index-content");
    let turns = vec![
        turn(Role::User, "How do we deploy?", 0),
        turn(Role::Assistant, "Run the deploy script.", 1),
        turn(Role::User, "How do we test?", 2),
    ];
    let index = build_session_index(&root.join("session.jsonl"), &turns);
    write_session_index(&root, &index).unwrap();

    let metadata = document_metadata(&root, "session").unwrap();
    assert_eq!(metadata.doc_id, "session");
    assert_eq!(metadata.turn_count, 3);

    let structure = document_structure(&root, "session").unwrap();
    let json = serde_json::to_string(&structure).unwrap();
    assert!(json.contains("000001"));
    assert!(!json.contains("Run the deploy script."));

    let node_content = document_content(&root, "session", "000001").unwrap();
    assert_eq!(
        node_content.text,
        "User: How do we deploy?\n\nAssistant: Run the deploy script.\n"
    );

    let range_content = document_content(&root, "session", "turns:1-2").unwrap();
    assert_eq!(
        range_content.text,
        "Assistant: Run the deploy script.\n\nUser: How do we test?\n"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn query_returns_traceable_transcript_node_hits() {
    let root = unique_temp_dir("transcript-page-index-query");
    let turns = vec![
        turn(Role::User, "How do we deploy?", 0),
        turn(Role::Assistant, "Run the deploy script.", 1),
    ];
    let index = build_session_index(&root.join("session.jsonl"), &turns);
    write_session_index(&root, &index).unwrap();

    let results = query_index(&root, "deploy script", 3).unwrap();

    assert_eq!(results[0].doc_id, "session");
    assert_eq!(results[0].node_id, "000001");
    assert_eq!(results[0].title, "How do we deploy?");
    assert!(results[0].reason.contains("deploy"));
    assert_eq!(
        results[0].next_content_command,
        "claude-memory transcript-page-index content session 000001"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn fixture_transcripts_prove_structure_content_query_and_no_memory_units() {
    let root = unique_temp_dir("transcript-page-index-fixtures");
    let claude_projects = root.join(".claude/projects/example");
    let claude_archive = root.join(".claude/archive");
    let codex_sessions = root.join(".codex/sessions/2026/05/10");
    let codex_archive = root.join(".codex/archived_sessions");
    let output_dir = root.join("index");
    std::fs::create_dir_all(&claude_projects).unwrap();
    std::fs::create_dir_all(&claude_archive).unwrap();
    std::fs::create_dir_all(&codex_sessions).unwrap();
    std::fs::create_dir_all(&codex_archive).unwrap();
    std::fs::write(
        claude_projects.join("claude-live.jsonl"),
        r#"{"type":"user","message":{"content":"How do we index markdown?"}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Build nested headings."}]}}
{"type":"user","message":{"content":"How do we query it?"}}
{"type":"assistant","message":{"content":[{"type":"text","text":"Fetch exact node content."}]}}
"#,
    )
    .unwrap();
    std::fs::write(
        codex_sessions.join("codex-live.jsonl"),
        r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"How should Codex cache commands?"}]}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Use explicit cache directories."}]}}
"#,
    )
    .unwrap();
    let sources = PageIndexSources {
        claude_projects_dir: &root.join(".claude/projects"),
        claude_archive_dir: &claude_archive,
        codex_sessions_dir: &root.join(".codex/sessions"),
        codex_archive_dir: &codex_archive,
    };

    let summary = build_page_index(&sources, &output_dir, None).unwrap();

    assert_eq!(summary.sessions, 2);
    assert_eq!(summary.nodes, 3);
    let claude_structure = document_structure(&output_dir, "claude-live").unwrap();
    let claude_json = serde_json::to_string(&claude_structure).unwrap();
    assert_eq!(claude_structure.nodes[1].node_id, "000002");
    assert!(!claude_json.contains("Fetch exact node content."));

    let claude_content = document_content(&output_dir, "claude-live", "000002").unwrap();
    assert_eq!(
        claude_content.text,
        "User: How do we query it?\n\nAssistant: Fetch exact node content.\n"
    );

    let codex_results = query_index(&output_dir, "codex cache commands", 5).unwrap();
    assert_eq!(codex_results[0].doc_id, "codex-live");
    assert_eq!(codex_results[0].node_id, "000001");
    assert_eq!(
        codex_results[0].next_content_command,
        "claude-memory transcript-page-index content codex-live 000001"
    );

    let output_files = std::fs::read_dir(&output_dir).unwrap().count();
    assert_eq!(output_files, 2);
    assert!(!output_dir.join("memory-units").exists());
    assert!(!output_dir.join("memories").exists());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn summary_separates_text_turns_from_tool_calls() {
    let turns = vec![
        turn(Role::User, "Please inspect the repo.", 0),
        Turn {
            role: Role::Assistant,
            text: String::new(),
            turn_index: 1,
            has_tool_use: true,
            tool_call_count: 3,
        },
        turn(Role::Assistant, "Done.", 2),
    ];

    let index = build_session_index(Path::new("session.jsonl"), &turns);

    assert_eq!(
        index.nodes[0].summary,
        "1 user text turn(s), 1 assistant text turn(s), 3 tool call(s)"
    );
}

#[test]
fn codex_parser_keeps_only_user_and_assistant_messages() {
    let input = r##"{"type":"session_meta","payload":{"id":"ignored"}}
{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"rules"}]}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"# AGENTS.md instructions for /tmp/repo\nrules"}]}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"try it"}]}}
{"type":"response_item","payload":{"type":"function_call","name":"exec_command","arguments":"{}"}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}
"##;

    let turns = read_codex_turns_from(BufReader::new(input.as_bytes())).unwrap();

    assert_eq!(turns.len(), 3);
    assert_eq!(turns[0].role, Role::User);
    assert_eq!(turns[0].text, "try it");
    assert_eq!(turns[1].tool_call_count, 1);
    assert_eq!(turns[2].role, Role::Assistant);
    assert_eq!(turns[2].text, "done");
}

#[test]
fn page_index_sources_collect_claude_archive_and_codex_sessions() {
    let root = std::env::temp_dir().join(format!("page-index-sources-{}", std::process::id()));
    let claude_projects = root.join("claude/projects");
    let claude_archive = root.join("claude/archive");
    let codex_sessions = root.join("codex/sessions/2026/05/06");
    let codex_archive = root.join("codex/archived_sessions");
    std::fs::create_dir_all(&claude_projects).unwrap();
    std::fs::create_dir_all(&claude_archive).unwrap();
    std::fs::create_dir_all(&codex_sessions).unwrap();
    std::fs::create_dir_all(&codex_archive).unwrap();
    std::fs::write(claude_projects.join("live.jsonl"), "").unwrap();
    std::fs::write(claude_archive.join("archive.jsonl.zst"), "").unwrap();
    std::fs::write(codex_sessions.join("codex-live.jsonl"), "").unwrap();
    std::fs::write(codex_archive.join("codex-archive.jsonl"), "").unwrap();

    let sources = PageIndexSources {
        claude_projects_dir: &claude_projects,
        claude_archive_dir: &claude_archive,
        codex_sessions_dir: &root.join("codex/sessions"),
        codex_archive_dir: &codex_archive,
    };

    let files = collect_session_files(&sources);

    assert_eq!(files.len(), 4);
    let _ = std::fs::remove_dir_all(root);
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()))
}
