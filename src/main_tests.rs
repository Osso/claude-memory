use super::*;

#[test]
fn index_defaults_cover_claude_codex_and_pi_sources() {
    let paths = indexing_cmds::default_index_source_paths(
        std::path::Path::new("/home/test"),
        std::path::Path::new("/home/test/.config"),
    );

    assert_eq!(
        paths.claude_projects,
        PathBuf::from("/home/test/.claude/projects")
    );
    assert_eq!(
        paths.claude_archive,
        PathBuf::from("/home/test/.claude/archive")
    );
    assert_eq!(
        paths.codex_sessions,
        PathBuf::from("/home/test/.codex/sessions")
    );
    assert_eq!(
        paths.codex_archive,
        PathBuf::from("/home/test/.codex/archived_sessions")
    );
    assert_eq!(
        paths.pi_sessions,
        PathBuf::from("/home/test/.config/pi/agent/sessions")
    );
}

#[test]
fn index_accepts_claude_codex_and_pi_paths() {
    let cli = Cli::parse_from([
        "claude-memory",
        "index",
        "--projects",
        "/tmp/claude/projects",
        "--archive",
        "/tmp/claude/archive",
        "--codex-sessions",
        "/tmp/codex/sessions",
        "--codex-archive",
        "/tmp/codex/archive",
        "--pi-sessions",
        "/tmp/pi/sessions",
    ]);
    let Command::Index {
        projects,
        archive,
        codex_sessions,
        codex_archive,
        pi_sessions,
        ..
    } = cli.command
    else {
        panic!("expected index command");
    };

    assert_eq!(projects, Some(PathBuf::from("/tmp/claude/projects")));
    assert_eq!(archive, Some(PathBuf::from("/tmp/claude/archive")));
    assert_eq!(codex_sessions, Some(PathBuf::from("/tmp/codex/sessions")));
    assert_eq!(codex_archive, Some(PathBuf::from("/tmp/codex/archive")));
    assert_eq!(pi_sessions, Some(PathBuf::from("/tmp/pi/sessions")));
}

#[test]
fn search_defaults_to_combined_prompt_and_answer_history() {
    let cli = Cli::parse_from(["claude-memory", "search", "ollama"]);
    let Command::Search {
        query,
        limit,
        target,
        json,
    } = cli.command
    else {
        panic!("expected search command");
    };

    assert_eq!(query, "ollama");
    assert_eq!(limit, 5);
    assert_eq!(target, None);
    assert!(!json);
}

#[test]
fn search_accepts_prompt_type() {
    let cli = Cli::parse_from(["claude-memory", "search", "--type", "prompts", "ollama"]);
    let Command::Search { target, .. } = cli.command else {
        panic!("expected search command");
    };
    assert_eq!(target, Some(SearchTarget::Prompts));
}

#[test]
fn search_accepts_answer_type() {
    let cli = Cli::parse_from(["claude-memory", "search", "--type", "answers", "ollama"]);
    let Command::Search { target, .. } = cli.command else {
        panic!("expected search command");
    };
    assert_eq!(target, Some(SearchTarget::Answers));
}

#[test]
fn search_accepts_json_output() {
    let cli = Cli::parse_from(["claude-memory", "search", "--json", "ollama"]);
    let Command::Search { json, .. } = cli.command else {
        panic!("expected search command");
    };
    assert!(json);
}

#[test]
fn search_json_output_is_stable_ndjson_in_rank_order() {
    let results = vec![
        index::SearchResult {
            record_type: "answer".to_string(),
            text: "first".to_string(),
            source: "session".to_string(),
            path: "/history/first".to_string(),
            session_id: "session-first".to_string(),
            score: 0.91,
        },
        index::SearchResult {
            record_type: "prompt".to_string(),
            text: "second".to_string(),
            source: "archive".to_string(),
            path: "/history/second".to_string(),
            session_id: "session-second".to_string(),
            score: 0.42,
        },
    ];
    let mut output = Vec::new();

    write_results_json(&results, &mut output).expect("write NDJSON");

    assert_eq!(
        String::from_utf8(output).expect("UTF-8 output"),
        concat!(
            "{\"type\":\"answer\",\"text\":\"first\",\"source\":\"session\",\"path\":\"/history/first\",\"session_id\":\"session-first\",\"score\":0.91}\n",
            "{\"type\":\"prompt\",\"text\":\"second\",\"source\":\"archive\",\"path\":\"/history/second\",\"session_id\":\"session-second\",\"score\":0.42}\n",
        )
    );
}

#[test]
fn manual_memory_commands_are_retired() {
    for command in ["memory-write", "memory-delete"] {
        let error = match Cli::try_parse_from(["claude-memory", command]) {
            Ok(_) => panic!("{command} should be rejected"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidSubcommand);
    }
}

#[test]
fn legacy_memory_commands_are_retired() {
    for command in [
        "analyze",
        "backfill",
        "deduplicate",
        "build-graph",
        "graph-clean",
        "graph-dump",
    ] {
        let error = match Cli::try_parse_from(["claude-memory", command]) {
            Ok(_) => panic!("{command} should be rejected"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidSubcommand);
    }
}

#[test]
fn ingest_kb_command_is_retired() {
    let error = match Cli::try_parse_from(["claude-memory", "ingest-kb"]) {
        Ok(_) => panic!("ingest-kb should be rejected"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), clap::error::ErrorKind::InvalidSubcommand);
}

#[test]
fn transcript_page_index_accepts_projects_archive_output_and_limit() {
    let cli = Cli::parse_from([
        "claude-memory",
        "transcript-page-index",
        "build",
        "--projects",
        "/tmp/projects",
        "--archive",
        "/tmp/archive",
        "--codex-sessions",
        "/tmp/codex/sessions",
        "--codex-archive",
        "/tmp/codex/archive",
        "--output",
        "/tmp/transcript-page-index",
        "--max-sessions",
        "5",
    ]);
    let Command::TranscriptPageIndex {
        command:
            TranscriptPageIndexCommand::Build {
                projects,
                archive,
                codex_sessions,
                codex_archive,
                output,
                max_sessions,
            },
    } = cli.command
    else {
        panic!("expected transcript-page-index command");
    };
    assert_eq!(projects, Some(PathBuf::from("/tmp/projects")));
    assert_eq!(archive, Some(PathBuf::from("/tmp/archive")));
    assert_eq!(codex_sessions, Some(PathBuf::from("/tmp/codex/sessions")));
    assert_eq!(codex_archive, Some(PathBuf::from("/tmp/codex/archive")));
    assert_eq!(output, Some(PathBuf::from("/tmp/transcript-page-index")));
    assert_eq!(max_sessions, Some(5));
}

#[test]
fn transcript_page_index_accepts_document_structure_content_and_query_commands() {
    let document = Cli::parse_from([
        "claude-memory",
        "transcript-page-index",
        "document",
        "session",
        "--index",
        "/tmp/transcript-index",
    ]);
    let Command::TranscriptPageIndex {
        command: TranscriptPageIndexCommand::Document { doc, index },
    } = document.command
    else {
        panic!("expected transcript-page-index document command");
    };
    assert_eq!(doc, "session");
    assert_eq!(index, Some(PathBuf::from("/tmp/transcript-index")));

    let structure = Cli::parse_from([
        "claude-memory",
        "transcript-page-index",
        "structure",
        "session",
        "--index",
        "/tmp/transcript-index",
    ]);
    let Command::TranscriptPageIndex {
        command: TranscriptPageIndexCommand::Structure { doc, index },
    } = structure.command
    else {
        panic!("expected transcript-page-index structure command");
    };
    assert_eq!(doc, "session");
    assert_eq!(index, Some(PathBuf::from("/tmp/transcript-index")));

    let content = Cli::parse_from([
        "claude-memory",
        "transcript-page-index",
        "content",
        "session",
        "000001",
        "--index",
        "/tmp/transcript-index",
    ]);
    let Command::TranscriptPageIndex {
        command:
            TranscriptPageIndexCommand::Content {
                doc,
                locator,
                index,
            },
    } = content.command
    else {
        panic!("expected transcript-page-index content command");
    };
    assert_eq!(doc, "session");
    assert_eq!(locator, "000001");
    assert_eq!(index, Some(PathBuf::from("/tmp/transcript-index")));

    let query = Cli::parse_from([
        "claude-memory",
        "transcript-page-index",
        "query",
        "deploy script",
        "--index",
        "/tmp/transcript-index",
        "--limit",
        "2",
    ]);
    let Command::TranscriptPageIndex {
        command:
            TranscriptPageIndexCommand::Query {
                query,
                limit,
                index,
            },
    } = query.command
    else {
        panic!("expected transcript-page-index query command");
    };
    assert_eq!(query, "deploy script");
    assert_eq!(limit, 2);
    assert_eq!(index, Some(PathBuf::from("/tmp/transcript-index")));

    let error = match Cli::try_parse_from([
        "claude-memory",
        "transcript-page-index",
        "query",
        "deploy script",
        "--mode",
        "agentic",
    ]) {
        Ok(_) => panic!("retired agentic mode should be rejected"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
}

#[test]
fn kb_page_index_accepts_build_paths() {
    let cli = Cli::parse_from([
        "claude-memory",
        "kb-page-index",
        "build",
        "--kb",
        "/tmp/kb",
        "--output",
        "/tmp/kb-index",
    ]);
    let Command::KbPageIndex {
        command: KbPageIndexCommand::Build { kb, output },
    } = cli.command
    else {
        panic!("expected kb-page-index build command");
    };
    assert_eq!(kb, Some(PathBuf::from("/tmp/kb")));
    assert_eq!(output, Some(PathBuf::from("/tmp/kb-index")));
}

#[test]
fn kb_page_index_accepts_query_paths_and_limit() {
    let cli = Cli::parse_from([
        "claude-memory",
        "kb-page-index",
        "query",
        "frontend design",
        "--limit",
        "2",
        "--kb",
        "/tmp/kb",
        "--index",
        "/tmp/kb-index",
    ]);
    let Command::KbPageIndex {
        command:
            KbPageIndexCommand::Query {
                query,
                limit,
                kb,
                index,
            },
    } = cli.command
    else {
        panic!("expected kb-page-index query command");
    };
    assert_eq!(query, "frontend design");
    assert_eq!(limit, 2);
    assert_eq!(kb, Some(PathBuf::from("/tmp/kb")));
    assert_eq!(index, Some(PathBuf::from("/tmp/kb-index")));
}

#[test]
fn kb_page_index_accepts_content_command() {
    let content = Cli::parse_from([
        "claude-memory",
        "kb-page-index",
        "content",
        "guides/router.md",
        "4-8",
        "--kb",
        "/tmp/kb",
        "--index",
        "/tmp/kb-index",
    ]);
    let Command::KbPageIndex {
        command:
            KbPageIndexCommand::Content {
                doc,
                locator,
                kb,
                index,
            },
    } = content.command
    else {
        panic!("expected kb-page-index content command");
    };
    assert_eq!(doc, "guides/router.md");
    assert_eq!(locator, "4-8");
    assert_eq!(kb, Some(PathBuf::from("/tmp/kb")));
    assert_eq!(index, Some(PathBuf::from("/tmp/kb-index")));
}
