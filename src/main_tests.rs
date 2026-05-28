use super::*;
use claude_memory::page_index_agentic;

#[test]
fn search_defaults_to_memories() {
    let cli = Cli::parse_from(["claude-memory", "search", "ollama"]);
    let Command::Search { target, .. } = cli.command else {
        panic!("expected search command");
    };
    assert_eq!(target, SearchTarget::Memories);
}

#[test]
fn memory_search_uses_semantic_query_when_enabled() {
    assert_eq!(memory_search_mode(true), MemorySearchMode::Semantic);
}

#[test]
fn memory_search_falls_back_to_substring_when_disabled() {
    assert_eq!(memory_search_mode(false), MemorySearchMode::Substring);
}

#[test]
fn search_accepts_prompt_type() {
    let cli = Cli::parse_from(["claude-memory", "search", "--type", "prompts", "ollama"]);
    let Command::Search { target, .. } = cli.command else {
        panic!("expected search command");
    };
    assert_eq!(target, SearchTarget::Prompts);
}

#[test]
fn search_accepts_answer_type() {
    let cli = Cli::parse_from(["claude-memory", "search", "--type", "answers", "ollama"]);
    let Command::Search { target, .. } = cli.command else {
        panic!("expected search command");
    };
    assert_eq!(target, SearchTarget::Answers);
}

#[test]
fn memory_write_accepts_no_project_scope_when_guidance_only() {
    let cli = Cli::parse_from(["claude-memory", "memory-write", "remember this"]);
    let Command::MemoryWrite { text, project } = cli.command else {
        panic!("expected memory-write command");
    };

    assert_eq!(text, "remember this");
    assert_eq!(project, None);
}

#[test]
fn manual_memory_write_guidance_points_to_docs_local() {
    let guidance = claude_memory::memory_unit::manual_memory_write_guidance();

    assert!(guidance.contains("disabled"));
    assert!(guidance.contains("docs/local/memory.md"));
    assert!(guidance.contains("/home/osso/AgentConfig/rules"));
}

#[test]
fn ingest_kb_accepts_dry_run_and_limit() {
    let cli = Cli::parse_from([
        "claude-memory",
        "ingest-kb",
        "--kb",
        "/tmp/kb",
        "--max-files",
        "3",
        "--dry-run",
    ]);
    let Command::IngestKb {
        kb,
        max_files,
        dry_run,
    } = cli.command
    else {
        panic!("expected ingest-kb command");
    };
    assert_eq!(kb, Some(PathBuf::from("/tmp/kb")));
    assert_eq!(max_files, Some(3));
    assert!(dry_run);
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
                mode,
            },
    } = query.command
    else {
        panic!("expected transcript-page-index query command");
    };
    assert_eq!(query, "deploy script");
    assert_eq!(limit, 2);
    assert_eq!(index, Some(PathBuf::from("/tmp/transcript-index")));
    assert_eq!(mode, page_index_agentic::RetrievalMode::Lexical);

    let agentic_query = Cli::parse_from([
        "claude-memory",
        "transcript-page-index",
        "query",
        "deploy script",
        "--mode",
        "agentic",
    ]);
    let Command::TranscriptPageIndex {
        command: TranscriptPageIndexCommand::Query { mode, .. },
    } = agentic_query.command
    else {
        panic!("expected transcript-page-index query command");
    };
    assert_eq!(mode, page_index_agentic::RetrievalMode::Agentic);
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
                mode,
            },
    } = cli.command
    else {
        panic!("expected kb-page-index query command");
    };
    assert_eq!(query, "frontend design");
    assert_eq!(limit, 2);
    assert_eq!(kb, Some(PathBuf::from("/tmp/kb")));
    assert_eq!(index, Some(PathBuf::from("/tmp/kb-index")));
    assert_eq!(mode, page_index_agentic::RetrievalMode::Lexical);
}

#[test]
fn kb_page_index_accepts_document_structure_and_content_commands() {
    let document = Cli::parse_from([
        "claude-memory",
        "kb-page-index",
        "document",
        "guides/router.md",
        "--index",
        "/tmp/kb-index",
    ]);
    let Command::KbPageIndex {
        command: KbPageIndexCommand::Document { doc, index },
    } = document.command
    else {
        panic!("expected kb-page-index document command");
    };
    assert_eq!(doc, "guides/router.md");
    assert_eq!(index, Some(PathBuf::from("/tmp/kb-index")));

    let structure = Cli::parse_from([
        "claude-memory",
        "kb-page-index",
        "structure",
        "guides/router.md",
        "--index",
        "/tmp/kb-index",
    ]);
    let Command::KbPageIndex {
        command: KbPageIndexCommand::Structure { doc, index },
    } = structure.command
    else {
        panic!("expected kb-page-index structure command");
    };
    assert_eq!(doc, "guides/router.md");
    assert_eq!(index, Some(PathBuf::from("/tmp/kb-index")));

    let content = Cli::parse_from([
        "claude-memory",
        "kb-page-index",
        "content",
        "guides/router.md",
        "000002",
        "--index",
        "/tmp/kb-index",
    ]);
    let Command::KbPageIndex {
        command:
            KbPageIndexCommand::Content {
                doc,
                locator,
                index,
            },
    } = content.command
    else {
        panic!("expected kb-page-index content command");
    };
    assert_eq!(doc, "guides/router.md");
    assert_eq!(locator, "000002");
    assert_eq!(index, Some(PathBuf::from("/tmp/kb-index")));
}
