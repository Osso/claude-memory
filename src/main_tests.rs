use super::*;

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
fn page_index_accepts_projects_archive_output_and_limit() {
    let cli = Cli::parse_from([
        "claude-memory",
        "page-index",
        "--projects",
        "/tmp/projects",
        "--archive",
        "/tmp/archive",
        "--codex-sessions",
        "/tmp/codex/sessions",
        "--codex-archive",
        "/tmp/codex/archive",
        "--output",
        "/tmp/page-index",
        "--max-sessions",
        "5",
    ]);
    let Command::PageIndex {
        projects,
        archive,
        codex_sessions,
        codex_archive,
        output,
        max_sessions,
    } = cli.command
    else {
        panic!("expected page-index command");
    };
    assert_eq!(projects, Some(PathBuf::from("/tmp/projects")));
    assert_eq!(archive, Some(PathBuf::from("/tmp/archive")));
    assert_eq!(codex_sessions, Some(PathBuf::from("/tmp/codex/sessions")));
    assert_eq!(codex_archive, Some(PathBuf::from("/tmp/codex/archive")));
    assert_eq!(output, Some(PathBuf::from("/tmp/page-index")));
    assert_eq!(max_sessions, Some(5));
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
