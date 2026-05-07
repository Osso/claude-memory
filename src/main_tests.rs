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
        "--output",
        "/tmp/page-index",
        "--max-sessions",
        "5",
    ]);
    let Command::PageIndex {
        projects,
        archive,
        output,
        max_sessions,
    } = cli.command
    else {
        panic!("expected page-index command");
    };
    assert_eq!(projects, Some(PathBuf::from("/tmp/projects")));
    assert_eq!(archive, Some(PathBuf::from("/tmp/archive")));
    assert_eq!(output, Some(PathBuf::from("/tmp/page-index")));
    assert_eq!(max_sessions, Some(5));
}
