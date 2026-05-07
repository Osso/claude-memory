use anyhow::{Context, Result};
use claude_memory::{index, kb_ingest, kb_search, page_index};
use std::path::PathBuf;

pub async fn run_index_cmd(
    archive: Option<PathBuf>,
    projects: Option<PathBuf>,
    kb: Option<PathBuf>,
    batch_size: usize,
    fresh: bool,
    delay_ms: u64,
) -> Result<()> {
    let home = dirs::home_dir().expect("no home directory");
    let archive_dir = archive.unwrap_or_else(|| home.join(".claude/archive"));
    let projects_dir = projects.unwrap_or_else(|| home.join(".claude/projects"));
    let kb_dir = kb.unwrap_or_else(|| PathBuf::from("/syncthing/Sync/KB"));
    index::run_index(
        &archive_dir,
        &projects_dir,
        &kb_dir,
        batch_size,
        fresh,
        delay_ms,
    )
    .await
}

pub async fn run_index_file_cmd(path: &PathBuf, batch_size: usize) -> Result<()> {
    let count = index::index_file(path, batch_size).await?;
    eprintln!("Indexed {} chunks from {}", count, path.display());
    Ok(())
}

pub async fn run_ingest_kb(
    kb: Option<PathBuf>,
    max_files: Option<usize>,
    dry_run: bool,
) -> Result<()> {
    let kb_dir = kb.unwrap_or_else(|| PathBuf::from("/syncthing/Sync/KB"));
    let summary = kb_ingest::ingest_kb_dir(&kb_dir, max_files, dry_run).await?;
    println!(
        "KB ingest: files={} sections={} facts={} inserted={} merged={}",
        summary.files, summary.sections, summary.facts, summary.inserted, summary.merged
    );
    Ok(())
}

pub fn run_kb_page_index_build(kb: Option<PathBuf>, output: Option<PathBuf>) -> Result<()> {
    let kb_dir = kb.unwrap_or_else(|| PathBuf::from(kb_search::DEFAULT_KB_DIR));
    let output_dir = output.unwrap_or_else(kb_search::default_index_dir);
    let summary = kb_search::build_index(&kb_dir, &output_dir)?;
    println!(
        "KB PageIndex: files={} nodes={} output={}",
        summary.files,
        summary.nodes,
        summary.index_path.display()
    );
    Ok(())
}

pub fn run_kb_page_index_query(
    query: &str,
    limit: usize,
    kb: Option<PathBuf>,
    index: Option<PathBuf>,
) -> Result<()> {
    let kb_dir = kb.unwrap_or_else(|| PathBuf::from(kb_search::DEFAULT_KB_DIR));
    let index_dir = index.unwrap_or_else(kb_search::default_index_dir);
    let results = kb_search::search_or_build(&kb_dir, &index_dir, query, limit)?;
    if results.is_empty() {
        println!("(no KB notes found)");
        return Ok(());
    }

    for (index, result) in results.iter().enumerate() {
        println!(
            "{}. [kb] {} > {} (score: {})",
            index + 1,
            result.path,
            result.heading,
            result.score
        );
        println!("   {}\n", result.text.replace('\n', " "));
    }
    Ok(())
}

pub async fn run_page_index(
    projects: Option<PathBuf>,
    archive: Option<PathBuf>,
    codex_sessions: Option<PathBuf>,
    codex_archive: Option<PathBuf>,
    output: Option<PathBuf>,
    max_sessions: Option<usize>,
) -> Result<()> {
    let home = dirs::home_dir().context("no home directory")?;
    let projects_dir = projects.unwrap_or_else(|| home.join(".claude/projects"));
    let archive_dir = archive.unwrap_or_else(|| home.join(".claude/archive"));
    let codex_sessions_dir = codex_sessions.unwrap_or_else(|| home.join(".codex/sessions"));
    let codex_archive_dir = codex_archive.unwrap_or_else(|| home.join(".codex/archived_sessions"));
    let output_dir = output.unwrap_or_else(page_index::default_output_dir);
    let sources = page_index::PageIndexSources {
        claude_projects_dir: &projects_dir,
        claude_archive_dir: &archive_dir,
        codex_sessions_dir: &codex_sessions_dir,
        codex_archive_dir: &codex_archive_dir,
    };
    let summary = page_index::build_page_index(&sources, &output_dir, max_sessions)?;
    println!(
        "PageIndex: sessions={} nodes={} output={}",
        summary.sessions,
        summary.nodes,
        summary.output_dir.display()
    );
    Ok(())
}
