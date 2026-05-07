use anyhow::{Context, Result};
use claude_memory::{index, kb_ingest, page_index};
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

pub async fn run_page_index(
    projects: Option<PathBuf>,
    archive: Option<PathBuf>,
    output: Option<PathBuf>,
    max_sessions: Option<usize>,
) -> Result<()> {
    let home = dirs::home_dir().context("no home directory")?;
    let projects_dir = projects.unwrap_or_else(|| home.join(".claude/projects"));
    let output_dir = output.unwrap_or_else(page_index::default_output_dir);
    let summary =
        page_index::build_page_index(&projects_dir, archive.as_deref(), &output_dir, max_sessions)?;
    println!(
        "PageIndex: sessions={} nodes={} output={}",
        summary.sessions,
        summary.nodes,
        summary.output_dir.display()
    );
    Ok(())
}
