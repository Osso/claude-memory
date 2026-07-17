use anyhow::{Context, Result};
use claude_memory::{index, kb_search, page_index, page_index_agentic};
use std::path::{Path, PathBuf};

pub async fn run_index_cmd(
    archive: Option<PathBuf>,
    projects: Option<PathBuf>,
    batch_size: usize,
    fresh: bool,
    delay_ms: u64,
) -> Result<()> {
    let home = dirs::home_dir().expect("no home directory");
    let archive_dir = archive.unwrap_or_else(|| home.join(".claude/archive"));
    let projects_dir = projects.unwrap_or_else(|| home.join(".claude/projects"));
    index::run_index(&archive_dir, &projects_dir, batch_size, fresh, delay_ms).await
}

pub async fn run_index_file_cmd(path: &Path, batch_size: usize) -> Result<()> {
    let count = index::index_file(path, batch_size).await?;
    eprintln!("Indexed {} chunks from {}", count, path.display());
    Ok(())
}

pub fn run_kb_page_index_build(kb: Option<PathBuf>, output: Option<PathBuf>) -> Result<()> {
    let kb_dir = kb.unwrap_or_else(|| PathBuf::from(kb_search::DEFAULT_KB_DIR));
    let output_dir = output.unwrap_or_else(kb_search::default_index_dir);
    let summary = kb_search::build_text_index(&kb_dir, &output_dir)?;
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
    let results = kb_search::search_kb(&kb_dir, &index_dir, query, limit)?;
    print_kb_query_results(&results);
    Ok(())
}

fn print_kb_query_results(results: &[kb_search::KbSearchResult]) {
    if results.is_empty() {
        println!("(no KB notes found)");
        return;
    }
    for (index, result) in results.iter().enumerate() {
        println!(
            "{}. [kb] {}#{} > {} (score: {})",
            index + 1,
            result.path,
            result.node_id,
            result.heading,
            result.score
        );
        println!("   reason: {}", result.reason);
        println!("   next: {}\n", result.next_content_command);
    }
}

pub fn run_kb_page_index_content(
    doc: &str,
    locator: &str,
    kb: Option<PathBuf>,
    index: Option<PathBuf>,
) -> Result<()> {
    let kb_dir = kb.unwrap_or_else(|| PathBuf::from(kb_search::DEFAULT_KB_DIR));
    let index_dir = index.unwrap_or_else(kb_search::default_index_dir);
    let content = kb_search::text_document_content(&kb_dir, &index_dir, Path::new(doc), locator)?;
    print!("{}", content.text);
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
        "Transcript PageIndex: sessions={} nodes={} output={}",
        summary.sessions,
        summary.nodes,
        summary.output_dir.display()
    );
    Ok(())
}

pub fn run_transcript_page_index_document(doc: &str, index: Option<PathBuf>) -> Result<()> {
    let index_dir = index.unwrap_or_else(page_index::default_output_dir);
    let metadata = page_index::document_metadata(&index_dir, Path::new(doc))?;
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

pub fn run_transcript_page_index_structure(doc: &str, index: Option<PathBuf>) -> Result<()> {
    let index_dir = index.unwrap_or_else(page_index::default_output_dir);
    let structure = page_index::document_structure(&index_dir, Path::new(doc))?;
    println!("{}", serde_json::to_string_pretty(&structure)?);
    Ok(())
}

pub fn run_transcript_page_index_content(
    doc: &str,
    locator: &str,
    index: Option<PathBuf>,
) -> Result<()> {
    let index_dir = index.unwrap_or_else(page_index::default_output_dir);
    let content = page_index::document_content(&index_dir, Path::new(doc), locator)?;
    print!("{}", content.text);
    Ok(())
}

pub async fn run_transcript_page_index_query(
    query: &str,
    limit: usize,
    index: Option<PathBuf>,
    mode: page_index_agentic::RetrievalMode,
) -> Result<()> {
    let index_dir = index.unwrap_or_else(page_index::default_output_dir);
    if mode == page_index_agentic::RetrievalMode::Agentic {
        let corpus = page_index_agentic::TranscriptTreeWalkCorpus::new(&index_dir);
        let response = page_index_agentic::retrieve_with_llm(query, &corpus, limit).await?;
        print_tree_walk_response(&response);
        return Ok(());
    }

    let results = page_index::query_index(&index_dir, query, limit)?;
    print_transcript_query_results(&results);
    Ok(())
}

fn print_transcript_query_results(results: &[page_index::PageIndexQueryResult]) {
    if results.is_empty() {
        println!("(no transcript notes found)");
        return;
    }
    for (index, result) in results.iter().enumerate() {
        println!(
            "{}. [transcript] {}#{} > {} (score: {})",
            index + 1,
            result.doc_id,
            result.node_id,
            result.title,
            result.score
        );
        println!("   source: {}", result.source_path);
        println!("   reason: {}", result.reason);
        println!("   next: {}\n", result.next_content_command);
    }
}

fn print_tree_walk_response(response: &page_index_agentic::TreeWalkResponse) {
    println!("PageIndex retrieval mode: {:?}", response.mode);
    println!("Answer:\n{}", response.answer);
    println!("\nReferences:");
    for reference in &response.references {
        println!("- {}#{}", reference.doc_id, reference.locator);
    }
    println!("\nRetrieval path:");
    for step in &response.steps {
        let doc = step.doc_id.as_deref().unwrap_or("-");
        let locator = step.locator.as_deref().unwrap_or("-");
        println!("- {} doc={} locator={}", step.action, doc, locator);
    }
}
