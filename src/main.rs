use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use claude_memory::{graph, index};
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

mod dedup;
use dedup::{cluster_similar, load_all_memories, merge_clusters, print_clusters};

#[derive(Parser)]
#[command(name = "claude-memory", about = "Semantic memory for Claude Code")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index conversations from archive and projects directories
    Index {
        /// Archive directory (default: ~/.claude/archive)
        #[arg(long)]
        archive: Option<PathBuf>,

        /// Projects directory (default: ~/.claude/projects)
        #[arg(long)]
        projects: Option<PathBuf>,

        /// Knowledge base directory (default: /syncthing/Sync/KB)
        #[arg(long)]
        kb: Option<PathBuf>,

        /// Batch size for embedding
        #[arg(long, default_value = "10")]
        batch_size: usize,

        /// Ignore existing index and re-index everything
        #[arg(long)]
        fresh: bool,

        /// Delay in milliseconds between batches (throttle GPU usage)
        #[arg(long, default_value = "0")]
        delay_ms: u64,
    },

    /// Index a single conversation file
    IndexFile {
        /// Path to .jsonl or .jsonl.zst file
        path: PathBuf,

        /// Batch size for embedding
        #[arg(long, default_value = "10")]
        batch_size: usize,
    },

    /// Search prompts (user messages, KB)
    SearchPrompts {
        /// Query text
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Filter by source (archive, session, summary, kb, memory)
        #[arg(long)]
        source: Option<String>,
    },

    /// Search answers (assistant responses)
    SearchAnswers {
        /// Query text
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Filter by source (archive, session)
        #[arg(long)]
        source: Option<String>,
    },

    /// Deduplicate existing memory entries (merge similar ones via LLM)
    Deduplicate {
        /// Similarity threshold (0.0-1.0) for merging
        #[arg(long, default_value = "0.88")]
        threshold: f32,

        /// Dry run: show what would be merged without actually merging
        #[arg(long)]
        dry_run: bool,
    },

    /// Build graph from existing memory entries (extract entities/relationships)
    BuildGraph {
        /// Also scan KB files for graph extraction
        #[arg(long)]
        kb: bool,
        /// Clear the existing graph before rebuilding
        #[arg(long)]
        fresh: bool,
    },
    /// Clean the existing graph in-place using current validation rules
    GraphClean {
        /// Maximum cleanup passes before stopping
        #[arg(long, default_value = "5")]
        max_passes: usize,
        /// Dry run: report what would change without writing
        #[arg(long)]
        dry_run: bool,
    },
    /// Enrich a prompt with memory context (for UserPromptSubmit hook)
    Enrich {
        /// Maximum memory results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Dump graph entities and relationships
    GraphDump {
        /// Maximum entries to show
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },

    /// Show collection statistics
    Stats,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    run_command(Cli::parse().command).await
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

async fn run_command(command: Command) -> Result<()> {
    use Command::{GraphClean, SearchAnswers, SearchPrompts};

    match command {
        Command::Index {
            archive,
            projects,
            kb,
            batch_size,
            fresh,
            delay_ms,
        } => run_index_cmd(archive, projects, kb, batch_size, fresh, delay_ms).await,
        Command::IndexFile { path, batch_size } => run_index_file_cmd(&path, batch_size).await,
        SearchPrompts {
            query,
            limit,
            source,
        } => run_search_prompts(query, limit, source).await,
        SearchAnswers {
            query,
            limit,
            source,
        } => run_search_answers(query, limit, source).await,
        Command::Deduplicate { threshold, dry_run } => run_deduplicate(threshold, dry_run).await,
        Command::BuildGraph { kb, fresh } => run_build_graph(kb, fresh).await,
        GraphClean {
            max_passes,
            dry_run,
        } => run_graph_clean_cmd(max_passes, dry_run),
        Command::Enrich { limit } => run_enrich(limit).await,
        Command::GraphDump { limit } => run_graph_dump(limit),
        Command::Stats => index::show_stats().await,
    }
}

async fn run_search_prompts(query: String, limit: usize, source: Option<String>) -> Result<()> {
    let results = index::search_prompts(&query, limit, source.as_deref()).await?;
    print_results(&results)
}

async fn run_search_answers(query: String, limit: usize, source: Option<String>) -> Result<()> {
    let results = index::search_answers(&query, limit, source.as_deref()).await?;
    print_results(&results)
}

fn run_graph_clean_cmd(max_passes: usize, dry_run: bool) -> Result<()> {
    let stats = graph::clean_graph(max_passes, dry_run)?;
    eprintln!(
        "Graph clean: {} pass(es), {} relationships seen, {} kept, {} removed, {} rewritten, {} entities removed",
        stats.passes,
        stats.relationships_seen,
        stats.relationships_kept,
        stats.relationships_removed,
        stats.relationships_rewritten,
        stats.entities_removed
    );
    Ok(())
}

async fn run_index_cmd(
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

async fn run_index_file_cmd(path: &PathBuf, batch_size: usize) -> Result<()> {
    let count = index::index_file(path, batch_size).await?;
    eprintln!("Indexed {} chunks from {}", count, path.display());
    Ok(())
}

fn print_results(results: &[index::SearchResult]) -> Result<()> {
    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. [{}] {} (score: {:.3})",
            i + 1,
            result.source,
            result.path,
            result.score
        );
        let text = result.text.replace('\n', " ");
        if text.len() > 200 {
            println!("   {}...", &text[..200]);
        } else {
            println!("   {}", text);
        }
        println!();
    }
    Ok(())
}

async fn run_deduplicate(threshold: f32, dry_run: bool) -> Result<()> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        anyhow::bail!("ANTHROPIC_API_KEY must be set for deduplication (needs LLM to merge)");
    }
    let entries = load_all_memories().await?;
    eprintln!("Loaded {} memory entries", entries.len());
    if entries.is_empty() {
        return Ok(());
    }
    let clusters = cluster_similar(&entries, threshold).await?;
    let merge_count = clusters.iter().filter(|c| c.len() > 1).count();
    eprintln!("Found {} clusters to merge", merge_count);
    if dry_run {
        print_clusters(&entries, &clusters);
        return Ok(());
    }
    merge_clusters(&entries, &clusters).await
}

async fn run_build_graph(kb: bool, fresh: bool) -> Result<()> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        anyhow::bail!("ANTHROPIC_API_KEY must be set for graph building (needs LLM to extract)");
    }
    if fresh {
        graph::clear_graph()?;
    }
    let entries = load_all_memories().await?;
    let mut extracted = extract_texts_to_graph(
        &entries.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(),
        "memory",
    )
    .await?;
    if kb {
        let kb_texts = load_kb_texts()?;
        extracted += extract_texts_to_graph(
            &kb_texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "KB",
        )
        .await?;
    }
    eprintln!("Total: {extracted} triplets");
    Ok(())
}

async fn extract_texts_to_graph(texts: &[&str], label: &str) -> Result<usize> {
    eprintln!(
        "Processing {} {label} entries for graph extraction",
        texts.len()
    );
    let mut extracted = 0;
    for (i, text) in texts.iter().enumerate() {
        match graph::extract_and_store(text).await {
            Ok(n) => extracted += n,
            Err(e) => eprintln!("  entry {}: {e}", i + 1),
        }
        eprint!("\r  {}/{} ({} triplets)", i + 1, texts.len(), extracted);
    }
    eprintln!();
    Ok(extracted)
}

fn load_kb_texts() -> Result<Vec<String>> {
    let kb_dir = PathBuf::from("/syncthing/Sync/KB");
    let include_dirs = ["dev", "guides", "research", "state", "memory"];
    let texts: Vec<String> = include_dirs
        .iter()
        .flat_map(|dir_name| collect_kb_chunks(&kb_dir.join(dir_name)))
        .collect();
    eprintln!("Loaded {} KB chunks from {:?}", texts.len(), include_dirs);
    Ok(texts)
}

fn collect_kb_chunks(dir: &Path) -> Vec<String> {
    if !dir.exists() {
        return vec![];
    }

    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| is_markdown(path))
        .flat_map(|path| read_chunked_markdown(&path))
        .collect()
}

fn is_markdown(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "md")
}

fn read_chunked_markdown(path: &Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return vec![];
    };

    claude_memory::chunk::chunk_text(&content)
        .into_iter()
        .map(|chunk| chunk.text)
        .collect()
}

fn run_graph_dump(limit: usize) -> Result<()> {
    let db = graph::get_graph()?;
    let entities = db
        .run_script(
            &format!("?[name, type] := *entities{{name, entity_type: type}} :limit {limit}"),
            std::collections::BTreeMap::new(),
            cozo::ScriptMutability::Immutable,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("=== Entities ({} shown) ===", entities.rows.len());
    for row in &entities.rows {
        let name = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let etype = row.get(1).and_then(|v| v.get_str()).unwrap_or("");
        println!("  {name} [{etype}]");
    }

    let rels = db
        .run_script(
            &format!(
                "?[src, rel, dst] := *relationships{{src, relation: rel, dst}} :limit {limit}"
            ),
            std::collections::BTreeMap::new(),
            cozo::ScriptMutability::Immutable,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("\n=== Relationships ({} shown) ===", rels.rows.len());
    for row in &rels.rows {
        let src = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let rel = row.get(1).and_then(|v| v.get_str()).unwrap_or("");
        let dst = row.get(2).and_then(|v| v.get_str()).unwrap_or("");
        println!("  {src} --[{rel}]--> {dst}");
    }
    Ok(())
}

// --- Enrich command (UserPromptSubmit hook) ---

async fn run_enrich(limit: usize) -> Result<()> {
    let input: serde_json::Value = read_hook_stdin()?;
    let prompt = input["prompt"].as_str().unwrap_or("");
    if prompt.is_empty() {
        print_hook_output("");
        return Ok(());
    }

    let mut sections = Vec::new();

    // Vector search for memories (filter low-score noise)
    const MIN_SCORE: f32 = 0.85;
    let memories = index::search_prompts(prompt, limit, Some("memory")).await;
    if let Ok(ref results) = memories {
        let relevant: Vec<_> = results.iter().filter(|r| r.score >= MIN_SCORE).collect();
        if !relevant.is_empty() {
            let mem_text = format_memory_results(&relevant);
            sections.push(mem_text);
        }
    }

    // Graph: extract entities from prompt, query relationships
    let entities = graph::find_concepts(prompt).await;
    if !entities.is_empty() {
        if let Ok(related) = graph::query_related(&entities) {
            if !related.is_empty() {
                let graph_text = format_graph_results(&related);
                sections.push(graph_text);
            }
        }
    }

    if sections.is_empty() {
        print_hook_output("");
    } else {
        print_hook_output(&sections.join("\n\n"));
    }
    Ok(())
}

fn read_hook_stdin() -> Result<serde_json::Value> {
    let mut buf = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
    serde_json::from_str(&buf).context("failed to parse hook input")
}

fn format_memory_results(results: &[&index::SearchResult]) -> String {
    let mut out = String::from("Relevant memories:");
    for r in results {
        let text = r.text.replace('\n', " ");
        let text = if text.len() > 300 {
            format!("{}...", &text[..300])
        } else {
            text
        };
        out.push_str(&format!("\n- ({:.2}) {}", r.score, text));
    }
    out
}

fn format_graph_results(related: &[String]) -> String {
    let mut out = String::from("Graph context:");
    for r in related.iter().take(20) {
        out.push_str(&format!("\n- {r}"));
    }
    if related.len() > 20 {
        out.push_str(&format!("\n  ...and {} more", related.len() - 20));
    }
    out
}

fn print_hook_output(context: &str) {
    if context.is_empty() {
        println!("{{}}");
        return;
    }
    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    });
    println!("{}", output);
}
