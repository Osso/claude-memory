use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use claude_memory::{analyze, backfill, config, graph, index, memory_unit};
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

    /// Analyze a single session JSONL for friction moments and extract memory units
    Analyze {
        /// Path to the session .jsonl file
        session_jsonl: PathBuf,
    },

    /// List stored memory units (id, source, text)
    MemoryList {
        #[arg(short, long, default_value = "20")]
        limit: usize,
        #[arg(long)]
        offset: Option<u64>,
        /// Case-insensitive substring filter on memory text
        #[arg(long)]
        substring: Option<String>,
        /// Semantic similarity search (returns top-K by cosine score)
        #[arg(long)]
        query: Option<String>,
    },

    /// Delete a memory unit by its numeric Qdrant point ID
    MemoryDelete {
        /// Point ID returned by `memory-list`
        id: u64,
    },

    /// Manually write a memory unit (alternative to the MCP memory_write tool)
    MemoryWrite {
        /// Memory text (1-3 sentences, encyclopedia-style)
        text: String,
    },

    /// Backfill: walk the live projects directory and analyze each session.
    /// Resumable via a sidecar state file.
    Backfill {
        /// Projects directory (default: ~/.claude/projects)
        #[arg(long)]
        projects: Option<PathBuf>,

        /// Archive directory of .jsonl.zst files (default: skip archives).
        /// Pass `~/.claude/archive` to also process compressed archives.
        #[arg(long)]
        archive: Option<PathBuf>,

        /// State file tracking processed session IDs
        /// (default: ~/.cache/claude-memory/backfill-processed.txt)
        #[arg(long)]
        state_file: Option<PathBuf>,

        /// Skip sessions with fewer than this many user turns
        #[arg(long, default_value = "3")]
        min_user_turns: usize,

        /// Stop after analysing this many sessions (useful for testing)
        #[arg(long)]
        max_sessions: Option<usize>,
    },
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
        Command::Analyze { session_jsonl } => run_analyze(&session_jsonl).await,
        Command::MemoryList {
            limit,
            offset,
            substring,
            query,
        } => run_memory_list(limit, offset, substring, query).await,
        Command::MemoryDelete { id } => run_memory_delete(id).await,
        Command::MemoryWrite { text } => run_memory_write(text).await,
        Command::Backfill {
            projects,
            archive,
            state_file,
            min_user_turns,
            max_sessions,
        } => run_backfill_cmd(projects, archive, state_file, min_user_turns, max_sessions).await,
    }
}

async fn run_backfill_cmd(
    projects: Option<PathBuf>,
    archive: Option<PathBuf>,
    state_file: Option<PathBuf>,
    min_user_turns: usize,
    max_sessions: Option<usize>,
) -> Result<()> {
    let home = dirs::home_dir().context("no home directory")?;
    let projects_dir = projects.unwrap_or_else(|| home.join(".claude/projects"));
    let state_file = state_file.unwrap_or_else(|| {
        dirs::cache_dir()
            .unwrap_or_else(|| home.join(".cache"))
            .join("claude-memory/backfill-processed.txt")
    });
    backfill::run_backfill(
        &projects_dir,
        archive.as_deref(),
        &state_file,
        min_user_turns,
        max_sessions,
    )
    .await
}

async fn run_memory_list(
    limit: usize,
    offset: Option<u64>,
    substring: Option<String>,
    query: Option<String>,
) -> Result<()> {
    let memories =
        memory_unit::list(limit, offset, substring.as_deref(), query.as_deref()).await?;
    if memories.is_empty() {
        println!("(no memory units stored)");
        return Ok(());
    }
    for m in memories {
        let preview = if m.text.len() > 200 {
            format!("{}…", &m.text[..200])
        } else {
            m.text.clone()
        };
        println!(
            "{}  seen={}  src={}#{}  {}\n  {}\n",
            m.id, m.seen_count, m.source_session, m.source_turn, m.created_at, preview
        );
    }
    Ok(())
}

async fn run_memory_delete(id: u64) -> Result<()> {
    memory_unit::delete(id).await?;
    println!("deleted memory unit {id}");
    Ok(())
}

async fn run_memory_write(text: String) -> Result<()> {
    use claude_memory::embed::Embedder;
    use claude_memory::memory_unit::{DedupOutcome, MemoryUnit, upsert_with_dedup};
    use chrono::Utc;
    use qdrant_client::Qdrant;

    let trimmed = text.trim();
    if trimmed.is_empty() {
        anyhow::bail!("memory text is empty");
    }
    let client = Qdrant::from_url(claude_memory::index::QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    memory_unit::ensure_memory_units_collection(&client).await?;
    let embedder = Embedder::new();
    let unit = MemoryUnit {
        text: trimmed.to_string(),
        created_at: Utc::now(),
        source_session: "manual".to_string(),
        source_turn: 0,
        seen_in_sessions: vec!["manual".to_string()],
    };
    match upsert_with_dedup(&client, &embedder, unit).await? {
        DedupOutcome::Inserted(uuid) => println!("inserted (uuid={uuid})"),
        DedupOutcome::Merged(_) => println!("merged with an existing similar memory"),
    }
    Ok(())
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
        if config::graph_enabled() {
            match graph::extract_and_store(text).await {
                Ok(n) => extracted += n,
                Err(e) => eprintln!("  entry {}: {e}", i + 1),
            }
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

    // Vector search across curated memories (prompts collection, source="memory")
    // and auto-extracted memory units (claude-memory-units collection)
    const MIN_SCORE: f32 = 0.65;
    let curated = match index::search_prompts(prompt, limit, Some("memory")).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("enrich: prompts search failed: {e:#}");
            Vec::new()
        }
    };
    let units = match memory_unit::search(prompt, limit).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("enrich: memory-units search failed: {e:#}");
            Vec::new()
        }
    };

    let curated_relevant: Vec<&index::SearchResult> =
        curated.iter().filter(|r| r.score >= MIN_SCORE).collect();
    if !curated_relevant.is_empty() {
        sections.push(format_memory_results(&curated_relevant));
    }

    let units_relevant: Vec<&index::SearchResult> = units
        .iter()
        .filter(|r| r.score >= MIN_SCORE)
        .filter(|r| {
            !curated_relevant
                .iter()
                .any(|c| c.text.trim() == r.text.trim())
        })
        .collect();
    if !units_relevant.is_empty() {
        sections.push(format_memory_unit_results(&units_relevant));
    }

    // Graph: extract entities from prompt, query relationships
    if config::graph_enabled() {
        let entities = graph::find_concepts(prompt).await;
        if !entities.is_empty() {
            if let Ok(related) = graph::query_related(&entities) {
                if !related.is_empty() {
                    let graph_text = format_graph_results(&related);
                    sections.push(graph_text);
                }
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

fn format_memory_unit_results(results: &[&index::SearchResult]) -> String {
    let mut out = String::from(
        "## Possibly-useful preloads (from prior sessions, may be stale or wrong; treat as hints, not facts)",
    );
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

async fn run_analyze(session_jsonl: &PathBuf) -> Result<()> {
    use analyze::AnalysisOutcome;

    println!("Analyzing: {}", session_jsonl.display());
    let outcomes = analyze::analyze_session(session_jsonl).await?;

    let mut no_friction = 0usize;
    let mut discarded = 0usize;
    let mut stored = 0usize;

    for outcome in &outcomes {
        match outcome {
            AnalysisOutcome::NoFriction { .. } => {
                no_friction += 1;
            }
            AnalysisOutcome::Discarded { turn, reason } => {
                discarded += 1;
                println!("[turn {turn}] DISCARDED: {reason}");
            }
            AnalysisOutcome::Stored {
                turn,
                unit,
                deduped,
            } => {
                stored += 1;
                let dedup_label = if *deduped {
                    " (merged with existing)"
                } else {
                    ""
                };
                println!("[turn {turn}] STORED{dedup_label}: {}", unit.text);
            }
        }
    }

    println!(
        "\nDone. Turns analysed: {}  |  no-friction: {no_friction}  |  discarded: {discarded}  |  stored: {stored}",
        outcomes.len()
    );
    Ok(())
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
