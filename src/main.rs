use anyhow::{Context, Result};
use claude_memory::{graph, index, llm};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

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
    BuildGraph,

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
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Index { archive, projects, kb, batch_size, fresh, delay_ms } => {
            run_index_cmd(archive, projects, kb, batch_size, fresh, delay_ms).await
        }
        Command::IndexFile { path, batch_size } => run_index_file_cmd(&path, batch_size).await,
        Command::SearchPrompts { query, limit, source } => {
            print_results(&index::search_prompts(&query, limit, source.as_deref()).await?)
        }
        Command::SearchAnswers { query, limit, source } => {
            print_results(&index::search_answers(&query, limit, source.as_deref()).await?)
        }
        Command::Deduplicate { threshold, dry_run } => run_deduplicate(threshold, dry_run).await,
        Command::BuildGraph => run_build_graph().await,
        Command::Enrich { limit } => run_enrich(limit).await,
        Command::GraphDump { limit } => run_graph_dump(limit),
        Command::Stats => index::show_stats().await,
    }
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
    index::run_index(&archive_dir, &projects_dir, &kb_dir, batch_size, fresh, delay_ms).await
}

async fn run_index_file_cmd(path: &PathBuf, batch_size: usize) -> Result<()> {
    let count = index::index_file(path, batch_size).await?;
    eprintln!("Indexed {} chunks from {}", count, path.display());
    Ok(())
}

fn print_results(results: &[index::SearchResult]) -> Result<()> {
    for (i, result) in results.iter().enumerate() {
        println!("{}. [{}] {} (score: {:.3})", i + 1, result.source, result.path, result.score);
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

// --- Deduplicate command ---

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

// --- Build graph command ---

async fn run_build_graph() -> Result<()> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        anyhow::bail!("ANTHROPIC_API_KEY must be set for graph building (needs LLM to extract)");
    }
    let entries = load_all_memories().await?;
    eprintln!("Processing {} memory entries for graph extraction", entries.len());
    let mut extracted = 0;
    for (i, entry) in entries.iter().enumerate() {
        match graph::extract_and_store(&entry.text).await {
            Ok(n) => extracted += n,
            Err(e) => eprintln!("  entry {}: {e}", i + 1),
        }
        eprint!("\r  {}/{} ({} triplets)", i + 1, entries.len(), extracted);
    }
    eprintln!("\nDone: extracted {} triplets from {} entries", extracted, entries.len());
    Ok(())
}

// --- Graph dump command ---

fn run_graph_dump(limit: usize) -> Result<()> {
    let db = graph::get_graph()?;
    let entities = db.run_script(
        &format!("?[name, type] := *entities{{name, entity_type: type}} :limit {limit}"),
        std::collections::BTreeMap::new(),
        cozo::ScriptMutability::Immutable,
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("=== Entities ({} shown) ===", entities.rows.len());
    for row in &entities.rows {
        let name = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let etype = row.get(1).and_then(|v| v.get_str()).unwrap_or("");
        println!("  {name} [{etype}]");
    }

    let rels = db.run_script(
        &format!("?[src, rel, dst] := *relationships{{src, relation: rel, dst}} :limit {limit}"),
        std::collections::BTreeMap::new(),
        cozo::ScriptMutability::Immutable,
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        let text = if text.len() > 300 { format!("{}...", &text[..300]) } else { text };
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

// --- Memory entry loading ---

struct MemoryEntry {
    id: u64,
    text: String,
    category: String,
    project: String,
}

async fn load_all_memories() -> Result<Vec<MemoryEntry>> {
    use claude_memory::qdrant_hybrid::ensure_hybrid_collection;
    use qdrant_client::qdrant::{Condition, Filter};

    let client = qdrant_client::Qdrant::from_url("http://localhost:6334")
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, "claude-memory").await?;

    let filter = Filter::must([Condition::matches("source", "memory".to_string())]);
    scroll_memory_entries(&client, filter).await
}

async fn scroll_memory_entries(
    client: &qdrant_client::Qdrant,
    filter: qdrant_client::qdrant::Filter,
) -> Result<Vec<MemoryEntry>> {
    use qdrant_client::qdrant::ScrollPointsBuilder;

    let mut entries = Vec::new();
    let mut offset: Option<qdrant_client::qdrant::PointId> = None;
    loop {
        let mut scroll = ScrollPointsBuilder::new("claude-memory")
            .limit(100)
            .with_payload(true)
            .filter(filter.clone());
        if let Some(off) = offset {
            scroll = scroll.offset(off);
        }
        let result = client.scroll(scroll).await.context("scroll failed")?;
        for point in &result.result {
            entries.push(point_to_entry(point));
        }
        offset = result.next_page_offset;
        if offset.is_none() { break; }
    }
    Ok(entries)
}

fn point_to_entry(point: &qdrant_client::qdrant::RetrievedPoint) -> MemoryEntry {
    let id = point.id.as_ref().and_then(|p| match &p.point_id_options {
        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => Some(*n),
        _ => None,
    }).unwrap_or(0);
    MemoryEntry {
        id,
        text: get_payload(&point.payload, "text"),
        category: get_payload(&point.payload, "category"),
        project: get_payload(&point.payload, "project"),
    }
}

fn get_payload(
    payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> String {
    payload.get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

// --- Clustering ---

async fn cluster_similar(entries: &[MemoryEntry], threshold: f32) -> Result<Vec<Vec<usize>>> {
    let embedder = claude_memory::embed::Embedder::new();
    eprintln!("Embedding {} entries...", entries.len());
    let texts: Vec<&str> = entries.iter().map(|e| e.text.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts).await?;
    eprintln!("Clustering...");
    Ok(greedy_cluster(&embeddings, threshold))
}

fn greedy_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let mut assigned = vec![false; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        if assigned[i] { continue; }
        let mut cluster = vec![i];
        assigned[i] = true;
        for j in (i + 1)..n {
            if assigned[j] { continue; }
            if cosine_sim(&embeddings[i], &embeddings[j]) >= threshold {
                cluster.push(j);
                assigned[j] = true;
            }
        }
        clusters.push(cluster);
        if (i + 1) % 50 == 0 || i + 1 == n {
            eprint!("\r  Clustering: {}/{} entries processed", i + 1, n);
        }
    }
    eprintln!();
    clusters
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

fn print_clusters(entries: &[MemoryEntry], clusters: &[Vec<usize>]) {
    for cluster in clusters {
        if cluster.len() < 2 { continue; }
        println!("--- Cluster ({} entries) ---", cluster.len());
        for &idx in cluster {
            let e = &entries[idx];
            let preview = e.text.replace('\n', " ");
            let preview = if preview.len() > 100 { format!("{}...", &preview[..100]) } else { preview };
            println!("  [id={}] {}", e.id, preview);
        }
        println!();
    }
}

// --- Merge clusters ---

async fn merge_clusters(entries: &[MemoryEntry], clusters: &[Vec<usize>]) -> Result<()> {
    let client = qdrant_client::Qdrant::from_url("http://localhost:6334")
        .build()
        .context("failed to connect to Qdrant")?;
    let embedder = claude_memory::embed::Embedder::new();
    let merge_total = clusters.iter().filter(|c| c.len() > 1).count();
    let mut merged_count = 0u32;
    let mut failed_count = 0u32;
    let mut deleted_count = 0u32;

    for cluster in clusters {
        if cluster.len() < 2 { continue; }
        let preview = preview_text(&entries[cluster[0]].text, 60);
        eprintln!("\n  Cluster {}/{} ({} entries): {}", merged_count + failed_count + 1, merge_total, cluster.len(), preview);
        for &idx in &cluster[1..] {
            eprintln!("    + {}", preview_text(&entries[idx].text, 60));
        }
        let Some(text) = merge_cluster_texts(entries, cluster).await else {
            failed_count += 1;
            eprintln!("    FAILED: LLM merge returned no result");
            continue;
        };
        upsert_merged(&client, &embedder, &entries[cluster[0]], &text).await?;
        deleted_count += delete_cluster_extras(&client, entries, cluster).await?;
        merged_count += 1;
        eprintln!("    OK: merged into {} chars, deleted {} dupes", text.len(), cluster.len() - 1);
    }
    eprintln!("\nDone: {merged_count} merged, {failed_count} failed, {deleted_count} duplicates removed");
    Ok(())
}

async fn upsert_merged(
    client: &qdrant_client::Qdrant,
    embedder: &claude_memory::embed::Embedder,
    keep: &MemoryEntry,
    text: &str,
) -> Result<()> {
    use claude_memory::qdrant_hybrid::build_named_vectors;
    use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};

    let embedding = embedder.embed(text).await?;
    let named = build_named_vectors(embedding, text);
    let payload = build_merged_payload(text, &keep.category, &keep.project);
    let point = PointStruct::new(keep.id, named, payload);
    client.upsert_points(UpsertPointsBuilder::new("claude-memory", vec![point]))
        .await.context("upsert failed")?;
    Ok(())
}

fn build_merged_payload(
    text: &str,
    category: &str,
    project: &str,
) -> std::collections::HashMap<String, qdrant_client::qdrant::Value> {
    [
        ("text", text.to_string().into()),
        ("source", "memory".to_string().into()),
        ("path", format!("daily/{}", chrono::Local::now().format("%Y-%m-%d.md")).into()),
        ("category", category.to_string().into()),
        ("project", project.to_string().into()),
        ("hash", claude_memory::chunk::hash_text(text).into()),
    ].into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

async fn delete_cluster_extras(
    client: &qdrant_client::Qdrant,
    entries: &[MemoryEntry],
    cluster: &[usize],
) -> Result<u32> {
    use qdrant_client::qdrant::DeletePointsBuilder;

    let ids: Vec<u64> = cluster[1..].iter().map(|&idx| entries[idx].id).collect();
    if ids.is_empty() { return Ok(0); }
    let count = ids.len() as u32;
    let point_ids: Vec<qdrant_client::qdrant::PointId> = ids.iter().map(|&id| id.into()).collect();
    client.delete_points(DeletePointsBuilder::new("claude-memory").points(point_ids))
        .await.context("delete failed")?;
    Ok(count)
}

/// Progressively merge texts in a cluster via LLM.
async fn merge_cluster_texts(entries: &[MemoryEntry], cluster: &[usize]) -> Option<String> {
    let mut result = entries[cluster[0]].text.clone();
    for (step, &idx) in cluster[1..].iter().enumerate() {
        eprint!("    merging {}/{}...", step + 1, cluster.len() - 1);
        match llm::merge_memories(&result, &entries[idx].text).await {
            Some(merged) => {
                eprintln!(" ok ({} chars)", merged.len());
                result = merged;
            }
            None => {
                eprintln!(" failed");
                return None;
            }
        }
    }
    Some(result)
}

fn preview_text(text: &str, max_len: usize) -> String {
    let oneline = text.replace('\n', " ");
    if oneline.len() > max_len {
        format!("{}...", &oneline[..max_len])
    } else {
        oneline
    }
}
