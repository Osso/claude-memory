use anyhow::Result;
use claude_memory::index;
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

    /// Show collection statistics
    Stats,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let home = dirs::home_dir().expect("no home directory");

    match cli.command {
        Command::Index {
            archive,
            projects,
            kb,
            batch_size,
            fresh,
        } => {
            let archive_dir = archive.unwrap_or_else(|| home.join(".claude/archive"));
            let projects_dir = projects.unwrap_or_else(|| home.join(".claude/projects"));
            let kb_dir = kb.unwrap_or_else(|| PathBuf::from("/syncthing/Sync/KB"));

            index::run_index(&archive_dir, &projects_dir, &kb_dir, batch_size, fresh).await?;
        }

        Command::IndexFile { path, batch_size } => {
            let count = index::index_file(&path, batch_size).await?;
            eprintln!("Indexed {} chunks from {}", count, path.display());
        }

        Command::SearchPrompts {
            query,
            limit,
            source,
        } => {
            let results = index::search_prompts(&query, limit, source.as_deref()).await?;
            for (i, result) in results.iter().enumerate() {
                println!(
                    "{}. [{}] {} (score: {:.3})",
                    i + 1,
                    result.source,
                    result.path,
                    result.score
                );
                println!("   {}", truncate(&result.text, 200));
                println!();
            }
        }

        Command::SearchAnswers {
            query,
            limit,
            source,
        } => {
            let results = index::search_answers(&query, limit, source.as_deref()).await?;
            for (i, result) in results.iter().enumerate() {
                println!(
                    "{}. [{}] {} (score: {:.3})",
                    i + 1,
                    result.source,
                    result.path,
                    result.score
                );
                println!("   {}", truncate(&result.text, 200));
                println!();
            }
        }

        Command::Stats => {
            index::show_stats().await?;
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        format!("{}...", &s[..max])
    }
}
