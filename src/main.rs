use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use claude_memory::index;
use std::io::Write;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

mod indexing_cmds;
mod kb_page_index_cli;
mod transcript_page_index_cli;
use indexing_cmds::{
    IndexCommandOptions, run_index_cmd, run_index_file_cmd, run_kb_page_index_build,
    run_kb_page_index_content, run_kb_page_index_query, run_page_index,
    run_transcript_page_index_content, run_transcript_page_index_document,
    run_transcript_page_index_query, run_transcript_page_index_structure,
};
use kb_page_index_cli::KbPageIndexCommand;
use transcript_page_index_cli::TranscriptPageIndexCommand;

#[cfg(test)]
#[path = "main_tests.rs"]
mod main_tests;

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

        /// Codex sessions directory (default: ~/.codex/sessions)
        #[arg(long)]
        codex_sessions: Option<PathBuf>,

        /// Codex archived sessions directory (default: ~/.codex/archived_sessions)
        #[arg(long)]
        codex_archive: Option<PathBuf>,

        /// Pi sessions directory (default: ~/.config/pi/agent/sessions)
        #[arg(long)]
        pi_sessions: Option<PathBuf>,

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

    /// Build local transcript outline PageIndex trees for Claude/Codex sessions
    #[command(name = "transcript-page-index")]
    TranscriptPageIndex {
        #[command(subcommand)]
        command: TranscriptPageIndexCommand,
    },

    /// Build or query the persistent KB PageIndex
    KbPageIndex {
        #[command(subcommand)]
        command: KbPageIndexCommand,
    },

    /// Search globally ranked prompts and answers
    Search {
        /// Query text
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Restrict search to prompts or answers
        #[arg(long = "type", value_enum)]
        target: Option<SearchTarget>,

        /// Output one JSON result object per line
        #[arg(long)]
        json: bool,
    },

    /// Enrich a prompt with memory context (for UserPromptSubmit hook)
    Enrich {
        /// Maximum memory results
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },

    /// Show collection statistics
    Stats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SearchTarget {
    Prompts,
    Answers,
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
    match command {
        Command::Index { .. } | Command::IndexFile { .. } => run_indexing_command(command).await,
        Command::TranscriptPageIndex { command } => {
            run_transcript_page_index_command(command).await
        }
        Command::KbPageIndex { command } => run_kb_page_index_command(command).await,
        Command::Search {
            query,
            limit,
            target,
            json,
        } => run_search(query, limit, target, json).await,
        Command::Enrich { limit } => claude_memory::enrich_cmd::run_enrich(limit).await,
        Command::Stats => index::show_stats().await,
    }
}

async fn run_transcript_page_index_command(command: TranscriptPageIndexCommand) -> Result<()> {
    match command {
        TranscriptPageIndexCommand::Build { .. } => run_transcript_page_index_build(command).await,
        command => run_transcript_page_index_lookup_command(command).await,
    }
}

async fn run_transcript_page_index_lookup_command(
    command: TranscriptPageIndexCommand,
) -> Result<()> {
    match command {
        TranscriptPageIndexCommand::Document { doc, index } => {
            run_transcript_page_index_document(&doc, index)
        }
        TranscriptPageIndexCommand::Structure { doc, index } => {
            run_transcript_page_index_structure(&doc, index)
        }
        TranscriptPageIndexCommand::Content {
            doc,
            locator,
            index,
        } => run_transcript_page_index_content(&doc, &locator, index),
        TranscriptPageIndexCommand::Query {
            query,
            limit,
            index,
        } => run_transcript_page_index_query(&query, limit, index),
        TranscriptPageIndexCommand::Build { .. } => {
            unreachable!("build command passed to run_transcript_page_index_lookup_command")
        }
    }
}

async fn run_transcript_page_index_build(command: TranscriptPageIndexCommand) -> Result<()> {
    let TranscriptPageIndexCommand::Build {
        projects,
        archive,
        codex_sessions,
        codex_archive,
        output,
        max_sessions,
    } = command
    else {
        unreachable!("non-build command passed to run_transcript_page_index_build");
    };

    run_page_index(
        projects,
        archive,
        codex_sessions,
        codex_archive,
        output,
        max_sessions,
    )
    .await
}

async fn run_kb_page_index_command(command: KbPageIndexCommand) -> Result<()> {
    match command {
        KbPageIndexCommand::Build { kb, output } => run_kb_page_index_build(kb, output),
        KbPageIndexCommand::Query {
            query,
            limit,
            kb,
            index,
        } => run_kb_page_index_query(&query, limit, kb, index),
        KbPageIndexCommand::Content {
            doc,
            locator,
            kb,
            index,
        } => run_kb_page_index_content(&doc, &locator, kb, index),
    }
}

async fn run_indexing_command(command: Command) -> Result<()> {
    match command {
        Command::Index {
            archive,
            projects,
            codex_sessions,
            codex_archive,
            pi_sessions,
            batch_size,
            fresh,
            delay_ms,
        } => {
            run_index_cmd(IndexCommandOptions {
                archive,
                projects,
                codex_sessions,
                codex_archive,
                pi_sessions,
                batch_size,
                fresh,
                delay_ms,
            })
            .await
        }
        Command::IndexFile { path, batch_size } => run_index_file_cmd(&path, batch_size).await,
        _ => unreachable!("non-indexing command passed to run_indexing_command"),
    }
}

async fn run_search(
    query: String,
    limit: usize,
    target: Option<SearchTarget>,
    json: bool,
) -> Result<()> {
    let results = match target {
        Some(SearchTarget::Prompts) => index::search_prompts(&query, limit, None).await?,
        Some(SearchTarget::Answers) => index::search_answers(&query, limit, None).await?,
        None => index::search_all(&query, limit, None).await?,
    };

    if json {
        let stdout = std::io::stdout();
        return write_results_json(&results, &mut stdout.lock());
    }
    print_results(&results)
}

fn write_results_json(results: &[index::SearchResult], output: &mut impl Write) -> Result<()> {
    for result in results {
        serde_json::to_writer(&mut *output, result)?;
        writeln!(output)?;
    }
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
