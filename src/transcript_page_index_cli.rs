use clap::Subcommand;
use claude_memory::page_index_agentic;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum TranscriptPageIndexCommand {
    /// Build the persistent transcript PageIndex
    Build {
        /// Projects directory (default: ~/.claude/projects)
        #[arg(long)]
        projects: Option<PathBuf>,

        /// Claude archive directory (default: ~/.claude/archive)
        #[arg(long)]
        archive: Option<PathBuf>,

        /// Codex sessions directory (default: ~/.codex/sessions)
        #[arg(long)]
        codex_sessions: Option<PathBuf>,

        /// Codex archived sessions directory (default: ~/.codex/archived_sessions)
        #[arg(long)]
        codex_archive: Option<PathBuf>,

        /// Output directory (default: ~/.cache/claude-memory/transcript-page-index)
        #[arg(long)]
        output: Option<PathBuf>,

        /// Stop after indexing this many sessions
        #[arg(long)]
        max_sessions: Option<usize>,
    },

    /// Print transcript document metadata
    Document {
        /// Document id or indexed JSON path
        doc: String,

        /// Index directory (default: ~/.cache/claude-memory/transcript-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },

    /// Print transcript structure without node text
    Structure {
        /// Document id or indexed JSON path
        doc: String,

        /// Index directory (default: ~/.cache/claude-memory/transcript-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },

    /// Print exact transcript text for a node id or turn range
    Content {
        /// Document id or indexed JSON path
        doc: String,

        /// Node id or inclusive turn range like 4-8
        locator: String,

        /// Index directory (default: ~/.cache/claude-memory/transcript-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },

    /// Query transcript PageIndex nodes and print traceable hits
    Query {
        /// Query text
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Index directory (default: ~/.cache/claude-memory/transcript-page-index)
        #[arg(long)]
        index: Option<PathBuf>,

        /// Retrieval mode
        #[arg(long, value_enum, default_value_t = page_index_agentic::RetrievalMode::Lexical)]
        mode: page_index_agentic::RetrievalMode,
    },
}
