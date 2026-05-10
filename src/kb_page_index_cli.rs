use clap::Subcommand;
use claude_memory::page_index_agentic;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum KbPageIndexCommand {
    /// Build the persistent KB PageIndex
    Build {
        /// Knowledge base directory (default: /syncthing/Sync/KB)
        #[arg(long)]
        kb: Option<PathBuf>,

        /// Output directory (default: ~/.cache/claude-memory/kb-page-index)
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Query the persistent KB PageIndex
    Query {
        /// Query text
        query: String,

        /// Maximum results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Knowledge base directory used for stale-index checks
        #[arg(long)]
        kb: Option<PathBuf>,

        /// Index directory (default: ~/.cache/claude-memory/kb-page-index)
        #[arg(long)]
        index: Option<PathBuf>,

        /// Retrieval mode
        #[arg(long, value_enum, default_value_t = page_index_agentic::RetrievalMode::Lexical)]
        mode: page_index_agentic::RetrievalMode,
    },

    /// Print document metadata from the persistent KB PageIndex
    Document {
        /// Document id or source path
        doc: String,

        /// Index directory (default: ~/.cache/claude-memory/kb-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },

    /// Print nested document structure without node text
    Structure {
        /// Document id or source path
        doc: String,

        /// Index directory (default: ~/.cache/claude-memory/kb-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },

    /// Print exact indexed text for a node id or line range
    Content {
        /// Document id or source path
        doc: String,

        /// Node id or inclusive line range like 4-8
        locator: String,

        /// Index directory (default: ~/.cache/claude-memory/kb-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },
}
