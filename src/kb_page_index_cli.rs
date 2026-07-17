use clap::Subcommand;
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
    },

    /// Print exact Markdown text for an indexed line range
    Content {
        /// Document source path
        doc: String,

        /// Inclusive line range like 4-8
        locator: String,

        /// Knowledge base directory
        #[arg(long)]
        kb: Option<PathBuf>,

        /// Index directory (default: ~/.cache/claude-memory/kb-page-index)
        #[arg(long)]
        index: Option<PathBuf>,
    },
}
