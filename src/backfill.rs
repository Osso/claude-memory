//! Backfill: walk live projects directory, run analyzer over each session,
//! persist resume state in a sidecar file.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::analyze::{AnalysisOutcome, analyze_session};
use crate::extract::{Role, read_session_turns};

/// Skip sessions whose mtime is newer than this — they're likely still active.
const ACTIVE_SKEW_SECS: u64 = 300;

/// Skip sessions with more user turns than this — analyzing 100s of turns
/// per session would dominate wall-clock time.
const MAX_USER_TURNS: usize = 100;

pub async fn run_backfill(
    projects_dir: &Path,
    archive_dir: Option<&Path>,
    state_file: &Path,
    min_user_turns: usize,
    max_sessions: Option<usize>,
) -> Result<()> {
    let processed = load_processed(state_file)?;
    let mut sessions = collect_sessions(projects_dir);
    let live_count = sessions.len();
    if let Some(adir) = archive_dir {
        let archive_sessions = collect_archive_sessions(adir);
        eprintln!(
            "Found {} live sessions in {} + {} archive sessions in {}",
            live_count,
            projects_dir.display(),
            archive_sessions.len(),
            adir.display()
        );
        sessions.extend(archive_sessions);
    } else {
        eprintln!(
            "Found {} sessions in {}",
            sessions.len(),
            projects_dir.display()
        );
    }
    eprintln!("Already processed: {}", processed.len());

    if let Some(parent) = state_file.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut state = OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_file)
        .with_context(|| format!("failed to open state file {}", state_file.display()))?;

    let pending = sessions.len() - processed.len();
    let mut count = 0usize;
    let mut total_stored = 0usize;
    let mut total_discarded = 0usize;
    let mut skipped_short = 0usize;
    let mut skipped_active = 0usize;

    for path in &sessions {
        let Some(session_id) = session_id_from_path(path) else {
            continue;
        };
        if processed.contains(&session_id) {
            continue;
        }

        if is_likely_active(path) {
            skipped_active += 1;
            eprintln!("  [{session_id}] skipping: recently modified (likely active)");
            continue;
        }

        let user_turn_count = match read_session_turns(path) {
            Ok(turns) => turns
                .iter()
                .filter(|t| matches!(t.role, Role::User))
                .count(),
            Err(e) => {
                eprintln!("  [{session_id}] read error: {e}");
                writeln!(state, "{session_id} read_error")?;
                continue;
            }
        };

        if user_turn_count < min_user_turns {
            skipped_short += 1;
            writeln!(state, "{session_id} short turns={user_turn_count}")?;
            continue;
        }
        if user_turn_count > MAX_USER_TURNS {
            eprintln!(
                "  [{session_id}] skipping: too long ({user_turn_count} user turns > {MAX_USER_TURNS})"
            );
            writeln!(state, "{session_id} too_long turns={user_turn_count}")?;
            continue;
        }

        count += 1;
        eprintln!(
            "\n[{}/{}] {} ({} user turns)",
            count, pending, session_id, user_turn_count
        );

        match analyze_session(path).await {
            Ok(outcomes) => {
                let stored = outcomes
                    .iter()
                    .filter(|o| matches!(o, AnalysisOutcome::Stored { .. }))
                    .count();
                let discarded = outcomes
                    .iter()
                    .filter(|o| matches!(o, AnalysisOutcome::Discarded { .. }))
                    .count();
                total_stored += stored;
                total_discarded += discarded;
                writeln!(
                    state,
                    "{session_id} ok stored={stored} discarded={discarded} turns={}",
                    outcomes.len()
                )?;
                state.flush().ok();
                eprintln!("  → stored: {stored}, discarded: {discarded}");
            }
            Err(e) => {
                eprintln!("  error: {e}");
                writeln!(state, "{session_id} error {e}")?;
                state.flush().ok();
            }
        }

        if let Some(max) = max_sessions
            && count >= max
        {
            eprintln!("\nReached max_sessions={max}");
            break;
        }
    }

    eprintln!(
        "\nBackfill done. Sessions analysed: {count} | stored: {total_stored} | discarded: {total_discarded} | skipped (too short): {skipped_short} | skipped (active): {skipped_active}"
    );
    Ok(())
}

fn is_likely_active(path: &Path) -> bool {
    let Ok(modified) = path.metadata().and_then(|m| m.modified()) else {
        return false;
    };
    let Ok(elapsed) = modified.elapsed() else {
        return false;
    };
    elapsed.as_secs() < ACTIVE_SKEW_SECS
}

fn collect_sessions(projects_dir: &Path) -> Vec<PathBuf> {
    if !projects_dir.exists() {
        return vec![];
    }
    let projects_canon = projects_dir
        .canonicalize()
        .unwrap_or_else(|_| projects_dir.to_path_buf());
    let mut sessions: Vec<PathBuf> = WalkDir::new(&projects_canon)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "jsonl")
                .unwrap_or(false)
        })
        .filter(|e| {
            // Direct child of an encoded-cwd dir: <projects>/<cwd>/<session>.jsonl
            e.path()
                .parent()
                .and_then(|p| p.parent())
                .map(|gp| gp == projects_canon)
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    sessions.sort_by_key(|p| std::cmp::Reverse(p.metadata().and_then(|m| m.modified()).ok()));
    sessions
}

fn collect_archive_sessions(archive_dir: &Path) -> Vec<PathBuf> {
    if !archive_dir.exists() {
        return vec![];
    }
    let mut sessions: Vec<PathBuf> = WalkDir::new(archive_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .to_str()
                .map(|s| s.ends_with(".jsonl.zst"))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    sessions.sort_by_key(|p| std::cmp::Reverse(p.metadata().and_then(|m| m.modified()).ok()));
    sessions
}

fn session_id_from_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    // archive: "<encoded-cwd>_<session-uuid>.jsonl.zst" → take part after final '_'
    if let Some(stripped) = name.strip_suffix(".jsonl.zst") {
        return stripped
            .rsplit_once('_')
            .map(|(_, id)| id.to_string())
            .or_else(|| Some(stripped.to_string()));
    }
    // live: "<session-uuid>.jsonl"
    if let Some(stripped) = name.strip_suffix(".jsonl") {
        return Some(stripped.to_string());
    }
    path.file_stem().map(|s| s.to_string_lossy().to_string())
}

fn load_processed(state_file: &Path) -> Result<HashSet<String>> {
    if !state_file.exists() {
        return Ok(HashSet::new());
    }
    let content = std::fs::read_to_string(state_file)
        .with_context(|| format!("failed to read state file {}", state_file.display()))?;
    Ok(content
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(|s| s.to_string()))
        .collect())
}
