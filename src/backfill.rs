//! Backfill: walk live projects directory, run analyzer over each session,
//! persist resume state in a sidecar file.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
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

#[derive(Default)]
struct BackfillTotals {
    analyzed: usize,
    stored: usize,
    discarded: usize,
    skipped_short: usize,
    skipped_active: usize,
}

struct SessionStats {
    notable_facts: usize,
    stored: usize,
    discarded: usize,
    outcomes: usize,
}

enum SessionStep {
    Analyzed(SessionStats),
    SkippedShort,
    SkippedActive,
    Ignored,
}

pub async fn run_backfill(
    projects_dir: &Path,
    archive_dir: Option<&Path>,
    state_file: &Path,
    min_user_turns: usize,
    max_sessions: Option<usize>,
) -> Result<()> {
    let processed = load_processed(state_file)?;
    let sessions = collect_all_sessions(projects_dir, archive_dir);
    eprintln!("Already processed: {}", processed.len());

    let mut state = open_state_file(state_file)?;
    let pending = sessions.len() - processed.len();
    let mut totals = BackfillTotals::default();

    for path in &sessions {
        let display_index = totals.analyzed + 1;
        update_backfill_totals(
            &mut totals,
            process_session(
                path,
                &processed,
                &mut state,
                min_user_turns,
                pending,
                display_index,
            )
            .await?,
        );

        if let Some(max) = max_sessions
            && totals.analyzed >= max
        {
            eprintln!("\nReached max_sessions={max}");
            break;
        }
    }

    eprintln!(
        "\nBackfill done. Sessions analysed: {} | stored: {} | discarded: {} | skipped (too short): {} | skipped (active): {}",
        totals.analyzed,
        totals.stored,
        totals.discarded,
        totals.skipped_short,
        totals.skipped_active
    );
    Ok(())
}

fn collect_all_sessions(projects_dir: &Path, archive_dir: Option<&Path>) -> Vec<PathBuf> {
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

    sessions
}

fn open_state_file(state_file: &Path) -> Result<File> {
    if let Some(parent) = state_file.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_file)
        .with_context(|| format!("failed to open state file {}", state_file.display()))
}

async fn process_session(
    path: &Path,
    processed: &HashSet<String>,
    state: &mut File,
    min_user_turns: usize,
    pending: usize,
    display_index: usize,
) -> Result<SessionStep> {
    let Some(session_id) = session_id_from_path(path) else {
        return Ok(SessionStep::Ignored);
    };
    if processed.contains(&session_id) {
        return Ok(SessionStep::Ignored);
    }
    if is_likely_active(path) {
        eprintln!("  [{session_id}] skipping: recently modified (likely active)");
        return Ok(SessionStep::SkippedActive);
    }

    let Some(user_turn_count) = count_user_turns(path, state, &session_id)? else {
        return Ok(SessionStep::Ignored);
    };
    if should_skip_short_session(user_turn_count, min_user_turns, state, &session_id)? {
        return Ok(SessionStep::SkippedShort);
    }
    if should_skip_long_session(user_turn_count, state, &session_id)? {
        return Ok(SessionStep::Ignored);
    }

    eprintln!(
        "\n[{}/{}] {} ({} user turns)",
        display_index, pending, session_id, user_turn_count
    );
    analyze_and_record_session(path, state, &session_id).await
}

fn count_user_turns(path: &Path, state: &mut File, session_id: &str) -> Result<Option<usize>> {
    match read_session_turns(path) {
        Ok(turns) => Ok(Some(
            turns
                .iter()
                .filter(|turn| matches!(turn.role, Role::User))
                .count(),
        )),
        Err(error) => {
            eprintln!("  [{session_id}] read error: {error}");
            writeln!(state, "{session_id} read_error")?;
            Ok(None)
        }
    }
}

fn should_skip_short_session(
    user_turn_count: usize,
    min_user_turns: usize,
    state: &mut File,
    session_id: &str,
) -> Result<bool> {
    if user_turn_count >= min_user_turns {
        return Ok(false);
    }

    writeln!(state, "{session_id} short turns={user_turn_count}")?;
    Ok(true)
}

fn should_skip_long_session(
    user_turn_count: usize,
    state: &mut File,
    session_id: &str,
) -> Result<bool> {
    if user_turn_count <= MAX_USER_TURNS {
        return Ok(false);
    }

    eprintln!(
        "  [{session_id}] skipping: too long ({user_turn_count} user turns > {MAX_USER_TURNS})"
    );
    writeln!(state, "{session_id} too_long turns={user_turn_count}")?;
    Ok(true)
}

async fn analyze_and_record_session(
    path: &Path,
    state: &mut File,
    session_id: &str,
) -> Result<SessionStep> {
    match analyze_session(path).await {
        Ok(outcomes) => {
            let stats = session_stats(&outcomes);
            writeln!(
                state,
                "{session_id} ok facts={} stored={} discarded={} turns={}",
                stats.notable_facts, stats.stored, stats.discarded, stats.outcomes
            )?;
            state.flush().ok();
            eprintln!(
                "  → notable facts: {}, stored: {}, discarded: {}",
                stats.notable_facts, stats.stored, stats.discarded
            );
            Ok(SessionStep::Analyzed(stats))
        }
        Err(error) => {
            eprintln!("  error: {error}");
            writeln!(state, "{session_id} error {error}")?;
            state.flush().ok();
            Ok(SessionStep::Ignored)
        }
    }
}

fn session_stats(outcomes: &[AnalysisOutcome]) -> SessionStats {
    SessionStats {
        notable_facts: notable_fact_count(outcomes),
        stored: stored_count(outcomes),
        discarded: discarded_count(outcomes),
        outcomes: outcomes.len(),
    }
}

fn notable_fact_count(outcomes: &[AnalysisOutcome]) -> usize {
    outcomes
        .iter()
        .map(|outcome| match outcome {
            AnalysisOutcome::NotableFacts { facts, .. } => *facts,
            _ => 0,
        })
        .sum()
}

fn stored_count(outcomes: &[AnalysisOutcome]) -> usize {
    outcomes
        .iter()
        .filter(|outcome| matches!(outcome, AnalysisOutcome::Stored { .. }))
        .count()
}

fn discarded_count(outcomes: &[AnalysisOutcome]) -> usize {
    outcomes
        .iter()
        .filter(|outcome| matches!(outcome, AnalysisOutcome::Discarded { .. }))
        .count()
}

fn update_backfill_totals(totals: &mut BackfillTotals, step: SessionStep) {
    match step {
        SessionStep::Analyzed(stats) => {
            totals.analyzed += 1;
            totals.stored += stats.stored;
            totals.discarded += stats.discarded;
        }
        SessionStep::SkippedShort => totals.skipped_short += 1,
        SessionStep::SkippedActive => totals.skipped_active += 1,
        SessionStep::Ignored => {}
    }
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
