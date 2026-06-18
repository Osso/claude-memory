//! Transcript-mining analyzer: friction classification, candidate extraction,
//! replay simulation, dual judging, and session-level orchestration.

use anyhow::{Context, Result};
use chrono::Utc;
use qdrant_client::Qdrant;
use serde::Deserialize;
use std::path::Path;

use crate::embed::Embedder;
use crate::extract::Turn;
use crate::extract::read_session_turns;
use crate::llm;
use crate::memory_unit::{DedupOutcome, MemoryUnit, upsert_with_dedup};
use crate::notable_fact::{NotableFactIngestSummary, ingest_session_notable_facts};

// ── Data types ────────────────────────────────────────────────────────────────

pub struct FrictionFlag {
    pub flagged: bool,
    pub reason: String,
}

pub struct JudgeResult {
    pub passed: bool,
    pub reason: String,
}

pub enum AnalysisOutcome {
    NotableFacts {
        facts: usize,
        inserted: usize,
        merged: usize,
    },
    NoFriction {
        turn: u32,
    },
    Discarded {
        turn: u32,
        reason: String,
    },
    Stored {
        turn: u32,
        unit: MemoryUnit,
        deduped: bool,
    },
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct FlaggedJson {
    flagged: bool,
    reason: String,
}

#[derive(Deserialize)]
struct PreloadJson {
    preload: Option<String>,
}

#[derive(Deserialize)]
struct JudgeJson {
    passed: bool,
    reason: String,
}

/// Strip markdown fences and extract the first JSON object from raw LLM output.
fn extract_json(text: &str) -> &str {
    let stripped = strip_markdown_json_fence(text.trim());
    slice_first_json_object(stripped)
}

fn strip_markdown_json_fence(text: &str) -> &str {
    let text = if let Some(inner) = text.strip_prefix("```json") {
        inner.trim_start()
    } else if let Some(inner) = text.strip_prefix("```") {
        inner.trim_start()
    } else {
        text
    };

    if let Some(inner) = text.strip_suffix("```") {
        inner.trim_end()
    } else {
        text
    }
}

fn slice_first_json_object(text: &str) -> &str {
    let start = text.find('{').unwrap_or(0);
    let end = text.rfind('}').map(|i| i + 1).unwrap_or(text.len());
    &text[start..end]
}

// ── Turn context helpers ──────────────────────────────────────────────────────

fn format_turn(t: &Turn) -> String {
    let role = match t.role {
        crate::extract::Role::User => "User",
        crate::extract::Role::Assistant => "Assistant",
    };
    let body = match (t.text.trim().is_empty(), t.has_tool_use) {
        (true, true) => format!(
            "<{} tool call{}>",
            t.tool_call_count,
            if t.tool_call_count == 1 { "" } else { "s" }
        ),
        (false, true) => format!(
            "{}\n<{} tool call{}>",
            t.text,
            t.tool_call_count,
            if t.tool_call_count == 1 { "" } else { "s" }
        ),
        _ => t.text.clone(),
    };
    format!("[Turn {}] {}: {}", t.turn_index, role, body)
}

fn build_context_window(turns: &[Turn], target: u32, prior_n: usize) -> String {
    let mut selected: Vec<&Turn> = Vec::new();

    // Up to `prior_n` turns before target
    for t in turns.iter().rev() {
        if t.turn_index < target {
            selected.push(t);
            if selected.len() >= prior_n {
                break;
            }
        }
    }
    selected.reverse();

    // Target turn itself
    if let Some(t) = turns.iter().find(|t| t.turn_index == target) {
        selected.push(t);
    }

    // Next assistant turn (the response)
    if let Some(t) = turns
        .iter()
        .find(|t| t.turn_index > target && matches!(t.role, crate::extract::Role::Assistant))
    {
        selected.push(t);
    }

    selected
        .into_iter()
        .map(format_turn)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn full_session_text(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(format_turn)
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Concatenate non-empty post-friction text. Many sessions resolve via multiple
/// assistant turns with tool-use intermixed; one turn's `text` may be empty even
/// when the resolution is plain. Falls back to user turns as last resort, since
/// the user's next message often acknowledges the resolution.
fn eventual_resolution(turns: &[Turn], friction_turn: u32) -> String {
    let window = 30u32;

    let assistant_texts: Vec<String> = turns
        .iter()
        .filter(|t| {
            t.turn_index > friction_turn
                && t.turn_index <= friction_turn + window
                && matches!(t.role, crate::extract::Role::Assistant)
                && !t.text.trim().is_empty()
        })
        .map(|t| t.text.clone())
        .collect();

    if !assistant_texts.is_empty() {
        return assistant_texts.join("\n\n");
    }

    turns
        .iter()
        .filter(|t| {
            t.turn_index > friction_turn
                && t.turn_index <= friction_turn + window
                && !t.text.trim().is_empty()
        })
        .map(format_turn)
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ── T3a: Friction classifier ──────────────────────────────────────────────────

const FRICTION_SYSTEM: &str = "\
You are a classifier for a Claude assistant transcript. \
Given a context window showing a user turn and the assistant's response (plus prior turns for context), \
decide whether the assistant struggled — e.g. ran many tool calls exploring for information, \
backtracked, repeated similar searches, said 'I couldn't find', or was corrected by the user. \
Respond ONLY with JSON: {\"flagged\": bool, \"reason\": \"...\"}. \
Keep reason under 30 words.";

pub async fn classify_friction(turns: &[Turn], target_turn_index: u32) -> Result<FrictionFlag> {
    let context = build_context_window(turns, target_turn_index, 2);
    let user_msg = format!(
        "Classify whether the assistant struggled at turn {target_turn_index}:\n\n{context}"
    );
    let raw = llm::complete(FRICTION_SYSTEM, &user_msg, 200, 90)
        .await
        .context("friction classifier LLM call failed")?;
    let json_str = extract_json(&raw);
    let parsed: FlaggedJson = serde_json::from_str(json_str)
        .with_context(|| format!("friction classifier JSON parse failed | raw: {raw}"))?;
    Ok(FrictionFlag {
        flagged: parsed.flagged,
        reason: parsed.reason,
    })
}

// ── T3b: Candidate extractor ──────────────────────────────────────────────────

const EXTRACTOR_SYSTEM: &str = r#"You extract operational preload memory from Claude transcripts.

Given a transcript where the assistant struggled, return the 1-3 sentence preload that would have let the assistant skip the struggle.
Prefer a compact, transferable fact or constraint drawn from earlier in the transcript when possible.
The preload is not a recipe, instruction, or preference. It is background knowledge a future assistant would want before answering.

Return null if no useful preload exists.
Respond ONLY with JSON: {"preload": "..." | null}."#;

pub async fn extract_candidate(
    full_session: &[Turn],
    friction_turn_index: u32,
    feedback: Option<&str>,
) -> Result<Option<String>> {
    let session_text = full_session_text(full_session);
    let feedback_clause = feedback
        .map(|f| format!("\n\nPrevious attempt failed: {f}. Adjust accordingly."))
        .unwrap_or_default();
    let user_msg = format!(
        "Session transcript:\n\n{session_text}\n\n\
        The assistant struggled at turn {friction_turn_index}. \
        Extract a 1-3 sentence preload that would have prevented the struggle.{feedback_clause}"
    );
    let raw = llm::complete(EXTRACTOR_SYSTEM, &user_msg, 300, 120)
        .await
        .context("extractor LLM call failed")?;
    let json_str = extract_json(&raw);
    let parsed: PreloadJson = serde_json::from_str(json_str)
        .with_context(|| format!("extractor JSON parse failed | raw: {raw}"))?;
    Ok(parsed.preload.filter(|s| !s.trim().is_empty()))
}

// ── T3c: Preload judge ──────────────────────────────────────────────────────────

// Single-pass judge: the model internally simulates a preload-only answer and grades it
// against the eventual resolution in one call, replacing the former replay + judge pair.
const PRELOAD_JUDGE_SYSTEM: &str = "\
You judge whether a preload (established background knowledge) would have let a fresh assistant \
answer a user prompt well enough to match how the real session eventually resolved.\n\n\
First, internally simulate the answer a fresh assistant would give to the user prompt using ONLY \
the preload as ground truth — no tool access, no conversation history — applying the preload \
directly rather than proposing investigation. Then judge that simulated answer against the \
eventual resolution.\n\n\
PASS if the simulated answer conveys information consistent with — and that would lead toward — \
the eventual resolution. Different stylistic framings, partial answers, or stating the relevant \
fact without taking an action all PASS as long as nothing said is wrong. Asking a clarifying \
question is NOT automatic failure if the preload's content is reflected in it.\n\n\
FAIL if the simulated answer would assert a fact that contradicts the resolution, take a wrong \
direction the resolution explicitly rejected, refuse or be empty (e.g. 'I don't know'), or if the \
preload is too vague to produce a substantive answer.\n\n\
Respond ONLY with JSON: {\"passed\": bool, \"reason\": \"...\"}. Keep reason under 40 words.";

pub async fn judge_candidate(
    preload: &str,
    original_user_prompt: &str,
    eventual_resolution: &str,
) -> Result<JudgeResult> {
    let user_msg = format!(
        "Preload:\n{preload}\n\nUser prompt:\n{original_user_prompt}\n\nEventual resolution:\n{eventual_resolution}"
    );
    let raw = llm::complete(PRELOAD_JUDGE_SYSTEM, &user_msg, 200, 120)
        .await
        .context("preload judge LLM call failed")?;
    let json_str = extract_json(&raw);
    let parsed: JudgeJson = serde_json::from_str(json_str)
        .with_context(|| format!("preload judge JSON parse failed | raw: {raw}"))?;
    Ok(JudgeResult {
        passed: parsed.passed,
        reason: parsed.reason,
    })
}

// ── T4: Orchestrator ──────────────────────────────────────────────────────────

const MAX_ITERATIONS: usize = 1;
const NULL_CANDIDATE_REASON: &str = "extractor returned null";

struct AnalysisSession {
    turns: Vec<Turn>,
    session_id: String,
    source: String,
    project_slug: Option<String>,
    client: Qdrant,
    embedder: Embedder,
}

enum CandidateAttempt {
    Stored(AnalysisOutcome),
    Retry(String),
    NullCandidate,
}

pub async fn analyze_session(session_path: &Path) -> Result<Vec<AnalysisOutcome>> {
    let session = load_analysis_session(session_path).await?;
    let mut outcomes = collect_session_notable_fact_outcomes(&session).await?;
    outcomes.extend(collect_user_turn_outcomes(&session).await?);
    Ok(outcomes)
}

async fn collect_session_notable_fact_outcomes(
    session: &AnalysisSession,
) -> Result<Vec<AnalysisOutcome>> {
    let summary = ingest_session_notable_facts(
        &session.client,
        &session.embedder,
        &session.turns,
        &session.source,
        &session.session_id,
        session.project_slug.as_deref(),
    )
    .await
    .context("notable fact ingestion failed")?;

    if summary.facts == 0 {
        return Ok(Vec::new());
    }

    eprintln!(
        "  notable facts: {} (inserted {}, merged {})",
        summary.facts, summary.inserted, summary.merged
    );
    Ok(vec![notable_fact_outcome(summary)])
}

fn notable_fact_outcome(summary: NotableFactIngestSummary) -> AnalysisOutcome {
    AnalysisOutcome::NotableFacts {
        facts: summary.facts,
        inserted: summary.inserted,
        merged: summary.merged,
    }
}

async fn collect_user_turn_outcomes(session: &AnalysisSession) -> Result<Vec<AnalysisOutcome>> {
    let mut outcomes = Vec::new();

    for turn in user_turns(&session.turns) {
        outcomes.push(analyze_user_turn(session, turn).await?);
    }

    Ok(outcomes)
}

async fn load_analysis_session(session_path: &Path) -> Result<AnalysisSession> {
    let turns = read_session_turns(session_path)
        .with_context(|| format!("failed to read turns from {}", session_path.display()))?;

    let session_id = session_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let source = memory_unit_source_from_path(session_path).to_string();
    let project_slug = project_slug_from_session_path(session_path);

    let client = Qdrant::from_url(crate::index::QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    crate::memory_unit::ensure_memory_units_collection(&client).await?;
    crate::notable_fact::ensure_notable_facts_collection(&client).await?;
    let embedder = Embedder::new();

    Ok(AnalysisSession {
        turns,
        session_id,
        source,
        project_slug,
        client,
        embedder,
    })
}

fn user_turns(turns: &[Turn]) -> impl Iterator<Item = &Turn> {
    turns
        .iter()
        .filter(|turn| matches!(turn.role, crate::extract::Role::User))
}

async fn analyze_user_turn(session: &AnalysisSession, turn: &Turn) -> Result<AnalysisOutcome> {
    let turn_index = turn.turn_index;
    let friction = classify_friction(&session.turns, turn_index)
        .await
        .with_context(|| format!("classify_friction failed at turn {turn_index}"))?;

    if !friction.flagged {
        eprintln!("  [turn {turn_index}] no friction: {}", friction.reason);
        return Ok(AnalysisOutcome::NoFriction { turn: turn_index });
    }

    eprintln!(
        "  [turn {turn_index}] friction flagged: {}",
        friction.reason
    );

    let resolution = eventual_resolution(&session.turns, turn_index);
    analyze_flagged_turn(session, turn, &resolution).await
}

async fn analyze_flagged_turn(
    session: &AnalysisSession,
    turn: &Turn,
    resolution: &str,
) -> Result<AnalysisOutcome> {
    let mut feedback = None;
    let mut final_fail_reason = String::new();

    for attempt in 0..MAX_ITERATIONS {
        let candidate_attempt =
            validate_candidate_attempt(session, turn, resolution, feedback.as_deref(), attempt)
                .await?;

        if let Some(outcome) =
            handle_candidate_attempt(candidate_attempt, &mut feedback, &mut final_fail_reason)
        {
            return Ok(outcome);
        }

        if should_stop_candidate_retries(&final_fail_reason) {
            break;
        }
    }

    Ok(discarded_validation_outcome(
        turn.turn_index,
        &final_fail_reason,
    ))
}

fn handle_candidate_attempt(
    attempt: CandidateAttempt,
    feedback: &mut Option<String>,
    final_fail_reason: &mut String,
) -> Option<AnalysisOutcome> {
    match attempt {
        CandidateAttempt::Stored(outcome) => Some(outcome),
        CandidateAttempt::NullCandidate => {
            *final_fail_reason = NULL_CANDIDATE_REASON.to_string();
            None
        }
        CandidateAttempt::Retry(fail_reason) => {
            *final_fail_reason = fail_reason.clone();
            *feedback = Some(fail_reason);
            None
        }
    }
}

fn should_stop_candidate_retries(final_fail_reason: &str) -> bool {
    final_fail_reason == NULL_CANDIDATE_REASON
}

async fn validate_candidate_attempt(
    session: &AnalysisSession,
    turn: &Turn,
    resolution: &str,
    feedback: Option<&str>,
    attempt: usize,
) -> Result<CandidateAttempt> {
    let turn_index = turn.turn_index;
    let Some(candidate) = extract_candidate(&session.turns, turn_index, feedback)
        .await
        .with_context(|| format!("extract_candidate failed at turn {turn_index}"))?
    else {
        return Ok(CandidateAttempt::NullCandidate);
    };

    log_candidate_attempt(turn_index, attempt, &candidate);
    let correctness = judge_candidate(&candidate, &turn.text, resolution)
        .await
        .with_context(|| format!("judge_candidate failed at turn {turn_index}"))?;

    if correctness.passed {
        return Ok(CandidateAttempt::Stored(
            store_candidate(session, turn_index, candidate).await,
        ));
    }

    Ok(candidate_validation_failure(
        turn_index,
        attempt,
        correctness,
    ))
}

fn candidate_validation_failure(
    turn_index: u32,
    attempt: usize,
    correctness: JudgeResult,
) -> CandidateAttempt {
    let fail_reason = format!("correctness: {}", correctness.reason);
    eprintln!(
        "  [turn {turn_index}] attempt {} failed: {fail_reason}",
        attempt + 1
    );
    CandidateAttempt::Retry(fail_reason)
}

fn log_candidate_attempt(turn_index: u32, attempt: usize, candidate: &str) {
    eprintln!(
        "  [turn {turn_index}] attempt {}: candidate: {}...",
        attempt + 1,
        candidate.chars().take(60).collect::<String>()
    );
}

async fn store_candidate(
    session: &AnalysisSession,
    turn_index: u32,
    candidate: String,
) -> AnalysisOutcome {
    let unit = MemoryUnit {
        text: candidate,
        created_at: Utc::now(),
        source: session.source.clone(),
        source_session: session.session_id.clone(),
        source_turn: turn_index,
        category: None,
        project: None,
        seen_in_sessions: vec![session.session_id.clone()],
    };

    match upsert_with_dedup(&session.client, &session.embedder, unit.clone()).await {
        Ok(DedupOutcome::Inserted(_)) => stored_outcome(turn_index, unit, false),
        Ok(DedupOutcome::Merged(_)) => stored_outcome(turn_index, unit, true),
        Err(e) => {
            eprintln!("  [turn {turn_index}] upsert error: {e}");
            AnalysisOutcome::Discarded {
                turn: turn_index,
                reason: format!("upsert failed: {e}"),
            }
        }
    }
}

fn stored_outcome(turn: u32, unit: MemoryUnit, deduped: bool) -> AnalysisOutcome {
    AnalysisOutcome::Stored {
        turn,
        unit,
        deduped,
    }
}

fn discarded_validation_outcome(turn: u32, final_fail_reason: &str) -> AnalysisOutcome {
    let reason = if final_fail_reason.is_empty() {
        NULL_CANDIDATE_REASON.to_string()
    } else {
        format!("validation failed 3x: {final_fail_reason}")
    };
    AnalysisOutcome::Discarded { turn, reason }
}

fn memory_unit_source_from_path(session_path: &Path) -> &'static str {
    let path = session_path.to_string_lossy();
    if path.ends_with(".jsonl.zst") {
        "archive"
    } else {
        "session"
    }
}

fn project_slug_from_session_path(session_path: &Path) -> Option<String> {
    if session_path.to_string_lossy().ends_with(".jsonl.zst") {
        return archive_project_slug(session_path);
    }

    session_path
        .parent()
        .and_then(|parent| parent.file_name())
        .map(|name| slugify_project_component(&name.to_string_lossy()))
        .filter(|slug| !slug.is_empty())
}

fn archive_project_slug(session_path: &Path) -> Option<String> {
    let file_name = session_path.file_name()?.to_string_lossy();
    let prefix = file_name.strip_suffix(".jsonl.zst").unwrap_or(&file_name);
    let project = prefix.rsplit_once('_').map(|(project, _)| project)?;
    let slug = slugify_project_component(project);
    if slug.is_empty() { None } else { Some(slug) }
}

fn slugify_project_component(component: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;

    for ch in component.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_dash = false;
            continue;
        }

        if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }

    slug.trim_matches('-').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_unit_source_uses_session_for_live_jsonl() {
        assert_eq!(
            memory_unit_source_from_path(Path::new("/tmp/session.jsonl")),
            "session"
        );
    }

    #[test]
    fn memory_unit_source_uses_archive_for_zst_jsonl_archive() {
        assert_eq!(
            memory_unit_source_from_path(Path::new("/tmp/session.jsonl.zst")),
            "archive"
        );
    }

    #[test]
    fn project_slug_uses_live_session_parent_dir() {
        let path = Path::new(
            "/home/osso/.claude/projects/-syncthing-Sync-Projects-claude-memory/session.jsonl",
        );

        assert_eq!(
            project_slug_from_session_path(path).as_deref(),
            Some("syncthing-sync-projects-claude-memory")
        );
    }

    #[test]
    fn project_slug_uses_archive_prefix_before_session_id() {
        let path = Path::new(
            "/home/osso/.claude/archive/-home-osso-Projects-claude-memory_abc123.jsonl.zst",
        );

        assert_eq!(
            project_slug_from_session_path(path).as_deref(),
            Some("home-osso-projects-claude-memory")
        );
    }
}
