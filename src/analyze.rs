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
    let s = text.trim();
    // Strip ```json ... ``` fences
    let s = if let Some(inner) = s.strip_prefix("```json") {
        inner.trim_start()
    } else if let Some(inner) = s.strip_prefix("```") {
        inner.trim_start()
    } else {
        s
    };
    let s = if let Some(inner) = s.strip_suffix("```") {
        inner.trim_end()
    } else {
        s
    };
    // Find first `{`
    let start = s.find('{').unwrap_or(0);
    let end = s.rfind('}').map(|i| i + 1).unwrap_or(s.len());
    &s[start..end]
}

// ── Turn context helpers ──────────────────────────────────────────────────────

fn format_turn(t: &Turn) -> String {
    let role = match t.role {
        crate::extract::Role::User => "User",
        crate::extract::Role::Assistant => "Assistant",
    };
    format!("[Turn {}] {}: {}", t.turn_index, role, t.text)
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
        .iter()
        .map(|t| format_turn(t))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn full_session_text(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|t| format_turn(t))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Find the assistant text within ~10 turns after the friction turn.
fn eventual_resolution(turns: &[Turn], friction_turn: u32) -> String {
    turns
        .iter()
        .filter(|t| {
            t.turn_index > friction_turn
                && t.turn_index <= friction_turn + 20
                && matches!(t.role, crate::extract::Role::Assistant)
        })
        .last()
        .or_else(|| {
            turns.iter().find(|t| {
                t.turn_index > friction_turn && matches!(t.role, crate::extract::Role::Assistant)
            })
        })
        .map(|t| t.text.clone())
        .unwrap_or_default()
}

// ── T3a: Friction classifier ──────────────────────────────────────────────────

const FRICTION_SYSTEM: &str = "\
You are a classifier for a Claude assistant transcript. \
Given a context window showing a user turn and the assistant's response (plus prior turns for context), \
decide whether the assistant struggled — e.g. ran many tool calls exploring for information, \
backtracked, repeated similar searches, said 'I couldn't find', or was corrected by the user. \
Respond ONLY with JSON: {\"flagged\": bool, \"reason\": \"...\"}. \
Keep reason under 30 words.";

pub async fn classify_friction(turns: &[Turn], target_turn_index: u32) -> Option<FrictionFlag> {
    let context = build_context_window(turns, target_turn_index, 2);
    let user_msg = format!(
        "Classify whether the assistant struggled at turn {target_turn_index}:\n\n{context}"
    );
    let raw = llm::complete(FRICTION_SYSTEM, &user_msg, 200, 15).await?;
    let json_str = extract_json(&raw);
    let parsed: FlaggedJson = serde_json::from_str(json_str)
        .map_err(|e| {
            eprintln!("  [friction] parse error: {e} | raw: {raw}");
        })
        .ok()?;
    Some(FrictionFlag {
        flagged: parsed.flagged,
        reason: parsed.reason,
    })
}

// ── T3b: Candidate extractor ──────────────────────────────────────────────────

const EXTRACTOR_SYSTEM: &str = "\
You extract DURABLE memory preloads from Claude session transcripts.\n\n\
A preload is a TECHNICAL FACT about how a tool, system, or environment behaves — knowledge that would apply to anyone, \
not just this user. The user's reaction in the transcript is just evidence; the real preload is the underlying mechanism that drove it.\n\n\
PRIORITY ORDER for what to extract:\n\n\
1. Tool-behavior facts (strongest — universal, non-subjective):\n\
   - \"`git absorb` uses blame-based target detection, so it fails when fixing up commits that contain newly-added lines (no blame history exists). Explicit `git commit --fixup=<sha>` works regardless.\"\n\
   - \"`pacman` does not manage AUR packages; AUR builds require an external helper.\"\n\n\
2. Configured artifacts that exist on this system:\n\
   - \"A `git fix <sha>` helper is documented in ~/.claude/rules/02-git.md (non-interactive autosquash).\"\n\
   - \"MCP server config lives at ~/.claude.json.\"\n\n\
3. Concrete environment / project facts:\n\
   - \"This is Arch Linux; package operations go through the `arch` CLI wrapper.\"\n\
   - \"Tests live in `tests/`, not co-located with `src/`.\"\n\n\
DO NOT WRITE preference-framed memories. If the user pushed back on something, find the TECHNICAL reason and write THAT. \
\"User prefers X\" is almost always wrong — the user's choice was driven by a constraint, find the constraint.\n\n\
BAD examples:\n\
- \"User prefers explicit-SHA fixup over `git absorb`.\" (preference framing — the real fact is `git absorb`'s failure mode)\n\
- \"When helping with fixups, always use `git fix <sha>`.\" (prescriptive rule — write what `git fix` IS, not what to always do)\n\
- \"User wants to fix commit X today.\" (session-specific task)\n\
- \"Assume the user prefers concise responses.\" (generic platitude with no anchoring fact)\n\n\
Forbidden phrasing: \"always\", \"never\", \"when X, do Y\", \"the user prefers\", \"assume the user wants\".\n\
Required phrasing: present-tense statements about what exists or how something works.\n\n\
Return null if the friction has no transferable technical fact — sometimes it's one-off task confusion. \
Respond ONLY with JSON: {\"preload\": \"...\" | null}.";

const DURABILITY_SYSTEM: &str = "\
You are a strict gatekeeper for a long-term memory system. You see ONLY a candidate memory.\n\n\
GOOD memories are TECHNICAL FACTS — about how a tool behaves, what's configured, what an environment is. \
They would be useful to anyone, not just one user. They state WHAT IS, in present tense.\n\n\
BAD memories frame things as user preferences or prescriptive rules. \"User prefers X\" is almost always lazy — \
the real fact is the technical constraint that drove the preference.\n\n\
PASS:\n\
- \"`pacman` does not manage AUR packages; an external helper is required.\"\n\
- \"Codex CLI is configured via ~/.codex/auth.json; it uses ChatGPT subscription for zero-cost API access.\"\n\
- \"`~/Projects/system/sentinel` is a Rust security daemon; `arch install` (custom CLI) rebuilds it via PKGBUILD.\"\n\
- \"Tests for this repo live in `tests/`, not in `src/`.\"\n\n\
FAIL:\n\
- \"When using LLMs, always prefer Codex CLI.\" (prescriptive + universalized)\n\
- \"User prefers explicit-SHA fixup over `git absorb`.\" (preference framing — what's the technical reason?)\n\
- \"When the user asks for fixups, do X.\" (prescriptive rule — write what X IS instead)\n\
- \"User prefers concise responses.\" (no anchoring fact)\n\
- \"User wants to fix commit X today.\" (session-specific)\n\n\
Red-flag phrasing: \"always\", \"never\", \"when X do Y\", \"the user prefers\", \"assume the user wants\", \"the user's goal is\". \
Fail anything containing these unless the rest of the sentence is a clean technical fact (rare).\n\n\
Respond ONLY with JSON: {\"passed\": bool, \"reason\": \"...\"}. Keep reason under 30 words.";

pub async fn extract_candidate(
    full_session: &[Turn],
    friction_turn_index: u32,
    feedback: Option<&str>,
) -> Option<String> {
    let session_text = full_session_text(full_session);
    let feedback_clause = feedback
        .map(|f| format!("\n\nPrevious attempt failed: {f}. Adjust accordingly."))
        .unwrap_or_default();
    let user_msg = format!(
        "Session transcript:\n\n{session_text}\n\n\
        The assistant struggled at turn {friction_turn_index}. \
        Extract a 1-3 sentence preload that would have prevented the struggle.{feedback_clause}"
    );
    let raw = llm::complete(EXTRACTOR_SYSTEM, &user_msg, 300, 20).await?;
    let json_str = extract_json(&raw);
    let parsed: PreloadJson = serde_json::from_str(json_str)
        .map_err(|e| {
            eprintln!("  [extractor] parse error: {e} | raw: {raw}");
        })
        .ok()?;
    parsed.preload.filter(|s| !s.trim().is_empty())
}

// ── T3c: Replay simulator ─────────────────────────────────────────────────────

const REPLAY_SYSTEM_PREFIX: &str = "\
You are an assistant. Use the following preload as your knowledge context. \
Respond to the user's prompt concisely and directly, using the preload.\n\nPreload:\n";

pub async fn replay_with_preload(preload: &str, original_user_prompt: &str) -> Option<String> {
    let system = format!("{REPLAY_SYSTEM_PREFIX}{preload}");
    llm::complete(&system, original_user_prompt, 500, 30).await
}

// ── T3d: Dual judge ───────────────────────────────────────────────────────────

const CORRECTNESS_SYSTEM: &str = "\
You are a judge comparing two assistant responses to the same user prompt. \
Does the simulated response reach substantively the same conclusion as the eventual resolution? \
Minor wording differences are fine; penalise only missing conclusions or wrong conclusions. \
Respond ONLY with JSON: {\"passed\": bool, \"reason\": \"...\"}. Keep reason under 30 words.";

const EFFICIENCY_SYSTEM: &str = "\
You are a judge evaluating whether a simulated assistant response uses a given preload. \
The simulated response should cite, build on, or apply the preload directly. \
Fail if the response ignores the preload and proposes exploration (file reads, greps, searches). \
Respond ONLY with JSON: {\"passed\": bool, \"reason\": \"...\"}. Keep reason under 30 words.";

pub async fn judge_correctness(simulated: &str, eventual_resolution: &str) -> Option<JudgeResult> {
    let user_msg =
        format!("Simulated response:\n{simulated}\n\nEventual resolution:\n{eventual_resolution}");
    let raw = llm::complete(CORRECTNESS_SYSTEM, &user_msg, 200, 15).await?;
    let json_str = extract_json(&raw);
    let parsed: JudgeJson = serde_json::from_str(json_str)
        .map_err(|e| eprintln!("  [correctness judge] parse error: {e} | raw: {raw}"))
        .ok()?;
    Some(JudgeResult {
        passed: parsed.passed,
        reason: parsed.reason,
    })
}

pub async fn judge_efficiency(simulated: &str, preload: &str) -> Option<JudgeResult> {
    let user_msg = format!("Preload:\n{preload}\n\nSimulated response:\n{simulated}");
    let raw = llm::complete(EFFICIENCY_SYSTEM, &user_msg, 200, 15).await?;
    let json_str = extract_json(&raw);
    let parsed: JudgeJson = serde_json::from_str(json_str)
        .map_err(|e| eprintln!("  [efficiency judge] parse error: {e} | raw: {raw}"))
        .ok()?;
    Some(JudgeResult {
        passed: parsed.passed,
        reason: parsed.reason,
    })
}

pub async fn judge_durability(preload: &str) -> Option<JudgeResult> {
    let user_msg = format!("Candidate memory:\n{preload}");
    let raw = llm::complete(DURABILITY_SYSTEM, &user_msg, 200, 15).await?;
    let json_str = extract_json(&raw);
    let parsed: JudgeJson = serde_json::from_str(json_str)
        .map_err(|e| eprintln!("  [durability judge] parse error: {e} | raw: {raw}"))
        .ok()?;
    Some(JudgeResult {
        passed: parsed.passed,
        reason: parsed.reason,
    })
}

// ── T4: Orchestrator ──────────────────────────────────────────────────────────

const MAX_ITERATIONS: usize = 3;

pub async fn analyze_session(session_path: &Path) -> Result<Vec<AnalysisOutcome>> {
    let turns = read_session_turns(session_path)
        .with_context(|| format!("failed to read turns from {}", session_path.display()))?;

    let session_id = session_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let client = Qdrant::from_url(crate::index::QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    crate::memory_unit::ensure_memory_units_collection(&client).await?;
    let embedder = Embedder::new();

    let mut outcomes = Vec::new();

    for turn in &turns {
        // Only classify user turns
        if !matches!(turn.role, crate::extract::Role::User) {
            continue;
        }

        let turn_index = turn.turn_index;

        let friction = match classify_friction(&turns, turn_index).await {
            Some(f) => f,
            None => {
                eprintln!("  [turn {turn_index}] friction classifier returned None, skipping");
                continue;
            }
        };

        if !friction.flagged {
            outcomes.push(AnalysisOutcome::NoFriction { turn: turn_index });
            continue;
        }

        eprintln!(
            "  [turn {turn_index}] friction flagged: {}",
            friction.reason
        );

        let resolution = eventual_resolution(&turns, turn_index);
        let user_prompt = turn.text.clone();

        let mut feedback: Option<String> = None;
        let mut stored = false;
        let mut final_fail_reason = String::new();

        for attempt in 0..MAX_ITERATIONS {
            // Extract candidate
            let candidate = match extract_candidate(&turns, turn_index, feedback.as_deref()).await {
                Some(c) => c,
                None => {
                    final_fail_reason = "extractor returned null".to_string();
                    break;
                }
            };

            eprintln!(
                "  [turn {turn_index}] attempt {}: candidate: {}...",
                attempt + 1,
                candidate.chars().take(60).collect::<String>()
            );

            // Durability gate (pre-replay): reject session-specific preloads cheaply
            let durability = judge_durability(&candidate).await.unwrap_or(JudgeResult {
                passed: false,
                reason: "judge unavailable".to_string(),
            });
            if !durability.passed {
                let fail_reason = format!("durability: {}", durability.reason);
                eprintln!(
                    "  [turn {turn_index}] attempt {} failed durability: {}",
                    attempt + 1,
                    durability.reason
                );
                final_fail_reason = fail_reason.clone();
                feedback = Some(fail_reason);
                continue;
            }

            // Replay
            let simulated = match replay_with_preload(&candidate, &user_prompt).await {
                Some(s) => s,
                None => {
                    feedback = Some("replay failed to produce output".to_string());
                    continue;
                }
            };

            // Judge correctness
            let correctness = judge_correctness(&simulated, &resolution).await;
            let efficiency = judge_efficiency(&simulated, &candidate).await;

            let c_result = correctness.unwrap_or(JudgeResult {
                passed: false,
                reason: "judge unavailable".to_string(),
            });
            let e_result = efficiency.unwrap_or(JudgeResult {
                passed: false,
                reason: "judge unavailable".to_string(),
            });

            if c_result.passed && e_result.passed {
                // Store
                let unit = MemoryUnit {
                    text: candidate,
                    created_at: Utc::now(),
                    source_session: session_id.clone(),
                    source_turn: turn_index,
                    seen_in_sessions: vec![session_id.clone()],
                };

                match upsert_with_dedup(&client, &embedder, unit.clone()).await {
                    Ok(DedupOutcome::Inserted(_)) => {
                        outcomes.push(AnalysisOutcome::Stored {
                            turn: turn_index,
                            unit,
                            deduped: false,
                        });
                    }
                    Ok(DedupOutcome::Merged(_)) => {
                        outcomes.push(AnalysisOutcome::Stored {
                            turn: turn_index,
                            unit,
                            deduped: true,
                        });
                    }
                    Err(e) => {
                        eprintln!("  [turn {turn_index}] upsert error: {e}");
                        outcomes.push(AnalysisOutcome::Discarded {
                            turn: turn_index,
                            reason: format!("upsert failed: {e}"),
                        });
                    }
                }
                stored = true;
                break;
            }

            // Build feedback from failing judge(s)
            let fail_reason = if !c_result.passed && !e_result.passed {
                format!(
                    "correctness: {}; efficiency: {}",
                    c_result.reason, e_result.reason
                )
            } else if !c_result.passed {
                format!("correctness: {}", c_result.reason)
            } else {
                format!("efficiency: {}", e_result.reason)
            };

            eprintln!(
                "  [turn {turn_index}] attempt {} failed: {fail_reason}",
                attempt + 1
            );
            final_fail_reason = fail_reason.clone();
            feedback = Some(fail_reason);
        }

        if !stored {
            outcomes.push(AnalysisOutcome::Discarded {
                turn: turn_index,
                reason: if final_fail_reason.is_empty() {
                    "extractor returned null".to_string()
                } else {
                    format!("validation failed 3x: {final_fail_reason}")
                },
            });
        }
    }

    Ok(outcomes)
}
