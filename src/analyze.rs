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
        .map(|t| format_turn(t))
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
You write encyclopedia-style entries for a long-term technical knowledge base.\n\n\
Given a transcript where the assistant struggled, identify what TECHNICAL FACT about the relevant tool, command, codebase, \
or environment — IF stated as a Wikipedia-style entry — would have given the assistant the knowledge it lacked.\n\n\
Reframe the question in your head: NOT \"what should the assistant DO\" but \"what IS true about <subject>\". \
The output should read like the first paragraph of a man page or a Wikipedia article — describing properties and behaviors, not giving advice.\n\n\
GOOD (encyclopedia-style facts):\n\
- \"`git absorb` selects fixup targets via `git blame` of changed lines; it cannot route hunks containing newly-added lines because no blame information exists for them.\"\n\
- \"`git cherry -v <upstream> <branch>` lists each commit in <branch> with `+` (not in upstream) or `-` (already present in upstream by patch-id), detecting cherry-picked or rebased commits despite different SHAs.\"\n\
- \"In the codex codebase, `PreToolUseOutcome` carries the `updatedInput` field from `events/pre_tool_use.rs` to `core/src/tools/registry.rs`, which mutates the shell tool invocation payload before dispatch.\"\n\
- \"The `arch` CLI on this system wraps `pacman` and `paru`, so AUR packages are installed via `arch install` rather than `pacman -S`.\"\n\n\
BAD (recipes, instructions, preferences):\n\
- \"Use `git cherry -v` to detect cherry-picks.\" (imperative recipe — say what `git cherry -v` IS instead)\n\
- \"When checking branch merge status, use patch-id comparison.\" (conditional recipe — describe the patch-id comparison itself)\n\
- \"User prefers explicit-SHA fixup.\" (preference — describe `git absorb`'s mechanism instead)\n\
- \"Always run X before Y.\" (universal rule)\n\n\
THE TEST: read your candidate aloud. Does it sound like a Wikipedia sentence (describing properties)? \
Or does it sound like advice (telling someone what to do)? Only the first kind is acceptable.\n\n\
Forbidden openers: \"Use\", \"Run\", \"For\", \"When\", \"To\", \"If\", \"Before\", \"After\", \"Always\", \"Never\", \"Verify\", \"Check\", \"Compare\".\n\
Forbidden subjects: \"The user\", \"The assistant\", \"You\".\n\
Required openers: a noun phrase naming the tool, command, file, or system that the fact is about.\n\n\
Return null if the friction has no transferable encyclopedia-style fact. \
Respond ONLY with JSON: {\"preload\": \"...\" | null}.";

const DURABILITY_SYSTEM: &str = "\
You are a strict gatekeeper for a long-term memory system. You see ONLY a candidate memory.\n\n\
A good memory is INDICATIVE — it states what something IS or DOES, in present tense.\n\
A bad memory is IMPERATIVE or CONDITIONAL — it tells the assistant what to DO, when, or how.\n\n\
PASS only declarative facts about properties, behaviors, or configurations:\n\
- \"`pacman` does not manage AUR packages; AUR builds need an external helper.\"\n\
- \"`git cherry -v <upstream> <branch>` outputs +/- markers for patch-equivalent commits, detecting cherry-picks even with different SHAs.\"\n\
- \"`git absorb` chooses fixup targets via `git blame`, so it cannot route hunks containing newly-added lines.\"\n\
- \"Codex CLI auth lives at ~/.codex/auth.json; calls bill against ChatGPT subscription, not the OpenAI API.\"\n\n\
FAIL imperative or conditional phrasing — these are recipes, not facts:\n\
- \"For branch-maturity checks, run X.\" (recipe: \"for X, do Y\")\n\
- \"When comparing branches, use `git cherry`.\" (conditional recipe)\n\
- \"Use `git cherry -v` to detect cherry-picks.\" (imperative \"use X to Y\")\n\
- \"To check merge status, do Z.\" (\"to X, do Y\")\n\
- \"Always prefer X.\" / \"Never use Y.\" (universalized rules)\n\
- \"The user prefers X.\" / \"The user wants Y.\" (preference framing)\n\
- \"Verify X before doing Y.\" (instruction, not fact)\n\n\
FAIL ephemeral snapshots — current state of specific entities, which goes stale fast:\n\
- \"`feature-x-branch` is 3346 commits ahead of master with 355 files changed.\" (specific branch's current diff stats; will change daily)\n\
- \"`config.toml` currently has `model = \"gpt-5.4\"` set.\" (current config value; user changes these)\n\
- \"The `widgets` table has 12,453 rows.\" (current data state)\n\
- \"PR #1234 is in review.\" (current PR status)\n\
- \"On 2026-04-28 the user is working on Y.\" (timestamped state)\n\n\
Snapshot red flags: specific commit counts/file counts/line counts/row counts, named branches/PRs/issues with current status, timestamped facts, or any \"X currently is/has Y\" where Y is volatile state. \
Generalizing to a class is fine (\"branches with rebased history\"), but specific named entities with numeric state are not.\n\n\
Red-flag openers (FAIL on sight unless paired with a pure declarative): \
\"For ...\", \"When ...\", \"To ...\", \"Use ...\", \"Always\", \"Never\", \"Before ...\", \"After ...\", \"If ...\", \
\"The user prefers\", \"Assume\", \"You should\".\n\n\
Convert mentally: if the candidate could be rewritten as \"X is/has/does Y\" without losing meaning, \
it might be a fact in disguise — pass it. If rewriting requires adding a subject like \"the assistant\" or \"someone\" \
or \"you\", it's a recipe — fail it.\n\n\
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
