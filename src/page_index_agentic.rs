use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::{llm, page_index};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RetrievalMode {
    Lexical,
    Agentic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TreeWalkCandidate {
    pub doc_id: String,
    pub node_id: String,
    pub title: String,
    pub score: usize,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentRequest {
    pub doc_id: String,
    pub locator: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeWalkReference {
    pub doc_id: String,
    pub locator: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeWalkStep {
    pub action: String,
    pub doc_id: Option<String>,
    pub locator: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeWalkResponse {
    pub mode: RetrievalMode,
    pub answer: String,
    pub references: Vec<TreeWalkReference>,
    pub steps: Vec<TreeWalkStep>,
}

#[derive(Debug, Clone)]
pub struct TreeWalkContext {
    pub query: String,
    pub candidates: Vec<TreeWalkCandidate>,
    pub metadata: Vec<(String, Value)>,
    pub structures: Vec<(String, Value)>,
}

pub trait TreeWalkCorpus {
    fn search_candidates(&self, query: &str, limit: usize) -> Result<Vec<TreeWalkCandidate>>;
    fn metadata(&self, doc_id: &str) -> Result<Value>;
    fn structure(&self, doc_id: &str) -> Result<Value>;
    fn content(&self, doc_id: &str, locator: &str) -> Result<String>;
    fn content_command(&self, doc_id: &str, locator: &str) -> String;
}

pub trait TreeWalkPlanner {
    fn plan_content(&self, context: &TreeWalkContext) -> Result<Vec<ContentRequest>>;
}

pub struct TranscriptTreeWalkCorpus {
    index_dir: PathBuf,
}

impl TranscriptTreeWalkCorpus {
    pub fn new(index_dir: impl AsRef<Path>) -> Self {
        Self {
            index_dir: index_dir.as_ref().to_path_buf(),
        }
    }
}

impl TreeWalkCorpus for TranscriptTreeWalkCorpus {
    fn search_candidates(&self, query: &str, limit: usize) -> Result<Vec<TreeWalkCandidate>> {
        let results = page_index::query_index(&self.index_dir, query, limit)?;
        let candidates = results
            .into_iter()
            .map(|result| {
                tree_walk_candidate(
                    result.doc_id,
                    result.node_id,
                    result.title,
                    result.score,
                    result.reason,
                )
            })
            .collect();
        Ok(candidates)
    }

    fn metadata(&self, doc_id: &str) -> Result<Value> {
        let metadata = page_index::document_metadata(&self.index_dir, doc_id)?;
        serde_json::to_value(metadata).context("failed to serialize transcript metadata")
    }

    fn structure(&self, doc_id: &str) -> Result<Value> {
        let structure = page_index::document_structure(&self.index_dir, doc_id)?;
        serde_json::to_value(structure).context("failed to serialize transcript structure")
    }

    fn content(&self, doc_id: &str, locator: &str) -> Result<String> {
        Ok(page_index::document_content(&self.index_dir, doc_id, locator)?.text)
    }

    fn content_command(&self, doc_id: &str, locator: &str) -> String {
        format!("claude-memory transcript-page-index content {doc_id} {locator}")
    }
}

fn tree_walk_candidate(
    doc_id: String,
    node_id: String,
    title: String,
    score: usize,
    reason: String,
) -> TreeWalkCandidate {
    TreeWalkCandidate {
        doc_id,
        node_id,
        title,
        score,
        reason,
    }
}

pub async fn retrieve_with_llm<C: TreeWalkCorpus>(
    query: &str,
    corpus: &C,
    limit: usize,
) -> Result<TreeWalkResponse> {
    let (context, mut steps) = inspect_tree(query, corpus, limit)?;
    let prompt = build_planner_prompt(&context);
    let raw = llm::complete(
        "Choose PageIndex content fetches. Return only JSON.",
        &prompt,
        400,
        90,
    )
    .await;

    let requests = match raw {
        Ok(text) => parse_content_requests(&text).unwrap_or_else(|_| Vec::new()),
        Err(_) => return deterministic_fallback(query, corpus, limit),
    };
    if requests.is_empty() {
        return deterministic_fallback(query, corpus, limit);
    }

    fetch_content(corpus, RetrievalMode::Agentic, requests, &mut steps)
}

pub fn retrieve_with_planner<C: TreeWalkCorpus, P: TreeWalkPlanner>(
    query: &str,
    corpus: &C,
    limit: usize,
    planner: &P,
) -> Result<TreeWalkResponse> {
    let (context, mut steps) = inspect_tree(query, corpus, limit)?;
    let requests = planner.plan_content(&context)?;
    if requests.is_empty() {
        return deterministic_fallback(query, corpus, limit);
    }

    fetch_content(corpus, RetrievalMode::Agentic, requests, &mut steps)
}

pub fn deterministic_fallback<C: TreeWalkCorpus>(
    query: &str,
    corpus: &C,
    limit: usize,
) -> Result<TreeWalkResponse> {
    let candidates = corpus.search_candidates(query, limit)?;
    let requests = candidates
        .iter()
        .map(|candidate| ContentRequest {
            doc_id: candidate.doc_id.clone(),
            locator: candidate.node_id.clone(),
        })
        .collect::<Vec<_>>();
    let mut steps = vec![TreeWalkStep {
        action: "deterministic lexical fallback".to_string(),
        doc_id: None,
        locator: None,
    }];
    fetch_content(corpus, RetrievalMode::Lexical, requests, &mut steps)
}

fn inspect_tree<C: TreeWalkCorpus>(
    query: &str,
    corpus: &C,
    limit: usize,
) -> Result<(TreeWalkContext, Vec<TreeWalkStep>)> {
    let candidates = corpus.search_candidates(query, limit)?;
    let mut inspection = TreeInspection::new();

    for candidate in &candidates {
        inspect_candidate(corpus, candidate, &mut inspection)?;
    }

    Ok((
        TreeWalkContext {
            query: query.to_string(),
            candidates,
            metadata: inspection.metadata,
            structures: inspection.structures,
        },
        inspection.steps,
    ))
}

struct TreeInspection {
    metadata: Vec<(String, Value)>,
    structures: Vec<(String, Value)>,
    steps: Vec<TreeWalkStep>,
}

impl TreeInspection {
    fn new() -> Self {
        Self {
            metadata: Vec::new(),
            structures: Vec::new(),
            steps: vec![TreeWalkStep {
                action: "search candidates".to_string(),
                doc_id: None,
                locator: None,
            }],
        }
    }
}

fn inspect_candidate<C: TreeWalkCorpus>(
    corpus: &C,
    candidate: &TreeWalkCandidate,
    inspection: &mut TreeInspection,
) -> Result<()> {
    let doc_id = candidate.doc_id.clone();
    inspection
        .metadata
        .push((doc_id.clone(), corpus.metadata(&doc_id)?));
    push_step(
        &mut inspection.steps,
        "inspect metadata",
        Some(&doc_id),
        None,
    );

    inspection
        .structures
        .push((doc_id.clone(), corpus.structure(&doc_id)?));
    push_step(
        &mut inspection.steps,
        "inspect structure",
        Some(&doc_id),
        None,
    );
    Ok(())
}

fn fetch_content<C: TreeWalkCorpus>(
    corpus: &C,
    mode: RetrievalMode,
    requests: Vec<ContentRequest>,
    steps: &mut Vec<TreeWalkStep>,
) -> Result<TreeWalkResponse> {
    let mut references = Vec::new();
    for request in requests {
        let text = corpus.content(&request.doc_id, &request.locator)?;
        push_step(
            steps,
            "fetch content",
            Some(&request.doc_id),
            Some(&request.locator),
        );
        references.push(TreeWalkReference {
            doc_id: request.doc_id,
            locator: request.locator,
            text,
        });
    }

    Ok(TreeWalkResponse {
        mode,
        answer: answer_from_references(&references),
        references,
        steps: steps.clone(),
    })
}

fn push_step(
    steps: &mut Vec<TreeWalkStep>,
    action: &str,
    doc_id: Option<&str>,
    locator: Option<&str>,
) {
    steps.push(TreeWalkStep {
        action: action.to_string(),
        doc_id: doc_id.map(ToString::to_string),
        locator: locator.map(ToString::to_string),
    });
}

fn answer_from_references(references: &[TreeWalkReference]) -> String {
    if references.is_empty() {
        return "No PageIndex content matched the query.".to_string();
    }

    references
        .iter()
        .map(|reference| {
            format!(
                "[{}#{}]\n{}",
                reference.doc_id,
                reference.locator,
                reference.text.trim()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn build_planner_prompt(context: &TreeWalkContext) -> String {
    let candidates = serde_json::to_string_pretty(&context.candidates).unwrap_or_default();
    let metadata = serde_json::to_string_pretty(&context.metadata).unwrap_or_default();
    let structures = serde_json::to_string_pretty(&context.structures).unwrap_or_default();
    format!(
        "Query: {}\n\nCandidates:\n{}\n\nMetadata:\n{}\n\nStructures:\n{}\n\nReturn JSON array of {{\"doc_id\":\"...\",\"locator\":\"node-or-range\"}}. Fetch only tight content ranges needed to answer.",
        context.query, candidates, metadata, structures
    )
}

fn parse_content_requests(raw: &str) -> Result<Vec<ContentRequest>> {
    let trimmed = raw.trim();
    serde_json::from_str(trimmed).context("failed to parse PageIndex content requests")
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use serde_json::json;
    use std::cell::RefCell;

    struct FakeCorpus {
        calls: RefCell<Vec<String>>,
    }

    impl FakeCorpus {
        fn calls(&self) -> Vec<String> {
            self.calls.borrow().clone()
        }
    }

    impl TreeWalkCorpus for FakeCorpus {
        fn search_candidates(&self, _query: &str, _limit: usize) -> Result<Vec<TreeWalkCandidate>> {
            self.calls.borrow_mut().push("search".to_string());
            Ok(vec![TreeWalkCandidate {
                doc_id: "router.md".to_string(),
                node_id: "000002".to_string(),
                title: "DHCP".to_string(),
                score: 42,
                reason: "matched query terms: dhcp".to_string(),
            }])
        }

        fn metadata(&self, doc_id: &str) -> Result<Value> {
            self.calls.borrow_mut().push(format!("metadata:{doc_id}"));
            Ok(json!({ "doc_id": doc_id }))
        }

        fn structure(&self, doc_id: &str) -> Result<Value> {
            self.calls.borrow_mut().push(format!("structure:{doc_id}"));
            Ok(json!({ "nodes": [{ "node_id": "000002", "title": "DHCP" }] }))
        }

        fn content(&self, doc_id: &str, locator: &str) -> Result<String> {
            self.calls
                .borrow_mut()
                .push(format!("content:{doc_id}:{locator}"));
            Ok("Use static DHCP reservations.".to_string())
        }

        fn content_command(&self, doc_id: &str, locator: &str) -> String {
            format!("fake content {doc_id} {locator}")
        }
    }

    struct FakePlanner;

    impl TreeWalkPlanner for FakePlanner {
        fn plan_content(&self, context: &TreeWalkContext) -> Result<Vec<ContentRequest>> {
            let candidate = context
                .candidates
                .first()
                .ok_or_else(|| anyhow!("missing candidate"))?;
            Ok(vec![ContentRequest {
                doc_id: candidate.doc_id.clone(),
                locator: candidate.node_id.clone(),
            }])
        }
    }

    struct EmptyPlanner;

    impl TreeWalkPlanner for EmptyPlanner {
        fn plan_content(&self, _context: &TreeWalkContext) -> Result<Vec<ContentRequest>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn tree_walk_inspects_metadata_structure_then_fetches_content() {
        let corpus = FakeCorpus {
            calls: RefCell::new(Vec::new()),
        };

        let response = retrieve_with_planner("dhcp", &corpus, 1, &FakePlanner).unwrap();

        assert_eq!(response.mode, RetrievalMode::Agentic);
        assert_eq!(
            corpus.calls(),
            vec![
                "search",
                "metadata:router.md",
                "structure:router.md",
                "content:router.md:000002"
            ]
        );
        assert!(response.answer.contains("Use static DHCP reservations."));
        assert_eq!(response.references[0].doc_id, "router.md");
        assert_eq!(response.references[0].locator, "000002");
    }

    #[test]
    fn empty_agentic_plan_uses_labeled_lexical_fallback() {
        let corpus = FakeCorpus {
            calls: RefCell::new(Vec::new()),
        };

        let response = retrieve_with_planner("dhcp", &corpus, 1, &EmptyPlanner).unwrap();

        assert_eq!(response.mode, RetrievalMode::Lexical);
        assert!(
            response
                .steps
                .iter()
                .any(|step| step.action == "deterministic lexical fallback")
        );
        assert!(response.answer.contains("Use static DHCP reservations."));
    }
}
