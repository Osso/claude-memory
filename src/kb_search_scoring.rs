use super::{KbSearchResult, TextIndexNode};
use std::collections::HashSet;

const HEADING_WEIGHT: usize = 600;
const PATH_WEIGHT: usize = 400;
const BODY_WEIGHT: usize = 100;
const TERM_FREQUENCY_CAP: usize = 3;
const PHRASE_BONUS: usize = 1_000;
const SCORE_SCALE: usize = 100;

pub(super) fn score_text_node(
    node: TextIndexNode,
    scoring_terms: &[String],
    phrase_tokens: &[String],
    coverage_stride: usize,
) -> Option<KbSearchResult> {
    let tokens = NodeTokens::from_node(&node);
    let matched_terms = tokens.matched_term_count(scoring_terms);
    if matched_terms == 0 {
        return None;
    }
    let score =
        tokens.calculate_score(scoring_terms, phrase_tokens, coverage_stride, matched_terms);
    Some(build_search_result(node, score))
}

struct NodeTokens {
    heading: Vec<String>,
    path: Vec<String>,
    body: Vec<String>,
}

impl NodeTokens {
    fn from_node(node: &TextIndexNode) -> Self {
        Self {
            heading: text_index_tokens(&node.heading_path),
            path: text_index_tokens(&node.path),
            body: text_index_tokens(&node.normalized_body),
        }
    }

    fn matched_term_count(&self, query_tokens: &[String]) -> usize {
        query_tokens
            .iter()
            .filter(|term| {
                self.heading.contains(term) || self.path.contains(term) || self.body.contains(term)
            })
            .count()
    }

    fn calculate_score(
        &self,
        query_tokens: &[String],
        phrase_tokens: &[String],
        coverage_stride: usize,
        matched_terms: usize,
    ) -> usize {
        let structural_frequency =
            structural_term_frequency(query_tokens, &self.heading, &self.path);
        let body_frequency = body_term_frequency(query_tokens, &self.body);
        let length_divisor = integer_sqrt(self.body.len().max(1)).max(1);
        let phrase_bonus =
            usize::from(contains_token_sequence(&self.body, phrase_tokens)) * PHRASE_BONUS;
        let secondary_score = structural_frequency * SCORE_SCALE
            + body_frequency * SCORE_SCALE / length_divisor
            + phrase_bonus;
        matched_terms
            .saturating_mul(coverage_stride)
            .saturating_add(secondary_score)
    }
}

fn build_search_result(node: TextIndexNode, score: usize) -> KbSearchResult {
    let node_id = format!("{}-{}", node.line_start, node.line_end);
    let title = node
        .heading_path
        .rsplit(" > ")
        .next()
        .unwrap_or(&node.heading_path)
        .to_string();
    KbSearchResult {
        doc_id: node.path.clone(),
        path: node.path,
        heading: node.heading_path,
        text: node.normalized_body,
        score,
        node_id,
        title,
        reason: format!("matched deterministic text index; score {score}"),
    }
}

pub(super) fn text_index_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

pub(super) fn unique_tokens(tokens: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    tokens
        .into_iter()
        .filter(|token| seen.insert(token.clone()))
        .collect()
}

pub(super) fn is_archive_path(path: &str) -> bool {
    path.split('/')
        .any(|component| component.to_ascii_lowercase().contains("archive"))
}

pub(super) fn coverage_score_stride(query_term_count: usize) -> usize {
    let maximum_score_per_term =
        TERM_FREQUENCY_CAP * (HEADING_WEIGHT + PATH_WEIGHT + BODY_WEIGHT) * SCORE_SCALE;
    query_term_count
        .saturating_mul(maximum_score_per_term)
        .saturating_add(PHRASE_BONUS)
        .saturating_add(1)
}

fn structural_term_frequency(
    query_tokens: &[String],
    heading_tokens: &[String],
    path_tokens: &[String],
) -> usize {
    query_tokens
        .iter()
        .map(|term| {
            capped_frequency(heading_tokens, term) * HEADING_WEIGHT
                + capped_frequency(path_tokens, term) * PATH_WEIGHT
        })
        .sum()
}

fn body_term_frequency(query_tokens: &[String], body_tokens: &[String]) -> usize {
    query_tokens
        .iter()
        .map(|term| capped_frequency(body_tokens, term) * BODY_WEIGHT)
        .sum()
}

fn contains_token_sequence(tokens: &[String], sequence: &[String]) -> bool {
    !sequence.is_empty()
        && tokens
            .windows(sequence.len())
            .any(|window| window == sequence)
}

fn capped_frequency(tokens: &[String], term: &str) -> usize {
    tokens
        .iter()
        .filter(|token| token.as_str() == term)
        .count()
        .min(TERM_FREQUENCY_CAP)
}

pub(super) fn integer_sqrt(value: usize) -> usize {
    let mut low = 0;
    let mut high = value;
    let mut root = 0;
    while low <= high {
        let middle = low + (high - low) / 2;
        if middle == 0 || middle <= value / middle {
            root = middle;
            low = middle.saturating_add(1);
        } else {
            high = middle - 1;
        }
    }
    root
}
