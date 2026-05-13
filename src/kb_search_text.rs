use std::collections::HashMap;

const SNIPPET_CHARS: usize = 420;
const STOPWORDS: &[&str] = &[
    "the", "and", "or", "to", "of", "in", "for", "a", "an", "is", "are", "we", "it", "with", "on",
    "as", "by", "be", "this", "that", "use", "did", "do", "does", "you", "your", "me", "my", "new",
];

pub(crate) fn token_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokenize(text) {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

pub(crate) fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '/' || ch == '.' {
            current.push(ch);
            continue;
        }
        push_token(&mut tokens, &mut current);
    }
    push_token(&mut tokens, &mut current);
    tokens
}

fn push_token(tokens: &mut Vec<String>, current: &mut String) {
    if current.len() > 1 && !is_stopword(current) {
        let token = std::mem::take(current);
        tokens.push(token.clone());
        for part in token.split(['-', '/', '.']) {
            if part.len() > 1 && !is_stopword(part) {
                tokens.push(part.to_string());
            }
        }
    } else {
        current.clear();
    }
}

fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

pub(crate) fn build_snippet(text: &str, query_tokens: &[String]) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower = compact.to_lowercase();
    let start = query_tokens
        .iter()
        .filter_map(|token| lower.find(token))
        .min()
        .unwrap_or(0)
        .saturating_sub(80);
    let start = previous_char_boundary(&compact, start);
    let start = previous_word_boundary(&compact, start);
    truncate_chars(&compact[start..], SNIPPET_CHARS)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn previous_char_boundary(text: &str, index: usize) -> usize {
    let mut boundary = index.min(text.len());
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

fn previous_word_boundary(text: &str, index: usize) -> usize {
    let mut boundary = index;
    while boundary > 0 && !text[..boundary].ends_with(char::is_whitespace) {
        boundary = previous_char_boundary(text, boundary.saturating_sub(1));
    }
    boundary
}
