use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{Condition, Filter, SearchPointsBuilder};

use super::search_results::build_search_results;
use super::{COLLECTION_SESSION_HISTORY, QDRANT_URL, SearchResult};
use crate::config;
use crate::embed::Embedder;
use crate::extract::HistoryType;
use crate::qdrant_hybrid::ensure_hybrid_collection;

/// Search prompts and answers in one globally ranked query.
pub async fn search_all(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, None).await
}

/// Search user prompts from session history.
pub async fn search_prompts(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, Some(HistoryType::Prompt)).await
}

/// Search answers (assistant responses).
pub async fn search_answers(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, Some(HistoryType::Answer)).await
}

pub async fn search_prompt_sources(
    query: &str,
    limit: usize,
    sources: &[&str],
) -> Result<Vec<SearchResult>> {
    search_collection(
        query,
        limit,
        sources.iter().copied(),
        Some(HistoryType::Prompt),
    )
    .await
}

pub async fn search_answer_sources(
    query: &str,
    limit: usize,
    sources: &[&str],
) -> Result<Vec<SearchResult>> {
    search_collection(
        query,
        limit,
        sources.iter().copied(),
        Some(HistoryType::Answer),
    )
    .await
}

async fn search_collection<'a>(
    query: &str,
    limit: usize,
    sources: impl IntoIterator<Item = &'a str>,
    history_type: Option<HistoryType>,
) -> Result<Vec<SearchResult>> {
    if !config::search_enabled() {
        return Ok(Vec::new());
    }

    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, COLLECTION_SESSION_HISTORY).await?;

    let embedder = Embedder::new();
    let query_vec = embedder.embed(query).await?;

    let sources: Vec<&str> = sources.into_iter().collect();
    let search = history_search(query_vec, limit, history_type, &sources);
    let results = client
        .search_points(search)
        .await
        .context("search failed")?;
    Ok(build_search_results(results.result))
}

fn history_search(
    query_vec: Vec<f32>,
    limit: usize,
    history_type: Option<HistoryType>,
    sources: &[&str],
) -> SearchPointsBuilder {
    SearchPointsBuilder::new(COLLECTION_SESSION_HISTORY, query_vec, limit as u64)
        .vector_name("dense")
        .with_payload(true)
        .filter(search_filter(history_type, sources))
}

pub fn history_filter(history_type: HistoryType, sources: &[&str]) -> Filter {
    search_filter(Some(history_type), sources)
}

pub fn global_history_filter(sources: &[&str]) -> Filter {
    search_filter(None, sources)
}

fn search_filter(history_type: Option<HistoryType>, sources: &[&str]) -> Filter {
    let type_condition = history_type
        .map(|history_type| Condition::matches("type", history_type.as_str().to_string()));
    let must: Vec<Condition> = type_condition
        .into_iter()
        .chain(sources_condition(sources))
        .collect();
    Filter::must(must)
}

fn sources_condition(sources: &[&str]) -> Option<Condition> {
    match sources {
        [] => None,
        [source] => Some(source_condition(source)),
        _ => Some(Filter::should(sources.iter().map(|source| source_condition(source))).into()),
    }
}

fn source_condition(source: &str) -> Condition {
    Condition::matches("source", source.to_string())
}
