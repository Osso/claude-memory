use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{Condition, Filter, ScrollPointsBuilder, SearchPointsBuilder};

use super::search_results::{build_search_results, get_string};
use super::{COLLECTION_ANSWERS, COLLECTION_PROMPTS, QDRANT_URL, SearchResult};
use crate::config;
use crate::embed::Embedder;
use crate::qdrant_hybrid::ensure_hybrid_collection;

/// Search prompts (user messages, KB).
pub async fn search_prompts(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source.into_iter(), COLLECTION_PROMPTS).await
}

/// Search answers (assistant responses).
pub async fn search_answers(
    query: &str,
    limit: usize,
    source: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source.into_iter(), COLLECTION_ANSWERS).await
}

pub async fn search_prompt_sources(
    query: &str,
    limit: usize,
    sources: &[&str],
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, sources.iter().copied(), COLLECTION_PROMPTS).await
}

pub async fn search_answer_sources(
    query: &str,
    limit: usize,
    sources: &[&str],
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, sources.iter().copied(), COLLECTION_ANSWERS).await
}

/// Search manually stored memories by substring without requiring embeddings.
pub async fn search_memories(query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let client = memory_search_client().await?;
    let filter = memory_source_filter();
    let needle = query.to_lowercase();
    collect_memory_matches(&client, &filter, &needle, limit).await
}

async fn memory_search_client() -> Result<Qdrant> {
    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, COLLECTION_PROMPTS).await?;
    Ok(client)
}

fn memory_source_filter() -> Filter {
    Filter::must([Condition::matches("source", "memory".to_string())])
}

async fn collect_memory_matches(
    client: &Qdrant,
    filter: &Filter,
    needle: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let mut matches = Vec::new();
    let mut offset: Option<qdrant_client::qdrant::PointId> = None;

    while matches.len() < limit {
        let scroll = memory_scroll(filter, offset);
        let result = client
            .scroll(scroll)
            .await
            .context("memory search scroll failed")?;

        append_memory_matches(&mut matches, result.result, needle, limit);

        offset = result.next_page_offset;
        if offset.is_none() {
            break;
        }
    }

    Ok(matches)
}

fn memory_scroll(
    filter: &Filter,
    offset: Option<qdrant_client::qdrant::PointId>,
) -> ScrollPointsBuilder {
    let scroll = ScrollPointsBuilder::new(COLLECTION_PROMPTS)
        .limit(100)
        .with_payload(true)
        .filter(filter.clone());

    if let Some(point_id) = offset {
        scroll.offset(point_id)
    } else {
        scroll
    }
}

fn append_memory_matches(
    matches: &mut Vec<SearchResult>,
    points: Vec<qdrant_client::qdrant::RetrievedPoint>,
    needle: &str,
    limit: usize,
) {
    for point in points {
        if let Some(result) = memory_point_result(&point, needle) {
            matches.push(result);
        }
        if matches.len() >= limit {
            break;
        }
    }
}

fn memory_point_result(
    point: &qdrant_client::qdrant::RetrievedPoint,
    needle: &str,
) -> Option<SearchResult> {
    let text = get_string(&point.payload, "text");
    if !text.to_lowercase().contains(needle) {
        return None;
    }

    Some(SearchResult {
        text,
        source: "memory".to_string(),
        path: get_string(&point.payload, "project"),
        session_id: String::new(),
        score: 1.0,
    })
}

async fn search_collection<'a>(
    query: &str,
    limit: usize,
    sources: impl IntoIterator<Item = &'a str>,
    collection: &str,
) -> Result<Vec<SearchResult>> {
    if !config::search_enabled() {
        return Ok(Vec::new());
    }

    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, collection).await?;

    let embedder = Embedder::new();
    let query_vec = embedder.embed(query).await?;

    let mut search = SearchPointsBuilder::new(collection, query_vec, limit as u64)
        .vector_name("dense")
        .with_payload(true);

    let sources: Vec<&str> = sources.into_iter().collect();
    if let Some(filter) = source_filter(&sources) {
        search = search.filter(filter);
    }

    let results = client
        .search_points(search)
        .await
        .context("search failed")?;
    Ok(build_search_results(results.result))
}

fn source_filter(sources: &[&str]) -> Option<Filter> {
    match sources {
        [] => None,
        [source] => Some(Filter::must([source_condition(source)])),
        _ => Some(Filter::should(
            sources.iter().map(|source| source_condition(source)),
        )),
    }
}

fn source_condition(source: &str) -> Condition {
    Condition::matches("source", source.to_string())
}
