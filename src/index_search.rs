use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, Filter, PayloadIncludeSelector, PointId, RetrievedPoint, ScrollPointsBuilder,
    SearchPointsBuilder,
};
use std::collections::BTreeSet;

use super::search_results::build_search_results;
use super::{COLLECTION_SESSION_HISTORY, QDRANT_URL, SearchResult};
use crate::config;
use crate::embed::Embedder;
use crate::extract::HistoryType;
use crate::qdrant_hybrid::ensure_hybrid_collection;

const SESSION_ID_PAGE_SIZE: u32 = 1000;

/// Search prompts and answers in one globally ranked query.
pub async fn search_all(
    query: &str,
    limit: usize,
    source: Option<&str>,
    session: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, None, session).await
}

/// Search user prompts from session history.
pub async fn search_prompts(
    query: &str,
    limit: usize,
    source: Option<&str>,
    session: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, Some(HistoryType::Prompt), session).await
}

/// Search answers (assistant responses).
pub async fn search_answers(
    query: &str,
    limit: usize,
    source: Option<&str>,
    session: Option<&str>,
) -> Result<Vec<SearchResult>> {
    search_collection(query, limit, source, Some(HistoryType::Answer), session).await
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
        None,
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
        None,
    )
    .await
}

async fn search_collection<'a>(
    query: &str,
    limit: usize,
    sources: impl IntoIterator<Item = &'a str>,
    history_type: Option<HistoryType>,
    session: Option<&str>,
) -> Result<Vec<SearchResult>> {
    if !config::search_enabled() {
        return Ok(Vec::new());
    }

    let client = Qdrant::from_url(QDRANT_URL)
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, COLLECTION_SESSION_HISTORY).await?;

    let sources: Vec<&str> = sources.into_iter().collect();
    let session_ids = match session {
        Some(substring) => {
            query_matching_session_ids(
                &client,
                COLLECTION_SESSION_HISTORY,
                history_type,
                &sources,
                substring,
            )
            .await?
        }
        None => Vec::new(),
    };
    if session.is_some() && session_ids.is_empty() {
        return Ok(Vec::new());
    }

    let embedder = Embedder::new();
    let query_vec = embedder.embed(query).await?;
    let search = history_search(query_vec, limit, history_type, &sources, &session_ids);
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
    session_ids: &[String],
) -> SearchPointsBuilder {
    SearchPointsBuilder::new(COLLECTION_SESSION_HISTORY, query_vec, limit as u64)
        .vector_name("dense")
        .with_payload(true)
        .filter(history_filter_for_sessions(
            history_type,
            sources,
            session_ids,
        ))
}

pub fn history_filter(history_type: HistoryType, sources: &[&str]) -> Filter {
    history_filter_for_sessions(Some(history_type), sources, &[])
}

pub fn global_history_filter(sources: &[&str]) -> Filter {
    history_filter_for_sessions(None, sources, &[])
}

pub(crate) fn history_filter_for_sessions(
    history_type: Option<HistoryType>,
    sources: &[&str],
    session_ids: &[String],
) -> Filter {
    let type_condition = history_type
        .map(|history_type| Condition::matches("type", history_type.as_str().to_string()));
    let must: Vec<Condition> = type_condition
        .into_iter()
        .chain(sources_condition(sources))
        .chain(session_condition(session_ids))
        .collect();
    Filter::must(must)
}

pub(crate) async fn query_matching_session_ids(
    client: &Qdrant,
    collection: &str,
    history_type: Option<HistoryType>,
    sources: &[&str],
    substring: &str,
) -> Result<Vec<String>> {
    let filter = history_filter_for_sessions(history_type, sources, &[]);
    let mut matches = BTreeSet::new();
    let mut offset = None;

    loop {
        let page = scroll_session_id_page(client, collection, filter.clone(), offset).await?;
        collect_matching_session_ids(&mut matches, &page.result, substring);
        offset = page.next_page_offset;
        if offset.is_none() {
            break;
        }
    }

    Ok(matches.into_iter().collect())
}

async fn scroll_session_id_page(
    client: &Qdrant,
    collection: &str,
    filter: Filter,
    offset: Option<PointId>,
) -> Result<qdrant_client::qdrant::ScrollResponse> {
    let payload = PayloadIncludeSelector::new(vec!["session_id".to_string()]);
    let mut scroll = ScrollPointsBuilder::new(collection)
        .filter(filter)
        .limit(SESSION_ID_PAGE_SIZE)
        .with_payload(payload)
        .with_vectors(false);
    if let Some(offset) = offset {
        scroll = scroll.offset(offset);
    }

    client
        .scroll(scroll)
        .await
        .context("failed to scan session IDs")
}

fn collect_matching_session_ids(
    matches: &mut BTreeSet<String>,
    points: &[RetrievedPoint],
    substring: &str,
) {
    for point in points {
        let Some(session_id) = point.payload.get("session_id") else {
            continue;
        };
        let Some(qdrant_client::qdrant::value::Kind::StringValue(session_id)) = &session_id.kind
        else {
            continue;
        };
        if session_id.contains(substring) {
            matches.insert(session_id.clone());
        }
    }
}

fn session_condition(session_ids: &[String]) -> Option<Condition> {
    if session_ids.is_empty() {
        return None;
    }
    Some(Condition::matches("session_id", session_ids.to_vec()))
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
