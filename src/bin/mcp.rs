//! MCP server for Claude memory operations.

use anyhow::{Context, Result};
use claude_memory::config;
use claude_memory::embed::Embedder;
use claude_memory::graph;
use claude_memory::llm::{self, RawResult};
use claude_memory::memory_unit::manual_memory_write_guidance;
use claude_memory::qdrant_hybrid::{BM25_MODEL, ensure_hybrid_collection};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, Document, Filter, Fusion, PrefetchQueryBuilder, Query, QueryPointsBuilder,
};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::OnceCell;

fn log(msg: &str) {
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/claude/memory-mcp.log")
    else {
        return;
    };
    let now = chrono::Local::now().format("%H:%M:%S%.3f");
    let pid = std::process::id();
    let _ = writeln!(f, "[{now} pid={pid}] {msg}");
}

const QDRANT_URL: &str = "http://localhost:6334";
const COLLECTION_PROMPTS: &str = "claude-memory";
const COLLECTION_ANSWERS: &str = "claude-answers";

/// Lazy-initialized Qdrant client.
static QDRANT: OnceCell<Qdrant> = OnceCell::const_new();

async fn get_qdrant() -> Result<&'static Qdrant> {
    log("get_qdrant: enter");
    let result = QDRANT
        .get_or_try_init(|| async {
            log("get_qdrant: initializing client");
            let client = Qdrant::from_url(QDRANT_URL)
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .context("failed to connect to Qdrant")?;
            log("get_qdrant: client built, ensuring collections");
            ensure_hybrid_collection(&client, COLLECTION_PROMPTS).await?;
            log("get_qdrant: prompts collection ok");
            ensure_hybrid_collection(&client, COLLECTION_ANSWERS).await?;
            log("get_qdrant: answers collection ok");
            claude_memory::memory_unit::ensure_memory_units_collection(&client).await?;
            log("get_qdrant: memory-units collection ok");
            Ok(client)
        })
        .await;
    log("get_qdrant: done");
    result
}

/// Build the prefetch+RRF hybrid QueryPointsBuilder.
fn make_hybrid_query(
    collection: &str,
    query_vec: Vec<f32>,
    query_text: &str,
    limit: u64,
    over_fetch: u64,
) -> QueryPointsBuilder {
    QueryPointsBuilder::new(collection)
        .add_prefetch(
            PrefetchQueryBuilder::default()
                .query(Query::new_nearest(query_vec))
                .using("dense")
                .limit(over_fetch),
        )
        .add_prefetch(
            PrefetchQueryBuilder::default()
                .query(Query::new_nearest(Document::new(query_text, BM25_MODEL)))
                .using("bm25")
                .limit(over_fetch),
        )
        .query(Fusion::Rrf)
        .limit(limit)
        .with_payload(true)
}

/// Format a single scored point into output text.
fn format_scored_point(i: usize, point: &qdrant_client::qdrant::ScoredPoint) -> String {
    let text = get_payload_str(&point.payload, "text");
    let source = get_payload_str(&point.payload, "source");
    let path = get_payload_str(&point.payload, "path");
    format!(
        "{}. [{}] {} (score: {:.3})\n   {}\n\n",
        i + 1,
        source,
        path,
        point.score,
        text.replace('\n', " "),
    )
}

fn get_payload_str(payload: &HashMap<String, qdrant_client::qdrant::Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

#[derive(Clone)]
pub struct MemoryService {
    embedder: Arc<Embedder>,
    tool_router: ToolRouter<Self>,
}

impl MemoryService {
    fn new() -> Self {
        Self {
            embedder: Arc::new(Embedder::new()),
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MemoryWriteParams {
    content: String,
    #[schemars(description = "Category: correction, preference, context, learning, decision")]
    category: Option<String>,
    #[schemars(
        description = "Project slug, or __global__ for memories that apply everywhere. Manual writes are disabled and this is kept for compatibility."
    )]
    project: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    query: String,
    #[schemars(description = "Maximum results (default: 5)")]
    limit: Option<usize>,
    #[schemars(
        description = "Filter by source type: \"memory\" (legacy manual), \"session\", \"archive\", \"summary\", \"kb\""
    )]
    source: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MemoryListParams {
    #[schemars(description = "Filter by category (exact match)")]
    category: Option<String>,
    #[schemars(description = "Filter by project (exact match)")]
    project: Option<String>,
}

#[tool_router]
impl MemoryService {
    #[tool(
        description = "Return guidance for manual memories. Storage is disabled; project-local durable context should be written to docs/local/memory.md."
    )]
    async fn memory_write(&self, Parameters(params): Parameters<MemoryWriteParams>) -> String {
        match self.do_memory_write(params).await {
            Ok(msg) => msg,
            Err(e) => format!("Error: {}", e),
        }
    }

    #[tool(
        description = "Search user prompts, questions, and knowledge base. Use to find what was asked or discussed."
    )]
    async fn prompt_search(&self, Parameters(params): Parameters<SearchParams>) -> String {
        log("tool: prompt_search called");
        match self.do_search(params, COLLECTION_PROMPTS).await {
            Ok(r) => {
                log("tool: prompt_search ok");
                r
            }
            Err(e) => {
                log(&format!("tool: prompt_search error: {e}"));
                format!("Error: {e}")
            }
        }
    }

    #[tool(
        description = "Search assistant responses and solutions. Use to find how problems were solved."
    )]
    async fn answer_search(&self, Parameters(params): Parameters<SearchParams>) -> String {
        log("tool: answer_search called");
        match self.do_search(params, COLLECTION_ANSWERS).await {
            Ok(r) => {
                log("tool: answer_search ok");
                r
            }
            Err(e) => {
                log(&format!("tool: answer_search error: {e}"));
                format!("Error: {e}")
            }
        }
    }

    #[tool(
        description = "List all memory entries matching exact filters. Returns full content, no truncation. Use for loading all memories of a specific category/project."
    )]
    async fn memory_list(&self, Parameters(params): Parameters<MemoryListParams>) -> String {
        log("tool: memory_list called");
        match self.do_memory_list(params).await {
            Ok(r) => {
                log("tool: memory_list ok");
                r
            }
            Err(e) => {
                log(&format!("tool: memory_list error: {e}"));
                format!("Error: {e}")
            }
        }
    }
}

impl MemoryService {
    async fn do_memory_write(&self, params: MemoryWriteParams) -> Result<String> {
        let MemoryWriteParams {
            content,
            category,
            project,
        } = params;
        log(&format!(
            "memory_write disabled; content_bytes={}, category={:?}, project={:?}",
            content.len(),
            category,
            project
        ));
        Ok(manual_memory_write_guidance().to_string())
    }

    async fn do_memory_list(&self, params: MemoryListParams) -> Result<String> {
        let entries = claude_memory::memory_unit::list_manual_entries(
            params.category.as_deref(),
            params.project.as_deref(),
        )
        .await?;
        if entries.is_empty() {
            return Ok("No entries found.".to_string());
        }
        Ok(format_entries(&entries))
    }

    async fn do_search(&self, params: SearchParams, collection: &str) -> Result<String> {
        log_search_request(collection, &params.query);
        let client = get_qdrant().await?;
        let limit = params.limit.unwrap_or(5);
        let points = self.search_points(client, collection, &params).await?;
        if points.is_empty() {
            return Ok("No results found.".to_string());
        }
        let filtered = filter_search_results(&params.query, &points, limit).await;
        if filtered.is_empty() {
            return Ok("No results found.".to_string());
        }
        let graph_context = enrich_with_graph(&params.query).await;
        Ok(format_search_output(&filtered, &graph_context))
    }

    async fn run_hybrid_search(
        &self,
        client: &Qdrant,
        collection: &str,
        query_vec: Vec<f32>,
        query_text: &str,
        limit: usize,
        source: Option<&str>,
    ) -> Result<Vec<qdrant_client::qdrant::ScoredPoint>> {
        let over_fetch = ((limit * 4) as u64).max(20);
        let mut qb = make_hybrid_query(collection, query_vec, query_text, limit as u64, over_fetch);
        if let Some(src) = source {
            qb = qb.filter(Filter::must([Condition::matches(
                "source",
                src.to_string(),
            )]));
        }
        Ok(client
            .query(qb)
            .await
            .context("hybrid search failed")?
            .result)
    }

    async fn search_points(
        &self,
        client: &Qdrant,
        collection: &str,
        params: &SearchParams,
    ) -> Result<Vec<qdrant_client::qdrant::ScoredPoint>> {
        if !config::search_enabled() {
            return Ok(Vec::new());
        }

        let query_vec = self.embedder.embed(&params.query).await?;
        let points = self
            .run_hybrid_search(
                client,
                collection,
                query_vec,
                &params.query,
                20,
                params.source.as_deref(),
            )
            .await?;
        log_scores("raw", &points);
        Ok(points)
    }
}

/// Use LLM to filter results to only relevant ones. Falls back to top-N on failure.
async fn filter_with_llm<'a>(
    query: &str,
    points: &'a [qdrant_client::qdrant::ScoredPoint],
    fallback_limit: usize,
) -> Vec<&'a qdrant_client::qdrant::ScoredPoint> {
    let raw: Vec<RawResult> = points
        .iter()
        .map(|p| RawResult {
            text: get_payload_str(&p.payload, "text"),
            score: p.score,
        })
        .collect();
    if let Some(indices) = llm::filter_relevant(query, &raw).await {
        log(&format!(
            "LLM filter: {} relevant of {}",
            indices.len(),
            points.len()
        ));
        indices
            .iter()
            .filter_map(|&i| points.get(i.saturating_sub(1)))
            .collect()
    } else {
        points.iter().take(fallback_limit).collect()
    }
}

async fn filter_search_results<'a>(
    query: &str,
    points: &'a [qdrant_client::qdrant::ScoredPoint],
    limit: usize,
) -> Vec<&'a qdrant_client::qdrant::ScoredPoint> {
    const MIN_SCORE: f32 = 0.65;
    let filtered: Vec<_> = filter_with_llm(query, points, limit)
        .await
        .into_iter()
        .filter(|point| point.score >= MIN_SCORE)
        .collect();
    log_ref_scores("after filter", &filtered);
    filtered
}

fn log_search_request(collection: &str, query: &str) {
    log(&format!(
        "do_search: collection={collection} query={:?}",
        &query[..query.len().min(80)]
    ));
}

fn log_scores(stage: &str, points: &[qdrant_client::qdrant::ScoredPoint]) {
    log(&format!(
        "do_search: {} {} results, scores: {:?}",
        points.len(),
        stage,
        points.iter().map(|point| point.score).collect::<Vec<_>>()
    ));
}

fn log_ref_scores(stage: &str, points: &[&qdrant_client::qdrant::ScoredPoint]) {
    log(&format!(
        "do_search: {} {}, scores: {:?}",
        points.len(),
        stage,
        points.iter().map(|point| point.score).collect::<Vec<_>>()
    ));
}

fn format_search_output(
    points: &[&qdrant_client::qdrant::ScoredPoint],
    graph_context: &[String],
) -> String {
    let mut output: String = points
        .iter()
        .enumerate()
        .map(|(index, point)| format_scored_point(index, point))
        .collect();
    if graph_context.is_empty() {
        return output;
    }

    output.push_str("Related (graph):\n");
    for relation in graph_context {
        output.push_str(&format!("  - {relation}\n"));
    }
    output
}

/// Query graph for related entities to enrich search results.
async fn enrich_with_graph(query: &str) -> Vec<String> {
    enrich_with_graph_when_enabled(query, config::graph_enabled()).await
}

async fn enrich_with_graph_when_enabled(query: &str, graph_enabled: bool) -> Vec<String> {
    if !graph_enabled {
        return vec![];
    }

    let entities = graph::find_concepts(query).await;
    if entities.is_empty() {
        return vec![];
    }
    graph::query_related(&entities).unwrap_or_default()
}

fn format_entries(entries: &[(String, String, String)]) -> String {
    const MAX_BYTES: usize = 50_000;
    let total = entries.len();
    let mut output = format!("{total} entries found.\n\n");
    let mut shown = 0;
    for (i, (category, project, text)) in entries.iter().enumerate() {
        let entry = format_entry(i, category, project, text);
        if output.len() + entry.len() > MAX_BYTES {
            output.push_str(&format!(
                "[truncated at 50KB — showing {shown}/{total} entries]\n"
            ));
            return output;
        }
        output.push_str(&entry);
        shown += 1;
    }
    output
}

fn format_entry(i: usize, category: &str, project: &str, text: &str) -> String {
    let mut entry = format!("{}.", i + 1);
    if !category.is_empty() {
        entry.push_str(&format!(" [{}]", category));
    }
    if !project.is_empty() {
        entry.push_str(&format!(" ({})", project));
    }
    entry.push('\n');
    entry.push_str(text);
    entry.push_str("\n\n");
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn enrich_with_graph_returns_empty_when_graph_disabled() {
        let related = enrich_with_graph_when_enabled("Rust", false).await;

        assert!(related.is_empty());
    }

    #[tokio::test]
    async fn mcp_memory_write_returns_guidance_without_storage() {
        let service = MemoryService::new();
        let response = service
            .do_memory_write(MemoryWriteParams {
                content: "Remember this project-local detail.".to_string(),
                category: Some("context".to_string()),
                project: Some("claude-memory".to_string()),
            })
            .await
            .unwrap();

        assert!(response.contains("Manual memory writes are disabled"));
        assert!(response.contains("Do not store this in Qdrant"));
        assert!(response.contains("docs/local/memory.md"));
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Claude memory MCP server - search semantic memories across sessions; manual memory_write storage is disabled and returns docs/local guidance".into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let service = MemoryService::new();
    let server = service.serve(stdio()).await?;
    server.waiting().await?;
    Ok(())
}
