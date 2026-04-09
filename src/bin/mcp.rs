//! MCP server for Claude memory operations.

use anyhow::{Context, Result};
use claude_memory::chunk::chunk_text;
use claude_memory::daily::{append_daily, append_kb_memory};
use claude_memory::embed::Embedder;
use claude_memory::graph;
use claude_memory::llm::{self, RawResult};
use claude_memory::qdrant_hybrid::{BM25_MODEL, build_named_vectors, ensure_hybrid_collection};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, Document, Filter, Fusion, PointStruct, PrefetchQueryBuilder, Query,
    QueryPointsBuilder, ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder,
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
use std::sync::atomic::{AtomicU64, Ordering};
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
            Ok(client)
        })
        .await;
    log("get_qdrant: done");
    result
}

/// Build a payload HashMap from a list of key-value pairs.
fn build_payload(
    fields: Vec<(&str, qdrant_client::qdrant::Value)>,
) -> HashMap<String, qdrant_client::qdrant::Value> {
    fields
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
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
    next_id: Arc<AtomicU64>,
    tool_router: ToolRouter<Self>,
}

impl MemoryService {
    fn new() -> Self {
        Self {
            embedder: Arc::new(Embedder::new()),
            next_id: Arc::new(AtomicU64::new(0)),
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MemoryWriteParams {
    content: String,
    #[schemars(description = "Category: correction, preference, context, learning, decision")]
    category: Option<String>,
    #[schemars(description = "Project name if this memory is project-specific")]
    project: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    query: String,
    #[schemars(description = "Maximum results (default: 5)")]
    limit: Option<usize>,
    #[schemars(
        description = "Filter by source type: \"memory\" (manually added), \"session\", \"archive\", \"summary\", \"kb\""
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
        description = "Store a memory entry. Use for: corrections (when user corrects a mistake), preferences (how user likes things done), learnings (new knowledge), decisions (architectural choices), context (ongoing project state)."
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
        append_daily(
            &params.content,
            params.category.as_deref(),
            params.project.as_deref(),
        )?;
        if let Err(e) = append_kb_memory(
            &params.content,
            params.category.as_deref(),
            params.project.as_deref(),
        ) {
            log(&format!("KB memory write failed (non-fatal): {e}"));
        }
        let chunks = chunk_text(&params.content);
        if chunks.is_empty() {
            return Ok("Memory stored (empty content, not indexed)".to_string());
        }
        let client = get_qdrant().await?;
        let (mut indexed, mut merged) = (0u32, 0u32);
        for chunk in &chunks {
            match self
                .try_dedup_or_insert(client, chunk, &params.category, &params.project)
                .await
            {
                Ok(true) => merged += 1,
                Ok(false) => indexed += 1,
                Err(e) => log(&format!("memory_write chunk error: {e}")),
            }
        }
        // Graph extraction (non-blocking, failures logged)
        if let Err(e) = graph::extract_and_store(&params.content).await {
            log(&format!("graph extract_and_store failed: {e}"));
        }
        Ok(format_write_result(indexed, merged))
    }

    /// Try to merge with an existing similar memory. Returns Ok(true) if merged, Ok(false) if inserted new.
    async fn try_dedup_or_insert(
        &self,
        client: &Qdrant,
        chunk: &claude_memory::chunk::Chunk,
        category: &Option<String>,
        project: &Option<String>,
    ) -> Result<bool> {
        const DEDUP_THRESHOLD: f32 = 0.88;
        if let Some((existing_id, existing_text, score)) = self
            .find_similar_memory(client, &chunk.text, COLLECTION_PROMPTS)
            .await?
        {
            if score >= DEDUP_THRESHOLD {
                if let Some(merged_text) = llm::merge_memories(&existing_text, &chunk.text).await {
                    log(&format!(
                        "dedup: merging with point {existing_id} (score {score:.3})"
                    ));
                    return self
                        .replace_memory_point(client, existing_id, &merged_text, category, project)
                        .await
                        .map(|()| true);
                }
            }
        }
        let point = self.build_memory_point(chunk, category, project).await?;
        client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_PROMPTS, vec![point]))
            .await
            .context("failed to index")?;
        Ok(false)
    }

    /// Replace an existing memory point with merged content.
    async fn replace_memory_point(
        &self,
        client: &Qdrant,
        id: u64,
        text: &str,
        category: &Option<String>,
        project: &Option<String>,
    ) -> Result<()> {
        let embedding = self
            .embedder
            .embed(text)
            .await
            .context("embedding failed")?;
        let named = build_named_vectors(embedding, text);
        let path = format!("daily/{}", chrono::Local::now().format("%Y-%m-%d.md"));
        let payload = build_payload(vec![
            ("text", text.to_string().into()),
            ("source", "memory".to_string().into()),
            ("path", path.into()),
            ("category", category.clone().unwrap_or_default().into()),
            ("project", project.clone().unwrap_or_default().into()),
            ("hash", claude_memory::chunk::hash_text(text).into()),
        ]);
        let point = PointStruct::new(id, named, payload);
        client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_PROMPTS, vec![point]))
            .await
            .context("failed to replace point")?;
        Ok(())
    }

    async fn build_memory_point(
        &self,
        chunk: &claude_memory::chunk::Chunk,
        category: &Option<String>,
        project: &Option<String>,
    ) -> Result<PointStruct> {
        let embedding = self
            .embedder
            .embed(&chunk.text)
            .await
            .context("embedding failed")?;
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let path = format!("daily/{}", chrono::Local::now().format("%Y-%m-%d.md"));
        let named = build_named_vectors(embedding, &chunk.text);
        let payload = build_payload(vec![
            ("text", chunk.text.clone().into()),
            ("source", "memory".to_string().into()),
            ("path", path.into()),
            ("category", category.clone().unwrap_or_default().into()),
            ("project", project.clone().unwrap_or_default().into()),
            ("hash", chunk.hash.clone().into()),
        ]);
        Ok(PointStruct::new(id, named, payload))
    }

    async fn do_memory_list(&self, params: MemoryListParams) -> Result<String> {
        let filter = build_memory_filter(&params);
        let entries = scroll_all_entries(filter).await?;
        if entries.is_empty() {
            return Ok("No entries found.".to_string());
        }
        Ok(format_entries(&entries))
    }

    async fn do_search(&self, params: SearchParams, collection: &str) -> Result<String> {
        log(&format!(
            "do_search: collection={collection} query={:?}",
            &params.query[..params.query.len().min(80)]
        ));
        let client = get_qdrant().await?;
        let limit = params.limit.unwrap_or(5);
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
        log(&format!(
            "do_search: {} raw results, scores: {:?}",
            points.len(),
            points.iter().map(|p| p.score).collect::<Vec<_>>()
        ));
        if points.is_empty() {
            return Ok("No results found.".to_string());
        }
        const MIN_SCORE: f32 = 0.65;
        let filtered = filter_with_llm(&params.query, &points, limit).await;
        let filtered: Vec<_> = filtered
            .into_iter()
            .filter(|p| p.score >= MIN_SCORE)
            .collect();
        log(&format!(
            "do_search: {} after filter, scores: {:?}",
            filtered.len(),
            filtered.iter().map(|p| p.score).collect::<Vec<_>>()
        ));
        if filtered.is_empty() {
            return Ok("No results found.".to_string());
        }
        let graph_context = enrich_with_graph(&params.query).await;
        let mut output: String = filtered
            .iter()
            .enumerate()
            .map(|(i, p)| format_scored_point(i, p))
            .collect();
        if !graph_context.is_empty() {
            output.push_str("Related (graph):\n");
            for r in &graph_context {
                output.push_str(&format!("  - {r}\n"));
            }
        }
        Ok(output)
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

    /// Find a similar existing memory for deduplication. Returns (id, text, score).
    async fn find_similar_memory(
        &self,
        client: &Qdrant,
        text: &str,
        collection: &str,
    ) -> Result<Option<(u64, String, f32)>> {
        let embedding = self.embedder.embed(text).await?;
        let search = SearchPointsBuilder::new(collection, embedding, 1)
            .vector_name("dense")
            .with_payload(true)
            .filter(Filter::must([Condition::matches(
                "source",
                "memory".to_string(),
            )]));
        let results = client.search_points(search).await?;
        let Some(point) = results.result.into_iter().next() else {
            return Ok(None);
        };
        let id = match point.id.and_then(|p| p.point_id_options) {
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n,
            _ => return Ok(None),
        };
        Ok(Some((
            id,
            get_payload_str(&point.payload, "text"),
            point.score,
        )))
    }
}

fn format_write_result(indexed: u32, merged: u32) -> String {
    match (indexed, merged) {
        (0, 0) => "Memory stored (no chunks indexed)".to_string(),
        (i, 0) => format!(
            "Memory stored and indexed ({i} chunk{})",
            if i == 1 { "" } else { "s" }
        ),
        (0, m) => format!(
            "Memory stored and merged with {m} existing entry{}",
            if m == 1 { "" } else { "ies" }
        ),
        (i, m) => format!("Memory stored ({i} indexed, {m} merged)"),
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

/// Query graph for related entities to enrich search results.
async fn enrich_with_graph(query: &str) -> Vec<String> {
    let entities = graph::find_concepts(query).await;
    if entities.is_empty() {
        return vec![];
    }
    graph::query_related(&entities).unwrap_or_default()
}

fn build_memory_filter(params: &MemoryListParams) -> qdrant_client::qdrant::Filter {
    let mut conditions = vec![Condition::matches("source", "memory".to_string())];
    if let Some(cat) = &params.category {
        conditions.push(Condition::matches("category", cat.clone()));
    }
    if let Some(proj) = &params.project {
        conditions.push(Condition::matches("project", proj.clone()));
    }
    Filter::must(conditions)
}

async fn scroll_all_entries(filter: Filter) -> Result<Vec<(String, String, String)>> {
    let client = get_qdrant().await?;
    let mut entries = Vec::new();
    let mut offset: Option<qdrant_client::qdrant::PointId> = None;
    loop {
        let mut scroll = ScrollPointsBuilder::new(COLLECTION_PROMPTS)
            .limit(100)
            .with_payload(true)
            .filter(filter.clone());
        if let Some(off) = offset {
            scroll = scroll.offset(off);
        }
        let result = client.scroll(scroll).await.context("scroll failed")?;
        for point in &result.result {
            entries.push((
                get_payload_str(&point.payload, "category"),
                get_payload_str(&point.payload, "project"),
                get_payload_str(&point.payload, "text"),
            ));
        }
        offset = result.next_page_offset;
        if offset.is_none() {
            break;
        }
    }
    Ok(entries)
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

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Claude memory MCP server - store and search semantic memories across sessions"
                    .into(),
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
