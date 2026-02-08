//! MCP server for Claude memory operations.

use anyhow::{Context, Result};
use claude_memory::chunk::chunk_text;
use claude_memory::daily::append_daily;
use claude_memory::embed::Embedder;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ServerHandler, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};
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
const VECTOR_SIZE: u64 = 4096;

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
            ensure_collection(&client, COLLECTION_PROMPTS).await?;
            log("get_qdrant: prompts collection ok");
            ensure_collection(&client, COLLECTION_ANSWERS).await?;
            log("get_qdrant: answers collection ok");
            Ok(client)
        })
        .await;
    log("get_qdrant: done");
    result
}

async fn ensure_collection(client: &Qdrant, name: &str) -> Result<()> {
    let collections = client.list_collections().await?;
    if !collections.collections.iter().any(|c| c.name == name) {
        client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine)),
            )
            .await
            .context("failed to create collection")?;
    }
    Ok(())
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
    /// The memory content to store. Can be multi-line.
    content: String,

    /// Category: correction, preference, context, learning, decision
    #[schemars(description = "Category: correction, preference, context, learning, decision")]
    category: Option<String>,

    /// Project context if relevant (e.g., "globalcomix", "sakuin")
    #[schemars(description = "Project name if this memory is project-specific")]
    project: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Search query
    query: String,

    /// Maximum results to return
    #[schemars(description = "Maximum results (default: 5)")]
    limit: Option<usize>,

    /// Filter by source type
    #[schemars(description = "Filter by source type: \"memory\" (manually added), \"session\", \"archive\", \"summary\", \"kb\"")]
    source: Option<String>,
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
            Ok(results) => { log("tool: prompt_search ok"); results },
            Err(e) => { log(&format!("tool: prompt_search error: {e}")); format!("Error: {}", e) },
        }
    }

    #[tool(
        description = "Search assistant responses and solutions. Use to find how problems were solved."
    )]
    async fn answer_search(&self, Parameters(params): Parameters<SearchParams>) -> String {
        log("tool: answer_search called");
        match self.do_search(params, COLLECTION_ANSWERS).await {
            Ok(results) => { log("tool: answer_search ok"); results },
            Err(e) => { log(&format!("tool: answer_search error: {e}")); format!("Error: {}", e) },
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

        let chunks = chunk_text(&params.content);
        if chunks.is_empty() {
            return Ok("Memory stored (empty content, not indexed)".to_string());
        }

        let client = get_qdrant().await?;
        let mut indexed = 0;
        for chunk in &chunks {
            let point = self
                .build_memory_point(chunk, &params.category, &params.project)
                .await?;
            client
                .upsert_points(UpsertPointsBuilder::new(COLLECTION_PROMPTS, vec![point]))
                .await
                .context("failed to index")?;
            indexed += 1;
        }

        Ok(format!(
            "Memory stored and indexed ({} chunk{})",
            indexed,
            if indexed == 1 { "" } else { "s" }
        ))
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

        Ok(PointStruct::new(
            id,
            embedding,
            [
                ("text", chunk.text.clone().into()),
                ("source", "memory".to_string().into()),
                ("path", path.into()),
                ("category", category.clone().unwrap_or_default().into()),
                ("project", project.clone().unwrap_or_default().into()),
                ("hash", chunk.hash.clone().into()),
            ],
        ))
    }

    async fn do_search(&self, params: SearchParams, collection: &str) -> Result<String> {
        log(&format!("do_search: collection={collection} query={:?}", &params.query[..params.query.len().min(80)]));

        let client = get_qdrant().await?;
        let limit = params.limit.unwrap_or(5);

        log("do_search: embedding...");
        let query_vec = self.embedder.embed(&params.query).await?;
        log("do_search: embedding done, searching qdrant...");

        let mut search =
            SearchPointsBuilder::new(collection, query_vec, limit as u64).with_payload(true);

        if let Some(src) = &params.source {
            search = search.filter(Filter::must([Condition::matches("source", src.clone())]));
        }

        let results = client
            .search_points(search)
            .await
            .context("search failed")?;
        log(&format!("do_search: done, {} results", results.result.len()));

        if results.result.is_empty() {
            return Ok("No results found.".to_string());
        }

        let mut output = String::new();
        for (i, point) in results.result.iter().enumerate() {
            let text = get_payload_string(&point.payload, "text");
            let source = get_payload_string(&point.payload, "source");
            let path = get_payload_string(&point.payload, "path");

            output.push_str(&format!(
                "{}. [{}] {} (score: {:.3})\n",
                i + 1,
                source,
                path,
                point.score
            ));

            // Truncate long texts (char-boundary safe)
            let display_text = if text.len() > 300 {
                let end = text.floor_char_boundary(300);
                format!("{}...", &text[..end])
            } else {
                text
            };
            output.push_str(&format!("   {}\n\n", display_text.replace('\n', " ")));
        }

        Ok(output)
    }
}

fn get_payload_string(
    payload: &std::collections::HashMap<String, qdrant_client::qdrant::Value>,
    key: &str,
) -> String {
    payload
        .get(key)
        .and_then(|v| v.kind.as_ref())
        .and_then(|k| match k {
            qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
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
