use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use claude_memory::kb_export::{
    ExportPlan, KnowledgeCollection, KnowledgePoint, classify_and_render,
    verify_source_plan_unchanged, verify_unclassified_count, verify_written_export, write_export,
};
use claude_memory::kb_search;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

const QDRANT_HTTP: &str = "http://127.0.0.1:6333";
const SCROLL_LIMIT: usize = 256;
const EXPECTED_UNCLASSIFIED_MEMORY_UNITS: usize = 222;

#[derive(Parser)]
#[command(name = "claude-memory-export-kb")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Classify durable knowledge without writing Markdown.
    Plan,
    /// Write canonical Markdown and rebuild the KB PageIndex.
    Apply {
        #[arg(long, default_value = kb_search::DEFAULT_KB_DIR)]
        kb_root: PathBuf,
    },
    /// Verify live source/count/content parity against the written export.
    Verify {
        #[arg(long, default_value = kb_search::DEFAULT_KB_DIR)]
        kb_root: PathBuf,
    },
}

#[derive(Deserialize)]
struct ApiResponse<T> {
    result: T,
}

#[derive(Deserialize)]
struct ScrollResult {
    points: Vec<RestPoint>,
    next_page_offset: Option<Value>,
}

#[derive(Deserialize)]
struct RestPoint {
    id: Value,
    payload: Value,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(1_800))
        .build()
        .context("failed to build Qdrant HTTP client")?;
    match Cli::parse().command {
        Command::Plan => print_plan(&build_live_plan(&client).await?),
        Command::Apply { kb_root } => apply_live_export(&client, &kb_root).await?,
        Command::Verify { kb_root } => {
            let plan = build_live_plan(&client).await?;
            verify_unclassified_count(&plan, EXPECTED_UNCLASSIFIED_MEMORY_UNITS)?;
            verify_export(&kb_root, &plan)?;
        }
    }
    Ok(())
}

async fn build_live_plan(client: &Client) -> Result<ExportPlan> {
    let mut points = scroll_collection(
        client,
        "claude-memory-units",
        KnowledgeCollection::MemoryUnits,
        false,
    )
    .await?;
    points.extend(
        scroll_collection(
            client,
            "claude-notable-facts",
            KnowledgeCollection::NotableFacts,
            false,
        )
        .await?,
    );
    points.extend(
        scroll_collection(
            client,
            "claude-memory",
            KnowledgeCollection::LegacyMemory,
            true,
        )
        .await?,
    );
    classify_and_render(&points)
}

async fn scroll_collection(
    client: &Client,
    collection_name: &str,
    collection: KnowledgeCollection,
    manual_only: bool,
) -> Result<Vec<KnowledgePoint>> {
    let mut points = Vec::new();
    let mut offset = None;
    loop {
        let result = scroll_page(client, collection_name, offset.take()).await?;
        for point in result.points {
            if manual_only && !is_manual_legacy_point(&point)? {
                continue;
            }
            points.push(KnowledgePoint {
                collection,
                id: point_id(&point.id),
                payload: point.payload,
            });
        }
        offset = result.next_page_offset;
        if offset.is_none() {
            return Ok(points);
        }
    }
}

async fn scroll_page(
    client: &Client,
    collection: &str,
    offset: Option<Value>,
) -> Result<ScrollResult> {
    let mut body = json!({
        "limit": SCROLL_LIMIT,
        "with_payload": true,
        "with_vector": false,
    });
    if let Some(offset) = offset {
        body["offset"] = offset;
    }
    let response: ApiResponse<ScrollResult> = client
        .post(format!(
            "{QDRANT_HTTP}/collections/{collection}/points/scroll"
        ))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .with_context(|| format!("failed to decode {collection} scroll response"))?;
    Ok(response.result)
}

fn point_id(id: &Value) -> String {
    id.as_str()
        .map(str::to_string)
        .unwrap_or_else(|| id.to_string())
}

fn is_manual_legacy_point(point: &RestPoint) -> Result<bool> {
    match point.payload.get("source") {
        Some(Value::String(source)) if source == "memory" => Ok(true),
        Some(Value::String(source))
            if matches!(source.as_str(), "session" | "archive" | "summary" | "kb") =>
        {
            Ok(false)
        }
        Some(Value::String(source)) => bail!(
            "legacy point {} has unsupported source {source}",
            point_id(&point.id)
        ),
        Some(_) => bail!("legacy point {} has non-string source", point_id(&point.id)),
        None => bail!("legacy point {} is missing source", point_id(&point.id)),
    }
}

async fn apply_live_export(client: &Client, kb_root: &Path) -> Result<()> {
    let before = build_live_plan(client).await?;
    verify_unclassified_count(&before, EXPECTED_UNCLASSIFIED_MEMORY_UNITS)?;
    let after = build_live_plan(client).await?;
    verify_source_plan_unchanged(&before, &after)?;
    write_export(kb_root, &after)?;
    let post_write = build_live_plan(client).await?;
    verify_source_plan_unchanged(&after, &post_write)?;
    verify_export(kb_root, &post_write)?;
    let index_dir = kb_search::default_index_dir();
    let summary = kb_search::build_index(kb_root, &index_dir)?;
    println!(
        "Rebuilt KB PageIndex: files={} nodes={} output={}",
        summary.files,
        summary.nodes,
        summary.index_path.display()
    );
    Ok(())
}

fn verify_export(kb_root: &Path, plan: &ExportPlan) -> Result<()> {
    let parity = verify_written_export(kb_root, plan)?;
    if parity.source_points != parity.manifest_entries {
        bail!(
            "source/manifest count mismatch: {} != {}",
            parity.source_points,
            parity.manifest_entries
        );
    }
    println!(
        "Verified KB export: source={} manifest={} unique={} content_hashes={}",
        parity.source_points,
        parity.manifest_entries,
        parity.unique_exported_records,
        parity.content_hashes_verified
    );
    print_plan(plan);
    Ok(())
}

fn print_plan(plan: &ExportPlan) {
    println!("raw source points: {}", plan.counts.raw);
    println!("unique durable exports: {}", plan.counts.exported_unique);
    println!("deduplicated source points: {}", plan.counts.duplicates);
    println!("quarantined records: {}", plan.counts.quarantined);
    println!(
        "excluded existing KB vectors: {}",
        plan.counts.excluded_kb_vectors
    );
    println!(
        "excluded non-durable records: {}",
        plan.counts.excluded_non_durable
    );
    for (destination, count) in &plan.counts.by_destination {
        println!("  {destination}: {count}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ids_preserve_strings_and_numbers() {
        assert_eq!(point_id(&json!(42)), "42");
        assert_eq!(point_id(&json!("point-1")), "point-1");
    }

    #[test]
    fn legacy_filter_validates_source_before_selecting_manual_records() {
        let manual = RestPoint {
            id: json!(1),
            payload: json!({"source":"memory","text":"manual"}),
        };
        let history = RestPoint {
            id: json!(2),
            payload: json!({"source":"session","text":"history"}),
        };
        let malformed = RestPoint {
            id: json!(3),
            payload: json!({"source":42,"text":"bad"}),
        };
        let unknown = RestPoint {
            id: json!(4),
            payload: json!({"source":"future","text":"unknown"}),
        };

        assert!(is_manual_legacy_point(&manual).unwrap());
        assert!(!is_manual_legacy_point(&history).unwrap());
        assert!(is_manual_legacy_point(&malformed).is_err());
        assert!(is_manual_legacy_point(&unknown).is_err());
    }
}
