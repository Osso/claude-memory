use anyhow::{Context, Result};
use claude_memory::{
    chunk::hash_text,
    embed::Embedder,
    llm,
    qdrant_hybrid::{build_named_vectors, ensure_hybrid_collection},
};
use qdrant_client::{
    Qdrant,
    qdrant::{
        Condition, DeletePointsBuilder, Filter, PointId, PointStruct, RetrievedPoint,
        ScrollPointsBuilder, UpsertPointsBuilder, Value, point_id::PointIdOptions,
        value::Kind::StringValue,
    },
};
use std::collections::HashMap;

const MEMORY_COLLECTION: &str = "claude-memory";

pub(crate) struct MemoryEntry {
    pub(crate) id: u64,
    pub(crate) text: String,
    pub(crate) category: String,
    pub(crate) project: String,
}

#[derive(Default)]
struct MergeStats {
    merged_count: u32,
    failed_count: u32,
    deleted_count: u32,
}

struct MergeSuccess {
    text_len: usize,
    deleted_count: u32,
}

pub(crate) async fn load_all_memories() -> Result<Vec<MemoryEntry>> {
    let client = memory_client().await?;
    let filter = Filter::must([Condition::matches("source", "memory".to_string())]);
    scroll_memory_entries(&client, filter).await
}

async fn memory_client() -> Result<Qdrant> {
    let client = Qdrant::from_url("http://localhost:6334")
        .build()
        .context("failed to connect to Qdrant")?;
    ensure_hybrid_collection(&client, MEMORY_COLLECTION).await?;
    Ok(client)
}

async fn scroll_memory_entries(client: &Qdrant, filter: Filter) -> Result<Vec<MemoryEntry>> {
    let mut entries = Vec::new();
    let mut offset: Option<PointId> = None;

    loop {
        let scroll = memory_scroll(filter.clone(), offset);
        let result = client.scroll(scroll).await.context("scroll failed")?;
        entries.extend(result.result.iter().map(point_to_entry));
        offset = result.next_page_offset;

        if offset.is_none() {
            return Ok(entries);
        }
    }
}

fn memory_scroll(filter: Filter, offset: Option<PointId>) -> ScrollPointsBuilder {
    let scroll = ScrollPointsBuilder::new(MEMORY_COLLECTION)
        .limit(100)
        .with_payload(true)
        .filter(filter);
    match offset {
        Some(offset) => scroll.offset(offset),
        None => scroll,
    }
}

fn point_to_entry(point: &RetrievedPoint) -> MemoryEntry {
    MemoryEntry {
        id: point_id(point),
        text: get_payload(&point.payload, "text"),
        category: get_payload(&point.payload, "category"),
        project: get_payload(&point.payload, "project"),
    }
}

fn point_id(point: &RetrievedPoint) -> u64 {
    point
        .id
        .as_ref()
        .and_then(|point| match &point.point_id_options {
            Some(PointIdOptions::Num(id)) => Some(*id),
            _ => None,
        })
        .unwrap_or(0)
}

fn get_payload(payload: &HashMap<String, Value>, key: &str) -> String {
    payload
        .get(key)
        .and_then(|value| value.kind.as_ref())
        .and_then(|kind| match kind {
            StringValue(text) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

pub(crate) async fn cluster_similar(
    entries: &[MemoryEntry],
    threshold: f32,
) -> Result<Vec<Vec<usize>>> {
    let embedder = Embedder::new();
    eprintln!("Embedding {} entries...", entries.len());
    let texts: Vec<&str> = entries.iter().map(|entry| entry.text.as_str()).collect();
    let embeddings = embedder.embed_batch(&texts).await?;
    eprintln!("Clustering...");
    Ok(greedy_cluster(&embeddings, threshold))
}

pub(crate) fn greedy_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let count = embeddings.len();
    let mut assigned = vec![false; count];
    let mut clusters = Vec::new();

    for i in 0..count {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![i];
        assigned[i] = true;
        for j in (i + 1)..count {
            if assigned[j] {
                continue;
            }
            if cosine_sim(&embeddings[i], &embeddings[j]) >= threshold {
                cluster.push(j);
                assigned[j] = true;
            }
        }

        clusters.push(cluster);
        if (i + 1) % 50 == 0 || i + 1 == count {
            eprint!("\r  Clustering: {}/{} entries processed", i + 1, count);
        }
    }

    eprintln!();
    clusters
}

pub(crate) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub(crate) fn print_clusters(entries: &[MemoryEntry], clusters: &[Vec<usize>]) {
    for cluster in clusters.iter().filter(|cluster| should_merge(cluster)) {
        println!("--- Cluster ({} entries) ---", cluster.len());
        for &idx in cluster {
            let entry = &entries[idx];
            println!("  [id={}] {}", entry.id, preview_text(&entry.text, 100));
        }
        println!();
    }
}

pub(crate) async fn merge_clusters(entries: &[MemoryEntry], clusters: &[Vec<usize>]) -> Result<()> {
    let client = memory_client().await?;
    let embedder = Embedder::new();
    let mergeable: Vec<&Vec<usize>> = clusters
        .iter()
        .filter(|cluster| should_merge(cluster))
        .collect();
    let mut stats = MergeStats::default();

    for (index, cluster) in mergeable.iter().enumerate() {
        print_cluster_header(entries, cluster, index + 1, mergeable.len());
        match merge_one_cluster(&client, &embedder, entries, cluster).await? {
            Some(success) => {
                stats.merged_count += 1;
                stats.deleted_count += success.deleted_count;
                eprintln!(
                    "    OK: merged into {} chars, deleted {} dupes",
                    success.text_len,
                    cluster.len() - 1
                );
            }
            None => {
                stats.failed_count += 1;
                eprintln!("    FAILED: LLM merge returned no result");
            }
        }
    }

    print_merge_summary(&stats);
    Ok(())
}

fn should_merge(cluster: &[usize]) -> bool {
    cluster.len() > 1
}

fn print_cluster_header(entries: &[MemoryEntry], cluster: &[usize], index: usize, total: usize) {
    let preview = preview_text(&entries[cluster[0]].text, 60);
    eprintln!(
        "\n  Cluster {index}/{total} ({} entries): {}",
        cluster.len(),
        preview
    );
    for &entry_index in &cluster[1..] {
        eprintln!("    + {}", preview_text(&entries[entry_index].text, 60));
    }
}

async fn merge_one_cluster(
    client: &Qdrant,
    embedder: &Embedder,
    entries: &[MemoryEntry],
    cluster: &[usize],
) -> Result<Option<MergeSuccess>> {
    let Some(text) = merge_cluster_texts(entries, cluster).await else {
        return Ok(None);
    };

    upsert_merged(client, embedder, &entries[cluster[0]], &text).await?;
    let deleted_count = delete_cluster_extras(client, entries, cluster).await?;
    Ok(Some(MergeSuccess {
        text_len: text.len(),
        deleted_count,
    }))
}

fn print_merge_summary(stats: &MergeStats) {
    eprintln!(
        "\nDone: {} merged, {} failed, {} duplicates removed",
        stats.merged_count, stats.failed_count, stats.deleted_count
    );
}

async fn upsert_merged(
    client: &Qdrant,
    embedder: &Embedder,
    keep: &MemoryEntry,
    text: &str,
) -> Result<()> {
    let embedding = embedder.embed(text).await?;
    let named = build_named_vectors(embedding, text);
    let payload = build_merged_payload(text, &keep.category, &keep.project);
    let point = PointStruct::new(keep.id, named, payload);
    client
        .upsert_points(UpsertPointsBuilder::new(MEMORY_COLLECTION, vec![point]))
        .await
        .context("upsert failed")?;
    Ok(())
}

fn build_merged_payload(text: &str, category: &str, project: &str) -> HashMap<String, Value> {
    [
        ("text", text.to_string().into()),
        ("source", "memory".to_string().into()),
        (
            "path",
            format!("daily/{}", chrono::Local::now().format("%Y-%m-%d.md")).into(),
        ),
        ("category", category.to_string().into()),
        ("project", project.to_string().into()),
        ("hash", hash_text(text).into()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value))
    .collect()
}

async fn delete_cluster_extras(
    client: &Qdrant,
    entries: &[MemoryEntry],
    cluster: &[usize],
) -> Result<u32> {
    let ids: Vec<u64> = cluster[1..].iter().map(|&idx| entries[idx].id).collect();
    if ids.is_empty() {
        return Ok(0);
    }

    let count = ids.len() as u32;
    let point_ids: Vec<PointId> = ids.into_iter().map(Into::into).collect();
    client
        .delete_points(DeletePointsBuilder::new(MEMORY_COLLECTION).points(point_ids))
        .await
        .context("delete failed")?;
    Ok(count)
}

async fn merge_cluster_texts(entries: &[MemoryEntry], cluster: &[usize]) -> Option<String> {
    let mut result = entries[cluster[0]].text.clone();
    for (step, &idx) in cluster[1..].iter().enumerate() {
        eprint!("    merging {}/{}...", step + 1, cluster.len() - 1);
        let merged = llm::merge_memories(&result, &entries[idx].text).await?;
        eprintln!(" ok ({} chars)", merged.len());
        result = merged;
    }
    Some(result)
}

pub(crate) fn preview_text(text: &str, max_len: usize) -> String {
    let oneline = text.replace('\n', " ");
    if oneline.len() > max_len {
        format!("{}...", &oneline[..max_len])
    } else {
        oneline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_sim_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let result = cosine_sim(&v, &v);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let result = cosine_sim(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_opposite() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0];
        let result = cosine_sim(&a, &b);
        assert!((result + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_sim_zero_vector() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let other = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine_sim(&zero, &other), 0.0);
        assert_eq!(cosine_sim(&zero, &zero), 0.0);
    }

    #[test]
    fn greedy_cluster_identical_items() {
        let v = vec![1.0f32, 0.0, 0.0];
        let embeddings = vec![v.clone(), v.clone(), v.clone()];
        let clusters = greedy_cluster(&embeddings, 0.99);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn greedy_cluster_completely_different() {
        let embeddings = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ];
        let clusters = greedy_cluster(&embeddings, 0.5);
        assert_eq!(clusters.len(), 3);
        assert!(clusters.iter().all(|cluster| cluster.len() == 1));
    }

    #[test]
    fn greedy_cluster_threshold_boundary() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert_eq!(greedy_cluster(&[a.clone(), b.clone()], 0.0).len(), 1);
        assert_eq!(greedy_cluster(&[a, b], 0.01).len(), 2);
    }

    #[test]
    fn greedy_cluster_empty() {
        let clusters = greedy_cluster(&[], 0.9);
        assert!(clusters.is_empty());
    }

    #[test]
    fn preview_text_short() {
        assert_eq!(preview_text("hello world", 50), "hello world");
    }

    #[test]
    fn preview_text_long() {
        let text = "a".repeat(100);
        assert_eq!(preview_text(&text, 20), format!("{}...", "a".repeat(20)));
    }

    #[test]
    fn preview_text_empty() {
        assert_eq!(preview_text("", 50), "");
    }

    #[test]
    fn preview_text_newlines_replaced() {
        assert_eq!(
            preview_text("line one\nline two\nline three", 200),
            "line one line two line three"
        );
    }

    #[test]
    fn preview_text_utf8_boundary() {
        let result = preview_text("😀😁😂", 4);
        assert!(result.starts_with("😀"));
        assert!(result.ends_with("..."));
    }
}
