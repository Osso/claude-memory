use anyhow::Result;
use claude_memory::{config, graph};
use std::path::{Path, PathBuf};

use crate::dedup::load_all_memories;

pub(crate) fn run_graph_clean_cmd(max_passes: usize, dry_run: bool) -> Result<()> {
    let stats = graph::clean_graph(max_passes, dry_run)?;
    eprintln!(
        "Graph clean: {} pass(es), {} relationships seen, {} kept, {} removed, {} rewritten, {} entities removed",
        stats.passes,
        stats.relationships_seen,
        stats.relationships_kept,
        stats.relationships_removed,
        stats.relationships_rewritten,
        stats.entities_removed
    );
    Ok(())
}

pub(crate) async fn run_build_graph(kb: bool, fresh: bool) -> Result<()> {
    if fresh {
        graph::clear_graph()?;
    }
    let entries = load_all_memories().await?;
    let mut extracted = extract_texts_to_graph(
        &entries.iter().map(|e| e.text.as_str()).collect::<Vec<_>>(),
        "memory",
    )
    .await?;
    if kb {
        let kb_texts = load_kb_texts();
        extracted += extract_texts_to_graph(
            &kb_texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            "KB",
        )
        .await?;
    }
    eprintln!("Total: {extracted} triplets");
    Ok(())
}

async fn extract_texts_to_graph(texts: &[&str], label: &str) -> Result<usize> {
    eprintln!(
        "Processing {} {label} entries for graph extraction",
        texts.len()
    );
    let mut extracted = 0;
    for (i, text) in texts.iter().enumerate() {
        if config::graph_enabled() {
            match graph::extract_and_store(text).await {
                Ok(n) => extracted += n,
                Err(e) => eprintln!("  entry {}: {e}", i + 1),
            }
        }
        eprint!("\r  {}/{} ({} triplets)", i + 1, texts.len(), extracted);
    }
    eprintln!();
    Ok(extracted)
}

fn load_kb_texts() -> Vec<String> {
    let kb_dir = PathBuf::from("/syncthing/Sync/KB");
    let include_dirs = ["dev", "guides", "research", "state", "memory"];
    let texts: Vec<String> = include_dirs
        .iter()
        .flat_map(|dir_name| collect_kb_chunks(&kb_dir.join(dir_name)))
        .collect();
    eprintln!("Loaded {} KB chunks from {:?}", texts.len(), include_dirs);
    texts
}

fn collect_kb_chunks(dir: &Path) -> Vec<String> {
    if !dir.exists() {
        return vec![];
    }

    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .map(|entry| entry.into_path())
        .filter(|path| is_markdown(path))
        .flat_map(|path| read_chunked_markdown(&path))
        .collect()
}

fn is_markdown(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "md")
}

fn read_chunked_markdown(path: &Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return vec![];
    };

    claude_memory::chunk::chunk_text(&content)
        .into_iter()
        .map(|chunk| chunk.text)
        .collect()
}

pub(crate) fn run_graph_dump(limit: usize) -> Result<()> {
    let db = graph::get_graph()?;
    let entities = db
        .run_script(
            &format!("?[name, type] := *entities{{name, entity_type: type}} :limit {limit}"),
            std::collections::BTreeMap::new(),
            cozo::ScriptMutability::Immutable,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("=== Entities ({} shown) ===", entities.rows.len());
    for row in &entities.rows {
        let name = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let etype = row.get(1).and_then(|v| v.get_str()).unwrap_or("");
        println!("  {name} [{etype}]");
    }

    let rels = db
        .run_script(
            &format!(
                "?[src, rel, dst] := *relationships{{src, relation: rel, dst}} :limit {limit}"
            ),
            std::collections::BTreeMap::new(),
            cozo::ScriptMutability::Immutable,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("\n=== Relationships ({} shown) ===", rels.rows.len());
    for row in &rels.rows {
        let src = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let rel = row.get(1).and_then(|v| v.get_str()).unwrap_or("");
        let dst = row.get(2).and_then(|v| v.get_str()).unwrap_or("");
        println!("  {src} --[{rel}]--> {dst}");
    }
    Ok(())
}
