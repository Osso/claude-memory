use super::{get_graph, sanitize, store_triplet};
use anyhow::Result;
use cozo::{DataValue, DbInstance, ScriptMutability};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Default, Clone, Copy)]
pub struct GraphCleanStats {
    pub passes: usize,
    pub relationships_seen: usize,
    pub relationships_kept: usize,
    pub relationships_removed: usize,
    pub relationships_rewritten: usize,
    pub entities_removed: usize,
}

pub fn clean_graph(max_passes: usize, dry_run: bool) -> Result<GraphCleanStats> {
    if dry_run {
        let mut stats = clean_pass(get_graph()?, true)?;
        stats.passes = 1;
        return Ok(stats);
    }

    let db = get_graph()?;
    backup_graph_db()?;
    let mut total = GraphCleanStats::default();

    for pass in 0..max_passes.max(1) {
        let pass_stats = clean_pass(db, dry_run)?;
        accumulate_stats(&mut total, pass + 1, pass_stats);

        if pass_is_stable(pass_stats) {
            break;
        }
    }

    Ok(total)
}

pub fn clear_graph() -> Result<()> {
    let db = get_graph()?;
    let relationships = load_relationship_rows(db)?;
    for (src, relation, dst, _) in &relationships {
        remove_relationship(db, src, relation, dst)?;
    }

    let entities = load_entity_rows(db)?;
    for (name, entity_type) in &entities {
        remove_entity(db, name, entity_type)?;
    }

    Ok(())
}

fn clean_pass(db: &DbInstance, dry_run: bool) -> Result<GraphCleanStats> {
    let relationships = load_relationship_rows(db)?;
    let entities = load_entity_rows(db)?;
    let (cleaned, mut stats) = sanitize_relationships(&relationships);
    stats.entities_removed = count_removed_entities(&entities, &cleaned);

    if dry_run {
        return Ok(stats);
    }

    rewrite_graph(db, &relationships, &entities, &cleaned)?;
    Ok(stats)
}

fn sanitize_relationships(
    relationships: &[(String, String, String, String)],
) -> (Vec<(String, String, String, String)>, GraphCleanStats) {
    let mut cleaned = Vec::new();
    let mut seen = BTreeSet::new();
    let mut stats = GraphCleanStats {
        relationships_seen: relationships.len(),
        ..GraphCleanStats::default()
    };

    for (src, relation, dst, source_text) in relationships {
        if let Some((new_src, new_relation, new_dst)) =
            sanitize::sanitize_triplet(src, relation, dst)
        {
            track_cleaned_triplet(
                &mut cleaned,
                &mut seen,
                &mut stats,
                src,
                relation,
                dst,
                source_text,
                new_src,
                new_relation,
                new_dst,
            );
        } else {
            stats.relationships_removed += 1;
        }
    }

    (cleaned, stats)
}

fn track_cleaned_triplet(
    cleaned: &mut Vec<(String, String, String, String)>,
    seen: &mut BTreeSet<(String, String, String)>,
    stats: &mut GraphCleanStats,
    src: &str,
    relation: &str,
    dst: &str,
    source_text: &str,
    new_src: String,
    new_relation: String,
    new_dst: String,
) {
    let was_rewritten = new_src != src || new_relation != relation || new_dst != dst;
    if was_rewritten {
        stats.relationships_rewritten += 1;
    }
    if seen.insert((new_src.clone(), new_relation.clone(), new_dst.clone())) {
        cleaned.push((new_src, new_relation, new_dst, source_text.to_string()));
        stats.relationships_kept += 1;
    } else {
        stats.relationships_removed += 1;
    }
}

fn count_removed_entities(
    entities: &[(String, String)],
    cleaned: &[(String, String, String, String)],
) -> usize {
    let valid_entities: BTreeSet<_> = cleaned
        .iter()
        .flat_map(|(src, _, dst, _)| [src.clone(), dst.clone()])
        .collect();

    entities
        .iter()
        .filter(|(name, _)| !valid_entities.contains(name))
        .count()
}

fn rewrite_graph(
    db: &DbInstance,
    relationships: &[(String, String, String, String)],
    entities: &[(String, String)],
    cleaned: &[(String, String, String, String)],
) -> Result<()> {
    for (src, relation, dst, _) in relationships {
        remove_relationship(db, src, relation, dst)?;
    }
    for (name, entity_type) in entities {
        remove_entity(db, name, entity_type)?;
    }
    for (src, relation, dst, source_text) in cleaned {
        store_triplet(db, src, relation, dst, source_text)?;
    }
    Ok(())
}

fn accumulate_stats(total: &mut GraphCleanStats, pass: usize, pass_stats: GraphCleanStats) {
    total.passes = pass;
    total.relationships_seen += pass_stats.relationships_seen;
    total.relationships_kept += pass_stats.relationships_kept;
    total.relationships_removed += pass_stats.relationships_removed;
    total.relationships_rewritten += pass_stats.relationships_rewritten;
    total.entities_removed += pass_stats.entities_removed;
}

fn pass_is_stable(stats: GraphCleanStats) -> bool {
    stats.relationships_removed == 0
        && stats.relationships_rewritten == 0
        && stats.entities_removed == 0
}

fn backup_graph_db() -> Result<()> {
    let path = dirs::home_dir()
        .expect("no home directory")
        .join(".claude/memory/graph.db");
    if !path.exists() {
        return Ok(());
    }
    let backup = path.with_extension("db.bak");
    std::fs::copy(&path, backup)?;
    Ok(())
}

fn load_relationship_rows(db: &DbInstance) -> Result<Vec<(String, String, String, String)>> {
    let rows = db
        .run_script(
            "?[src, relation, dst, source_text] := *relationships{src, relation, dst, source_text}",
            BTreeMap::new(),
            ScriptMutability::Immutable,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(rows
        .rows
        .into_iter()
        .map(|row| {
            (
                str_row(&row, 0),
                str_row(&row, 1),
                str_row(&row, 2),
                str_row(&row, 3),
            )
        })
        .collect())
}

fn load_entity_rows(db: &DbInstance) -> Result<Vec<(String, String)>> {
    let rows = db
        .run_script(
            "?[name, entity_type] := *entities{name, entity_type}",
            BTreeMap::new(),
            ScriptMutability::Immutable,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(rows
        .rows
        .into_iter()
        .map(|row| (str_row(&row, 0), str_row(&row, 1)))
        .collect())
}

fn remove_relationship(db: &DbInstance, src: &str, relation: &str, dst: &str) -> Result<()> {
    let params = BTreeMap::from([
        ("src".to_string(), DataValue::Str(src.into())),
        ("rel".to_string(), DataValue::Str(relation.into())),
        ("dst".to_string(), DataValue::Str(dst.into())),
    ]);

    db.run_script(
        "?[src, relation, dst] <- [[$src, $rel, $dst]] :rm relationships {src, relation, dst}",
        params,
        ScriptMutability::Mutable,
    )
    .map_err(|e| anyhow::anyhow!("graph remove relationship failed: {e}"))?;

    Ok(())
}

fn remove_entity(db: &DbInstance, name: &str, entity_type: &str) -> Result<()> {
    let params = BTreeMap::from([
        ("name".to_string(), DataValue::Str(name.into())),
        (
            "entity_type".to_string(),
            DataValue::Str(entity_type.into()),
        ),
    ]);

    db.run_script(
        "?[name, entity_type] <- [[$name, $entity_type]] :rm entities {name, entity_type}",
        params,
        ScriptMutability::Mutable,
    )
    .map_err(|e| anyhow::anyhow!("graph remove entity failed: {e}"))?;

    Ok(())
}

fn str_row(row: &[DataValue], idx: usize) -> String {
    row.get(idx)
        .and_then(|value| value.get_str())
        .unwrap_or("")
        .to_string()
}
