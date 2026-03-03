//! CozoDB embedded graph database for entity-relationship storage.

use anyhow::{Context, Result};
use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability};
use std::collections::BTreeMap;
use std::sync::OnceLock;

static GRAPH: OnceLock<DbInstance> = OnceLock::new();

const EXTRACT_SYSTEM: &str = "Extract entity-relationship triplets from the text. \
Return a JSON array of [subject, relation, object] arrays. \
Focus on: project names, technologies, people, tools, architectural decisions, preferences. \
Entity names should be proper nouns or short noun phrases (1-3 words). \
NEVER use as entities: file paths, CLI flags, code snippets, numbers, coordinates, regex patterns, variable names. \
Good entities: \"authd\", \"Rust\", \"Retribution Paladin\", \"Qdrant\", \"GlobalComix\" \
Bad entities: \"/etc/authd/policies.d/\", \"--json\", \"0.9998\", \"$parent substitution\" \
Example: [[\"authd\", \"written_in\", \"Rust\"], [\"authd\", \"replaces\", \"polkit\"]] \
If no clear relationships exist, return: []";

pub fn get_graph() -> Result<&'static DbInstance> {
    if let Some(db) = GRAPH.get() {
        return Ok(db);
    }

    let path = dirs::home_dir()
        .expect("no home directory")
        .join(".claude/memory/graph.db");

    std::fs::create_dir_all(path.parent().unwrap())?;

    let db = DbInstance::new("sqlite", path.to_str().unwrap(), Default::default())
        .map_err(|e| anyhow::anyhow!("failed to open graph db: {e}"))?;

    ensure_schema(&db)?;

    GRAPH
        .set(db)
        .map_err(|_| anyhow::anyhow!("graph already initialized"))?;
    Ok(GRAPH.get().unwrap())
}

fn ensure_schema(db: &DbInstance) -> Result<()> {
    let _ = db.run_script(
        ":create entities {name: String, entity_type: String => source_text: String}",
        BTreeMap::new(),
        ScriptMutability::Mutable,
    );

    let _ = db.run_script(
        ":create relationships {src: String, relation: String, dst: String => source_text: String}",
        BTreeMap::new(),
        ScriptMutability::Mutable,
    );

    Ok(())
}

pub async fn extract_and_store(text: &str) -> Result<usize> {
    let db = get_graph()?;
    let triplets = extract_triplets(text).await?;

    let mut stored = 0;
    for (subject, relation, object) in &triplets {
        if let Err(e) = store_triplet(db, subject, relation, object, text) {
            tracing::warn!("graph store_triplet failed: {e}");
        } else {
            stored += 1;
        }
    }

    Ok(stored)
}

async fn extract_triplets(text: &str) -> Result<Vec<(String, String, String)>> {
    let key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => return Ok(vec![]),
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()?;

    let body = serde_json::json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "system": EXTRACT_SYSTEM,
        "messages": [{"role": "user", "content": text}]
    });

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .context("failed to call Anthropic API")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Anthropic API error {}: {}", status, body);
    }

    let json: serde_json::Value = resp.json().await.context("failed to parse API response")?;

    let content_text = json["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|block| block["text"].as_str())
        .unwrap_or("[]");

    parse_triplets(content_text)
}

fn parse_triplets(text: &str) -> Result<Vec<(String, String, String)>> {
    // Find the JSON array in the response (model may wrap it in prose)
    let start = text.find('[').unwrap_or(0);
    let end = text.rfind(']').map(|i| i + 1).unwrap_or(text.len());
    let json_str = &text[start..end];

    let arr: serde_json::Value = serde_json::from_str(json_str).unwrap_or(serde_json::json!([]));

    let mut triplets = Vec::new();
    if let Some(outer) = arr.as_array() {
        for item in outer {
            if let Some(inner) = item.as_array() {
                if inner.len() == 3 {
                    let s = inner[0].as_str().unwrap_or("").to_string();
                    let r = inner[1].as_str().unwrap_or("").to_string();
                    let o = inner[2].as_str().unwrap_or("").to_string();
                    if !s.is_empty() && !r.is_empty() && !o.is_empty() {
                        triplets.push((s, r, o));
                    }
                }
            }
        }
    }

    Ok(triplets)
}

fn store_triplet(
    db: &DbInstance,
    subject: &str,
    relation: &str,
    object: &str,
    source_text: &str,
) -> Result<()> {
    store_entity(db, subject, source_text)?;
    store_entity(db, object, source_text)?;

    let params = BTreeMap::from([
        ("src".to_string(), DataValue::Str(subject.into())),
        ("rel".to_string(), DataValue::Str(relation.into())),
        ("dst".to_string(), DataValue::Str(object.into())),
        ("source".to_string(), DataValue::Str(source_text.into())),
    ]);

    db.run_script(
        "?[src, relation, dst, source_text] <- [[$src, $rel, $dst, $source]] \
         :put relationships {src, relation, dst => source_text}",
        params,
        ScriptMutability::Mutable,
    )
    .map_err(|e| anyhow::anyhow!("graph store relationship failed: {e}"))?;

    Ok(())
}

fn store_entity(db: &DbInstance, name: &str, source_text: &str) -> Result<()> {
    let params = BTreeMap::from([
        ("name".to_string(), DataValue::Str(name.into())),
        ("source".to_string(), DataValue::Str(source_text.into())),
    ]);

    db.run_script(
        "?[name, entity_type, source_text] <- [[$name, 'entity', $source]] \
         :put entities {name, entity_type => source_text}",
        params,
        ScriptMutability::Mutable,
    )
    .map_err(|e| anyhow::anyhow!("graph store entity failed: {e}"))?;

    Ok(())
}

pub fn query_related(entities: &[String]) -> Result<Vec<String>> {
    let db = get_graph()?;
    let mut related = Vec::new();

    for entity in entities {
        let outgoing = query_outgoing(db, entity);
        let incoming = query_incoming(db, entity);

        match outgoing {
            Ok(rows) => {
                for row in rows.rows {
                    if let (Some(dst), Some(rel)) = (row.first(), row.get(1)) {
                        related.push(format!("{entity} {} {}", str_val(rel), str_val(dst)));
                    }
                }
            }
            Err(e) => tracing::warn!("graph outgoing query failed for {entity}: {e}"),
        }

        match incoming {
            Ok(rows) => {
                for row in rows.rows {
                    if let (Some(src), Some(rel)) = (row.first(), row.get(1)) {
                        related.push(format!("{} {} {entity}", str_val(src), str_val(rel)));
                    }
                }
            }
            Err(e) => tracing::warn!("graph incoming query failed for {entity}: {e}"),
        }
    }

    related.dedup();
    Ok(related)
}

fn query_outgoing(db: &DbInstance, entity: &str) -> Result<NamedRows, String> {
    let params = BTreeMap::from([("entity".to_string(), DataValue::Str(entity.into()))]);
    db.run_script(
        "?[dst, relation] := *relationships{src: $entity, relation, dst}",
        params,
        ScriptMutability::Immutable,
    )
    .map_err(|e| format!("{e}"))
}

fn query_incoming(db: &DbInstance, entity: &str) -> Result<NamedRows, String> {
    let params = BTreeMap::from([("entity".to_string(), DataValue::Str(entity.into()))]);
    db.run_script(
        "?[src, relation] := *relationships{src, relation, dst: $entity}",
        params,
        ScriptMutability::Immutable,
    )
    .map_err(|e| format!("{e}"))
}

fn str_val(v: &DataValue) -> &str {
    v.get_str().unwrap_or("")
}

/// Find entities related to a query via word overlap on entities and relationships.
pub async fn find_concepts(query: &str) -> Vec<String> {
    let db = match get_graph() {
        Ok(db) => db,
        Err(_) => return vec![],
    };

    let keywords = extract_keywords(query);
    if keywords.is_empty() {
        return vec![];
    }

    let mut matched = std::collections::HashSet::new();
    match_entity_names(db, &keywords, &mut matched);
    match_relationships(db, &keywords, &mut matched);
    matched.into_iter().collect()
}

fn match_entity_names(
    db: &DbInstance,
    keywords: &[String],
    matched: &mut std::collections::HashSet<String>,
) {
    let Ok(r) = db.run_script(
        "?[name] := *entities{name, entity_type, source_text}",
        BTreeMap::new(),
        ScriptMutability::Immutable,
    ) else { return };

    for row in &r.rows {
        if let Some(name) = row.first().and_then(|v| v.get_str()) {
            if entity_matches_keywords(name, keywords) {
                matched.insert(name.to_string());
            }
        }
    }
}

fn match_relationships(
    db: &DbInstance,
    keywords: &[String],
    matched: &mut std::collections::HashSet<String>,
) {
    let Ok(r) = db.run_script(
        "?[src, rel, dst] := *relationships{src, relation: rel, dst}",
        BTreeMap::new(),
        ScriptMutability::Immutable,
    ) else { return };

    for row in &r.rows {
        let src = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let rel = row.get(1).and_then(|v| v.get_str()).unwrap_or("");
        let dst = row.get(2).and_then(|v| v.get_str()).unwrap_or("");
        if triplet_matches_keywords(src, rel, dst, keywords) {
            matched.insert(src.to_string());
            matched.insert(dst.to_string());
        }
    }
}

fn triplet_matches_keywords(src: &str, rel: &str, dst: &str, keywords: &[String]) -> bool {
    entity_matches_keywords(src, keywords)
        || entity_matches_keywords(dst, keywords)
        || words_overlap(rel, keywords)
}

fn extract_keywords(query: &str) -> Vec<String> {
    let stops = stop_words();
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|w| w.len() >= 2 && !stops.contains(w))
        .map(|w| w.to_string())
        .collect()
}

fn stop_words() -> std::collections::HashSet<&'static str> {
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "about", "between",
        "through", "after", "before", "and", "but", "or", "not", "no", "if",
        "then", "so", "than", "that", "this", "these", "those", "it", "its",
        "my", "your", "his", "her", "our", "their", "what", "which", "who",
        "when", "where", "how", "why", "all", "each", "every", "any", "some",
        "i", "me", "we", "you", "he", "she", "they", "them", "let", "us",
        "yes", "no", "just", "also", "very", "much", "more", "most", "well",
        "now", "here", "there", "still", "already", "yet", "too", "only",
        "work", "use", "make", "get", "set", "run", "fix", "add", "try",
        "want", "need", "know", "think", "look", "find", "give", "tell",
        "first", "new", "old", "last", "next", "same", "other", "like",
    ].iter().copied().collect()
}

fn entity_matches_keywords(entity: &str, keywords: &[String]) -> bool {
    let entity_lower = entity.to_lowercase();
    let entity_words: Vec<&str> = entity_lower
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|w| w.len() >= 2)
        .collect();
    keywords.iter().any(|kw| entity_words.iter().any(|ew| ew == kw || ew.contains(kw.as_str())))
}

fn words_overlap(text: &str, keywords: &[String]) -> bool {
    let lower = text.to_lowercase();
    keywords.iter().any(|kw| lower.contains(kw.as_str()))
}
