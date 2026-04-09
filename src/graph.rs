//! CozoDB embedded graph database for entity-relationship storage.

mod clean;
mod sanitize;

use anyhow::{Context, Result};
use cozo::{DataValue, DbInstance, NamedRows, ScriptMutability};
use std::collections::BTreeMap;
use std::sync::OnceLock;

static GRAPH: OnceLock<DbInstance> = OnceLock::new();

const EXTRACT_SYSTEM: &str = "Extract entity-relationship triplets from the text. \
Return a JSON array of [subject, relation, object] arrays. \
Focus on: project names, technologies, people, tools, architectural decisions, preferences. \
Entity names MUST be proper nouns or short noun phrases (1-3 words). \
NEVER use as entities: file paths, CLI flags, code snippets, numbers, measurements, \
quantities, percentages, filenames, version numbers, coordinates, regex, variable names. \
Good: \"authd\", \"Rust\", \"Qdrant\", \"AMD RDNA 4\", \"GlobalComix\", \"YNAB\" \
Bad: \"/etc/authd/\", \"--json\", \"120 FPS\", \"575 Watts\", \"04-external-services.md\", \
\"23000+ organizations\", \"0.8.x\", \"1Gi memory limit\", \"49 rotation keyframes\" \
Example: [[\"authd\", \"written_in\", \"Rust\"], [\"authd\", \"replaces\", \"polkit\"]] \
If no clear relationships exist, return: []";

pub use clean::GraphCleanStats;

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

pub fn clean_graph(max_passes: usize, dry_run: bool) -> Result<GraphCleanStats> {
    clean::clean_graph(max_passes, dry_run)
}

pub fn clear_graph() -> Result<()> {
    clean::clear_graph()
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
    let key = match anthropic_api_key() {
        Some(key) => key,
        _ => return Ok(vec![]),
    };

    let response = anthropic_client()?
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &key)
        .header("anthropic-version", "2023-06-01")
        .json(&anthropic_request_body(text))
        .send()
        .await
        .context("failed to call Anthropic API")?;
    let json = parse_anthropic_response(response).await?;
    let content_text = response_content_text(&json);
    parse_triplets(content_text)
}

fn parse_triplets(text: &str) -> Result<Vec<(String, String, String)>> {
    Ok(parse_triplet_array(text)
        .iter()
        .filter_map(parse_triplet_fields)
        .filter_map(|(subject, relation, object)| {
            sanitize::sanitize_triplet(subject, relation, object)
        })
        .collect())
}

fn store_triplet(
    db: &DbInstance,
    subject: &str,
    relation: &str,
    object: &str,
    source_text: &str,
) -> Result<()> {
    let Some((subject, relation, object)) = sanitize::sanitize_triplet(subject, relation, object)
    else {
        return Ok(());
    };
    store_entity(db, &subject, source_text)?;
    store_entity(db, &object, source_text)?;

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

/// Reject entities that are code artifacts, not real concepts.
#[cfg(test)]
fn is_valid_entity(name: &str) -> bool {
    sanitize::is_valid_entity_name(name)
}

/// Known-good numeric prefixes: 2D, 3D, 2FA, 4K, etc.
#[cfg(test)]
fn looks_like_number_or_hash(name: &str) -> bool {
    sanitize::looks_like_number_or_hash(name)
}

/// Maximum triplets returned from graph enrichment.
const MAX_RELATED_TRIPLETS: usize = 50;

pub fn query_related(entities: &[String]) -> Result<Vec<String>> {
    let db = get_graph()?;
    let mut related = Vec::new();
    for entity in entities {
        if related.len() >= MAX_RELATED_TRIPLETS {
            break;
        }
        related.extend(collect_related_rows(db, entity, RelatedDirection::Outgoing));
        if related.len() >= MAX_RELATED_TRIPLETS {
            break;
        }
        related.extend(collect_related_rows(db, entity, RelatedDirection::Incoming));
    }
    related.dedup();
    related.truncate(MAX_RELATED_TRIPLETS);
    Ok(related)
}

fn query_related_rows(
    db: &DbInstance,
    entity: &str,
    direction: RelatedDirection,
) -> Result<NamedRows, String> {
    let params = BTreeMap::from([("entity".to_string(), DataValue::Str(entity.into()))]);
    let script = match direction {
        RelatedDirection::Outgoing => {
            "?[dst, relation] := *relationships{src: $entity, relation, dst}"
        }
        RelatedDirection::Incoming => {
            "?[src, relation] := *relationships{src, relation, dst: $entity}"
        }
    };
    db.run_script(script, params, ScriptMutability::Immutable)
        .map_err(|e| format!("{e}"))
}

fn str_val(v: &DataValue) -> &str {
    v.get_str().unwrap_or("")
}

fn parse_triplet_array(text: &str) -> Vec<serde_json::Value> {
    let json_str = extract_json_array(text);
    serde_json::from_str(json_str).unwrap_or_default()
}

fn anthropic_api_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|key| !key.is_empty())
}

fn anthropic_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(Into::into)
}

fn anthropic_request_body(text: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "system": EXTRACT_SYSTEM,
        "messages": [{"role": "user", "content": text}]
    })
}

async fn parse_anthropic_response(response: reqwest::Response) -> Result<serde_json::Value> {
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Anthropic API error {}: {}", status, body);
    }

    response
        .json()
        .await
        .context("failed to parse API response")
}

fn response_content_text(json: &serde_json::Value) -> &str {
    json["content"]
        .as_array()
        .and_then(|array| array.first())
        .and_then(|block| block["text"].as_str())
        .unwrap_or("[]")
}

fn extract_json_array(text: &str) -> &str {
    let start = text.find('[').unwrap_or(0);
    let end = text.rfind(']').map(|index| index + 1).unwrap_or(text.len());
    &text[start..end]
}

fn parse_triplet_fields(item: &serde_json::Value) -> Option<(&str, &str, &str)> {
    let values = item.as_array()?;
    let [subject, relation, object] = values.as_slice() else {
        return None;
    };

    let subject = subject.as_str()?;
    let relation = relation.as_str()?;
    let object = object.as_str()?;
    if subject.is_empty() || relation.is_empty() || object.is_empty() {
        return None;
    }

    Some((subject, relation, object))
}

fn collect_related_rows(db: &DbInstance, entity: &str, direction: RelatedDirection) -> Vec<String> {
    let Ok(rows) = query_related_rows(db, entity, direction) else {
        return vec![];
    };

    rows.rows
        .iter()
        .filter_map(|row| format_related_row(entity, row, direction))
        .collect()
}

fn format_related_row(
    entity: &str,
    row: &[DataValue],
    direction: RelatedDirection,
) -> Option<String> {
    let first = row.first()?;
    let relation = row.get(1)?;
    let formatted = match direction {
        RelatedDirection::Outgoing => format!("{entity} {} {}", str_val(relation), str_val(first)),
        RelatedDirection::Incoming => format!("{} {} {entity}", str_val(first), str_val(relation)),
    };
    Some(formatted)
}

#[derive(Clone, Copy)]
enum RelatedDirection {
    Outgoing,
    Incoming,
}

/// Maximum entities returned from concept matching.
const MAX_MATCHED_ENTITIES: usize = 15;

/// Find entities related to a query via word overlap on entities and relationships.
/// Returns up to MAX_MATCHED_ENTITIES entities, ranked by keyword match count.
pub async fn find_concepts(query: &str) -> Vec<String> {
    let db = match get_graph() {
        Ok(db) => db,
        Err(_) => return vec![],
    };

    let keywords = extract_keywords(query);
    if keywords.is_empty() {
        return vec![];
    }

    let mut scored: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    score_entity_names(db, &keywords, &mut scored);
    score_relationships(db, &keywords, &mut scored);

    let mut ranked: Vec<(String, usize)> = scored.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(MAX_MATCHED_ENTITIES);
    ranked.into_iter().map(|(name, _)| name).collect()
}

fn score_entity_names(
    db: &DbInstance,
    keywords: &[String],
    scored: &mut std::collections::HashMap<String, usize>,
) {
    let Ok(r) = db.run_script(
        "?[name] := *entities{name, entity_type, source_text}",
        BTreeMap::new(),
        ScriptMutability::Immutable,
    ) else {
        return;
    };

    for row in &r.rows {
        if let Some(name) = row.first().and_then(|v| v.get_str()) {
            let score = entity_keyword_score(name, keywords);
            if score > 0 {
                *scored.entry(name.to_string()).or_default() += score;
            }
        }
    }
}

fn score_relationships(
    db: &DbInstance,
    keywords: &[String],
    scored: &mut std::collections::HashMap<String, usize>,
) {
    let Ok(r) = db.run_script(
        "?[src, dst] := *relationships{src, dst}",
        BTreeMap::new(),
        ScriptMutability::Immutable,
    ) else {
        return;
    };

    for row in &r.rows {
        let src = row.first().and_then(|v| v.get_str()).unwrap_or("");
        let dst = row.get(1).and_then(|v| v.get_str()).unwrap_or("");

        let src_score = entity_keyword_score(src, keywords);
        let dst_score = entity_keyword_score(dst, keywords);
        // Only add the side(s) that actually matched by entity name
        if src_score > 0 {
            *scored.entry(src.to_string()).or_default() += src_score;
        }
        if dst_score > 0 {
            *scored.entry(dst.to_string()).or_default() += dst_score;
        }
        // Relation-only matches (neither src nor dst matched) are skipped —
        // generic relation words like "protocol", "uses" cause too much fan-out.
    }
}

#[cfg(test)]
fn triplet_matches_keywords(src: &str, rel: &str, dst: &str, keywords: &[String]) -> bool {
    entity_keyword_score(src, keywords) > 0
        || entity_keyword_score(dst, keywords) > 0
        || words_overlap_exact(rel, keywords)
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
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "about",
        "between", "through", "after", "before", "and", "but", "or", "not", "no", "if", "then",
        "so", "than", "that", "this", "these", "those", "it", "its", "my", "your", "his", "her",
        "our", "their", "what", "which", "who", "when", "where", "how", "why", "all", "each",
        "every", "any", "some", "i", "me", "we", "you", "he", "she", "they", "them", "let", "us",
        "yes", "no", "just", "also", "very", "much", "more", "most", "well", "now", "here",
        "there", "still", "already", "yet", "too", "only", "work", "use", "make", "get", "set",
        "run", "fix", "add", "try", "want", "need", "know", "think", "look", "find", "give",
        "tell", "first", "new", "old", "last", "next", "same", "other", "like",
    ]
    .iter()
    .copied()
    .collect()
}

/// Count how many keywords match entity words (exact word match only).
fn entity_keyword_score(entity: &str, keywords: &[String]) -> usize {
    let entity_lower = entity.to_lowercase();
    let entity_words: Vec<&str> = entity_lower
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|w| w.len() >= 2)
        .collect();
    keywords
        .iter()
        .filter(|kw| entity_words.iter().any(|ew| ew == kw))
        .count()
}

#[cfg(test)]
fn entity_matches_keywords(entity: &str, keywords: &[String]) -> bool {
    entity_keyword_score(entity, keywords) > 0
}

#[cfg(test)]
fn words_overlap_exact(text: &str, keywords: &[String]) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 2)
        .collect();
    keywords.iter().any(|kw| words.iter().any(|w| w == kw))
}

#[cfg(test)]
fn words_overlap(text: &str, keywords: &[String]) -> bool {
    words_overlap_exact(text, keywords)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod clean_tests;
