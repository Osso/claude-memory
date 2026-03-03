//! CozoDB embedded graph database for entity-relationship storage.

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
    if !is_valid_entity(subject) || !is_valid_entity(object) {
        return Ok(());
    }
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

/// Reject entities that are code artifacts, not real concepts.
fn is_valid_entity(name: &str) -> bool {
    if name.len() < 2 || name.len() > 60 {
        return false;
    }
    if name.starts_with('/') || name.starts_with('.') || name.contains("/.") {
        return false;
    }
    if name.contains('/') {
        return false;
    }
    if name.starts_with('-') || name.starts_with('+') || name.starts_with('@')
        || name.starts_with('$') || name.starts_with('#') || name.starts_with('&')
    {
        return false;
    }
    if name.split_whitespace().count() > 4 {
        return false;
    }
    if looks_like_number_or_hash(name) {
        return false;
    }
    if name.contains(".*") || name.contains("$(") || name.contains("=>") {
        return false;
    }
    if name.contains('(') || name.contains(')') || name.contains('<') || name.contains('>') {
        return false;
    }
    if name.contains('%') || name.contains('\'') || name.contains('"') {
        return false;
    }
    if has_file_extension(name) {
        return false;
    }
    if first_word_is_numeric(name) {
        return false;
    }
    true
}

/// Known-good numeric prefixes: 2D, 3D, 2FA, 4K, etc.
const GOOD_NUM_PREFIXES: &[&str] = &["2d", "3d", "2fa", "4k"];

/// Reject entities whose first word is a bare number or number+unit.
/// Allows well-known prefixes like 2D, 3D, 2FA, 4K.
fn first_word_is_numeric(name: &str) -> bool {
    let first = match name.split_whitespace().next() {
        Some(w) => w,
        None => return true,
    };
    if GOOD_NUM_PREFIXES.iter().any(|p| first.eq_ignore_ascii_case(p)) {
        return false;
    }
    if !first.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    // Pure digits
    if first.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    // Digits followed by short suffix (units): 1Gi, 10c, 0.8.x, 23000+
    let digit_prefix: String = first
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect();
    let rest = &first[digit_prefix.len()..];
    !digit_prefix.is_empty() && rest.len() <= 3
}

fn has_file_extension(name: &str) -> bool {
    const EXTS: &[&str] = &[
        ".md", ".rs", ".js", ".ts", ".tsx", ".php", ".py", ".toml",
        ".yaml", ".yml", ".json", ".html", ".css", ".go", ".sh",
    ];
    EXTS.iter().any(|ext| name.ends_with(ext))
}

fn looks_like_number_or_hash(name: &str) -> bool {
    let stripped = name.replace(['.', ',', ' ', '-', '_', '%', '+'], "");
    if stripped.is_empty() {
        return true;
    }
    stripped.chars().all(|c| c.is_ascii_digit() || c.is_ascii_hexdigit()
        || "KMGBikb".contains(c))
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

        if let Ok(rows) = query_outgoing(db, entity) {
            for row in rows.rows {
                if let (Some(dst), Some(rel)) = (row.first(), row.get(1)) {
                    related.push(format!("{entity} {} {}", str_val(rel), str_val(dst)));
                }
            }
        }

        if let Ok(rows) = query_incoming(db, entity) {
            for row in rows.rows {
                if let (Some(src), Some(rel)) = (row.first(), row.get(1)) {
                    related.push(format!("{} {} {entity}", str_val(src), str_val(rel)));
                }
            }
        }
    }

    related.dedup();
    related.truncate(MAX_RELATED_TRIPLETS);
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
    ) else { return };

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
    ) else { return };

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

/// Count how many keywords match entity words (exact word match only).
fn entity_keyword_score(entity: &str, keywords: &[String]) -> usize {
    let entity_lower = entity.to_lowercase();
    let entity_words: Vec<&str> = entity_lower
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        .filter(|w| w.len() >= 2)
        .collect();
    keywords.iter().filter(|kw| entity_words.iter().any(|ew| ew == kw)).count()
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
mod tests {
    use super::*;

    // --- is_valid_entity ---

    #[test]
    fn valid_entity_accepts_real_entities() {
        assert!(is_valid_entity("Qdrant"));
        assert!(is_valid_entity("Claude Code"));
        assert!(is_valid_entity("Rust"));
        assert!(is_valid_entity("JSONL parser"));
        assert!(is_valid_entity("GlobalComix"));
        assert!(is_valid_entity("authd"));
    }

    #[test]
    fn valid_entity_rejects_file_paths() {
        assert!(!is_valid_entity("/usr/bin/foo"));
        assert!(!is_valid_entity("src/main.rs"));
        assert!(!is_valid_entity(".hidden"));
    }

    #[test]
    fn valid_entity_rejects_cli_flags() {
        assert!(!is_valid_entity("--verbose"));
        assert!(!is_valid_entity("-n"));
    }

    #[test]
    fn valid_entity_rejects_numbers() {
        assert!(!is_valid_entity("12345"));
        assert!(!is_valid_entity("99999"));
    }

    #[test]
    fn valid_entity_rejects_hex_hashes() {
        assert!(!is_valid_entity("a1b2c3d4e5"));
        assert!(!is_valid_entity("abcdef1234"));
    }

    #[test]
    fn valid_entity_rejects_code_artifacts() {
        assert!(!is_valid_entity("fn()"));
        assert!(!is_valid_entity("Vec<String>"));
        assert!(!is_valid_entity("$(cmd)"));
    }

    #[test]
    fn valid_entity_rejects_empty_and_single_char() {
        assert!(!is_valid_entity(""));
        assert!(!is_valid_entity("x"));
    }

    #[test]
    fn valid_entity_rejects_numeric_first_word() {
        assert!(!is_valid_entity("120 FPS"));
        assert!(!is_valid_entity("575 Watts"));
        assert!(!is_valid_entity("1000000 lumens"));
        assert!(!is_valid_entity("49 rotation keyframes"));
        assert!(!is_valid_entity("10 Million downloads"));
        assert!(!is_valid_entity("1Gi memory limit"));
        assert!(!is_valid_entity("10c battery"));
        assert!(!is_valid_entity("0.8.x"));
    }

    #[test]
    fn valid_entity_allows_known_numeric_prefixes() {
        assert!(is_valid_entity("2D rendering"));
        assert!(is_valid_entity("3D Models"));
        assert!(is_valid_entity("2FA app"));
        assert!(is_valid_entity("4K display"));
    }

    #[test]
    fn valid_entity_rejects_file_extensions() {
        assert!(!is_valid_entity("04-external-services.md"));
        assert!(!is_valid_entity("main.rs"));
        assert!(!is_valid_entity("config.toml"));
        assert!(!is_valid_entity("index.ts"));
    }

    #[test]
    fn valid_entity_rejects_percent_and_quotes() {
        assert!(!is_valid_entity("11% to 1%"));
        assert!(!is_valid_entity("30% smaller"));
        assert!(!is_valid_entity("5'11\""));
    }

    #[test]
    fn valid_entity_rejects_ampersand_prefix() {
        assert!(!is_valid_entity("&& chaining"));
        assert!(!is_valid_entity("&mut reference"));
    }

    // --- looks_like_number_or_hash ---

    #[test]
    fn number_or_hash_true_for_pure_hex() {
        assert!(looks_like_number_or_hash("abcdef1234"));
        assert!(looks_like_number_or_hash("deadbeef"));
    }

    #[test]
    fn number_or_hash_true_for_pure_digits() {
        assert!(looks_like_number_or_hash("99999"));
        assert!(looks_like_number_or_hash("0"));
    }

    #[test]
    fn number_or_hash_true_for_version_strings() {
        // '.' is stripped, leaving only digits — detected as number
        assert!(looks_like_number_or_hash("1.2.3"));
        assert!(looks_like_number_or_hash("1.0.0"));
    }

    #[test]
    fn number_or_hash_false_for_normal_words() {
        assert!(!looks_like_number_or_hash("hello"));
        assert!(!looks_like_number_or_hash("Rust"));
    }

    #[test]
    fn number_or_hash_false_for_mixed_entity_names() {
        assert!(!looks_like_number_or_hash("Qdrant"));
        assert!(!looks_like_number_or_hash("Claude"));
    }

    // --- extract_keywords ---

    #[test]
    fn extract_keywords_lowercases_and_splits() {
        let kws = extract_keywords("Qdrant Vector Store");
        assert!(kws.contains(&"qdrant".to_string()));
        assert!(kws.contains(&"vector".to_string()));
        assert!(kws.contains(&"store".to_string()));
    }

    #[test]
    fn extract_keywords_removes_stop_words() {
        let kws = extract_keywords("how is the Rust compiler");
        assert!(!kws.contains(&"how".to_string()));
        assert!(!kws.contains(&"is".to_string()));
        assert!(!kws.contains(&"the".to_string()));
        assert!(kws.contains(&"rust".to_string()));
        assert!(kws.contains(&"compiler".to_string()));
    }

    #[test]
    fn extract_keywords_drops_single_char_tokens() {
        let kws = extract_keywords("a b c Rust");
        assert!(!kws.contains(&"a".to_string()));
        assert!(!kws.contains(&"b".to_string()));
        assert!(!kws.contains(&"c".to_string()));
        assert!(kws.contains(&"rust".to_string()));
    }

    // --- entity_matches_keywords / words_overlap ---

    #[test]
    fn entity_matches_keywords_true_on_word_match() {
        let kws = vec!["qdrant".to_string(), "vector".to_string()];
        assert!(entity_matches_keywords("Qdrant", &kws));
        assert!(entity_matches_keywords("Vector Store", &kws));
    }

    #[test]
    fn entity_matches_keywords_false_on_no_match() {
        let kws = vec!["postgres".to_string()];
        assert!(!entity_matches_keywords("Qdrant", &kws));
    }

    #[test]
    fn words_overlap_true_when_keyword_in_text() {
        let kws = vec!["written".to_string()];
        assert!(words_overlap("written_in", &kws));
        assert!(words_overlap("written in rust", &kws));
    }

    #[test]
    fn words_overlap_false_when_no_keyword_in_text() {
        let kws = vec!["replaces".to_string()];
        assert!(!words_overlap("written_in", &kws));
    }

    // --- parse_triplets ---

    #[test]
    fn parse_triplets_valid_json_array() {
        let input = r#"[["authd", "written_in", "Rust"], ["authd", "replaces", "polkit"]]"#;
        let result = parse_triplets(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("authd".to_string(), "written_in".to_string(), "Rust".to_string()));
        assert_eq!(result[1], ("authd".to_string(), "replaces".to_string(), "polkit".to_string()));
    }

    #[test]
    fn parse_triplets_empty_array() {
        let result = parse_triplets("[]").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn parse_triplets_malformed_json_returns_empty() {
        let result = parse_triplets("not json at all").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn parse_triplets_skips_incomplete_triplets() {
        // Inner arrays with fewer than 3 elements are skipped
        let input = r#"[["only", "two"], ["a", "b", "c"]]"#;
        let result = parse_triplets(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), "b".to_string(), "c".to_string()));
    }

    #[test]
    fn parse_triplets_skips_empty_string_fields() {
        let input = r#"[["", "rel", "obj"], ["sub", "rel", "obj"]]"#;
        let result = parse_triplets(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("sub".to_string(), "rel".to_string(), "obj".to_string()));
    }

    #[test]
    fn parse_triplets_strips_prose_wrapper() {
        let input = r#"Here are the triplets: [["Rust", "is", "fast"]] done."#;
        let result = parse_triplets(input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("Rust".to_string(), "is".to_string(), "fast".to_string()));
    }

    // --- triplet_matches_keywords ---

    #[test]
    fn triplet_matches_keywords_on_src() {
        let kws = vec!["rust".to_string()];
        assert!(triplet_matches_keywords("Rust", "written_in", "authd", &kws));
    }

    #[test]
    fn triplet_matches_keywords_on_dst() {
        let kws = vec!["polkit".to_string()];
        assert!(triplet_matches_keywords("authd", "replaces", "polkit", &kws));
    }

    #[test]
    fn triplet_matches_keywords_on_relation() {
        let kws = vec!["replaces".to_string()];
        assert!(triplet_matches_keywords("authd", "replaces", "polkit", &kws));
    }

    #[test]
    fn triplet_matches_keywords_false_on_no_match() {
        let kws = vec!["postgres".to_string()];
        assert!(!triplet_matches_keywords("authd", "replaces", "polkit", &kws));
    }
}
