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
    assert_eq!(
        result[0],
        (
            "authd".to_string(),
            "written_in".to_string(),
            "Rust".to_string()
        )
    );
    assert_eq!(
        result[1],
        (
            "authd".to_string(),
            "replaces".to_string(),
            "polkit".to_string()
        )
    );
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
    let input = r#"[["only", "two"], ["authd", "uses", "Rust"]]"#;
    let result = parse_triplets(input).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0],
        ("authd".to_string(), "uses".to_string(), "Rust".to_string())
    );
}

#[test]
fn parse_triplets_skips_empty_string_fields() {
    let input = r#"[["", "rel", "obj"], ["Qdrant", "uses", "Rust"]]"#;
    let result = parse_triplets(input).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0],
        ("Qdrant".to_string(), "uses".to_string(), "Rust".to_string())
    );
}

#[test]
fn parse_triplets_strips_prose_wrapper() {
    let input = r#"Here are the triplets: [["Qdrant", "uses", "Rust"]] done."#;
    let result = parse_triplets(input).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0],
        ("Qdrant".to_string(), "uses".to_string(), "Rust".to_string())
    );
}

// --- triplet_matches_keywords ---

#[test]
fn triplet_matches_keywords_on_src() {
    let kws = vec!["rust".to_string()];
    assert!(triplet_matches_keywords(
        "Rust",
        "written_in",
        "authd",
        &kws
    ));
}

#[test]
fn triplet_matches_keywords_on_dst() {
    let kws = vec!["polkit".to_string()];
    assert!(triplet_matches_keywords(
        "authd", "replaces", "polkit", &kws
    ));
}

#[test]
fn triplet_matches_keywords_on_relation() {
    let kws = vec!["replaces".to_string()];
    assert!(triplet_matches_keywords(
        "authd", "replaces", "polkit", &kws
    ));
}

#[test]
fn triplet_matches_keywords_false_on_no_match() {
    let kws = vec!["postgres".to_string()];
    assert!(!triplet_matches_keywords(
        "authd", "replaces", "polkit", &kws
    ));
}
