use super::*;

#[test]
fn valid_entity_rejects_permission_scopes_and_artifacts() {
    assert!(!is_valid_entity("Mail.ReadWrite"));
    assert!(!is_valid_entity("MailboxSettings.Read"));
    assert!(!is_valid_entity("offline_access"));
    assert!(!is_valid_entity("(( ))"));
    assert!(!is_valid_entity("=="));
    assert!(!is_valid_entity("0x1.pt"));
    assert!(!is_valid_entity("74:ac:b9:3d:53:a1"));
}

#[test]
fn parse_triplets_filters_documentation_and_scope_noise() {
    let input = r#"[
        ["cargo-apk", "documented_in", "package-reasons.md"],
        ["Microsoft Graph", "provides", "Mail.ReadWrite"],
        ["Qdrant", "written_in", "Rust"]
    ]"#;
    let result = parse_triplets(input).unwrap();
    assert_eq!(
        result,
        vec![(
            "Qdrant".to_string(),
            "written_in".to_string(),
            "Rust".to_string()
        )]
    );
}

#[test]
fn parse_triplets_filters_meta_relationship_noise() {
    let input = r#"[
        ["ChatGPT", "feature", "memory"],
        ["cagent", "includes_tool", "memory"],
        ["database monitoring", "demonstrates", "memory improvements"],
        ["claude-memory", "uses", "Qdrant"]
    ]"#;
    let result = parse_triplets(input).unwrap();
    assert_eq!(
        result,
        vec![(
            "claude-memory".to_string(),
            "uses".to_string(),
            "Qdrant".to_string()
        )]
    );
}
