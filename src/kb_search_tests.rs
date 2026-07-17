use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn text_index_files_round_trip_generated_sections() {
    let root = unique_temp_dir("kb-text-index-contract");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("state")).unwrap();
    std::fs::write(
        kb_dir.join("state/router.md"),
        "# Router\nLocal network overview.\n\n## Firewall\nBlock unsolicited WAN traffic.\n",
    )
    .unwrap();

    let summary = build_text_index(&kb_dir, &index_dir).unwrap();
    let nodes = load_text_nodes(&index_dir).unwrap();
    let manifest = load_text_manifest(&index_dir).unwrap();

    assert_eq!(summary.files, 1);
    assert_eq!(summary.nodes, 2);
    assert_eq!(summary.index_path, index_dir.join("nodes.tsv"));
    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].path, "state/router.md");
    assert_eq!(nodes[0].heading_path, "Router");
    assert_eq!(nodes[1].heading_path, "Router > Firewall");
    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest[0].path, "state/router.md");
    assert!(manifest[0].mtime_ns > 0);
    assert!(manifest[0].size > 0);

    let nodes_text = std::fs::read_to_string(index_dir.join("nodes.tsv")).unwrap();
    let manifest_text = std::fs::read_to_string(index_dir.join("manifest.tsv")).unwrap();
    assert!(!nodes_text.starts_with('{'));
    assert!(!manifest_text.starts_with('{'));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_index_stores_full_normalized_section_body() {
    let root = unique_temp_dir("kb-text-index-normalized-body");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("notes.md"),
        "# Notes\n  First line with spacing.\nSecond\tline.\n\nThird line remains searchable.\n",
    )
    .unwrap();

    build_text_index(&kb_dir, &index_dir).unwrap();
    let nodes = load_text_nodes(&index_dir).unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].normalized_body,
        "First line with spacing. Second line. Third line remains searchable."
    );

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn manifest_detects_size_or_mtime_changes() {
    let root = unique_temp_dir("kb-text-index-stale");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    let path = kb_dir.join("notes.md");
    std::fs::write(&path, "# Notes\nOriginal.\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    assert!(validate_text_manifest(&kb_dir, &index_dir).is_ok());

    std::fs::write(&path, "# Notes\nChanged content with a different size.\n").unwrap();
    let error = validate_text_manifest(&kb_dir, &index_dir).unwrap_err();

    assert!(error.to_string().contains("stale KB text index"));
    assert!(error.to_string().contains("notes.md"));

    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn headingless_hash_prefix_remains_in_normalized_body() {
    let root = unique_temp_dir("kb-text-index-leading-hashes");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("notes.md"),
        "####### literal text\nimportant body\n",
    )
    .unwrap();

    build_text_index(&kb_dir, &index_dir).unwrap();
    let nodes = load_text_nodes(&index_dir).unwrap();

    assert_eq!(
        nodes[0].normalized_body,
        "####### literal text important body"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn empty_markdown_heading_marker_remains_in_normalized_body() {
    let root = unique_temp_dir("kb-text-index-empty-heading");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("notes.md"), "# \nimportant body\n").unwrap();

    build_text_index(&kb_dir, &index_dir).unwrap();
    let nodes = load_text_nodes(&index_dir).unwrap();

    assert_eq!(nodes[0].normalized_body, "# important body");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_index_order_ranges_and_escaping_are_deterministic() {
    let root = unique_temp_dir("kb-text-index-deterministic");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("z.md"), "# Zed\nLast.\n").unwrap();
    std::fs::write(
        kb_dir.join("a\tb.md"),
        "# Alpha\tBeta\nFirst.\n\n## Child\nSecond.\n",
    )
    .unwrap();

    build_text_index(&kb_dir, &index_dir).unwrap();
    let nodes = load_text_nodes(&index_dir).unwrap();

    assert_eq!(nodes[0].path, "a\tb.md");
    assert_eq!(nodes[0].heading_path, "Alpha\tBeta");
    assert_eq!((nodes[0].line_start, nodes[0].line_end), (1, 3));
    assert_eq!((nodes[1].line_start, nodes[1].line_end), (4, 5));
    assert_eq!(nodes[2].path, "z.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_index_parser_rejects_invalid_ranges_with_row_context() {
    let root = unique_temp_dir("kb-text-index-invalid-range");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("nodes.tsv"), "notes.md\t10\t2\tNotes\tbody\n").unwrap();

    let error = load_text_nodes(&root).unwrap_err();

    let message = format!("{error:#}");
    assert!(message.contains("nodes.tsv row 1"));
    assert!(message.contains("line range"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn manifest_compares_mtime_and_size_independently() {
    let baseline = TextManifestEntry {
        path: "notes.md".to_string(),
        mtime_ns: 10,
        size: 20,
    };

    assert!(!manifest_entries_match(
        &baseline,
        &TextManifestEntry {
            mtime_ns: 11,
            ..baseline.clone()
        }
    ));
    assert!(!manifest_entries_match(
        &baseline,
        &TextManifestEntry {
            size: 21,
            ..baseline.clone()
        }
    ));
}

#[test]
fn integer_sqrt_handles_boundaries_without_rounding() {
    for (input, expected) in [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 2),
        (8, 2),
        (9, 3),
        (15, 3),
        (16, 4),
        (17, 4),
    ] {
        assert_eq!(integer_sqrt(input), expected, "input={input}");
    }

    let max_root = (1usize << (usize::BITS / 2)) - 1;
    assert_eq!(integer_sqrt(usize::MAX), max_root);
}

#[test]
fn text_search_weights_heading_over_path_over_body() {
    let root = unique_temp_dir("kb-text-search-field-weights");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("body.md"), "# Notes\nrouter\n").unwrap();
    std::fs::write(kb_dir.join("router-path.md"), "# Notes\nunrelated\n").unwrap();
    std::fs::write(kb_dir.join("heading.md"), "# Router\nunrelated\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router", 3).unwrap();
    let paths: Vec<&str> = results.iter().map(|result| result.path.as_str()).collect();

    assert_eq!(paths, ["heading.md", "router-path.md", "body.md"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_caps_term_frequency() {
    let root = unique_temp_dir("kb-text-search-frequency-cap");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("a-three.md"),
        format!("# Notes\nrouter router router {}\n", "noise ".repeat(17)),
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("z-twenty.md"),
        format!("# Notes\n{}\n", "router ".repeat(20)),
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router", 2).unwrap();

    assert_eq!(results[0].score, results[1].score);
    assert_eq!(results[0].path, "a-three.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_normalizes_for_section_length() {
    let root = unique_temp_dir("kb-text-search-length");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("focused.md"), "# Notes\nrouter configuration\n").unwrap();
    std::fs::write(
        kb_dir.join("long.md"),
        format!("# Notes\nrouter {}\n", "noise ".repeat(100)),
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router", 2).unwrap();

    assert_eq!(results[0].path, "focused.md");
    assert!(results[0].score > results[1].score);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn structural_fields_outrank_capped_body_frequency() {
    let root = unique_temp_dir("kb-text-search-absolute-field-priority");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("heading.md"), "# Router\nunrelated\n").unwrap();
    std::fs::write(kb_dir.join("router-path.md"), "# Notes\nunrelated\n").unwrap();
    std::fs::write(kb_dir.join("body.md"), "# Notes\nrouter router router\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router", 3).unwrap();
    let paths: Vec<&str> = results.iter().map(|result| result.path.as_str()).collect();

    assert_eq!(paths, ["heading.md", "router-path.md", "body.md"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_does_not_length_normalize_heading_or_path_weights() {
    let root = unique_temp_dir("kb-text-search-structural-length");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("heading.md"),
        format!("# Router\n{}\n", "noise ".repeat(100)),
    )
    .unwrap();
    std::fs::write(kb_dir.join("body.md"), "# Notes\nrouter\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router", 2).unwrap();

    assert_eq!(results[0].path, "heading.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_rewards_token_sequence_phrase_across_punctuation() {
    let root = unique_temp_dir("kb-text-search-token-phrase");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("a-false-substring.md"),
        "# Notes\nbiosolar batteryless solar planning battery\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("z-token-sequence.md"),
        "# Notes\nsolar-battery planning notes\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "solar battery", 2).unwrap();

    assert_eq!(results[0].path, "z-token-sequence.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_preserves_unicode_alphanumeric_tokens() {
    let root = unique_temp_dir("kb-text-search-unicode");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("resume.md"), "# Notes\nrésumé guidance\n").unwrap();
    std::fs::write(kb_dir.join("network.md"), "# Notes\n家庭网络 guidance\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    assert_eq!(
        search_text_index(&kb_dir, &index_dir, "résumé", 1).unwrap()[0].path,
        "resume.md"
    );
    assert_eq!(
        search_text_index(&kb_dir, &index_dir, "家庭网络", 1).unwrap()[0].path,
        "network.md"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn repeated_query_tokens_remain_required_for_phrase_bonus() {
    let root = unique_temp_dir("kb-text-search-repeated-phrase");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("a-reordered.md"), "# Notes\nvery good very\n").unwrap();
    std::fs::write(kb_dir.join("z-exact.md"), "# Notes\nvery very good\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "very very good", 2).unwrap();

    assert_eq!(results[0].path, "z-exact.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn repeated_query_terms_do_not_amplify_score() {
    let root = unique_temp_dir("kb-text-search-query-dedup");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("notes.md"), "# Notes\nrouter configuration\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let single = search_text_index(&kb_dir, &index_dir, "router", 1).unwrap();
    let repeated = search_text_index(&kb_dir, &index_dir, "router router router", 1).unwrap();

    assert!(repeated[0].score <= single[0].score);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_rewards_exact_phrase() {
    let root = unique_temp_dir("kb-text-search-phrase");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("a-separated.md"),
        "# Notes\nsolar production and home battery storage\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("z-exact.md"),
        "# Notes\nsolar battery planning for home storage\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "solar battery", 2).unwrap();

    assert_eq!(results[0].path, "z-exact.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn text_search_uses_deterministic_path_order_for_ties() {
    let root = unique_temp_dir("kb-text-search-ties");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(kb_dir.join("b.md"), "# Notes\nrouter\n").unwrap();
    std::fs::write(kb_dir.join("a.md"), "# Notes\nrouter\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router", 2).unwrap();
    let paths: Vec<&str> = results.iter().map(|result| result.path.as_str()).collect();

    assert_eq!(paths, ["a.md", "b.md"]);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn stale_text_search_rejects_without_automatic_rebuild() {
    let root = unique_temp_dir("kb-text-search-stale");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    let path = kb_dir.join("notes.md");
    std::fs::write(&path, "# Notes\noriginal router note\n").unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();
    let original_index = std::fs::read(index_dir.join("nodes.tsv")).unwrap();
    std::fs::write(&path, "# Notes\nchanged router note with more text\n").unwrap();

    let error = search_text_index(&kb_dir, &index_dir, "router", 3).unwrap_err();

    assert!(format!("{error:#}").contains("stale KB text index"));
    assert_eq!(
        std::fs::read(index_dir.join("nodes.tsv")).unwrap(),
        original_index
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn build_and_search_persisted_kb_index() {
    let root = unique_temp_dir("kb-page-index-search");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("memory")).unwrap();
    std::fs::write(
        kb_dir.join("memory/corrections.md"),
        "# Corrections\n\n## Process\nLoad frontend design skill immediately.\n",
    )
    .unwrap();

    let summary = build_index(&kb_dir, &index_dir).unwrap();
    let results = search_index(&index_dir, "frontend design skill load immediately", 3).unwrap();

    assert_eq!(summary.files, 1);
    assert_eq!(summary.nodes, 2);
    assert_eq!(results[0].path, "memory/corrections.md");
    assert_eq!(results[0].heading, "Corrections > Process");
}

#[test]
fn search_or_build_refreshes_stale_index() {
    let root = unique_temp_dir("kb-page-index-refresh");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    let path = kb_dir.join("notes.md");
    std::fs::write(&path, "# Notes\nOld content.\n").unwrap();
    build_index(&kb_dir, &index_dir).unwrap();

    std::fs::write(&path, "# Notes\nUse uv instead of pip.\n").unwrap();
    let results = search_or_build(&kb_dir, &index_dir, "uv instead pip", 3).unwrap();

    assert_eq!(results[0].path, "notes.md");
    assert!(results[0].text.contains("uv instead of pip"));
}

#[test]
fn build_doc_uses_nested_page_index_document_model() {
    let markdown = "# Router\nLocal network router note.\n\n## DHCP\nLease reservations.\n\n### Static leases\nPin important devices.\n\n## Firewall\nWAN block rules.\n";
    let doc = build_doc_from_text("state/router.md", markdown);

    assert_eq!(doc.doc_id, "state/router.md");
    assert_eq!(doc.doc_name, "router");
    assert_eq!(doc.doc_description.as_deref(), Some("Router"));
    assert_eq!(doc.source_path, "state/router.md");
    assert_eq!(doc.line_count, 11);
    assert_eq!(doc.nodes.len(), 1);

    let root = &doc.nodes[0];
    assert_eq!(root.node_id, "000001");
    assert_eq!(root.title, "Router");
    assert_eq!(root.heading_path, "Router");
    assert_eq!(root.source_line, 1);
    assert_eq!(root.nodes.len(), 2);
    assert!(root.text.contains("Local network router note."));

    let dhcp = &root.nodes[0];
    assert_eq!(dhcp.node_id, "000002");
    assert_eq!(dhcp.title, "DHCP");
    assert_eq!(dhcp.heading_path, "Router > DHCP");
    assert_eq!(dhcp.nodes[0].node_id, "000003");
    assert_eq!(dhcp.nodes[0].heading_path, "Router > DHCP > Static leases");
    assert!(dhcp.nodes[0].text.contains("Pin important devices."));
}

#[test]
fn structure_view_omits_internal_node_text() {
    let markdown = "# Router\nSecret body text.\n\n## DHCP\nLease reservations.\n";
    let doc = build_doc_from_text("state/router.md", markdown);

    let structure = doc.structure_without_text();
    let json = serde_json::to_string(&structure).unwrap();

    assert!(json.contains("Router"));
    assert!(json.contains("000002"));
    assert!(!json.contains("Secret body text"));
    assert!(!json.contains("Lease reservations"));
    assert!(!json.contains("token_counts"));
}

#[test]
fn document_metadata_and_structure_resolve_doc_id_or_path_without_text() {
    let root = unique_temp_dir("kb-page-index-parity-doc");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    let guide_dir = kb_dir.join("guides");
    std::fs::create_dir_all(&guide_dir).unwrap();
    std::fs::write(
        guide_dir.join("router.md"),
        "# Router\nSecret body text\n## DHCP\nLease reservations\n",
    )
    .unwrap();
    build_index(&kb_dir, &index_dir).unwrap();

    let metadata = document_metadata(&index_dir, "guides/router.md").unwrap();
    assert_eq!(metadata.doc_id, "guides/router.md");
    assert_eq!(metadata.doc_name, "router");
    assert_eq!(metadata.source_path, "guides/router.md");
    assert_eq!(metadata.line_count, 4);

    let structure = document_structure(&index_dir, guide_dir.join("router.md")).unwrap();
    assert_eq!(structure.nodes[0].node_id, "000001");
    assert_eq!(structure.nodes[0].nodes[0].node_id, "000002");
    let json = serde_json::to_string(&structure).unwrap();
    assert!(!json.contains("Secret body text"));
    assert!(!json.contains("Lease reservations"));
}

#[test]
fn content_fetch_returns_exact_node_or_line_range_text() {
    let root = unique_temp_dir("kb-page-index-parity-content");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("router.md"),
        "# Router\nIntro line\n## DHCP\nLease reservations\n- static hosts\n",
    )
    .unwrap();
    build_index(&kb_dir, &index_dir).unwrap();

    let node_content = document_content(&index_dir, "router.md", "000002").unwrap();
    assert_eq!(node_content.doc_id, "router.md");
    assert_eq!(node_content.locator, "000002");
    assert_eq!(
        node_content.text,
        "## DHCP\nLease reservations\n- static hosts\n"
    );

    let line_content = document_content(&index_dir, "router.md", "2-4").unwrap();
    assert_eq!(line_content.locator, "2-4");
    assert_eq!(
        line_content.text,
        "Intro line\n## DHCP\nLease reservations\n"
    );
}

#[test]
fn fixture_markdown_proves_nested_structure_content_and_query() {
    let root = unique_temp_dir("kb-page-index-fixture-parity");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("guides")).unwrap();
    std::fs::write(
        kb_dir.join("guides/router.md"),
        "# Router\nRoot overview.\n\n## DHCP\nLease reservations.\n\n### Static leases\nPin important devices.\n",
    )
    .unwrap();

    build_index(&kb_dir, &index_dir).unwrap();

    let structure = document_structure(&index_dir, "guides/router.md").unwrap();
    assert_eq!(structure.nodes[0].heading_path, "Router");
    assert_eq!(
        structure.nodes[0].nodes[0].nodes[0].heading_path,
        "Router > DHCP > Static leases"
    );
    let structure_json = serde_json::to_string(&structure).unwrap();
    assert!(!structure_json.contains("Pin important devices."));

    let content = document_content(&index_dir, "guides/router.md", "000003").unwrap();
    assert_eq!(content.text, "### Static leases\nPin important devices.\n");

    let results = query_index(&index_dir, "static leases important devices", 3).unwrap();
    assert_eq!(results[0].doc_id, "guides/router.md");
    assert_eq!(results[0].node_id, "000003");
}

#[test]
fn query_returns_traceable_node_hits_without_snippets() {
    let root = unique_temp_dir("kb-page-index-parity-query");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("router.md"),
        "# Router\nIntro line\n## DHCP\nLease reservations\n",
    )
    .unwrap();
    build_index(&kb_dir, &index_dir).unwrap();

    let hits = query_index(&index_dir, "dhcp lease reservations", 5).unwrap();
    assert_eq!(hits[0].doc_id, "router.md");
    assert_eq!(hits[0].node_id, "000002");
    assert_eq!(hits[0].title, "DHCP");
    assert!(hits[0].reason.contains("lease"));
    assert!(
        hits[0]
            .next_content_command
            .contains("kb-page-index content router.md 000002")
    );
}

#[test]
fn search_or_build_context_fetches_exact_node_content_for_enrich() {
    let root = unique_temp_dir("kb-page-index-enrich-content");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("agent.md"),
        "# Agent Rules\nIntro text.\n\n## Frontend\nLoad the frontend design skill immediately before UI work.\n",
    )
    .unwrap();

    let results =
        search_or_build_context(&kb_dir, &index_dir, "frontend design skill immediately", 3)
            .unwrap();

    assert_eq!(results[0].path, "agent.md");
    assert_eq!(results[0].node_id, "000002");
    assert_eq!(
        results[0].text,
        "## Frontend\nLoad the frontend design skill immediately before UI work.\n"
    );
}

#[test]
fn search_or_build_refreshes_added_and_deleted_markdown_files() {
    let root = unique_temp_dir("kb-page-index-add-delete-refresh");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    let old_path = kb_dir.join("old.md");
    let new_path = kb_dir.join("new.md");
    std::fs::write(&old_path, "# Old\nRemove me after first build.\n").unwrap();
    build_index(&kb_dir, &index_dir).unwrap();

    std::fs::remove_file(&old_path).unwrap();
    std::fs::write(&new_path, "# New\nFresh page index content.\n").unwrap();

    let fresh_results = search_or_build(&kb_dir, &index_dir, "fresh page index content", 3)
        .expect("new file should trigger refresh");
    let old_results = search_index(&index_dir, "remove me after first build", 3)
        .expect("refreshed index should remain readable");

    assert_eq!(fresh_results[0].path, "new.md");
    assert!(old_results.is_empty());
}

#[test]
fn long_queries_require_three_distinct_terms() {
    let markdown = "# Notes\n\nDesign links and profession skill points.\n";
    let doc = build_doc_from_text("bookmarks.md", markdown);
    let query = "frontend design skill load immediately";

    let result = score_node(&doc, &doc.nodes[0], query, &tokenize(query));

    assert!(result.is_none());
}

#[test]
fn link_dumps_do_not_match_scattered_prompt_terms() {
    let markdown = format!(
        "# Firefox Tab Archive\n\n## Firefox Tabs\n{}\n",
        [
            "- [Deployment guide](https://example.test/deploy)",
            "- [Mini secrets manager](https://example.test/secrets)",
            "- [New shell tools](https://example.test/new)",
            "- [Would you use this tool?](https://example.test/you)",
            "- [Did the release ship?](https://example.test/did)",
            "- [Git performance improvements](https://example.test/git)",
            "- [Kubernetes news](https://example.test/kubernetes)",
            "- [Claude memory project](https://example.test/claude)",
            "- [Rust CLI parser](https://example.test/rust)",
            "- [Home automation notes](https://example.test/home)",
            "- [GlobalComix pull requests](https://example.test/prs)",
            "- [Task queue article](https://example.test/task)",
        ]
        .join("\n")
    );
    let doc = build_doc_from_text("state/firefox-tab-archive.md", &markdown);
    let query = "did you deploy the new secrets";

    let result = score_node(&doc, &doc.nodes[0].nodes[0], query, &tokenize(query));

    assert!(result.is_none());
}

#[test]
fn link_dumps_still_match_exact_link_phrases() {
    let markdown = format!(
        "# Firefox Tab Archive\n\n## Reddit\n{}\n",
        [
            "- [Deployment guide](https://example.test/deploy)",
            "- [VOA mini secrets manager](https://example.test/secrets)",
            "- [New shell tools](https://example.test/new)",
            "- [Would you use this tool?](https://example.test/you)",
            "- [Did the release ship?](https://example.test/did)",
            "- [Git performance improvements](https://example.test/git)",
            "- [Kubernetes news](https://example.test/kubernetes)",
            "- [Claude memory project](https://example.test/claude)",
            "- [Rust CLI parser](https://example.test/rust)",
            "- [Home automation notes](https://example.test/home)",
            "- [GlobalComix pull requests](https://example.test/prs)",
            "- [Task queue article](https://example.test/task)",
        ]
        .join("\n")
    );
    let doc = build_doc_from_text("state/firefox-tab-archive.md", &markdown);
    let query = "mini secrets manager";

    let result = score_node(&doc, &doc.nodes[0].nodes[0], query, &tokenize(query));

    assert!(result.is_some());
}

fn build_doc_from_text(path: &str, markdown: &str) -> KbIndexedDoc {
    let sections = split_markdown_sections(path, markdown);
    let doc_description = sections.first().map(|section| section.title.clone());
    KbIndexedDoc {
        doc_id: path.to_string(),
        doc_name: fallback_heading(path),
        doc_description,
        source_path: path.to_string(),
        line_count: markdown.lines().count(),
        text: markdown.to_string(),
        nodes: build_nested_nodes(&sections, None),
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}
