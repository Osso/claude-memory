use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

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
