use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

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
fn multi_term_body_match_outranks_single_structural_term() {
    let root = unique_temp_dir("kb-text-search-coverage");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("dev/ecdysis")).unwrap();
    std::fs::create_dir_all(kb_dir.join("memory")).unwrap();
    std::fs::write(
        kb_dir.join("dev/ecdysis/SKILL.md"),
        "# SKILL\nUnrelated reference.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("memory/corrections.md"),
        "# Corrections\n\n## Process\nFrontend design skill must load immediately.\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(
        &kb_dir,
        &index_dir,
        "frontend design skill load immediately",
        2,
    )
    .unwrap();

    assert_eq!(results[0].path, "memory/corrections.md");
    assert_eq!(results[1].path, "dev/ecdysis/SKILL.md");
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn adjacent_coverage_levels_outrank_maximum_structural_frequency() {
    let root = unique_temp_dir("kb-text-search-adjacent-coverage");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    let noisy_dir =
        kb_dir.join("alpha-beta-gamma-delta/alpha-beta-gamma-delta/alpha-beta-gamma-delta");
    std::fs::create_dir_all(&noisy_dir).unwrap();
    std::fs::write(
        noisy_dir.join("noise.md"),
        "# Alpha Beta Gamma Delta Alpha Beta Gamma Delta Alpha Beta Gamma Delta\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("complete.md"),
        "# Complete\nAlpha beta gamma delta epsilon.\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results =
        search_text_index(&kb_dir, &index_dir, "alpha beta gamma delta epsilon", 2).unwrap();

    assert_eq!(results[0].path, "complete.md");
    assert!(results[0].score > results[1].score);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn frontend_quality_gate_excludes_archive_noise_from_top_three() {
    let root = unique_temp_dir("kb-text-quality-frontend");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("memory/manual-memories")).unwrap();
    std::fs::create_dir_all(kb_dir.join("dev")).unwrap();
    std::fs::create_dir_all(kb_dir.join("state")).unwrap();
    std::fs::write(
        kb_dir.join("memory/corrections.md"),
        "# Corrections\n\n## Process\nFrontend design skill must load immediately.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("memory/manual-memories/__global__.md"),
        "# Durable memory\nRemember to load the frontend design skill immediately before interface work.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("dev/claude-skills-guide.md"),
        "# Guide\nThe frontend design skill should activate for interface tasks.\n",
    )
    .unwrap();
    std::fs::create_dir_all(kb_dir.join("personal/bookmarks/tech")).unwrap();
    std::fs::write(
        kb_dir.join("personal/bookmarks/tech/programming.md"),
        "# Programming\nDesign philosophy and skill references.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("state/firefox-tab-archive-2024-12.md"),
        "# Firefox Tab Archive\n\n## Firefox Tabs\nFrontend design skill load immediately bookmark.\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(
        &kb_dir,
        &index_dir,
        "frontend design skill load immediately",
        3,
    )
    .unwrap();
    let paths = results
        .iter()
        .map(|result| result.path.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        paths,
        [
            "memory/corrections.md",
            "memory/manual-memories/__global__.md",
        ]
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn bash_hook_quality_gate_returns_distinct_documents() {
    let root = unique_temp_dir("kb-text-quality-bash-hook");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("memory/notable-facts")).unwrap();
    std::fs::create_dir_all(kb_dir.join("memory/manual-memories")).unwrap();
    std::fs::write(
        kb_dir.join("memory/notable-facts/project-claude-bash-hook.md"),
        "# Claude Bash Hook\n\n## First\nClaude bash hook blocks unsafe Codex commands.\n\n## Second\nClaude bash hook rejects unsafe Codex execution.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("memory/manual-memories/project-claude-bash-hook.md"),
        "# Manual memory\nClaude bash hook handles unsafe Codex commands.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("memory/decision.md"),
        "# Decision\nClaude bash hook marks unsafe Codex commands explicitly.\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results =
        search_text_index(&kb_dir, &index_dir, "claude bash hook codex unsafe", 3).unwrap();
    let paths = results
        .iter()
        .map(|result| result.path.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        paths,
        [
            "memory/notable-facts/project-claude-bash-hook.md",
            "memory/manual-memories/project-claude-bash-hook.md",
            "memory/decision.md",
        ]
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn quality_gate_preserves_amdgpu_first_and_absent_query_behavior() {
    let root = unique_temp_dir("kb-text-quality-preservation");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(kb_dir.join("memory/briefs")).unwrap();
    std::fs::write(
        kb_dir.join("memory/briefs/kernel-panic-suspend-amdgpu.md"),
        "# Kernel Panic\nRouter suspend AMDGPU investigation evidence.\n",
    )
    .unwrap();
    std::fs::write(
        kb_dir.join("router-suspend.md"),
        "# Router Suspend\nUnrelated networking note.\n",
    )
    .unwrap();
    build_text_index(&kb_dir, &index_dir).unwrap();

    let results = search_text_index(&kb_dir, &index_dir, "router suspend amdgpu", 3).unwrap();
    let absent = search_text_index(&kb_dir, &index_dir, "zzzzqvwxjkl987654321", 3).unwrap();

    assert_eq!(
        results[0].path,
        "memory/briefs/kernel-panic-suspend-amdgpu.md"
    );
    assert!(absent.is_empty());
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
fn search_kb_context_fetches_exact_node_content_for_enrich() {
    let root = unique_temp_dir("kb-page-index-enrich-content");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    std::fs::create_dir_all(&kb_dir).unwrap();
    std::fs::write(
        kb_dir.join("agent.md"),
        "# Agent Rules\nIntro text.\n\n## Frontend\nLoad the frontend design skill immediately before UI work.\n",
    )
    .unwrap();

    build_text_index(&kb_dir, &index_dir).unwrap();
    let results =
        search_kb_context(&kb_dir, &index_dir, "frontend design skill immediately", 3).unwrap();

    assert_eq!(results[0].path, "agent.md");
    assert_eq!(results[0].node_id, "4-5");
    assert_eq!(
        results[0].text,
        "## Frontend\nLoad the frontend design skill immediately before UI work.\n"
    );
}
