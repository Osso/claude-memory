use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("claude-memory-{name}-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&path).expect("create temporary directory");
    path
}

fn cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_claude-memory"))
        .args(args)
        .output()
        .expect("run claude-memory")
}

fn write_fixture(kb_dir: &Path) -> PathBuf {
    let path = kb_dir.join("guide.md");
    fs::write(
        &path,
        "# Router Guide\n\nIntro line.\n\n## Recovery\n\nExact recovery command.\nFinal line.\n",
    )
    .expect("write fixture");
    path
}

fn build(kb_dir: &Path, index_dir: &Path) -> Output {
    cli(&[
        "kb-page-index",
        "build",
        "--kb",
        kb_dir.to_str().expect("UTF-8 KB path"),
        "--output",
        index_dir.to_str().expect("UTF-8 index path"),
    ])
}

#[test]
fn explicit_build_writes_only_text_index_files() {
    let root = temp_dir("kb-cli-build");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);
    fs::create_dir_all(&index_dir).expect("create old index directory");
    fs::write(index_dir.join("index.json"), "legacy").expect("write legacy index");

    let output = build(&kb_dir, &index_dir);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut names = fs::read_dir(&index_dir)
        .expect("read index directory")
        .map(|entry| entry.expect("read index entry").file_name())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["manifest.tsv", "nodes.tsv"]);
}

#[test]
fn build_creates_missing_nested_output_parents() {
    let root = temp_dir("kb-cli-nested-output");
    let kb_dir = root.join("kb");
    let index_dir = root.join("new/child/index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);

    let output = build(&kb_dir, &index_dir);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(index_dir.join("nodes.tsv").is_file());
    assert!(index_dir.join("manifest.tsv").is_file());
}

#[test]
fn build_creates_relative_nested_output() {
    let root = temp_dir("kb-cli-relative-output");
    let kb_dir = root.join("kb");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_claude-memory"))
        .current_dir(&root)
        .args([
            "kb-page-index",
            "build",
            "--kb",
            "kb",
            "--output",
            "new/child/index",
        ])
        .output()
        .expect("run claude-memory");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(root.join("new/child/index/nodes.tsv").is_file());
    assert!(root.join("new/child/index/manifest.tsv").is_file());
}

#[test]
fn build_rejects_output_that_contains_the_kb() {
    let root = temp_dir("kb-cli-overlap");
    let kb_dir = root.join("kb");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    let source = write_fixture(&kb_dir);
    let source_before = fs::read(&source).expect("read source before build");

    let output = build(&kb_dir, &root);

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("overlap"));
    assert_eq!(
        fs::read(&source).expect("read source after build"),
        source_before
    );
}

#[test]
fn query_reads_explicit_text_index() {
    let root = temp_dir("kb cli's query");
    let kb_dir = root.join("knowledge base");
    let index_dir = root.join("text index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);
    assert!(build(&kb_dir, &index_dir).status.success());

    let output = cli(&[
        "kb-page-index",
        "query",
        "recovery command",
        "--kb",
        kb_dir.to_str().expect("UTF-8 KB path"),
        "--index",
        index_dir.to_str().expect("UTF-8 index path"),
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    assert!(
        stdout.contains("guide.md#5-8 > Router Guide > Recovery"),
        "{stdout}"
    );
    assert!(stdout.contains("Exact recovery command."), "{stdout}");
    assert!(!stdout.contains("next:"), "{stdout}");
}

#[test]
fn query_builds_a_missing_index() {
    let root = temp_dir("kb-cli-missing-index");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);

    let output = cli(&[
        "kb-page-index",
        "query",
        "recovery command",
        "--kb",
        kb_dir.to_str().expect("UTF-8 KB path"),
        "--index",
        index_dir.to_str().expect("UTF-8 index path"),
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(index_dir.join("nodes.tsv").is_file());
    assert!(index_dir.join("manifest.tsv").is_file());
}

#[test]
fn stale_query_rebuilds_changed_source() {
    let root = temp_dir("kb-cli-stale");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    let source = write_fixture(&kb_dir);
    assert!(build(&kb_dir, &index_dir).status.success());
    let manifest_before = fs::read(index_dir.join("manifest.tsv")).expect("read manifest");

    fs::write(
        &source,
        "# Changed\n\nReplacement phrase from the changed source.\n",
    )
    .expect("change source");
    let output = cli(&[
        "kb-page-index",
        "query",
        "replacement phrase",
        "--kb",
        kb_dir.to_str().expect("UTF-8 KB path"),
        "--index",
        index_dir.to_str().expect("UTF-8 index path"),
    ]);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    assert!(stdout.contains("Replacement phrase from the changed source."));
    assert_ne!(
        fs::read(index_dir.join("manifest.tsv")).expect("read rebuilt manifest"),
        manifest_before
    );
}

#[test]
fn query_rebuilds_after_added_or_deleted_kb_files() {
    for change in ["added", "deleted"] {
        let root = temp_dir(&format!("kb-cli-{change}"));
        let kb_dir = root.join("kb");
        let index_dir = root.join("index");
        fs::create_dir_all(&kb_dir).expect("create KB directory");
        let source = write_fixture(&kb_dir);
        assert!(build(&kb_dir, &index_dir).status.success());
        let query = match change {
            "added" => {
                fs::write(
                    kb_dir.join("added.md"),
                    "# Added\nUnique addition phrase.\n",
                )
                .unwrap();
                "unique addition"
            }
            "deleted" => {
                fs::remove_file(source).unwrap();
                "recovery command"
            }
            _ => unreachable!(),
        };

        let output = cli(&[
            "kb-page-index",
            "query",
            query,
            "--kb",
            kb_dir.to_str().expect("UTF-8 KB path"),
            "--index",
            index_dir.to_str().expect("UTF-8 index path"),
        ]);

        assert!(
            output.status.success(),
            "{change}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
        if change == "added" {
            assert!(stdout.contains("Unique addition phrase."), "{stdout}");
        } else {
            assert!(stdout.contains("(no KB notes found)"), "{stdout}");
        }
    }
}

#[test]
fn query_fails_when_the_kb_directory_is_missing() {
    let root = temp_dir("kb-cli-missing-kb");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);
    assert!(build(&kb_dir, &index_dir).status.success());
    fs::remove_dir_all(&kb_dir).expect("remove KB directory");

    let output = cli(&[
        "kb-page-index",
        "query",
        "recovery command",
        "--kb",
        kb_dir.to_str().expect("UTF-8 KB path"),
        "--index",
        index_dir.to_str().expect("UTF-8 index path"),
    ]);

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("does not exist"));
}

#[test]
fn retired_kb_commands_are_rejected() {
    for command in ["document", "structure", "content"] {
        let output = cli(&["kb-page-index", command, "guide.md"]);
        assert!(!output.status.success(), "{command} should be rejected");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn concurrent_queries_share_one_valid_rebuild() {
    let root = temp_dir("kb-cli-concurrent-rebuild");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);

    let mut children = (0..4)
        .map(|_| {
            Command::new(env!("CARGO_BIN_EXE_claude-memory"))
                .args([
                    "kb-page-index",
                    "query",
                    "recovery command",
                    "--kb",
                    kb_dir.to_str().expect("UTF-8 KB path"),
                    "--index",
                    index_dir.to_str().expect("UTF-8 index path"),
                ])
                .spawn()
                .expect("spawn query")
        })
        .collect::<Vec<_>>();

    for child in &mut children {
        assert!(child.wait().expect("wait for query").success());
    }
    let output = cli(&[
        "kb-page-index",
        "query",
        "recovery command",
        "--kb",
        kb_dir.to_str().expect("UTF-8 KB path"),
        "--index",
        index_dir.to_str().expect("UTF-8 index path"),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Exact recovery command."));
}
