use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn temp_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("claude-memory-{name}-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&path).expect("create temporary directory");
    path
}

fn shell_quote(value: &Path) -> String {
    format!("'{}'", value.to_string_lossy().replace('\'', "'\"'\"'"))
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
    assert!(
        stdout.contains(&format!(
            "--kb {} --index {}",
            shell_quote(&kb_dir),
            shell_quote(&index_dir)
        )),
        "{stdout}"
    );
}

#[test]
fn stale_query_fails_without_rebuilding() {
    let root = temp_dir("kb-cli-stale");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    let source = write_fixture(&kb_dir);
    assert!(build(&kb_dir, &index_dir).status.success());
    let nodes_before = fs::read(index_dir.join("nodes.tsv")).expect("read nodes before query");
    let manifest_before =
        fs::read(index_dir.join("manifest.tsv")).expect("read manifest before query");

    fs::write(
        &source,
        "# Changed\n\nDifferent content with another size.\n",
    )
    .expect("change source");
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
    assert!(String::from_utf8_lossy(&output.stderr).contains("stale"));
    assert_eq!(fs::read(index_dir.join("nodes.tsv")).unwrap(), nodes_before);
    assert_eq!(
        fs::read(index_dir.join("manifest.tsv")).unwrap(),
        manifest_before
    );
}

#[test]
fn query_rejects_added_deleted_and_missing_kb_files() {
    for change in ["added", "deleted", "missing"] {
        let root = temp_dir(&format!("kb-cli-{change}"));
        let kb_dir = root.join("kb");
        let index_dir = root.join("index");
        fs::create_dir_all(&kb_dir).expect("create KB directory");
        let source = (change != "missing").then(|| write_fixture(&kb_dir));
        assert!(build(&kb_dir, &index_dir).status.success());
        match change {
            "added" => fs::write(kb_dir.join("added.md"), "# Added\nNew page.\n").unwrap(),
            "deleted" => fs::remove_file(source.expect("source exists")).unwrap(),
            "missing" => fs::remove_dir_all(&kb_dir).unwrap(),
            _ => unreachable!(),
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

        assert!(!output.status.success(), "{change} should be rejected");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(if change == "missing" {
                "does not exist"
            } else {
                "stale"
            }),
            "{change}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn json_only_kb_commands_are_retired() {
    for command in ["document", "structure"] {
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
fn content_fetch_preserves_exact_line_endings() {
    let root = temp_dir("kb-cli-content-crlf");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    fs::write(
        kb_dir.join("windows.md"),
        b"# Windows\r\nFirst\r\nSecond\r\n",
    )
    .expect("write CRLF fixture");
    assert!(build(&kb_dir, &index_dir).status.success());

    let output = cli(&[
        "kb-page-index",
        "content",
        "windows.md",
        "2-3",
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
    assert_eq!(output.stdout, b"First\r\nSecond\r\n");
}

#[test]
fn content_fetch_reads_exact_markdown_line_range() {
    let root = temp_dir("kb-cli-content");
    let kb_dir = root.join("kb");
    let index_dir = root.join("index");
    fs::create_dir_all(&kb_dir).expect("create KB directory");
    write_fixture(&kb_dir);
    assert!(build(&kb_dir, &index_dir).status.success());

    let output = cli(&[
        "kb-page-index",
        "content",
        "guide.md",
        "5-8",
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
    assert_eq!(
        String::from_utf8(output.stdout).expect("UTF-8 stdout"),
        "## Recovery\n\nExact recovery command.\nFinal line.\n"
    );
}
