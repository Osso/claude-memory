use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::{Parser, Subcommand};
use claude_memory::index::COLLECTION_SESSION_HISTORY;
use claude_memory::migration::{
    DestinationPoint, HistoryClassification, LegacyCollection, LegacyPoint, classify_history_points,
};
use claude_memory::qdrant_hybrid::create_hybrid_collection;
use qdrant_client::Qdrant;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const QDRANT_HTTP: &str = "http://127.0.0.1:6333";
const QDRANT_GRPC: &str = "http://127.0.0.1:6334";
const LEGACY_COLLECTIONS: [&str; 4] = [
    "claude-memory",
    "claude-answers",
    "claude-memory-units",
    "claude-notable-facts",
];
const SCROLL_LIMIT: usize = 256;
const UPSERT_BATCH_SIZE: usize = 128;

#[derive(Parser)]
#[command(name = "claude-memory-migrate")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Classify legacy history points without writing anything.
    Plan,
    /// Back up all legacy collections, migrate session history, and verify parity.
    Apply {
        /// Parent directory for a unique permission-locked backup directory.
        #[arg(long)]
        backup_dir: PathBuf,
    },
    /// Verify legacy-to-destination session-history parity without writing.
    Verify,
}

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    result: T,
}

#[derive(Debug, Deserialize)]
struct ScrollResult {
    points: Vec<RestPoint>,
    next_page_offset: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RestPoint {
    id: Value,
    vector: Option<Value>,
    payload: Value,
}

#[derive(Debug, Deserialize)]
struct SnapshotDescription {
    name: String,
    size: u64,
    checksum: Option<String>,
}

#[derive(Debug)]
struct BackupArtifact {
    collection: String,
    path: PathBuf,
    size: u64,
    checksum: String,
}

#[derive(Debug, Eq, PartialEq)]
struct SourceWatermark {
    memory_points: usize,
    answer_points: usize,
    digest: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = http_client()?;
    match Cli::parse().command {
        Command::Plan => print_plan(&classify_sources(&client, false).await?),
        Command::Apply { backup_dir } => apply_migration(&client, &backup_dir).await?,
        Command::Verify => verify_live_parity(&client).await?,
    }
    Ok(())
}

fn http_client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(1_800))
        .build()
        .context("failed to build Qdrant HTTP client")
}

async fn apply_migration(client: &Client, backup_root: &Path) -> Result<()> {
    let before_backup = source_watermark(client).await?;
    let backup_dir = create_backup_dir(backup_root)?;
    let artifacts = backup_legacy_collections(client, &backup_dir).await?;
    verify_backup_set(&artifacts)?;
    print_backup_artifacts(&backup_dir, &artifacts);

    let classification = classify_sources(client, true).await?;
    verify_stable_source(&before_backup, &source_watermark(client).await?)?;
    verify_stable_source(&before_backup, &source_watermark(client).await?)?;

    create_destination_collection().await?;
    let created = true;
    let migration = migrate_destination(client, &classification).await;
    if migration.is_err() && created {
        rollback_destination().await?;
    }
    migration?;
    print_plan(&classification);
    Ok(())
}

fn create_backup_dir(root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(root).with_context(|| format!("failed to create {}", root.display()))?;
    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let path = root.join(format!(
        "qdrant-migration-{timestamp}-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    fs::create_dir(&path).with_context(|| format!("failed to create {}", path.display()))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
    Ok(path)
}

async fn backup_legacy_collections(
    client: &Client,
    backup_dir: &Path,
) -> Result<Vec<BackupArtifact>> {
    let mut artifacts = Vec::new();
    for collection in LEGACY_COLLECTIONS {
        artifacts.push(backup_collection(client, collection, backup_dir).await?);
    }
    Ok(artifacts)
}

async fn backup_collection(
    client: &Client,
    collection: &str,
    backup_dir: &Path,
) -> Result<BackupArtifact> {
    let create_url = format!("{QDRANT_HTTP}/collections/{collection}/snapshots");
    let response: ApiResponse<SnapshotDescription> = client
        .post(create_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    download_and_verify_snapshot(client, collection, response.result, backup_dir).await
}

async fn download_and_verify_snapshot(
    client: &Client,
    collection: &str,
    snapshot: SnapshotDescription,
    backup_dir: &Path,
) -> Result<BackupArtifact> {
    validate_snapshot_name(&snapshot.name)?;
    let url = format!(
        "{QDRANT_HTTP}/collections/{collection}/snapshots/{}",
        snapshot.name
    );
    let bytes = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    persist_verified_snapshot(collection, snapshot, backup_dir, &bytes)
}

fn persist_verified_snapshot(
    collection: &str,
    snapshot: SnapshotDescription,
    backup_dir: &Path,
    bytes: &[u8],
) -> Result<BackupArtifact> {
    let path = backup_dir.join(format!("{collection}--{}", snapshot.name));
    fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    let checksum = sha256_hex(bytes);
    verify_snapshot_metadata(&snapshot, bytes.len() as u64, &checksum)?;
    verify_written_snapshot(&path, bytes, &checksum)?;
    Ok(BackupArtifact {
        collection: collection.to_string(),
        path,
        size: bytes.len() as u64,
        checksum,
    })
}

fn verify_snapshot_metadata(
    snapshot: &SnapshotDescription,
    size: u64,
    checksum: &str,
) -> Result<()> {
    if snapshot.size != size {
        bail!(
            "snapshot size mismatch for {}: {} != {size}",
            snapshot.name,
            snapshot.size
        );
    }
    let expected = snapshot
        .checksum
        .as_deref()
        .context("Qdrant snapshot metadata omitted checksum")?;
    if !expected.eq_ignore_ascii_case(checksum) {
        bail!("snapshot checksum mismatch for {}", snapshot.name);
    }
    Ok(())
}

fn validate_snapshot_name(name: &str) -> Result<()> {
    let safe = !name.is_empty()
        && !matches!(name, "." | "..")
        && name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
        });
    if !safe {
        bail!("unsafe snapshot name returned by Qdrant: {name:?}");
    }
    Ok(())
}

fn verify_written_snapshot(path: &Path, expected_bytes: &[u8], expected_hash: &str) -> Result<()> {
    let written = fs::read(path).with_context(|| format!("failed to reread {}", path.display()))?;
    if written != expected_bytes || sha256_hex(&written) != expected_hash {
        bail!("written snapshot integrity mismatch: {}", path.display());
    }
    Ok(())
}

fn verify_backup_set(artifacts: &[BackupArtifact]) -> Result<()> {
    if artifacts.len() != LEGACY_COLLECTIONS.len() {
        bail!("incomplete backup set: {} artifacts", artifacts.len());
    }
    for artifact in artifacts {
        let bytes = fs::read(&artifact.path)?;
        let mode = fs::metadata(&artifact.path)?.permissions().mode() & 0o777;
        if bytes.len() as u64 != artifact.size
            || sha256_hex(&bytes) != artifact.checksum
            || mode != 0o600
        {
            bail!(
                "backup set verification failed: {}",
                artifact.path.display()
            );
        }
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

async fn classify_sources(client: &Client, with_vectors: bool) -> Result<HistoryClassification> {
    let mut points = scroll_legacy_collection(
        client,
        "claude-memory",
        LegacyCollection::Memory,
        with_vectors,
    )
    .await?;
    points.extend(
        scroll_legacy_collection(
            client,
            "claude-answers",
            LegacyCollection::Answers,
            with_vectors,
        )
        .await?,
    );
    classify_history_points(points).map_err(Into::into)
}

async fn scroll_legacy_collection(
    client: &Client,
    collection_name: &str,
    collection: LegacyCollection,
    with_vectors: bool,
) -> Result<Vec<LegacyPoint>> {
    let points = scroll_collection(client, collection_name, with_vectors).await?;
    points
        .into_iter()
        .map(|point| {
            let vector = required_vector(point.vector, with_vectors)?;
            Ok(LegacyPoint {
                collection,
                vector,
                payload: point.payload,
            })
        })
        .collect()
}

fn required_vector(vector: Option<Value>, required: bool) -> Result<Value> {
    match (vector, required) {
        (Some(vector), _) => Ok(vector),
        (None, false) => Ok(Value::Null),
        (None, true) => bail!("eligible legacy point is missing vectors"),
    }
}

async fn scroll_collection(
    client: &Client,
    collection: &str,
    with_vectors: bool,
) -> Result<Vec<RestPoint>> {
    let mut points = Vec::new();
    let mut offset = None;
    loop {
        let page = scroll_page(client, collection, with_vectors, offset.take()).await?;
        points.extend(page.points);
        offset = page.next_page_offset;
        if offset.is_none() {
            return Ok(points);
        }
    }
}

async fn scroll_page(
    client: &Client,
    collection: &str,
    with_vectors: bool,
    offset: Option<Value>,
) -> Result<ScrollResult> {
    let url = format!("{QDRANT_HTTP}/collections/{collection}/points/scroll");
    let body = json!({
        "limit": SCROLL_LIMIT,
        "offset": offset,
        "with_payload": true,
        "with_vector": with_vectors
    });
    let response: ApiResponse<ScrollResult> = client
        .post(url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(response.result)
}

async fn create_destination_collection() -> Result<()> {
    let qdrant = Qdrant::from_url(QDRANT_GRPC).build()?;
    create_hybrid_collection(&qdrant, COLLECTION_SESSION_HISTORY).await
}

async fn migrate_destination(
    client: &Client,
    classification: &HistoryClassification,
) -> Result<()> {
    upsert_destination_points(client, &classification.destination_points).await?;
    verify_destination_parity(client, classification).await
}

async fn rollback_destination() -> Result<()> {
    let qdrant = Qdrant::from_url(QDRANT_GRPC).build()?;
    qdrant.delete_collection(COLLECTION_SESSION_HISTORY).await?;
    Ok(())
}

async fn upsert_destination_points(client: &Client, points: &[DestinationPoint]) -> Result<()> {
    for batch in points.chunks(UPSERT_BATCH_SIZE) {
        let body = json!({"points": batch.iter().map(destination_json).collect::<Vec<_>>()});
        let url =
            format!("{QDRANT_HTTP}/collections/{COLLECTION_SESSION_HISTORY}/points?wait=true");
        client
            .put(url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
    }
    Ok(())
}

fn destination_json(point: &DestinationPoint) -> Value {
    json!({
        "id": point.id,
        "vector": point.vector,
        "payload": point.payload
    })
}

async fn verify_live_parity(client: &Client) -> Result<()> {
    let expected = classify_sources(client, true).await?;
    verify_destination_parity(client, &expected).await
}

async fn verify_destination_parity(
    client: &Client,
    expected: &HistoryClassification,
) -> Result<()> {
    let actual = scroll_collection(client, COLLECTION_SESSION_HISTORY, true).await?;
    verify_destination_points(&expected.destination_points, &actual)?;
    let keys = destination_keys(&expected.destination_points)?;
    print_grouped_parity(&keys);
    Ok(())
}

async fn source_watermark(client: &Client) -> Result<SourceWatermark> {
    let memory = scroll_collection(client, "claude-memory", true).await?;
    let answers = scroll_collection(client, "claude-answers", true).await?;
    build_source_watermark(&memory, &answers)
}

fn build_source_watermark(memory: &[RestPoint], answers: &[RestPoint]) -> Result<SourceWatermark> {
    let mut entries = source_entries("claude-memory", memory)?;
    entries.extend(source_entries("claude-answers", answers)?);
    entries.sort();
    let mut hasher = Sha256::new();
    for entry in entries {
        hash_watermark_part(&mut hasher, &entry);
    }
    Ok(SourceWatermark {
        memory_points: memory.len(),
        answer_points: answers.len(),
        digest: hex::encode(hasher.finalize()),
    })
}

fn source_entries(collection: &str, points: &[RestPoint]) -> Result<Vec<Vec<u8>>> {
    points
        .iter()
        .map(|point| {
            serde_json::to_vec(&json!({
                "collection": collection,
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            }))
            .map_err(Into::into)
        })
        .collect()
}

fn hash_watermark_part(hasher: &mut Sha256, part: &[u8]) {
    hasher.update((part.len() as u64).to_be_bytes());
    hasher.update(part);
}

fn verify_stable_source(expected: &SourceWatermark, actual: &SourceWatermark) -> Result<()> {
    if expected != actual {
        bail!("legacy source watermark changed during migration");
    }
    Ok(())
}

fn verify_destination_points(expected: &[DestinationPoint], actual: &[RestPoint]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "destination point count mismatch: expected={} actual={}",
            expected.len(),
            actual.len()
        );
    }
    let expected = expected_point_map(expected);
    let actual = actual_point_map(actual)?;
    if expected != actual {
        bail!(
            "full destination parity failed: expected={} actual={}",
            expected.len(),
            actual.len()
        );
    }
    Ok(())
}

fn expected_point_map(points: &[DestinationPoint]) -> BTreeMap<String, (Value, Value)> {
    points
        .iter()
        .map(|point| {
            (
                point.id.clone(),
                (point.vector.clone(), point.payload.clone()),
            )
        })
        .collect()
}

fn actual_point_map(points: &[RestPoint]) -> Result<BTreeMap<String, (Value, Value)>> {
    points
        .iter()
        .map(|point| {
            let id = point
                .id
                .as_str()
                .context("destination point ID is not a UUID string")?;
            let vector = point
                .vector
                .clone()
                .context("destination point is missing vectors")?;
            Ok((id.to_string(), (vector, point.payload.clone())))
        })
        .collect()
}

fn destination_keys(points: &[DestinationPoint]) -> Result<BTreeSet<(String, String, String)>> {
    points
        .iter()
        .map(|point| payload_key(&point.payload))
        .collect()
}

fn payload_key(payload: &Value) -> Result<(String, String, String)> {
    let payload = payload
        .as_object()
        .context("point payload is not an object")?;
    Ok((
        json_string(payload, "type")?,
        json_string(payload, "source")?,
        json_string(payload, "hash")?,
    ))
}

fn json_string(payload: &serde_json::Map<String, Value>, field: &'static str) -> Result<String> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("missing destination payload field: {field}"))
}

fn print_grouped_parity(keys: &BTreeSet<(String, String, String)>) {
    let mut grouped = BTreeMap::new();
    for (history_type, source, _) in keys {
        *grouped.entry((history_type, source)).or_insert(0_usize) += 1;
    }
    println!("Verified {} exact session-history keys", keys.len());
    for ((history_type, source), count) in grouped {
        println!("  {history_type}/{source}: {count}");
    }
}

fn print_plan(plan: &HistoryClassification) {
    println!("raw points: {}", plan.raw_points);
    println!("eligible points: {}", plan.eligible_points);
    println!("unique destination points: {}", plan.unique_points);
    println!("duplicate points: {}", plan.duplicate_points);
    println!("skipped unsupported points: {}", plan.skipped_points);
    for ((history_type, source), count) in &plan.grouped_unique {
        println!("  {history_type}/{source}: {count}");
    }
}

fn print_backup_artifacts(backup_dir: &Path, artifacts: &[BackupArtifact]) {
    println!("Backup directory: {}", backup_dir.display());
    for artifact in artifacts {
        println!(
            "  {}: {} bytes sha256={} {}",
            artifact.collection,
            artifact.size,
            artifact.checksum,
            artifact.path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backup_directories_are_unique_and_permission_locked() {
        let root = unique_temp_dir("backup-dir");
        let first = create_backup_dir(&root).unwrap();
        let second = create_backup_dir(&root).unwrap();

        assert_ne!(first, second);
        assert_eq!(
            fs::metadata(first).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(second).unwrap().permissions().mode() & 0o777,
            0o700
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn snapshot_metadata_rejects_size_and_checksum_mismatch() {
        let snapshot = SnapshotDescription {
            name: "fixture.snapshot".to_string(),
            size: 4,
            checksum: Some("deadbeef".to_string()),
        };

        assert!(verify_snapshot_metadata(&snapshot, 3, "deadbeef").is_err());
        assert!(verify_snapshot_metadata(&snapshot, 4, "badcafe").is_err());
    }

    #[test]
    fn snapshot_metadata_requires_advertised_checksum() {
        let snapshot = SnapshotDescription {
            name: "fixture.snapshot".to_string(),
            size: 4,
            checksum: None,
        };

        assert!(verify_snapshot_metadata(&snapshot, 4, "deadbeef").is_err());
    }

    #[test]
    fn unsafe_snapshot_names_are_rejected() {
        for name in [
            "../escape.snapshot",
            "/tmp/escape",
            "nested/file",
            ".",
            "..",
            "",
        ] {
            assert!(validate_snapshot_name(name).is_err(), "accepted {name:?}");
        }
        assert!(validate_snapshot_name("collection-2026_07_17.snapshot").is_ok());
    }

    #[test]
    fn missing_requested_vectors_fail_closed() {
        assert!(required_vector(None, true).is_err());
        assert_eq!(required_vector(None, false).unwrap(), Value::Null);
    }

    #[test]
    fn written_snapshot_is_byte_and_hash_verified() {
        let root = unique_temp_dir("snapshot-write");
        fs::create_dir_all(&root).unwrap();
        let path = root.join("fixture.snapshot");
        let bytes = b"snapshot-bytes";
        fs::write(&path, bytes).unwrap();

        assert!(verify_written_snapshot(&path, bytes, &sha256_hex(bytes)).is_ok());
        assert!(verify_written_snapshot(&path, b"different", &sha256_hex(bytes)).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_watermark_detects_vector_only_changes() {
        let memory = vec![rest_fixture()];
        let watermark = build_source_watermark(&memory, &[]).unwrap();
        let mut changed = vec![rest_fixture()];
        changed[0].vector = Some(json!({"dense": [2.0]}));
        let changed = build_source_watermark(&changed, &[]).unwrap();

        assert!(verify_stable_source(&watermark, &changed).is_err());
    }

    #[test]
    fn full_parity_rejects_vector_payload_id_and_count_changes() {
        let expected = vec![destination_fixture()];
        let matching = vec![rest_fixture()];
        assert!(verify_destination_points(&expected, &matching).is_ok());

        let mut wrong_vector = rest_fixture();
        wrong_vector.vector = Some(json!({"dense": [2.0]}));
        assert!(verify_destination_points(&expected, &[wrong_vector]).is_err());

        let mut wrong_payload = rest_fixture();
        wrong_payload.payload["path"] = "wrong.jsonl".into();
        assert!(verify_destination_points(&expected, &[wrong_payload]).is_err());

        let mut wrong_id = rest_fixture();
        wrong_id.id = "other-id".into();
        assert!(verify_destination_points(&expected, &[wrong_id]).is_err());

        assert!(verify_destination_points(&expected, &[rest_fixture(), rest_fixture()]).is_err());
    }

    fn destination_fixture() -> DestinationPoint {
        DestinationPoint {
            id: "point-id".to_string(),
            vector: json!({"dense": [1.0]}),
            payload: json!({
                "type": "prompt",
                "source": "session",
                "hash": "prompt:session:hash",
                "text": "text",
                "path": "session.jsonl",
                "session_id": "session-1"
            }),
        }
    }

    fn rest_fixture() -> RestPoint {
        let destination = destination_fixture();
        RestPoint {
            id: destination.id.into(),
            vector: Some(destination.vector),
            payload: destination.payload,
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "claude-memory-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }
}
