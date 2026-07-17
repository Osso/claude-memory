use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KnowledgeCollection {
    MemoryUnits,
    NotableFacts,
    LegacyMemory,
}

impl KnowledgeCollection {
    fn as_str(self) -> &'static str {
        match self {
            Self::MemoryUnits => "claude-memory-units",
            Self::NotableFacts => "claude-notable-facts",
            Self::LegacyMemory => "claude-memory",
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnowledgePoint {
    pub collection: KnowledgeCollection,
    pub id: String,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportDocument {
    pub relative_path: PathBuf,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ManifestEntry {
    pub collection: String,
    pub source_point_id: String,
    pub source: Option<String>,
    pub source_path: Option<String>,
    pub legacy_hash: Option<String>,
    pub disposition: String,
    pub relative_path: Option<PathBuf>,
    pub anchor: Option<String>,
    pub content_hash: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ExportCounts {
    pub raw: usize,
    pub exported_unique: usize,
    pub duplicates: usize,
    pub quarantined: usize,
    pub quarantined_source_points: usize,
    pub excluded_kb_vectors: usize,
    pub excluded_non_durable: usize,
    pub by_destination: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportPlan {
    pub counts: ExportCounts,
    pub documents: Vec<ExportDocument>,
    pub manifest: Vec<ManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportParity {
    pub source_points: usize,
    pub manifest_entries: usize,
    pub unique_exported_records: usize,
    pub content_hashes_verified: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Destination {
    Friction,
    NotableFacts,
    ManualMemories,
    Quarantine,
    ExcludedKbVector,
    ExcludedNonDurable,
}

impl Destination {
    fn label(&self) -> &'static str {
        match self {
            Self::Friction => "friction",
            Self::NotableFacts => "notable-facts",
            Self::ManualMemories => "manual-memories",
            Self::Quarantine => "quarantine",
            Self::ExcludedKbVector => "excluded-kb-vector",
            Self::ExcludedNonDurable => "excluded-non-durable",
        }
    }

    fn is_exported(&self) -> bool {
        !matches!(self, Self::ExcludedKbVector | Self::ExcludedNonDurable)
    }
}

#[derive(Debug, Clone)]
struct ClassifiedPoint {
    collection: KnowledgeCollection,
    id: String,
    destination: Destination,
    source: Option<String>,
    source_path: Option<String>,
    legacy_hash: Option<String>,
    project: String,
    text: String,
    content_hash: String,
    source_session: Option<String>,
    seen_in_sessions: Vec<String>,
    source_turn: Option<i64>,
    category: Option<String>,
    topics: Vec<String>,
    created_at: Option<String>,
}

#[derive(Debug, Clone)]
struct ExportRecord {
    destination: Destination,
    project: String,
    text: String,
    content_hash: String,
    source_points: Vec<(KnowledgeCollection, String)>,
    sources: BTreeSet<String>,
    source_paths: BTreeSet<String>,
    legacy_hashes: BTreeSet<String>,
    source_sessions: BTreeSet<String>,
    source_turns: BTreeSet<i64>,
    categories: BTreeSet<String>,
    topics: BTreeSet<String>,
    created_at: BTreeSet<String>,
}

pub fn classify_and_render(points: &[KnowledgePoint]) -> Result<ExportPlan> {
    let classified = points
        .iter()
        .map(classify_point)
        .collect::<Result<Vec<_>>>()?;
    let records = deduplicate_points(&classified);
    let documents = render_documents(&records);
    let manifest = build_manifest(&classified, &records)?;
    let counts = build_counts(points.len(), &classified, &records);

    Ok(ExportPlan {
        counts,
        documents,
        manifest,
    })
}

pub fn write_export(kb_root: &Path, plan: &ExportPlan) -> Result<()> {
    let manifest_path = PathBuf::from("memory/export-manifest.json");
    let manifest = render_manifest(plan)?;
    let writes = plan
        .documents
        .iter()
        .map(|document| {
            (
                document.relative_path.as_path(),
                document.content.as_bytes(),
            )
        })
        .chain(std::iter::once((
            manifest_path.as_path(),
            manifest.as_bytes(),
        )))
        .collect::<Vec<_>>();

    preflight_writes(kb_root, &writes)?;
    write_files(kb_root, &writes)
}

pub fn verify_written_export(kb_root: &Path, plan: &ExportPlan) -> Result<ExportParity> {
    verify_manifest(kb_root, plan)?;
    verify_documents(kb_root, &plan.documents)?;
    let exported = exported_manifest_entries(&plan.manifest);
    let verified_hashes = verify_persisted_content_hashes(kb_root, &exported)?;

    Ok(ExportParity {
        source_points: plan.counts.raw,
        manifest_entries: plan.manifest.len(),
        unique_exported_records: unique_record_count(&exported),
        content_hashes_verified: verified_hashes,
    })
}

pub fn verify_unclassified_count(plan: &ExportPlan, expected: usize) -> Result<()> {
    if plan.counts.quarantined_source_points != expected {
        bail!(
            "unclassified count changed: expected {expected}, found {}",
            plan.counts.quarantined_source_points
        );
    }
    Ok(())
}

pub fn verify_source_plan_unchanged(before: &ExportPlan, after: &ExportPlan) -> Result<()> {
    if before.counts != after.counts
        || before.manifest != after.manifest
        || before.documents != after.documents
    {
        bail!("durable knowledge source changed during export");
    }
    Ok(())
}

fn exported_manifest_entries(manifest: &[ManifestEntry]) -> Vec<&ManifestEntry> {
    manifest
        .iter()
        .filter(|entry| entry.relative_path.is_some())
        .collect()
}

fn verify_persisted_content_hashes(kb_root: &Path, entries: &[&ManifestEntry]) -> Result<usize> {
    let expected = expected_persisted_hashes(entries)?;
    let mut verified = 0;
    for ((relative_path, anchor), expected_hash) in expected {
        let content = fs::read_to_string(kb_root.join(&relative_path))?;
        let text = extract_record_text(&content, &anchor).with_context(|| {
            format!(
                "missing export anchor {anchor} in {}",
                relative_path.display()
            )
        })?;
        if hash_text(&text) != expected_hash {
            bail!(
                "content hash mismatch for {anchor} in {}",
                relative_path.display()
            );
        }
        verified += 1;
    }
    Ok(verified)
}

fn expected_persisted_hashes(
    entries: &[&ManifestEntry],
) -> Result<BTreeMap<(PathBuf, String), String>> {
    let mut expected = BTreeMap::new();
    for entry in entries {
        let path = entry
            .relative_path
            .clone()
            .context("exported manifest entry has no path")?;
        let anchor = entry
            .anchor
            .clone()
            .context("exported manifest entry has no anchor")?;
        let key = (path.clone(), anchor.clone());
        if let Some(existing_hash) = expected.get(&key) {
            if existing_hash != &entry.content_hash {
                bail!("conflicting export identity {anchor} in {}", path.display());
            }
            continue;
        }
        expected.insert(key, entry.content_hash.clone());
    }
    Ok(expected)
}

fn unique_record_count(entries: &[&ManifestEntry]) -> usize {
    entries
        .iter()
        .map(|entry| (entry.relative_path.as_ref(), entry.anchor.as_ref()))
        .collect::<BTreeSet<_>>()
        .len()
}

fn classify_point(point: &KnowledgePoint) -> Result<ClassifiedPoint> {
    let payload = point
        .payload
        .as_object()
        .with_context(|| format!("point {} payload is not an object", point.id))?;
    let text = required_string(payload.get("text"), "text", &point.id)?;
    let source = optional_string(payload.get("source"), "source", &point.id)?;
    let destination = classify_destination(point.collection, source.as_deref())?;
    let (source_path, legacy_hash) = source_path_and_hash(point, payload, &destination)?;

    Ok(ClassifiedPoint {
        collection: point.collection,
        id: point.id.clone(),
        destination,
        source,
        source_path,
        legacy_hash,
        project: optional_string(payload.get("project"), "project", &point.id)?.unwrap_or_default(),
        content_hash: hash_text(&text),
        text,
        source_session: optional_string(
            payload.get("source_session"),
            "source_session",
            &point.id,
        )?,
        seen_in_sessions: optional_string_list(
            payload.get("seen_in_sessions"),
            "seen_in_sessions",
            &point.id,
        )?,
        source_turn: optional_integer(payload.get("source_turn"), "source_turn", &point.id)?,
        category: optional_string(payload.get("category"), "category", &point.id)?,
        topics: optional_string_list(payload.get("topics"), "topics", &point.id)?,
        created_at: optional_string(payload.get("created_at"), "created_at", &point.id)?,
    })
}

fn source_path_and_hash(
    point: &KnowledgePoint,
    payload: &serde_json::Map<String, Value>,
    destination: &Destination,
) -> Result<(Option<String>, Option<String>)> {
    if point.collection == KnowledgeCollection::LegacyMemory
        && destination == &Destination::ManualMemories
    {
        return Ok((
            Some(required_string(payload.get("path"), "path", &point.id)?),
            Some(required_string(payload.get("hash"), "hash", &point.id)?),
        ));
    }
    Ok((
        optional_string(payload.get("path"), "path", &point.id)?,
        optional_string(payload.get("hash"), "hash", &point.id)?,
    ))
}

fn classify_destination(
    collection: KnowledgeCollection,
    source: Option<&str>,
) -> Result<Destination> {
    match (collection, source) {
        (_, Some("kb")) => Ok(Destination::ExcludedKbVector),
        (KnowledgeCollection::NotableFacts, Some("session")) => Ok(Destination::NotableFacts),
        (KnowledgeCollection::NotableFacts, _) => bail!("unsupported notable-fact source"),
        (KnowledgeCollection::LegacyMemory, Some("memory")) => Ok(Destination::ManualMemories),
        (KnowledgeCollection::LegacyMemory, Some("session" | "archive" | "summary")) => {
            Ok(Destination::ExcludedNonDurable)
        }
        (KnowledgeCollection::MemoryUnits, Some("session")) => Ok(Destination::Friction),
        (KnowledgeCollection::MemoryUnits, Some("memory")) => Ok(Destination::ManualMemories),
        (KnowledgeCollection::MemoryUnits, None | Some("")) => Ok(Destination::Quarantine),
        (KnowledgeCollection::MemoryUnits, _) => Ok(Destination::Quarantine),
        (KnowledgeCollection::LegacyMemory, _) => bail!("unsupported legacy memory source"),
    }
}

fn required_string(value: Option<&Value>, field: &str, point_id: &str) -> Result<String> {
    let value = optional_string(value, field, point_id)?
        .with_context(|| format!("point {point_id} is missing {field}"))?;
    if value.is_empty() {
        bail!("point {point_id} has empty {field}");
    }
    Ok(value)
}

fn optional_string(value: Option<&Value>, field: &str, point_id: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => bail!("point {point_id} has non-string {field}"),
    }
}

fn optional_string_list(value: Option<&Value>, field: &str, point_id: &str) -> Result<Vec<String>> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .with_context(|| format!("point {point_id} has non-string {field} entry"))
            })
            .collect(),
        Some(_) => bail!("point {point_id} has non-list {field}"),
    }
}

fn optional_integer(value: Option<&Value>, field: &str, point_id: &str) -> Result<Option<i64>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_i64()
            .map(Some)
            .with_context(|| format!("point {point_id} has non-integer {field}")),
        Some(_) => bail!("point {point_id} has non-integer {field}"),
    }
}

fn deduplicate_points(points: &[ClassifiedPoint]) -> Vec<ExportRecord> {
    let mut records = BTreeMap::<(Destination, String, String), ExportRecord>::new();
    for point in points {
        let key = (
            point.destination.clone(),
            point.project.clone(),
            point.content_hash.clone(),
        );
        let record = records
            .entry(key)
            .or_insert_with(|| ExportRecord::from(point));
        record.merge(point);
    }
    records.into_values().collect()
}

impl From<&ClassifiedPoint> for ExportRecord {
    fn from(point: &ClassifiedPoint) -> Self {
        Self {
            destination: point.destination.clone(),
            project: point.project.clone(),
            text: point.text.clone(),
            content_hash: point.content_hash.clone(),
            source_points: Vec::new(),
            sources: BTreeSet::new(),
            source_paths: BTreeSet::new(),
            legacy_hashes: BTreeSet::new(),
            source_sessions: BTreeSet::new(),
            source_turns: BTreeSet::new(),
            categories: BTreeSet::new(),
            topics: BTreeSet::new(),
            created_at: BTreeSet::new(),
        }
    }
}

impl ExportRecord {
    fn merge(&mut self, point: &ClassifiedPoint) {
        self.merge_source_identity(point);
        self.merge_session_provenance(point);
        self.merge_descriptive_provenance(point);
    }

    fn merge_source_identity(&mut self, point: &ClassifiedPoint) {
        if !self
            .source_points
            .iter()
            .any(|(collection, id)| *collection == point.collection && id == &point.id)
        {
            self.source_points
                .push((point.collection, point.id.clone()));
        }
        self.sources.insert(
            point
                .source
                .clone()
                .unwrap_or_else(|| "<missing>".to_string()),
        );
        insert_non_empty(&mut self.source_paths, point.source_path.as_ref());
        insert_non_empty(&mut self.legacy_hashes, point.legacy_hash.as_ref());
    }

    fn merge_session_provenance(&mut self, point: &ClassifiedPoint) {
        insert_non_empty(&mut self.source_sessions, point.source_session.as_ref());
        self.source_sessions.extend(
            point
                .seen_in_sessions
                .iter()
                .filter(|value| !value.is_empty())
                .cloned(),
        );
        if let Some(value) = point.source_turn {
            self.source_turns.insert(value);
        }
    }

    fn merge_descriptive_provenance(&mut self, point: &ClassifiedPoint) {
        insert_non_empty(&mut self.categories, point.category.as_ref());
        self.topics.extend(
            point
                .topics
                .iter()
                .filter(|value| !value.is_empty())
                .cloned(),
        );
        insert_non_empty(&mut self.created_at, point.created_at.as_ref());
    }

    fn relative_path(&self) -> Option<PathBuf> {
        match self.destination {
            Destination::Friction => Some(PathBuf::from(format!(
                "memory/friction/{}.md",
                project_filename(&self.project)
            ))),
            Destination::NotableFacts => Some(PathBuf::from(format!(
                "memory/notable-facts/{}.md",
                project_filename(&self.project)
            ))),
            Destination::ManualMemories => Some(PathBuf::from(format!(
                "memory/manual-memories/{}.md",
                project_filename(&self.project)
            ))),
            Destination::Quarantine => Some(PathBuf::from(
                "memory/quarantine/unclassified-memory-units.md",
            )),
            Destination::ExcludedKbVector | Destination::ExcludedNonDurable => None,
        }
    }

    fn anchor(&self) -> String {
        let scope = format!("{}\0{}", self.destination.label(), self.project);
        format!("memory-{}-{}", &self.content_hash[..16], hash_text(&scope))
    }
}

fn insert_non_empty(values: &mut BTreeSet<String>, value: Option<&String>) {
    if let Some(value) = value
        && !value.is_empty()
    {
        values.insert(value.clone());
    }
}

fn render_documents(records: &[ExportRecord]) -> Vec<ExportDocument> {
    let mut grouped = BTreeMap::<PathBuf, Vec<&ExportRecord>>::new();
    for record in records {
        if let Some(path) = record.relative_path() {
            grouped.entry(path).or_default().push(record);
        }
    }
    grouped
        .into_iter()
        .map(|(relative_path, records)| ExportDocument {
            content: render_document(&relative_path, &records),
            relative_path,
        })
        .collect()
}

fn render_document(path: &Path, records: &[&ExportRecord]) -> String {
    let title = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("durable-memory");
    let mut output = format!("# Durable memory: {title}\n\n");
    for record in records {
        output.push_str(&render_record(record));
    }
    output
}

fn render_record(record: &ExportRecord) -> String {
    let mut output = format!(
        "<a id=\"{}\"></a>\n## {}\n\n{}\n<!-- end-source-text -->\n\n",
        record.anchor(),
        &record.content_hash[..16],
        encode_source_text(&record.text)
    );
    output.push_str(&format!("- content hash: `{}`\n", record.content_hash));
    output.push_str(&format!(
        "- project: {}\n",
        encode_metadata(display_project(&record.project))
    ));
    for (collection, id) in &record.source_points {
        output.push_str(&format!("- source point: {}\n", encode_metadata(id)));
        output.push_str(&format!("- source collection: {}\n", collection.as_str()));
    }
    append_metadata(&mut output, "sources", &record.sources);
    append_metadata(&mut output, "source paths", &record.source_paths);
    append_metadata(&mut output, "legacy hashes", &record.legacy_hashes);
    append_metadata(&mut output, "seen in sessions", &record.source_sessions);
    append_metadata(&mut output, "source turns", &record.source_turns);
    append_metadata(&mut output, "categories", &record.categories);
    append_metadata(&mut output, "topics", &record.topics);
    append_metadata(&mut output, "created at", &record.created_at);
    output.push('\n');
    output
}

fn append_metadata<T: ToString + Ord>(output: &mut String, label: &str, values: &BTreeSet<T>) {
    if !values.is_empty() {
        output.push_str(&format!(
            "- {label}: {}\n",
            values
                .iter()
                .map(|value| encode_metadata(&value.to_string()))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
}

fn build_manifest(
    points: &[ClassifiedPoint],
    records: &[ExportRecord],
) -> Result<Vec<ManifestEntry>> {
    let record_map = records
        .iter()
        .map(|record| (record_key(record), record))
        .collect::<BTreeMap<_, _>>();
    points
        .iter()
        .map(|point| manifest_entry(point, &record_map))
        .collect()
}

fn record_key(record: &ExportRecord) -> (Destination, String, String) {
    (
        record.destination.clone(),
        record.project.clone(),
        record.content_hash.clone(),
    )
}

fn manifest_entry(
    point: &ClassifiedPoint,
    record_map: &BTreeMap<(Destination, String, String), &ExportRecord>,
) -> Result<ManifestEntry> {
    let key = (
        point.destination.clone(),
        point.project.clone(),
        point.content_hash.clone(),
    );
    let record = record_map
        .get(&key)
        .context("classified point has no export record")?;
    Ok(ManifestEntry {
        collection: point.collection.as_str().to_string(),
        source_point_id: point.id.clone(),
        source: point.source.clone(),
        source_path: point.source_path.clone(),
        legacy_hash: point.legacy_hash.clone(),
        disposition: point.destination.label().to_string(),
        relative_path: record.relative_path(),
        anchor: record.relative_path().map(|_| record.anchor()),
        content_hash: point.content_hash.clone(),
    })
}

fn build_counts(
    raw: usize,
    classified: &[ClassifiedPoint],
    records: &[ExportRecord],
) -> ExportCounts {
    let mut counts = initial_counts(raw, classified, records);
    for record in records {
        count_record(&mut counts, record);
    }
    counts
}

fn initial_counts(
    raw: usize,
    classified: &[ClassifiedPoint],
    records: &[ExportRecord],
) -> ExportCounts {
    let exported_sources = classified
        .iter()
        .filter(|point| point.destination.is_exported())
        .count();
    let exported_records = records
        .iter()
        .filter(|record| record.destination.is_exported())
        .count();
    ExportCounts {
        raw,
        duplicates: exported_sources.saturating_sub(exported_records),
        quarantined_source_points: count_classified(
            classified,
            Some(KnowledgeCollection::MemoryUnits),
            Destination::Quarantine,
        ),
        excluded_kb_vectors: count_classified(classified, None, Destination::ExcludedKbVector),
        excluded_non_durable: count_classified(classified, None, Destination::ExcludedNonDurable),
        ..ExportCounts::default()
    }
}

fn count_classified(
    classified: &[ClassifiedPoint],
    collection: Option<KnowledgeCollection>,
    destination: Destination,
) -> usize {
    classified
        .iter()
        .filter(|point| {
            collection.is_none_or(|expected| point.collection == expected)
                && point.destination == destination
        })
        .count()
}

fn count_record(counts: &mut ExportCounts, record: &ExportRecord) {
    match record.destination {
        Destination::Quarantine => counts.quarantined += 1,
        Destination::Friction | Destination::NotableFacts | Destination::ManualMemories => {
            counts.exported_unique += 1;
            *counts
                .by_destination
                .entry(record.destination.label().to_string())
                .or_default() += 1;
        }
        Destination::ExcludedKbVector | Destination::ExcludedNonDurable => {}
    }
}

#[derive(Serialize)]
struct ExportManifest<'a> {
    counts: &'a ExportCounts,
    entries: &'a [ManifestEntry],
}

fn render_manifest(plan: &ExportPlan) -> Result<String> {
    let manifest = ExportManifest {
        counts: &plan.counts,
        entries: &plan.manifest,
    };
    let mut json =
        serde_json::to_string_pretty(&manifest).context("failed to serialize export manifest")?;
    json.push('\n');
    Ok(json)
}

fn verify_manifest(kb_root: &Path, plan: &ExportPlan) -> Result<()> {
    let path = kb_root.join("memory/export-manifest.json");
    let actual = fs::read_to_string(&path)?;
    let expected = render_manifest(plan)?;
    if actual != expected {
        bail!("export manifest differs from plan: {}", path.display());
    }
    Ok(())
}

fn verify_documents(kb_root: &Path, documents: &[ExportDocument]) -> Result<()> {
    for document in documents {
        let path = kb_root.join(&document.relative_path);
        if fs::read(&path)? != document.content.as_bytes() {
            bail!("exported document differs from plan: {}", path.display());
        }
    }
    Ok(())
}

fn preflight_writes(kb_root: &Path, writes: &[(&Path, &[u8])]) -> Result<()> {
    for (relative_path, content) in writes {
        validate_relative_path(relative_path)?;
        let path = kb_root.join(relative_path);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if !metadata.file_type().is_file() => {
                bail!("refusing non-file export target: {}", path.display())
            }
            Ok(_) if fs::read(&path)? != *content => {
                bail!(
                    "refusing to replace different existing KB document: {}",
                    path.display()
                )
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn write_files(kb_root: &Path, writes: &[(&Path, &[u8])]) -> Result<()> {
    for (relative_path, content) in writes {
        let path = kb_root.join(relative_path);
        if path.exists() {
            continue;
        }
        let parent = path.parent().context("export path has no parent")?;
        fs::create_dir_all(parent)?;
        fs::write(&path, content).with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<()> {
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!("unsafe export path: {}", path.display());
    }
    Ok(())
}

fn project_filename(project: &str) -> String {
    if project.is_empty() {
        return "__global__".to_string();
    }
    let encoded = project
        .as_bytes()
        .iter()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' => (*byte as char).to_string(),
            _ => format!("%{byte:02X}"),
        })
        .collect::<String>();
    format!("project-{encoded}")
}

fn display_project(project: &str) -> &str {
    if project.is_empty() {
        "<global>"
    } else {
        project
    }
}

fn encode_metadata(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('`', "&#96;")
        .replace('\r', "&#13;")
        .replace('\n', "&#10;")
}

fn encode_source_text(text: &str) -> String {
    let body = text
        .split('\n')
        .map(|line| {
            if line.is_empty() {
                ">".to_string()
            } else {
                format!("> {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("<!-- source-text -->\n{body}")
}

fn extract_record_text(document: &str, anchor: &str) -> Result<String> {
    let anchor_marker = format!("<a id=\"{anchor}\"></a>");
    let mut lines = document.split('\n');
    lines
        .find(|line| *line == anchor_marker)
        .context("anchor not found")?;
    lines
        .find(|line| *line == "<!-- source-text -->")
        .context("source text start not found")?;
    let encoded = lines
        .take_while(|line| *line != "<!-- end-source-text -->")
        .map(|line| {
            line.strip_prefix("> ")
                .or_else(|| (line == ">").then_some(""))
                .context("invalid source text blockquote")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(encoded.join("\n"))
}

fn hash_text(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}
