use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde_json::{Map, Value};

use crate::extract::HistoryType;
use crate::index::point_id_for_history_hash;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LegacyCollection {
    Memory,
    Answers,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LegacyHistoryRecord {
    pub collection: LegacyCollection,
    pub text: String,
    pub source: String,
    pub path: String,
    pub session_id: String,
    pub hash: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionHistoryRecord {
    pub history_type: HistoryType,
    pub text: String,
    pub source: String,
    pub path: String,
    pub session_id: String,
    pub hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MigrationError {
    UnsupportedHistorySource(String),
    MissingPayloadField(&'static str),
    InvalidPayload,
    InvalidPayloadField(&'static str),
}

impl fmt::Display for MigrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedHistorySource(source) => {
                write!(formatter, "unsupported history source: {source}")
            }
            Self::MissingPayloadField(field) => write!(formatter, "missing payload field: {field}"),
            Self::InvalidPayload => write!(formatter, "point payload is not an object"),
            Self::InvalidPayloadField(field) => write!(formatter, "invalid payload field: {field}"),
        }
    }
}

impl std::error::Error for MigrationError {}

impl TryFrom<LegacyHistoryRecord> for SessionHistoryRecord {
    type Error = MigrationError;

    fn try_from(record: LegacyHistoryRecord) -> Result<Self, Self::Error> {
        if !matches!(record.source.as_str(), "session" | "archive") {
            return Err(MigrationError::UnsupportedHistorySource(record.source));
        }
        let history_type = match record.collection {
            LegacyCollection::Memory => HistoryType::Prompt,
            LegacyCollection::Answers => HistoryType::Answer,
        };
        let hash = format!(
            "{}:{}:{}",
            history_type.as_str(),
            record.source,
            record.hash
        );
        Ok(Self {
            history_type,
            text: record.text,
            source: record.source,
            path: record.path,
            session_id: record.session_id,
            hash,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LegacyPoint {
    pub collection: LegacyCollection,
    pub vector: Value,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DestinationPoint {
    pub id: String,
    pub vector: Value,
    pub payload: Value,
}

pub fn build_destination_point(point: LegacyPoint) -> Result<DestinationPoint, MigrationError> {
    let payload = point
        .payload
        .as_object()
        .ok_or(MigrationError::InvalidPayload)?;
    let legacy = legacy_record_from_payload(point.collection, payload)?;
    let history = SessionHistoryRecord::try_from(legacy)?;
    let mut destination_payload = payload.clone();
    destination_payload.insert("type".to_string(), history.history_type.as_str().into());
    destination_payload.insert("hash".to_string(), history.hash.clone().into());
    destination_payload.insert("session_id".to_string(), history.session_id.clone().into());
    Ok(DestinationPoint {
        id: point_id_for_history_hash(&history.hash),
        vector: point.vector,
        payload: Value::Object(destination_payload),
    })
}

fn legacy_record_from_payload(
    collection: LegacyCollection,
    payload: &Map<String, Value>,
) -> Result<LegacyHistoryRecord, MigrationError> {
    Ok(LegacyHistoryRecord {
        collection,
        text: payload_string(payload, "text")?,
        source: payload_string(payload, "source")?,
        path: payload_string(payload, "path")?,
        session_id: optional_payload_string(payload, "session_id")?,
        hash: payload_string(payload, "hash")?,
    })
}

fn optional_payload_string(
    payload: &Map<String, Value>,
    field: &'static str,
) -> Result<String, MigrationError> {
    match payload.get(field) {
        None => Ok(String::new()),
        Some(value) => value
            .as_str()
            .map(str::to_string)
            .ok_or(MigrationError::InvalidPayloadField(field)),
    }
}

fn payload_string(
    payload: &Map<String, Value>,
    field: &'static str,
) -> Result<String, MigrationError> {
    let value = payload
        .get(field)
        .ok_or(MigrationError::MissingPayloadField(field))?;
    let value = value
        .as_str()
        .ok_or(MigrationError::InvalidPayloadField(field))?;
    if value.is_empty() {
        return Err(MigrationError::InvalidPayloadField(field));
    }
    Ok(value.to_string())
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoryClassification {
    pub raw_points: usize,
    pub eligible_points: usize,
    pub unique_points: usize,
    pub duplicate_points: usize,
    pub skipped_points: usize,
    pub grouped_unique: BTreeMap<(String, String), usize>,
    pub destination_points: Vec<DestinationPoint>,
}

pub fn classify_history_points(
    points: Vec<LegacyPoint>,
) -> Result<HistoryClassification, MigrationError> {
    let raw_points = points.len();
    let mut eligible_points = 0;
    let mut skipped_points = 0;
    let mut unique = BTreeMap::new();

    for point in points {
        match build_destination_point(point) {
            Ok(destination) => {
                eligible_points += 1;
                unique.entry(destination.id.clone()).or_insert(destination);
            }
            Err(MigrationError::UnsupportedHistorySource(_)) => skipped_points += 1,
            Err(error) => return Err(error),
        }
    }

    Ok(finish_classification(
        raw_points,
        eligible_points,
        skipped_points,
        unique,
    ))
}

fn finish_classification(
    raw_points: usize,
    eligible_points: usize,
    skipped_points: usize,
    unique: BTreeMap<String, DestinationPoint>,
) -> HistoryClassification {
    let unique_points = unique.len();
    let destination_points: Vec<_> = unique.into_values().collect();
    let grouped_unique = grouped_destination_counts(&destination_points);
    HistoryClassification {
        raw_points,
        eligible_points,
        unique_points,
        duplicate_points: eligible_points - unique_points,
        skipped_points,
        grouped_unique,
        destination_points,
    }
}

fn grouped_destination_counts(points: &[DestinationPoint]) -> BTreeMap<(String, String), usize> {
    let mut grouped = BTreeMap::new();
    for point in points {
        let payload = point
            .payload
            .as_object()
            .expect("validated destination payload");
        let history_type = payload_string(payload, "type").expect("validated history type");
        let source = payload_string(payload, "source").expect("validated history source");
        *grouped.entry((history_type, source)).or_insert(0) += 1;
    }
    grouped
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParityMismatch {
    pub missing: BTreeSet<(String, String, String)>,
    pub unexpected: BTreeSet<(String, String, String)>,
}

pub fn verify_history_parity(
    expected: &[SessionHistoryRecord],
    actual: &[SessionHistoryRecord],
) -> Result<(), ParityMismatch> {
    let expected_keys = history_keys(expected);
    let actual_keys = history_keys(actual);
    let missing: BTreeSet<_> = expected_keys.difference(&actual_keys).cloned().collect();
    let unexpected: BTreeSet<_> = actual_keys.difference(&expected_keys).cloned().collect();
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }
    Err(ParityMismatch {
        missing,
        unexpected,
    })
}

fn history_keys(records: &[SessionHistoryRecord]) -> BTreeSet<(String, String, String)> {
    records
        .iter()
        .map(|record| {
            (
                record.history_type.as_str().to_string(),
                record.source.clone(),
                record.hash.clone(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        LegacyCollection, LegacyHistoryRecord, LegacyPoint, MigrationError, SessionHistoryRecord,
        build_destination_point, classify_history_points, verify_history_parity,
    };
    use crate::extract::HistoryType;

    fn legacy(collection: LegacyCollection, source: &str, hash: &str) -> LegacyHistoryRecord {
        LegacyHistoryRecord {
            collection,
            text: format!("text-{hash}"),
            source: source.to_string(),
            path: "session.jsonl".to_string(),
            session_id: "session-1".to_string(),
            hash: hash.to_string(),
        }
    }

    #[test]
    fn prompt_and_answer_records_map_to_typed_history() {
        let prompt =
            SessionHistoryRecord::try_from(legacy(LegacyCollection::Memory, "session", "aaa"))
                .unwrap();
        let answer =
            SessionHistoryRecord::try_from(legacy(LegacyCollection::Answers, "archive", "aaa"))
                .unwrap();

        assert_eq!(prompt.history_type, HistoryType::Prompt);
        assert_eq!(prompt.source, "session");
        assert_eq!(prompt.hash, "prompt:session:aaa");
        assert_eq!(answer.history_type, HistoryType::Answer);
        assert_eq!(answer.source, "archive");
        assert_eq!(answer.hash, "answer:archive:aaa");
    }

    #[test]
    fn identical_type_and_text_from_different_sources_have_distinct_identity() {
        let session =
            SessionHistoryRecord::try_from(legacy(LegacyCollection::Memory, "session", "same"))
                .unwrap();
        let archive =
            SessionHistoryRecord::try_from(legacy(LegacyCollection::Memory, "archive", "same"))
                .unwrap();

        assert_ne!(session.hash, archive.hash);
    }

    #[test]
    fn non_session_sources_are_rejected() {
        for source in ["summary", "kb", "memory"] {
            let error =
                SessionHistoryRecord::try_from(legacy(LegacyCollection::Memory, source, source))
                    .unwrap_err();

            assert_eq!(
                error,
                MigrationError::UnsupportedHistorySource(source.to_string())
            );
        }
    }

    #[test]
    fn destination_point_preserves_vectors_and_extra_payload() {
        let vector = serde_json::json!({
            "dense": [0.25, 0.5],
            "bm25": {"indices": [1, 4], "values": [0.7, 0.3]}
        });
        let payload = serde_json::json!({
            "text": "same text",
            "source": "session",
            "path": "session.jsonl",
            "session_id": "session-1",
            "hash": "legacy-hash",
            "legacy_extra": "preserve"
        });
        let point = LegacyPoint {
            collection: LegacyCollection::Memory,
            vector: vector.clone(),
            payload,
        };

        let destination = build_destination_point(point).unwrap();

        assert_eq!(destination.vector, vector);
        assert_eq!(destination.payload["type"], "prompt");
        assert_eq!(destination.payload["hash"], "prompt:session:legacy-hash");
        assert_eq!(destination.payload["legacy_extra"], "preserve");
        assert_eq!(destination.id, "4af11796-0553-6475-8491-eb07de31a620");
    }

    #[test]
    fn dry_run_classification_accounts_for_deduplication_and_skips() {
        let points = vec![
            legacy_point(LegacyCollection::Memory, "session", "same"),
            legacy_point(LegacyCollection::Memory, "session", "same"),
            legacy_point(LegacyCollection::Memory, "archive", "same"),
            legacy_point(LegacyCollection::Memory, "kb", "skip"),
            legacy_point(LegacyCollection::Answers, "session", "answer"),
        ];

        let classification = classify_history_points(points).unwrap();

        assert_eq!(classification.raw_points, 5);
        assert_eq!(classification.eligible_points, 4);
        assert_eq!(classification.unique_points, 3);
        assert_eq!(classification.duplicate_points, 1);
        assert_eq!(classification.skipped_points, 1);
        assert_eq!(classification.destination_points.len(), 3);
        assert_eq!(
            classification
                .grouped_unique
                .get(&("prompt".to_string(), "session".to_string())),
            Some(&1)
        );
        assert_eq!(
            classification
                .grouped_unique
                .get(&("prompt".to_string(), "archive".to_string())),
            Some(&1)
        );
        assert_eq!(
            classification
                .grouped_unique
                .get(&("answer".to_string(), "session".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn missing_optional_session_id_normalizes_to_empty_string() {
        let mut point = legacy_point(LegacyCollection::Memory, "archive", "legacy");
        point.payload.as_object_mut().unwrap().remove("session_id");

        let destination = build_destination_point(point).unwrap();

        assert_eq!(destination.payload["session_id"], "");
    }

    #[test]
    fn non_string_optional_session_id_fails_fast() {
        let mut point = legacy_point(LegacyCollection::Memory, "session", "broken");
        point.payload["session_id"] = serde_json::json!(42);

        let error = build_destination_point(point).unwrap_err();

        assert_eq!(error, MigrationError::InvalidPayloadField("session_id"));
    }

    #[test]
    fn empty_required_field_fails_fast() {
        let mut point = legacy_point(LegacyCollection::Memory, "session", "broken");
        point.payload["hash"] = "".into();

        let error = build_destination_point(point).unwrap_err();

        assert_eq!(error, MigrationError::InvalidPayloadField("hash"));
    }

    #[test]
    fn malformed_eligible_point_fails_fast() {
        let mut point = legacy_point(LegacyCollection::Memory, "session", "broken");
        point.payload.as_object_mut().unwrap().remove("hash");

        let error = classify_history_points(vec![point]).unwrap_err();

        assert_eq!(error, MigrationError::MissingPayloadField("hash"));
    }

    fn legacy_point(collection: LegacyCollection, source: &str, hash: &str) -> LegacyPoint {
        LegacyPoint {
            collection,
            vector: serde_json::json!({"dense": [1.0], "bm25": {"indices": [1], "values": [1.0]}}),
            payload: serde_json::json!({
                "text": format!("text-{hash}"),
                "source": source,
                "path": "session.jsonl",
                "session_id": "session-1",
                "hash": hash
            }),
        }
    }

    #[test]
    fn parity_requires_exact_type_source_and_hash_set() {
        let expected = vec![
            SessionHistoryRecord::try_from(legacy(LegacyCollection::Memory, "session", "same"))
                .unwrap(),
            SessionHistoryRecord::try_from(legacy(LegacyCollection::Answers, "session", "same"))
                .unwrap(),
        ];
        let mut actual = expected.clone();
        actual[1].source = "archive".to_string();

        let mismatch = verify_history_parity(&expected, &actual).unwrap_err();

        assert_eq!(mismatch.missing.len(), 1);
        assert_eq!(mismatch.unexpected.len(), 1);
    }
}
