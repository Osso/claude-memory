use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KbPageIndex {
    pub source_dir: String,
    pub built_at: DateTime<Utc>,
    pub files: Vec<KbIndexedFile>,
    pub docs: Vec<KbIndexedDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KbIndexedFile {
    pub path: String,
    pub fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KbIndexedDoc {
    pub doc_id: String,
    pub doc_name: String,
    pub doc_description: Option<String>,
    pub source_path: String,
    pub line_count: usize,
    pub text: String,
    pub nodes: Vec<KbIndexedNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KbIndexedNode {
    pub node_id: String,
    pub title: String,
    pub heading_path: String,
    pub level: usize,
    pub source_line: usize,
    pub text: String,
    pub token_counts: HashMap<String, usize>,
    pub nodes: Vec<KbIndexedNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KbDocStructure {
    pub doc_id: String,
    pub doc_name: String,
    pub doc_description: Option<String>,
    pub source_path: String,
    pub line_count: usize,
    pub nodes: Vec<KbNodeStructure>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KbNodeStructure {
    pub node_id: String,
    pub title: String,
    pub heading_path: String,
    pub level: usize,
    pub source_line: usize,
    pub nodes: Vec<KbNodeStructure>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KbDocMetadata {
    pub doc_id: String,
    pub doc_name: String,
    pub doc_description: Option<String>,
    pub source_path: String,
    pub line_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KbDocContent {
    pub doc_id: String,
    pub source_path: String,
    pub locator: String,
    pub text: String,
}

impl KbIndexedDoc {
    pub fn metadata(&self) -> KbDocMetadata {
        KbDocMetadata {
            doc_id: self.doc_id.clone(),
            doc_name: self.doc_name.clone(),
            doc_description: self.doc_description.clone(),
            source_path: self.source_path.clone(),
            line_count: self.line_count,
        }
    }

    pub fn structure_without_text(&self) -> KbDocStructure {
        KbDocStructure {
            doc_id: self.doc_id.clone(),
            doc_name: self.doc_name.clone(),
            doc_description: self.doc_description.clone(),
            source_path: self.source_path.clone(),
            line_count: self.line_count,
            nodes: self
                .nodes
                .iter()
                .map(KbIndexedNode::structure_without_text)
                .collect(),
        }
    }
}

impl KbIndexedNode {
    pub fn structure_without_text(&self) -> KbNodeStructure {
        KbNodeStructure {
            node_id: self.node_id.clone(),
            title: self.title.clone(),
            heading_path: self.heading_path.clone(),
            level: self.level,
            source_line: self.source_line,
            nodes: self
                .nodes
                .iter()
                .map(KbIndexedNode::structure_without_text)
                .collect(),
        }
    }
}
