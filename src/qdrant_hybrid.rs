//! Shared helpers for hybrid (dense + BM25 sparse) Qdrant collection management.

use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, Document, SparseVectorParamsBuilder,
    SparseVectorsConfigBuilder, VectorParamsBuilder, Vector, VectorsConfigBuilder,
};
use qdrant_client::Qdrant;
use std::collections::HashMap;

pub const VECTOR_SIZE: u64 = 1024;
pub const BM25_MODEL: &str = "Qdrant/bm25";

/// Check whether an existing collection already has sparse vectors configured.
pub async fn collection_has_sparse(client: &Qdrant, name: &str) -> Result<bool> {
    let info = client.collection_info(name).await?;
    Ok(info
        .result
        .as_ref()
        .and_then(|r| r.config.as_ref())
        .and_then(|c| c.params.as_ref())
        .map(|p| p.sparse_vectors_config.is_some())
        .unwrap_or(false))
}

/// Create a new hybrid collection with named dense + BM25 sparse vectors.
pub async fn create_hybrid_collection(client: &Qdrant, name: &str) -> Result<()> {
    let mut vectors_config = VectorsConfigBuilder::default();
    vectors_config.add_named_vector_params(
        "dense",
        VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine),
    );
    let mut sparse_config = SparseVectorsConfigBuilder::default();
    sparse_config.add_named_vector_params("bm25", SparseVectorParamsBuilder::default());
    client
        .create_collection(
            CreateCollectionBuilder::new(name)
                .vectors_config(vectors_config)
                .sparse_vectors_config(sparse_config),
        )
        .await
        .context("failed to create hybrid collection")?;
    Ok(())
}

/// Ensure a collection exists with the hybrid format, migrating (delete+recreate) if needed.
pub async fn ensure_hybrid_collection(client: &Qdrant, name: &str) -> Result<()> {
    let collections = client.list_collections().await?;
    let exists = collections.collections.iter().any(|c| c.name == name);
    if exists {
        if !collection_has_sparse(client, name).await? {
            client.delete_collection(name).await?;
            create_hybrid_collection(client, name).await?;
        }
        return Ok(());
    }
    create_hybrid_collection(client, name).await
}

/// Build named vectors map: dense (pre-computed embedding) + BM25 (server-side tokenization).
pub fn build_named_vectors(embedding: Vec<f32>, text: &str) -> HashMap<String, Vector> {
    let mut named = HashMap::new();
    named.insert("dense".to_string(), Vector::from(embedding));
    named.insert(
        "bm25".to_string(),
        Vector::from(Document::new(text, BM25_MODEL)),
    );
    named
}
