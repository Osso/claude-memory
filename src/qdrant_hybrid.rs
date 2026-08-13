//! Shared helpers for hybrid (dense + BM25 sparse) Qdrant collection management.

use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, Document, GetCollectionInfoResponse,
    SparseVectorParamsBuilder, SparseVectorsConfigBuilder, Vector, VectorParamsBuilder,
    VectorsConfigBuilder, vectors_config,
};
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
    create_hybrid_collection_with_vector_size(client, name, VECTOR_SIZE).await
}

async fn create_hybrid_collection_with_vector_size(
    client: &Qdrant,
    name: &str,
    vector_size: u64,
) -> Result<()> {
    let mut vectors_config = VectorsConfigBuilder::default();
    vectors_config.add_named_vector_params(
        "dense",
        VectorParamsBuilder::new(vector_size, Distance::Cosine),
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

/// Ensure a collection exists with the default hybrid vector size.
pub async fn ensure_hybrid_collection(client: &Qdrant, name: &str) -> Result<()> {
    ensure_hybrid_collection_with_vector_size(client, name, VECTOR_SIZE).await
}

pub async fn ensure_hybrid_collection_with_vector_size(
    client: &Qdrant,
    name: &str,
    vector_size: u64,
) -> Result<()> {
    let collections = client.list_collections().await?;
    let exists = collections
        .collections
        .iter()
        .any(|collection| collection.name == name);
    if !exists {
        return create_hybrid_collection_with_vector_size(client, name, vector_size).await;
    }

    let info = client.collection_info(name).await?;
    validate_dense_vector_size(&info, name, vector_size)?;
    if !collection_has_sparse(client, name).await? {
        client.delete_collection(name).await?;
        create_hybrid_collection_with_vector_size(client, name, vector_size).await?;
    }
    Ok(())
}

fn validate_dense_vector_size(
    info: &GetCollectionInfoResponse,
    collection: &str,
    expected_size: u64,
) -> Result<()> {
    let actual_size = dense_vector_size(info)
        .with_context(|| format!("collection `{collection}` has no named `dense` vector"))?;
    if actual_size != expected_size {
        anyhow::bail!(
            "collection `{collection}` dense vector size is {actual_size}, expected {expected_size}"
        );
    }
    Ok(())
}

fn dense_vector_size(info: &GetCollectionInfoResponse) -> Option<u64> {
    let config = info
        .result
        .as_ref()?
        .config
        .as_ref()?
        .params
        .as_ref()?
        .vectors_config
        .as_ref()?
        .config
        .as_ref()?;
    match config {
        vectors_config::Config::Params(params) => Some(params.size),
        vectors_config::Config::ParamsMap(params) => {
            params.map.get("dense").map(|dense| dense.size)
        }
    }
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
