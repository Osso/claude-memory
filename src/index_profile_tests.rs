use std::collections::{HashMap, HashSet};

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::{IndexState, QDRANT_URL, index_new_chunks};
use crate::chunk::Chunk;
use crate::embed::Embedder;
use crate::extract::{HistoryType, IndexedChunk};
use crate::qdrant_hybrid::{
    build_named_vectors, ensure_hybrid_collection, ensure_hybrid_collection_with_vector_size,
};

fn make_chunk(hash: &str) -> IndexedChunk {
    IndexedChunk {
        chunk: Chunk {
            text: format!("text for {hash}"),
            hash: hash.to_string(),
        },
        history_type: HistoryType::Prompt,
        source: "session".to_string(),
        path: "/some/path".to_string(),
        session_id: None,
    }
}

async fn start_embedding_server(status: &'static str) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_request_headers(&mut stream).await;
        let body = serde_json::json!({"embedding": vec![1.0_f32; 1024]}).to_string();
        write_response(&mut stream, status, &body).await;
    });
    (url, server)
}

async fn read_request_headers(stream: &mut TcpStream) {
    let mut request = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let read = stream.read(&mut chunk).await.unwrap();
        if read == 0 {
            return;
        }
        request.extend_from_slice(&chunk[..read]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            return;
        }
    }
}

async fn write_response(stream: &mut TcpStream, status: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    stream.write_all(response.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
}

async fn collection_point_count(client: &Qdrant, collection: &str) -> u64 {
    client
        .collection_info(collection)
        .await
        .unwrap()
        .result
        .and_then(|result| result.points_count)
        .unwrap_or(0)
}

#[tokio::test]
async fn indexing_writes_only_the_configured_collection() {
    let selected = format!("test-selected-index-{}", uuid::Uuid::new_v4());
    let untouched = format!("test-untouched-index-{}", uuid::Uuid::new_v4());
    let client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    ensure_hybrid_collection(&client, &selected).await.unwrap();
    ensure_hybrid_collection(&client, &untouched).await.unwrap();
    let (url, server) = start_embedding_server("200 OK").await;
    let state = IndexState {
        client,
        embedder: Embedder::with_url(url),
        hashes: Mutex::new(HashSet::new()),
        batch_size: 1,
        delay_ms: 0,
        collection: selected.clone(),
    };

    let indexed = index_new_chunks(&state, &[make_chunk("selected-collection")])
        .await
        .unwrap();
    server.await.unwrap();
    let selected_count = collection_point_count(&state.client, &selected).await;
    let untouched_count = collection_point_count(&state.client, &untouched).await;
    state.client.delete_collection(&selected).await.unwrap();
    state.client.delete_collection(&untouched).await.unwrap();

    assert_eq!(indexed, 1);
    assert_eq!(selected_count, 1);
    assert_eq!(untouched_count, 0);
}

#[tokio::test]
async fn indexing_stops_when_an_embedding_batch_fails() {
    let collection = format!("test-failed-index-{}", uuid::Uuid::new_v4());
    let client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    ensure_hybrid_collection(&client, &collection)
        .await
        .unwrap();
    let (url, server) = start_embedding_server("500 Internal Server Error").await;
    let state = IndexState {
        client,
        embedder: Embedder::with_url(url),
        hashes: Mutex::new(HashSet::new()),
        batch_size: 1,
        delay_ms: 0,
        collection: collection.clone(),
    };

    let error = index_new_chunks(&state, &[make_chunk("failed-embedding")])
        .await
        .unwrap_err();
    server.await.unwrap();
    let points = collection_point_count(&state.client, &collection).await;
    state.client.delete_collection(&collection).await.unwrap();

    assert!(format!("{error:#}").contains("Ollama error"));
    assert_eq!(points, 0);
}

#[tokio::test]
async fn qdrant_collection_uses_requested_dense_dimension() {
    let collection = format!("test-vector-size-{}", uuid::Uuid::new_v4());
    let client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    ensure_hybrid_collection_with_vector_size(&client, &collection, 4)
        .await
        .unwrap();
    let point = PointStruct::new(
        1,
        build_named_vectors(vec![1.0; 4], "four-dimensional point"),
        HashMap::<String, Value>::new(),
    );

    client
        .upsert_points(UpsertPointsBuilder::new(&collection, vec![point]))
        .await
        .unwrap();
    client.delete_collection(&collection).await.unwrap();
}

#[tokio::test]
async fn qdrant_collection_rejects_dense_dimension_mismatch_without_recreating() {
    let collection = format!("test-vector-mismatch-{}", uuid::Uuid::new_v4());
    let client = Qdrant::from_url(QDRANT_URL).build().unwrap();
    ensure_hybrid_collection_with_vector_size(&client, &collection, 4)
        .await
        .unwrap();

    let error = ensure_hybrid_collection_with_vector_size(&client, &collection, 8)
        .await
        .unwrap_err();
    let point = PointStruct::new(
        1,
        build_named_vectors(vec![1.0; 4], "original collection remains"),
        HashMap::<String, Value>::new(),
    );
    client
        .upsert_points(UpsertPointsBuilder::new(&collection, vec![point]))
        .await
        .unwrap();
    let points = collection_point_count(&client, &collection).await;
    client.delete_collection(&collection).await.unwrap();

    assert!(error.to_string().contains("dense vector size"));
    assert_eq!(points, 1);
}
