//! Ollama embedding client.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

const OLLAMA_URL: &str = "http://localhost:11434";
const MODEL: &str = "qwen3-embedding:0.6b-ctx2048";
/// Timeout for embedding requests (allows for cold model loading)
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

pub struct Embedder {
    client: Client,
    url: String,
}

impl Embedder {
    pub fn new() -> Self {
        Self::from_url(OLLAMA_URL.to_string())
    }

    #[cfg(test)]
    pub(crate) fn with_url(url: String) -> Self {
        Self::from_url(url)
    }

    fn from_url(url: String) -> Self {
        Self {
            client: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .pool_idle_timeout(Duration::from_secs(30))
                .pool_max_idle_per_host(1)
                .build()
                .expect("failed to build HTTP client"),
            url,
        }
    }

    /// Embed a single text.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let resp = self
            .client
            .post(format!("{}/api/embeddings", self.url))
            .json(&EmbedRequest {
                model: MODEL,
                prompt: text,
            })
            .send()
            .await
            .context("failed to connect to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama error {}: {}", status, body);
        }

        let resp: EmbedResponse = resp.json().await.context("failed to parse embedding")?;
        Ok(resp.embedding)
    }

    /// Embed multiple texts sequentially.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new()
    }
}
