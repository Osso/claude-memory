//! Embedding clients for local Ollama and hosted OpenRouter models.

use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::config::{EmbeddingBackend, EmbeddingConfig, embedding_config};

const OLLAMA_URL: &str = "http://localhost:11434";
const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/embeddings";
const OPENROUTER_MAX_ATTEMPTS: usize = 3;
const RETRY_BASE_DELAY_MS: u64 = 250;
const RETRY_JITTER_MS: u64 = 100;
/// Timeout for embedding requests (allows for cold model loading).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct OpenRouterEmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    dimensions: u64,
    encoding_format: &'static str,
    input_type: &'static str,
    provider: OpenRouterProviderPreferences,
}

#[derive(Serialize)]
struct OpenRouterProviderPreferences {
    zdr: bool,
}

#[derive(Deserialize)]
struct OpenRouterCredentials {
    api_key: Option<String>,
}

#[derive(Deserialize)]
struct OpenRouterEmbedResponse {
    data: Vec<OpenRouterEmbedding>,
}

#[derive(Deserialize)]
struct OpenRouterEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

fn read_openrouter_api_key(path: &Path) -> Result<String> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read OpenRouter config at {}", path.display()))?;
    let credentials: OpenRouterCredentials = toml::from_str(&contents)
        .with_context(|| format!("failed to parse OpenRouter config at {}", path.display()))?;
    let api_key = credentials
        .api_key
        .context("OpenRouter config missing required `api_key`")?;
    if api_key.trim().is_empty() {
        anyhow::bail!("OpenRouter config `api_key` must not be blank");
    }
    Ok(api_key)
}

pub struct Embedder {
    client: Client,
    config: EmbeddingConfig,
    url: String,
    api_key: Option<String>,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let config = embedding_config()?;
        let openrouter_config_path = match config.backend {
            EmbeddingBackend::Ollama => PathBuf::new(),
            EmbeddingBackend::OpenRouter => default_openrouter_config_path()?,
        };
        Self::from_runtime_config(config, &openrouter_config_path)
    }

    fn from_runtime_config(config: EmbeddingConfig, openrouter_config_path: &Path) -> Result<Self> {
        match config.backend {
            EmbeddingBackend::Ollama => Ok(Self::from_config(config, OLLAMA_URL.to_string(), None)),
            EmbeddingBackend::OpenRouter => {
                let api_key = read_openrouter_api_key(openrouter_config_path)?;
                Ok(Self::from_config(
                    config,
                    OPENROUTER_URL.to_string(),
                    Some(api_key),
                ))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn with_url(url: String) -> Self {
        Self::from_config(EmbeddingConfig::default(), url, None)
    }

    #[cfg(test)]
    fn with_config(config: EmbeddingConfig, url: String, api_key: Option<String>) -> Self {
        Self::from_config(config, url, api_key)
    }

    fn from_config(config: EmbeddingConfig, url: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .pool_idle_timeout(Duration::from_secs(30))
                .pool_max_idle_per_host(1)
                .build()
                .expect("failed to build HTTP client"),
            config,
            url,
            api_key,
        }
    }

    /// Embed one search query.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self.config.backend {
            EmbeddingBackend::Ollama => self.embed_ollama(text).await,
            EmbeddingBackend::OpenRouter => {
                let query = self.format_query(text);
                let mut embeddings = self
                    .embed_openrouter(&[query.as_str()], "search_query")
                    .await?;
                Ok(embeddings.remove(0))
            }
        }
    }

    /// Embed search documents in one provider batch when supported.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match self.config.backend {
            EmbeddingBackend::Ollama => self.embed_ollama_batch(texts).await,
            EmbeddingBackend::OpenRouter => self.embed_openrouter(texts, "search_document").await,
        }
    }

    async fn embed_ollama(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post(format!("{}/api/embeddings", self.url))
            .json(&OllamaEmbedRequest {
                model: &self.config.model,
                prompt: text,
            })
            .send()
            .await
            .context("failed to connect to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama error {status}: {body}");
        }

        let response: OllamaEmbedResponse = response
            .json()
            .await
            .context("failed to parse Ollama embedding")?;
        self.validate_embedding_dimension(&response.embedding)?;
        Ok(response.embedding)
    }

    async fn embed_ollama_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed_ollama(text).await?);
        }
        Ok(embeddings)
    }

    async fn embed_openrouter(
        &self,
        texts: &[&str],
        input_type: &'static str,
    ) -> Result<Vec<Vec<f32>>> {
        let api_key = self
            .api_key
            .as_deref()
            .context("OpenRouter API key is not configured")?;
        let request = OpenRouterEmbedRequest {
            model: &self.config.model,
            input: texts,
            dimensions: self.config.vector_size,
            encoding_format: "float",
            input_type,
            provider: OpenRouterProviderPreferences { zdr: true },
        };
        let response = self.send_openrouter_request(api_key, &request).await?;
        let response: OpenRouterEmbedResponse = response
            .json()
            .await
            .context("failed to parse OpenRouter embedding response")?;
        self.order_openrouter_embeddings(response.data, texts.len())
    }

    async fn send_openrouter_request(
        &self,
        api_key: &str,
        request: &OpenRouterEmbedRequest<'_>,
    ) -> Result<Response> {
        for attempt in 0..OPENROUTER_MAX_ATTEMPTS {
            let response = self
                .client
                .post(&self.url)
                .bearer_auth(api_key)
                .json(request)
                .send()
                .await;
            match response {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) if should_retry_status(response.status(), attempt) => {
                    let delay = retry_delay(response.headers(), attempt);
                    tokio::time::sleep(delay).await;
                }
                Ok(response) => return Err(openrouter_response_error(response).await),
                Err(error) if attempt + 1 < OPENROUTER_MAX_ATTEMPTS => {
                    tokio::time::sleep(exponential_retry_delay(attempt)).await;
                    tracing::debug!("retrying OpenRouter embedding request after error: {error}");
                }
                Err(error) => return Err(error).context("failed to connect to OpenRouter"),
            }
        }
        unreachable!("OpenRouter retry loop returns on its final attempt")
    }

    fn order_openrouter_embeddings(
        &self,
        mut embeddings: Vec<OpenRouterEmbedding>,
        expected_count: usize,
    ) -> Result<Vec<Vec<f32>>> {
        embeddings.sort_by_key(|embedding| embedding.index);
        if embeddings.len() != expected_count {
            anyhow::bail!(
                "OpenRouter returned {} embeddings for {expected_count} inputs",
                embeddings.len()
            );
        }
        embeddings
            .into_iter()
            .enumerate()
            .map(|(expected_index, embedding)| {
                if embedding.index != expected_index {
                    anyhow::bail!(
                        "OpenRouter embedding index is {}, expected {expected_index}",
                        embedding.index
                    );
                }
                self.validate_embedding_dimension(&embedding.embedding)?;
                Ok(embedding.embedding)
            })
            .collect()
    }

    fn format_query(&self, query: &str) -> String {
        match &self.config.query_instruction {
            Some(instruction) => format!("Instruct: {instruction}\nQuery:{query}"),
            None => query.to_string(),
        }
    }

    fn validate_embedding_dimension(&self, embedding: &[f32]) -> Result<()> {
        let actual_size = embedding.len() as u64;
        if actual_size != self.config.vector_size {
            anyhow::bail!(
                "embedding dimension is {actual_size}, expected {}",
                self.config.vector_size
            );
        }
        Ok(())
    }
}

fn default_openrouter_config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("no home directory for OpenRouter config")?;
    Ok(home.join(".config/openrouter/config.toml"))
}

fn should_retry_status(status: StatusCode, attempt: usize) -> bool {
    let has_attempt_remaining = attempt + 1 < OPENROUTER_MAX_ATTEMPTS;
    has_attempt_remaining
        && matches!(
            status.as_u16(),
            408 | 429 | 500 | 502 | 503 | 504 | 524 | 529
        )
}

fn retry_delay(headers: &HeaderMap, attempt: usize) -> Duration {
    retry_after_delay(headers).unwrap_or_else(|| exponential_retry_delay(attempt))
}

fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    retry_at
        .signed_duration_since(chrono::Utc::now())
        .to_std()
        .ok()
}

fn exponential_retry_delay(attempt: usize) -> Duration {
    let multiplier = 1_u64 << attempt.min(8);
    let base_delay = RETRY_BASE_DELAY_MS.saturating_mul(multiplier);
    let jitter = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64
        % (RETRY_JITTER_MS + 1);
    Duration::from_millis(base_delay + jitter)
}

async fn openrouter_response_error(response: Response) -> anyhow::Error {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    anyhow::anyhow!("OpenRouter error {status}: {body}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EmbeddingBackend, EmbeddingConfig};
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::oneshot;
    use tokio::task::JoinHandle;

    struct TestResponse {
        status: &'static str,
        headers: Vec<(&'static str, &'static str)>,
        body: Value,
    }

    struct CapturedRequest {
        headers: String,
        body: Value,
    }

    async fn start_test_server(
        responses: Vec<TestResponse>,
    ) -> (
        String,
        oneshot::Receiver<Vec<CapturedRequest>>,
        JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (requests_tx, requests_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let mut requests = Vec::new();
            for response in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                requests.push(read_request(&mut stream).await);
                write_response(&mut stream, response).await;
            }
            requests_tx.send(requests).ok();
        });
        (url, requests_rx, server)
    }

    async fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            let read = stream.read(&mut chunk).await.unwrap();
            request.extend_from_slice(&chunk[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let body_start = header_end + 4;
            let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(str::to_string)
                })
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() < body_start + content_length {
                continue;
            }
            let body =
                serde_json::from_slice(&request[body_start..body_start + content_length]).unwrap();
            return CapturedRequest { headers, body };
        }
    }

    async fn write_response(stream: &mut TcpStream, response: TestResponse) {
        let body = response.body.to_string();
        let extra_headers = response
            .headers
            .iter()
            .map(|(name, value)| format!("{name}: {value}\r\n"))
            .collect::<String>();
        let response = format!(
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
            response.status,
            body.len(),
            extra_headers,
            body,
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
    }

    fn openrouter_config(vector_size: u64) -> EmbeddingConfig {
        EmbeddingConfig {
            backend: EmbeddingBackend::OpenRouter,
            model: "qwen/qwen3-embedding-8b".to_string(),
            vector_size,
            collection: "test-openrouter".to_string(),
            query_instruction: Some("Retrieve relevant conversation passages".to_string()),
        }
    }

    #[test]
    fn default_openrouter_config_path_uses_standard_location() {
        let path = default_openrouter_config_path().unwrap();

        assert!(path.ends_with(".config/openrouter/config.toml"));
    }

    #[test]
    fn openrouter_runtime_config_fails_when_credential_file_is_missing() {
        let path =
            std::env::temp_dir().join(format!("missing-openrouter-{}.toml", uuid::Uuid::new_v4()));

        let error = match Embedder::from_runtime_config(openrouter_config(4096), &path) {
            Ok(_) => panic!("missing OpenRouter config should fail"),
            Err(error) => error,
        };

        assert!(format!("{error:#}").contains(path.to_str().unwrap()));
    }

    #[test]
    fn openrouter_api_key_is_read_from_standard_config_shape() {
        let path = std::env::temp_dir().join(format!(
            "claude-memory-openrouter-{}.toml",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &path,
            "api_key = \"test-inference-key\"\nmgmt_key = \"unused-management-key\"\n",
        )
        .unwrap();

        let api_key = read_openrouter_api_key(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(api_key, "test-inference-key");
    }

    #[tokio::test]
    async fn openrouter_batch_sends_one_document_request_and_restores_response_order() {
        let responses = vec![TestResponse {
            status: "200 OK",
            headers: Vec::new(),
            body: serde_json::json!({
                "data": [
                    {"index": 1, "embedding": [2.0, 0.0, 0.0, 0.0]},
                    {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}
                ]
            }),
        }];
        let (url, requests, server) = start_test_server(responses).await;
        let embedder = Embedder::with_config(
            openrouter_config(4),
            url,
            Some("test-openrouter-key".to_string()),
        );

        let embeddings = embedder
            .embed_batch(&["first document", "second document"])
            .await
            .unwrap();
        let requests = requests.await.unwrap();
        server.await.unwrap();

        assert_eq!(embeddings[0][0], 1.0);
        assert_eq!(embeddings[1][0], 2.0);
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .headers
                .contains("authorization: Bearer test-openrouter-key")
                || requests[0]
                    .headers
                    .contains("Authorization: Bearer test-openrouter-key")
        );
        assert_eq!(requests[0].body["model"], "qwen/qwen3-embedding-8b");
        assert_eq!(
            requests[0].body["input"],
            serde_json::json!(["first document", "second document"])
        );
        assert_eq!(requests[0].body["dimensions"], 4);
        assert_eq!(requests[0].body["input_type"], "search_document");
        assert_eq!(requests[0].body["provider"]["zdr"], true);
    }

    #[tokio::test]
    async fn openrouter_query_applies_instruction_without_modifying_documents() {
        let responses = vec![TestResponse {
            status: "200 OK",
            headers: Vec::new(),
            body: serde_json::json!({
                "data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]
            }),
        }];
        let (url, requests, server) = start_test_server(responses).await;
        let embedder = Embedder::with_config(
            openrouter_config(4),
            url,
            Some("test-openrouter-key".to_string()),
        );

        embedder.embed("find auth backup race").await.unwrap();
        let requests = requests.await.unwrap();
        server.await.unwrap();

        assert_eq!(
            requests[0].body["input"],
            serde_json::json!([
                "Instruct: Retrieve relevant conversation passages\nQuery:find auth backup race"
            ])
        );
        assert_eq!(requests[0].body["input_type"], "search_query");
    }

    #[test]
    fn retry_after_http_date_uses_future_delay() {
        let retry_at = chrono::Utc::now() + chrono::Duration::seconds(2);
        let mut headers = HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            retry_at
                .to_rfc2822()
                .parse()
                .expect("valid Retry-After header"),
        );

        let delay = retry_after_delay(&headers).unwrap();

        assert!(delay <= Duration::from_secs(2));
        assert!(delay >= Duration::from_millis(500));
    }

    #[tokio::test]
    async fn openrouter_retries_rate_limits_using_retry_after() {
        let responses = vec![
            TestResponse {
                status: "429 Too Many Requests",
                headers: vec![("Retry-After", "0")],
                body: serde_json::json!({"error": {"message": "slow down"}}),
            },
            TestResponse {
                status: "200 OK",
                headers: Vec::new(),
                body: serde_json::json!({
                    "data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]
                }),
            },
        ];
        let (url, requests, server) = start_test_server(responses).await;
        let embedder = Embedder::with_config(
            openrouter_config(4),
            url,
            Some("test-openrouter-key".to_string()),
        );

        let result = embedder.embed_batch(&["retry this document"]).await;
        if result.is_err() {
            server.abort();
        }
        let embeddings = result.unwrap();
        let requests = requests.await.unwrap();
        server.await.unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(requests.len(), 2);
    }
}
