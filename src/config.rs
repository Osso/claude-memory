use anyhow::{Result, bail};
use std::sync::OnceLock;

const EMBEDDING_BACKEND_ENV: &str = "CLAUDE_MEMORY_EMBEDDING_BACKEND";
const EMBEDDING_MODEL_ENV: &str = "CLAUDE_MEMORY_EMBEDDING_MODEL";
const VECTOR_SIZE_ENV: &str = "CLAUDE_MEMORY_VECTOR_SIZE";
const COLLECTION_ENV: &str = "CLAUDE_MEMORY_COLLECTION";
const QUERY_INSTRUCTION_ENV: &str = "CLAUDE_MEMORY_QUERY_INSTRUCTION";

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBackend {
    Ollama,
    OpenRouter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub backend: EmbeddingBackend,
    pub model: String,
    pub vector_size: u64,
    pub collection: String,
    pub query_instruction: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: EmbeddingBackend::Ollama,
            model: "qwen3-embedding:0.6b-ctx2048".to_string(),
            vector_size: 1024,
            collection: "claude-session-history".to_string(),
            query_instruction: None,
        }
    }
}

pub fn embedding_config() -> Result<EmbeddingConfig> {
    resolve_embedding_config_for_config(load(), |key| std::env::var(key).ok())
}

pub fn resolve_embedding_config<F>(get: F) -> Result<EmbeddingConfig>
where
    F: Fn(&str) -> Option<String>,
{
    resolve_embedding_config_for_config(&Config::default(), get)
}

fn resolve_embedding_config_for_config<F>(config: &Config, get: F) -> Result<EmbeddingConfig>
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(error) = &config.embedding_error {
        bail!("{error}");
    }

    let mut embedding = config.embedding.clone().unwrap_or_default();

    if let Some(value) = get(EMBEDDING_BACKEND_ENV) {
        embedding.backend = parse_backend(&value, EMBEDDING_BACKEND_ENV)?;
    }
    if let Some(value) = get(EMBEDDING_MODEL_ENV) {
        validate_nonblank(EMBEDDING_MODEL_ENV, &value)?;
        embedding.model = value;
    }
    if let Some(value) = get(VECTOR_SIZE_ENV) {
        embedding.vector_size = parse_vector_size(&value, VECTOR_SIZE_ENV)?;
    }
    if let Some(value) = get(COLLECTION_ENV) {
        validate_nonblank(COLLECTION_ENV, &value)?;
        embedding.collection = value;
    }
    if let Some(value) = get(QUERY_INSTRUCTION_ENV) {
        validate_nonblank(QUERY_INSTRUCTION_ENV, &value)?;
        embedding.query_instruction = Some(value);
    }

    Ok(embedding)
}

fn parse_embedding_config(table: &toml::Table) -> Result<Option<EmbeddingConfig>> {
    let Some(value) = table.get("embedding") else {
        return Ok(None);
    };
    let section = value
        .as_table()
        .ok_or_else(|| anyhow::anyhow!("invalid [embedding]: expected a table"))?;

    let backend_value = required_file_string(section, "backend")?;
    let model = required_file_string(section, "model")?;
    let vector_size = required_file_vector_size(section, "vector_size")?;
    let collection = required_file_string(section, "collection")?;
    let query_instruction = optional_file_string(section, "query_instruction")?;

    Ok(Some(EmbeddingConfig {
        backend: parse_backend(&backend_value, "embedding.backend")?,
        model,
        vector_size,
        collection,
        query_instruction,
    }))
}

fn required_file_string(table: &toml::Table, key: &str) -> Result<String> {
    let value = table
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("invalid embedding.{key}: value is required"))?;
    parse_file_string(value, &format!("embedding.{key}"))
}

fn optional_file_string(table: &toml::Table, key: &str) -> Result<Option<String>> {
    table
        .get(key)
        .map(|value| parse_file_string(value, &format!("embedding.{key}")))
        .transpose()
}

fn parse_file_string(value: &toml::Value, key: &str) -> Result<String> {
    let value = value
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("invalid {key}: expected a string"))?;
    validate_nonblank(key, value)?;
    Ok(value.to_string())
}

fn required_file_vector_size(table: &toml::Table, key: &str) -> Result<u64> {
    let value = table
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("invalid embedding.{key}: value is required"))?;
    let value = value
        .as_integer()
        .ok_or_else(|| anyhow::anyhow!("invalid embedding.{key}: expected a positive integer"))?;
    if value <= 0 {
        bail!("invalid embedding.{key}: expected a positive integer");
    }
    Ok(value as u64)
}

fn parse_backend(value: &str, key: &str) -> Result<EmbeddingBackend> {
    match value {
        "ollama" => Ok(EmbeddingBackend::Ollama),
        "openrouter" => Ok(EmbeddingBackend::OpenRouter),
        _ => bail!("invalid {key}: expected `ollama` or `openrouter`, got `{value}`"),
    }
}

fn parse_vector_size(value: &str, key: &str) -> Result<u64> {
    let vector_size = value
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("invalid {key}: expected a positive integer"))?;
    if vector_size == 0 {
        bail!("invalid {key}: expected a positive integer");
    }
    Ok(vector_size)
}

fn validate_nonblank(key: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("invalid {key}: value must not be blank");
    }
    Ok(())
}

#[derive(Debug)]
pub struct Config {
    pub search: SearchConfig,
    embedding: Option<EmbeddingConfig>,
    embedding_error: Option<String>,
}

#[derive(Debug)]
pub struct SearchConfig {
    pub enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            search: SearchConfig { enabled: false },
            embedding: None,
            embedding_error: None,
        }
    }
}

pub fn load() -> &'static Config {
    CONFIG.get_or_init(load_inner)
}

pub fn search_enabled() -> bool {
    load().search.enabled
}

fn load_inner() -> Config {
    let Some(config_dir) = dirs::config_dir() else {
        return Config::default();
    };
    let path = config_dir.join("claude-memory/config.toml");
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Config::default(),
        Err(e) => {
            log_warn(&format!("config read error {}: {e}", path.display()));
            return Config::default();
        }
    };
    parse_config(&raw)
}

fn parse_config(raw: &str) -> Config {
    match toml::from_str::<toml::Table>(raw) {
        Ok(table) => match parse_embedding_config(&table) {
            Ok(embedding) => Config {
                search: SearchConfig {
                    enabled: table_enabled(&table, "search"),
                },
                embedding,
                embedding_error: None,
            },
            Err(error) => Config {
                search: SearchConfig {
                    enabled: table_enabled(&table, "search"),
                },
                embedding: None,
                embedding_error: Some(format!("{error:#}")),
            },
        },
        Err(e) => {
            log_warn(&format!("config parse error: {e}"));
            Config::default()
        }
    }
}

fn table_enabled(table: &toml::Table, section: &str) -> bool {
    table
        .get(section)
        .and_then(|v| v.get("enabled"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

pub(crate) const LOG_PATH: &str = "/tmp/claude/claude-memory.log";

fn log_warn(msg: &str) {
    use std::io::Write;
    let log_path = LOG_PATH;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let _ = writeln!(f, "[claude-memory config] WARN: {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_search_disabled() {
        let cfg = Config::default();
        assert!(!cfg.search.enabled);
    }

    #[test]
    fn parse_search_enabled_true() {
        let cfg = parse_config("[search]\nenabled = true");
        assert!(cfg.search.enabled);
    }

    #[test]
    fn parse_malformed_toml_returns_default() {
        let cfg = parse_config("not = [ valid toml !!!@#");
        assert!(!cfg.search.enabled);
    }

    #[test]
    fn embedding_config_uses_current_defaults() {
        let cfg = resolve_embedding_config(|_| None).unwrap();

        assert_eq!(cfg.backend, EmbeddingBackend::Ollama);
        assert_eq!(cfg.model, "qwen3-embedding:0.6b-ctx2048");
        assert_eq!(cfg.vector_size, 1024);
        assert_eq!(cfg.collection, "claude-session-history");
        assert_eq!(cfg.query_instruction, None);
    }

    #[test]
    fn embedding_config_uses_file_profile() {
        let config = parse_config(
            "[embedding]\nbackend = \"openrouter\"\nmodel = \"qwen/qwen3-embedding-8b\"\nvector_size = 4096\ncollection = \"claude-session-history-8b\"\nquery_instruction = \"Represent this query for retrieval\"",
        );

        let cfg = resolve_embedding_config_for_config(&config, |_| None).unwrap();

        assert_eq!(cfg.backend, EmbeddingBackend::OpenRouter);
        assert_eq!(cfg.model, "qwen/qwen3-embedding-8b");
        assert_eq!(cfg.vector_size, 4096);
        assert_eq!(cfg.collection, "claude-session-history-8b");
        assert_eq!(
            cfg.query_instruction.as_deref(),
            Some("Represent this query for retrieval")
        );
    }

    #[test]
    fn embedding_config_environment_overrides_file_profile() {
        let config = parse_config(
            "[embedding]\nbackend = \"ollama\"\nmodel = \"file-model\"\nvector_size = 1024\ncollection = \"file-collection\"\nquery_instruction = \"file instruction\"",
        );
        let values = std::collections::HashMap::from([
            ("CLAUDE_MEMORY_EMBEDDING_BACKEND", "openrouter"),
            ("CLAUDE_MEMORY_EMBEDDING_MODEL", "env-model"),
            ("CLAUDE_MEMORY_VECTOR_SIZE", "4096"),
            ("CLAUDE_MEMORY_COLLECTION", "env-collection"),
            ("CLAUDE_MEMORY_QUERY_INSTRUCTION", "env instruction"),
        ]);

        let cfg = resolve_embedding_config_for_config(&config, |key| {
            values.get(key).map(|value| (*value).to_string())
        })
        .unwrap();

        assert_eq!(cfg.backend, EmbeddingBackend::OpenRouter);
        assert_eq!(cfg.model, "env-model");
        assert_eq!(cfg.vector_size, 4096);
        assert_eq!(cfg.collection, "env-collection");
        assert_eq!(cfg.query_instruction.as_deref(), Some("env instruction"));
    }

    #[test]
    fn embedding_config_rejects_invalid_file_backend() {
        let config = parse_config(
            "[embedding]\nbackend = \"unsupported\"\nmodel = \"model\"\nvector_size = 1024\ncollection = \"collection\"",
        );

        let error = resolve_embedding_config_for_config(&config, |_| None).unwrap_err();

        assert!(format!("{error:#}").contains("embedding.backend"));
    }

    #[test]
    fn embedding_config_applies_environment_overrides() {
        let values = std::collections::HashMap::from([
            ("CLAUDE_MEMORY_EMBEDDING_BACKEND", "openrouter"),
            ("CLAUDE_MEMORY_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"),
            ("CLAUDE_MEMORY_VECTOR_SIZE", "4096"),
            ("CLAUDE_MEMORY_COLLECTION", "claude-session-history-8b"),
            (
                "CLAUDE_MEMORY_QUERY_INSTRUCTION",
                "Represent this query for retrieval",
            ),
        ]);

        let cfg = resolve_embedding_config(|key| values.get(key).map(|value| (*value).to_string()))
            .unwrap();

        assert_eq!(cfg.backend, EmbeddingBackend::OpenRouter);
        assert_eq!(cfg.model, "qwen/qwen3-embedding-8b");
        assert_eq!(cfg.vector_size, 4096);
        assert_eq!(cfg.collection, "claude-session-history-8b");
        assert_eq!(
            cfg.query_instruction.as_deref(),
            Some("Represent this query for retrieval")
        );
    }

    #[test]
    fn embedding_config_rejects_invalid_backend() {
        let values =
            std::collections::HashMap::from([("CLAUDE_MEMORY_EMBEDDING_BACKEND", "unsupported")]);

        let error =
            resolve_embedding_config(|key| values.get(key).map(|value| (*value).to_string()))
                .unwrap_err();

        assert!(format!("{error:#}").contains("CLAUDE_MEMORY_EMBEDDING_BACKEND"));
    }

    #[test]
    fn embedding_config_rejects_blank_model() {
        let values = std::collections::HashMap::from([("CLAUDE_MEMORY_EMBEDDING_MODEL", "  ")]);

        let error =
            resolve_embedding_config(|key| values.get(key).map(|value| (*value).to_string()))
                .unwrap_err();

        assert!(format!("{error:#}").contains("CLAUDE_MEMORY_EMBEDDING_MODEL"));
    }

    #[test]
    fn embedding_config_rejects_nonpositive_vector_size() {
        let values = std::collections::HashMap::from([("CLAUDE_MEMORY_VECTOR_SIZE", "0")]);

        let error =
            resolve_embedding_config(|key| values.get(key).map(|value| (*value).to_string()))
                .unwrap_err();

        assert!(format!("{error:#}").contains("CLAUDE_MEMORY_VECTOR_SIZE"));
    }

    #[test]
    fn embedding_config_rejects_blank_collection() {
        let values = std::collections::HashMap::from([("CLAUDE_MEMORY_COLLECTION", "  ")]);

        let error =
            resolve_embedding_config(|key| values.get(key).map(|value| (*value).to_string()))
                .unwrap_err();

        assert!(format!("{error:#}").contains("CLAUDE_MEMORY_COLLECTION"));
    }

    #[test]
    fn embedding_config_rejects_blank_query_instruction() {
        let values = std::collections::HashMap::from([("CLAUDE_MEMORY_QUERY_INSTRUCTION", "  ")]);

        let error =
            resolve_embedding_config(|key| values.get(key).map(|value| (*value).to_string()))
                .unwrap_err();

        assert!(format!("{error:#}").contains("CLAUDE_MEMORY_QUERY_INSTRUCTION"));
    }
}
