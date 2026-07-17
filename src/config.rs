use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();

#[derive(Debug)]
pub struct Config {
    pub search: SearchConfig,
}

#[derive(Debug)]
pub struct SearchConfig {
    pub enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            search: SearchConfig { enabled: false },
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
        Ok(table) => Config {
            search: SearchConfig {
                enabled: table_enabled(&table, "search"),
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
}
