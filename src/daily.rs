//! Daily log file management.

use anyhow::{Context, Result};
use chrono::Local;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

/// Get the memory directory path.
pub fn memory_dir() -> PathBuf {
    dirs::home_dir()
        .expect("no home directory")
        .join(".claude/memory")
}

/// Get today's daily log path.
pub fn daily_log_path() -> PathBuf {
    let date = Local::now().format("%Y-%m-%d");
    memory_dir().join("daily").join(format!("{}.md", date))
}

/// Ensure the memory directory structure exists.
pub fn ensure_dirs() -> Result<()> {
    let base = memory_dir();
    fs::create_dir_all(base.join("daily")).context("failed to create daily dir")?;
    fs::create_dir_all(base.join("projects")).context("failed to create projects dir")?;
    Ok(())
}

/// Append an entry to today's daily log.
pub fn append_daily(content: &str, category: Option<&str>, project: Option<&str>) -> Result<()> {
    ensure_dirs()?;

    let path = daily_log_path();
    let time = Local::now().format("%H:%M");

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .context("failed to open daily log")?;

    // Format the entry
    let mut entry = String::new();
    entry.push_str(&format!("\n## {}", time));

    if let Some(cat) = category {
        entry.push_str(&format!(" [{}]", cat));
    }
    if let Some(proj) = project {
        entry.push_str(&format!(" ({})", proj));
    }
    entry.push('\n');
    entry.push_str(content);
    entry.push('\n');

    file.write_all(entry.as_bytes())
        .context("failed to write to daily log")?;

    Ok(())
}

/// Get the project memory file path.
pub fn project_path(project: &str) -> PathBuf {
    memory_dir().join("projects").join(format!("{}.md", project))
}

/// Append to a project's memory file.
pub fn append_project(project: &str, content: &str) -> Result<()> {
    ensure_dirs()?;

    let path = project_path(project);
    let time = Local::now().format("%Y-%m-%d %H:%M");

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .context("failed to open project memory")?;

    let entry = format!("\n## {}\n{}\n", time, content);
    file.write_all(entry.as_bytes())
        .context("failed to write to project memory")?;

    Ok(())
}
