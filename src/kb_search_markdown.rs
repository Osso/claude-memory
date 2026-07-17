use std::path::Path;

#[derive(Debug, Clone)]
pub(super) struct MarkdownSection {
    pub(super) heading_path: String,
    pub(super) source_line: usize,
    pub(super) text: String,
}

pub(super) fn split_markdown_sections(path: &str, markdown: &str) -> Vec<MarkdownSection> {
    let mut heading_stack = Vec::new();
    let mut current = None;
    let mut sections = Vec::new();
    let mut in_code_fence = false;

    for (line_index, line) in markdown.lines().enumerate() {
        let line_number = line_index + 1;
        if is_fence_line(line) {
            in_code_fence = !in_code_fence;
            push_line(path, &mut current, line_number, line);
            continue;
        }
        if !in_code_fence && let Some((level, title)) = parse_heading(line) {
            push_section(&mut current, &mut sections);
            heading_stack.truncate(level.saturating_sub(1));
            heading_stack.push(title);
            current = Some(SectionBuilder::new(heading_stack.join(" > "), line_number));
            current.as_mut().expect("section exists").push_line(line);
            continue;
        }
        push_line(path, &mut current, line_number, line);
    }

    push_section(&mut current, &mut sections);
    sections
}

struct SectionBuilder {
    heading_path: String,
    source_line: usize,
    text: String,
}

impl SectionBuilder {
    fn new(heading_path: String, source_line: usize) -> Self {
        Self {
            heading_path,
            source_line,
            text: String::new(),
        }
    }

    fn push_line(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
    }

    fn finish(self) -> MarkdownSection {
        MarkdownSection {
            heading_path: self.heading_path,
            source_line: self.source_line,
            text: self.text.trim().to_string(),
        }
    }
}

fn push_line(path: &str, current: &mut Option<SectionBuilder>, line_number: usize, line: &str) {
    if current.is_none() && line.trim().is_empty() {
        return;
    }
    if current.is_none() {
        *current = Some(SectionBuilder::new(fallback_heading(path), line_number));
    }
    current.as_mut().expect("section exists").push_line(line);
}

fn push_section(current: &mut Option<SectionBuilder>, sections: &mut Vec<MarkdownSection>) {
    if let Some(section) = current.take() {
        sections.push(section.finish());
    }
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let title = trimmed.get(level..)?.trim();
    if title.is_empty() {
        return None;
    }
    Some((level, title.trim_matches('#').trim().to_string()))
}

fn is_fence_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

fn fallback_heading(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}
