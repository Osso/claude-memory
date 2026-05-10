use std::path::Path;

#[derive(Debug, Clone)]
pub(super) struct MarkdownSection {
    pub(super) node_id: String,
    pub(super) title: String,
    pub(super) heading_path: String,
    pub(super) level: usize,
    pub(super) parent: Option<String>,
    pub(super) source_line: usize,
    pub(super) text: String,
}

pub(super) fn split_markdown_sections(path: &str, markdown: &str) -> Vec<MarkdownSection> {
    let mut state = MarkdownSplitState::new(path);

    for (line_index, line) in markdown.lines().enumerate() {
        state.process_line(line_index + 1, line);
    }

    state.finish()
}

struct MarkdownSplitState<'a> {
    path: &'a str,
    heading_stack: Vec<String>,
    id_stack: Vec<String>,
    current: Option<SectionBuilder>,
    sections: Vec<MarkdownSection>,
    heading_count: usize,
    in_code_fence: bool,
}

impl<'a> MarkdownSplitState<'a> {
    fn new(path: &'a str) -> Self {
        Self {
            path,
            heading_stack: Vec::new(),
            id_stack: Vec::new(),
            current: None,
            sections: Vec::new(),
            heading_count: 0,
            in_code_fence: false,
        }
    }

    fn process_line(&mut self, line_number: usize, line: &str) {
        if is_fence_line(line) {
            self.in_code_fence = !self.in_code_fence;
            self.push_markdown_line(line_number, line);
            return;
        }

        if !self.in_code_fence && self.try_start_heading_section(line, line_number) {
            return;
        }

        self.push_markdown_line(line_number, line);
    }

    fn try_start_heading_section(&mut self, line: &str, line_number: usize) -> bool {
        try_start_heading_section(
            line,
            line_number,
            &mut self.heading_stack,
            &mut self.id_stack,
            &mut self.heading_count,
            &mut self.current,
            &mut self.sections,
        )
    }

    fn push_markdown_line(&mut self, line_number: usize, line: &str) {
        push_markdown_line(
            self.path,
            &mut self.current,
            &mut self.heading_count,
            line_number,
            line,
        );
    }

    fn finish(mut self) -> Vec<MarkdownSection> {
        if let Some(section) = self.current {
            section.push(&mut self.sections);
        }
        self.sections
    }
}

fn try_start_heading_section(
    line: &str,
    line_number: usize,
    heading_stack: &mut Vec<String>,
    id_stack: &mut Vec<String>,
    heading_count: &mut usize,
    current: &mut Option<SectionBuilder>,
    sections: &mut Vec<MarkdownSection>,
) -> bool {
    let Some((level, title)) = parse_heading(line) else {
        return false;
    };

    if let Some(section) = current.take() {
        section.push(sections);
    }
    *heading_count += 1;
    heading_stack.truncate(level.saturating_sub(1));
    id_stack.truncate(level.saturating_sub(1));
    heading_stack.push(title.clone());

    let node_id = format_node_id(*heading_count);
    let parent = id_stack.last().cloned();
    id_stack.push(node_id.clone());
    *current = Some(SectionBuilder::new(
        &node_id,
        &title,
        &heading_stack.join(" > "),
        level,
        parent,
        line_number,
    ));
    if let Some(section) = current {
        section.push_line(line);
    }
    true
}

struct SectionBuilder {
    node_id: String,
    title: String,
    heading_path: String,
    level: usize,
    parent: Option<String>,
    source_line: usize,
    text: String,
}

impl SectionBuilder {
    fn new(
        node_id: &str,
        title: &str,
        heading_path: &str,
        level: usize,
        parent: Option<String>,
        source_line: usize,
    ) -> Self {
        Self {
            node_id: node_id.to_string(),
            title: title.to_string(),
            heading_path: heading_path.to_string(),
            level,
            parent,
            source_line,
            text: String::new(),
        }
    }

    fn push_line(&mut self, line: &str) {
        self.text.push_str(line);
        self.text.push('\n');
    }

    fn push(self, sections: &mut Vec<MarkdownSection>) {
        sections.push(MarkdownSection {
            node_id: self.node_id,
            title: self.title,
            heading_path: self.heading_path,
            level: self.level,
            parent: self.parent,
            source_line: self.source_line,
            text: self.text.trim().to_string(),
        });
    }
}

fn push_markdown_line(
    path: &str,
    current: &mut Option<SectionBuilder>,
    heading_count: &mut usize,
    line_number: usize,
    line: &str,
) {
    if current.is_none() && line.trim().is_empty() {
        return;
    }

    if current.is_none() {
        *heading_count += 1;
        let title = fallback_heading(path);
        let node_id = format_node_id(*heading_count);
        *current = Some(SectionBuilder::new(
            &node_id,
            &title,
            &title,
            1,
            None,
            line_number,
        ));
    }

    if let Some(section) = current {
        section.push_line(line);
    }
}

fn format_node_id(position: usize) -> String {
    format!("{position:06}")
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

pub(super) fn fallback_heading(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}
