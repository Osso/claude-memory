pub(super) fn sanitize_triplet(
    subject: &str,
    relation: &str,
    object: &str,
) -> Option<(String, String, String)> {
    let subject = normalize_entity(subject)?;
    let relation = normalize_relation(relation)?;
    let object = normalize_entity(object)?;

    if is_blocked_triplet(&subject, &relation, &object) {
        return None;
    }

    Some((subject, relation, object))
}

#[cfg(test)]
pub(super) fn is_valid_entity_name(name: &str) -> bool {
    normalize_entity(name).is_some()
}

fn normalize_entity(name: &str) -> Option<String> {
    let normalized = collapse_whitespace(name);
    if !has_valid_entity_shape(&normalized) {
        return None;
    }
    if has_rejected_entity_prefix(&normalized) {
        return None;
    }
    if looks_like_rejected_entity(&normalized) {
        return None;
    }
    Some(normalized)
}

fn has_valid_entity_shape(name: &str) -> bool {
    if name.len() < 2 || name.len() > 60 {
        return false;
    }
    if name.starts_with(|c: char| !c.is_ascii_alphanumeric()) {
        return false;
    }
    name.split_whitespace().count() <= 3
}

fn has_rejected_entity_prefix(name: &str) -> bool {
    name.starts_with('/')
        || name.starts_with('.')
        || name.contains("/.")
        || name.contains('/')
        || name.starts_with('-')
        || name.starts_with('+')
        || name.starts_with('@')
        || name.starts_with('$')
        || name.starts_with('#')
        || name.starts_with('&')
}

fn looks_like_rejected_entity(name: &str) -> bool {
    looks_like_permission_scope(name)
        || looks_like_network_identifier(name)
        || looks_like_symbolic_artifact(name)
        || looks_like_number_or_hash(name)
        || looks_like_code_artifact(name)
        || has_file_extension(name)
        || first_word_is_numeric(name)
}

fn looks_like_code_artifact(name: &str) -> bool {
    name.contains(".*")
        || name.contains("$(")
        || name.contains("=>")
        || name.contains('(')
        || name.contains(')')
        || name.contains('<')
        || name.contains('>')
        || name.contains('%')
        || name.contains('\'')
        || name.contains('"')
}

fn normalize_relation(relation: &str) -> Option<String> {
    let normalized = relation.trim().to_lowercase().replace([' ', '-'], "_");
    if normalized.is_empty() {
        return None;
    }

    let relation = match normalized.as_str() {
        "writtenin" => "written_in",
        "builtin" => "built_in",
        "usedin" => "used_in",
        "requires_api" => return None,
        _ => normalized.as_str(),
    };

    if BLOCKED_RELATIONS.contains(&relation) {
        return None;
    }
    if !ALLOWED_RELATIONS.contains(&relation) {
        return None;
    }

    Some(relation.to_string())
}

fn is_blocked_triplet(subject: &str, relation: &str, object: &str) -> bool {
    if relation == "provides" && looks_like_permission_scope(object) {
        return true;
    }
    if relation == "requires" && subject.ends_with(" CLI") && object == "Microsoft Graph" {
        return true;
    }
    GENERIC_OBJECTS.contains(&object)
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn looks_like_permission_scope(name: &str) -> bool {
    if name == "offline_access" {
        return true;
    }
    if name.contains('.') && !name.contains(' ') {
        return true;
    }
    name.contains('_') && !name.contains(' ')
}

fn looks_like_network_identifier(name: &str) -> bool {
    if name.matches(':').count() >= 2 || name.contains("::") {
        return true;
    }
    name.chars().filter(|c| c.is_ascii_digit()).count() >= 4
        && (name.contains('.') || name.contains('/'))
}

fn looks_like_symbolic_artifact(name: &str) -> bool {
    let alnum = name.chars().filter(|c| c.is_ascii_alphanumeric()).count();
    if alnum == 0 {
        return true;
    }
    let punctuation = name
        .chars()
        .filter(|c| !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace() && *c != '-')
        .count();
    punctuation >= alnum
}

const BLOCKED_RELATIONS: &[&str] = &[
    "documented_in",
    "contains",
    "feature",
    "has_feature",
    "includes_tool",
    "demonstrates",
    "sometimes_conflicts_with",
    "publishes_about",
    "should_be",
    "is_program",
    "describes",
    "full_name",
    "represents",
    "purpose",
    "targets",
    "workflow_preference",
    "coding_practice",
    "practices",
    "experience",
    "experienced_in",
    "location",
    "compares_with",
    "compared_with",
    "competes_with",
    "stronger_at",
    "acted_in",
    "worked_on",
    "follows",
    "follows_pattern",
    "follows_standard",
    "groups",
    "knows",
    "avoids",
    "should_avoid",
    "should_create",
    "defends_against",
    "measures",
    "decreases_in",
    "is",
    "is_alias_for",
    "is_theme_for",
];

const ALLOWED_RELATIONS: &[&str] = &[
    "applies_mixin",
    "authenticates_with",
    "benchmarked_on",
    "built_in",
    "built_with",
    "calls",
    "compatible_with",
    "component_in",
    "configures",
    "connects_to",
    "depends_on",
    "deploys_to",
    "developed_by",
    "distributed_via",
    "enables",
    "exposes",
    "featured_in",
    "has",
    "has_issue",
    "has_method",
    "hosts",
    "implemented_in",
    "implements",
    "includes",
    "incompatible_with",
    "indexed_in",
    "integrates_with",
    "license",
    "loads_data_from",
    "maintains",
    "manages",
    "maps_to",
    "monitors",
    "needed_for",
    "owned_by",
    "packaged_as",
    "part_of",
    "powers",
    "product_of",
    "produces",
    "project_of",
    "provides",
    "queries",
    "references",
    "replaces",
    "requires",
    "reviewed_by",
    "runs_on",
    "stores_in",
    "supports",
    "tested_on",
    "tested_with",
    "used_by",
    "used_for",
    "used_in",
    "uses",
    "uses_architecture",
    "uses_mixin",
    "use_protocol",
    "written_in",
];

const GENERIC_OBJECTS: &[&str] = &[
    "instructions",
    "memory",
    "memory improvements",
    "performance critical",
    "use cases",
];

const GOOD_NUM_PREFIXES: &[&str] = &["2d", "3d", "2fa", "4k"];

fn first_word_is_numeric(name: &str) -> bool {
    let first = match name.split_whitespace().next() {
        Some(w) => w,
        None => return true,
    };
    if GOOD_NUM_PREFIXES
        .iter()
        .any(|p| first.eq_ignore_ascii_case(p))
    {
        return false;
    }
    if !first.starts_with(|c: char| c.is_ascii_digit()) {
        return false;
    }
    if first.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    let digit_prefix: String = first
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect();
    let rest = &first[digit_prefix.len()..];
    !digit_prefix.is_empty() && rest.len() <= 3
}

fn has_file_extension(name: &str) -> bool {
    const EXTS: &[&str] = &[
        ".md", ".rs", ".js", ".ts", ".tsx", ".php", ".py", ".toml", ".yaml", ".yml", ".json",
        ".html", ".css", ".go", ".sh",
    ];
    EXTS.iter().any(|ext| name.ends_with(ext))
}

pub(super) fn looks_like_number_or_hash(name: &str) -> bool {
    let stripped = name.replace(['.', ',', ' ', '-', '_', '%', '+'], "");
    if stripped.is_empty() {
        return true;
    }
    stripped
        .chars()
        .all(|c| c.is_ascii_digit() || c.is_ascii_hexdigit() || "KMGBikb".contains(c))
}
