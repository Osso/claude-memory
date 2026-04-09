//! Text chunking with overlap for embedding.

use sha2::{Digest, Sha256};

/// Target chunk size in characters (~400 tokens)
const CHUNK_SIZE: usize = 1600;
/// Overlap between chunks (~80 tokens)
const CHUNK_OVERLAP: usize = 320;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub hash: String,
}

/// Split text into overlapping chunks, breaking at newlines when possible.
/// Uses char indices to avoid splitting multi-byte UTF-8 characters.
pub fn chunk_text(text: &str) -> Vec<Chunk> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let char_indices: Vec<(usize, char)> = text.char_indices().collect();
    let char_count = char_indices.len();
    if char_count <= CHUNK_SIZE {
        return vec![build_chunk(text)];
    }

    let mut chunks = Vec::new();
    let mut start_char = 0;
    while start_char < char_count {
        let end_char = (start_char + CHUNK_SIZE).min(char_count);
        let (start_byte, end_byte) =
            chunk_byte_bounds(&char_indices, start_char, end_char, text.len());
        let actual_end_byte = choose_chunk_end(text, start_byte, end_byte, end_char, char_count);
        let chunk_text = text[start_byte..actual_end_byte].trim();
        if !chunk_text.is_empty() {
            chunks.push(build_chunk(chunk_text));
        }
        start_char = next_start_char(text, start_char, start_byte, actual_end_byte, end_char);
    }

    chunks
}

fn build_chunk(text: &str) -> Chunk {
    Chunk {
        text: text.to_string(),
        hash: hash_text(text),
    }
}

fn chunk_byte_bounds(
    char_indices: &[(usize, char)],
    start_char: usize,
    end_char: usize,
    text_len: usize,
) -> (usize, usize) {
    let start_byte = char_indices[start_char].0;
    let end_byte = if end_char >= char_indices.len() {
        text_len
    } else {
        char_indices[end_char].0
    };
    (start_byte, end_byte)
}

fn choose_chunk_end(
    text: &str,
    start_byte: usize,
    end_byte: usize,
    end_char: usize,
    char_count: usize,
) -> usize {
    if end_char >= char_count {
        return end_byte;
    }

    let slice = &text[start_byte..end_byte];
    let Some(newline_pos) = slice.rfind('\n') else {
        return end_byte;
    };

    let chars_before_newline = slice[..newline_pos].chars().count();
    if chars_before_newline > CHUNK_OVERLAP {
        start_byte + newline_pos + 1
    } else {
        end_byte
    }
}

fn next_start_char(
    text: &str,
    start_char: usize,
    start_byte: usize,
    end_byte: usize,
    end_char: usize,
) -> usize {
    let chunk_chars = text[start_byte..end_byte].chars().count();
    let advance = chunk_chars.saturating_sub(CHUNK_OVERLAP);
    if advance == 0 {
        end_char
    } else {
        start_char + advance
    }
}

pub fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    hex::encode(&result[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_text() {
        let chunks = chunk_text("Hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
    }

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_utf8_text() {
        // Test with multi-byte characters
        let text = "Hello 世界! ".repeat(500);
        let chunks = chunk_text(&text);
        assert!(!chunks.is_empty());
        // Verify no panic and all chunks are valid UTF-8
        for chunk in &chunks {
            assert!(chunk.text.is_char_boundary(0));
        }
    }

    #[test]
    fn test_large_text_overlap() {
        // Build text long enough to produce at least 2 chunks.
        // CHUNK_SIZE=1600, CHUNK_OVERLAP=320, so we need > 1600 chars.
        let line = "abcdefghij".repeat(20); // 200 chars per line
        let text = (0..12)
            .map(|_| line.as_str())
            .collect::<Vec<_>>()
            .join("\n"); // ~2412 chars

        let chunks = chunk_text(&text);
        assert!(
            chunks.len() >= 2,
            "expected multiple chunks, got {}",
            chunks.len()
        );

        // Consecutive chunks must share some content (overlap).
        for pair in chunks.windows(2) {
            let first = &pair[0].text;
            let second = &pair[1].text;
            // Take the last 50 chars of the first chunk and verify they appear in the second.
            let tail: String = first
                .chars()
                .rev()
                .take(50)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            assert!(
                second.contains(&tail),
                "no overlap found between consecutive chunks"
            );
        }
    }

    #[test]
    fn test_newline_only_break_points() {
        // Lines with no spaces — only newlines separate content.
        // Each line is shorter than CHUNK_SIZE but together they exceed it.
        let line = "x".repeat(200); // 200 chars, no spaces
        let text = (0..12)
            .map(|_| line.as_str())
            .collect::<Vec<_>>()
            .join("\n"); // ~2412 chars

        let chunks = chunk_text(&text);
        assert!(
            chunks.len() >= 2,
            "expected multiple chunks, got {}",
            chunks.len()
        );

        // Every chunk must be valid UTF-8 and non-empty.
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
            assert!(chunk.text.is_char_boundary(0));
        }
    }

    #[test]
    fn test_chunk_hash_uniqueness() {
        // Produce at least 2 chunks and verify their hashes differ.
        let line = "abcdefghij".repeat(20);
        let text = (0..12)
            .map(|_| line.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let chunks = chunk_text(&text);
        assert!(
            chunks.len() >= 2,
            "expected multiple chunks for hash uniqueness test"
        );

        let hashes: Vec<&str> = chunks.iter().map(|c| c.hash.as_str()).collect();
        let unique: std::collections::HashSet<&str> = hashes.iter().copied().collect();
        assert_eq!(
            hashes.len(),
            unique.len(),
            "chunk hashes are not all unique"
        );
    }
}
