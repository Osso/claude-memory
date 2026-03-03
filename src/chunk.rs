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

    // Convert to char indices for safe slicing
    let char_indices: Vec<(usize, char)> = text.char_indices().collect();
    let char_count = char_indices.len();

    if char_count <= CHUNK_SIZE {
        return vec![Chunk {
            text: text.to_string(),
            hash: hash_text(text),
        }];
    }

    let mut chunks = Vec::new();
    let mut start_char = 0;

    while start_char < char_count {
        let end_char = (start_char + CHUNK_SIZE).min(char_count);

        // Get byte positions
        let start_byte = char_indices[start_char].0;
        let end_byte = if end_char >= char_count {
            text.len()
        } else {
            char_indices[end_char].0
        };

        // Try to break at newline
        let slice = &text[start_byte..end_byte];
        let actual_end_byte = if end_char < char_count {
            if let Some(newline_pos) = slice.rfind('\n') {
                // Only use newline if it's not too close to start
                let chars_before_newline = slice[..newline_pos].chars().count();
                if chars_before_newline > CHUNK_OVERLAP {
                    start_byte + newline_pos + 1
                } else {
                    end_byte
                }
            } else {
                end_byte
            }
        } else {
            end_byte
        };

        let chunk_text = text[start_byte..actual_end_byte].trim();
        if !chunk_text.is_empty() {
            chunks.push(Chunk {
                text: chunk_text.to_string(),
                hash: hash_text(chunk_text),
            });
        }

        // Move start, accounting for overlap in chars
        let chunk_chars = text[start_byte..actual_end_byte].chars().count();
        let advance = chunk_chars.saturating_sub(CHUNK_OVERLAP);
        if advance == 0 {
            start_char = end_char; // Prevent infinite loop
        } else {
            start_char += advance;
        }
    }

    chunks
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
}
