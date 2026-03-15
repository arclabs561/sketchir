//! Text blocking helpers built on MinHash + LSH.
//!
//! This module provides a convenience layer for *text inputs*:
//! you can insert raw strings, and it will compute shingles, MinHash signatures,
//! and use LSH banding to produce candidate pairs.
//!
//! Scope: index-only blocking primitives (no thresholds/policy beyond the config).
//! If you need end-to-end entity resolution, keep that in the calling crate.
//!
//! # Example
//!
//! ```rust
//! use sketchir::blocking::{BlockingConfig, MinHashTextLSH};
//!
//! let mut lsh = MinHashTextLSH::new(BlockingConfig::default()).unwrap();
//! lsh.insert_text("1", "Barack Obama");
//! lsh.insert_text("2", "Barack Obama");
//! lsh.insert_text("3", "Donald Trump");
//!
//! let pairs = lsh.candidate_pairs();
//! assert!(!pairs.is_empty());
//! ```
//!
//! # Notes
//!
//! - Shingles are computed on **Unicode scalar values** (`char`), not bytes.
//! - The banding index uses deterministic hashing for stability.

use std::collections::HashSet;

use crate::lsh::MinHashLSH;
use crate::minhash::{MinHash, MinHashSignature};
use crate::Error;

/// Configuration for MinHash-based text blocking.
#[derive(Debug, Clone)]
pub struct BlockingConfig {
    /// Number of hash functions per band (higher = stricter matching).
    pub num_hashes_per_band: usize,
    /// Number of bands (higher = more candidates, better recall).
    pub num_bands: usize,
    /// N-gram size for text shingling. Must be >= 1.
    pub ngram_size: usize,
    /// Whether to use character n-grams (vs word n-grams).
    pub char_ngrams: bool,
}

impl Default for BlockingConfig {
    fn default() -> Self {
        Self {
            num_hashes_per_band: 4,
            num_bands: 25,
            ngram_size: 3,
            char_ngrams: true,
        }
    }
}

impl BlockingConfig {
    /// Create config optimized for high recall (more candidates).
    pub fn high_recall() -> Self {
        Self {
            num_bands: 50,
            num_hashes_per_band: 2,
            ..Default::default()
        }
    }

    /// Create config optimized for high precision (fewer, better candidates).
    pub fn high_precision() -> Self {
        Self {
            num_bands: 10,
            num_hashes_per_band: 8,
            ..Default::default()
        }
    }

    /// Estimate the probability that two items with given Jaccard similarity
    /// will be placed in the same bucket (i.e., become candidates).
    ///
    /// `P(candidate) = 1 - (1 - s^r)^b`
    /// where `s` is similarity, `r` is `num_hashes_per_band`, and `b` is `num_bands`.
    pub fn candidate_probability(&self, jaccard_similarity: f64) -> f64 {
        let s = jaccard_similarity;
        let r = self.num_hashes_per_band as f64;
        let b = self.num_bands as f64;
        1.0 - (1.0 - s.powf(r)).powf(b)
    }
}

/// A text item stored in the index.
#[derive(Debug, Clone)]
pub(crate) struct TextItem {
    /// External identifier.
    id: String,
    signature: MinHashSignature,
}

/// MinHash + banding LSH over raw text.
#[derive(Debug)]
pub struct MinHashTextLSH {
    config: BlockingConfig,
    minhash: MinHash,
    index: MinHashLSH,
    items: Vec<TextItem>,
}

impl MinHashTextLSH {
    /// Create a new MinHashTextLSH.
    pub fn new(config: BlockingConfig) -> Result<Self, Error> {
        if config.ngram_size == 0 {
            return Err(Error::InvalidParam("ngram_size must be >= 1"));
        }

        let total_hashes = config
            .num_bands
            .checked_mul(config.num_hashes_per_band)
            .ok_or(Error::InvalidParam(
                "num_bands * num_hashes_per_band overflow",
            ))?;

        let minhash = MinHash::new(total_hashes)?;
        let index = MinHashLSH::new(config.num_bands, config.num_hashes_per_band)?;

        Ok(Self {
            config,
            minhash,
            index,
            items: Vec::new(),
        })
    }

    /// Insert a new text item.
    pub fn insert_text(&mut self, id: impl Into<String>, text: impl AsRef<str>) {
        let id = id.into();

        let shingles = shingle(
            text.as_ref(),
            self.config.ngram_size,
            self.config.char_ngrams,
        );
        let signature = self.minhash.signature(&shingles);

        // Safety: signature length is guaranteed to match by construction
        // (MinHash and MinHashLSH were created with the same total_hashes).
        let doc_id = self
            .index
            .insert(signature.clone())
            .expect("signature length matches by construction");
        debug_assert_eq!(
            doc_id,
            self.items.len(),
            "doc_id should follow insertion order"
        );

        self.items.push(TextItem { id, signature });
    }

    /// Query for candidate indices that collide with `text` in any bucket.
    pub fn query(&self, text: &str) -> Vec<usize> {
        let shingles = shingle(text, self.config.ngram_size, self.config.char_ngrams);
        let signature = self.minhash.signature(&shingles);
        self.index.query(&signature)
    }

    /// Get all candidate pairs (sorted for deterministic output).
    pub fn candidate_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs: HashSet<(usize, usize)> = HashSet::new();

        for i in 0..self.items.len() {
            for j in self.index.query(&self.items[i].signature) {
                if i == j {
                    continue;
                }
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                pairs.insert((a, b));
            }
        }

        let mut v: Vec<(usize, usize)> = pairs.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Estimated similarity from MinHash signatures.
    pub fn estimated_similarity(&self, i: usize, j: usize) -> Option<f64> {
        let a = self.items.get(i)?;
        let b = self.items.get(j)?;
        a.signature.jaccard(&b.signature)
    }

    /// Get the external ID of the item at a given index.
    pub fn get_id(&self, idx: usize) -> Option<&str> {
        self.items.get(idx).map(|item| item.id.as_str())
    }

    /// Number of items indexed.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

fn shingle(text: &str, n: usize, char_ngrams: bool) -> HashSet<String> {
    debug_assert!(
        n >= 1,
        "ngram_size must be >= 1 (validated at construction)"
    );
    let normalized = text.to_lowercase();

    if char_ngrams {
        let chars: Vec<char> = normalized.chars().collect();
        if chars.len() < n {
            return HashSet::from([normalized]);
        }
        chars
            .windows(n)
            .map(|w| w.iter().collect::<String>())
            .collect()
    } else {
        let words: Vec<&str> = normalized.split_whitespace().collect();
        if words.len() < n {
            return HashSet::from([normalized]);
        }
        words.windows(n).map(|w| w.join(" ")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings_become_candidates() {
        let mut lsh = MinHashTextLSH::new(BlockingConfig::default()).unwrap();
        lsh.insert_text("1", "Barack Obama");
        lsh.insert_text("2", "Barack Obama");
        assert!(!lsh.candidate_pairs().is_empty());
    }

    #[test]
    fn estimated_similarity_is_one_for_identical() {
        let mut lsh = MinHashTextLSH::new(BlockingConfig::default()).unwrap();
        lsh.insert_text("1", "New York");
        lsh.insert_text("2", "New York");
        let sim = lsh.estimated_similarity(0, 1).unwrap();
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn candidate_pairs_are_sorted() {
        let mut lsh = MinHashTextLSH::new(BlockingConfig::default()).unwrap();
        lsh.insert_text("a", "hello world foo bar");
        lsh.insert_text("b", "hello world foo bar");
        lsh.insert_text("c", "hello world foo bar");
        let pairs = lsh.candidate_pairs();
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        assert_eq!(pairs, sorted);
    }

    #[test]
    fn rejects_zero_ngram_size() {
        let cfg = BlockingConfig {
            ngram_size: 0,
            ..Default::default()
        };
        assert!(MinHashTextLSH::new(cfg).is_err());
    }

    #[test]
    fn get_id_returns_inserted_id() {
        let mut lsh = MinHashTextLSH::new(BlockingConfig::default()).unwrap();
        lsh.insert_text("my-id", "some text");
        assert_eq!(lsh.get_id(0), Some("my-id"));
        assert_eq!(lsh.get_id(1), None);
    }
}
