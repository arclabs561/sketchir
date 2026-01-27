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

use crate::lsh::{Error, MinHashLSH};
use crate::minhash::{MinHash, MinHashSignature};

/// Configuration for MinHash-based text blocking.
#[derive(Debug, Clone)]
pub struct BlockingConfig {
    /// Number of hash functions per band (higher = stricter matching).
    pub num_hashes_per_band: usize,
    /// Number of bands (higher = more candidates, better recall).
    pub num_bands: usize,
    /// N-gram size for text shingling.
    pub ngram_size: usize,
    /// Whether to use character n-grams (vs word n-grams).
    pub char_ngrams: bool,
    /// Minimum Jaccard similarity threshold for convenience filtering.
    pub similarity_threshold: f64,
}

impl Default for BlockingConfig {
    fn default() -> Self {
        Self {
            num_hashes_per_band: 4,
            num_bands: 25,
            ngram_size: 3,
            char_ngrams: true,
            similarity_threshold: 0.5,
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
    /// \(P(\text{candidate}) = 1 - (1 - s^r)^b\)
    /// where \(s\) is similarity, \(r\) is `num_hashes_per_band`, and \(b\) is `num_bands`.
    pub fn candidate_probability(&self, jaccard_similarity: f64) -> f64 {
        let s = jaccard_similarity;
        let r = self.num_hashes_per_band as f64;
        let b = self.num_bands as f64;
        1.0 - (1.0 - s.powf(r)).powf(b)
    }
}

/// A text item stored in the index.
#[derive(Debug, Clone)]
pub struct TextItem {
    /// External identifier.
    pub id: String,
    /// The raw text content.
    pub text: String,
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
        let total_hashes = config
            .num_bands
            .checked_mul(config.num_hashes_per_band)
            .ok_or(Error::InvalidParam(
                "num_bands * num_hashes_per_band overflow",
            ))?;

        let minhash = MinHash::new(total_hashes);
        let index = MinHashLSH::new(config.num_bands, config.num_hashes_per_band)?;

        Ok(Self {
            config,
            minhash,
            index,
            items: Vec::new(),
        })
    }

    /// Insert a new text item.
    pub fn insert_text(&mut self, id: impl Into<String>, text: impl Into<String>) {
        let id = id.into();
        let text = text.into();

        let shingles = shingle(&text, self.config.ngram_size, self.config.char_ngrams);
        let signature = self.minhash.signature(&shingles);

        let doc_id = self.index.insert(signature.clone());
        debug_assert_eq!(
            doc_id,
            self.items.len(),
            "doc_id should follow insertion order"
        );

        self.items.push(TextItem {
            id,
            text,
            signature,
        });
    }

    /// Query for candidate indices that collide with `text` in any bucket.
    pub fn query(&self, text: &str) -> Vec<usize> {
        let shingles = shingle(text, self.config.ngram_size, self.config.char_ngrams);
        let signature = self.minhash.signature(&shingles);
        self.index.query(&signature)
    }

    /// Get all candidate pairs.
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

        pairs.into_iter().collect()
    }

    /// Candidate pairs with estimated similarity (signature Jaccard).
    pub fn candidate_pairs_with_similarity(&self) -> Vec<(usize, usize, f64)> {
        self.candidate_pairs()
            .into_iter()
            .filter_map(|(i, j)| {
                let sim = self.estimated_similarity(i, j)?;
                if sim >= self.config.similarity_threshold {
                    Some((i, j, sim))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Estimated similarity from MinHash signatures.
    pub fn estimated_similarity(&self, i: usize, j: usize) -> Option<f64> {
        let a = self.items.get(i)?;
        let b = self.items.get(j)?;
        Some(a.signature.jaccard(&b.signature))
    }

    /// Exact Jaccard similarity on shingles (slower than signature estimate).
    pub fn exact_similarity(&self, i: usize, j: usize) -> Option<f64> {
        let a = self.items.get(i)?;
        let b = self.items.get(j)?;
        let sa = shingle(&a.text, self.config.ngram_size, self.config.char_ngrams);
        let sb = shingle(&b.text, self.config.ngram_size, self.config.char_ngrams);
        Some(jaccard_similarity(&sa, &sb))
    }

    /// Get the item at a given index.
    pub fn get(&self, idx: usize) -> Option<&TextItem> {
        self.items.get(idx)
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
    let normalized = text.to_lowercase();
    if n == 0 {
        return HashSet::new();
    }

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

fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
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
}
