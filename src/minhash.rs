//! MinHash for Jaccard similarity estimation.
//!
//! MinHash provides locality-sensitive hashing for set similarity by estimating:
//! \(J(A,B) = |A ∩ B| / |A ∪ B|\).

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::lcg_next;

/// A small stable 64-bit FNV-1a hasher.
///
/// This avoids relying on `std`'s `DefaultHasher` stability guarantees.
pub(crate) struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    pub(crate) fn new() -> Self {
        // FNV offset basis
        Self {
            state: 0xcbf29ce484222325,
        }
    }
}

impl Hasher for Fnv1a64 {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        // FNV-1a
        const PRIME: u64 = 0x00000100000001B3;
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(PRIME);
        }
    }
}

/// MinHash signature generator.
#[derive(Debug, Clone)]
pub struct MinHash {
    /// Number of hash functions (signature length).
    num_hashes: usize,
    /// Seeds for hash functions.
    seeds: Vec<u64>,
}

impl MinHash {
    /// Create a new MinHash with `num_hashes` hash functions, using a fixed seed.
    pub fn new(num_hashes: usize) -> Self {
        Self::with_seed(num_hashes, 42)
    }

    /// Create MinHash with a specific seed (deterministic).
    pub fn with_seed(num_hashes: usize, seed: u64) -> Self {
        let mut seeds = Vec::with_capacity(num_hashes);
        let mut rng_state = seed;
        for _ in 0..num_hashes {
            seeds.push(lcg_next(&mut rng_state));
        }
        Self { num_hashes, seeds }
    }

    /// Compute a MinHash signature for a set of items.
    pub fn signature<T: Hash>(&self, items: &HashSet<T>) -> MinHashSignature {
        let mut mins = vec![u64::MAX; self.num_hashes];
        for item in items {
            for (i, &seed) in self.seeds.iter().enumerate() {
                let h = self.hash_with_seed(item, seed);
                if h < mins[i] {
                    mins[i] = h;
                }
            }
        }
        MinHashSignature { values: mins }
    }

    fn hash_with_seed<T: Hash>(&self, item: &T, seed: u64) -> u64 {
        let mut hasher = Fnv1a64::new();
        seed.to_le_bytes().hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }
}

/// A MinHash signature (fingerprint) of a set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashSignature {
    /// The min-hash values for each hash function.
    pub(crate) values: Vec<u64>,
}

impl MinHashSignature {
    /// View the raw min-hash values.
    pub fn as_slice(&self) -> &[u64] {
        &self.values
    }

    /// Estimate Jaccard similarity from two signatures.
    pub fn jaccard(&self, other: &Self) -> f64 {
        if self.values.len() != other.values.len() || self.values.is_empty() {
            return 0.0;
        }
        let matches = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.values.len() as f64
    }
}
