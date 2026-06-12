//! MinHash for Jaccard similarity estimation.
//!
//! MinHash provides locality-sensitive hashing for set similarity by estimating
//! `J(A,B) = |A & B| / |A | B|`.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::{lcg_next, Error, Fnv1a64};

/// MinHash signature generator.
#[derive(Debug, Clone)]
pub struct MinHash {
    /// Seeds for hash functions.
    seeds: Vec<u64>,
}

impl MinHash {
    /// Create a new MinHash with `num_hashes` hash functions, using a fixed seed.
    ///
    /// `num_hashes` must be >= 1.
    pub fn new(num_hashes: usize) -> Result<Self, Error> {
        Self::with_seed(num_hashes, 42)
    }

    /// Create MinHash with a specific seed (deterministic).
    ///
    /// `num_hashes` must be >= 1.
    pub fn with_seed(num_hashes: usize, seed: u64) -> Result<Self, Error> {
        if num_hashes == 0 {
            return Err(Error::InvalidParam("num_hashes must be >= 1"));
        }
        let mut seeds = Vec::with_capacity(num_hashes);
        let mut rng_state = seed;
        for _ in 0..num_hashes {
            seeds.push(lcg_next(&mut rng_state));
        }
        Ok(Self { seeds })
    }

    /// Compute a MinHash signature for a set of items.
    pub fn signature<T: Hash>(&self, items: &HashSet<T>) -> MinHashSignature {
        let mut mins = vec![u64::MAX; self.seeds.len()];
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

    /// Number of hash values in the signature.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True if the signature has no hash values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Estimate Jaccard similarity from two signatures.
    ///
    /// Returns `None` if the signatures have different lengths or are empty.
    pub fn jaccard(&self, other: &Self) -> Option<f64> {
        if self.values.len() != other.values.len() || self.values.is_empty() {
            return None;
        }
        let len = self.values.len();
        // The estimator (matches/len) is the domain meaning and lives here; the
        // differing-slot count is an agnostic primitive. With the `innr`
        // feature it dispatches to SIMD; otherwise it is the scalar loop.
        #[cfg(feature = "innr")]
        let differing = innr::slot_hamming_u64(&self.values, &other.values) as usize;
        #[cfg(not(feature = "innr"))]
        let differing = self
            .values
            .iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a != b)
            .count();
        Some((len - differing) as f64 / len as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // DETERMINISM CANARY -- these values must remain stable across versions.
    // If a change to hashing or PRNG breaks these, the determinism guarantee is violated.

    #[test]
    fn minhash_signature_determinism() {
        let mh = MinHash::new(4).unwrap();
        let items: HashSet<&str> = ["alpha", "beta", "gamma"].into_iter().collect();
        let sig = mh.signature(&items);
        assert_eq!(
            sig.as_slice(),
            &[
                5997730903145785187,
                8107540397845570824,
                5710066622574848307,
                548850101144292092,
            ],
        );
    }

    #[test]
    fn jaccard_identical_signatures() {
        let mh = MinHash::new(8).unwrap();
        let items: HashSet<&str> = ["a", "b"].into_iter().collect();
        let sig = mh.signature(&items);
        assert_eq!(sig.jaccard(&sig), Some(1.0));
    }

    #[test]
    fn jaccard_mismatched_lengths_returns_none() {
        let s4 = MinHash::new(4).unwrap();
        let s8 = MinHash::new(8).unwrap();
        let items: HashSet<&str> = ["x"].into_iter().collect();
        assert_eq!(s4.signature(&items).jaccard(&s8.signature(&items)), None);
    }

    #[test]
    fn new_rejects_zero() {
        assert!(MinHash::new(0).is_err());
    }
}
