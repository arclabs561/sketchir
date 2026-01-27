//! SimHash: binary fingerprints for fast near-duplicate detection.
//!
//! SimHash (Charikar, 2002) maps weighted feature vectors to a fixed-width bitstring such that
//! similar items have small Hamming distance.

use rand::{RngCore, SeedableRng};

/// A SimHash fingerprint (64-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimHashFingerprint(pub u64);

impl SimHashFingerprint {
    /// Hamming distance between two fingerprints (XOR + popcount).
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }
}

/// SimHash generator.
#[derive(Debug, Clone)]
pub struct SimHash {
    seed: u64,
}

impl SimHash {
    /// Create a SimHash generator with a fixed seed (deterministic).
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Fingerprint a bag of (feature_hash, weight) pairs.
    ///
    /// `feature_hash` is assumed to be a 64-bit hash of the feature.
    pub fn fingerprint_weighted(&self, features: &[(u64, f32)]) -> SimHashFingerprint {
        let mut acc = [0f32; 64];
        for (h, w) in features {
            let bits = *h;
            for i in 0..64 {
                let bit = (bits >> i) & 1;
                if bit == 1 {
                    acc[i] += *w;
                } else {
                    acc[i] -= *w;
                }
            }
        }
        let mut out = 0u64;
        for i in 0..64 {
            if acc[i] > 0.0 {
                out |= 1u64 << i;
            }
        }
        SimHashFingerprint(out)
    }

    /// A tiny helper to generate random hyperplane-like feature hashes for demos/tests.
    ///
    /// Not a general-purpose hashing API.
    pub fn demo_random_feature_hashes(&self, n: usize) -> Vec<u64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        (0..n).map(|_| rng.next_u64()).collect()
    }
}
