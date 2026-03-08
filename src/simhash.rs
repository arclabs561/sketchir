//! SimHash: binary fingerprints for fast near-duplicate detection.
//!
//! SimHash (Charikar, 2002) maps weighted feature vectors to a fixed-width bitstring such that
//! similar items have small Hamming distance.

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
///
/// Stateless: `fingerprint_weighted` is a pure function of its inputs.
#[derive(Debug, Clone)]
pub struct SimHash;

impl SimHash {
    /// Create a SimHash generator.
    pub fn new() -> Self {
        Self
    }

    /// Fingerprint a bag of (feature_hash, weight) pairs.
    ///
    /// `feature_hash` is assumed to be a 64-bit hash of the feature.
    pub fn fingerprint_weighted(&self, features: &[(u64, f32)]) -> SimHashFingerprint {
        let mut acc = [0f32; 64];
        for (h, w) in features {
            let bits = *h;
            #[allow(clippy::needless_range_loop)]
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
        #[allow(clippy::needless_range_loop)]
        for i in 0..64 {
            if acc[i] > 0.0 {
                out |= 1u64 << i;
            }
        }
        SimHashFingerprint(out)
    }
}

impl Default for SimHash {
    fn default() -> Self {
        Self::new()
    }
}
