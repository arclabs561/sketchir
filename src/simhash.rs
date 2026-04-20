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

/// Fingerprint a bag of `(feature_hash, weight)` pairs.
///
/// `feature_hash` is assumed to be a 64-bit hash of the feature.
/// This is a pure function -- no state required.
pub fn simhash_fingerprint(features: &[(u64, f32)]) -> SimHashFingerprint {
    let mut acc = [0f32; 64];
    for &(bits, w) in features {
        // Branch-free: weight is added as +w or -w based on the bit.
        // The sign pattern 2*bit-1 is computed without branching.
        // Unrolled into 4-wide chunks so LLVM can use multiple FP accumulators.
        let mut i = 0;
        while i + 3 < 64 {
            acc[i] += if (bits >> i) & 1 != 0 { w } else { -w };
            acc[i + 1] += if (bits >> (i + 1)) & 1 != 0 { w } else { -w };
            acc[i + 2] += if (bits >> (i + 2)) & 1 != 0 { w } else { -w };
            acc[i + 3] += if (bits >> (i + 3)) & 1 != 0 { w } else { -w };
            i += 4;
        }
        while i < 64 {
            acc[i] += if (bits >> i) & 1 != 0 { w } else { -w };
            i += 1;
        }
    }
    let mut out = 0u64;
    for (i, a) in acc.iter().enumerate() {
        if *a > 0.0 {
            out |= 1u64 << i;
        }
    }
    SimHashFingerprint(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // DETERMINISM CANARY
    #[test]
    fn simhash_fingerprint_determinism() {
        let fp = simhash_fingerprint(&[(0xDEADBEEF, 1.0), (0xCAFEBABE, 0.5)]);
        assert_eq!(fp.0, 3735928559);
    }

    #[test]
    fn identical_features_zero_distance() {
        let fp1 = simhash_fingerprint(&[(1, 1.0), (2, 1.0)]);
        let fp2 = simhash_fingerprint(&[(1, 1.0), (2, 1.0)]);
        assert_eq!(fp1.hamming_distance(&fp2), 0);
    }

    #[test]
    fn empty_features_produce_zero_fingerprint() {
        let fp = simhash_fingerprint(&[]);
        assert_eq!(fp.0, 0);
    }
}
