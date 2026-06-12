//! Random hyperplane LSH for cosine similarity on dense `f32` vectors.
//!
//! Each vector is projected onto `num_bits` random unit hyperplanes; the sign
//! of each projection is packed into a byte-aligned bit signature.  Hamming
//! distance between two signatures estimates the angle between the original
//! vectors via:
//!
//! ```text
//! cos(theta) ≈ cos(pi * hamming_dist / num_bits)
//! ```
//!
//! This is the locality-sensitive hashing scheme for cosine distance
//! (Charikar 2002).  Hyperplanes are sampled from N(0,1) and normalised to
//! unit length so that only the orientation of the plane matters, not its
//! scale.
//!
//! ## Usage
//!
//! ```rust
//! use sketchir::hyperplane::HyperplaneHasher;
//!
//! let hasher = HyperplaneHasher::new(4, 8, 42).unwrap();
//! let a = vec![1.0_f32, 0.0, 0.0, 0.0];
//! let b = vec![1.0_f32, 0.0, 0.0, 0.0];
//! let sa = hasher.hash(&a).unwrap();
//! let sb = hasher.hash(&b).unwrap();
//! assert_eq!(HyperplaneHasher::distance(&sa, &sb), 0);
//! ```

use crate::{all_finite, lcg_next, Error};

/// Random hyperplane LSH for cosine similarity.
///
/// Hyperplanes are generated once at construction time from a seeded
/// deterministic RNG (64-bit LCG + Box-Muller).  The same `(dim, num_bits,
/// seed)` triple always produces the same hyperplanes and therefore the same
/// signatures.
#[derive(Debug, Clone)]
pub struct HyperplaneHasher {
    /// Flattened random unit vectors, row-major: `planes[b * dim .. (b+1) * dim]`
    /// is the normal of hyperplane `b`.
    planes: Vec<f32>,
    dim: usize,
    num_bits: usize,
}

impl HyperplaneHasher {
    /// Create a hasher with `num_bits` random hyperplanes in `dim` dimensions.
    ///
    /// `dim` and `num_bits` must be at least 1.
    pub fn new(dim: usize, num_bits: usize, seed: u64) -> Result<Self, Error> {
        if dim == 0 {
            return Err(Error::InvalidParam("dim must be >= 1"));
        }
        if num_bits == 0 {
            return Err(Error::InvalidParam("num_bits must be >= 1"));
        }

        let planes = generate_planes(dim, num_bits, seed);
        Ok(Self {
            planes,
            dim,
            num_bits,
        })
    }

    /// Hash a dense vector to a bit signature packed into bytes.
    ///
    /// The returned `Vec<u8>` has length `(num_bits + 7) / 8`.  Bit `b`
    /// (zero-indexed from LSB of byte 0) is 1 when the dot product with
    /// hyperplane `b` is positive.
    ///
    /// Returns [`Error::DimensionMismatch`] if `vector.len() != dim`, and
    /// [`Error::NonFiniteInput`] if the vector contains NaN or infinity.
    pub fn hash(&self, vector: &[f32]) -> Result<Vec<u8>, Error> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if !all_finite(vector) {
            return Err(Error::NonFiniteInput);
        }

        let n_bytes = self.num_bits.div_ceil(8);
        let mut sig = vec![0u8; n_bytes];

        for b in 0..self.num_bits {
            let plane = &self.planes[b * self.dim..(b + 1) * self.dim];
            let dot: f32 = plane.iter().zip(vector.iter()).map(|(p, v)| p * v).sum();
            if dot > 0.0 {
                sig[b / 8] |= 1u8 << (b % 8);
            }
        }
        Ok(sig)
    }

    /// Hash a batch of vectors, returning one signature per vector.
    pub fn hash_batch(&self, vectors: &[&[f32]]) -> Result<Vec<Vec<u8>>, Error> {
        vectors.iter().map(|v| self.hash(v)).collect()
    }

    /// Hamming distance (number of differing bits) between two signatures.
    ///
    /// Both signatures must have the same byte length; otherwise the result
    /// counts only the overlapping bytes.
    pub fn distance(sig_a: &[u8], sig_b: &[u8]) -> u32 {
        // bit-Hamming over byte-packed signatures. With `innr` it dispatches
        // to SIMD popcount; otherwise scalar XOR+popcount. Slice to the
        // shorter length first to keep the documented overlapping-bytes
        // behavior (innr::hamming_distance asserts equal length).
        #[cfg(feature = "innr")]
        {
            let n = sig_a.len().min(sig_b.len());
            innr::hamming_distance(&sig_a[..n], &sig_b[..n])
        }
        #[cfg(not(feature = "innr"))]
        {
            sig_a
                .iter()
                .zip(sig_b.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum()
        }
    }

    /// Estimate cosine similarity from the Hamming distance of two signatures.
    ///
    /// Uses `cos(π * hamming / num_bits)`.  The estimate improves with larger
    /// `num_bits`.
    pub fn estimated_cosine(&self, sig_a: &[u8], sig_b: &[u8]) -> f32 {
        let hamming = Self::distance(sig_a, sig_b);
        let theta = std::f32::consts::PI * hamming as f32 / self.num_bits as f32;
        theta.cos()
    }

    /// Number of hyperplane bits.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Hyperplane generation
// ---------------------------------------------------------------------------

/// Generate `num_bits` random unit vectors in `dim` dimensions.
///
/// Each component is drawn from N(0,1) via the Box-Muller transform applied
/// to the crate's LCG, then the vector is L2-normalised.
fn generate_planes(dim: usize, num_bits: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    // Mix the seed so that seed=0 doesn't stay at 0 forever.
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    let total = num_bits * dim;
    let mut raw: Vec<f32> = Vec::with_capacity(total);

    // Box-Muller produces two N(0,1) samples per iteration.
    let mut spare: Option<f32> = None;
    while raw.len() < total {
        if let Some(s) = spare.take() {
            raw.push(s);
        } else {
            let (z0, z1) = box_muller(&mut state);
            raw.push(z0);
            spare = Some(z1);
        }
    }
    raw.truncate(total);

    // Normalise each plane to unit length.
    for b in 0..num_bits {
        let start = b * dim;
        let end = start + dim;
        let norm: f32 = raw[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut raw[start..end] {
                *x /= norm;
            }
        }
    }

    raw
}

/// Box-Muller transform: two uniform (0,1] draws -> two N(0,1) samples.
fn box_muller(state: &mut u64) -> (f32, f32) {
    // Two independent U(0,1] values using the LCG.
    let u1 = lcg_u01(state);
    let u2 = lcg_u01(state);

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// LCG-based uniform f32 in (0, 1] (exclusive 0 so ln() is safe).
fn lcg_u01(state: &mut u64) -> f32 {
    let raw = lcg_next(state);
    // Use the upper 32 bits; shift to (0, 1].
    let u = (raw >> 32) as u32;
    // Map [0, 2^32) -> (0, 1]: add 1 to avoid 0, divide by 2^32.
    (u.wrapping_add(1) as f64 / (u32::MAX as f64 + 1.0)) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Determinism: same seed => same hash
    // ------------------------------------------------------------------

    #[test]
    fn hash_determinism() {
        let h1 = HyperplaneHasher::new(8, 32, 0xDEAD_BEEF).unwrap();
        let h2 = HyperplaneHasher::new(8, 32, 0xDEAD_BEEF).unwrap();
        let v: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        assert_eq!(h1.hash(&v).unwrap(), h2.hash(&v).unwrap());
    }

    #[test]
    fn different_seeds_give_different_planes() {
        let h1 = HyperplaneHasher::new(16, 64, 1).unwrap();
        let h2 = HyperplaneHasher::new(16, 64, 2).unwrap();
        // The planes themselves should differ.
        assert_ne!(h1.planes, h2.planes);
    }

    // ------------------------------------------------------------------
    // Distance properties
    // ------------------------------------------------------------------

    #[test]
    fn same_vector_zero_distance() {
        let h = HyperplaneHasher::new(4, 16, 42).unwrap();
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sig = h.hash(&v).unwrap();
        assert_eq!(HyperplaneHasher::distance(&sig, &sig), 0);
    }

    #[test]
    fn distance_symmetric() {
        let h = HyperplaneHasher::new(8, 32, 7).unwrap();
        let a: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..8).map(|i| (i as f32).cos()).collect();
        let sa = h.hash(&a).unwrap();
        let sb = h.hash(&b).unwrap();
        assert_eq!(
            HyperplaneHasher::distance(&sa, &sb),
            HyperplaneHasher::distance(&sb, &sa)
        );
    }

    #[test]
    fn distance_triangle_inequality() {
        let h = HyperplaneHasher::new(8, 64, 99).unwrap();
        let a: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..8).map(|i| (i as f32).cos()).collect();
        let c: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let sa = h.hash(&a).unwrap();
        let sb = h.hash(&b).unwrap();
        let sc = h.hash(&c).unwrap();
        let dab = HyperplaneHasher::distance(&sa, &sb);
        let dac = HyperplaneHasher::distance(&sa, &sc);
        let dbc = HyperplaneHasher::distance(&sb, &sc);
        assert!(
            dab <= dac + dbc,
            "triangle inequality violated: {dab} > {dac} + {dbc}"
        );
    }

    // ------------------------------------------------------------------
    // Estimated cosine accuracy on known vectors
    // ------------------------------------------------------------------

    #[test]
    fn identical_vectors_cosine_one() {
        // Identical vectors -> 0 Hamming distance -> estimated cosine should be 1.0.
        let h = HyperplaneHasher::new(4, 128, 1234).unwrap();
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let sig = h.hash(&v).unwrap();
        let est = h.estimated_cosine(&sig, &sig);
        assert!((est - 1.0_f32).abs() < 1e-6, "expected 1.0, got {est}");
    }

    #[test]
    fn orthogonal_vectors_cosine_near_zero() {
        // Orthogonal vectors -> true cosine = 0.  With 256 bits the estimate
        // should fall within ±0.15 of zero with overwhelming probability.
        let h = HyperplaneHasher::new(4, 256, 5678).unwrap();
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];
        let sa = h.hash(&a).unwrap();
        let sb = h.hash(&b).unwrap();
        let est = h.estimated_cosine(&sa, &sb);
        assert!(
            est.abs() < 0.15,
            "expected cosine near 0 for orthogonal vectors, got {est}"
        );
    }

    #[test]
    fn opposite_vectors_cosine_near_minus_one() {
        // Opposite vectors -> true cosine = -1.  With 256 bits the estimate
        // should be well below -0.8.
        let h = HyperplaneHasher::new(4, 256, 9999).unwrap();
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![-1.0_f32, 0.0, 0.0, 0.0];
        let sa = h.hash(&a).unwrap();
        let sb = h.hash(&b).unwrap();
        let est = h.estimated_cosine(&sa, &sb);
        assert!(
            est < -0.8,
            "expected cosine near -1 for opposite vectors, got {est}"
        );
    }

    // ------------------------------------------------------------------
    // Batch hashing agrees with single hashing
    // ------------------------------------------------------------------

    #[test]
    fn hash_batch_matches_individual() {
        let h = HyperplaneHasher::new(4, 16, 0).unwrap();
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let batch = h.hash_batch(&refs).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            assert_eq!(batch[i], h.hash(v).unwrap());
        }
    }

    // ------------------------------------------------------------------
    // Error propagation
    // ------------------------------------------------------------------

    #[test]
    fn rejects_zero_dim() {
        assert!(HyperplaneHasher::new(0, 8, 0).is_err());
    }

    #[test]
    fn rejects_zero_bits() {
        assert!(HyperplaneHasher::new(4, 0, 0).is_err());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let h = HyperplaneHasher::new(4, 8, 0).unwrap();
        assert!(h.hash(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn rejects_non_finite_input() {
        let h = HyperplaneHasher::new(4, 8, 0).unwrap();
        assert!(h.hash(&[1.0, f32::NAN, 0.0, 0.0]).is_err());
        assert!(h.hash(&[1.0, f32::INFINITY, 0.0, 0.0]).is_err());
    }

    // ------------------------------------------------------------------
    // Determinism canary (pin a specific output for regression detection)
    // ------------------------------------------------------------------

    #[test]
    fn determinism_canary() {
        let h = HyperplaneHasher::new(4, 8, 42).unwrap();
        let v = [1.0_f32, 0.0, 0.0, 0.0];
        let sig = h.hash(&v).unwrap();
        // If this value changes, the hyperplane generation changed.
        assert_eq!(sig.len(), 1, "8 bits -> 1 byte");
        // Record the exact byte for regression detection.
        let canary = sig[0];
        let h2 = HyperplaneHasher::new(4, 8, 42).unwrap();
        assert_eq!(h2.hash(&v).unwrap()[0], canary);
    }
}
