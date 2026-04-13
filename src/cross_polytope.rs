//! Cross-polytope LSH for angular distance on dense `f32` vectors.
//!
//! Maps each vector to a vertex of the cross-polytope (the L1 unit ball) by
//! applying a random rotation and selecting the coordinate with maximum absolute
//! value. This achieves optimal sensitivity for angular distance (Andoni et al.,
//! NeurIPS 2015).
//!
//! For dimensions > 64, uses a randomized Hadamard transform (HD^3 construction)
//! in O(d log d) time instead of dense rotation (O(d^2)).
//!
//! ## Usage
//!
//! ```rust
//! use sketchir::cross_polytope::CrossPolytopeHasher;
//!
//! let hasher = CrossPolytopeHasher::new(8, 42).unwrap();
//! let v = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! let bucket = hasher.hash(&v).unwrap();
//! // bucket is in 0..2*dim (one of 2d cross-polytope vertices)
//! assert!(bucket < 16);
//! ```
//!
//! ## References
//!
//! - Andoni, Indyk, Laarhoven, Razenshteyn, Schmidt (2015). "Practical and Optimal
//!   LSH for Angular Distance." NeurIPS.

use crate::{all_finite, lcg_next, Error};

/// Cross-polytope LSH hasher for angular distance.
///
/// Maps vectors to vertices of the cross-polytope (L1 unit ball) via random
/// rotation. Returns a bucket ID in `0..2*dim`.
///
/// Supports multiple independent rotations via `hash_multi` for multi-table LSH.
#[derive(Debug, Clone)]
pub struct CrossPolytopeHasher {
    dim: usize,
    /// Rotation data. For d <= 64: a flat d×d row-major orthogonal matrix.
    /// For d > 64: [padded_d, num_rounds, sign_vectors...] (Hadamard format).
    rotation: Vec<f32>,
}

impl CrossPolytopeHasher {
    /// Create a hasher with a single random rotation for `dim` dimensions.
    pub fn new(dim: usize, seed: u64) -> Result<Self, Error> {
        if dim == 0 {
            return Err(Error::InvalidParam("dim must be >= 1"));
        }

        let rotation = generate_rotation(dim, seed);
        Ok(Self { dim, rotation })
    }

    /// Hash a vector to a cross-polytope vertex (bucket ID).
    ///
    /// Returns a value in `0..2*dim`. The encoding is:
    /// `2 * argmax_i(|Rv|_i) + (1 if Rv[argmax] < 0 else 0)`.
    pub fn hash(&self, vector: &[f32]) -> Result<u32, Error> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if !all_finite(vector) {
            return Err(Error::NonFiniteInput);
        }

        let rotated = apply_rotation(vector, &self.rotation, self.dim);
        Ok(cross_polytope_vertex(&rotated))
    }

    /// Return the top-k cross-polytope vertices sorted by coordinate magnitude
    /// (descending). Useful for multiprobe: the first entry is the primary hash,
    /// subsequent entries are the nearest neighbor vertices to probe.
    pub fn hash_ranked(&self, vector: &[f32], k: usize) -> Result<Vec<u32>, Error> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if !all_finite(vector) {
            return Err(Error::NonFiniteInput);
        }

        let rotated = apply_rotation(vector, &self.rotation, self.dim);

        let mut ranked: Vec<(usize, f32, bool)> = rotated
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs(), v < 0.0))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        Ok(ranked
            .iter()
            .take(k)
            .map(|&(idx, _, is_neg)| {
                if is_neg {
                    (idx as u32) * 2 + 1
                } else {
                    (idx as u32) * 2
                }
            })
            .collect())
    }

    /// Hash a batch of vectors.
    pub fn hash_batch(&self, vectors: &[&[f32]]) -> Result<Vec<u32>, Error> {
        vectors.iter().map(|v| self.hash(v)).collect()
    }

    /// Number of possible buckets (2 * dim).
    pub fn num_buckets(&self) -> usize {
        self.dim * 2
    }

    /// Input dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Create multiple independent hashers (one per table) from a base seed.
///
/// Each hasher gets a distinct rotation derived from `base_seed + table_index`.
pub fn multi_hasher(
    dim: usize,
    num_tables: usize,
    base_seed: u64,
) -> Result<Vec<CrossPolytopeHasher>, Error> {
    (0..num_tables)
        .map(|i| CrossPolytopeHasher::new(dim, base_seed.wrapping_add(i as u64)))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Map a rotated vector to a cross-polytope vertex index.
fn cross_polytope_vertex(rotated: &[f32]) -> u32 {
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    let mut max_neg = false;

    for (i, &v) in rotated.iter().enumerate() {
        let abs_v = v.abs();
        if abs_v > max_abs {
            max_abs = abs_v;
            max_idx = i;
            max_neg = v < 0.0;
        }
    }

    (max_idx as u32) * 2 + u32::from(max_neg)
}

/// Generate a rotation (dense or Hadamard depending on dimension).
fn generate_rotation(dim: usize, seed: u64) -> Vec<f32> {
    if dim <= 64 {
        generate_dense_rotation(dim, seed)
    } else {
        generate_hadamard_rotation(dim, seed)
    }
}

/// Apply rotation (dispatches between dense and Hadamard).
fn apply_rotation(vector: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    if rotation.len() == dim * dim {
        apply_dense_rotation(vector, rotation, dim)
    } else {
        apply_hadamard_rotation(vector, rotation, dim)
    }
}

/// Dense O(d^2) rotation via Gram-Schmidt QR.
fn generate_dense_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    // Mix seed.
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    let mut matrix = vec![0.0f32; dim * dim];
    let mut spare: Option<f32> = None;
    for val in &mut matrix {
        if let Some(s) = spare.take() {
            *val = s;
        } else {
            let (z0, z1) = box_muller(&mut state);
            *val = z0;
            spare = Some(z1);
        }
    }

    // Gram-Schmidt (column-major).
    for i in 0..dim {
        let mut norm = 0.0f32;
        for row in 0..dim {
            norm += matrix[row * dim + i] * matrix[row * dim + i];
        }
        let norm = norm.sqrt();
        if norm > 1e-10 {
            for row in 0..dim {
                matrix[row * dim + i] /= norm;
            }
        }
        for j in (i + 1)..dim {
            let mut dot = 0.0f32;
            for row in 0..dim {
                dot += matrix[row * dim + i] * matrix[row * dim + j];
            }
            for row in 0..dim {
                matrix[row * dim + j] -= dot * matrix[row * dim + i];
            }
        }
    }

    // Transpose to row-major.
    let mut row_major = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            row_major[i * dim + j] = matrix[j * dim + i];
        }
    }
    row_major
}

/// HD^3 randomized Hadamard rotation.
fn generate_hadamard_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let padded = dim.next_power_of_two();
    let num_rounds: usize = 3;
    let mut state = seed;
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    let mut result = Vec::with_capacity(2 + num_rounds * padded);
    result.push(padded as f32);
    result.push(num_rounds as f32);

    for _ in 0..num_rounds {
        for _ in 0..padded {
            let bit = (lcg_next(&mut state) >> 63) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
    }

    result
}

/// Dense matrix-vector multiply (row-major).
fn apply_dense_rotation(vector: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for (i, out) in result.iter_mut().enumerate() {
        let row = &rotation[i * dim..(i + 1) * dim];
        *out = row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum();
    }
    result
}

/// Hadamard rotation: D * H repeated num_rounds times.
fn apply_hadamard_rotation(vector: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let padded = rotation[0] as usize;
    let num_rounds = rotation[1] as usize;

    let mut buf = vec![0.0f32; padded];
    buf[..dim].copy_from_slice(vector);

    for round in 0..num_rounds {
        let signs_offset = 2 + round * padded;
        let signs = &rotation[signs_offset..signs_offset + padded];
        for (v, &s) in buf.iter_mut().zip(signs.iter()) {
            *v *= s;
        }
        walsh_hadamard_transform(&mut buf);
    }

    buf.truncate(dim);
    buf
}

/// In-place Walsh-Hadamard transform (normalized by 1/sqrt(n)).
fn walsh_hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }

    let scale = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
}

/// Box-Muller transform.
fn box_muller(state: &mut u64) -> (f32, f32) {
    let u1 = lcg_u01(state);
    let u2 = lcg_u01(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// LCG-based uniform f32 in (0, 1].
fn lcg_u01(state: &mut u64) -> f32 {
    let raw = lcg_next(state);
    let u = (raw >> 32) as u32;
    (u.wrapping_add(1) as f64 / (u32::MAX as f64 + 1.0)) as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_determinism() {
        let h1 = CrossPolytopeHasher::new(8, 42).unwrap();
        let h2 = CrossPolytopeHasher::new(8, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        assert_eq!(h1.hash(&v).unwrap(), h2.hash(&v).unwrap());
    }

    #[test]
    fn different_seeds_give_different_rotations() {
        let h1 = CrossPolytopeHasher::new(16, 1).unwrap();
        let h2 = CrossPolytopeHasher::new(16, 2).unwrap();
        assert_ne!(h1.rotation, h2.rotation);
    }

    #[test]
    fn bucket_in_range() {
        let h = CrossPolytopeHasher::new(8, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let bucket = h.hash(&v).unwrap();
        assert!(bucket < 16, "bucket {} out of range for dim=8", bucket);
    }

    #[test]
    fn similar_vectors_same_bucket() {
        let h = CrossPolytopeHasher::new(32, 42).unwrap();
        let v1: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..32).map(|i| i as f32 + 0.001).collect();
        assert_eq!(h.hash(&v1).unwrap(), h.hash(&v2).unwrap());
    }

    #[test]
    fn hash_ranked_returns_sorted_by_magnitude() {
        let h = CrossPolytopeHasher::new(8, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let ranked = h.hash_ranked(&v, 4).unwrap();
        assert_eq!(ranked.len(), 4);
        // First element should be the primary hash.
        assert_eq!(ranked[0], h.hash(&v).unwrap());
    }

    #[test]
    fn hash_batch_matches_individual() {
        let h = CrossPolytopeHasher::new(4, 42).unwrap();
        let vecs: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let batch = h.hash_batch(&refs).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            assert_eq!(batch[i], h.hash(v).unwrap());
        }
    }

    #[test]
    fn multi_hasher_produces_independent_hashes() {
        let hashers = multi_hasher(8, 4, 42).unwrap();
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let buckets: Vec<u32> = hashers.iter().map(|h| h.hash(&v).unwrap()).collect();
        // Not all should be the same (probabilistically).
        // With 4 independent rotations on an 8d vector, at least 2 should differ.
        let unique: std::collections::HashSet<u32> = buckets.iter().copied().collect();
        assert!(
            unique.len() > 1,
            "4 independent hashers should produce some different buckets"
        );
    }

    #[test]
    fn hadamard_preserves_norm_approx() {
        // d=128 triggers Hadamard path.
        let h = CrossPolytopeHasher::new(128, 42).unwrap();
        let v: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        let rotated = apply_rotation(&v, &h.rotation, 128);
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        let ratio = norm_after / norm_before;
        assert!(
            (ratio - 1.0).abs() < 0.3,
            "Hadamard should approximately preserve norm: ratio={}",
            ratio
        );
    }

    #[test]
    fn dense_rotation_is_orthogonal() {
        let rot = generate_dense_rotation(8, 42);
        for i in 0..8 {
            for j in 0..8 {
                let dot: f32 = (0..8).map(|k| rot[i * 8 + k] * rot[j * 8 + k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "R*R^T[{},{}] = {} (expected {})",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn rejects_zero_dim() {
        assert!(CrossPolytopeHasher::new(0, 42).is_err());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let h = CrossPolytopeHasher::new(4, 42).unwrap();
        assert!(h.hash(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn rejects_non_finite() {
        let h = CrossPolytopeHasher::new(4, 42).unwrap();
        assert!(h.hash(&[1.0, f32::NAN, 0.0, 0.0]).is_err());
    }

    #[test]
    fn vertex_encoding() {
        // [3, -1, 2] -> argmax_abs=0 (positive) -> bucket 0
        assert_eq!(cross_polytope_vertex(&[3.0, -1.0, 2.0]), 0);
        // [1, -5, 2] -> argmax_abs=1 (negative) -> bucket 3
        assert_eq!(cross_polytope_vertex(&[1.0, -5.0, 2.0]), 3);
        // [0, 0, 7] -> argmax_abs=2 (positive) -> bucket 4
        assert_eq!(cross_polytope_vertex(&[0.0, 0.0, 7.0]), 4);
    }
}
