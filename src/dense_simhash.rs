//! Dense-vector SimHash and an embedding LSH index.
//!
//! Produces candidate sets from embedding vectors:
//! - map an embedding to a 64-bit SimHash fingerprint using fixed random hyperplanes
//! - bucket by fingerprint (plus Hamming-1 neighbors) to get a candidate set
//!
//! Supports incremental insertion (no build step). For batch indexing with multi-table
//! hashing, see [`LSHIndex`](crate::LSHIndex).
//!
//! Scope: primitives for *candidate generation*. Downstream policy (thresholding, scoring,
//! clustering) belongs in the caller.

use std::collections::{HashMap, HashSet};

use crate::hyperplane::box_muller;
use crate::{all_finite, Error};

/// Deterministic SimHash for dense vectors using fixed random hyperplanes.
#[derive(Debug)]
pub(crate) struct DenseSimHash {
    embedding_dim: usize,
    num_bits: usize,
    hyperplanes: Vec<Vec<f32>>,
}

impl DenseSimHash {
    /// Create a new DenseSimHash generator.
    ///
    /// `num_bits` must be in `[1, 64]`.
    pub(crate) fn new(embedding_dim: usize, num_bits: usize) -> Result<Self, Error> {
        if embedding_dim == 0 {
            return Err(Error::InvalidParam("embedding_dim must be >= 1"));
        }
        if num_bits == 0 || num_bits > 64 {
            return Err(Error::InvalidParam("num_bits must be in [1, 64]"));
        }

        // Deterministic Gaussian hyperplanes (LCG + Box-Muller). Gaussian
        // normals make P[bit agrees] = 1 - theta/pi for any input direction;
        // uniform-cube normals only give that for isotropic inputs.
        let mut hyperplanes = Vec::with_capacity(num_bits);
        let mut rng_state = 0x12345678u64;
        for _ in 0..num_bits {
            let mut plane = Vec::with_capacity(embedding_dim);
            for _ in 0..embedding_dim {
                plane.push(box_muller(&mut rng_state).0);
            }
            hyperplanes.push(plane);
        }

        Ok(Self {
            embedding_dim,
            num_bits,
            hyperplanes,
        })
    }

    /// Compute a 64-bit SimHash fingerprint for an embedding.
    pub(crate) fn fingerprint(&self, embedding: &[f32]) -> Result<u64, Error> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        if !all_finite(embedding) {
            return Err(Error::NonFiniteInput);
        }

        let mut hash = 0u64;
        for (i, plane) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = plane.iter().zip(embedding.iter()).map(|(a, b)| a * b).sum();
            if dot > 0.0 {
                hash |= 1u64 << i;
            }
        }
        Ok(hash)
    }

    /// Number of bits used in the fingerprint.
    pub(crate) fn num_bits(&self) -> usize {
        self.num_bits
    }
}

/// A minimal embedding LSH index using DenseSimHash fingerprints.
///
/// Supports incremental insertion (no build step). Queries return exact-bucket collisions
/// plus Hamming-distance-1 neighbors. For batch indexing with multi-table hashing,
/// see [`LSHIndex`](crate::LSHIndex).
#[derive(Debug)]
pub struct DenseSimHashLSH {
    simhash: DenseSimHash,
    buckets: HashMap<u64, Vec<usize>>,
    ids: Vec<String>,
}

impl DenseSimHashLSH {
    /// Create a new index for embeddings of `embedding_dim`.
    ///
    /// `num_bits` must be in `[1, 64]`.
    pub fn new(embedding_dim: usize, num_bits: usize) -> Result<Self, Error> {
        Ok(Self {
            simhash: DenseSimHash::new(embedding_dim, num_bits)?,
            buckets: HashMap::new(),
            ids: Vec::new(),
        })
    }

    /// Insert an embedding vector and return its assigned index.
    pub fn insert(&mut self, id: impl Into<String>, embedding: &[f32]) -> Result<usize, Error> {
        let idx = self.ids.len();
        let fp = self.simhash.fingerprint(embedding)?;
        self.buckets.entry(fp).or_default().push(idx);
        self.ids.push(id.into());
        Ok(idx)
    }

    /// Query for candidates similar to an embedding.
    ///
    /// Returns exact-bucket collisions plus Hamming-distance-1 neighbors.
    pub fn query(&self, embedding: &[f32]) -> Result<Vec<usize>, Error> {
        let fp = self.simhash.fingerprint(embedding)?;

        let mut candidates: HashSet<usize> = HashSet::new();
        if let Some(indices) = self.buckets.get(&fp) {
            candidates.extend(indices.iter().copied());
        }

        for bit in 0..self.simhash.num_bits() {
            let neighbor = fp ^ (1u64 << bit);
            if let Some(indices) = self.buckets.get(&neighbor) {
                candidates.extend(indices.iter().copied());
            }
        }

        let mut v: Vec<usize> = candidates.into_iter().collect();
        v.sort_unstable();
        Ok(v)
    }

    /// Get the external ID for an item by index.
    pub fn get_id(&self, idx: usize) -> Option<&str> {
        self.ids.get(idx).map(|s| s.as_str())
    }

    /// Number of items indexed.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_query_returns_candidates() {
        let mut lsh = DenseSimHashLSH::new(8, 64).unwrap();
        let v1: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let v2: Vec<f32> = (0..8).map(|i| (i as f32).sin() + 0.01).collect();
        let v3: Vec<f32> = (0..8).map(|i| (i as f32).cos()).collect();

        lsh.insert("1", &v1).unwrap();
        lsh.insert("2", &v2).unwrap();
        lsh.insert("3", &v3).unwrap();

        let candidates = lsh.query(&v1).unwrap();
        assert!(!candidates.is_empty());
    }

    #[test]
    fn rejects_num_bits_over_64() {
        assert!(DenseSimHashLSH::new(8, 65).is_err());
        assert!(DenseSimHashLSH::new(8, 128).is_err());
    }

    #[test]
    fn rejects_zero_num_bits() {
        assert!(DenseSimHashLSH::new(8, 0).is_err());
    }

    #[test]
    fn simhash_bit_agreement_matches_one_minus_theta_over_pi() {
        // Charikar 2002: for Gaussian hyperplanes P[bit agrees] = 1 - theta/pi.
        // u = e0, v = (e0 + 2 e1) / sqrt(5): theta = atan(2) = 63.43 deg, so
        // the rate is 0.6476 (a uniform-cube plane gives 0.625 here). Pool
        // many independent plane sets by varying the dimension.
        let theta = 2.0_f64.atan();
        let expected = 1.0 - theta / std::f64::consts::PI;
        let (mut agree, mut total) = (0usize, 0usize);
        for dim in 2..=300usize {
            let dsh = DenseSimHash::new(dim, 64).unwrap();
            let mut u = vec![0.0f32; dim];
            let mut v = vec![0.0f32; dim];
            u[0] = 1.0;
            v[0] = 1.0;
            v[1] = 2.0;
            let diff = dsh.fingerprint(&u).unwrap() ^ dsh.fingerprint(&v).unwrap();
            agree += 64 - diff.count_ones() as usize;
            total += 64;
        }
        let rate = agree as f64 / total as f64;
        // sd = sqrt(0.65 * 0.35 / 19000) = 0.0035
        assert!((rate - expected).abs() < 0.012, "rate {rate} vs {expected}");
    }

    // DETERMINISM CANARY
    #[test]
    fn dense_simhash_fingerprint_determinism() {
        let dsh = DenseSimHash::new(4, 16).unwrap();
        let fp = dsh.fingerprint(&[1.0, -0.5, 0.3, 0.8]).unwrap();
        assert_eq!(fp, 33302);
    }
}
