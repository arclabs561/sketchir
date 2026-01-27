//! Dense-vector SimHash and an embedding LSH index.
//!
//! This is a small, deterministic helper for producing candidate sets from embedding vectors:
//! - map an embedding to a 64-bit SimHash fingerprint using fixed random hyperplanes
//! - bucket by fingerprint (plus Hamming-1 neighbors) to get a candidate set
//!
//! Scope: primitives for *candidate generation*. Downstream policy (thresholding, scoring,
//! clustering) belongs in the caller.

use std::collections::{HashMap, HashSet};

use crate::lsh::Error;

/// Deterministic SimHash for dense vectors using fixed random hyperplanes.
#[derive(Debug)]
pub struct DenseSimHash {
    embedding_dim: usize,
    num_bits: usize,
    hyperplanes: Vec<Vec<f32>>,
}

impl DenseSimHash {
    /// Create a new DenseSimHash generator.
    ///
    /// `num_bits` is capped at 64.
    pub fn new(embedding_dim: usize, num_bits: usize) -> Result<Self, Error> {
        if embedding_dim == 0 {
            return Err(Error::InvalidParam("embedding_dim must be >= 1"));
        }
        if num_bits == 0 {
            return Err(Error::InvalidParam("num_bits must be >= 1"));
        }

        // Deterministic hyperplanes (LCG).
        let mut hyperplanes = Vec::with_capacity(num_bits.min(64));
        let mut rng_state = 0x12345678u64;
        for _ in 0..num_bits.min(64) {
            let mut plane = Vec::with_capacity(embedding_dim);
            for _ in 0..embedding_dim {
                rng_state = rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                let val = ((rng_state >> 16) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                plane.push(val);
            }
            hyperplanes.push(plane);
        }

        Ok(Self {
            embedding_dim,
            num_bits: num_bits.min(64),
            hyperplanes,
        })
    }

    /// Compute a 64-bit SimHash fingerprint for an embedding.
    pub fn fingerprint(&self, embedding: &[f32]) -> Result<u64, Error> {
        if embedding.len() != self.embedding_dim {
            return Err(Error::DimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }

        let mut hash = 0u64;
        for (i, plane) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = plane.iter().zip(embedding.iter()).map(|(a, b)| a * b).sum();
            if dot > 0.0 {
                hash |= 1u64 << (i % 64);
            }
        }
        Ok(hash)
    }

    /// Number of bits used in the fingerprint.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Embedding dimension expected by this generator.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// A minimal embedding LSH index using DenseSimHash fingerprints.
#[derive(Debug)]
pub struct DenseSimHashLSH {
    simhash: DenseSimHash,
    buckets: HashMap<u64, Vec<usize>>,
    items: Vec<(String, Vec<f32>)>,
}

impl DenseSimHashLSH {
    /// Create a new index for embeddings of `embedding_dim`.
    pub fn new(embedding_dim: usize, num_bits: usize) -> Result<Self, Error> {
        Ok(Self {
            simhash: DenseSimHash::new(embedding_dim, num_bits)?,
            buckets: HashMap::new(),
            items: Vec::new(),
        })
    }

    /// Insert an embedding vector and return its assigned index.
    pub fn insert(&mut self, id: impl Into<String>, embedding: Vec<f32>) -> Result<usize, Error> {
        if embedding.len() != self.simhash.embedding_dim() {
            return Err(Error::DimensionMismatch {
                expected: self.simhash.embedding_dim(),
                got: embedding.len(),
            });
        }

        let idx = self.items.len();
        let fp = self.simhash.fingerprint(&embedding)?;
        self.buckets.entry(fp).or_default().push(idx);
        self.items.push((id.into(), embedding));
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

        for bit in 0..self.simhash.num_bits().min(64) {
            let neighbor = fp ^ (1u64 << bit);
            if let Some(indices) = self.buckets.get(&neighbor) {
                candidates.extend(indices.iter().copied());
            }
        }

        Ok(candidates.into_iter().collect())
    }

    /// Get item by index.
    pub fn get(&self, idx: usize) -> Option<(&str, &[f32])> {
        self.items
            .get(idx)
            .map(|(id, emb)| (id.as_str(), emb.as_slice()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_query_returns_candidates() {
        let mut lsh = DenseSimHashLSH::new(8, 64).unwrap();
        let v1: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let v2: Vec<f32> = (0..8).map(|i| (i as f32).sin() + 0.01).collect();
        let v3: Vec<f32> = (0..8).map(|i| (i as f32).cos()).collect();

        lsh.insert("1", v1.clone()).unwrap();
        lsh.insert("2", v2).unwrap();
        lsh.insert("3", v3).unwrap();

        let candidates = lsh.query(&v1).unwrap();
        assert!(!candidates.is_empty());
    }
}
