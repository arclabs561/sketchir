//! LSH-style indexing helpers.
//!
//! This module is intentionally small and deterministic. It provides:
//! - banding-based LSH for [`MinHashSignature`]
//! - bit-sampling LSH for [`SimHashFingerprint`]
//! - random-projection LSH for dense vectors (cosine-ish)

use std::collections::{HashMap, HashSet};

use crate::minhash::MinHashSignature;
use crate::simhash::SimHashFingerprint;

/// Errors for LSH indexes.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A parameter is out of range or inconsistent.
    #[error("invalid parameter: {0}")]
    InvalidParam(&'static str),
    /// Dimension mismatch between expected and provided vectors.
    #[error("dimension mismatch (expected {expected}, got {got})")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual provided dimension.
        got: usize,
    },
    /// The index has no inserted items.
    #[error("empty index")]
    EmptyIndex,
    /// The index has not been built yet.
    #[error("index not built")]
    NotBuilt,
    /// Attempted to add after the index was built.
    #[error("cannot add after build")]
    AddAfterBuild,
}

/// LSH index using MinHash banding (near-duplicate detection).
#[derive(Debug)]
pub struct MinHashLSH {
    bands: usize,
    rows_per_band: usize,
    buckets: Vec<HashMap<u64, Vec<usize>>>,
    signatures: Vec<MinHashSignature>,
}

impl MinHashLSH {
    /// Create a MinHash banding LSH index.
    pub fn new(bands: usize, rows_per_band: usize) -> Result<Self, Error> {
        if bands == 0 || rows_per_band == 0 {
            return Err(Error::InvalidParam("bands and rows_per_band must be >= 1"));
        }
        Ok(Self {
            bands,
            rows_per_band,
            buckets: (0..bands).map(|_| HashMap::new()).collect(),
            signatures: Vec::new(),
        })
    }

    /// Insert a signature and return its assigned document id.
    pub fn insert(&mut self, signature: MinHashSignature) -> usize {
        let doc_id = self.signatures.len();
        for (band_idx, chunk) in signature.values.chunks(self.rows_per_band).enumerate() {
            if band_idx >= self.bands {
                break;
            }
            let band_hash = hash_band(chunk);
            self.buckets[band_idx]
                .entry(band_hash)
                .or_default()
                .push(doc_id);
        }
        self.signatures.push(signature);
        doc_id
    }

    /// Return candidate document IDs that share at least one band bucket.
    pub fn query(&self, signature: &MinHashSignature) -> Vec<usize> {
        let mut candidates: HashSet<usize> = HashSet::new();
        for (band_idx, chunk) in signature.values.chunks(self.rows_per_band).enumerate() {
            if band_idx >= self.bands {
                break;
            }
            let band_hash = hash_band(chunk);
            if let Some(docs) = self.buckets[band_idx].get(&band_hash) {
                candidates.extend(docs.iter().copied());
            }
        }
        let mut v: Vec<usize> = candidates.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Query and return candidates with estimated Jaccard similarity.
    pub fn query_with_similarity(&self, signature: &MinHashSignature) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = self
            .query(signature)
            .into_iter()
            .map(|id| (id, signature.jaccard(&self.signatures[id])))
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        results
    }
}

fn hash_band(values: &[u64]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for v in values {
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// LSH index for 64-bit SimHash fingerprints using bit sampling.
#[derive(Debug)]
pub struct SimHashLSH {
    num_tables: usize,
    bits_per_table: usize,
    bit_indices: Vec<Vec<usize>>,
    tables: Vec<HashMap<u64, Vec<usize>>>,
    fingerprints: Vec<SimHashFingerprint>,
}

impl SimHashLSH {
    /// Create a SimHash bit-sampling LSH index.
    pub fn new(num_tables: usize, bits_per_table: usize) -> Result<Self, Error> {
        if num_tables == 0 || bits_per_table == 0 || bits_per_table > 64 {
            return Err(Error::InvalidParam(
                "num_tables must be >=1 and bits_per_table must be in [1,64]",
            ));
        }

        // Deterministic bit selection (LCG).
        let mut rng_state = 12345u64;
        let mut bit_indices = Vec::with_capacity(num_tables);
        for _ in 0..num_tables {
            let mut idxs = Vec::with_capacity(bits_per_table);
            while idxs.len() < bits_per_table {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bit = (rng_state as usize) % 64;
                if !idxs.contains(&bit) {
                    idxs.push(bit);
                }
            }
            idxs.sort_unstable();
            bit_indices.push(idxs);
        }

        Ok(Self {
            num_tables,
            bits_per_table,
            bit_indices,
            tables: (0..num_tables).map(|_| HashMap::new()).collect(),
            fingerprints: Vec::new(),
        })
    }

    /// Insert a fingerprint and return its assigned document id.
    pub fn insert(&mut self, fp: SimHashFingerprint) -> usize {
        let doc_id = self.fingerprints.len();
        for (t, idxs) in self.bit_indices.iter().enumerate() {
            let key = extract_bits(fp.0, idxs);
            self.tables[t].entry(key).or_default().push(doc_id);
        }
        self.fingerprints.push(fp);
        doc_id
    }

    /// Return candidate document IDs that collide in any table bucket.
    pub fn query(&self, fp: SimHashFingerprint) -> Vec<usize> {
        let mut candidates: HashSet<usize> = HashSet::new();
        for (t, idxs) in self.bit_indices.iter().enumerate() {
            let key = extract_bits(fp.0, idxs);
            if let Some(docs) = self.tables[t].get(&key) {
                candidates.extend(docs.iter().copied());
            }
        }
        let mut v: Vec<usize> = candidates.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Query and return candidates with Hamming distance to the query fingerprint.
    pub fn query_with_distance(&self, fp: SimHashFingerprint) -> Vec<(usize, u32)> {
        let mut results: Vec<(usize, u32)> = self
            .query(fp)
            .into_iter()
            .map(|id| (id, fp.hamming_distance(&self.fingerprints[id])))
            .collect();
        results.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        results
    }

    /// Number of hash tables.
    pub fn num_tables(&self) -> usize {
        self.num_tables
    }

    /// Number of sampled bits per table.
    pub fn bits_per_table(&self) -> usize {
        self.bits_per_table
    }
}

fn extract_bits(v: u64, bit_indices: &[usize]) -> u64 {
    let mut key = 0u64;
    for (i, &bit_idx) in bit_indices.iter().enumerate() {
        if (v >> bit_idx) & 1 == 1 {
            key |= 1u64 << i;
        }
    }
    key
}

/// Random-projection LSH over dense vectors.
#[derive(Debug, Clone)]
pub struct LSHParams {
    /// Number of hash tables.
    pub num_tables: usize,
    /// Number of random hyperplanes per table.
    pub num_functions: usize,
    /// Target number of candidates to verify (not currently enforced; kept for API shape).
    pub num_candidates: usize,
}

impl Default for LSHParams {
    fn default() -> Self {
        Self {
            num_tables: 10,
            num_functions: 10,
            num_candidates: 100,
        }
    }
}

/// A minimal random-projection LSH index.
#[derive(Debug)]
pub struct LSHIndex {
    dimension: usize,
    params: LSHParams,
    built: bool,
    vectors: Vec<f32>, // SoA [n * d]
    num_vectors: usize,
    hash_functions: Vec<Vec<f32>>,
    hash_tables: Vec<HashMap<u64, Vec<u32>>>,
}

impl LSHIndex {
    /// Create a new index for vectors of length `dimension`.
    pub fn new(dimension: usize, params: LSHParams) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::InvalidParam("dimension must be >= 1"));
        }
        if params.num_tables == 0 || params.num_functions == 0 {
            return Err(Error::InvalidParam(
                "num_tables and num_functions must be >= 1",
            ));
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            hash_functions: Vec::new(),
            hash_tables: Vec::new(),
        })
    }

    /// Add a vector (before calling [`Self::build`]).
    pub fn add(&mut self, _doc_id: u32, vector: Vec<f32>) -> Result<(), Error> {
        if self.built {
            return Err(Error::AddAfterBuild);
        }
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        self.vectors.extend_from_slice(&vector);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build hash tables for search.
    pub fn build(&mut self) -> Result<(), Error> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(Error::EmptyIndex);
        }

        // Deterministic hyperplanes: stable across runs for the same params.
        fn next_f32(state: &mut u64) -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (*state >> 16) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        }

        let mut rng_state = 0x9E3779B97F4A7C15u64
            ^ (self.dimension as u64)
            ^ ((self.params.num_tables as u64) << 32)
            ^ (self.params.num_functions as u64);
        let total_functions = self.params.num_tables * self.params.num_functions;
        self.hash_functions = (0..total_functions)
            .map(|_| {
                (0..self.dimension)
                    .map(|_| next_f32(&mut rng_state))
                    .collect()
            })
            .collect();

        self.hash_tables = vec![HashMap::new(); self.params.num_tables];

        let mut hash_values: Vec<Vec<u64>> = Vec::with_capacity(self.num_vectors);
        for vector_idx in 0..self.num_vectors {
            let v = self.get_vector(vector_idx);
            let mut hashes = Vec::with_capacity(self.params.num_tables);
            for table_idx in 0..self.params.num_tables {
                hashes.push(self.compute_hash(v, table_idx));
            }
            hash_values.push(hashes);
        }

        for vector_idx in 0..self.num_vectors {
            for table_idx in 0..self.params.num_tables {
                let hash = hash_values[vector_idx][table_idx];
                self.hash_tables[table_idx]
                    .entry(hash)
                    .or_default()
                    .push(vector_idx as u32);
            }
        }

        self.built = true;
        Ok(())
    }

    /// Search for the top-k nearest candidates using hash buckets + exact verification.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, Error> {
        if !self.built {
            return Err(Error::NotBuilt);
        }
        if query.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut candidate_set: HashSet<u32> = HashSet::new();
        for table_idx in 0..self.params.num_tables {
            let hash = self.compute_hash(query, table_idx);
            if let Some(indices) = self.hash_tables[table_idx].get(&hash) {
                candidate_set.extend(indices.iter().copied());
            }
        }

        let mut candidates: Vec<(u32, f32)> = candidate_set
            .iter()
            .map(|&idx| {
                let v = self.get_vector(idx as usize);
                let dist = 1.0 - dot(query, v);
                (idx, dist)
            })
            .collect();

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Ok(candidates.into_iter().take(k).collect())
    }

    fn compute_hash(&self, vector: &[f32], table_idx: usize) -> u64 {
        let mut hash = 0u64;
        for func_idx in 0..self.params.num_functions {
            let hash_func_idx = table_idx * self.params.num_functions + func_idx;
            let proj = dot(vector, &self.hash_functions[hash_func_idx]);
            let bit = if proj >= 0.0 { 1 } else { 0 };
            hash = (hash << 1) | bit;
        }
        hash
    }

    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minhash::MinHash;
    use crate::simhash::SimHash;
    use std::collections::HashSet as StdHashSet;

    #[test]
    fn minhash_lsh_smoke() {
        let mh = MinHash::new(64);
        let mut ix = MinHashLSH::new(8, 8).unwrap();
        let a: StdHashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let b: StdHashSet<&str> = ["a", "b", "d"].into_iter().collect();
        ix.insert(mh.signature(&a));
        ix.insert(mh.signature(&b));
        let q = mh.signature(&a);
        assert!(!ix.query(&q).is_empty());
    }

    #[test]
    fn simhash_lsh_smoke() {
        let sh = SimHash::new(42);
        let mut ix = SimHashLSH::new(8, 8).unwrap();
        let fp1 = sh.fingerprint_weighted(&[(1, 1.0)]);
        let fp2 = sh.fingerprint_weighted(&[(2, 1.0)]);
        ix.insert(fp1);
        ix.insert(fp2);
        assert!(!ix.query(fp1).is_empty());
    }

    #[test]
    fn random_projection_lsh_smoke() {
        let mut ix = LSHIndex::new(2, LSHParams::default()).unwrap();
        ix.add(0, vec![1.0, 0.0]).unwrap();
        ix.add(1, vec![0.9, 0.1]).unwrap();
        ix.build().unwrap();
        let r = ix.search(&[1.0, 0.0], 2).unwrap();
        assert!(!r.is_empty());
    }
}
