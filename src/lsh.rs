//! LSH-style indexing helpers.
//!
//! This module is intentionally small and deterministic. It provides:
//! - banding-based LSH for [`MinHashSignature`]
//! - bit-sampling LSH for [`SimHashFingerprint`]
//! - random-projection LSH for dense vectors (cosine-ish)

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::minhash::MinHashSignature;
use crate::simhash::SimHashFingerprint;
use crate::{all_finite, lcg_f32, lcg_next, Error, Fnv1a64};

/// LSH index using MinHash banding (near-duplicate detection).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        bands
            .checked_mul(rows_per_band)
            .ok_or(Error::InvalidParam("bands * rows_per_band overflow"))?;
        Ok(Self {
            bands,
            rows_per_band,
            buckets: (0..bands).map(|_| HashMap::new()).collect(),
            signatures: Vec::new(),
        })
    }

    /// Expected signature length for this index configuration.
    pub fn expected_signature_len(&self) -> usize {
        self.bands * self.rows_per_band
    }

    /// Insert a signature and return its assigned document id.
    ///
    /// Returns an error if the signature length does not match
    /// `bands * rows_per_band`.
    pub fn insert(&mut self, signature: MinHashSignature) -> Result<usize, Error> {
        let expected = self.expected_signature_len();
        if signature.values.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: signature.values.len(),
            });
        }
        let doc_id = self.signatures.len();
        for (band_idx, chunk) in signature.values.chunks(self.rows_per_band).enumerate() {
            let band_hash = hash_band(chunk);
            self.buckets[band_idx]
                .entry(band_hash)
                .or_default()
                .push(doc_id);
        }
        self.signatures.push(signature);
        Ok(doc_id)
    }

    /// Return candidate document IDs that share at least one band bucket.
    ///
    /// A signature whose length differs from [`Self::expected_signature_len`]
    /// cannot belong to this index and returns no candidates.
    pub fn query(&self, signature: &MinHashSignature) -> Vec<usize> {
        if signature.values.len() != self.expected_signature_len() {
            return Vec::new();
        }
        let mut candidates: HashSet<usize> = HashSet::new();
        for (band_idx, chunk) in signature
            .values
            .chunks(self.rows_per_band)
            .enumerate()
            .take(self.bands)
        {
            let band_hash = hash_band(chunk);
            if let Some(docs) = self.buckets[band_idx].get(&band_hash) {
                candidates.extend(docs.iter().copied());
            }
        }
        let mut v: Vec<usize> = candidates.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Return candidate document IDs that share at least `min_shared_bands` band
    /// buckets with `signature`.
    ///
    /// A threshold of `0` is treated as `1`, matching [`Self::query`]. A threshold
    /// larger than the configured band count has no possible matches. A
    /// signature with the wrong length also returns no candidates.
    pub fn query_min_shared_bands(
        &self,
        signature: &MinHashSignature,
        min_shared_bands: usize,
    ) -> Vec<usize> {
        if signature.values.len() != self.expected_signature_len() {
            return Vec::new();
        }
        let min_shared_bands = min_shared_bands.max(1);
        if min_shared_bands == 1 {
            return self.query(signature);
        }
        if min_shared_bands > self.bands {
            return Vec::new();
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for (band_idx, chunk) in signature
            .values
            .chunks(self.rows_per_band)
            .enumerate()
            .take(self.bands)
        {
            let band_hash = hash_band(chunk);
            if let Some(docs) = self.buckets[band_idx].get(&band_hash) {
                for doc in docs {
                    *counts.entry(*doc).or_insert(0) += 1;
                }
            }
        }

        let mut v: Vec<usize> = counts
            .into_iter()
            .filter_map(|(doc, count)| (count >= min_shared_bands).then_some(doc))
            .collect();
        v.sort_unstable();
        v
    }

    /// Query and return candidates with estimated Jaccard similarity.
    pub fn query_with_similarity(&self, signature: &MinHashSignature) -> Vec<(usize, f64)> {
        self.query_with_similarity_min_shared_bands(signature, 1)
    }

    /// Query candidates sharing at least `min_shared_bands` bands and return them
    /// ranked by estimated Jaccard similarity.
    pub fn query_with_similarity_min_shared_bands(
        &self,
        signature: &MinHashSignature,
        min_shared_bands: usize,
    ) -> Vec<(usize, f64)> {
        let mut results: Vec<(usize, f64)> = self
            .query_min_shared_bands(signature, min_shared_bands)
            .into_iter()
            .filter_map(|id| {
                let sim = signature.jaccard(&self.signatures[id])?;
                Some((id, sim))
            })
            .collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        results
    }
}

fn hash_band(values: &[u64]) -> u64 {
    let mut hasher = Fnv1a64::new();
    for v in values {
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// LSH index for 64-bit SimHash fingerprints using bit sampling.
#[derive(Debug)]
pub struct SimHashLSH {
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
                let bit = (lcg_next(&mut rng_state) as usize) % 64;
                if !idxs.contains(&bit) {
                    idxs.push(bit);
                }
            }
            idxs.sort_unstable();
            bit_indices.push(idxs);
        }

        Ok(Self {
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

/// A minimal random-projection LSH index for dense vectors.
///
/// Two-phase lifecycle: call [`add`](Self::add) to insert vectors, then
/// [`build`](Self::build) to construct hash tables, then [`search`](Self::search) to query.
/// For incremental insertion without a build step, see [`DenseSimHashLSH`](crate::DenseSimHashLSH).
#[derive(Debug)]
pub struct LSHIndex {
    dimension: usize,
    num_tables: usize,
    num_functions: usize,
    built: bool,
    vectors: Vec<f32>, // SoA [n * d]
    num_vectors: usize,
    hash_functions: Vec<Vec<f32>>,
    hash_tables: Vec<HashMap<u64, Vec<u32>>>,
}

impl LSHIndex {
    /// Create a new index for vectors of length `dimension`.
    pub fn new(dimension: usize, num_tables: usize, num_functions: usize) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::InvalidParam("dimension must be >= 1"));
        }
        if num_tables == 0 || num_functions == 0 {
            return Err(Error::InvalidParam(
                "num_tables and num_functions must be >= 1",
            ));
        }
        Ok(Self {
            dimension,
            num_tables,
            num_functions,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            hash_functions: Vec::new(),
            hash_tables: Vec::new(),
        })
    }

    /// Create a new index with default parameters (10 tables, 10 functions).
    pub fn with_defaults(dimension: usize) -> Result<Self, Error> {
        Self::new(dimension, 10, 10)
    }

    /// Add a vector (before calling [`Self::build`]).
    pub fn add(&mut self, vector: Vec<f32>) -> Result<(), Error> {
        if self.built {
            return Err(Error::AddAfterBuild);
        }
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if !all_finite(&vector) {
            return Err(Error::NonFiniteInput);
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
        let mut rng_state = 0x9E3779B97F4A7C15u64
            ^ (self.dimension as u64)
            ^ ((self.num_tables as u64) << 32)
            ^ (self.num_functions as u64);
        let total_functions = self.num_tables * self.num_functions;
        self.hash_functions = (0..total_functions)
            .map(|_| {
                (0..self.dimension)
                    .map(|_| lcg_f32(&mut rng_state))
                    .collect()
            })
            .collect();

        self.hash_tables = vec![HashMap::new(); self.num_tables];

        let mut hash_values: Vec<Vec<u64>> = Vec::with_capacity(self.num_vectors);
        for vector_idx in 0..self.num_vectors {
            let v = self.get_vector(vector_idx);
            let mut hashes = Vec::with_capacity(self.num_tables);
            for table_idx in 0..self.num_tables {
                hashes.push(self.compute_hash(v, table_idx));
            }
            hash_values.push(hashes);
        }

        #[allow(clippy::needless_range_loop)]
        for vector_idx in 0..self.num_vectors {
            for table_idx in 0..self.num_tables {
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
        if !all_finite(query) {
            return Err(Error::NonFiniteInput);
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut candidate_set: HashSet<u32> = HashSet::new();
        for table_idx in 0..self.num_tables {
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
        for func_idx in 0..self.num_functions {
            let hash_func_idx = table_idx * self.num_functions + func_idx;
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

use crate::multibit::dot;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minhash::MinHash;
    use crate::simhash::simhash_fingerprint;
    use std::collections::HashSet as StdHashSet;

    #[test]
    fn minhash_lsh_smoke() {
        let mh = MinHash::new(64).unwrap();
        let mut ix = MinHashLSH::new(8, 8).unwrap();
        let a: StdHashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let b: StdHashSet<&str> = ["a", "b", "d"].into_iter().collect();
        ix.insert(mh.signature(&a)).unwrap();
        ix.insert(mh.signature(&b)).unwrap();
        let q = mh.signature(&a);
        assert!(!ix.query(&q).is_empty());
    }

    #[test]
    fn minhash_lsh_rejects_wrong_signature_length() {
        let mut ix = MinHashLSH::new(4, 4).unwrap(); // expects 16
        let mh = MinHash::new(8).unwrap(); // produces 8
        let items: StdHashSet<&str> = ["x"].into_iter().collect();
        assert!(ix.insert(mh.signature(&items)).is_err());
    }

    #[test]
    fn minhash_lsh_rejects_signature_length_overflow() {
        assert!(matches!(
            MinHashLSH::new(usize::MAX, 2),
            Err(Error::InvalidParam("bands * rows_per_band overflow"))
        ));
    }

    #[test]
    fn minhash_lsh_wrong_length_queries_return_no_candidates() {
        let mut ix = MinHashLSH::new(2, 2).unwrap();
        let indexed = MinHashSignature {
            values: vec![1, 2, 3, 4],
        };
        ix.insert(indexed).unwrap();

        for query in [
            MinHashSignature { values: vec![1, 2] },
            MinHashSignature {
                values: vec![1, 2, 3, 4, 5, 6],
            },
        ] {
            assert!(ix.query(&query).is_empty());
            assert!(ix.query_min_shared_bands(&query, 2).is_empty());
            assert!(ix.query_with_similarity(&query).is_empty());
            assert!(ix
                .query_with_similarity_min_shared_bands(&query, 2)
                .is_empty());
        }
    }

    #[test]
    fn simhash_lsh_smoke() {
        let mut ix = SimHashLSH::new(8, 8).unwrap();
        let fp1 = simhash_fingerprint(&[(1, 1.0)]);
        let fp2 = simhash_fingerprint(&[(2, 1.0)]);
        ix.insert(fp1);
        ix.insert(fp2);
        assert!(!ix.query(fp1).is_empty());
    }

    #[test]
    fn random_projection_lsh_smoke() {
        let mut ix = LSHIndex::with_defaults(2).unwrap();
        ix.add(vec![1.0, 0.0]).unwrap();
        ix.add(vec![0.9, 0.1]).unwrap();
        ix.build().unwrap();
        let r = ix.search(&[1.0, 0.0], 2).unwrap();
        assert!(!r.is_empty());
    }

    // DETERMINISM CANARY
    #[test]
    fn minhash_lsh_banding_determinism() {
        let mh = MinHash::new(8).unwrap();
        let mut ix = MinHashLSH::new(2, 4).unwrap();
        let items: StdHashSet<&str> = ["hello", "world"].into_iter().collect();
        let sig = mh.signature(&items);
        let doc_id = ix.insert(sig.clone()).unwrap();
        assert_eq!(doc_id, 0);
        let candidates = ix.query(&sig);
        assert_eq!(candidates, vec![0]);
    }

    #[test]
    fn minhash_lsh_min_shared_bands_filters_single_band_hits() {
        let mut ix = MinHashLSH::new(3, 1).unwrap();
        let query = MinHashSignature {
            values: vec![1, 2, 3],
        };
        ix.insert(query.clone()).unwrap();
        ix.insert(MinHashSignature {
            values: vec![1, 20, 30],
        })
        .unwrap();
        ix.insert(MinHashSignature {
            values: vec![1, 2, 30],
        })
        .unwrap();
        ix.insert(MinHashSignature {
            values: vec![10, 20, 30],
        })
        .unwrap();

        assert_eq!(ix.query(&query), vec![0, 1, 2]);
        assert_eq!(ix.query_min_shared_bands(&query, 2), vec![0, 2]);
        assert_eq!(ix.query_min_shared_bands(&query, 4), Vec::<usize>::new());
    }
}
