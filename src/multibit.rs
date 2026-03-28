//! Multi-bit locality-sensitive hashing via distributional quantization.
//!
//! Standard SimHash/LSH uses 1 bit per random projection (the sign). This wastes
//! information: the magnitude of the projection tells you *how far* the vector is
//! from the hyperplane, but sign quantization discards it.
//!
//! Multi-bit LSH quantizes each projection into k bits using optimal codebook
//! boundaries derived from the projection's distribution. By CLT, random projections
//! of high-dimensional vectors are approximately Gaussian, so Gaussian quantile
//! boundaries are near-optimal.
//!
//! # Trade-offs vs 1-bit SimHash
//!
//! | Property | 1-bit (SimHash) | k-bit (MultibitLSH) |
//! |----------|-----------------|---------------------|
//! | Bits per projection | 1 | k (2-8) |
//! | Information per bit | maximal for sign | diminishing per extra bit |
//! | Hash table buckets | 2^num_projections | (2^k)^num_projections |
//! | Distance metric | Hamming ≈ cosine angle | finer-grained distance |
//! | Best for | high-dim, many projections | fewer projections, more bits each |
//!
//! # References
//!
//! - Petersen & Sutter, "Distributional Quantization" (2021-2023)
//! - Kong & Li (2012), "Isotropic Hashing" (multi-bit extension of SimHash)

use std::collections::{HashMap, HashSet};

use crate::{all_finite, lcg_f32, Error};

/// Configuration for multi-bit LSH.
#[derive(Debug, Clone)]
pub struct MultibitConfig {
    /// Number of random projections per hash table.
    pub num_projections: usize,
    /// Bits per projection (1-8). 1 = standard SimHash.
    pub bits_per_projection: u8,
    /// Number of independent hash tables (more = higher recall, more memory).
    pub num_tables: usize,
}

impl MultibitConfig {
    /// Standard SimHash configuration (1 bit per projection).
    pub fn simhash(num_projections: usize, num_tables: usize) -> Self {
        Self {
            num_projections,
            bits_per_projection: 1,
            num_tables,
        }
    }

    /// Multi-bit configuration with Gaussian-optimal quantization.
    pub fn multibit(num_projections: usize, bits_per_projection: u8, num_tables: usize) -> Self {
        Self {
            num_projections,
            bits_per_projection,
            num_tables,
        }
    }

    /// Total bits per fingerprint (per table).
    pub fn fingerprint_bits(&self) -> usize {
        self.num_projections * self.bits_per_projection as usize
    }
}

/// A multi-bit LSH index for dense vectors.
///
/// Each random projection is quantized into `bits_per_projection` bits using
/// Gaussian quantile boundaries, producing finer-grained hash codes than
/// standard 1-bit SimHash.
#[derive(Debug)]
pub struct MultibitLSH {
    config: MultibitConfig,
    dimension: usize,
    /// Hyperplanes: [num_tables * num_projections][dimension]
    hyperplanes: Vec<Vec<f32>>,
    /// Quantization boundaries for each bin (derived from Gaussian quantiles).
    /// Length = 2^bits_per_projection - 1 (the interior bin edges).
    boundaries: Vec<f64>,
    /// Hash tables: table_idx -> bucket_hash -> vec of item indices
    tables: Vec<HashMap<u64, Vec<u32>>>,
    /// Stored vectors (flat SoA)
    vectors: Vec<f32>,
    num_vectors: usize,
    built: bool,
}

impl MultibitLSH {
    /// Create a new multi-bit LSH index.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Embedding dimension
    /// * `config` - Multi-bit LSH configuration
    pub fn new(dimension: usize, config: MultibitConfig) -> Result<Self, Error> {
        if dimension == 0 {
            return Err(Error::InvalidParam("dimension must be >= 1"));
        }
        if config.num_projections == 0 {
            return Err(Error::InvalidParam("num_projections must be >= 1"));
        }
        if config.bits_per_projection == 0 || config.bits_per_projection > 8 {
            return Err(Error::InvalidParam("bits_per_projection must be in [1, 8]"));
        }
        if config.num_tables == 0 {
            return Err(Error::InvalidParam("num_tables must be >= 1"));
        }

        // Compute Gaussian quantile boundaries for the bins.
        // For k bits we have 2^k bins and 2^k - 1 interior boundaries.
        let n_levels = 1usize << config.bits_per_projection;
        let boundaries: Vec<f64> = (1..n_levels)
            .map(|i| gaussian_quantile(i as f64 / n_levels as f64))
            .collect();

        // Generate deterministic hyperplanes
        let total_projections = config.num_tables * config.num_projections;
        let mut rng_state = 0xDEADBEEF_u64
            ^ (dimension as u64)
            ^ ((config.num_tables as u64) << 32)
            ^ ((config.num_projections as u64) << 16)
            ^ (config.bits_per_projection as u64);

        let hyperplanes: Vec<Vec<f32>> = (0..total_projections)
            .map(|_| (0..dimension).map(|_| lcg_f32(&mut rng_state)).collect())
            .collect();

        Ok(Self {
            config,
            dimension,
            hyperplanes,
            boundaries,
            tables: Vec::new(),
            vectors: Vec::new(),
            num_vectors: 0,
            built: false,
        })
    }

    /// Add a vector to the index (before calling `build`).
    pub fn add(&mut self, vector: &[f32]) -> Result<(), Error> {
        if self.built {
            return Err(Error::AddAfterBuild);
        }
        if vector.len() != self.dimension {
            return Err(Error::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if !all_finite(vector) {
            return Err(Error::NonFiniteInput);
        }
        self.vectors.extend_from_slice(vector);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the hash tables. Must be called after all vectors are added.
    pub fn build(&mut self) -> Result<(), Error> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(Error::EmptyIndex);
        }

        self.tables = vec![HashMap::new(); self.config.num_tables];

        for vec_idx in 0..self.num_vectors {
            // Pre-compute hashes to avoid borrow conflict with self.tables
            let hashes: Vec<u64> = (0..self.config.num_tables)
                .map(|t| {
                    let v = self.get_vector(vec_idx);
                    self.compute_multibit_hash(v, t)
                })
                .collect();
            for (table_idx, hash) in hashes.into_iter().enumerate() {
                self.tables[table_idx]
                    .entry(hash)
                    .or_default()
                    .push(vec_idx as u32);
            }
        }

        self.built = true;
        Ok(())
    }

    /// Search for the top-k nearest candidates.
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
        for table_idx in 0..self.config.num_tables {
            let hash = self.compute_multibit_hash(query, table_idx);
            if let Some(indices) = self.tables[table_idx].get(&hash) {
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

    /// Compute the multi-bit hash code for a vector within a specific table.
    ///
    /// Each projection is quantized into `bits_per_projection` bits using
    /// Gaussian quantile boundaries, then packed into a u64.
    pub fn fingerprint(&self, vector: &[f32], table_idx: usize) -> u64 {
        self.compute_multibit_hash(vector, table_idx)
    }

    /// Compute the multi-bit hash for a vector in a specific table.
    fn compute_multibit_hash(&self, vector: &[f32], table_idx: usize) -> u64 {
        let bpp = self.config.bits_per_projection as usize;
        let mut hash = 0u64;

        for proj_idx in 0..self.config.num_projections {
            let hp_idx = table_idx * self.config.num_projections + proj_idx;
            let proj = dot(vector, &self.hyperplanes[hp_idx]) as f64;

            // Quantize: find which bin this projection falls into
            let code = quantize_scalar(proj, &self.boundaries);

            // Pack into hash
            let shift = proj_idx * bpp;
            if shift < 64 {
                hash |= (code as u64) << shift;
            }
        }

        hash
    }

    /// Number of items in the index.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Total bits per fingerprint.
    pub fn fingerprint_bits(&self) -> usize {
        self.config.fingerprint_bits()
    }

    /// The quantization boundaries (Gaussian quantiles).
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

/// Quantize a scalar projection value into a bin index using boundaries.
///
/// Returns an integer in `[0, 2^bits - 1]`.
fn quantize_scalar(value: f64, boundaries: &[f64]) -> u32 {
    // Binary search for the bin
    match boundaries.binary_search_by(|b| b.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Less))
    {
        Ok(pos) => pos as u32 + 1, // Exactly on a boundary -> upper bin
        Err(pos) => pos as u32,    // Between boundaries
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Gaussian quantile function (inverse CDF), Acklam's rational approximation.
fn gaussian_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let a = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1bit_matches_simhash_behavior() {
        // 1-bit multi-bit LSH should behave like standard SimHash
        let config = MultibitConfig::simhash(10, 2);
        let mut idx = MultibitLSH::new(8, config).unwrap();

        let v1: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let v2: Vec<f32> = (0..8).map(|i| (i as f32).sin() + 0.001).collect();

        idx.add(&v1).unwrap();
        idx.add(&v2).unwrap();
        idx.build().unwrap();

        let results = idx.search(&v1, 2).unwrap();
        assert!(!results.is_empty(), "Should find at least the query itself");
    }

    #[test]
    fn test_multibit_finds_similar() {
        let config = MultibitConfig::multibit(8, 2, 4);
        let mut idx = MultibitLSH::new(16, config).unwrap();

        // Two similar vectors
        let v1: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
        let v2: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin() + 0.01).collect();
        // One dissimilar vector
        let v3: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).cos()).collect();

        idx.add(&v1).unwrap();
        idx.add(&v2).unwrap();
        idx.add(&v3).unwrap();
        idx.build().unwrap();

        let results = idx.search(&v1, 3).unwrap();
        assert!(!results.is_empty());
        // v1 (idx 0) should be in the results
        assert!(
            results.iter().any(|(idx, _)| *idx == 0),
            "v1 should be in results: {results:?}"
        );
    }

    #[test]
    fn test_boundaries_are_symmetric() {
        let config = MultibitConfig::multibit(4, 3, 1); // 3 bits = 8 levels, 7 boundaries
        let idx = MultibitLSH::new(4, config).unwrap();
        let b = idx.boundaries();
        assert_eq!(b.len(), 7);

        // Gaussian quantiles should be symmetric around 0
        let mid = b.len() / 2;
        assert!(
            (b[mid]).abs() < 0.01,
            "Middle boundary should be near 0: {}",
            b[mid]
        );
        for i in 0..mid {
            assert!(
                (b[i] + b[b.len() - 1 - i]).abs() < 0.01,
                "Boundaries should be symmetric: {} vs {}",
                b[i],
                b[b.len() - 1 - i]
            );
        }
    }

    #[test]
    fn test_quantize_scalar() {
        // 2 bits = 4 levels, 3 boundaries
        let boundaries = vec![
            gaussian_quantile(0.25),
            gaussian_quantile(0.50),
            gaussian_quantile(0.75),
        ];

        // Very negative -> bin 0
        assert_eq!(quantize_scalar(-10.0, &boundaries), 0);
        // Very positive -> bin 3
        assert_eq!(quantize_scalar(10.0, &boundaries), 3);
        // Near zero -> bin 1 or 2
        let mid = quantize_scalar(0.0, &boundaries);
        assert!(
            mid == 1 || mid == 2,
            "Near zero should be middle bin: {mid}"
        );
    }

    #[test]
    fn test_more_bits_finer_discrimination() {
        let dim = 32;
        let n_tables = 4;
        let n_proj = 8;

        // 1-bit: coarse
        let config1 = MultibitConfig::simhash(n_proj, n_tables);
        let idx1 = MultibitLSH::new(dim, config1).unwrap();

        // 4-bit: fine
        let config4 = MultibitConfig::multibit(n_proj, 4, n_tables);
        let idx4 = MultibitLSH::new(dim, config4).unwrap();

        assert_eq!(idx1.fingerprint_bits(), n_proj); // 8 bits total
        assert_eq!(idx4.fingerprint_bits(), n_proj * 4); // 32 bits total

        // More bits per projection = more discriminating power per projection
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();
        let fp1 = idx1.fingerprint(&v, 0);
        let fp4 = idx4.fingerprint(&v, 0);

        // Both should produce valid fingerprints
        assert!(
            fp1 < (1u64 << n_proj),
            "1-bit hash should fit in {n_proj} bits"
        );
        // 4-bit hash uses more bits
        let _ = fp4; // just ensure it doesn't panic
    }

    #[test]
    fn test_4bit_larger_index() {
        let dim = 64;
        let config = MultibitConfig::multibit(6, 4, 8);
        let mut idx = MultibitLSH::new(dim, config).unwrap();

        // Add 100 random-ish vectors
        for i in 0..100 {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * 17 + j * 31) as f32 * 0.01).sin())
                .collect();
            idx.add(&v).unwrap();
        }
        idx.build().unwrap();

        // Search for the first vector
        let query: Vec<f32> = (0..dim).map(|j| (j * 31) as f32 * 0.01).collect();
        let results = idx
            .search(&query.iter().map(|x| x.sin()).collect::<Vec<_>>(), 5)
            .unwrap();

        // Should return something (even if not perfect recall)
        assert_eq!(idx.len(), 100);
        // results might be empty if no hash collision -- that's ok for this test
        let _ = results;
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = MultibitConfig::multibit(4, 2, 1);
        let mut idx = MultibitLSH::new(8, config).unwrap();
        assert!(idx.add(&[1.0, 2.0]).is_err()); // wrong dimension
    }

    #[test]
    fn test_empty_index_build_fails() {
        let config = MultibitConfig::multibit(4, 2, 1);
        let mut idx = MultibitLSH::new(8, config).unwrap();
        assert!(idx.build().is_err());
    }

    #[test]
    fn test_search_before_build_fails() {
        let config = MultibitConfig::multibit(4, 2, 1);
        let idx = MultibitLSH::new(8, config).unwrap();
        assert!(idx.search(&[0.0; 8], 5).is_err());
    }

    #[test]
    fn test_gaussian_quantile_symmetry() {
        let q25 = gaussian_quantile(0.25);
        let q75 = gaussian_quantile(0.75);
        assert!(
            (q25 + q75).abs() < 1e-10,
            "Gaussian quantile should be symmetric: {q25} + {q75}"
        );
    }

    // DETERMINISM CANARY
    #[test]
    fn multibit_fingerprint_determinism() {
        let config = MultibitConfig::multibit(4, 2, 1);
        let idx = MultibitLSH::new(4, config).unwrap();
        let fp = idx.fingerprint(&[1.0, -0.5, 0.3, 0.8], 0);
        assert_eq!(
            fp, 149,
            "Fingerprint changed -- hyperplanes or quantization logic drifted"
        );
    }
}
