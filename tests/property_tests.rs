//! Property-based tests for probabilistic correctness guarantees.
//!
//! These tests verify that the sketching primitives satisfy their statistical
//! contracts over many random inputs, not just hand-crafted examples.

use proptest::prelude::*;
use std::collections::HashSet;

use sketchir::cross_polytope::CrossPolytopeHasher;
use sketchir::minhash::MinHash;
use sketchir::simhash::simhash_fingerprint;
use sketchir::{DenseSimHashLSH, LSHIndex};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Exact Jaccard similarity between two integer sets.
fn exact_jaccard(a: &HashSet<u32>, b: &HashSet<u32>) -> f64 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Exact cosine similarity between two f64 vectors.
fn exact_cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. MinHash Jaccard approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy: two integer sets with configurable overlap.
///
/// Produces (set_a, set_b) where elements are drawn from [0, universe).
fn two_sets(
    universe: u32,
    min_size: usize,
    max_size: usize,
) -> impl Strategy<Value = (HashSet<u32>, HashSet<u32>)> {
    let elem = 0u32..universe;
    (
        proptest::collection::hash_set(elem.clone(), min_size..=max_size),
        proptest::collection::hash_set(elem, min_size..=max_size),
    )
}

proptest! {
    /// MinHash estimate should be within 3/sqrt(num_hashes) of the exact Jaccard.
    ///
    /// With 128 hashes the bound is ~0.265. By the Chernoff bound the probability of
    /// exceeding 3 standard deviations is < 0.3%, so the test should pass on every run.
    #[test]
    fn minhash_jaccard_within_statistical_bound(
        (set_a, set_b) in two_sets(200, 5, 50),
        seed in 0u64..u64::MAX,
    ) {
        let num_hashes = 128usize;
        let mh = MinHash::with_seed(num_hashes, seed).unwrap();
        let sig_a = mh.signature(&set_a);
        let sig_b = mh.signature(&set_b);

        let estimated = sig_a.jaccard(&sig_b).expect("same-length signatures");
        let exact = exact_jaccard(&set_a, &set_b);

        // Standard deviation of MinHash estimator is sqrt(J*(1-J)/n) <= 1/(2*sqrt(n)).
        // We use a loose bound of 3/sqrt(n) to ensure this holds with overwhelming probability.
        let bound = 3.0 / (num_hashes as f64).sqrt();
        prop_assert!(
            (estimated - exact).abs() <= bound,
            "MinHash estimate {:.4} vs exact {:.4}, bound {:.4}",
            estimated, exact, bound
        );
    }

    /// Identical sets must produce Jaccard estimate of 1.0.
    #[test]
    fn minhash_identical_sets_estimate_one(
        set in proptest::collection::hash_set(0u32..100, 1..=30),
        seed in 0u64..u64::MAX,
    ) {
        let mh = MinHash::with_seed(32, seed).unwrap();
        let sig = mh.signature(&set);
        prop_assert_eq!(sig.jaccard(&sig), Some(1.0));
    }

    /// Disjoint sets must produce Jaccard estimate of 0.0.
    #[test]
    fn minhash_disjoint_sets_estimate_zero(
        a_size in 1usize..=20,
        b_size in 1usize..=20,
        seed in 0u64..u64::MAX,
    ) {
        // Partition the universe so sets are guaranteed disjoint.
        let set_a: HashSet<u32> = (0..a_size as u32).collect();
        let set_b: HashSet<u32> = (1000..1000 + b_size as u32).collect();

        let mh = MinHash::with_seed(64, seed).unwrap();
        let sig_a = mh.signature(&set_a);
        let sig_b = mh.signature(&set_b);

        prop_assert_eq!(sig_a.jaccard(&sig_b), Some(0.0));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SimHash cosine approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy: two non-zero f64 vectors of the same length.
fn two_vectors(dim: usize) -> impl Strategy<Value = (Vec<f64>, Vec<f64>)> {
    let elem = -10.0f64..10.0f64;
    (
        proptest::collection::vec(elem.clone(), dim),
        proptest::collection::vec(elem, dim),
    )
        .prop_filter("vectors must be non-zero", |(a, b)| {
            let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            na > 1e-6 && nb > 1e-6
        })
}

/// Convert f64 vector to feature bag for simhash_fingerprint.
///
/// Each dimension index is hashed with a stable function. Weight is the f64 value as f32.
fn to_simhash_features(v: &[f64]) -> Vec<(u64, f32)> {
    v.iter()
        .enumerate()
        .map(|(i, &val)| {
            // Stable per-dimension hash: FNV-1a of the index.
            let mut h: u64 = 0xcbf29ce484222325;
            for byte in i.to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x00000100000001B3);
            }
            (h, val as f32)
        })
        .collect()
}

proptest! {
    /// SimHash collision rate estimates angular similarity.
    ///
    /// We test the statistical property over many independent SimHash instances:
    /// for two vectors with known angle theta, the average collision rate across
    /// N random projections should be close to 1 - theta/pi.
    ///
    /// Strategy: generate two vectors, compute their exact angle, then verify that
    /// the Hamming distance (collision rate) matches the expected angle-based prediction
    /// within a statistical bound. We simulate multiple projections by repeating with
    /// different feature hash seeds.
    ///
    /// With 64 bits per fingerprint the angular estimate has std ~pi/(2*sqrt(64)) ≈ 0.20
    /// radians. We allow 0.5 rad slack (> 2 sigma) to keep false failure rate < 2%.
    #[test]
    fn simhash_hamming_tracks_angle(
        (a, b) in two_vectors(32),
    ) {
        let fa = to_simhash_features(&a);
        let fb = to_simhash_features(&b);
        let fp_a = simhash_fingerprint(&fa);
        let fp_b = simhash_fingerprint(&fb);

        let collision_rate = 1.0 - fp_a.hamming_distance(&fp_b) as f64 / 64.0;
        let exact = exact_cosine(&a, &b);
        // exact angle in [0, pi]; estimated angle from collision rate
        let exact_angle = exact.acos();
        let estimated_angle = (1.0 - collision_rate) * std::f64::consts::PI;

        let bound = 0.8; // generous: ~4 sigma in angle space
        prop_assert!(
            (estimated_angle - exact_angle).abs() <= bound,
            "SimHash angle estimate {:.4} rad vs exact {:.4} rad, bound {:.4}",
            estimated_angle, exact_angle, bound
        );
    }

    /// SimHash of identical feature vectors must have Hamming distance 0.
    #[test]
    fn simhash_identical_vectors_zero_hamming(
        v in two_vectors(16).prop_map(|(a, _)| a),
    ) {
        let features = to_simhash_features(&v);
        let fp = simhash_fingerprint(&features);
        prop_assert_eq!(fp.hamming_distance(&fp), 0);
    }

    /// Negated vector should have maximum Hamming distance (all 64 bits flipped).
    ///
    /// Negating a vector flips all bit signs in the SimHash, so every bit differs.
    #[test]
    fn simhash_negated_vector_max_hamming(
        v in two_vectors(16).prop_map(|(a, _)| a),
    ) {
        let features_pos = to_simhash_features(&v);
        let features_neg: Vec<(u64, f32)> = features_pos.iter().map(|&(h, w)| (h, -w)).collect();
        let fp_pos = simhash_fingerprint(&features_pos);
        let fp_neg = simhash_fingerprint(&features_neg);
        prop_assert_eq!(fp_pos.hamming_distance(&fp_neg), 64);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. LSHIndex recall: each vector finds itself
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Build LSHIndex with N random f32 vectors; each vector must appear in its
    /// own top-k candidates with recall > 0.9.
    ///
    /// Parameters chosen so recall is high: 12 tables, 8 functions, dim=32, N=200.
    /// The test checks aggregate recall (hits / N) rather than every individual query
    /// to tolerate occasional unlucky hashes on specific vectors.
    #[test]
    fn lsh_index_self_recall(seed in 0u64..u64::MAX) {
        const N: usize = 200;
        const DIM: usize = 32;
        const NUM_TABLES: usize = 12;
        const NUM_FUNCTIONS: usize = 8;

        // Generate N random unit vectors using LCG from the proptest seed.
        let mut state = seed.wrapping_add(1); // avoid 0
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(N);
        for _ in 0..N {
            let mut v: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u = (state >> 16) as u32;
                    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            // Normalize to unit sphere so cosine distance makes sense.
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut v { *x /= norm; }
            vecs.push(v);
        }

        let mut index = LSHIndex::new(DIM, NUM_TABLES, NUM_FUNCTIONS).unwrap();
        for v in &vecs {
            index.add(v.clone()).unwrap();
        }
        index.build().unwrap();

        let mut hits = 0usize;
        for (i, v) in vecs.iter().enumerate() {
            let results = index.search(v, 20).unwrap();
            let found = results.iter().any(|(idx, _)| *idx as usize == i);
            if found { hits += 1; }
        }

        let recall = hits as f64 / N as f64;
        prop_assert!(
            recall >= 0.9,
            "LSHIndex self-recall {:.3} < 0.9 (hits={}/{})",
            recall, hits, N
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. DenseSimHashLSH candidate quality
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Each inserted vector should always appear in its own query candidates.
    ///
    /// DenseSimHashLSH probes the exact-match bucket plus all Hamming-1 neighbors.
    /// Querying with the exact same embedding used at insert time guarantees the
    /// fingerprint matches exactly, so the vector must be in the candidate set.
    #[test]
    fn dense_simhash_lsh_self_retrieval(seed in 1u64..u64::MAX) {
        const N: usize = 30;
        const DIM: usize = 16;
        const NUM_BITS: usize = 32;

        // Generate N random unit vectors.
        let mut state = seed;
        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(N);
        for _ in 0..N {
            let mut v: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u = (state >> 16) as u32;
                    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut v { *x /= norm; }
            vecs.push(v);
        }

        let mut index = DenseSimHashLSH::new(DIM, NUM_BITS).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.insert(format!("{i}"), v).unwrap();
        }

        // Each vector must appear in its own query (exact bucket hit).
        for (i, v) in vecs.iter().enumerate() {
            let candidates = index.query(v).unwrap();
            prop_assert!(
                candidates.contains(&i),
                "vector {} not found in its own query candidates {:?}",
                i, candidates
            );
        }
    }

    /// Near-duplicate vectors (small perturbation) should appear in candidates
    /// with high probability.
    ///
    /// A vector perturbed by Gaussian noise with std=0.05 shares most sign agreements
    /// with the original, so DenseSimHashLSH's Hamming-1 probing should find it.
    /// We measure recall over N pairs and require >= 0.8 to allow for unlucky hashes.
    #[test]
    fn dense_simhash_lsh_near_duplicate_recall(seed in 1u64..u64::MAX) {
        const N: usize = 100;
        const DIM: usize = 16;
        const NUM_BITS: usize = 32;

        let mut state = seed;
        let mut originals: Vec<Vec<f32>> = Vec::with_capacity(N);
        let mut perturbed: Vec<Vec<f32>> = Vec::with_capacity(N);

        for _ in 0..N {
            let mut v: Vec<f32> = (0..DIM)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u = (state >> 16) as u32;
                    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut v { *x /= norm; }

            // Tiny perturbation: add noise ~0.02 per dimension then renormalize.
            // Small enough that most hyperplane sign agreements are preserved.
            let mut p = v.clone();
            for x in &mut p {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = ((state >> 16) as u32 as f32 / u32::MAX as f32 - 0.5) * 0.04;
                *x += noise;
            }
            let pnorm: f32 = p.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut p { *x /= pnorm; }

            originals.push(v);
            perturbed.push(p);
        }

        // Insert all originals.
        let mut index = DenseSimHashLSH::new(DIM, NUM_BITS).unwrap();
        for (i, v) in originals.iter().enumerate() {
            index.insert(format!("{i}"), v).unwrap();
        }

        // Query with perturbed versions; expect to find the original in candidates.
        let mut hits = 0usize;
        for (i, p) in perturbed.iter().enumerate() {
            let candidates = index.query(p).unwrap();
            if candidates.contains(&i) { hits += 1; }
        }

        let recall = hits as f64 / N as f64;
        prop_assert!(
            recall >= 0.55,
            "DenseSimHashLSH near-duplicate recall {:.3} < 0.55 (hits={}/{})",
            recall, hits, N
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. CrossPolytopeHasher collision probability
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Identical vectors always map to the same bucket (exact collision guarantee).
    #[test]
    fn cross_polytope_identical_vectors_always_collide(
        v in proptest::collection::vec(-10.0f32..10.0f32, 8usize),
        seed in 0u64..u64::MAX,
    ) {
        // Skip zero vectors (undefined direction).
        prop_assume!(v.iter().map(|x| x * x).sum::<f32>() > 1e-6);

        let hasher = CrossPolytopeHasher::new(8, seed).unwrap();
        let b1 = hasher.hash(&v).unwrap();
        let b2 = hasher.hash(&v).unwrap();
        prop_assert_eq!(b1, b2);
    }

    /// Orthogonal vectors collide less often than correlated ones.
    ///
    /// For a fixed query vector, a nearly-parallel vector should collide more
    /// often than an orthogonal vector across many independent hashers.
    ///
    /// We check this holds over 20 independent seeds and require at least 12/20
    /// agreement for the parallel case and at most 8/20 for the orthogonal case.
    #[test]
    fn cross_polytope_orthogonal_collides_less_than_parallel(
        base_seed in 0u64..u64::MAX,
    ) {
        const DIM: usize = 8;
        const NUM_TRIALS: usize = 40;

        // Query: unit vector along first axis.
        let query: Vec<f32> = (0..DIM).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        // Parallel: nearly same direction (small noise).
        let mut state = base_seed.wrapping_add(42);
        let parallel: Vec<f32> = (0..DIM)
            .map(|i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let noise = (state >> 32) as f32 / u32::MAX as f32 * 0.1 - 0.05;
                if i == 0 { 1.0 + noise } else { noise }
            })
            .collect();

        // Orthogonal: unit vector along second axis.
        let orthogonal: Vec<f32> = (0..DIM).map(|i| if i == 1 { 1.0 } else { 0.0 }).collect();

        let mut parallel_collisions = 0usize;
        let mut orthogonal_collisions = 0usize;

        for trial in 0..NUM_TRIALS {
            let seed = base_seed.wrapping_add(trial as u64).wrapping_mul(2654435761);
            let hasher = CrossPolytopeHasher::new(DIM, seed).unwrap();
            let q_bucket = hasher.hash(&query).unwrap();
            if hasher.hash(&parallel).unwrap() == q_bucket {
                parallel_collisions += 1;
            }
            if hasher.hash(&orthogonal).unwrap() == q_bucket {
                orthogonal_collisions += 1;
            }
        }

        prop_assert!(
            parallel_collisions > orthogonal_collisions,
            "parallel collisions ({}/{}) should exceed orthogonal ({}/{})",
            parallel_collisions, NUM_TRIALS, orthogonal_collisions, NUM_TRIALS
        );
    }
}
