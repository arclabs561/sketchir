//! Rosetta correctness fixtures: sketchir MinHash Jaccard estimation, the
//! DISTRIBUTIONAL tolerance class.
//!
//! Reference values in `fixtures/rosetta/sketchir_minhash.json` come from
//! `gen_sketchir.py` (their provenance). MinHash estimates the Jaccard
//! similarity of two sets; the estimate is stochastic (depends on the hash
//! family), so a cross-library hash-for-hash match is impossible. Instead the
//! oracle is the EXACT true Jaccard (computed by the generator, so this test
//! never recomputes it and cannot repeat sketchir's own potential error), and
//! the test asserts the estimate is within MinHash sampling error:
//! std = sqrt(J(1-J)/num_hashes), allowing 4 sigma plus a small floor.
//!
//! Two set pairs with true Jaccard ~1/3 and ~2/3 check that the estimator tracks
//! similarity across the range, not just at one point.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_sketchir.py`.

use serde::Deserialize;
use sketchir::MinHash;
use std::collections::HashSet;

const FIXTURE: &str = include_str!("fixtures/rosetta/sketchir_minhash.json");

#[derive(Deserialize)]
struct Fixture {
    num_hashes: usize,
    seed: u64,
    pairs: Vec<Pair>,
}

#[derive(Deserialize)]
struct Pair {
    a: Vec<u64>,
    b: Vec<u64>,
    true_jaccard: f64,
}

#[test]
fn rosetta_minhash_estimates_true_jaccard() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let mh = MinHash::with_seed(fx.num_hashes, fx.seed).expect("minhash");

    for (idx, pair) in fx.pairs.iter().enumerate() {
        let a: HashSet<u64> = pair.a.iter().copied().collect();
        let b: HashSet<u64> = pair.b.iter().copied().collect();
        let est = mh
            .signature(&a)
            .jaccard(&mh.signature(&b))
            .expect("jaccard estimate");

        let j = pair.true_jaccard;
        // MinHash variance: std = sqrt(J(1-J)/k). 4 sigma bound (+ floor) is loose
        // enough to never be flaky yet tight enough to catch a broken estimator.
        let sigma = (j * (1.0 - j) / fx.num_hashes as f64).sqrt();
        let bound = (4.0 * sigma).max(0.02);
        let diff = (est - j).abs();
        assert!(
            diff < bound,
            "pair {idx}: estimate={est} true={j} |diff|={diff} bound={bound}"
        );
    }
}
