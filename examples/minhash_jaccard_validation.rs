//! Validate MinHash Jaccard estimation against the exact Jaccard index.
//!
//! For sets with a known overlap, the true Jaccard is |A∩B|/|A∪B|. MinHash
//! estimates it from k hashes with error ~1/sqrt(k); the estimate must be
//! unbiased (mean error -> 0 as k grows) and exact at the anchors (identical
//! sets -> 1.0, disjoint -> 0.0). Systematic bias is a bug.
//!
//! ```sh
//! cargo run --release --example minhash_jaccard_validation
//! ```

use std::collections::HashSet;

use sketchir::minhash::MinHash;

fn exact_jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

/// Two sets of size `n` sharing exactly `overlap` elements (deterministic).
fn pair(n: u64, overlap: u64, salt: u64) -> (HashSet<u64>, HashSet<u64>) {
    let a: HashSet<u64> = (0..n).map(|i| i + salt * 1_000_000).collect();
    // b shares [0, overlap) with a, then disjoint high elements.
    let b: HashSet<u64> = (0..overlap)
        .map(|i| i + salt * 1_000_000)
        .chain((0..(n - overlap)).map(|i| 9_000_000 + i + salt * 1_000_000))
        .collect();
    (a, b)
}

fn main() {
    println!("=== anchors ===");
    let mh = MinHash::new(256).unwrap();
    let s: HashSet<u64> = (0..500).collect();
    let d: HashSet<u64> = (1000..1500).collect();
    let (sig_s, sig_s2, sig_d) = (mh.signature(&s), mh.signature(&s), mh.signature(&d));
    println!(
        "  identical sets: est = {:.4} (exact 1.0)",
        sig_s.jaccard(&sig_s2).unwrap()
    );
    println!(
        "  disjoint sets:  est = {:.4} (exact 0.0)",
        sig_s.jaccard(&sig_d).unwrap()
    );

    println!("\n=== mean |est - exact| over 40 set pairs, by hash count ===");
    for k in [64usize, 256, 1024] {
        let mh = MinHash::new(k).unwrap();
        let mut err = 0.0;
        let n_pairs = 40;
        for salt in 0..n_pairs {
            let overlap = (salt * 25) % 1000; // vary true Jaccard across pairs
            let (a, b) = pair(1000, overlap, salt as u64 + 1);
            let exact = exact_jaccard(&a, &b);
            let est = mh.signature(&a).jaccard(&mh.signature(&b)).unwrap();
            err += (est - exact).abs();
        }
        let mean = err / n_pairs as f64;
        let expected = 1.0 / (k as f64).sqrt();
        println!(
            "  k={k:<5} mean abs error = {mean:.4}   (~1/sqrt(k) = {expected:.4})  [{}]",
            if mean < 2.0 * expected { "OK" } else { "HIGH" }
        );
    }
}
