//! `sketchir`: sketching primitives for IR.
//!
//! This crate is intended for **index-only** similarity sketches used in:
//! - near-duplicate detection (MinHash / shingles)
//! - text fingerprinting (SimHash)
//! - approximate similarity search (LSH-style candidate generation)
//!
//! Scope here is *primitives*: signatures, basic indexing, deterministic behavior.
//! Higher-level workflows (crawl dedupe pipelines, content extraction, etc.) belong elsewhere.

#![warn(missing_docs)]

pub mod blocking;
pub mod dense_simhash;
pub mod lsh;
pub mod minhash;
pub mod simhash;

pub use blocking::{BlockingConfig, MinHashTextLSH, TextItem};
pub use dense_simhash::DenseSimHashLSH;
pub use lsh::{LSHIndex, LSHParams, MinHashLSH, SimHashLSH};
pub use minhash::{MinHash, MinHashSignature};
pub use simhash::{SimHash, SimHashFingerprint};

/// Deterministic 64-bit LCG (Knuth constants). Not cryptographic.
pub(crate) fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// LCG-based uniform f32 in [-1, 1].
pub(crate) fn lcg_f32(state: &mut u64) -> f32 {
    lcg_next(state);
    let u = (*state >> 16) as u32;
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// True if all values are finite (not NaN or infinity).
pub(crate) fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}
