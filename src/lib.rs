//! `sketchir`: sketching primitives for IR.
//!
//! This crate is intended for **index-only** similarity sketches used in:
//! - near-duplicate detection (MinHash / shingles)
//! - text fingerprinting (SimHash)
//! - approximate similarity search (LSH-style candidate generation)
//!
//! Scope here is *primitives*: signatures, basic indexing, deterministic behavior.
//! Higher-level workflows (crawl dedupe pipelines, content extraction, etc.) belong elsewhere.
//!
//! ## Dense-vector LSH: `LSHIndex` vs `DenseSimHashLSH`
//!
//! Both index dense `f32` vectors but differ in lifecycle and recall strategy:
//!
//! - [`LSHIndex`]: batch workflow (add, build, search). Multi-table random projection
//!   with tunable `num_tables` / `num_functions` for recall control.
//! - [`DenseSimHashLSH`]: incremental insertion (no build step). Single SimHash table
//!   with Hamming-distance-1 neighbor probing for approximate recall.

#![warn(missing_docs)]

pub mod blocking;
pub mod dense_simhash;
pub mod hyperplane;
pub mod lsh;
pub mod minhash;
pub mod multibit;
pub mod simhash;

pub use blocking::{BlockingConfig, MinHashTextLSH};
pub use dense_simhash::DenseSimHashLSH;
pub use hyperplane::HyperplaneHasher;
pub use lsh::{LSHIndex, MinHashLSH, SimHashLSH};
pub use minhash::{MinHash, MinHashSignature};
pub use multibit::{MultibitConfig, MultibitLSH};
pub use simhash::{simhash_fingerprint, SimHashFingerprint};

use std::fmt;
use std::hash::Hasher;

/// Errors for sketchir indexes and operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A parameter is out of range or inconsistent.
    InvalidParam(&'static str),
    /// Dimension mismatch between expected and provided vectors.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual provided dimension.
        got: usize,
    },
    /// The index has no inserted items.
    EmptyIndex,
    /// The index has not been built yet.
    NotBuilt,
    /// Attempted to add after the index was built.
    AddAfterBuild,
    /// Input data contains non-finite values (NaN or infinity).
    NonFiniteInput,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Error::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch (expected {expected}, got {got})")
            }
            Error::EmptyIndex => f.write_str("empty index"),
            Error::NotBuilt => f.write_str("index not built"),
            Error::AddAfterBuild => f.write_str("cannot add after build"),
            Error::NonFiniteInput => f.write_str("input contains non-finite values (NaN or Inf)"),
        }
    }
}

impl std::error::Error for Error {}

/// A small stable 64-bit FNV-1a hasher.
///
/// This avoids relying on `std`'s `DefaultHasher` stability guarantees.
pub(crate) struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    pub(crate) fn new() -> Self {
        // FNV offset basis
        Self {
            state: 0xcbf29ce484222325,
        }
    }
}

impl Hasher for Fnv1a64 {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        // FNV-1a
        const PRIME: u64 = 0x00000100000001B3;
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(PRIME);
        }
    }
}

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
