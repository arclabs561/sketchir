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
pub use dense_simhash::{DenseSimHash, DenseSimHashLSH};
pub use lsh::{LSHIndex, LSHParams, MinHashLSH, SimHashLSH};
pub use minhash::{MinHash, MinHashSignature};
pub use simhash::{SimHash, SimHashFingerprint};
