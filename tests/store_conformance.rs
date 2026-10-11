//! segstore's shared conformance suite against sketchir's updatable index.
#![cfg(feature = "store")]

use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::conformance::{run_all, ConformanceStore};
use sketchir::blocking::BlockingConfig;
use sketchir::store::UpdatableIndex;

struct Adapter(UpdatableIndex);

/// 24 pseudo-random 6-letter words: unrelated documents share almost no
/// shingles, so their MinHash similarity is near 0 while a document's own is 1.
fn text(seed: u64) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    (0..24)
        .map(|_| {
            (0..6)
                .map(|_| char::from(b'a' + (next() % 26) as u8))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl ConformanceStore for Adapter {
    type Doc = String;

    fn open(dir: Arc<dyn Directory>) -> PersistenceResult<Self> {
        UpdatableIndex::open(dir, 3, BlockingConfig::default()).map(Self)
    }
    fn add(&mut self, id: u32, doc: String) -> PersistenceResult<()> {
        self.0.add(id, doc)
    }
    fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.0.delete(id)
    }
    fn compact(&mut self) -> PersistenceResult<()> {
        self.0.compact()
    }
    fn compact_tiers(&mut self) -> PersistenceResult<()> {
        self.0.compact_tiers()
    }
    fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.0.reclaim(min_live_ratio)
    }
    fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.0.checkpoint()
    }
    fn doc(seed: u64) -> String {
        text(seed)
    }
    /// Near-duplicates whose estimated Jaccard similarity is essentially 1.
    fn matching(&self, doc: &String) -> Vec<u32> {
        self.0
            .near_duplicates_with_similarity(doc)
            .into_iter()
            .filter(|&(_, similarity)| similarity > 0.95)
            .map(|(id, _)| id)
            .collect()
    }
}

#[test]
fn updatable_index_meets_segstore_conformance() {
    run_all::<Adapter>();
}
