//! Updatable, durable near-duplicate index (MinHash LSH) via `segstore`.
//!
//! Enabled by the optional `store` feature. [`MinHashTextLSH`] is built per
//! batch; this wraps the corpus in a segstore `SegmentedStore` so documents can
//! be added and deleted incrementally with a write-ahead log + checkpoint +
//! compaction, and the blocking index survives a restart.
//!
//! Each segment stores the source `(id, text)` pairs; a real `MinHashTextLSH`
//! over the live documents of each segment answers a query. Because the LSH
//! returns insertion-order indices rather than the caller's ids, this maps them
//! back through a per-segment id vector. The `BlockingConfig` is a query-time
//! parameter (not persisted), so it is supplied at [`UpdatableIndex::open`].

use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{BlockingConfig, MinHashTextLSH};

/// segstore payload: items are document texts, a segment is a batch of source
/// texts (the LSH is rebuilt from the live ones per query).
struct TextBacking;

impl Store for TextBacking {
    type Id = u32;
    type Item = String;
    type Segment = Vec<(u32, String)>;

    fn build_segment(&self, batch: &[(u32, String)]) -> Vec<(u32, String)> {
        batch.to_vec()
    }

    fn merge_segments(
        &self,
        segs: &[Vec<(u32, String)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, String)> {
        segs.iter()
            .flatten()
            .filter(|(id, _)| live(id))
            .cloned()
            .collect()
    }
}

/// An updatable, durable MinHash near-duplicate index.
pub struct UpdatableIndex {
    inner: SegmentedStore<TextBacking>,
    config: BlockingConfig,
}

impl UpdatableIndex {
    /// Open (or recover) an index under `dir`, using `config` for the per-segment
    /// MinHash blocking. Up to `flush_threshold` documents are buffered before a
    /// new immutable segment is sealed.
    pub fn open(
        dir: Arc<dyn Directory>,
        flush_threshold: usize,
        config: BlockingConfig,
    ) -> PersistenceResult<Self> {
        Ok(Self {
            inner: SegmentedStore::open(dir, TextBacking, flush_threshold)?,
            config,
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        self.inner.add(id, text.into())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Document ids that are near-duplicate candidates of `text`, unioned over
    /// every live document.
    pub fn near_duplicates(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for seg in self.inner.segments() {
            out.extend(self.candidates_in(seg, text));
        }
        let buffered = self.inner.buffer().to_vec();
        out.extend(self.candidates_in(&buffered, text));
        out.sort_unstable();
        out.dedup();
        out
    }

    fn candidates_in(&self, batch: &[(u32, String)], text: &str) -> Vec<u32> {
        let mut lsh = match MinHashTextLSH::new(self.config.clone()) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };
        let mut ids: Vec<u32> = Vec::new();
        for (id, doc) in batch {
            if self.inner.is_live(id) {
                lsh.insert_text(id.to_string(), doc);
                ids.push(*id);
            }
        }
        if ids.is_empty() {
            return Vec::new();
        }
        // The LSH returns insertion-order indices; map them back to caller ids.
        lsh.query(text)
            .into_iter()
            .filter_map(|i| ids.get(i).copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    const A: &str = "the quick brown fox jumps over the lazy dog";
    const B: &str = "lorem ipsum dolor sit amet consectetur adipiscing elit";

    #[test]
    fn add_delete_compact_recover_through_real_lsh() {
        let dir = MemoryDirectory::arc();
        {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
            store.add(1, A).unwrap();
            store.add(2, A).unwrap(); // identical to doc 1; flush
            store.add(3, B).unwrap(); // unrelated; buffered

            let dups = store.near_duplicates(A);
            assert!(
                dups.contains(&1) && dups.contains(&2),
                "identical docs are near-duplicates"
            );
            assert!(!dups.contains(&3), "unrelated doc is not");

            store.delete(2).unwrap();
            assert!(
                !store.near_duplicates(A).contains(&2),
                "deleted doc drops out"
            );

            store.compact().unwrap();
            let dups = store.near_duplicates(A);
            assert!(
                dups.contains(&1) && !dups.contains(&2),
                "compaction preserves the result"
            );
        }
        let store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        let dups = store.near_duplicates(A);
        assert!(
            dups.contains(&1) && !dups.contains(&2),
            "recovery preserves the result"
        );
    }
}
