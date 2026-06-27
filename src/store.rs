//! Updatable, durable near-duplicate index (MinHash LSH) via `segstore`.
//!
//! Enabled by the optional `store` feature. [`MinHashTextLSH`] is built per
//! batch; this wraps the corpus in a segstore `SegmentedStore` so documents can
//! be added and deleted incrementally with a write-ahead log + checkpoint +
//! compaction, and the blocking index survives a restart.
//!
//! Each segment stores the source `(id, text)` pairs; a real `MinHashTextLSH`
//! over the live documents of each segment is built and **cached**, rebuilt only
//! when the index is mutated (an add that seals a segment, a delete, or a
//! compaction), not on every query. The small unflushed buffer is built per
//! query. Because the LSH returns insertion-order indices rather than the
//! caller's ids, each cached index carries a parallel id vector to map them back.
//! The `BlockingConfig` is a query-time parameter (not persisted), so it is
//! supplied at [`UpdatableIndex::open`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use durability::{Directory, PersistenceResult};
use segstore::{SegmentedStore, Store};

use crate::{BlockingConfig, MinHashTextLSH};

/// segstore payload: items are document texts, a segment is a batch of source
/// texts (the LSH is built + cached from the live ones).
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

    fn segment_len(&self, seg: &Vec<(u32, String)>) -> usize {
        seg.len()
    }

    fn live_len(&self, seg: &Vec<(u32, String)>, live: &dyn Fn(&u32) -> bool) -> Option<usize> {
        Some(seg.iter().filter(|(id, _)| live(id)).count())
    }
}

/// A built per-segment LSH plus the insertion-order id map needed to translate
/// its results back to caller ids.
type Block = (MinHashTextLSH, Vec<u32>);

/// Per-segment LSH blocks keyed by the segment's stable `Arc` identity. Because
/// segstore keeps an unchanged segment's `Arc` across mutations, a sealed add
/// only builds the one new segment's block (the rest are reused) instead of
/// rebuilding the whole corpus -- the dominant cost in an add-then-query loop.
struct Cache {
    by_ptr: HashMap<usize, Option<Block>>,
}

/// An updatable, durable MinHash near-duplicate index.
pub struct UpdatableIndex {
    inner: SegmentedStore<TextBacking>,
    config: BlockingConfig,
    cache: RefCell<Cache>,
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
            cache: RefCell::new(Cache {
                by_ptr: HashMap::new(),
            }),
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        // A sealed add introduces a new segment (a new Arc identity); existing
        // segments keep theirs, so the cache reuses them and builds only the new one.
        self.inner.add(id, text.into())?;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone changes the live filter used to build every segment's block,
        // so invalidate the cache (deletes are far rarer than adds).
        self.cache.borrow_mut().by_ptr.clear();
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned documents -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        self.inner.reclaim_tombstones(min_live_ratio)?;
        Ok(())
    }

    /// Storage amplification: stored documents divided by live documents (`1.0`
    /// with no tombstones, higher as deletes accumulate).
    pub fn space_amplification(&self) -> Option<f64> {
        self.inner.space_amplification()
    }

    /// Document ids that are near-duplicate candidates of `text`, unioned over
    /// every live document.
    pub fn near_duplicates(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Drop cached blocks for segments no longer present (post-compaction).
            let current: std::collections::HashSet<usize> =
                segs.iter().map(|a| Arc::as_ptr(a) as usize).collect();
            cache.by_ptr.retain(|key, _| current.contains(key));
            // Build only segments not already cached (i.e. new ones).
            for seg in segs {
                let key = Arc::as_ptr(seg) as usize;
                cache
                    .by_ptr
                    .entry(key)
                    .or_insert_with(|| self.build_live_index(&seg[..]));
            }
            for block in cache.by_ptr.values().flatten() {
                out.extend(query_block(block, text));
            }
        }
        let buffered = self.inner.buffer().to_vec();
        if let Some(block) = self.build_live_index(&buffered) {
            out.extend(query_block(&block, text));
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Build a MinHash LSH over the live documents of `batch` (None if empty),
    /// keeping the insertion-order id map alongside.
    fn build_live_index(&self, batch: &[(u32, String)]) -> Option<Block> {
        let mut lsh = match MinHashTextLSH::new(self.config.clone()) {
            Ok(l) => l,
            Err(_) => return None,
        };
        let mut ids: Vec<u32> = Vec::new();
        for (id, doc) in batch {
            if self.inner.is_live(id) {
                lsh.insert_text(id.to_string(), doc);
                ids.push(*id);
            }
        }
        if ids.is_empty() {
            return None;
        }
        Some((lsh, ids))
    }
}

/// Run a query against a cached block, mapping insertion-order results to ids.
fn query_block((lsh, ids): &Block, text: &str) -> Vec<u32> {
    lsh.query(text)
        .into_iter()
        .filter_map(|i| ids.get(i).copied())
        .collect()
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
            assert_eq!(store.near_duplicates(A), dups, "cached query is stable");

            store.delete(2).unwrap();
            assert!(
                !store.near_duplicates(A).contains(&2),
                "delete invalidates the cache; deleted doc drops out"
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
