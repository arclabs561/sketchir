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
use std::collections::{HashMap, HashSet};
use std::io::Read;
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
        segs: &[&Vec<(u32, String)>],
        live: &dyn Fn(&u32) -> bool,
    ) -> Vec<(u32, String)> {
        segs.iter()
            .flat_map(|s| s.iter())
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

/// Per-segment LSH blocks keyed by segstore's stable segment id. A sealed add
/// creates one new segment id, so cached blocks for existing segments are reused
/// instead of rebuilding the whole corpus on the next query.
struct Cache {
    by_segment_id: HashMap<u64, Option<Block>>,
}

/// The `kind` tag for a persisted per-segment MinHash LSH sidecar.
const INDEX_KIND: &str = "minhash";
const SIDECAR_MAGIC: &[u8; 8] = b"SKIRIDX1";
const SIDECAR_VERSION: u32 = 1;

#[derive(serde::Serialize, serde::Deserialize)]
struct BlockSidecar {
    block: Block,
}

/// An updatable, durable MinHash near-duplicate index.
pub struct UpdatableIndex {
    inner: SegmentedStore<TextBacking>,
    config: BlockingConfig,
    sidecar_recipe: String,
    cache: RefCell<Cache>,
    /// Segment ids whose on-disk MinHash sidecar was validated or written in
    /// this process, so checkpoint persistence stays O(new segments).
    persisted: RefCell<HashSet<u64>>,
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
            sidecar_recipe: Self::make_sidecar_recipe(&config),
            config,
            cache: RefCell::new(Cache {
                by_segment_id: HashMap::new(),
            }),
            persisted: RefCell::new(HashSet::new()),
        })
    }

    /// Add (or re-add) a document by id.
    pub fn add(&mut self, id: u32, text: impl Into<String>) -> PersistenceResult<()> {
        // A sealed add introduces a new segment id; existing segment ids stay
        // stable, so the cache reuses them and builds only the new one.
        self.inner.add(id, text.into())?;
        Ok(())
    }

    /// Add (or re-add) many documents, syncing the write-ahead log once for the
    /// whole batch instead of once per document. This is the bulk-ingest path (the
    /// corpus-load phase): per-item WAL sync is the dominant cost on a real disk, so
    /// one sync per batch is several times faster than a loop of [`Self::add`]. A
    /// crash mid-batch recovers a consistent prefix (each document is an
    /// independently CRC-checked WAL record).
    pub fn extend(
        &mut self,
        docs: impl IntoIterator<Item = (u32, String)>,
    ) -> PersistenceResult<()> {
        self.inner.extend(docs)?;
        Ok(())
    }

    /// Tombstone a document.
    pub fn delete(&mut self, id: u32) -> PersistenceResult<()> {
        self.inner.delete(id)?;
        // A tombstone only changes the live-set of the segment that holds `id`, so
        // invalidate just that segment's cached block -- not the whole cache --
        // and remove its now-stale sidecar. The live-id guard would reject it
        // anyway; deleting avoids a wasted load on the next query.
        let mut cache = self.cache.borrow_mut();
        let ids = self.inner.segment_ids();
        for (seg_idx, seg) in self.inner.segments().iter().enumerate() {
            if seg.iter().any(|(sid, _)| *sid == id) {
                let seg_id = ids[seg_idx];
                cache.by_segment_id.remove(&seg_id);
                self.persisted.borrow_mut().remove(&seg_id);
                let _ = self
                    .inner
                    .dir()
                    .delete(&self.inner.index_name(seg_id, INDEX_KIND));
            }
        }
        Ok(())
    }

    /// Merge segments (dropping tombstoned docs) and persist a checkpoint.
    pub fn compact(&mut self) -> PersistenceResult<()> {
        self.inner.compact()?;
        self.prune_cache_to_current_segments();
        self.persist_new_segments();
        Ok(())
    }

    /// Persist a checkpoint without merging.
    pub fn checkpoint(&mut self) -> PersistenceResult<()> {
        self.inner.checkpoint()?;
        self.persist_new_segments();
        Ok(())
    }

    /// Run one round of size-tiered compaction, merging similarly-sized segments
    /// so the segment count stays bounded without a full [`compact`](Self::compact).
    pub fn compact_tiers(&mut self) -> PersistenceResult<()> {
        let stats = self.inner.compact_tiers()?;
        if stats.merges > 0 {
            self.prune_cache_to_current_segments();
            self.persist_new_segments();
        }
        Ok(())
    }

    /// Merge only the segments whose live ratio is below `min_live_ratio`,
    /// reclaiming tombstoned documents -- the cheap alternative to a full
    /// [`compact`](Self::compact) when a few segments are delete-heavy.
    pub fn reclaim(&mut self, min_live_ratio: f64) -> PersistenceResult<()> {
        let stats = self.inner.reclaim_tombstones(min_live_ratio)?;
        if stats.merges > 0 {
            self.prune_cache_to_current_segments();
            self.persist_new_segments();
        }
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
        let mut out = self.collect_from_blocks(text, |lsh, ids, sig| {
            lsh.query_sig(sig)
                .into_iter()
                .filter_map(|i| ids.get(i).copied())
                .collect()
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Near-duplicate candidates of `text`, ranked by estimated Jaccard
    /// similarity and deduplicated by document id.
    pub fn near_duplicates_with_similarity(&self, text: &str) -> Vec<(u32, f64)> {
        let mut by_id: HashMap<u32, f64> = HashMap::new();
        for (id, sim) in self.collect_from_blocks(text, |lsh, ids, sig| {
            lsh.query_sig_with_similarity(sig)
                .into_iter()
                .filter_map(|(i, sim)| ids.get(i).copied().map(|id| (id, sim)))
                .collect()
        }) {
            by_id
                .entry(id)
                .and_modify(|existing| *existing = existing.max(sim))
                .or_insert(sim);
        }
        let mut out: Vec<(u32, f64)> = by_id.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    fn collect_from_blocks<T>(
        &self,
        text: &str,
        mut f: impl FnMut(&MinHashTextLSH, &[u32], &crate::MinHashSignature) -> Vec<T>,
    ) -> Vec<T> {
        let mut out: Vec<T> = Vec::new();
        // The query's MinHash signature is config-determined (fixed seed), so it is
        // identical for every per-segment block. Compute it once and reuse it,
        // rather than re-shingling and re-hashing the query once per segment.
        let mut sig = None;
        {
            let segs = self.inner.segments();
            let mut cache = self.cache.borrow_mut();
            // Build only current segments not already cached, loading a persisted
            // sidecar first when one matches the current recipe and live id set.
            let ids = self.inner.segment_ids();
            for (i, seg) in segs.iter().enumerate() {
                let seg_id = ids[i];
                let block = cache
                    .by_segment_id
                    .entry(seg_id)
                    .or_insert_with(|| self.build_or_load(&seg[..], seg_id));
                if let Some((lsh, ids)) = block {
                    let s = sig.get_or_insert_with(|| lsh.signature(text));
                    out.extend(f(lsh, ids, s));
                }
            }
        }
        let buffered = self.inner.buffer().to_vec();
        if let Some((lsh, ids)) = self.build_live_index(&buffered) {
            let s = sig.get_or_insert_with(|| lsh.signature(text));
            out.extend(f(&lsh, &ids, s));
        }
        out
    }

    fn prune_cache_to_current_segments(&self) {
        let current: HashSet<u64> = self.inner.segment_ids().iter().copied().collect();
        self.cache
            .borrow_mut()
            .by_segment_id
            .retain(|id, _| current.contains(id));
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

    /// Load segment `seg_id`'s persisted MinHash sidecar, or build it over the
    /// segment's live documents and persist it for the next restart.
    fn build_or_load(&self, seg: &[(u32, String)], seg_id: u64) -> Option<Block> {
        if let Some(block) = self.load_sidecar(seg, seg_id) {
            self.persisted.borrow_mut().insert(seg_id);
            return Some(block);
        }
        let block = self.build_live_index(seg)?;
        self.persist_sidecar(&block, seg, seg_id);
        Some(block)
    }

    /// Load a sidecar only if its recipe matches and its ids match the segment's
    /// current live ids. A stale sidecar can never serve a tombstoned document.
    fn load_sidecar(&self, seg: &[(u32, String)], seg_id: u64) -> Option<Block> {
        let name = self.inner.index_name(seg_id, INDEX_KIND);
        if !self.inner.dir().exists(&name) {
            return None;
        }
        let mut bytes = Vec::new();
        self.inner
            .dir()
            .open_file(&name)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        let block_bytes = self.decode_sidecar(&bytes)?;
        let sidecar: BlockSidecar = postcard::from_bytes(block_bytes).ok()?;
        let live = self.live_ids(seg);
        let ids = &sidecar.block.1;
        if ids.len() == live.len() && ids.iter().all(|id| live.contains(id)) {
            Some(sidecar.block)
        } else {
            None
        }
    }

    /// Persist a built per-segment MinHash block as its sidecar. Best-effort: a
    /// failed write leaves the in-memory block usable and simply rebuilds later.
    fn persist_sidecar(&self, block: &Block, seg: &[(u32, String)], seg_id: u64) {
        let sidecar = BlockSidecar {
            block: (block.0.clone(), self.live_ids_vec(seg)),
        };
        if let Ok(index) = postcard::to_allocvec(&sidecar) {
            let Some(bytes) = self.encode_sidecar(&index) else {
                return;
            };
            if self
                .inner
                .dir()
                .atomic_write(&self.inner.index_name(seg_id, INDEX_KIND), &bytes)
                .is_ok()
            {
                self.persisted.borrow_mut().insert(seg_id);
            }
        }
    }

    fn live_ids(&self, seg: &[(u32, String)]) -> HashSet<u32> {
        seg.iter()
            .filter_map(|(id, _)| self.inner.is_live(id).then_some(*id))
            .collect()
    }

    fn live_ids_vec(&self, seg: &[(u32, String)]) -> Vec<u32> {
        let mut ids: Vec<u32> = seg
            .iter()
            .filter_map(|(id, _)| self.inner.is_live(id).then_some(*id))
            .collect();
        ids.sort_unstable();
        ids
    }

    fn make_sidecar_recipe(config: &BlockingConfig) -> String {
        format!(
            "sketchir-store-minhash-v1;\
             codec=postcard-minhash-text-lsh-v1;\
             num_hashes_per_band={};num_bands={};ngram_size={};char_ngrams={}",
            config.num_hashes_per_band, config.num_bands, config.ngram_size, config.char_ngrams
        )
    }

    fn encode_sidecar(&self, index: &[u8]) -> Option<Vec<u8>> {
        let recipe = self.sidecar_recipe.as_bytes();
        let recipe_len = u32::try_from(recipe.len()).ok()?;
        let mut bytes = Vec::with_capacity(16 + recipe.len() + index.len());
        bytes.extend_from_slice(SIDECAR_MAGIC);
        bytes.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
        bytes.extend_from_slice(&recipe_len.to_le_bytes());
        bytes.extend_from_slice(recipe);
        bytes.extend_from_slice(index);
        Some(bytes)
    }

    fn decode_sidecar<'a>(&self, bytes: &'a [u8]) -> Option<&'a [u8]> {
        if bytes.len() < 16 {
            return None;
        }
        if &bytes[..8] != SIDECAR_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        if version != SIDECAR_VERSION {
            return None;
        }
        let recipe_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let recipe_start = 16usize;
        let recipe_end = recipe_start.checked_add(recipe_len)?;
        if bytes.len() < recipe_end {
            return None;
        }
        if &bytes[recipe_start..recipe_end] != self.sidecar_recipe.as_bytes() {
            return None;
        }
        Some(&bytes[recipe_end..])
    }

    /// Persist sidecars for sealed segments that lack a current one. This is
    /// incremental: already validated/written segment ids are skipped.
    fn persist_new_segments(&self) {
        let ids = self.inner.segment_ids();
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        self.persisted.borrow_mut().retain(|id| id_set.contains(id));
        for (i, seg) in self.inner.segments().iter().enumerate() {
            let seg_id = ids[i];
            if self.persisted.borrow().contains(&seg_id) {
                continue;
            }
            if self.load_sidecar(&seg[..], seg_id).is_some() {
                self.persisted.borrow_mut().insert(seg_id);
                continue;
            }
            if let Some(block) = self.build_live_index(&seg[..]) {
                self.persist_sidecar(&block, &seg[..], seg_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use durability::MemoryDirectory;

    const A: &str = "the quick brown fox jumps over the lazy dog";
    const B: &str = "lorem ipsum dolor sit amet consectetur adipiscing elit";
    const C: &str = "the quick brown fox jumps over the lazy dog";

    fn read_file(dir: &Arc<dyn Directory>, name: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        dir.open_file(name)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    fn checkpointed_store(dir: Arc<dyn Directory>) -> (String, Vec<u8>) {
        let mut store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        store.add(1, A).unwrap();
        store.add(2, C).unwrap();
        store.add(3, B).unwrap();
        store.add(4, "a separate unrelated document").unwrap();
        store.checkpoint().unwrap();
        let seg_id = store.inner.segment_ids()[0];
        let name = store.inner.index_name(seg_id, INDEX_KIND);
        let bytes = read_file(store.inner.dir(), &name);
        (name, bytes)
    }

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

    #[test]
    fn near_duplicates_with_similarity_covers_segments_and_buffer() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        store.add(1, A).unwrap();
        store.add(2, C).unwrap();
        store.add(3, A).unwrap();

        assert_eq!(store.near_duplicates(A), vec![1, 2, 3]);
        let ranked = store.near_duplicates_with_similarity(A);
        assert_eq!(
            ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(ranked.iter().all(|(_, sim)| (sim - 1.0).abs() < 1e-9));
    }

    #[test]
    fn checkpoint_persists_sidecars_and_reopen_loads_them() {
        let dir = MemoryDirectory::arc();
        {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
            store.add(1, A).unwrap();
            store.add(2, C).unwrap();
            store.add(3, B).unwrap();
            store.add(4, "a separate unrelated document").unwrap();
            store.checkpoint().unwrap();

            let ids: Vec<u64> = store.inner.segment_ids().to_vec();
            assert!(
                !ids.is_empty(),
                "4 docs at flush 2 seal at least one segment"
            );
            for id in &ids {
                assert!(
                    store
                        .inner
                        .dir()
                        .exists(&store.inner.index_name(*id, INDEX_KIND)),
                    "segment {id} must have a persisted sidecar after checkpoint"
                );
            }
        }

        let store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        let dups = store.near_duplicates(A);
        assert!(
            dups.contains(&1) && dups.contains(&2),
            "search over loaded sidecars returns duplicate candidates"
        );
    }

    #[test]
    fn compact_persists_sidecar_for_merged_segment() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        store.add(1, A).unwrap();
        store.add(2, C).unwrap();
        store.add(3, B).unwrap();
        store.add(4, "a separate unrelated document").unwrap();
        store.compact().unwrap();

        let ids: Vec<u64> = store.inner.segment_ids().to_vec();
        assert_eq!(ids.len(), 1, "compact should merge the sealed segments");
        assert!(
            store
                .inner
                .dir()
                .exists(&store.inner.index_name(ids[0], INDEX_KIND)),
            "merged segment should have a sidecar immediately after compact"
        );
    }

    #[test]
    fn compact_prunes_cached_segment_blocks() {
        let dir = MemoryDirectory::arc();
        let mut store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        store.add(1, A).unwrap();
        store.add(2, C).unwrap();
        store.add(3, B).unwrap();
        store.add(4, "a separate unrelated document").unwrap();

        let before_ids = store.inner.segment_ids().to_vec();
        assert!(
            before_ids.len() >= 2,
            "test setup should create multiple sealed segments"
        );
        let _ = store.near_duplicates(A);
        assert_eq!(
            store.cache.borrow().by_segment_id.len(),
            before_ids.len(),
            "warm query should cache each sealed segment"
        );

        store.compact().unwrap();

        let after_ids = store.inner.segment_ids().to_vec();
        assert_eq!(
            after_ids.len(),
            1,
            "compact should merge the sealed segments"
        );
        assert!(
            store
                .cache
                .borrow()
                .by_segment_id
                .keys()
                .all(|id| after_ids.contains(id)),
            "cache should not retain blocks for compacted-away segment ids"
        );
    }

    #[test]
    fn minhash_sidecar_recipe_mismatch_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, before) = checkpointed_store(dir.clone());
        assert_eq!(
            &before[..SIDECAR_MAGIC.len()],
            SIDECAR_MAGIC,
            "new sidecars carry the sketchir MinHash envelope"
        );

        let store = UpdatableIndex::open(dir.clone(), 2, BlockingConfig::high_precision()).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "sidecar built with default blocking config must not load under high-precision config"
        );
        assert!(
            !store.near_duplicates(A).is_empty(),
            "mismatched sidecar falls back to rebuild"
        );

        let after = read_file(store.inner.dir(), &name);
        assert_ne!(before, after, "rebuild overwrites the stale-recipe sidecar");
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar now matches the current recipe"
        );
    }

    #[test]
    fn minhash_sidecar_envelope_rejects_corrupt_headers() {
        let store =
            UpdatableIndex::open(MemoryDirectory::arc(), 2, BlockingConfig::default()).unwrap();
        let block = b"block-bytes";
        let bytes = store.encode_sidecar(block).unwrap();
        assert_eq!(store.decode_sidecar(&bytes), Some(block.as_slice()));

        assert!(store.decode_sidecar(&bytes[..8]).is_none());

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(store.decode_sidecar(&bad_magic).is_none());

        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&(SIDECAR_VERSION + 1).to_le_bytes());
        assert!(store.decode_sidecar(&bad_version).is_none());

        let mut bad_recipe_len = bytes.clone();
        bad_recipe_len[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(store.decode_sidecar(&bad_recipe_len).is_none());

        let mut bad_recipe = bytes.clone();
        bad_recipe[16] ^= 0x01;
        assert!(store.decode_sidecar(&bad_recipe).is_none());
    }

    #[test]
    fn minhash_sidecar_invalid_payload_rebuilds() {
        let dir = MemoryDirectory::arc();
        let (name, _) = checkpointed_store(dir.clone());
        {
            let store = UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
            let corrupt = store
                .encode_sidecar(b"not-a-postcard-minhash-block")
                .unwrap();
            store.inner.dir().atomic_write(&name, &corrupt).unwrap();
        }

        let store = UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "valid envelope with invalid MinHash payload is rejected"
        );
        assert!(
            !store.near_duplicates(A).is_empty(),
            "invalid payload falls back to rebuild"
        );
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_some(),
            "rebuilt sidecar loads after the fallback"
        );
    }

    #[test]
    fn deleted_id_does_not_resurface_through_a_sidecar() {
        let dir = MemoryDirectory::arc();
        {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
            store.add(1, A).unwrap();
            store.add(2, C).unwrap();
            store.add(3, B).unwrap();
            store.checkpoint().unwrap();
            store.delete(2).unwrap();
            store.checkpoint().unwrap();
        }

        let store = UpdatableIndex::open(dir, 2, BlockingConfig::default()).unwrap();
        let dups = store.near_duplicates(A);
        assert!(
            !dups.contains(&2),
            "deleted id 2 must not resurface from a persisted sidecar"
        );
        assert!(dups.contains(&1), "live duplicate should remain searchable");
    }

    #[test]
    fn checkpoint_after_replayed_delete_rewrites_stale_sidecar() {
        let dir = MemoryDirectory::arc();
        let (name, stale_bytes) = {
            let mut store =
                UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
            store.add(1, A).unwrap();
            store.add(2, C).unwrap();
            store.add(3, B).unwrap();
            store.checkpoint().unwrap();

            let seg_id = store.inner.segment_ids()[0];
            let name = store.inner.index_name(seg_id, INDEX_KIND);
            let bytes = read_file(store.inner.dir(), &name);

            // Simulate a crash after the delete is durably logged but before
            // `UpdatableIndex::delete` removes the now-stale sidecar.
            store.inner.delete(2).unwrap();
            (name, bytes)
        };

        let mut store = UpdatableIndex::open(dir.clone(), 2, BlockingConfig::default()).unwrap();
        let seg_id = store.inner.segment_ids()[0];
        assert!(
            store
                .load_sidecar(&store.inner.segments()[0][..], seg_id)
                .is_none(),
            "replayed tombstone must make the old sidecar stale"
        );

        store.checkpoint().unwrap();

        let rewritten = read_file(&dir, &name);
        assert_ne!(
            rewritten, stale_bytes,
            "checkpoint should rewrite stale sidecars even before search"
        );
        let block = store
            .load_sidecar(&store.inner.segments()[0][..], seg_id)
            .expect("rewritten sidecar should be valid");
        let sig = block.0.signature(A);
        let dups: Vec<u32> = block
            .0
            .query_sig(&sig)
            .into_iter()
            .filter_map(|i| block.1.get(i).copied())
            .collect();
        assert!(
            !dups.contains(&2),
            "rewritten sidecar must exclude the replayed delete"
        );
        assert!(
            dups.contains(&1),
            "rewritten sidecar should keep live ids from the segment"
        );
    }
}
