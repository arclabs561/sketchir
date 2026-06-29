# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.4] - 2026-06-28

### Added

- `store::UpdatableIndex::extend(docs)`: bulk ingest that syncs the write-ahead log
  once per batch instead of once per document. ~4.9x faster than a loop of `add`
  for a corpus load on a real filesystem (bench `ingest_fs`: 8.2ms vs 1.7ms / 4000
  docs).
- `MinHashTextLSH::signature(text)` + `query_sig(&signature)`: split a query into
  its (segment-independent) MinHash signature and the band probe.

### Changed

- A multi-segment `near_duplicates` query now computes the query MinHash signature
  once and probes every per-segment block with it, instead of re-shingling and
  re-MinHashing the query per segment. Results are identical (the MinHash seed is
  fixed). ~7.4x faster over 10 segments (bench `query_hash_amortization`: 196us vs
  27us).
- The `store` feature now requires `segstore = "0.3"`; the internal `merge_segments`
  takes `&[&Segment]` (segstore 0.3's by-reference signature).

## [0.5.3] - 2026-06-27

### Changed

- A `delete` now invalidates only the cached block of the segment that holds the
  id, not the whole cache, so one delete no longer forces every segment to
  rebuild on the next query.

## [0.5.2] - 2026-06-27

### Added

- `store::UpdatableIndex::compact_tiers()`: one round of size-tiered compaction
  (merge similarly-sized segments), keeping segment count bounded without a full
  `compact()`.

## [0.5.1] - 2026-06-27

### Added

- `store::UpdatableIndex::reclaim(min_live_ratio)` and `space_amplification()`
  (via the new `Store::live_len`): cheap tombstone reclamation, merging only the
  delete-heavy segments instead of a full compaction.

## [0.5.0] - 2026-06-27

### Changed

- `store::UpdatableIndex` now caches each segment's MinHash block by the segment's
  stable `Arc` identity (via segstore 0.2), so a mutation rebuilds only the new or
  changed segments instead of the whole corpus on the next query.
- Requires `segstore` 0.2 (only affects the optional `store` feature; the on-disk
  store format changed, so a `store` index written by 0.4.x is not read by 0.5.0).

## [0.4.1] - 2026-06-26

### Fixed

- `store::UpdatableIndex` caches the per-segment MinHash LSH blocks and rebuilds
  them only on mutation (add/delete/compact), instead of rebuilding every segment
  on every query.

## [0.4.0] - 2026-06-26

### Added
- Optional `store` feature: `store::UpdatableIndex`, an updatable, durable
  MinHash near-duplicate index backed by
  [`segstore`](https://crates.io/crates/segstore) (write-ahead log, checkpoint,
  compaction, crash recovery). Opt-in; the default build does not depend on
  segstore.

## [0.3.2] - 2026-04-20

### Added
- Multibit locality-sensitive hashing module.
- Hyperplane LSH for dense vector hashing.
- Cross-polytope LSH hasher.
- `dedup_documents` example.
- Probabilistic property tests and determinism tests.

### Changed
- Optimized simhash fingerprint computation.
- Consolidated the duplicate `dot` function.
- Validated inputs and trimmed the API.
- Dropped all dependencies.

### Fixed
- Flaky near-duplicate recall test.
- Determinism issues.
- Clippy `needless_range_loop` warnings.
