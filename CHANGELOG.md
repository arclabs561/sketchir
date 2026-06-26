# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
