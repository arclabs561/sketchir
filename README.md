# sketchir

[![crates.io](https://img.shields.io/crates/v/sketchir.svg)](https://crates.io/crates/sketchir)
[![Documentation](https://docs.rs/sketchir/badge.svg)](https://docs.rs/sketchir)
[![CI](https://github.com/arclabs561/sketchir/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sketchir/actions/workflows/ci.yml)

Sketching primitives for IR: MinHash, SimHash, and LSH indexes for
near-duplicate detection and approximate similarity search.

```toml
[dependencies]
sketchir = "0.5"
```

## Best starting points

- **Near-duplicate detection**: `MinHashTextLSH` + `BlockingConfig`
- **SimHash fingerprints**: `SimHashFingerprint` / `SimHashLSH`
- **Dense-vector LSH (batch)**: `LSHIndex` -- multi-table random projection, add/build/search lifecycle
- **Dense-vector LSH (incremental)**: `DenseSimHashLSH` -- SimHash + Hamming-1 probing, no build step

## Tuning knobs (`BlockingConfig`)

| Param | Typical | Tradeoff |
|---|---|---|
| `ngram_size` | 3-9 chars | Smaller = more sensitive to noise; Larger = stricter. |
| `num_bands * num_hashes_per_band` | 100-256 | More = better Jaccard estimation, higher storage. |
| `num_bands` | 20-50 | Controls recall/precision curve (S-curve). |

## Example (MinHash blocking)

```rust
use sketchir::{BlockingConfig, MinHashTextLSH};

let cfg = BlockingConfig::default();
let mut index = MinHashTextLSH::new(cfg).unwrap();
index.insert_text("a", "hello world");
index.insert_text("b", "hello  world!");

let pairs = index.candidate_pairs();
assert!(!pairs.is_empty());
```

## Examples

Runnable examples live in [`examples/`](examples/):

- `dedup_documents` indexes a document set and returns candidate near-duplicate pairs via MinHash LSH blocking, the corpus-hygiene use (the engine behind a dedup tool like siftr).
- `minhash_jaccard_validation` checks the MinHash Jaccard estimate against the exact Jaccard index, confirming the sketch is accurate before trusting it at scale.

## Updatable index (`store` feature)

`store::UpdatableIndex` wraps the MinHash near-duplicate index in a durable,
segmented store ([`segstore`](https://crates.io/crates/segstore)): incremental
add/delete, a write-ahead log, checkpoint, compaction, and crash recovery. The
per-segment LSH blocks are cached and persisted as sidecars, so restart loads
unchanged MinHash blocks instead of rebuilding them.
The source text batches are still loaded by the current `segstore` open path;
the sidecars avoid rebuilding LSH blocks.
Opt-in; the default build does not depend on segstore.

## License

MIT OR Apache-2.0
