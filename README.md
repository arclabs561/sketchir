# sketchir

[![crates.io](https://img.shields.io/crates/v/sketchir.svg)](https://crates.io/crates/sketchir)
[![Documentation](https://docs.rs/sketchir/badge.svg)](https://docs.rs/sketchir)
[![CI](https://github.com/arclabs561/sketchir/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sketchir/actions/workflows/ci.yml)

Sketching primitives for IR.

## What it is

`sketchir` is the index-only layer for:
- near-duplicate detection (shingles + MinHash)
- text fingerprinting (SimHash)
- approximate similarity candidate generation (LSH)

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

## License

MIT OR Apache-2.0
