# sketchir

Sketching primitives for IR: MinHash/SimHash/LSH-style signatures.

## Tuning Knobs

| Param | Typical | Tradeoff |
|---|---|---|
| `shingle_len` | 5-9 chars | Smaller = more sensitive to noise; Larger = stricter. |
| `num_perm` | 128-256 | More = better Jaccard estimation, higher storage. |
| `lsh_bands` | 20-50 | Controls recall/precision curve (S-curve). |

## What it is

`sketchir` is the index-only layer for:
- near-duplicate detection (shingles + MinHash)
- text fingerprinting (SimHash)
- approximate similarity candidate generation (LSH)

## Best starting points

- **Near-duplicate detection**: `MinHashTextLSH` + `BlockingConfig`
- **SimHash fingerprints**: `SimHashFingerprint` / `SimHashLSH`
- **Generic LSH interface**: `LSHIndex`

## Example (MinHash blocking)

```rust
use sketchir::{BlockingConfig, MinHashTextLSH, TextItem};

let items = vec![
    TextItem { id: "a".into(), text: "hello world".into() },
    TextItem { id: "b".into(), text: "hello  world!".into() },
];

let cfg = BlockingConfig::default();
let mut index = MinHashTextLSH::new(cfg);
index.add_all(&items);

let candidates = index.candidates_for(&items[0]);
assert!(!candidates.is_empty());
```

## License

MIT OR Apache-2.0
