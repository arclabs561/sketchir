# sketchir examples

These examples are small demonstrations on synthetic text and integer sets.
Their output, including the error figures from `minhash_jaccard_validation`
and the timings from `store_reopen_diagnostics`, is not benchmark evidence;
use `cargo bench` for measurement.

## Where to start

| I want to... | Run |
|---|---|
| Find near-duplicate documents with MinHash LSH blocking | `dedup_documents` |
| See how close MinHash Jaccard estimates are to exact Jaccard | `minhash_jaccard_validation` |
| Add, checkpoint, delete, and reopen a durable near-duplicate index | `updatable_store` |
| Compare first-query time after reopening with and without persisted sidecars | `store_reopen_diagnostics` |

## Default features

| Example | What it shows |
|---|---|
| `dedup_documents` | Inserts six short documents into `MinHashTextLSH`, prints the candidate near-duplicate pairs, and counts documents left after dropping the second member of each pair |
| `minhash_jaccard_validation` | Prints MinHash estimates for identical and disjoint sets, then the mean absolute error against exact Jaccard over 40 set pairs for k = 64, 256, 1024, next to 1/sqrt(k) |

```sh
cargo run --release --example dedup_documents
cargo run --release --example minhash_jaccard_validation
```

## Feature-gated examples

| Feature | Example | What it shows |
|---|---|---|
| `store` | `updatable_store` | `store::UpdatableIndex` in memory: indexes four documents, queries ranked near-duplicates, deletes one, reopens from the same directory, and checks the deletion survived |
| `store` | `store_reopen_diagnostics` | Builds a 1,000-document checkpointed store on disk twice, deletes the MinHash sidecars in one copy, and prints the first `SnapshotIndex` query time for each path; asserts both paths return the same candidates |

```sh
cargo run --release --features store --example updatable_store
cargo run --release --features store --example store_reopen_diagnostics
```
