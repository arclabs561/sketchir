//! Benchmarks for the `store` feature (segstore-backed updatable MinHash index).
//!
//! Run: `cargo bench --features store --bench store`. Without the feature the
//! harness is an empty no-op so the target still compiles. Measures build
//! throughput, warm query latency (per-segment block cached), and the cold
//! "rebuild every segment" cost -- what a delete that clears the whole cache
//! pays, which the targeted-invalidation delete avoids (one segment instead).

#[cfg(not(feature = "store"))]
fn main() {}

#[cfg(feature = "store")]
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

#[cfg(feature = "store")]
const N: usize = 20_000;
#[cfg(feature = "store")]
const FLUSH: usize = 2_000; // ~10 segments

#[cfg(feature = "store")]
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(feature = "store")]
fn text(state: &mut u64) -> String {
    (0..40)
        .map(|_| (b'a' + (xorshift(state) % 26) as u8) as char)
        .collect()
}

#[cfg(feature = "store")]
fn fresh_store(warm: bool) -> (sketchir::store::UpdatableIndex, String) {
    use durability::MemoryDirectory;
    use sketchir::BlockingConfig;
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut store = sketchir::store::UpdatableIndex::open(
        MemoryDirectory::arc(),
        FLUSH,
        BlockingConfig::default(),
    )
    .unwrap();
    for i in 0..N {
        store.add(i as u32, text(&mut s)).unwrap();
    }
    store.checkpoint().unwrap();
    let q = text(&mut s);
    if warm {
        let _ = store.near_duplicates(&q);
    }
    (store, q)
}

#[cfg(feature = "store")]
fn benches(c: &mut Criterion) {
    let mut g = c.benchmark_group("store");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("build", |b| {
        b.iter_batched(
            || (),
            |_| {
                let _ = fresh_store(false);
            },
            BatchSize::SmallInput,
        )
    });

    let (warm, q) = fresh_store(true);
    g.bench_function("search_warm", |b| b.iter(|| warm.near_duplicates(&q)));

    g.bench_function("search_cold_rebuild_all", |b| {
        b.iter_batched(
            || fresh_store(false),
            |(store, q)| store.near_duplicates(&q),
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
fn ingest_fs(c: &mut Criterion) {
    // The extend() win is invisible on MemoryDirectory (flush is free); on a real
    // filesystem the per-item WAL flush is the cost extend amortizes into one batch
    // sync. add-per-item vs extend over the same documents.
    use durability::FsDirectory;
    use sketchir::BlockingConfig;
    let mut g = c.benchmark_group("ingest_fs");
    let n = 4_000usize;
    g.throughput(Throughput::Elements(n as u64));
    let mk = |tag: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!("sketchir-bench-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    };
    g.bench_function("add", |b| {
        b.iter_batched(
            || mk("add"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store = sketchir::store::UpdatableIndex::open(
                    FsDirectory::arc(&p).unwrap(),
                    FLUSH,
                    BlockingConfig::default(),
                )
                .unwrap();
                for i in 0..n {
                    store.add(i as u32, text(&mut s)).unwrap();
                }
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.bench_function("extend", |b| {
        b.iter_batched(
            || mk("extend"),
            |p| {
                let mut s = 0x1234_5678_9abc_def0u64;
                let mut store = sketchir::store::UpdatableIndex::open(
                    FsDirectory::arc(&p).unwrap(),
                    FLUSH,
                    BlockingConfig::default(),
                )
                .unwrap();
                store
                    .extend((0..n).map(|i| (i as u32, text(&mut s))))
                    .unwrap();
                let _ = std::fs::remove_dir_all(&p);
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

#[cfg(feature = "store")]
fn query_hash_amortization(c: &mut Criterion) {
    // The query-once win: near_duplicates computes the query MinHash signature once
    // and probes every per-segment block with it, instead of re-shingling and
    // re-MinHashing the query per segment. A/B at the block level over 10 segments:
    // hash_per_segment (old) vs hash_once (new). Both run identically, so the ratio
    // is the redundant query-hashing the change removed.
    use sketchir::{BlockingConfig, MinHashTextLSH};
    let cfg = BlockingConfig::default();
    let mut s = 0x9999_1234_5678_9abcu64;
    let n_seg = 10;
    let blocks: Vec<MinHashTextLSH> = (0..n_seg)
        .map(|_| {
            let mut lsh = MinHashTextLSH::new(cfg.clone()).unwrap();
            for j in 0..50u32 {
                lsh.insert_text(j.to_string(), text(&mut s));
            }
            lsh
        })
        .collect();
    let q = text(&mut s);
    let mut g = c.benchmark_group("query_hash_amortization_10seg");
    g.bench_function("hash_per_segment", |b| {
        b.iter(|| {
            let mut out = 0usize;
            for lsh in &blocks {
                out += lsh.query(&q).len(); // re-shingles + re-hashes per segment
            }
            out
        })
    });
    g.bench_function("hash_once", |b| {
        b.iter(|| {
            let sig = blocks[0].signature(&q); // hash once, reuse across all blocks
            let mut out = 0usize;
            for lsh in &blocks {
                out += lsh.query_sig(&sig).len();
            }
            out
        })
    });
    g.finish();
}

#[cfg(feature = "store")]
criterion_group!(g, benches, ingest_fs, query_hash_amortization);
#[cfg(feature = "store")]
criterion_main!(g);
