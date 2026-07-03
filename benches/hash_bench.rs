use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use sketchir::{
    hyperplane::HyperplaneHasher,
    lsh::MinHashLSH,
    minhash::MinHash,
    multibit::{MultibitConfig, MultibitLSH},
    simhash::simhash_fingerprint,
};
use std::collections::HashSet;

// ============================================================================
// Helpers
// ============================================================================

fn make_f32_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_feature_hashes(n: usize, seed: u64) -> Vec<(u64, f32)> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let h = state;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = (state >> 33) as u32 as f32 / u32::MAX as f32;
            (h, w)
        })
        .collect()
}

fn make_set(shared: usize, unique: usize, doc: u64) -> HashSet<u64> {
    let mut items: HashSet<u64> = (0..shared as u64).collect();
    for i in 0..unique as u64 {
        items.insert(1_000_000 + doc * unique as u64 + i);
    }
    items
}

// ============================================================================
// SimHash fingerprint (sparse feature vector)
// ============================================================================

fn bench_simhash_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("simhash_fingerprint");
    for &n_features in &[16usize, 64, 256] {
        let features = make_feature_hashes(n_features, 42);
        group.throughput(Throughput::Elements(n_features as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            &n_features,
            |b, _| {
                b.iter(|| simhash_fingerprint(black_box(&features)));
            },
        );
    }
    group.finish();
}

// ============================================================================
// HyperplaneHasher -- dense vector fingerprinting
// ============================================================================

fn bench_hyperplane_hasher(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperplane_hash");
    for &dim in &[384usize, 768] {
        let hasher = HyperplaneHasher::new(dim, 64, 42).unwrap();
        let v = make_f32_vec(dim, 1);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| hasher.hash(black_box(&v)).unwrap());
        });
    }
    group.finish();
}

// ============================================================================
// MinHash -- set fingerprinting
// ============================================================================

fn bench_minhash(c: &mut Criterion) {
    let mut group = c.benchmark_group("minhash_signature");
    let mh = MinHash::new(128).unwrap();
    for &n_items in &[16usize, 64, 256] {
        let items: HashSet<u64> = (0..n_items as u64).collect();
        group.throughput(Throughput::Elements(n_items as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_items), &n_items, |b, _| {
            b.iter(|| mh.signature(black_box(&items)));
        });
    }
    group.finish();
}

fn bench_minhash_lsh_query(c: &mut Criterion) {
    let mh = MinHash::new(128).unwrap();
    let mut index = MinHashLSH::new(32, 4).unwrap();
    let query_items = make_set(24, 16, 0);
    let query_sig = mh.signature(&query_items);

    for doc in 0..5_000u64 {
        let items = if doc % 25 == 0 {
            query_items.clone()
        } else {
            make_set(24, 16, doc + 1)
        };
        index.insert(mh.signature(&items)).unwrap();
    }

    let mut group = c.benchmark_group("minhash_lsh_query");
    group.throughput(Throughput::Elements(5_000));
    for min_shared_bands in [1usize, 2, 4] {
        group.bench_with_input(
            BenchmarkId::new("with_similarity_min_shared_bands", min_shared_bands),
            &min_shared_bands,
            |b, &min_shared_bands| {
                b.iter(|| {
                    black_box(
                        index
                            .query_with_similarity_min_shared_bands(
                                black_box(&query_sig),
                                min_shared_bands,
                            )
                            .len(),
                    )
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// MultibitLSH fingerprinting (dense vector, 1-bit and 4-bit)
// ============================================================================

fn bench_multibit_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("multibit_fingerprint");
    for &dim in &[384usize, 768] {
        let v = make_f32_vec(dim, 2);

        // 1-bit (standard SimHash)
        let cfg1 = MultibitConfig::simhash(16, 1);
        let idx1 = MultibitLSH::new(dim, cfg1).unwrap();
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::new("1bit", dim), &dim, |b, _| {
            b.iter(|| idx1.fingerprint(black_box(&v), 0));
        });

        // 4-bit
        let cfg4 = MultibitConfig::multibit(16, 4, 1);
        let idx4 = MultibitLSH::new(dim, cfg4).unwrap();
        group.bench_with_input(BenchmarkId::new("4bit", dim), &dim, |b, _| {
            b.iter(|| idx4.fingerprint(black_box(&v), 0));
        });
    }
    group.finish();
}

// ============================================================================
// Registration
// ============================================================================

criterion_group!(
    benches,
    bench_simhash_fingerprint,
    bench_hyperplane_hasher,
    bench_minhash,
    bench_minhash_lsh_query,
    bench_multibit_fingerprint,
);
criterion_main!(benches);
