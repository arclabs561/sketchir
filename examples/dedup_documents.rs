/// Find near-duplicate documents in a collection.
///
/// A common preprocessing step before indexing: detect and remove
/// near-duplicates that would inflate results and waste storage.
/// MinHash + LSH gives approximate Jaccard similarity in O(n) time
/// instead of O(n^2) pairwise comparison.
///
/// ```sh
/// cargo run --example dedup_documents
/// ```
use sketchir::{BlockingConfig, MinHashTextLSH};

fn main() {
    let documents = vec![
        (
            "original",
            "The quick brown fox jumps over the lazy dog near the riverbank",
        ),
        (
            "minor_edit",
            "A quick brown fox jumped over the lazy dog near the riverbank",
        ),
        (
            "reordered",
            "Near the riverbank the lazy dog was jumped over by the quick brown fox",
        ),
        (
            "different_1",
            "Machine learning models require large datasets for training",
        ),
        (
            "different_2",
            "The weather forecast predicts rain for the entire week ahead",
        ),
        (
            "different_3",
            "Rust provides memory safety without garbage collection overhead",
        ),
    ];

    let cfg = BlockingConfig {
        ngram_size: 3,
        num_bands: 20,
        num_hashes_per_band: 5,
        ..BlockingConfig::default()
    };
    let mut index = MinHashTextLSH::new(cfg).unwrap();

    for (id, text) in &documents {
        index.insert_text(*id, *text);
    }
    println!("Indexed {} documents\n", documents.len());

    // candidate_pairs returns (i, j) index pairs into insertion order.
    let pairs = index.candidate_pairs();
    println!("Near-duplicate pairs:");
    if pairs.is_empty() {
        println!("  (none found)");
    }
    for (i, j) in &pairs {
        println!("  {:?} <-> {:?}", documents[*i].0, documents[*j].0);
    }

    // Dedup: keep the first document from each duplicate pair.
    let mut removed = std::collections::HashSet::new();
    for (_, j) in &pairs {
        removed.insert(*j);
    }
    let unique: Vec<_> = documents
        .iter()
        .enumerate()
        .filter(|(i, _)| !removed.contains(i))
        .collect();
    println!("\nUnique documents after dedup: {}", unique.len());
}
