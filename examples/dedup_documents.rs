use sketchir::{BlockingConfig, MinHashTextLSH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = [
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

    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.is_empty() {
        for (id, text) in samples {
            index.insert_text(id, text);
        }
    } else {
        for path in paths {
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("opening {path:?}: {e}"))?;
            index.insert_text(path.to_string_lossy(), text);
        }
    }
    println!("Indexed {} documents", index.len());
    println!("Candidate pairs (estimated Jaccard):");
    let pairs = index.candidate_pairs();
    if pairs.is_empty() {
        println!("  (none found)");
    }
    for (i, j) in pairs {
        let left = index.get_id(i).unwrap();
        let right = index.get_id(j).unwrap();
        let similarity = index.estimated_similarity(i, j).unwrap();
        println!("  {left:?} <-> {right:?}  {similarity:.3}");
    }
    Ok(())
}
