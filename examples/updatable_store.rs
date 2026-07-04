//! Durable, updatable MinHash near-duplicate search.
//!
//! Run:
//! `cargo run --features store --example updatable_store`

use durability::MemoryDirectory;
use sketchir::{store::UpdatableIndex, BlockingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = MemoryDirectory::arc();
    let config = BlockingConfig::default();

    let mut index = UpdatableIndex::open(dir.clone(), 2, config.clone())?;
    index.add(1, "New York City travel guide")?;
    index.add(2, "New York City travel guide")?;
    index.add(3, "San Francisco restaurant notes")?;
    index.add(4, "Rust search index implementation")?;
    index.checkpoint()?;

    let query = "New York City guide";
    let before_delete = index.near_duplicates_with_similarity(query);
    assert!(before_delete.iter().any(|(id, _)| *id == 1));
    assert!(before_delete.iter().any(|(id, _)| *id == 2));

    index.delete(2)?;
    index.checkpoint()?;
    drop(index);

    let recovered = UpdatableIndex::open(dir, 2, config)?;
    let after_reopen = recovered.near_duplicates_with_similarity(query);
    assert!(after_reopen.iter().any(|(id, _)| *id == 1));
    assert!(!after_reopen.iter().any(|(id, _)| *id == 2));

    println!("before delete:");
    for (id, sim) in before_delete {
        println!("  doc {id}: {sim:.3}");
    }
    println!("after reopen:");
    for (id, sim) in after_reopen {
        println!("  doc {id}: {sim:.3}");
    }

    Ok(())
}
