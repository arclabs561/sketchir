//! Measure sidecar-first snapshot reopen vs rebuild for the segstore-backed store.
//!
//! Run:
//! `cargo run --release --features store --example store_reopen_diagnostics`

use std::sync::Arc;
use std::time::{Duration, Instant};

use durability::{Directory, FsDirectory};
use sketchir::{
    store::{SnapshotIndex, UpdatableIndex},
    BlockingConfig,
};

const N: usize = 1_000;
const FLUSH: usize = 200;
const MIN_SHARED_BANDS: usize = 2;
const QUERY_ID: u32 = 377;

type DynError = Box<dyn std::error::Error>;
type StoreDir = Arc<dyn Directory>;
type Candidates = Vec<u32>;
type BuiltStore = (tempfile::TempDir, StoreDir, String);

fn main() -> Result<(), DynError> {
    let config = BlockingConfig::default();

    let (_load_root, load_dir, query) = build_checkpointed_dir(config.clone())?;
    let load_sidecars = sidecar_count(&load_dir)?;
    let (load_elapsed, load_hits) = first_snapshot_query(load_dir.clone(), config.clone(), &query)?;

    let (_rebuild_root, rebuild_dir, rebuild_query) = build_checkpointed_dir(config.clone())?;
    let sidecars_before_delete = sidecar_count(&rebuild_dir)?;
    delete_sidecars(&rebuild_dir)?;
    let sidecars_after_delete = sidecar_count(&rebuild_dir)?;
    let (rebuild_elapsed, rebuild_hits) =
        first_snapshot_query(rebuild_dir, config, &rebuild_query)?;

    assert_eq!(load_hits, rebuild_hits);
    assert!(load_hits.contains(&QUERY_ID));

    println!("documents: {N}, flush threshold: {FLUSH}");
    println!("sidecars loaded path: {load_sidecars}");
    println!(
        "sidecars rebuild path before/after delete: {sidecars_before_delete}/{sidecars_after_delete}"
    );
    println!(
        "first snapshot query with sidecars: {}",
        micros(load_elapsed)
    );
    println!(
        "first snapshot query after deleting sidecars: {}",
        micros(rebuild_elapsed)
    );
    println!("matching candidates: {}", load_hits.len());
    println!("query doc present: {}", load_hits.contains(&QUERY_ID));

    Ok(())
}

fn build_checkpointed_dir(config: BlockingConfig) -> Result<BuiltStore, DynError> {
    let root = tempfile::tempdir()?;
    let dir: StoreDir = FsDirectory::arc(root.path())?;
    let mut index = UpdatableIndex::open(dir.clone(), FLUSH, config)?;
    index.extend((0..N).map(|id| (id as u32, text(id as u32))))?;
    index.checkpoint()?;
    Ok((root, dir, text(QUERY_ID)))
}

fn first_snapshot_query(
    dir: StoreDir,
    config: BlockingConfig,
    query: &str,
) -> Result<(Duration, Candidates), DynError> {
    let snapshot = SnapshotIndex::open(dir, config)?;
    let start = Instant::now();
    let hits = snapshot.near_duplicates_min_shared_bands(query, MIN_SHARED_BANDS)?;
    Ok((start.elapsed(), hits))
}

fn delete_sidecars(dir: &StoreDir) -> Result<(), DynError> {
    for name in dir.list_dir("")? {
        if name.starts_with("segstore.idx.") {
            dir.delete(&name)?;
        }
    }
    Ok(())
}

fn sidecar_count(dir: &StoreDir) -> Result<usize, DynError> {
    Ok(dir
        .list_dir("")?
        .into_iter()
        .filter(|name| name.starts_with("segstore.idx."))
        .count())
}

fn text(id: u32) -> String {
    let mut state = 0x1234_5678_9abc_def0u64 ^ id as u64;
    format!(
        "near duplicate record {id:04} {} {} {} {}",
        token(&mut state),
        token(&mut state),
        token(&mut state),
        token(&mut state)
    )
}

fn token(state: &mut u64) -> String {
    (0..12)
        .map(|_| (b'a' + (xorshift(state) % 26) as u8) as char)
        .collect()
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn micros(duration: Duration) -> String {
    format!("{} us", duration.as_micros())
}
