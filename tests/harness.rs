//! Behavior must not change unless intended: every scenario played
//! deterministically (see `mandelbrot::harness`) must match the fingerprint
//! recorded for the platform's backend, `tests/data/compat-<backend>.txt`
//! (see `render::BACKEND`), pixel for pixel and batch for batch. GPUs round
//! transcendentals differently, and one escape count off by one changes
//! everything after it, so fingerprints only match on the GPU (and driver)
//! they were recorded on: `compat-metal.txt` on Apple silicon,
//! `compat-wgpu.txt` on llvmpipe.
//!
//! Slow (it plays every scenario on the GPU), so ignored by default:
//! `cargo test --release --test harness -- --ignored`
//!
//! After an intended behavior change, re-record (for each backend):
//! `cargo run --release --example harness -- --compat record tests/data/compat-<backend>.txt`

use mandelbrot::harness::{self, Config};
use mandelbrot::render;

#[test]
#[ignore]
fn every_scenario_matches_the_recorded_fingerprint() {
    let path = format!("{}/tests/data/compat-{}.txt", env!("CARGO_MANIFEST_DIR"), render::BACKEND);
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut r = render::create_headless().expect("GPU device");
    let cfg = Config::compat();
    let mut actual = String::new();
    for s in harness::scenarios() {
        actual += &harness::run(&s, &cfg, r.as_mut()).unwrap().fingerprint();
    }
    if let Err(e) = harness::compare_fingerprints(&expected, &actual) {
        panic!("behavior changed:\n{e}");
    }
}
