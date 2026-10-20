//! Behavior must not change unless intended: every scenario played
//! deterministically (see `mandelbrot::harness`) must match the fingerprint
//! recorded in `tests/data/compat.txt`, pixel for pixel and batch for batch.
//!
//! Slow (it plays every scenario on the GPU), so ignored by default:
//! `cargo test --release --test harness -- --ignored`
//!
//! After an intended behavior change, re-record:
//! `cargo run --release --example harness -- --compat record tests/data/compat.txt`

use mandelbrot::harness::{self, Config};
use mandelbrot::render;

#[test]
#[ignore]
fn every_scenario_matches_the_recorded_fingerprint() {
    let expected = include_str!("data/compat.txt");
    let mut r = render::create_headless().expect("Metal device");
    let cfg = Config::compat();
    let mut actual = String::new();
    for s in harness::scenarios() {
        actual += &harness::run(&s, &cfg, r.as_mut()).unwrap().fingerprint();
    }
    if let Err(e) = harness::compare_fingerprints(expected, &actual) {
        panic!("behavior changed:\n{e}");
    }
}
