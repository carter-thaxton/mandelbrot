//! Shared test helpers: a mock renderer that records what it's asked to do.
//!
//! (Each file in `tests/` is its own crate and uses a different subset of
//! these, hence the `dead_code` allowance.)
#![allow(dead_code)]

use mandelbrot::app::{Colors, Explorer, FrameParams};
use mandelbrot::progressive::Scheduler;
use mandelbrot::render::{BatchStats, FrameStats, Output, Renderer, Reprojection, Result, TILE, TileBatch, TileStats};

/// Every renderer call, in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Call {
    DrawPixels,
    DrawStats,
    Reproject,
    Present,
}

/// Records calls; reports statistics from `stats` (given the frame and the
/// batch's sample count), at a fixed GPU cost per sample. Per-tile stats
/// only count samples.
pub struct MockRenderer {
    pub batches: Vec<TileBatch>,
    pub reprojections: Vec<Reprojection>,
    pub log: Vec<Call>,
    pub ms_per_sample: f64,
    pub stats: Box<dyn Fn(&FrameParams, &TileBatch, u64) -> FrameStats>,
}

impl MockRenderer {
    pub fn new(ms_per_sample: f64, stats: impl Fn(&FrameParams, &TileBatch, u64) -> FrameStats + 'static) -> Self {
        Self { batches: vec![], reprojections: vec![], log: vec![], ms_per_sample, stats: Box::new(stats) }
    }

    /// A view whose iteration limit is already right: nothing escapes near
    /// the limit, and nothing suggests it could shrink.
    pub fn settled() -> Self {
        Self::new(1e-4, |p, _, samples| stats(samples, p.max_iter / 2 - 1, 0, 0))
    }

    /// Full-resolution tiles computed from batch `from` onward.
    pub fn exact_tiles_since(&self, from: usize) -> usize {
        self.batches[from..].iter().filter(|b| b.step == 1 && b.output == Output::Pixels).map(|b| b.tiles.len()).sum()
    }
}

pub fn stats(samples: u64, max_escaped: u32, near_limit: u32, interior: u32) -> FrameStats {
    FrameStats { gpu_ms: 0.0, samples, max_escaped, near_limit, interior }
}

impl Renderer for MockRenderer {
    fn resize(&mut self, _: u32, _: u32, _: f64) {}

    fn draw(&mut self, p: &FrameParams, batch: &TileBatch) -> Result<BatchStats> {
        self.batches.push(batch.clone());
        self.log.push(match batch.output {
            Output::Pixels => Call::DrawPixels,
            Output::StatsOnly => Call::DrawStats,
        });
        let per_side = TILE / batch.step;
        let samples = batch.tiles.len() as u64 * per_side.pow(2) as u64;
        let s = (self.stats)(p, batch, samples);
        let tile = TileStats { samples: per_side.pow(2), pairs: TileStats::pairs_in(per_side, per_side), ..Default::default() };
        Ok(BatchStats {
            total: FrameStats { gpu_ms: samples as f64 * self.ms_per_sample, samples, ..s },
            tiles: vec![tile; batch.tiles.len()],
        })
    }

    fn reproject(&mut self, xf: &Reprojection) -> Result<()> {
        self.reprojections.push(*xf);
        self.log.push(Call::Reproject);
        Ok(())
    }

    fn present(&mut self, _: &Colors) -> Result<()> {
        self.log.push(Call::Present);
        Ok(())
    }

    fn snapshot(&mut self, _: &Colors) -> Result<Vec<u8>> {
        unimplemented!("no pixels in a mock")
    }

    fn capture(&mut self, _: &FrameParams, _: u32, _: &Colors) -> Result<(Vec<u8>, FrameStats)> {
        unimplemented!("no pixels in a mock")
    }
}

pub fn run_to_completion(s: &mut Scheduler, e: &mut Explorer, r: &mut MockRenderer) {
    for _ in 0..10_000 {
        if !s.step(e, r).unwrap() {
            return;
        }
    }
    panic!("scheduler never finished");
}
