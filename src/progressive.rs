//! Progressive rendering with reuse: keep every frame within a small GPU
//! budget so the event loop stays responsive, and never throw away pixels
//! that are still (approximately) right.
//!
//! The canvas is divided into TILE x TILE tiles. For each tile we track how
//! good its current contents are:
//!   - `res`: effective sample spacing in pixels (1 = full detail, 4 = 4x4
//!     blocks or equivalent blur, infinity = nothing there yet)
//!   - `exact`: computed at full resolution for exactly the current view
//!
//! When the view changes, the old canvas is *reprojected* into the new view on
//! the GPU. A whole-pixel pan copies pixels unchanged, so they stay exact and
//! only the revealed strip is computed. A zoom resamples them: still a much
//! better preview than blocks, with `res` scaled by the zoom factor. Tiles
//! with no source (revealed by panning or zooming out) start empty.
//!
//! Each display refresh then spends the budget on the worst tiles first:
//! tiles coarser than the preview step are computed at that step (so blur
//! from a long zoom-in stays bounded), then the rest at full resolution,
//! center outward.
//!
//! The automatic iteration limit is judged on whole-view statistics only:
//! batches are accumulated into a *sweep* until every tile has been computed
//! for the current view and look. A single batch is a biased sample (a few
//! tiles along the set's boundary always look under-iterated), and acting on
//! those would ratchet the limit upward.

use crate::app::{Explorer, FrameParams, Precision};
use crate::bigfixed::Fixed;
use crate::render::{FrameStats, Renderer, Reprojection, Result, TILE, TileBatch};

/// GPU time to aim for per frame (ms). Leaves headroom inside a 60 Hz frame.
const FRAME_BUDGET_MS: f64 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    /// Effective sample spacing of the current contents, in pixels.
    res: f32,
    /// Computed at full resolution for the canvas's current view and look.
    exact: bool,
}

const MISSING: Tile = Tile { res: f32::INFINITY, exact: false };

/// Everything besides geometry that changes pixel colors.
#[derive(Clone, PartialEq)]
struct Look {
    palette: u32,
    color_density: f32,
    precision: Precision,
    max_iter: u32,
}

impl Look {
    fn of(p: &FrameParams) -> Self {
        Self { palette: p.palette, color_density: p.color_density, precision: p.precision, max_iter: p.max_iter }
    }
}

/// What the canvas currently shows, tile by tile.
struct Canvas {
    center: (Fixed, Fixed),
    scale: f64,
    width: u32,
    height: u32,
    look: Look,
    cols: u32,
    rows: u32,
    tiles: Vec<Tile>,
}

impl Canvas {
    fn new(e: &Explorer, look: Look) -> Self {
        let (cols, rows) = (e.width.div_ceil(TILE), e.height.div_ceil(TILE));
        Self {
            center: e.center.clone(),
            scale: e.scale,
            width: e.width,
            height: e.height,
            look,
            cols,
            rows,
            tiles: vec![MISSING; (cols * rows) as usize],
        }
    }

    fn origin(&self, i: usize) -> [u32; 2] {
        [(i as u32 % self.cols) * TILE, (i as u32 / self.cols) * TILE]
    }

    /// Squared distance from a tile's center to the canvas center (pixels).
    fn dist2(&self, i: usize) -> f64 {
        let [x, y] = self.origin(i);
        let dx = (x + TILE / 2) as f64 - self.width as f64 / 2.0;
        let dy = (y + TILE / 2) as f64 - self.height as f64 / 2.0;
        dx * dx + dy * dy
    }

    fn exact_fraction(&self) -> f32 {
        self.tiles.iter().filter(|t| t.exact).count() as f32 / self.tiles.len().max(1) as f32
    }
}

/// How to map the old canvas into a new view, plus the tile bookkeeping.
fn reprojected(old: &Canvas, e: &Explorer, look: Look) -> (Canvas, Reprojection) {
    let ratio = e.scale / old.scale; // new pixel size in old pixels
    let mut shift = (
        (&e.center.0 - &old.center.0).to_f64() / old.scale,
        -(&e.center.1 - &old.center.1).to_f64() / old.scale,
    );
    // A same-scale move by (almost exactly) whole pixels copies old pixels
    // unchanged, so they remain exact.
    let near_int = |v: f64| (v - v.round()).abs() < 1e-6;
    let pure_pan = (ratio - 1.0).abs() < 1e-12 && near_int(shift.0) && near_int(shift.1);
    if pure_pan {
        shift = (shift.0.round(), shift.1.round());
    }

    let mut new = Canvas::new(e, look);
    // Old-canvas position (pixels) of new-canvas position (x, y).
    let to_old = |x: f64, y: f64| {
        (
            old.width as f64 / 2.0 + (x - new.width as f64 / 2.0) * ratio + shift.0,
            old.height as f64 / 2.0 + (y - new.height as f64 / 2.0) * ratio + shift.1,
        )
    };
    let eps = 1e-6;
    let mut tiles = vec![MISSING; new.tiles.len()];
    for (i, tile) in tiles.iter_mut().enumerate() {
        let [x0, y0] = new.origin(i);
        let x1 = (x0 + TILE).min(new.width);
        let y1 = (y0 + TILE).min(new.height);
        let (u0, v0) = to_old(x0 as f64, y0 as f64);
        let (u1, v1) = to_old(x1 as f64, y1 as f64);
        if u0 < -eps || v0 < -eps || u1 > old.width as f64 + eps || v1 > old.height as f64 + eps {
            continue; // part of this tile has no source: compute it fresh
        }
        // Old tiles this one draws from (bilinear can touch one texel beyond).
        let c0 = ((u0 - 0.5).max(0.0) / TILE as f64) as u32;
        let r0 = ((v0 - 0.5).max(0.0) / TILE as f64) as u32;
        let c1 = (((u1 + 0.5) / TILE as f64).ceil() as u32).min(old.cols);
        let r1 = (((v1 + 0.5) / TILE as f64).ceil() as u32).min(old.rows);
        let (mut res, mut exact) = (0.0f32, pure_pan);
        for r in r0..r1 {
            for c in c0..c1 {
                let t = old.tiles[(r * old.cols + c) as usize];
                res = res.max(t.res);
                exact &= t.exact;
            }
        }
        // Old samples r old pixels apart are r / ratio new pixels apart.
        *tile = Tile { res: (res as f64 / ratio) as f32, exact };
    }
    new.tiles = tiles;
    let xf = Reprojection { ratio, shift, old_size: (old.width, old.height) };
    (new, xf)
}

/// Statistics accumulated until every tile has been computed once.
struct Sweep {
    stats: FrameStats,
    swept: Vec<bool>,
    remaining: usize,
}

impl Sweep {
    fn new(tiles: usize) -> Self {
        Self { stats: FrameStats::default(), swept: vec![false; tiles], remaining: tiles }
    }

    /// Add one batch; returns true once the whole view has been covered.
    fn add(&mut self, s: &FrameStats, tiles: &[usize]) -> bool {
        let t = &mut self.stats;
        t.gpu_ms += s.gpu_ms;
        t.samples += s.samples;
        t.max_escaped = t.max_escaped.max(s.max_escaped);
        t.near_limit += s.near_limit;
        t.interior += s.interior;
        for &i in tiles {
            if !std::mem::replace(&mut self.swept[i], true) {
                self.remaining -= 1;
            }
        }
        self.remaining == 0
    }
}

#[derive(Default)]
pub struct Scheduler {
    canvas: Option<Canvas>,
    /// Current sweep for the automatic iteration limit (reset on any change).
    sweep: Option<Sweep>,
    /// Smoothed GPU cost of one sample (ms); 0 until first measured.
    ms_per_sample: f64,
    /// GPU time spent since the view last changed.
    work_ms: f64,
}

impl Scheduler {
    /// The renderer's canvas was recreated (e.g. window resize): forget it.
    pub fn reset(&mut self) {
        self.canvas = None;
    }

    /// Fraction of the canvas at full detail, or None once all of it is.
    pub fn progress(&self) -> Option<f32> {
        let f = self.canvas.as_ref()?.exact_fraction();
        (f < 1.0).then_some(f)
    }

    /// GPU time spent since the view last changed.
    pub fn work_ms(&self) -> f64 {
        self.work_ms
    }

    fn record(&mut self, stats: &FrameStats) {
        self.work_ms += stats.gpu_ms;
        if stats.samples > 0 {
            let cost = stats.gpu_ms / stats.samples as f64;
            self.ms_per_sample = if self.ms_per_sample == 0.0 {
                cost
            } else {
                0.5 * (self.ms_per_sample + cost)
            };
        }
    }

    /// Samples affordable in one frame.
    fn budget_samples(&self, p: &FrameParams) -> f64 {
        if self.ms_per_sample == 0.0 {
            // Unknown cost: assume a 4x4-block preview of the view fits.
            (p.width as f64 * p.height as f64 / 16.0).max(1.0)
        } else {
            FRAME_BUDGET_MS / self.ms_per_sample
        }
    }

    /// Coarsest step a whole-view preview needs to fit the budget (power of 2).
    fn preview_step(&self, p: &FrameParams) -> u32 {
        let full = p.width as f64 * p.height as f64;
        let s = (full / self.budget_samples(p)).sqrt().ceil().max(1.0) as u32;
        s.next_power_of_two().min(TILE)
    }

    /// Bring the canvas bookkeeping up to date with the explorer's view,
    /// reprojecting the old image if that's worthwhile.
    fn sync(&mut self, e: &Explorer, p: &FrameParams, renderer: &mut dyn Renderer) -> Result<bool> {
        let look = Look::of(p);
        let Some(old) = self.canvas.take() else {
            self.sweep = None;
            self.canvas = Some(Canvas::new(e, look));
            self.work_ms = 0.0;
            return Ok(true);
        };
        if old.width != e.width || old.height != e.height {
            self.sweep = None;
            self.canvas = Some(Canvas::new(e, look));
            self.work_ms = 0.0;
            return Ok(true);
        }
        let moved = old.scale != e.scale || old.center != e.center;
        let mut canvas = if moved {
            let (new, xf) = reprojected(&old, e, look.clone());
            if new.tiles.iter().any(|t| t.res.is_finite()) {
                renderer.reproject(&xf)?;
            }
            self.work_ms = 0.0;
            self.sweep = None;
            new
        } else {
            old
        };
        if canvas.look != look {
            // Colors or iteration counts changed: everything needs redoing.
            // Keep the old pixels on screen until the coarse pass replaces them.
            canvas.tiles.fill(MISSING);
            canvas.look = look;
            self.sweep = None;
            self.work_ms = 0.0;
        }
        self.canvas = Some(canvas);
        Ok(moved)
    }

    /// Choose this frame's work: worst tiles first, within budget.
    fn pick(&self, p: &FrameParams) -> Option<(Vec<usize>, u32)> {
        let c = self.canvas.as_ref()?;
        let budget = self.budget_samples(p);
        let preview = self.preview_step(p);
        let by_priority = |mut v: Vec<usize>| {
            // Worst first; among equals, nearest the center first.
            v.sort_by(|&a, &b| {
                c.tiles[b].res.total_cmp(&c.tiles[a].res).then(c.dist2(a).total_cmp(&c.dist2(b)))
            });
            v
        };

        // 1. Anything coarser than the preview step.
        let coarse: Vec<usize> = (0..c.tiles.len()).filter(|&i| c.tiles[i].res > preview as f32).collect();
        if !coarse.is_empty() {
            // Use the finest step at which all of them fit; else the preview
            // step, worst first.
            let per_tile = |s: u32| ((TILE / s) * (TILE / s)) as f64;
            let fits = |s: u32| coarse.len() as f64 * per_tile(s) <= budget;
            let step = [1, 2, 4, 8, 16].into_iter().find(|&s| s <= preview && fits(s)).unwrap_or(preview);
            let n = ((budget / per_tile(step)) as usize).max(1);
            let mut tiles = by_priority(coarse);
            tiles.truncate(n);
            return Some((tiles, step));
        }

        // 2. Full-resolution refinement.
        let stale: Vec<usize> = (0..c.tiles.len()).filter(|&i| !c.tiles[i].exact).collect();
        if stale.is_empty() {
            return None;
        }
        let n = ((budget / (TILE * TILE) as f64) as usize).max(1);
        let mut tiles = by_priority(stale);
        tiles.truncate(n);
        Some((tiles, 1))
    }

    /// Do one frame's worth of work. Returns true if more work remains (the
    /// caller should request another redraw).
    pub fn step(&mut self, explorer: &mut Explorer, renderer: &mut dyn Renderer) -> Result<bool> {
        let params = explorer.frame_params();
        let moved = self.sync(explorer, &params, renderer)?;

        let Some((tiles, step)) = self.pick(&params) else {
            if moved {
                // Reprojection alone: still needs presenting.
                renderer.draw(&params, &TileBatch { tiles: vec![], step: 1 })?;
            }
            return Ok(false);
        };
        let canvas = self.canvas.as_ref().unwrap();
        let batch = TileBatch { tiles: tiles.iter().map(|&i| canvas.origin(i)).collect(), step };
        let total = canvas.tiles.len();

        let stats = renderer.draw(&params, &batch)?;
        self.record(&stats);
        let canvas = self.canvas.as_mut().unwrap();
        for &i in &tiles {
            canvas.tiles[i] = Tile { res: step as f32, exact: step == 1 };
        }
        // Judge the iteration limit once the whole view has been covered. A
        // changed limit shows up as a new Look on the next step.
        let sweep = self.sweep.get_or_insert_with(|| Sweep::new(total));
        if sweep.add(&stats, &tiles) {
            explorer.observe(&sweep.stats, true);
            self.sweep = None;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records batches instead of drawing; reports a fixed cost per sample.
    #[derive(Default)]
    struct Mock {
        batches: Vec<TileBatch>,
        reprojections: Vec<Reprojection>,
    }

    impl Renderer for Mock {
        fn resize(&mut self, _: u32, _: u32, _: f64) {}
        fn draw(&mut self, p: &FrameParams, batch: &TileBatch) -> Result<FrameStats> {
            self.batches.push(batch.clone());
            let per = (TILE / batch.step).pow(2) as u64;
            let samples = batch.tiles.len() as u64 * per;
            // A settled iteration limit: nothing near it, nothing far below.
            let max_escaped = p.max_iter / 2 - 1;
            Ok(FrameStats { gpu_ms: samples as f64 * 1e-4, samples, max_escaped, near_limit: 0, interior: 0 })
        }
        fn reproject(&mut self, xf: &Reprojection) -> Result<()> {
            self.reprojections.push(*xf);
            Ok(())
        }
        fn capture(&mut self, _: &FrameParams, _: u32) -> Result<(Vec<u8>, FrameStats)> {
            unimplemented!()
        }
    }

    fn run_to_completion(s: &mut Scheduler, e: &mut Explorer, m: &mut Mock) -> usize {
        let mut frames = 0;
        while s.step(e, m).unwrap() {
            frames += 1;
            assert!(frames < 10_000, "never finished");
        }
        frames
    }

    /// Full-resolution tiles computed since `from`.
    fn exact_tiles(m: &Mock, from: usize) -> usize {
        m.batches[from..].iter().filter(|b| b.step == 1).map(|b| b.tiles.len()).sum()
    }

    #[test]
    fn fresh_view_previews_then_refines_every_tile_once() {
        let mut e = Explorer::new(400, 300);
        let (mut s, mut m) = (Scheduler::default(), Mock::default());
        run_to_completion(&mut s, &mut e, &mut m);
        assert!(m.batches[0].step > 1, "first frame should be a coarse preview");
        let mut seen = std::collections::HashMap::new();
        for b in m.batches.iter().filter(|b| b.step == 1) {
            for t in &b.tiles {
                *seen.entry(*t).or_insert(0) += 1;
            }
        }
        let total = (400u32.div_ceil(TILE) * 300u32.div_ceil(TILE)) as usize;
        assert_eq!(seen.len(), total);
        assert!(seen.values().all(|&n| n == 1));
        assert_eq!(s.progress(), None);
    }

    #[test]
    fn whole_pixel_pan_only_computes_revealed_tiles() {
        let mut e = Explorer::new(400, 304); // 25 x 19 tiles
        let (mut s, mut m) = (Scheduler::default(), Mock::default());
        run_to_completion(&mut s, &mut e, &mut m);

        let before = m.batches.len();
        e.pan_pixels(-40.0, 0.0); // content moves left: 40 new columns on the right
        run_to_completion(&mut s, &mut e, &mut m);
        let xf = m.reprojections.last().unwrap();
        assert_eq!((xf.ratio, xf.shift), (1.0, (40.0, 0.0)));
        // Exact tiles survive where their whole source was exact; the 40 new
        // columns touch 3 tile columns, plus one more misaligned column.
        let redone = exact_tiles(&m, before);
        assert!(redone <= 4 * 19, "recomputed {redone} tiles for a 40 px pan");
        assert!(redone >= 3 * 19);
    }

    /// Every batch looks under-iterated (as tiles along the boundary do).
    struct Pessimist;

    impl Renderer for Pessimist {
        fn resize(&mut self, _: u32, _: u32, _: f64) {}
        fn draw(&mut self, p: &FrameParams, batch: &TileBatch) -> Result<FrameStats> {
            let samples = batch.tiles.len() as u64 * (TILE / batch.step).pow(2) as u64;
            // Only full-resolution batches near the boundary look cut off.
            let near_limit = if batch.step == 1 { (samples / 50) as u32 } else { 0 };
            Ok(FrameStats { gpu_ms: samples as f64 * 1e-3, samples, max_escaped: p.max_iter - 1, near_limit, interior: 0 })
        }
        fn reproject(&mut self, _: &Reprojection) -> Result<()> {
            Ok(())
        }
        fn capture(&mut self, _: &FrameParams, _: u32) -> Result<(Vec<u8>, FrameStats)> {
            unimplemented!()
        }
    }

    #[test]
    fn iteration_limit_is_judged_per_sweep_not_per_batch() {
        let mut e = Explorer::new(400, 304);
        let mut s = Scheduler::default();
        let start = e.iter_limit;
        let mut batches = 0;
        while e.iter_limit == start {
            assert!(s.step(&mut e, &mut Pessimist).unwrap());
            batches += 1;
            assert!(batches < 10_000);
        }
        // The pessimist's batches all looked cut off, but the limit only grew
        // once the whole view had been refined.
        assert!(batches > 5, "expected several batches, got {batches}");
        assert_eq!(s.progress(), None, "limit changed mid-sweep");
        assert_eq!(e.iter_limit, start * 2);
    }

    #[test]
    fn zoom_reuses_old_image_as_preview() {
        let mut e = Explorer::new(400, 304);
        let (mut s, mut m) = (Scheduler::default(), Mock::default());
        run_to_completion(&mut s, &mut e, &mut m);

        // Zoom in 1.5x: everything has a (blurry) source, nothing is missing.
        let look = s.canvas.as_ref().unwrap().look.clone();
        e.zoom_at_center(1.5);
        let (c, xf) = reprojected(s.canvas.as_ref().unwrap(), &e, look.clone());
        assert!((xf.ratio - 1.0 / 1.5).abs() < 1e-12);
        assert!(c.tiles.iter().all(|t| !t.exact && (t.res - 1.5).abs() < 1e-3));
        // So no coarse pass is needed: straight to full-resolution refinement.
        let before = m.batches.len();
        run_to_completion(&mut s, &mut e, &mut m);
        assert!(m.batches[before..].iter().all(|b| b.step == 1));

        // Zoom out 2x: the middle quarter has sharp sources, the border none.
        e.zoom_at_center(0.5);
        let (c, _) = reprojected(s.canvas.as_ref().unwrap(), &e, look);
        assert!(c.tiles.iter().filter(|t| t.res.is_finite()).all(|t| t.res <= 0.5 + 1e-6));
        let missing = c.tiles.iter().filter(|t| t.res.is_infinite()).count() as f64;
        let frac = missing / c.tiles.len() as f64;
        assert!(frac > 0.6 && frac < 0.85, "missing fraction {frac}");
    }
}
