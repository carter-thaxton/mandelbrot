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
//! The automatic iteration limit is judged on whole-view statistics only. A
//! single batch is a biased sample (a few tiles along the set's boundary
//! always look under-iterated), and acting on those would ratchet the limit
//! upward. Two sources:
//!   - While the view is changing, a *probe* every PROBE_INTERVAL: one frame
//!     spent on a coarse whole-view pass that gathers statistics but writes
//!     no pixels. This keeps the limit current during continuous zooming and
//!     panning, when tiles are mostly reused and no sweep would complete.
//!   - A *sweep*: batches accumulated until every tile has been computed for
//!     the current view and look (full-resolution evidence once settled).

use crate::app::{Colors, Explorer, FrameParams, MIN_DECISION_SAMPLES, Precision};
use crate::bigfixed::Fixed;
use crate::render::{FrameStats, Output, Renderer, Reprojection, Result, TILE, TileBatch};

/// GPU time to aim for per frame (ms). Leaves headroom inside a 60 Hz frame.
const FRAME_BUDGET_MS: f64 = 12.0;
/// Never compute fewer samples per frame than this. A dispatch lasts at least
/// as long as its slowest pixel, so below the size that fills the GPU a
/// smaller batch saves no time, it only idles the GPU. On deep views that
/// floor (one slow pixel, ~16 ms at 32k iterations) exceeds the budget, and
/// without this minimum the batches shrank to ~8 tiles: 30x slower overall.
const MIN_SAMPLES_PER_FRAME: f64 = 65536.0;
/// Minimum time between statistics probes while the view keeps changing.
const PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(120);
/// Probes sample at least this coarsely (they only need statistics).
const PROBE_MIN_STEP: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    /// Effective sample spacing of the current contents, in pixels.
    res: f32,
    /// Computed at full resolution for the canvas's current view and look.
    exact: bool,
}

const MISSING: Tile = Tile { res: f32::INFINITY, exact: false };

/// Everything besides geometry that changes the computed values (mu).
/// Colors aren't part of it: they're applied when presenting.
#[derive(Clone, PartialEq)]
struct Look {
    precision: Precision,
    max_iter: u32,
}

impl Look {
    fn of(p: &FrameParams) -> Self {
        Self { precision: p.precision, max_iter: p.max_iter }
    }
}

/// The geometry of a view: which point of the plane each pixel shows.
#[derive(Clone, PartialEq)]
struct View {
    center: (Fixed, Fixed),
    /// Complex-plane units per pixel.
    scale: f64,
    width: u32,
    height: u32,
}

impl View {
    fn of(e: &Explorer) -> Self {
        Self { center: e.center.clone(), scale: e.scale, width: e.width, height: e.height }
    }
}

/// What the canvas currently shows, tile by tile.
struct Canvas {
    view: View,
    look: Look,
    cols: u32,
    rows: u32,
    tiles: Vec<Tile>,
}

impl Canvas {
    fn new(view: View, look: Look) -> Self {
        let (cols, rows) = (view.width.div_ceil(TILE), view.height.div_ceil(TILE));
        Self {
            view,
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
        let dx = (x + TILE / 2) as f64 - self.view.width as f64 / 2.0;
        let dy = (y + TILE / 2) as f64 - self.view.height as f64 / 2.0;
        dx * dx + dy * dy
    }

    fn exact_fraction(&self) -> f32 {
        self.tiles.iter().filter(|t| t.exact).count() as f32 / self.tiles.len().max(1) as f32
    }
}

/// How to map the old canvas into a new view, plus the tile bookkeeping.
fn reprojected(old: &Canvas, view: View, look: Look) -> (Canvas, Reprojection) {
    let o = &old.view;
    let ratio = view.scale / o.scale; // new pixel size in old pixels
    let mut shift = (
        (&view.center.0 - &o.center.0).to_f64() / o.scale,
        -(&view.center.1 - &o.center.1).to_f64() / o.scale,
    );
    // A same-scale move by (almost exactly) whole pixels copies old pixels
    // unchanged, so they remain exact.
    let near_int = |v: f64| (v - v.round()).abs() < 1e-6;
    let pure_pan = (ratio - 1.0).abs() < 1e-12 && near_int(shift.0) && near_int(shift.1);
    if pure_pan {
        shift = (shift.0.round(), shift.1.round());
    }

    let mut new = Canvas::new(view, look);
    let n = &new.view;
    // Old-canvas position (pixels) of new-canvas position (x, y).
    let to_old = |x: f64, y: f64| {
        (
            o.width as f64 / 2.0 + (x - n.width as f64 / 2.0) * ratio + shift.0,
            o.height as f64 / 2.0 + (y - n.height as f64 / 2.0) * ratio + shift.1,
        )
    };
    let eps = 1e-6;
    let mut tiles = vec![MISSING; new.tiles.len()];
    for (i, tile) in tiles.iter_mut().enumerate() {
        let [x0, y0] = new.origin(i);
        let x1 = (x0 + TILE).min(n.width);
        let y1 = (y0 + TILE).min(n.height);
        let (u0, v0) = to_old(x0 as f64, y0 as f64);
        let (u1, v1) = to_old(x1 as f64, y1 as f64);
        if u0 < -eps || v0 < -eps || u1 > o.width as f64 + eps || v1 > o.height as f64 + eps {
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
    let xf = Reprojection { ratio, shift, old_size: (o.width, o.height) };
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
        self.stats.merge(s);
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
    /// Samples to compute per frame to fill FRAME_BUDGET_MS; 0 until first
    /// measured (see [`Scheduler::record`]).
    samples_per_frame: f64,
    /// GPU time spent since the view last changed.
    work_ms: f64,
    /// The view or look changed since the last statistics probe.
    probe_due: bool,
    last_probe: Option<std::time::Instant>,
    /// Probe statistics accumulated until there are enough to decide on
    /// (coarse probes of expensive views have only a few thousand samples).
    probe_stats: FrameStats,
    /// Override for PROBE_INTERVAL.
    probe_interval: Option<std::time::Duration>,
    /// Colors of the last present (a color change alone needs a re-present).
    presented: Option<Colors>,
}

impl Scheduler {
    /// A scheduler that probes statistics at most every `interval` while the
    /// view changes (default PROBE_INTERVAL). Zero probes on every change.
    pub fn with_probe_interval(interval: std::time::Duration) -> Self {
        Self { probe_interval: Some(interval), ..Default::default() }
    }

    /// The renderer's canvas was recreated (e.g. window resize): forget it.
    pub fn reset(&mut self) {
        self.canvas = None;
        self.presented = None;
    }

    fn present(&mut self, renderer: &mut dyn Renderer, colors: Colors) -> Result<()> {
        renderer.present(&colors)?;
        self.presented = Some(colors);
        Ok(())
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

    /// Size the next frame's work from how long this draw took: drop at once
    /// when over budget, grow at most 2x per frame when under (see also
    /// MIN_SAMPLES_PER_FRAME).
    fn record(&mut self, stats: &FrameStats) {
        self.work_ms += stats.gpu_ms;
        if stats.samples == 0 {
            return;
        }
        let target = stats.samples as f64 * FRAME_BUDGET_MS / stats.gpu_ms.max(1e-3);
        let current = self.samples_per_frame;
        self.samples_per_frame = if current == 0.0 { target } else { target.min(current * 2.0) };
    }

    /// Samples affordable in one frame.
    fn budget_samples(&self, p: &FrameParams) -> f64 {
        if self.samples_per_frame == 0.0 {
            // Unknown cost: assume a 4x4-block preview of the view fits.
            (p.width as f64 * p.height as f64 / 16.0).max(MIN_SAMPLES_PER_FRAME)
        } else {
            self.samples_per_frame.max(MIN_SAMPLES_PER_FRAME)
        }
    }

    /// Coarsest step a whole-view preview needs to fit the budget (power of 2).
    fn preview_step(&self, p: &FrameParams) -> u32 {
        let full = p.width as f64 * p.height as f64;
        let s = (full / self.budget_samples(p)).sqrt().ceil().max(1.0) as u32;
        s.next_power_of_two().min(TILE)
    }

    /// Bring the canvas bookkeeping up to date with the current view,
    /// reprojecting the old image if that's worthwhile.
    fn sync(&mut self, view: View, p: &FrameParams, renderer: &mut dyn Renderer) -> Result<bool> {
        let look = Look::of(p);
        let Some(old) = self.canvas.take() else {
            self.sweep = None;
            self.canvas = Some(Canvas::new(view, look));
            self.work_ms = 0.0;
            return Ok(true);
        };
        if old.view.width != view.width || old.view.height != view.height {
            self.sweep = None;
            self.canvas = Some(Canvas::new(view, look));
            self.work_ms = 0.0;
            return Ok(true);
        }
        let moved = old.view != view;
        let mut canvas = if moved {
            let (new, xf) = reprojected(&old, view, look.clone());
            if new.tiles.iter().any(|t| t.res.is_finite()) {
                renderer.reproject(&xf)?;
            }
            self.work_ms = 0.0;
            self.sweep = None;
            self.probe_due = true;
            new
        } else {
            old
        };
        if canvas.look != look {
            let only_iters = Look { max_iter: look.max_iter, ..canvas.look.clone() } == look;
            let ratio = look.max_iter as f64 / canvas.look.max_iter as f64;
            if only_iters && ratio > 0.5 && ratio < 2.0 {
                // A modest iteration change barely alters the image: keep it
                // as the preview and just refine.
                canvas.tiles.iter_mut().for_each(|t| t.exact = false);
            } else {
                // Colors changed, or iterations by 2x or more: redo everything.
                // Old pixels stay on screen until the coarse pass replaces them.
                canvas.tiles.fill(MISSING);
            }
            canvas.look = look;
            self.probe_due = true;
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
        self.step_at(std::time::Instant::now(), explorer, renderer)
    }

    /// [`Scheduler::step`] at time `now` (probes are spaced by this clock),
    /// for replaying input on a simulated clock.
    pub fn step_at(&mut self, now: std::time::Instant, explorer: &mut Explorer, renderer: &mut dyn Renderer) -> Result<bool> {
        let params = explorer.frame_params();
        let colors = explorer.colors();
        let moved = self.sync(View::of(explorer), &params, renderer)?;
        // What's on screen is stale if the canvas moved or the colors changed.
        let stale = moved || self.presented != Some(colors);

        if self.probe_due
            && explorer.auto_iter
            && self.last_probe.is_none_or(|t| now.duration_since(t) >= self.probe_interval.unwrap_or(PROBE_INTERVAL))
        {
            // Spend this frame on whole-view statistics, which don't change
            // the canvas.
            if stale {
                self.present(renderer, colors)?;
            }
            let canvas = self.canvas.as_ref().unwrap();
            let step = self.preview_step(&params).max(PROBE_MIN_STEP);
            let probe = TileBatch::all(canvas.view.width, canvas.view.height, step, Output::StatsOnly);
            let stats = renderer.draw(&params, &probe)?;
            self.record(&stats);
            self.probe_stats.merge(&stats);
            if self.probe_stats.samples >= MIN_DECISION_SAMPLES {
                let before = explorer.iter_limit;
                explorer.observe(&self.probe_stats, true);
                if std::env::var_os("MANDEL_DEBUG_ITERS").is_some() {
                    eprintln!(
                        "probe at zoom {:.2e}: limit {before} -> {}  {:?}",
                        explorer.zoom(),
                        explorer.iter_limit,
                        self.probe_stats
                    );
                }
                self.probe_stats = FrameStats::default();
            }
            self.probe_due = false;
            self.last_probe = Some(now);
            return Ok(true);
        }

        let Some((tiles, step)) = self.pick(&params) else {
            if stale {
                // A reprojection or color change with nothing left to compute.
                self.present(renderer, colors)?;
            }
            return Ok(false);
        };
        let canvas = self.canvas.as_ref().unwrap();
        let batch = TileBatch::new(tiles.iter().map(|&i| canvas.origin(i)).collect(), step, Output::Pixels);
        let total = canvas.tiles.len();

        let stats = renderer.draw(&params, &batch)?;
        self.present(renderer, colors)?;
        self.record(&stats);
        let canvas = self.canvas.as_mut().unwrap();
        for &i in &tiles {
            canvas.tiles[i] = Tile { res: step as f32, exact: step == 1 };
        }
        // Judge the iteration limit once the whole view has been covered. A
        // changed limit shows up as a new Look on the next step.
        let sweep = self.sweep.get_or_insert_with(|| Sweep::new(total));
        if sweep.add(&stats, &tiles) {
            // Too few samples (a very coarse pass) means no decision; start a
            // new sweep rather than topping this one up with partial batches,
            // which would bias it.
            if sweep.stats.samples >= MIN_DECISION_SAMPLES {
                let before = explorer.iter_limit;
                explorer.observe(&sweep.stats, true);
                if std::env::var_os("MANDEL_DEBUG_ITERS").is_some() {
                    eprintln!(
                        "sweep at zoom {:.2e}: limit {before} -> {}  {:?}",
                        explorer.zoom(),
                        explorer.iter_limit,
                        sweep.stats
                    );
                }
            }
            self.sweep = None;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(width: u32, height: u32, cx: f64, scale: f64) -> View {
        View { center: (Fixed::from_f64(cx, 2), Fixed::zero(2)), scale, width, height }
    }

    fn look() -> Look {
        Look { precision: Precision::F32, max_iter: 1000 }
    }

    /// A canvas whose every tile was computed exactly.
    fn finished(view: View) -> Canvas {
        let mut c = Canvas::new(view, look());
        c.tiles.fill(Tile { res: 1.0, exact: true });
        c
    }

    #[test]
    fn whole_pixel_pan_keeps_overlap_exact() {
        // 25 x 19 tiles; move the view 40 px right: content shifts left.
        let old = finished(view(400, 304, 0.0, 0.01));
        let (new, xf) = reprojected(&old, view(400, 304, 0.40, 0.01), look());
        assert_eq!(xf.ratio, 1.0);
        assert_eq!(xf.shift, (40.0, 0.0));
        let stale: Vec<usize> = (0..new.tiles.len()).filter(|&i| !new.tiles[i].exact).collect();
        // 40 revealed columns touch the last 3 tile columns, plus one column
        // whose source straddles the old edge: at most 4 of 25 columns.
        assert!(stale.len() <= 4 * 19 && stale.len() >= 3 * 19, "{} stale", stale.len());
        assert!(stale.iter().all(|&i| new.origin(i)[0] >= 400 - 4 * TILE));
        assert!(new.tiles.iter().filter(|t| t.exact).all(|t| t.res == 1.0));
    }

    #[test]
    fn sub_pixel_pan_is_not_exact() {
        let old = finished(view(400, 304, 0.0, 0.01));
        let (new, _) = reprojected(&old, view(400, 304, 0.005, 0.01), look());
        assert!(new.tiles.iter().all(|t| !t.exact));
    }

    #[test]
    fn zoom_in_scales_blur_and_keeps_every_tile() {
        let old = finished(view(400, 304, 0.0, 0.01));
        let (new, xf) = reprojected(&old, view(400, 304, 0.0, 0.01 / 1.5), look());
        assert!((xf.ratio - 1.0 / 1.5).abs() < 1e-12);
        assert!(new.tiles.iter().all(|t| !t.exact && (t.res - 1.5).abs() < 1e-3));
    }

    #[test]
    fn zoom_out_leaves_border_missing() {
        let old = finished(view(400, 304, 0.0, 0.01));
        let (new, _) = reprojected(&old, view(400, 304, 0.0, 0.02), look());
        // The middle quarter comes from denser old samples...
        assert!(new.tiles.iter().filter(|t| t.res.is_finite()).all(|t| t.res <= 0.5 + 1e-6));
        // ...and the rest has no source at all.
        let missing = new.tiles.iter().filter(|t| t.res.is_infinite()).count() as f64;
        let frac = missing / new.tiles.len() as f64;
        assert!(frac > 0.6 && frac < 0.85, "missing fraction {frac}");
    }

    #[test]
    fn reprojection_inherits_worst_source_tile() {
        let mut old = finished(view(400, 304, 0.0, 0.01));
        old.tiles[0] = MISSING; // top-left corner never computed
        let (new, _) = reprojected(&old, view(400, 304, 0.0, 0.01 / 1.5), look());
        // Zooming in toward the center pulls nothing from the corner...
        assert!(new.tiles.iter().all(|t| t.res.is_finite()));
        // ...but a pure pan that brings the corner into the middle does.
        let (moved, _) = reprojected(&old, view(400, 304, -2.0, 0.01), look());
        assert!(moved.tiles.iter().any(|t| t.res.is_infinite()));
    }

    #[test]
    fn sweep_completes_once_every_tile_is_covered() {
        let mut sweep = Sweep::new(4);
        let stats = |max_escaped, near_limit| FrameStats {
            gpu_ms: 1.0,
            samples: 100,
            max_escaped,
            near_limit,
            interior: 10,
        };
        assert!(!sweep.add(&stats(50, 1), &[0, 1]));
        assert!(!sweep.add(&stats(70, 2), &[1, 2]), "tile 1 twice must not count double");
        assert!(sweep.add(&stats(60, 3), &[3]));
        assert_eq!(sweep.stats.samples, 300);
        assert_eq!(sweep.stats.max_escaped, 70);
        assert_eq!(sweep.stats.near_limit, 6);
        assert_eq!(sweep.stats.interior, 30);
    }
}
