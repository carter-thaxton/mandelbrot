//! Progressive rendering with reuse: keep every frame within a small GPU
//! budget so the event loop stays responsive, and never throw away pixels
//! that are still (approximately) right.
//!
//! The canvas is divided into TILE x TILE tiles. For each tile we track how
//! good its current contents are ([`TileState`]):
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
//! This module is the mechanism; what to compute each frame, and to what
//! iteration limits, is up to its [`Policy`] (see `policy.rs`). Each display
//! refresh, [`Scheduler::step`] brings the canvas up to date with the view,
//! asks the policy for a [`Plan`], carries it out, and reports the results
//! back to the policy.

use crate::app::{Colors, Explorer, FrameParams, Precision};
use crate::bigfixed::Fixed;
use crate::policy::{Change, Plan, PlanInput, Policy, TileState};
use crate::render::{BatchStats, Output, Renderer, Reprojection, Result, TILE, TileBatch, TileJob};

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
    tiles: Vec<TileState>,
}

impl Canvas {
    fn new(view: View, look: Look) -> Self {
        let (cols, rows) = (view.width.div_ceil(TILE), view.height.div_ceil(TILE));
        let tiles = (0..rows).flat_map(|r| (0..cols).map(move |c| TileState::missing([c * TILE, r * TILE]))).collect();
        Self { view, look, cols, rows, tiles }
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
    let mut tiles = new.tiles.clone();
    for tile in tiles.iter_mut() {
        let [x0, y0] = tile.origin;
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
        let (mut res, mut exact, mut limit) = (0.0f32, pure_pan, 0);
        for r in r0..r1 {
            for c in c0..c1 {
                let t = &old.tiles[(r * old.cols + c) as usize];
                res = res.max(t.res);
                exact &= t.exact;
                limit = limit.max(t.limit);
            }
        }
        // Old samples r old pixels apart are r / ratio new pixels apart.
        // Statistics describe the old view, so they don't carry over.
        *tile = TileState { res: (res as f64 / ratio) as f32, exact, limit, ..*tile };
    }
    new.tiles = tiles;
    let xf = Reprojection { ratio, shift, old_size: (o.width, o.height) };
    (new, xf)
}

#[derive(Default)]
pub struct Scheduler {
    canvas: Option<Canvas>,
    policy: Policy,
    /// GPU time spent since the view last changed.
    work_ms: f64,
    /// Colors of the last present (a color change alone needs a re-present).
    presented: Option<Colors>,
}

impl Scheduler {
    pub fn with_policy(policy: Policy) -> Self {
        Self { policy, ..Default::default() }
    }

    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// The renderer's canvas was recreated (e.g. window resize): forget it.
    pub fn reset(&mut self) {
        self.canvas = None;
        self.presented = None;
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

    /// Bring the canvas bookkeeping up to date with the current view,
    /// reprojecting the old image if that's worthwhile. Returns whether the
    /// canvas moved (what's on screen is stale) and what changed.
    fn sync(&mut self, view: View, p: &FrameParams, renderer: &mut dyn Renderer) -> Result<(bool, Change)> {
        let look = Look::of(p);
        let fresh = match &self.canvas {
            None => true,
            Some(old) => old.view.width != view.width || old.view.height != view.height,
        };
        if fresh {
            self.canvas = Some(Canvas::new(view, look));
            self.work_ms = 0.0;
            return Ok((true, Change::Fresh));
        }
        let old = self.canvas.take().unwrap();
        let moved = old.view != view;
        let mut changed = moved;
        let mut canvas = if moved {
            let (new, xf) = reprojected(&old, view, look.clone());
            if new.tiles.iter().any(|t| t.res.is_finite()) {
                renderer.reproject(&xf)?;
            }
            self.work_ms = 0.0;
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
                // Precision changed, or iterations by 2x or more: redo
                // everything. Old pixels stay on screen until the coarse
                // pass replaces them.
                canvas.tiles.iter_mut().for_each(|t| *t = TileState::missing(t.origin));
            }
            canvas.look = look;
            changed = true;
            self.work_ms = 0.0;
        }
        self.canvas = Some(canvas);
        Ok((moved, if changed { Change::Changed } else { Change::None }))
    }

    /// Do one frame's worth of work. Returns true if more work remains (the
    /// caller should request another redraw).
    pub fn step(&mut self, explorer: &mut Explorer, renderer: &mut dyn Renderer) -> Result<bool> {
        self.step_at(std::time::Instant::now(), explorer, renderer)
    }

    /// [`Scheduler::step`] at time `now` (the policy's clock), for replaying
    /// input on a simulated clock.
    pub fn step_at(&mut self, now: std::time::Instant, explorer: &mut Explorer, renderer: &mut dyn Renderer) -> Result<bool> {
        for event in explorer.take_inputs() {
            self.policy.input(event);
        }
        explorer.iter_limit = self.policy.ceiling();
        let params = explorer.frame_params();
        let colors = explorer.colors();
        let (moved, change) = self.sync(View::of(explorer), &params, renderer)?;
        // What's on screen is stale if the canvas moved or the colors changed.
        let stale = moved || self.presented != Some(colors);

        let canvas = self.canvas.as_ref().unwrap();
        let input = PlanInput {
            now,
            width: canvas.view.width,
            height: canvas.view.height,
            zoom: explorer.zoom(),
            tiles: &canvas.tiles,
            change,
        };
        let plan = self.policy.plan(&input);
        let present = |renderer: &mut dyn Renderer, presented: &mut Option<Colors>| -> Result<()> {
            renderer.present(&colors)?;
            *presented = Some(colors);
            Ok(())
        };
        let stats: Option<BatchStats> = match &plan {
            Plan::Idle => {
                if stale {
                    // A reprojection or color change with nothing to compute.
                    present(renderer, &mut self.presented)?;
                }
                None
            }
            Plan::Probe { step } => {
                // Statistics don't change the canvas: show what's there.
                if stale {
                    present(renderer, &mut self.presented)?;
                }
                let probe = TileBatch::all(input.width, input.height, params.max_iter, *step, Output::StatsOnly);
                Some(renderer.draw(&params, &probe)?)
            }
            Plan::Draw { tiles, step, limits } => {
                let jobs = tiles.iter().zip(limits).map(|(&i, &l)| TileJob::new(input.tiles[i].origin, l)).collect();
                let stats = renderer.draw(&params, &TileBatch::new(jobs, *step, Output::Pixels))?;
                present(renderer, &mut self.presented)?;
                Some(stats)
            }
        };
        if let Some(s) = &stats {
            self.work_ms += s.total.gpu_ms;
        }
        self.policy.observe(&input, &plan, stats.as_ref());

        if let (Plan::Draw { tiles, step, limits }, Some(s)) = (&plan, &stats) {
            let canvas = self.canvas.as_mut().unwrap();
            for ((&i, &limit), tile_stats) in tiles.iter().zip(limits).zip(&s.tiles) {
                let t = &mut canvas.tiles[i];
                *t = TileState { res: *step as f32, exact: *step == 1, limit, stats: Some(*tile_stats), ..*t };
            }
        }
        // A changed limit shows up as a new look on the next step.
        explorer.iter_limit = self.policy.ceiling();
        Ok(plan != Plan::Idle)
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
        c.tiles.iter_mut().for_each(|t| *t = TileState { res: 1.0, exact: true, limit: 1000, ..*t });
        c
    }

    #[test]
    fn whole_pixel_pan_keeps_overlap_exact() {
        // 25 x 19 tiles; move the view 40 px right: content shifts left.
        let old = finished(view(400, 304, 0.0, 0.01));
        let (new, xf) = reprojected(&old, view(400, 304, 0.40, 0.01), look());
        assert_eq!(xf.ratio, 1.0);
        assert_eq!(xf.shift, (40.0, 0.0));
        let stale: Vec<&TileState> = new.tiles.iter().filter(|t| !t.exact).collect();
        // 40 revealed columns touch the last 3 tile columns, plus one column
        // whose source straddles the old edge: at most 4 of 25 columns.
        assert!(stale.len() <= 4 * 19 && stale.len() >= 3 * 19, "{} stale", stale.len());
        assert!(stale.iter().all(|t| t.origin[0] >= 400 - 4 * TILE));
        assert!(new.tiles.iter().filter(|t| t.exact).all(|t| t.res == 1.0 && t.limit == 1000));
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
        old.tiles[0] = TileState::missing(old.tiles[0].origin); // top-left corner never computed
        let (new, _) = reprojected(&old, view(400, 304, 0.0, 0.01 / 1.5), look());
        // Zooming in toward the center pulls nothing from the corner...
        assert!(new.tiles.iter().all(|t| t.res.is_finite()));
        // ...but a pure pan that brings the corner into the middle does.
        let (moved, _) = reprojected(&old, view(400, 304, -2.0, 0.01), look());
        assert!(moved.tiles.iter().any(|t| t.res.is_infinite()));
    }
}
