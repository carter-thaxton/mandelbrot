//! The progressive renderer's decisions: what work each frame does, and to
//! what iteration limits.
//!
//! The scheduler (see `progressive.rs`) is the mechanism: it keeps the
//! canvas, reprojects it when the view moves, and carries out plans. A
//! [`Policy`] makes the decisions, in two calls per frame:
//!   - [`Policy::plan`] looks at the canvas ([`PlanInput`]) and chooses this
//!     frame's work ([`Plan`]). It changes nothing, so it can be evaluated
//!     (and tested) on its own.
//!   - [`Policy::observe`] learns from what that work measured: batch sizes,
//!     statistics sweeps, the iteration limit.
//!
//! User input that affects iteration depth (zooming, the detail keys) reaches
//! the policy as [`Input`] events, queued by the explorer.
//!
//! [`Policy::master`] is the behavior the explorer had before policies
//! existed, exactly: one automatic iteration limit for the whole view,
//! predicted from the zoom depth and corrected from GPU statistics, judged
//! on whole-view evidence only. A single batch is a biased sample (a few
//! tiles along the set's boundary always look under-iterated), and acting on
//! those would ratchet the limit upward. Two sources of evidence:
//!   - While the view is changing, a *probe* every probe interval: one frame
//!     spent on a coarse whole-view pass that gathers statistics but writes
//!     no pixels. This keeps the limit current during continuous zooming and
//!     panning, when tiles are mostly reused and no sweep would complete.
//!   - A *sweep*: batches accumulated until every tile has been computed for
//!     the current view and limit (full-resolution evidence once settled).
//!
//! The direction (see the design doc) is per-tile limits steered by each
//! tile's statistics, with the frame-wide limit reduced to what the
//! reference orbit must cover ([`Policy::ceiling`]).

use std::time::{Duration, Instant};

use crate::render::{BatchStats, FrameStats, TILE, TileStats};

pub const MIN_ITER: u32 = 256;
pub const MAX_ITER: u32 = 200_000;
/// Automatic iteration decisions need at least this many samples: the 0.1%
/// cut-off test is meaningless on a few thousand (it fired on 8 pixels).
pub const MIN_DECISION_SAMPLES: u64 = 20_000;

/// In auto mode, each zoom by a factor f scales the iteration limit by
/// f^ITERS_PER_ZOOM, so it keeps roughly in step while zooming; GPU
/// statistics then correct it for the actual location. Fitted to the seahorse
/// valley (5.4k needed at 1e10, 62k at 1e30); flatter places just get
/// corrected down.
const ITERS_PER_ZOOM: f64 = 0.04;
/// Most the predictor may raise the limit in one view change. A sudden deep
/// jump would otherwise multiply it by thousands (straight to MAX_ITER), and
/// from above it can get stuck: around a minibrot a few pixels always escape
/// very late, so the shrink rule never fires. From below, the statistics
/// double it cheaply until it's enough, then stop.
const MAX_PREDICTED_GROWTH: f64 = 2.0;

/// GPU time to aim for per frame (ms). Leaves headroom inside a 60 Hz frame.
const FRAME_BUDGET_MS: f64 = 12.0;
/// Never compute fewer samples per frame than this. A dispatch lasts at least
/// as long as its slowest pixel, so below the size that fills the GPU a
/// smaller batch saves no time, it only idles the GPU. On deep views that
/// floor (one slow pixel, ~16 ms at 32k iterations) exceeds the budget, and
/// without this minimum the batches shrank to ~8 tiles: 30x slower overall.
const MIN_SAMPLES_PER_FRAME: f64 = 65536.0;
/// Minimum time between statistics probes while the view keeps changing.
const PROBE_INTERVAL: Duration = Duration::from_millis(120);
/// Probes sample at least this coarsely (they only need statistics).
const PROBE_MIN_STEP: u32 = 4;

/// Starting guess for the iteration limit at `zoom` (auto mode then corrects
/// it from what the GPU actually sees).
pub fn estimate_iter(zoom: f64) -> u32 {
    let decades = zoom.max(1.0).log10();
    (200.0 + 250.0 * decades).clamp(MIN_ITER as f64, 20_000.0) as u32
}

/// User input that affects iteration depth, in the order it happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Input {
    /// The view zoomed by this factor (> 1 zooms in).
    Zoom(f64),
    /// Back to automatic, starting from the estimate at `zoom`.
    Reset { zoom: f64 },
    /// The detail keys: scale the limit by this factor.
    ScaleDetail(f64),
    /// A fixed limit (turns automatic off).
    SetLimit(u32),
    ToggleAuto,
}

/// One canvas tile as the policy sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileState {
    /// Top-left pixel.
    pub origin: [u32; 2],
    /// Effective sample spacing of the current contents, in pixels
    /// (1 = full detail, infinity = nothing there yet).
    pub res: f32,
    /// Computed at full resolution for the canvas's current view and limit.
    pub exact: bool,
    /// The iteration limit the contents were computed at (0: none yet).
    pub limit: u32,
    /// Statistics from the tile's last draw for the current view.
    pub stats: Option<TileStats>,
}

impl TileState {
    pub fn missing(origin: [u32; 2]) -> Self {
        Self { origin, res: f32::INFINITY, exact: false, limit: 0, stats: None }
    }
}

/// What changed about the canvas since the last frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Change {
    None,
    /// A new canvas (first frame, or the window was resized).
    Fresh,
    /// The view moved, or the canvas's limit changed.
    Changed,
}

/// What the policy sees each frame.
pub struct PlanInput<'a> {
    pub now: Instant,
    /// Canvas size in pixels.
    pub width: u32,
    pub height: u32,
    /// Magnification relative to the home view.
    pub zoom: f64,
    /// Every canvas tile, row-major.
    pub tiles: &'a [TileState],
    pub change: Change,
}

impl PlanInput<'_> {
    /// Squared distance from a tile's center to the canvas center (pixels).
    fn dist2(&self, i: usize) -> f64 {
        let [x, y] = self.tiles[i].origin;
        let dx = (x + TILE / 2) as f64 - self.width as f64 / 2.0;
        let dy = (y + TILE / 2) as f64 - self.height as f64 / 2.0;
        dx * dx + dy * dy
    }
}

/// One frame's work.
#[derive(Clone, Debug, PartialEq)]
pub enum Plan {
    /// Nothing to compute.
    Idle,
    /// Gather statistics for the whole view at this step; write no pixels.
    Probe { step: u32 },
    /// Compute these tiles (indices into [`PlanInput::tiles`]) at this step,
    /// each to its own limit.
    Draw { tiles: Vec<usize>, step: u32, limits: Vec<u32> },
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

pub struct Policy {
    /// The iteration limit for the whole view.
    limit: u32,
    /// Adjust `limit` from GPU statistics (see [`Policy::judge`]).
    auto: bool,
    /// Zoom-driven change to `limit` not yet applied (applied in >= 5%
    /// steps so continuous zooming doesn't change it every frame).
    drift: f64,
    /// Current sweep (reset on any change).
    sweep: Option<Sweep>,
    /// Samples to compute per frame to fill FRAME_BUDGET_MS; 0 until first
    /// measured (see [`Policy::record`]).
    samples_per_frame: f64,
    /// The view or limit changed since the last statistics probe.
    probe_due: bool,
    last_probe: Option<Instant>,
    /// Probe statistics accumulated until there are enough to decide on
    /// (coarse probes of expensive views have only a few thousand samples).
    probe_stats: FrameStats,
    probe_interval: Duration,
}

impl Default for Policy {
    fn default() -> Self {
        Self::master()
    }
}

impl Policy {
    /// The explorer's behavior before policies: see the module docs.
    pub fn master() -> Self {
        Self {
            limit: MIN_ITER,
            auto: true,
            drift: 1.0,
            sweep: None,
            samples_per_frame: 0.0,
            probe_due: false,
            last_probe: None,
            probe_stats: FrameStats::default(),
            probe_interval: PROBE_INTERVAL,
        }
    }

    /// Probe statistics at most every `interval` while the view changes
    /// (zero: on every change).
    pub fn with_probe_interval(mut self, interval: Duration) -> Self {
        self.probe_interval = interval;
        self
    }

    /// The limit the reference orbit must cover: no tile goes deeper.
    pub fn ceiling(&self) -> u32 {
        self.limit
    }

    pub fn is_auto(&self) -> bool {
        self.auto
    }

    /// Iteration settings, for the window title.
    pub fn describe(&self) -> String {
        format!("iters {} {}", self.limit, if self.auto { "auto" } else { "manual" })
    }

    pub fn input(&mut self, event: Input) {
        match event {
            Input::Zoom(factor) => self.track_depth(factor),
            Input::Reset { zoom } => {
                self.auto = true;
                self.limit = estimate_iter(zoom);
            }
            Input::ScaleDetail(factor) => {
                self.auto = false;
                self.limit = (self.limit as f64 * factor).clamp(32.0, MAX_ITER as f64) as u32;
            }
            Input::SetLimit(n) => {
                self.auto = false;
                self.limit = n.clamp(32, MAX_ITER);
            }
            Input::ToggleAuto => self.auto = !self.auto,
        }
    }

    /// In auto mode, scale the limit with a zoom by `factor` (> 1 zooms in),
    /// in >= 5% steps. Growth is capped per call (MAX_PREDICTED_GROWTH);
    /// shrinking isn't, since it's cheap and the statistics raise it again
    /// if needed.
    fn track_depth(&mut self, factor: f64) {
        if self.auto {
            self.drift *= factor.powf(ITERS_PER_ZOOM).min(MAX_PREDICTED_GROWTH);
            if !(1.0 / 1.05..=1.05).contains(&self.drift) {
                let n = self.limit as f64 * self.drift;
                self.limit = n.round().clamp(MIN_ITER as f64, MAX_ITER as f64) as u32;
                self.drift = 1.0;
            }
        }
    }

    /// Judge the limit from whole-view statistics drawn with it. Returns
    /// true if the limit changed.
    ///
    /// - More than 0.1% of the *escaped* samples landing in the top quarter
    ///   of the range means the escape-count tail runs past the limit, cutting
    ///   off real detail (those pixels would otherwise go black): double it.
    ///   Measured on real views, this lands within one doubling of the point
    ///   where the next doubling changes < 0.01% of pixels. (A wider window,
    ///   the top half, overshoots 2x when escape counts cluster just below the
    ///   limit. Counting against all samples instead of escaped ones stalls
    ///   when most of the view is still black.)
    /// - If even the slowest escaping sample used less than a third of the
    ///   limit, the extra iterations only cost time on interior pixels: shrink
    ///   to 2x that sample. That leaves the top quarter empty, so the two
    ///   rules don't oscillate.
    pub fn judge(&mut self, s: &FrameStats) -> bool {
        if !self.auto || s.samples < MIN_DECISION_SAMPLES {
            return false;
        }
        let l = self.limit;
        let escaped = s.samples.saturating_sub(s.interior as u64);
        let cut_off = s.near_limit >= 4 && s.near_limit as f64 > escaped as f64 * 1e-3;
        let all_black = s.max_escaped == 0 && s.interior > 0;
        let new = if cut_off || all_black {
            l.saturating_mul(2)
        } else if s.max_escaped < l / 3 {
            s.max_escaped.saturating_mul(2)
        } else {
            l
        }
        .clamp(MIN_ITER, MAX_ITER);
        let changed = new != l;
        self.limit = new;
        changed
    }

    /// [`Policy::judge`], logging the decision if MANDEL_DEBUG_ITERS is set.
    fn judge_logged(&mut self, what: &str, zoom: f64, s: &FrameStats) {
        let before = self.limit;
        self.judge(s);
        if std::env::var_os("MANDEL_DEBUG_ITERS").is_some() {
            eprintln!("{what} at zoom {zoom:.2e}: limit {before} -> {}  {s:?}", self.limit);
        }
    }

    /// Samples affordable in one frame.
    fn budget_samples(&self, input: &PlanInput) -> f64 {
        if self.samples_per_frame == 0.0 {
            // Unknown cost: assume a 4x4-block preview of the view fits.
            (input.width as f64 * input.height as f64 / 16.0).max(MIN_SAMPLES_PER_FRAME)
        } else {
            self.samples_per_frame.max(MIN_SAMPLES_PER_FRAME)
        }
    }

    /// Coarsest step a whole-view preview needs to fit the budget (power of 2).
    fn preview_step(&self, input: &PlanInput) -> u32 {
        let full = input.width as f64 * input.height as f64;
        let s = (full / self.budget_samples(input)).sqrt().ceil().max(1.0) as u32;
        s.next_power_of_two().min(TILE)
    }

    /// Choose this frame's work.
    pub fn plan(&self, input: &PlanInput) -> Plan {
        let probe_due = self.probe_due || input.change == Change::Changed;
        if probe_due && self.auto && self.last_probe.is_none_or(|t| input.now.duration_since(t) >= self.probe_interval) {
            // Spend this frame on whole-view statistics.
            return Plan::Probe { step: self.preview_step(input).max(PROBE_MIN_STEP) };
        }
        match self.pick(input) {
            Some((tiles, step)) => {
                let limits = vec![self.limit; tiles.len()];
                Plan::Draw { tiles, step, limits }
            }
            None => Plan::Idle,
        }
    }

    /// Worst tiles first, within budget.
    fn pick(&self, input: &PlanInput) -> Option<(Vec<usize>, u32)> {
        let t = input.tiles;
        let budget = self.budget_samples(input);
        let preview = self.preview_step(input);
        let by_priority = |mut v: Vec<usize>| {
            // Worst first; among equals, nearest the center first.
            v.sort_by(|&a, &b| t[b].res.total_cmp(&t[a].res).then(input.dist2(a).total_cmp(&input.dist2(b))));
            v
        };

        // 1. Anything coarser than the preview step.
        let coarse: Vec<usize> = (0..t.len()).filter(|&i| t[i].res > preview as f32).collect();
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
        let stale: Vec<usize> = (0..t.len()).filter(|&i| !t[i].exact).collect();
        if stale.is_empty() {
            return None;
        }
        let n = ((budget / (TILE * TILE) as f64) as usize).max(1);
        let mut tiles = by_priority(stale);
        tiles.truncate(n);
        Some((tiles, 1))
    }

    /// Learn from carrying out `plan` (planned from `input`), which measured
    /// `stats` (None if nothing was drawn).
    pub fn observe(&mut self, input: &PlanInput, plan: &Plan, stats: Option<&BatchStats>) {
        if input.change != Change::None {
            self.sweep = None;
        }
        if input.change == Change::Changed {
            self.probe_due = true;
        }
        match (plan, stats) {
            (Plan::Probe { .. }, Some(s)) => {
                self.record(&s.total);
                self.probe_stats.merge(&s.total);
                if self.probe_stats.samples >= MIN_DECISION_SAMPLES {
                    let probe = std::mem::take(&mut self.probe_stats);
                    self.judge_logged("probe", input.zoom, &probe);
                }
                self.probe_due = false;
                self.last_probe = Some(input.now);
            }
            (Plan::Draw { tiles, .. }, Some(s)) => {
                self.record(&s.total);
                // Judge the limit once the whole view has been covered.
                let sweep = self.sweep.get_or_insert_with(|| Sweep::new(input.tiles.len()));
                if sweep.add(&s.total, tiles) {
                    // Too few samples (a very coarse pass) means no decision;
                    // start a new sweep rather than topping this one up with
                    // partial batches, which would bias it.
                    let sweep = self.sweep.take().unwrap();
                    if sweep.stats.samples >= MIN_DECISION_SAMPLES {
                        self.judge_logged("sweep", input.zoom, &sweep.stats);
                    }
                }
            }
            _ => {}
        }
    }

    /// Size the next frame's work from how long this draw took: drop at once
    /// when over budget, grow at most 2x per frame when under (see also
    /// MIN_SAMPLES_PER_FRAME).
    fn record(&mut self, stats: &FrameStats) {
        if stats.samples == 0 {
            return;
        }
        let target = stats.samples as f64 * FRAME_BUDGET_MS / stats.gpu_ms.max(1e-3);
        let current = self.samples_per_frame;
        self.samples_per_frame = if current == 0.0 { target } else { target.min(current * 2.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(samples: u64, max_escaped: u32, near_limit: u32, interior: u32) -> FrameStats {
        FrameStats { gpu_ms: 0.0, samples, max_escaped, near_limit, interior }
    }

    fn with_limit(limit: u32) -> Policy {
        Policy { limit, ..Policy::master() }
    }

    #[test]
    fn auto_iterations_grow_shrink_and_settle() {
        let mut p = with_limit(1000);
        // Lots of samples escaping just under the limit: cut off, so grow.
        assert!(p.judge(&stats(100_000, 999, 5_000, 20_000)));
        assert_eq!(p.ceiling(), 2000);
        // Mostly black, and most of what escaped did so near the limit: the
        // view is under-iterated even though few samples are near the limit.
        assert!(p.judge(&stats(100_000, 1990, 50, 99_900)));
        assert_eq!(p.ceiling(), 4000);
        let mut p = with_limit(2000);
        // The slowest escape far below the limit: shrink to 2x it.
        assert!(p.judge(&stats(100_000, 300, 0, 20_000)));
        assert_eq!(p.ceiling(), 600);
        // Same evidence again: stable.
        assert!(!p.judge(&stats(100_000, 300, 0, 20_000)));
        // Too few samples to decide on.
        assert!(!p.judge(&stats(1_000, 1, 0, 0)));
        // Manual mode ignores stats.
        p.input(Input::ScaleDetail(2.0));
        assert!(!p.judge(&stats(100_000, 1, 0, 0)));
    }

    #[test]
    fn iterations_track_zoom_depth() {
        let mut p = with_limit(1000);
        // 1e5 zoom in many small steps, like a pinch.
        for _ in 0..100 {
            p.input(Input::Zoom(1e5f64.powf(0.01)));
        }
        let expected = 1000.0 * 1e5f64.powf(ITERS_PER_ZOOM);
        let got = p.ceiling() as f64;
        assert!((got / expected - 1.0).abs() < 0.06, "{got} vs {expected}");
        // Manual mode leaves it alone.
        p.input(Input::SetLimit(500));
        p.input(Input::Zoom(1e5));
        assert_eq!(p.ceiling(), 500);
    }

    #[test]
    fn sudden_jumps_raise_iterations_only_a_little() {
        let mut p = with_limit(256);
        p.input(Input::Zoom(1e80));
        // The full prediction would be ~1500x; statistics ramp up from here.
        assert!(p.ceiling() > 256 && p.ceiling() as f64 <= 256.0 * MAX_PREDICTED_GROWTH);
        // Jumping back out shrinks freely.
        let mut p = with_limit(50_000);
        p.input(Input::Zoom(1e-80));
        assert!(p.ceiling() < 50_000 / 100, "{}", p.ceiling());
    }

    #[test]
    fn detail_and_reset_inputs() {
        let mut p = with_limit(1000);
        p.input(Input::ScaleDetail(1.5));
        assert_eq!((p.ceiling(), p.is_auto()), (1500, false));
        p.input(Input::ScaleDetail(1e-9));
        assert_eq!(p.ceiling(), 32, "clamped");
        p.input(Input::ToggleAuto);
        assert!(p.is_auto());
        p.input(Input::Reset { zoom: 1e4 });
        assert_eq!((p.ceiling(), p.is_auto()), (estimate_iter(1e4), true));
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

    fn input(tiles: &[TileState], change: Change) -> PlanInput<'_> {
        PlanInput { now: Instant::now(), width: 64, height: 32, zoom: 1.0, tiles, change }
    }

    fn canvas() -> Vec<TileState> {
        (0..2).flat_map(|y| (0..4).map(move |x| TileState::missing([x * TILE, y * TILE]))).collect()
    }

    #[test]
    fn plans_previews_then_refinement_then_nothing() {
        let p = Policy::master();
        let mut tiles = canvas();
        // Nothing there yet: everything fits the budget at full resolution.
        let Plan::Draw { tiles: picked, step, limits } = p.plan(&input(&tiles, Change::Fresh)) else { panic!() };
        assert_eq!((picked.len(), step), (8, 1));
        assert!(limits.iter().all(|&l| l == p.ceiling()));
        // Present at full resolution but inexact (as after a small zoom
        // within the preview step): refine them, nearest the center first.
        for t in &mut tiles {
            *t = TileState { res: 1.0, ..*t };
        }
        tiles[0].exact = true;
        let Plan::Draw { tiles: picked, step, .. } = p.plan(&input(&tiles, Change::None)) else { panic!() };
        assert_eq!(step, 1);
        assert_eq!(picked.len(), 7);
        assert!([1, 2, 5, 6].contains(&picked[0]), "{picked:?}");
        // All exact: idle.
        tiles.iter_mut().for_each(|t| t.exact = true);
        assert_eq!(p.plan(&input(&tiles, Change::None)), Plan::Idle);
    }

    #[test]
    fn view_changes_trigger_one_probe_per_interval() {
        let mut p = Policy::master();
        let tiles: Vec<TileState> =
            canvas().into_iter().map(|t| TileState { res: 1.0, exact: true, ..t }).collect();
        let changed = input(&tiles, Change::Changed);
        let plan = p.plan(&changed);
        assert!(matches!(plan, Plan::Probe { step } if step >= PROBE_MIN_STEP), "{plan:?}");
        p.observe(&changed, &plan, Some(&BatchStats::default()));
        // Within the interval, another change doesn't probe again.
        let again = PlanInput { now: changed.now + PROBE_INTERVAL / 2, ..input(&tiles, Change::Changed) };
        assert_eq!(p.plan(&again), Plan::Idle);
        p.observe(&again, &Plan::Idle, None);
        // After it, the pending change gets its probe.
        let later = PlanInput { now: changed.now + PROBE_INTERVAL, ..input(&tiles, Change::None) };
        assert!(matches!(p.plan(&later), Plan::Probe { .. }));
        // A new canvas alone isn't a reason to probe; nor is manual mode.
        let mut q = Policy::master();
        assert_eq!(q.plan(&input(&tiles, Change::Fresh)), Plan::Idle);
        q.input(Input::ToggleAuto);
        assert_eq!(q.plan(&input(&tiles, Change::Changed)), Plan::Idle);
    }
}
