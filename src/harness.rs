//! Evaluation harness for progressive rendering: scripted scenarios played
//! on a fixed 120 fps clock, with per-frame timing, image quality against
//! exact references, and fingerprints for comparing builds.
//!
//! A [`Scenario`] starts settled at a place, plays a script of input
//! (zooming about a point, dragging, flying, jumping) one frame at a time,
//! and ends by holding still until the scheduler is idle. A frame is one
//! input update plus one [`Scheduler::step_at`], as in the app. The clock
//! advances 1/120 s per frame however long the frame took, so the same input
//! lands on the same frame in every run. (In the app a slow frame makes the
//! input jump ahead; here its cost shows up in the jank figures instead.)
//!
//! Two kinds of run ([`Config`]):
//!   - Real ([`Config::real`]): window-sized, measured GPU time, background
//!     reference orbits. Reports frame times and image quality.
//!   - Deterministic ([`Config::compat`]): small, with GPU time modeled from
//!     the sample count and reference orbits computed synchronously, so a
//!     build always produces exactly the same frames. Every batch,
//!     reprojection and changed canvas is hashed into a fingerprint, and
//!     comparing fingerprints ([`compare_fingerprints`]) shows whether a
//!     change altered behavior (it must not, for a new mechanism in its
//!     degenerate setting).
//!
//! Quality, as fractions of pixels whose color differs noticeably
//! ([`differing`]):
//!   - error: what's displayed vs. a deep reference, an exact render of the
//!     same view at TRUNCATION_FACTOR times the frame's iteration limit. Each
//!     wrong pixel is attributed by the state of its tile ([`error_by_cause`]):
//!       - unfinished: the tile isn't computed exactly for this view yet
//!         (blockiness, blur, staleness)
//!       - truncation: it is, so the difference is detail its iteration
//!         limit cuts off; *lowered* counts the part in tiles the policy gave
//!         a lower limit than the frame's
//!   - speckle: adjacent pixel pairs with very different colors ([`speckle`])
//!   - popping: pixels changing from one frame to the next while still

use std::time::{Duration, Instant};

use crate::app::{Colors, Explorer, FrameParams, Place};
use crate::bookmarks::PRESETS;
use crate::flight::Flight;
use crate::policy::{MAX_ITER, Policy, Settings, TileState};
use crate::progressive::Scheduler;
use crate::render::{BatchStats, FrameStats, Output, Renderer, Reprojection, Result, TILE, TileBatch};

pub const FPS: f64 = 120.0;
/// Frame time at FPS (ms); longer frames count as jank.
pub const FRAME_MS: f64 = 1000.0 / FPS;
/// Every scenario ends holding still at least this long.
const SETTLE_MIN_SECONDS: f64 = 2.0;
/// The deep reference uses this many times the frame's limit.
const TRUNCATION_FACTOR: u32 = 4;
/// Frames between fingerprint checkpoints.
const CHECKPOINT_FRAMES: usize = 60;
/// Modeled GPU cost in deterministic runs, per sample per iteration of the
/// limit: 12 ms buys 600k samples at 1000 iterations, or the 65536-sample
/// minimum at ~9000.
const MODEL_MS_PER_SAMPLE_ITER: f64 = 2e-8;
/// Flight speed in the scenarios, zoom decades per second (faster than the
/// app's default so the runs stay short).
const FLIGHT_SPEED: f64 = 1.5;
/// "Quiet" once fewer than this fraction of pixels change per still frame.
const QUIET_FRACTION: f64 = 0.001;

/// One piece of a scenario's input script.
#[derive(Clone, Debug)]
pub enum Action {
    /// Zoom by `per_second`x each second (< 1 zooms out) about a point given
    /// as a fraction of the view, like pinching under the cursor.
    Zoom { at: (f64, f64), per_second: f64, seconds: f64 },
    /// Drag the content by `per_second` view widths and heights per second.
    Pan { per_second: (f64, f64), seconds: f64 },
    /// Fly to a place at `speed` zoom decades per second.
    Fly { to: Place, speed: f64 },
    /// Show a place at once (a bookmark key).
    Jump(Place),
    /// No input.
    Still(f64),
}

#[derive(Clone, Debug)]
pub struct Scenario {
    pub name: &'static str,
    /// Where it starts, rendered to completion before the first frame.
    pub start: Place,
    pub script: Vec<Action>,
    /// Longest the final hold may last waiting for the scheduler to finish.
    pub settle_max: f64,
}

fn preset(name: &str) -> Place {
    let p = PRESETS.iter().find(|p| p.name == name).expect("preset");
    Place::parse(p.x, p.y, p.span).expect("preset coordinates parse")
}

fn home() -> Place {
    Explorer::new(1, 1).home_place()
}

/// The standard scenarios, most important first.
pub fn scenarios() -> Vec<Scenario> {
    use Action::*;
    vec![
        Scenario {
            name: "zoom",
            start: preset("seahorse valley"),
            script: vec![
                Zoom { at: (0.62, 0.41), per_second: 5.0, seconds: 1.0 },
                Still(0.25),
                Zoom { at: (0.62, 0.41), per_second: 5.0, seconds: 1.0 },
                Still(0.25),
                Zoom { at: (0.40, 0.60), per_second: 0.2, seconds: 0.75 },
            ],
            settle_max: 10.0,
        },
        Scenario {
            name: "flight-in",
            start: preset("seahorse valley"),
            script: vec![Fly { to: preset("spiral sea"), speed: FLIGHT_SPEED }],
            settle_max: 10.0,
        },
        Scenario {
            name: "flight-out",
            start: preset("elephant valley minibrot"),
            script: vec![Fly { to: home(), speed: FLIGHT_SPEED }],
            settle_max: 10.0,
        },
        Scenario {
            name: "pan",
            start: preset("elephant crossroads"),
            script: vec![
                Pan { per_second: (0.5, 0.0), seconds: 1.0 },
                Still(0.25),
                Pan { per_second: (-0.3, 0.4), seconds: 1.0 },
            ],
            settle_max: 10.0,
        },
        Scenario {
            name: "jump",
            start: home(),
            script: vec![Jump(preset("minibrot at 1e-82"))],
            settle_max: 20.0,
        },
    ]
}

#[derive(Clone, Debug)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    /// Modeled GPU time and synchronous reference orbits (reproducible).
    pub deterministic: bool,
    /// Hash every frame into a fingerprint.
    pub fingerprint: bool,
    /// Measure image quality (snapshots every frame, deep references).
    pub metrics: bool,
    /// While moving, measure error every this many frames (always while
    /// still).
    pub error_every: usize,
    /// Settings of the policy under test.
    pub settings: Settings,
}

impl Config {
    /// Window-sized, real timing, with quality metrics, with the app's
    /// settings.
    pub fn real() -> Self {
        Self {
            width: 2400,
            height: 1600,
            deterministic: false,
            fingerprint: false,
            metrics: true,
            error_every: 12,
            settings: Settings::tuned(),
        }
    }

    /// Small and reproducible, fingerprinted, for comparing behavior.
    pub fn compat() -> Self {
        Self {
            width: 640,
            height: 416,
            deterministic: true,
            fingerprint: true,
            metrics: false,
            error_every: 0,
            settings: Settings::master(),
        }
    }
}

/// One frame of a scenario.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// Seconds since the scenario's first frame.
    pub t: f64,
    /// Input changed the view this frame.
    pub moving: bool,
    /// Part of the final hold.
    pub settling: bool,
    pub wall_ms: f64,
    pub gpu_ms: f64,
    pub samples: u64,
    pub pixel_draws: u32,
    pub stats_draws: u32,
    pub reprojections: u32,
    /// The iteration limit the frame was drawn with.
    pub max_iter: u32,
    /// The scheduler had more work to do.
    pub busy: bool,
    /// See the module docs; error = unfinished + truncation.
    pub error: Option<f64>,
    pub unfinished: Option<f64>,
    pub truncation: Option<f64>,
    pub lowered: Option<f64>,
    pub speckle: Option<f64>,
    pub popping: Option<f64>,
}

/// A played scenario.
pub struct Run {
    pub name: &'static str,
    pub frames: Vec<Frame>,
    /// (frame, running hash) every CHECKPOINT_FRAMES frames and at the end.
    pub checkpoints: Vec<(usize, u64)>,
    /// Frames (and seconds) spent settling at the start, before frame 0.
    pub warmup_frames: usize,
    /// The final frame as displayed, and its deep reference (RGBA rows;
    /// with metrics only).
    pub final_image: Option<Vec<u8>>,
    pub final_deep: Option<Vec<u8>>,
}

/// 64-bit FNV-1a: tiny, and stable across Rust versions and runs.
#[derive(Clone, Copy)]
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn bytes(&mut self, data: &[u8]) {
        for &b in data {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }

    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// What the renderer was asked to do during one frame.
#[derive(Default)]
struct FrameLog {
    gpu_ms: f64,
    samples: u64,
    pixel_draws: u32,
    stats_draws: u32,
    reprojections: u32,
}

/// Passes calls through to a renderer, logging (and optionally hashing)
/// them, and in deterministic runs replacing measured GPU time with a model.
struct Recorder<'a> {
    inner: &'a mut dyn Renderer,
    /// Model GPU time as samples x limit x this (deterministic runs).
    model_cost: Option<f64>,
    log: FrameLog,
    hash: Option<Fnv>,
    /// Parameters of the most recent draw: what the canvas was computed with.
    params: Option<FrameParams>,
}

impl Recorder<'_> {
    fn hash_draw(&mut self, p: &FrameParams, b: &TileBatch) {
        let Some(h) = &mut self.hash else { return };
        h.u64(1);
        h.u64(p.center.0.to_bits());
        h.u64(p.center.1.to_bits());
        h.u64(p.scale.to_bits());
        h.u64(p.max_iter as u64);
        h.u64(p.precision as u64);
        h.u64(p.series_n as u64);
        h.u64(b.step as u64);
        h.u64((b.output == Output::StatsOnly) as u64);
        h.u64(b.tiles.len() as u64);
        for t in &b.tiles {
            h.u64(((t.origin[0] as u64) << 32) | t.origin[1] as u64);
            // Only a limit other than the frame's (so runs from before
            // per-tile limits hash the same).
            if t.max_iter != p.max_iter {
                h.u64(t.max_iter as u64);
            }
        }
    }
}

impl Renderer for Recorder<'_> {
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        self.inner.resize(width, height, scale_factor)
    }

    fn draw(&mut self, params: &FrameParams, batch: &TileBatch) -> Result<BatchStats> {
        self.hash_draw(params, batch);
        let mut batch_stats = self.inner.draw(params, batch)?;
        let stats = &mut batch_stats.total;
        if let Some(cost) = self.model_cost {
            stats.gpu_ms = stats.samples as f64 * params.max_iter as f64 * cost;
        }
        if let Some(h) = &mut self.hash {
            h.u64(stats.samples);
            h.u64(stats.max_escaped as u64);
            h.u64(stats.near_limit as u64);
            h.u64(stats.interior as u64);
        }
        self.log.gpu_ms += stats.gpu_ms;
        self.log.samples += stats.samples;
        match batch.output {
            Output::Pixels => self.log.pixel_draws += 1,
            Output::StatsOnly => self.log.stats_draws += 1,
        }
        self.params = Some(params.clone());
        Ok(batch_stats)
    }

    fn present(&mut self, colors: &Colors) -> Result<()> {
        self.inner.present(colors)
    }

    fn snapshot(&mut self, colors: &Colors) -> Result<Vec<u8>> {
        self.inner.snapshot(colors)
    }

    fn reproject(&mut self, xf: &Reprojection) -> Result<()> {
        if let Some(h) = &mut self.hash {
            h.u64(2);
            h.u64(xf.ratio.to_bits());
            h.u64(xf.shift.0.to_bits());
            h.u64(xf.shift.1.to_bits());
        }
        self.log.reprojections += 1;
        self.inner.reproject(xf)
    }

    fn capture(&mut self, params: &FrameParams, step: u32, colors: &Colors) -> Result<(Vec<u8>, FrameStats)> {
        self.inner.capture(params, step, colors)
    }
}

/// A deep render for one view and limit, kept while the view holds still.
struct Reference {
    key: (u64, u32),
    deep: Vec<u8>,
}

/// Plays one scenario.
struct Player<'a> {
    cfg: &'a Config,
    e: Explorer,
    s: Scheduler,
    rec: Recorder<'a>,
    /// Simulated time of frame 0, and frames played (warm-up included).
    base: Instant,
    tick: u64,
    frames: Vec<Frame>,
    checkpoints: Vec<(usize, u64)>,
    /// Counts view changes, so cached references know when they're stale.
    epoch: u64,
    reference: Option<Reference>,
    last_snapshot: Option<Vec<u8>>,
    /// Measuring quality (not during the warm-up).
    measuring: bool,
}

impl Player<'_> {
    /// Apply `input`, then let the scheduler do one frame's work.
    fn frame(&mut self, moving: bool, settling: bool, input: impl FnOnce(&mut Explorer)) -> Result<Frame> {
        input(&mut self.e);
        let now = self.base + Duration::from_secs_f64(self.tick as f64 / FPS);
        let start = Instant::now();
        let busy = self.s.step_at(now, &mut self.e, &mut self.rec)?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.tick += 1;
        if moving {
            self.epoch += 1;
        }
        let log = std::mem::take(&mut self.rec.log);
        let changed = log.pixel_draws + log.reprojections > 0;
        let colors = self.e.colors();

        let mut f = Frame {
            t: self.frames.len() as f64 / FPS,
            moving,
            settling,
            wall_ms,
            gpu_ms: log.gpu_ms,
            samples: log.samples,
            pixel_draws: log.pixel_draws,
            stats_draws: log.stats_draws,
            reprojections: log.reprojections,
            max_iter: self.rec.params.as_ref().map_or(0, |p| p.max_iter),
            busy,
            ..Default::default()
        };

        if let Some(h) = &mut self.rec.hash {
            let pixels = if changed { Some(self.rec.inner.snapshot(&colors)?) } else { None };
            h.u64(busy as u64);
            h.u64(self.e.iter_limit as u64);
            if let Some(px) = pixels {
                h.bytes(&px);
            }
        }

        if self.cfg.metrics && self.measuring {
            let shown = if changed || self.last_snapshot.is_none() {
                self.rec.inner.snapshot(&colors)?
            } else {
                self.last_snapshot.clone().unwrap()
            };
            f.speckle = Some(speckle(&shown, self.cfg.width as usize));
            if !moving {
                f.popping = Some(match (&self.last_snapshot, changed) {
                    (Some(last), true) => differing(last, &shown),
                    _ => 0.0,
                });
            }
            let n = self.frames.len();
            if !moving || (self.cfg.error_every > 0 && n.is_multiple_of(self.cfg.error_every)) {
                let limit = self.rec.params.as_ref().expect("a frame was drawn").max_iter;
                self.update_reference()?;
                let deep = &self.reference.as_ref().unwrap().deep;
                let e = error_by_cause(&shown, deep, self.s.tiles(), self.cfg.width, limit);
                f.unfinished = Some(e.unfinished);
                f.truncation = Some(e.truncation);
                f.lowered = Some(e.lowered);
                f.error = Some(e.unfinished + e.truncation);
            }
            self.last_snapshot = Some(shown);
        }
        Ok(f)
    }

    /// Make `reference` a deep render of the current view: TRUNCATION_FACTOR
    /// times the frame's limit.
    fn update_reference(&mut self) -> Result<()> {
        let p = self.rec.params.as_ref().expect("a frame was drawn");
        let key = (self.epoch, p.max_iter);
        if self.reference.as_ref().is_none_or(|r| r.key != key) {
            let mut x = Explorer::new(self.e.width, self.e.height);
            x.center = self.e.center.clone();
            x.scale = self.e.scale;
            x.precision_mode = self.e.precision_mode;
            x.series_skip = self.e.series_skip;
            x.set_iter_limit(p.max_iter.saturating_mul(TRUNCATION_FACTOR).min(MAX_ITER));
            let (deep, _) = self.rec.inner.capture(&x.frame_params(), 1, &self.e.colors())?;
            self.reference = Some(Reference { key, deep });
        }
        Ok(())
    }

    fn push(&mut self, f: Frame) {
        self.frames.push(f);
        if self.frames.len().is_multiple_of(CHECKPOINT_FRAMES) {
            self.checkpoint();
        }
    }

    fn checkpoint(&mut self) {
        if let Some(h) = self.rec.hash {
            self.checkpoints.push((self.frames.len(), h.0));
        }
    }

    fn play(&mut self, action: &Action) -> Result<()> {
        let frames = |seconds: f64| (seconds * FPS).round() as usize;
        match action {
            Action::Zoom { at, per_second, seconds } => {
                let factor = per_second.powf(1.0 / FPS);
                for _ in 0..frames(*seconds) {
                    let f = self.frame(true, false, |e| {
                        e.zoom_at(at.0 * e.width as f64, at.1 * e.height as f64, factor)
                    })?;
                    self.push(f);
                }
            }
            Action::Pan { per_second, seconds } => {
                for _ in 0..frames(*seconds) {
                    let f = self.frame(true, false, |e| {
                        e.pan_pixels(per_second.0 * e.width as f64 / FPS, per_second.1 * e.height as f64 / FPS)
                    })?;
                    self.push(f);
                }
            }
            Action::Fly { to, speed } => {
                let mut flight = Flight::new(self.e.place(), to.clone(), *speed);
                loop {
                    let (place, arrived) = flight.advance(1.0 / FPS);
                    let f = self.frame(true, false, |e| e.go_to(&place))?;
                    self.push(f);
                    if arrived {
                        break;
                    }
                }
            }
            Action::Jump(place) => {
                let f = self.frame(true, false, |e| e.go_to(place))?;
                self.push(f);
            }
            Action::Still(seconds) => {
                for _ in 0..frames(*seconds) {
                    let f = self.frame(false, false, |_| {})?;
                    self.push(f);
                }
            }
        }
        Ok(())
    }

    /// Hold still for at least SETTLE_MIN_SECONDS and until idle, or `max`.
    fn settle(&mut self, max: f64) -> Result<()> {
        for i in 0.. {
            let f = self.frame(false, true, |_| {})?;
            let busy = f.busy;
            self.push(f);
            let t = i as f64 / FPS;
            if t >= max || (t >= SETTLE_MIN_SECONDS && !busy) {
                return Ok(());
            }
        }
        unreachable!()
    }
}

/// Play `scenario` on `renderer` (which is resized to the config's size).
pub fn run(scenario: &Scenario, cfg: &Config, renderer: &mut dyn Renderer) -> Result<Run> {
    renderer.resize(cfg.width, cfg.height, 1.0);
    let mut e = Explorer::new(cfg.width, cfg.height);
    e.color_cycling = false;
    e.background_references = !cfg.deterministic;
    let mut p = Player {
        cfg,
        e,
        s: Scheduler::with_policy(Policy::new(cfg.settings.clone())),
        rec: Recorder {
            inner: renderer,
            model_cost: cfg.deterministic.then_some(MODEL_MS_PER_SAMPLE_ITER),
            log: FrameLog::default(),
            hash: cfg.fingerprint.then(Fnv::new),
            params: None,
        },
        base: Instant::now(),
        tick: 0,
        frames: vec![],
        checkpoints: vec![],
        epoch: 0,
        reference: None,
        last_snapshot: None,
        measuring: false,
    };

    // Start from a finished render of the starting place.
    p.e.go_to(&scenario.start);
    let mut warmup = 0;
    while p.frame(false, false, |_| {})?.busy {
        warmup += 1;
        if warmup > 1_000_000 {
            return Err(format!("{}: never settled at the start", scenario.name));
        }
    }
    p.measuring = true;

    for action in &scenario.script {
        p.play(action)?;
    }
    p.settle(scenario.settle_max)?;
    if p.checkpoints.last().is_none_or(|c| c.0 != p.frames.len()) {
        p.checkpoint();
    }
    Ok(Run {
        name: scenario.name,
        frames: p.frames,
        checkpoints: p.checkpoints,
        warmup_frames: warmup,
        final_image: p.last_snapshot,
        final_deep: p.reference.map(|r| r.deep),
    })
}

/// Fraction of pixels whose color differs noticeably (any channel by more
/// than 16) between two RGBA images.
pub fn differing(a: &[u8], b: &[u8]) -> f64 {
    let bad = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| (0..3).any(|i| x[i].abs_diff(y[i]) > 16)).count();
    bad as f64 / (a.len() / 4).max(1) as f64
}

/// Error against a deep reference, split by the cause each wrong pixel's
/// tile suggests (see the module docs). Fractions of all pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorByCause {
    pub unfinished: f64,
    pub truncation: f64,
    /// The part of `truncation` in tiles computed below `frame_limit`.
    pub lowered: f64,
}

/// Compare `shown` with `deep` (RGBA, `width` wide), attributing each
/// differing pixel by its canvas tile (row-major [`TileState`]s).
pub fn error_by_cause(shown: &[u8], deep: &[u8], tiles: &[TileState], width: u32, frame_limit: u32) -> ErrorByCause {
    let cols = width.div_ceil(TILE) as usize;
    let (mut unfinished, mut truncated, mut lowered) = (0u64, 0u64, 0u64);
    for (i, (a, b)) in shown.chunks(4).zip(deep.chunks(4)).enumerate() {
        if !(0..3).any(|c| a[c].abs_diff(b[c]) > 16) {
            continue;
        }
        let (x, y) = (i % width as usize, i / width as usize);
        let t = &tiles[(y / TILE as usize) * cols + x / TILE as usize];
        if !t.exact {
            unfinished += 1;
        } else {
            truncated += 1;
            lowered += (t.limit < frame_limit) as u64;
        }
    }
    let n = (shown.len() / 4).max(1) as f64;
    ErrorByCause { unfinished: unfinished as f64 / n, truncation: truncated as f64 / n, lowered: lowered as f64 / n }
}

/// Fraction of horizontally or vertically adjacent pixel pairs whose colors
/// differ sharply (any channel by more than 96): isolated pixels that don't
/// match their neighbors read as noise rather than detail.
pub fn speckle(rgba: &[u8], width: usize) -> f64 {
    let px: Vec<&[u8]> = rgba.chunks(4).collect();
    let height = px.len() / width.max(1);
    let sharp = |a: &[u8], b: &[u8]| (0..3).any(|i| a[i].abs_diff(b[i]) > 96);
    let (mut pairs, mut rough) = (0u64, 0u64);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if x + 1 < width {
                pairs += 1;
                rough += sharp(px[i], px[i + 1]) as u64;
            }
            if y + 1 < height {
                pairs += 1;
                rough += sharp(px[i], px[i + width]) as u64;
            }
        }
    }
    rough as f64 / pairs.max(1) as f64
}

/// The `q`-quantile (0..=1) of `values` (0 if empty).
fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v.get(((v.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0)
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

/// A scenario's figures.
#[derive(Clone, Debug)]
pub struct Summary {
    pub name: &'static str,
    pub frames: usize,
    pub seconds: f64,
    /// Wall time per frame (ms): median, 90th and 99th percentile, max.
    pub wall: [f64; 4],
    /// Frames over FRAME_MS, as a fraction.
    pub over_budget: f64,
    /// Mean squared overrun past FRAME_MS per frame (ms²).
    pub jank: f64,
    /// GPU time per frame (ms): median and 90th percentile.
    pub gpu: [f64; 2],
    pub error_moving: Option<f64>,
    pub error_still: Option<f64>,
    pub truncation_moving: Option<f64>,
    /// Truncation of the final image.
    pub truncation_final: Option<f64>,
    /// The part of `truncation_final` in tiles given lower limits.
    pub lowered_final: Option<f64>,
    pub speckle_moving: Option<f64>,
    pub speckle_final: Option<f64>,
    /// Sum of per-frame popping over still frames: how many times, on
    /// average, each pixel changed after motion stopped.
    pub popping: Option<f64>,
    /// Seconds into the final hold when the scheduler went idle (None: not
    /// within the scenario's settle_max).
    pub idle_after: Option<f64>,
    /// Seconds into the final hold after which every frame changed fewer
    /// than QUIET_FRACTION of the pixels.
    pub quiet_after: Option<f64>,
    pub final_iters: u32,
}

impl Run {
    pub fn summary(&self) -> Summary {
        let f = &self.frames;
        let wall: Vec<f64> = f.iter().map(|f| f.wall_ms).collect();
        let gpu: Vec<f64> = f.iter().map(|f| f.gpu_ms).collect();
        let settle_start = f.iter().position(|f| f.settling).unwrap_or(f.len());
        let hold = &f[settle_start..];
        let t0 = hold.first().map_or(0.0, |f| f.t);
        let idle_after = hold.iter().find(|f| !f.busy).map(|f| f.t - t0);
        let quiet_after = hold.iter().any(|f| f.popping.is_some()).then(|| {
            hold.iter().rposition(|f| f.popping.unwrap_or(0.0) >= QUIET_FRACTION).map_or(0.0, |i| (i + 1) as f64 / FPS)
        });
        let pick = |moving: bool, m: fn(&Frame) -> Option<f64>| mean(f.iter().filter(|f| f.moving == moving).filter_map(m));
        Summary {
            name: self.name,
            frames: f.len(),
            seconds: f.len() as f64 / FPS,
            wall: [quantile(&wall, 0.5), quantile(&wall, 0.9), quantile(&wall, 0.99), quantile(&wall, 1.0)],
            over_budget: wall.iter().filter(|&&w| w > FRAME_MS).count() as f64 / wall.len().max(1) as f64,
            jank: mean(wall.iter().map(|w| (w - FRAME_MS).max(0.0).powi(2))).unwrap_or(0.0),
            gpu: [quantile(&gpu, 0.5), quantile(&gpu, 0.9)],
            error_moving: pick(true, |f| f.error),
            error_still: pick(false, |f| f.error),
            truncation_moving: pick(true, |f| f.truncation),
            truncation_final: f.last().and_then(|f| f.truncation),
            lowered_final: f.last().and_then(|f| f.lowered),
            speckle_moving: pick(true, |f| f.speckle),
            speckle_final: f.last().and_then(|f| f.speckle),
            popping: f.iter().any(|f| f.popping.is_some()).then(|| f.iter().filter_map(|f| f.popping).sum()),
            idle_after,
            quiet_after,
            final_iters: f.last().map_or(0, |f| f.max_iter),
        }
    }

    /// Fingerprint lines: `<scenario> <frame> <hash>` per checkpoint.
    pub fn fingerprint(&self) -> String {
        self.checkpoints.iter().map(|(n, h)| format!("{} {n} {h:016x}\n", self.name)).collect()
    }

    /// Per-frame figures as CSV, for plotting.
    pub fn csv(&self) -> String {
        let opt = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.6}"));
        let mut out = String::from(
            "t,moving,settling,wall_ms,gpu_ms,samples,pixel_draws,stats_draws,reprojections,max_iter,busy,error,unfinished,truncation,lowered,speckle,popping\n",
        );
        for f in &self.frames {
            out += &format!(
                "{:.4},{},{},{:.3},{:.3},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                f.t,
                f.moving as u8,
                f.settling as u8,
                f.wall_ms,
                f.gpu_ms,
                f.samples,
                f.pixel_draws,
                f.stats_draws,
                f.reprojections,
                f.max_iter,
                f.busy as u8,
                opt(f.error),
                opt(f.unfinished),
                opt(f.truncation),
                opt(f.lowered),
                opt(f.speckle),
                opt(f.popping),
            );
        }
        out
    }
}

impl Summary {
    pub const HEADER: &str = "scenario     frames  wall ms p50/p90/p99/max   >8.3ms  jank ms²  gpu p50/p90  \
                              error mov/still  trunc mov/final  lowered  speckle mov/final  popping  idle s  quiet s  iters";

    pub fn row(&self) -> String {
        let pct = |v: Option<f64>| v.map_or("    -".to_string(), |v| format!("{:5.2}", v * 100.0));
        let secs = |v: Option<f64>| v.map_or("   -".to_string(), |v| format!("{v:4.2}"));
        format!(
            "{:<11} {:>7}  {:5.1}/{:5.1}/{:5.1}/{:6.1}  {:5.1}%  {:8.1}  {:5.1}/{:5.1}  {}/{}%      {}/{}%   {}%      {}/{}%      {}  {:>6}  {:>7}  {}",
            self.name,
            self.frames,
            self.wall[0],
            self.wall[1],
            self.wall[2],
            self.wall[3],
            self.over_budget * 100.0,
            self.jank,
            self.gpu[0],
            self.gpu[1],
            pct(self.error_moving),
            pct(self.error_still),
            pct(self.truncation_moving),
            pct(self.truncation_final),
            pct(self.lowered_final),
            pct(self.speckle_moving),
            pct(self.speckle_final),
            self.popping.map_or("    -".to_string(), |v| format!("{v:5.2}")),
            secs(self.idle_after),
            secs(self.quiet_after),
            self.final_iters,
        )
    }
}

/// Compare fingerprints (concatenated [`Run::fingerprint`]s). Err lists,
/// per scenario, where `actual` first diverges from `expected`.
pub fn compare_fingerprints(expected: &str, actual: &str) -> std::result::Result<(), String> {
    use std::collections::BTreeMap;
    let parse = |text: &str| {
        let mut m: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
            let mut w = line.split_whitespace();
            if let (Some(name), Some(frame), Some(hash)) = (w.next(), w.next(), w.next()) {
                m.entry(name.to_string()).or_default().push((frame.parse().unwrap_or(0), hash.to_string()));
            }
        }
        m
    };
    let (want, got) = (parse(expected), parse(actual));
    let mut problems = vec![];
    for (name, w) in &want {
        let Some(g) = got.get(name) else {
            problems.push(format!("{name}: not run"));
            continue;
        };
        match w.iter().zip(g).position(|(a, b)| a != b) {
            Some(i) => {
                let previous = if i == 0 { 0 } else { w[i - 1].0 };
                if w[i].0 != g[i].0 {
                    let (wl, gl) = (w.last().unwrap().0, g.last().unwrap().0);
                    problems.push(format!("{name}: diverges after frame {previous} ({wl} frames expected, {gl} played)"));
                } else {
                    problems.push(format!("{name}: diverges between frames {previous} and {}", w[i].0));
                }
            }
            None if w.len() != g.len() => problems.push(format!(
                "{name}: {} frames expected, {} played",
                w.last().unwrap().0,
                g.last().unwrap().0
            )),
            None => {}
        }
    }
    for name in got.keys().filter(|n| !want.contains_key(*n)) {
        problems.push(format!("{name}: no recorded fingerprint"));
    }
    if problems.is_empty() { Ok(()) } else { Err(problems.join("\n")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speckle_counts_sharp_neighbor_pairs() {
        // 3x2: one bright pixel among dark ones touches 3 of the 7 pairs.
        let dark = [10, 10, 10, 255];
        let bright = [250, 10, 10, 255];
        let img: Vec<u8> = [dark, bright, dark, dark, dark, dark].concat();
        assert!((speckle(&img, 3) - 3.0 / 7.0).abs() < 1e-12);
        // A smooth gradient has none.
        let smooth: Vec<u8> = (0..6).flat_map(|i| [i * 20, 0, 0, 255]).collect();
        assert_eq!(speckle(&smooth, 3), 0.0);
    }

    #[test]
    fn differing_ignores_small_changes_and_alpha() {
        let a = [100, 100, 100, 255, 0, 0, 0, 255];
        let b = [110, 90, 116, 0, 0, 40, 0, 255];
        assert_eq!(differing(&a, &b), 0.5);
    }

    #[test]
    fn errors_are_attributed_by_tile_state() {
        // Two tiles side by side, one pixel row: 32 x 1.
        let width = 2 * TILE;
        let px = |v: u8| [v, v, v, 255];
        let deep: Vec<u8> = (0..width).flat_map(|_| px(100)).collect();
        let mut shown = deep.clone();
        // 4 wrong pixels in the left tile, 2 in the right; one off by too
        // little to count.
        for x in [0, 1, 2, 3, 20, 21] {
            shown[x * 4..x * 4 + 4].copy_from_slice(&px(200));
        }
        shown[25 * 4..25 * 4 + 4].copy_from_slice(&px(110));
        let tile = |x, exact, limit| TileState { exact, limit, res: 1.0, ..TileState::missing([x, 0]) };
        let n = width as f64;

        let unfinished = [tile(0, false, 1000), tile(TILE, true, 1000)];
        let e = error_by_cause(&shown, &deep, &unfinished, width, 1000);
        assert_eq!(e, ErrorByCause { unfinished: 4.0 / n, truncation: 2.0 / n, lowered: 0.0 });

        let lowered = [tile(0, true, 1000), tile(TILE, true, 500)];
        let e = error_by_cause(&shown, &deep, &lowered, width, 1000);
        assert_eq!(e, ErrorByCause { unfinished: 0.0, truncation: 6.0 / n, lowered: 2.0 / n });
    }

    #[test]
    fn quantiles_and_means() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(quantile(&v, 0.5), 3.0);
        assert_eq!(quantile(&v, 1.0), 5.0);
        assert_eq!(quantile(&[], 0.5), 0.0);
        assert_eq!(mean([1.0, 2.0].into_iter()), Some(1.5));
        assert_eq!(mean(std::iter::empty()), None);
    }

    #[test]
    fn fingerprint_comparison_reports_where_runs_diverge() {
        let expected = "zoom 60 aa\nzoom 120 bb\nzoom 150 cc\npan 60 dd\n";
        assert!(compare_fingerprints(expected, expected).is_ok());
        let changed = "zoom 60 aa\nzoom 120 xx\nzoom 150 cc\npan 60 dd\n";
        let err = compare_fingerprints(expected, changed).unwrap_err();
        assert_eq!(err, "zoom: diverges between frames 60 and 120");
        let shorter = "zoom 60 aa\nzoom 120 bb\nzoom 140 cc\npan 60 dd\n";
        let err = compare_fingerprints(expected, shorter).unwrap_err();
        assert!(err.contains("after frame 120") && err.contains("150 frames expected, 140 played"), "{err}");
        let missing = compare_fingerprints(expected, "zoom 60 aa\nzoom 120 bb\nzoom 150 cc\n").unwrap_err();
        assert_eq!(missing, "pan: not run");
    }
}
