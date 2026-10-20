//! Platform-independent explorer state: navigation, colors, and the
//! perturbation reference orbit's lifecycle. Input that affects iteration
//! depth is queued for the policy (see `policy.rs`).
//!
//! Nothing in here knows about Metal, AppKit, or even winit. Input is fed in as
//! plain numbers (physical pixels), and the output is a [`FrameParams`] that any
//! renderer backend can consume.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use crate::bigfixed::{Fixed, frac_limbs_for_scale};
use crate::perturb::{ReferenceOrbit, compute_reference};
use crate::policy::{Input, MAX_ITER, MIN_ITER, estimate_iter};

/// Which arithmetic the GPU kernel should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Plain 32-bit floats. Fast; pixelates around 1e-5 zoom.
    F32,
    /// Double-float (two f32s, ~48-bit mantissa). ~3-5x slower; good to ~1e-12.
    DoubleFloat,
    /// Deltas against a high-precision CPU reference orbit; f32 where possible,
    /// floatexp where values are too small for f32. Good to ~1e300.
    Perturbation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    Auto,
    Forced(Precision),
}

/// A location in the plane and how much of it is visible: everything needed
/// to return to a view, independent of window size (bookmarks, flights).
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub center: (Fixed, Fixed),
    /// Height of the view in complex-plane units.
    pub span: f64,
}

impl Place {
    /// A place from exact decimal coordinates, keeping every digit given.
    pub fn parse(x: &str, y: &str, span: f64) -> Option<Place> {
        Some(Place { center: parse_center(x, y)?, span })
    }
}

/// Decimal strings to a center, with enough precision for every digit
/// (~3.3 bits each) and at least 512 bits.
fn parse_center(x: &str, y: &str) -> Option<(Fixed, Fixed)> {
    let frac = (x.len().max(y.len()) * 4).div_ceil(64).max(8);
    Some((Fixed::parse(x, frac)?, Fixed::parse(y, frac)?))
}

/// How iteration counts become colors. Applied each time the canvas is shown,
/// so changing it never requires recomputing anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colors {
    pub palette: u32,
    /// Palette cycles per 50 iterations.
    pub density: f32,
    /// Palette phase, in cycles.
    pub offset: f32,
}

/// Everything a renderer needs to compute one frame (colors are separate:
/// see [`Colors`]).
#[derive(Clone)]
pub struct FrameParams {
    /// Center rounded to f64 (all the direct f32/df64 paths can use).
    pub center: (f64, f64),
    /// Complex-plane units per physical pixel.
    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub max_iter: u32,
    pub precision: Precision,
    /// Reference orbit (Perturbation only). Renderers can use `Arc::ptr_eq`
    /// to skip re-uploading an unchanged orbit.
    pub reference: Option<Arc<ReferenceOrbit>>,
    /// View center minus the reference point (Perturbation only). The
    /// reference need not be at the center, which lets it be reused while
    /// panning.
    pub ref_offset: (f64, f64),
    /// Series approximation: every pixel starts at iteration `series_n` with
    /// dz = A * dc, A given as complex floatexp (mantissa, exponent).
    pub series_n: u32,
    pub series_a: ([f32; 2], i32),
}

pub const PALETTE_COUNT: u32 = 6;
pub const PALETTE_NAMES: [&str; PALETTE_COUNT as usize] =
    ["Classic", "Fire", "Ocean", "Neon", "Pastel", "Grayscale"];

const HOME_CENTER: (f64, f64) = (-0.75, 0.0);
/// Complex-plane height visible at zoom 1.
const HOME_SPAN: f64 = 2.6;
/// The scale (and the reference orbit's conversion through f64) is f64.
const MIN_SCALE: f64 = 1e-300;
/// Switch away from plain f32 once a pixel is smaller than this (Auto mode).
const F32_LIMIT: f64 = 2e-6;

/// Initial color cycling speed, in palette cycles per second: slow enough to
/// read as drifting rather than flashing.
pub const DEFAULT_COLOR_SPEED: f32 = 0.1;

/// Every palette repeats after this many cycles, so the color offset can wrap
/// here seamlessly. (Not 1: Fire's channel frequencies are 1, 0.7 and 0.4,
/// which only all complete whole waves after 10 cycles. Palettes in the
/// shader must keep frequency × 10 whole.)
pub const PALETTE_PERIOD: f32 = 10.0;

/// A reference may be used while within this many view sizes of the center;
/// beyond it, f32 deltas lose too much precision and we recompute in place.
const REF_USABLE_VIEWS: f64 = 4.0;
/// Beyond this many view sizes, a better reference is computed in the background.
const REF_IDEAL_VIEWS: f64 = 0.5;

/// A computed reference orbit and the point it belongs to.
struct RefState {
    c: (Fixed, Fixed),
    /// Fraction limbs it was computed with (must cover the current scale).
    limbs: usize,
    orbit: Arc<ReferenceOrbit>,
}

impl RefState {
    fn compute(c: (Fixed, Fixed), limbs: usize, max_iter: u32) -> Self {
        // Any point near the view works as a reference, so drop extra
        // precision the center may carry: it only slows the orbit down.
        let c = (c.0.with_frac_limbs(limbs), c.1.with_frac_limbs(limbs));
        let orbit = Arc::new(compute_reference(&c.0, &c.1, max_iter));
        Self { c, limbs, orbit }
    }
}

pub struct Explorer {
    pub center: (Fixed, Fixed),
    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub precision_mode: PrecisionMode,
    pub palette: u32,
    pub color_density: f32,
    /// Palette phase in cycles (see [`Colors::offset`]).
    pub color_offset: f32,
    /// Advance `color_offset` over time (see [`Explorer::advance_colors`]).
    pub color_cycling: bool,
    /// Cycling speed, palette cycles per second.
    pub color_speed: f32,
    /// The iteration limit frames are computed for: the reference orbit and
    /// series skip cover it. Set by the scheduler from its policy (see
    /// `policy.rs`), or directly when rendering without one.
    pub iter_limit: u32,
    /// Use the series approximation to skip early iterations (Perturbation).
    pub series_skip: bool,
    /// Compute better-centered reference orbits on a background thread
    /// (false: synchronously, adopted on the next frame as if the thread had
    /// just finished, which makes runs reproducible).
    pub background_references: bool,
    /// Input that affects iteration depth, for the policy (see
    /// [`Explorer::take_inputs`]).
    inputs: Vec<Input>,
    drag_anchor: Option<(f64, f64)>,
    pub cursor: (f64, f64),
    /// Sub-pixel pan not yet applied: pans move the center in whole pixels so
    /// the previous frame can be reused exactly (see progressive.rs).
    pan_remainder: (f64, f64),
    reference: Option<RefState>,
    /// A better reference being computed on a background thread.
    pending: Option<Receiver<RefState>>,
    /// Iterations skipped by the series approximation in the last frame (HUD).
    last_series_n: u32,
}

impl Explorer {
    pub fn new(width: u32, height: u32) -> Self {
        let mut e = Self {
            center: (Fixed::zero(2), Fixed::zero(2)),
            scale: 1.0,
            width: width.max(1),
            height: height.max(1),
            precision_mode: PrecisionMode::Auto,
            palette: 0,
            color_density: 1.0,
            color_offset: 0.0,
            // On by default: it only re-colors already computed values.
            color_cycling: true,
            color_speed: DEFAULT_COLOR_SPEED,
            iter_limit: MIN_ITER,
            series_skip: true,
            background_references: true,
            inputs: vec![],
            drag_anchor: None,
            cursor: (0.0, 0.0),
            pan_remainder: (0.0, 0.0),
            reference: None,
            pending: None,
            last_series_n: 0,
        };
        e.reset_view();
        e
    }

    fn home_scale(&self) -> f64 {
        HOME_SPAN / self.height as f64
    }

    pub fn reset_view(&mut self) {
        self.scale = self.home_scale();
        let frac = frac_limbs_for_scale(self.scale);
        self.center = (Fixed::from_f64(HOME_CENTER.0, frac), Fixed::from_f64(HOME_CENTER.1, frac));
        self.reset_iter_estimate();
    }

    /// Automatic iterations, starting from the estimate for the current depth.
    pub fn reset_iter_estimate(&mut self) {
        self.iter_limit = estimate_iter(self.zoom());
        self.inputs.push(Input::Reset { zoom: self.zoom() });
    }

    /// Input for the policy since the last call, in order.
    pub fn take_inputs(&mut self) -> Vec<Input> {
        std::mem::take(&mut self.inputs)
    }

    /// The current view as a window-size-independent [`Place`].
    pub fn place(&self) -> Place {
        Place { center: self.center.clone(), span: self.scale * self.height as f64 }
    }

    /// The starting view.
    pub fn home_place(&self) -> Place {
        let frac = frac_limbs_for_scale(self.home_scale());
        Place {
            center: (Fixed::from_f64(HOME_CENTER.0, frac), Fixed::from_f64(HOME_CENTER.1, frac)),
            span: HOME_SPAN,
        }
    }

    /// Show `place`. Only the view changes (not colors or iteration mode);
    /// the policy sees the depth change as a zoom.
    pub fn go_to(&mut self, place: &Place) {
        let new_scale = (place.span / self.height as f64).clamp(MIN_SCALE, self.home_scale() * 4.0);
        let zoom_factor = self.scale / new_scale;
        self.scale = new_scale;
        let frac = |c: &Fixed| frac_limbs_for_scale(new_scale).max(c.frac_limbs());
        self.center = (
            place.center.0.with_frac_limbs(frac(&place.center.0)),
            place.center.1.with_frac_limbs(frac(&place.center.1)),
        );
        self.pan_remainder = (0.0, 0.0);
        self.inputs.push(Input::Zoom(zoom_factor));
    }

    /// Set the center from decimal strings (arbitrary precision).
    pub fn set_center_str(&mut self, x: &str, y: &str) -> Option<()> {
        self.center = parse_center(x, y)?;
        Some(())
    }

    /// Keeps the visible region stable (by height) when the window resizes.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        self.scale *= self.height as f64 / height as f64;
        self.width = width;
        self.height = height;
    }

    /// Magnification relative to the home view.
    pub fn zoom(&self) -> f64 {
        self.home_scale() / self.scale
    }

    pub fn precision(&self) -> Precision {
        match self.precision_mode {
            PrecisionMode::Forced(p) => p,
            PrecisionMode::Auto if self.scale < F32_LIMIT => Precision::Perturbation,
            PrecisionMode::Auto => Precision::F32,
        }
    }

    pub fn cycle_precision_mode(&mut self) {
        use Precision::*;
        self.precision_mode = match self.precision_mode {
            PrecisionMode::Auto => PrecisionMode::Forced(F32),
            PrecisionMode::Forced(F32) => PrecisionMode::Forced(DoubleFloat),
            PrecisionMode::Forced(DoubleFloat) => PrecisionMode::Forced(Perturbation),
            PrecisionMode::Forced(Perturbation) => PrecisionMode::Auto,
        };
    }

    /// The detail keys: scale the iteration limit (turns automatic off).
    pub fn scale_detail(&mut self, factor: f64) {
        self.inputs.push(Input::ScaleDetail(factor));
    }

    pub fn toggle_auto_iter(&mut self) {
        self.inputs.push(Input::ToggleAuto);
    }

    /// A fixed iteration limit (turns automatic off), effective at once.
    pub fn set_iter_limit(&mut self, n: u32) {
        self.iter_limit = n.clamp(32, MAX_ITER);
        self.inputs.push(Input::SetLimit(n));
    }

    /// Move the center by a complex-plane offset, growing precision as needed.
    fn shift_center(&mut self, dx: f64, dy: f64) {
        let frac = frac_limbs_for_scale(self.scale).max(self.center.0.frac_limbs());
        self.center = (
            self.center.0.with_frac_limbs(frac).add_f64(dx),
            self.center.1.with_frac_limbs(frac).add_f64(dy),
        );
    }

    /// Zoom by `factor` (>1 zooms in) keeping the point under (px, py) fixed.
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        // Offset of the anchor from the center, in complex units (small: f64 is fine).
        let ox = (px - self.width as f64 / 2.0) * self.scale;
        let oy = -(py - self.height as f64 / 2.0) * self.scale;
        let new_scale = (self.scale / factor).clamp(MIN_SCALE, self.home_scale() * 4.0);
        let k = new_scale / self.scale;
        self.scale = new_scale;
        self.pan_remainder = (0.0, 0.0);
        self.inputs.push(Input::Zoom(k.recip()));
        self.shift_center(ox * (1.0 - k), oy * (1.0 - k));
    }

    pub fn zoom_at_cursor(&mut self, factor: f64) {
        self.zoom_at(self.cursor.0, self.cursor.1, factor);
    }

    pub fn zoom_at_center(&mut self, factor: f64) {
        self.zoom_at(self.width as f64 / 2.0, self.height as f64 / 2.0, factor);
    }

    /// Move the content by (dx, dy) physical pixels (content follows the finger).
    pub fn pan_pixels(&mut self, dx: f64, dy: f64) {
        let (rx, ry) = (self.pan_remainder.0 + dx, self.pan_remainder.1 + dy);
        let (wx, wy) = (rx.round(), ry.round());
        self.pan_remainder = (rx - wx, ry - wy);
        if wx != 0.0 || wy != 0.0 {
            self.shift_center(-wx * self.scale, wy * self.scale);
        }
    }

    /// Pan by a fraction of the view size (keyboard arrows).
    pub fn pan_fraction(&mut self, fx: f64, fy: f64) {
        self.pan_pixels(fx * self.width as f64, fy * self.height as f64);
    }

    pub fn begin_drag(&mut self) {
        self.drag_anchor = Some(self.cursor);
    }

    pub fn end_drag(&mut self) {
        self.drag_anchor = None;
    }

    /// Returns true if the view changed (i.e. we were dragging).
    pub fn cursor_moved(&mut self, x: f64, y: f64) -> bool {
        self.cursor = (x, y);
        if let Some((ax, ay)) = self.drag_anchor {
            self.pan_pixels(x - ax, y - ay);
            self.drag_anchor = Some((x, y));
            true
        } else {
            false
        }
    }

    pub fn cycle_palette(&mut self, forward: bool) {
        self.palette = if forward {
            (self.palette + 1) % PALETTE_COUNT
        } else {
            (self.palette + PALETTE_COUNT - 1) % PALETTE_COUNT
        };
    }

    /// Offset of the view center from `r`'s point, and that distance in view sizes.
    fn ref_offset(&self, r: &RefState) -> ((f64, f64), f64) {
        let off = ((&self.center.0 - &r.c.0).to_f64(), (&self.center.1 - &r.c.1).to_f64());
        let view = self.width.max(self.height) as f64 * self.scale;
        (off, off.0.abs().max(off.1.abs()) / view)
    }

    /// The reference orbit to use for the current view.
    ///
    /// With rebasing, any reference near the view is correct, so an existing
    /// one is kept while panning and zooming. It is recomputed synchronously
    /// when unusable; otherwise a better-centered one is computed on a
    /// background thread and adopted when ready.
    ///
    /// Unusable means too imprecise for the current scale, too far away, or
    /// too short. A reference that *escaped* may end early (pixels near it
    /// escape too, and rebasing handles the rest), but one that was merely
    /// cut off at an older iteration limit may not: rebasing at its end
    /// would replace a pixel's tiny dz with the full z in f32, discarding
    /// exactly the bits that distinguish neighboring pixels.
    fn reference_orbit(&mut self) -> (Arc<ReferenceOrbit>, (f64, f64)) {
        if let Some(rx) = &self.pending {
            match rx.try_recv() {
                Ok(r) => {
                    self.reference = Some(r);
                    self.pending = None;
                }
                Err(TryRecvError::Disconnected) => self.pending = None,
                Err(TryRecvError::Empty) => {}
            }
        }

        let limbs = frac_limbs_for_scale(self.scale);
        let limit = self.iter_limit;
        let usable = self.reference.as_ref().is_some_and(|r| {
            let long_enough = r.orbit.escaped || r.orbit.points.len() as u32 > limit;
            r.limbs >= limbs && long_enough && self.ref_offset(r).1 < REF_USABLE_VIEWS
        });
        // Headroom so the next automatic doubling doesn't force a recompute.
        let ref_iters = limit.saturating_mul(2).min(MAX_ITER).max(limit);
        if !usable {
            self.reference = Some(RefState::compute(self.center.clone(), limbs, ref_iters));
            self.pending = None; // whatever it was computing is now stale
        }

        let r = self.reference.as_ref().unwrap();
        let (offset, dist) = self.ref_offset(r);
        if dist > REF_IDEAL_VIEWS && self.pending.is_none() {
            let (tx, rx) = mpsc::channel();
            let (c, n) = (self.center.clone(), ref_iters);
            let job = move || {
                let _ = tx.send(RefState::compute(c, limbs, n));
            };
            if self.background_references {
                std::thread::spawn(job);
            } else {
                job();
            }
            self.pending = Some(rx);
        }
        (r.orbit.clone(), offset)
    }

    pub fn frame_params(&mut self) -> FrameParams {
        let precision = self.precision();
        let (mut series_n, mut series_a) = (0, ([0.0, 0.0], 0));
        let (reference, ref_offset) = if precision == Precision::Perturbation {
            let (orbit, offset) = self.reference_orbit();
            if self.series_skip {
                // Largest |dc| in the view: half its diagonal, plus how far the
                // view center is from the reference.
                let half_diag = 0.5 * (self.width as f64).hypot(self.height as f64) * self.scale;
                let dc_max = half_diag + offset.0.hypot(offset.1);
                let (n, a) = orbit.series_skip(dc_max, self.iter_limit);
                (series_n, series_a) = (n, a.to_gpu());
            }
            (Some(orbit), offset)
        } else {
            (None, (0.0, 0.0))
        };
        self.last_series_n = series_n;
        FrameParams {
            center: (self.center.0.to_f64(), self.center.1.to_f64()),
            scale: self.scale,
            width: self.width,
            height: self.height,
            max_iter: self.iter_limit,
            precision,
            reference,
            ref_offset,
            series_n,
            series_a,
        }
    }

    /// Move the palette phase by `cycles`, kept in [0, PALETTE_PERIOD): every
    /// palette repeats over that range, so wrapping is seamless, and f32
    /// stays precise however long colors cycle.
    pub fn shift_colors(&mut self, cycles: f32) {
        self.color_offset = (self.color_offset + cycles).rem_euclid(PALETTE_PERIOD);
    }

    /// Advance color cycling by `dt` seconds (nothing unless cycling).
    pub fn advance_colors(&mut self, dt: f64) {
        if self.color_cycling {
            self.shift_colors(self.color_speed * dt as f32);
        }
    }

    pub fn scale_color_speed(&mut self, factor: f32) {
        self.color_speed = (self.color_speed * factor).clamp(0.001, 10.0);
    }

    pub fn colors(&self) -> Colors {
        Colors { palette: self.palette, density: self.color_density, offset: self.color_offset }
    }

    /// Center as decimal strings with enough digits to reproduce this view.
    pub fn center_strings(&self) -> (String, String) {
        let digits = ((-self.scale.log10()).ceil() as usize + 3).max(8);
        (self.center.0.to_decimal(digits), self.center.1.to_decimal(digits))
    }

    /// Window title. `iters` describes the iteration policy (see
    /// `Policy::describe`); `progress` is the refinement fraction (None when
    /// done).
    pub fn hud(&self, iters: &str, gpu_ms: Option<f64>, progress: Option<f32>) -> String {
        let mode = match (self.precision_mode, self.precision()) {
            (PrecisionMode::Auto, Precision::F32) => "auto/f32",
            (PrecisionMode::Auto, Precision::DoubleFloat) => "auto/df64",
            (PrecisionMode::Auto, Precision::Perturbation) => "auto/perturb",
            (PrecisionMode::Forced(Precision::F32), _) => "f32",
            (PrecisionMode::Forced(Precision::DoubleFloat), _) => "df64",
            (PrecisionMode::Forced(Precision::Perturbation), _) => "perturb",
        };
        let reference = match (&self.reference, self.precision()) {
            (Some(r), Precision::Perturbation) => format!(
                " (ref {}{} it, {:.0} ms{})",
                r.orbit.points.len() - 1,
                if r.orbit.escaped { " escaped" } else { "" },
                r.orbit.compute_ms,
                match self.last_series_n {
                    0 => String::new(),
                    n => format!(", skip {n}"),
                }
            ),
            _ => String::new(),
        };
        let status = match (progress, gpu_ms) {
            (Some(p), _) => format!("  |  refining {:.0}%", p * 100.0),
            (None, Some(ms)) => format!("  |  {ms:.0} ms"),
            (None, None) => String::new(),
        };
        let cycling = if self.color_cycling { format!(", cycling {:.3}/s", self.color_speed) } else { String::new() };
        // The title bar can't show 35 digits usefully; `c` prints them all.
        let (cx, cy) = (self.center.0.to_f64(), self.center.1.to_f64());
        format!(
            "Mandelbrot  |  {cx:+.15}, {cy:+.15}  |  zoom {:.3e}  |  {iters}  |  {mode}{reference}  |  {}{cycling}{status}",
            self.zoom(),
            PALETTE_NAMES[self.palette as usize],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel_to_complex(e: &Explorer, px: f64, py: f64) -> (Fixed, Fixed) {
        (
            e.center.0.add_f64((px - e.width as f64 / 2.0) * e.scale),
            e.center.1.add_f64(-(py - e.height as f64 / 2.0) * e.scale),
        )
    }

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut e = Explorer::new(800, 600);
        e.zoom_at_center(1e20);
        let before = pixel_to_complex(&e, 123.0, 456.0);
        e.zoom_at(123.0, 456.0, 7.5);
        let after = pixel_to_complex(&e, 123.0, 456.0);
        // Must hold to well below a pixel even at 1e20 zoom.
        assert!((&before.0 - &after.0).to_f64().abs() < e.scale * 1e-3);
        assert!((&before.1 - &after.1).to_f64().abs() < e.scale * 1e-3);
    }

    #[test]
    fn auto_precision_switches_when_deep() {
        let mut e = Explorer::new(800, 600);
        assert_eq!(e.precision(), Precision::F32);
        e.zoom_at_center(1e6);
        assert_eq!(e.precision(), Precision::Perturbation);
    }

    #[test]
    fn place_round_trips_across_window_sizes() {
        let mut e = Explorer::new(800, 600);
        e.set_center_str("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139");
        e.zoom_at_center(1e25);
        let saved = e.place();
        // Same place in a window twice as tall: same region, half the scale.
        let mut tall = Explorer::new(800, 1200);
        tall.go_to(&saved);
        assert_eq!(tall.place().span, saved.span);
        assert!((tall.scale / e.scale - 0.5).abs() < 1e-12);
        // (Fixed's == compares representation too: compare values.)
        assert_eq!((&tall.center.0 - &saved.center.0).to_f64(), 0.0);
        assert_eq!((&tall.center.1 - &saved.center.1).to_f64(), 0.0);
        // Coming back home.
        tall.go_to(&tall.home_place());
        assert_eq!(tall.zoom(), 1.0);
    }

    #[test]
    fn go_to_grows_precision_for_deep_places() {
        let mut e = Explorer::new(800, 600);
        let mut deep = e.home_place();
        deep.span = 1e-150;
        e.go_to(&deep);
        assert!(e.center.0.frac_limbs() >= frac_limbs_for_scale(e.scale));
    }

    #[test]
    fn depth_input_is_queued_for_the_policy_in_order() {
        let mut e = Explorer::new(800, 600);
        assert_eq!(e.take_inputs(), [Input::Reset { zoom: 1.0 }]);
        e.zoom_at(10.0, 20.0, 4.0);
        e.scale_detail(1.5);
        let mut deep = e.home_place();
        deep.span /= 1e6;
        e.go_to(&deep);
        e.set_iter_limit(5000);
        let got = e.take_inputs();
        assert!(matches!(got[0], Input::Zoom(f) if (f - 4.0).abs() < 1e-9), "{got:?}");
        assert_eq!(got[1], Input::ScaleDetail(1.5));
        assert!(matches!(got[2], Input::Zoom(f) if (f - 1e6 / 4.0).abs() < 1e-3), "{got:?}");
        assert_eq!(got[3], Input::SetLimit(5000));
        assert_eq!(e.iter_limit, 5000, "a fixed limit applies at once");
        assert!(e.take_inputs().is_empty());
    }

    #[test]
    fn color_cycling_advances_only_when_on_and_wraps() {
        let mut e = Explorer::new(800, 600);
        assert!(e.color_cycling, "on by default");
        e.color_cycling = false;
        e.advance_colors(10.0);
        assert_eq!(e.color_offset, 0.0, "moved while off");
        e.color_cycling = true;
        e.color_speed = 0.25;
        e.advance_colors(1.0);
        assert!((e.color_offset - 0.25).abs() < 1e-6);
        // Wraps into [0, PALETTE_PERIOD) after many cycles, forward and back.
        e.advance_colors(1000.0);
        assert!((0.0..PALETTE_PERIOD).contains(&e.color_offset), "{}", e.color_offset);
        assert!((e.color_offset - 0.25).abs() < 1e-3, "{}", e.color_offset);
        e.shift_colors(-0.5);
        assert!((e.color_offset - (PALETTE_PERIOD - 0.25)).abs() < 1e-3, "{}", e.color_offset);
        // The offset reaches the colors the renderer uses.
        assert_eq!(e.colors().offset, e.color_offset);
    }

    #[test]
    fn pans_move_in_whole_pixels() {
        let mut e = Explorer::new(800, 600);
        let start = e.center.clone();
        e.pan_pixels(0.4, 0.0);
        assert_eq!(e.center, start, "sub-pixel pan is deferred");
        e.pan_pixels(0.4, 0.0);
        let moved = (&e.center.0 - &start.0).to_f64() / e.scale;
        assert!((moved + 1.0).abs() < 1e-9, "{moved}");
    }

    #[test]
    fn reference_is_reused_while_panning() {
        let mut e = Explorer::new(800, 600);
        e.zoom_at_center(1e8);
        let first = e.frame_params().reference.unwrap();
        e.pan_pixels(50.0, -30.0);
        let p = e.frame_params();
        assert!(Arc::ptr_eq(&first, p.reference.as_ref().unwrap()));
        // The offset is exactly the pan, in complex units.
        assert!((p.ref_offset.0 - (-50.0 * e.scale)).abs() < e.scale * 1e-9);
        assert!((p.ref_offset.1 - (-30.0 * e.scale)).abs() < e.scale * 1e-9);
        // Raising the limit past the reference's length forces a longer one
        // (a cut-off reference can't be rebased past its end).
        let len = first.points.len() as u32;
        e.set_iter_limit(len * 2);
        let longer = e.frame_params().reference.unwrap();
        assert!(longer.escaped || longer.points.len() as u32 > len * 2);
        // Panning many screens away forces a fresh reference.
        e.pan_pixels(10_000.0, 0.0);
        assert!(!Arc::ptr_eq(&longer, e.frame_params().reference.as_ref().unwrap()));
    }
}
