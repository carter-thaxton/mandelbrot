//! Platform-independent explorer state: navigation, iteration policy, colors,
//! and the perturbation reference orbit's lifecycle.
//!
//! Nothing in here knows about Metal, AppKit, or even winit. Input is fed in as
//! plain numbers (physical pixels), and the output is a [`FrameParams`] that any
//! renderer backend can consume.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};

use crate::bigfixed::{Fixed, frac_limbs_for_scale};
use crate::perturb::{ReferenceOrbit, compute_reference};
use crate::render::FrameStats;

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

/// Everything a renderer needs to draw one frame.
#[derive(Clone)]
pub struct FrameParams {
    /// Center rounded to f64 (all the direct f32/df64 paths can use).
    pub center: (f64, f64),
    /// Complex-plane units per physical pixel.
    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub max_iter: u32,
    pub palette: u32,
    pub precision: Precision,
    /// Palette cycles per 50 iterations.
    pub color_density: f32,
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

pub const MIN_ITER: u32 = 256;
/// Automatic iteration decisions need at least this many samples: the 0.1%
/// cut-off test is meaningless on a few thousand (it fired on 8 pixels).
pub const MIN_DECISION_SAMPLES: u64 = 20_000;
pub const MAX_ITER: u32 = 200_000;

/// In auto mode, each zoom by a factor f scales the iteration limit by
/// f^ITERS_PER_ZOOM, so it keeps roughly in step while zooming; GPU
/// statistics then correct it for the actual location. Fitted to the seahorse
/// valley (5.4k needed at 1e10, 62k at 1e30); flatter places just get
/// corrected down.
const ITERS_PER_ZOOM: f64 = 0.04;

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
    pub iter_limit: u32,
    /// Adjust `iter_limit` from GPU statistics (see [`Explorer::observe`]).
    pub auto_iter: bool,
    /// Use the series approximation to skip early iterations (Perturbation).
    pub series_skip: bool,
    /// Zoom-driven change to `iter_limit` not yet applied (applied in >=5%
    /// steps so continuous zooming doesn't change it every frame).
    iter_drift: f64,
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
            iter_limit: MIN_ITER,
            auto_iter: true,
            series_skip: true,
            iter_drift: 1.0,
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
        self.auto_iter = true;
        self.iter_limit = self.estimate_iter();
    }

    /// Set the center from decimal strings (arbitrary precision).
    pub fn set_center_str(&mut self, x: &str, y: &str) -> Option<()> {
        // ~3.3 bits per decimal digit; keep everything that was typed.
        let frac = (x.len().max(y.len()) * 4).div_ceil(64).max(8);
        self.center = (Fixed::parse(x, frac)?, Fixed::parse(y, frac)?);
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

    /// Starting guess for the iteration limit at the current depth (auto mode
    /// then corrects it from what the GPU actually sees).
    pub fn estimate_iter(&self) -> u32 {
        let decades = self.zoom().max(1.0).log10();
        (200.0 + 250.0 * decades).clamp(MIN_ITER as f64, 20_000.0) as u32
    }

    /// Manually scale the iteration limit (turns auto off).
    pub fn scale_iter_limit(&mut self, factor: f64) {
        self.auto_iter = false;
        self.iter_limit = (self.iter_limit as f64 * factor).clamp(32.0, MAX_ITER as f64) as u32;
    }

    pub fn set_iter_limit(&mut self, n: u32) {
        self.auto_iter = false;
        self.iter_limit = n.clamp(32, MAX_ITER);
    }

    /// Automatic iteration limit, driven by GPU statistics from a frame drawn
    /// with the current limit. Returns true if the limit changed.
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
    ///   rules don't oscillate. Only applied when `whole_view` (a strip
    ///   revealed by panning says nothing about the rest of the image).
    pub fn observe(&mut self, s: &FrameStats, whole_view: bool) -> bool {
        if !self.auto_iter || s.samples < MIN_DECISION_SAMPLES {
            return false;
        }
        let l = self.iter_limit;
        let escaped = s.samples.saturating_sub(s.interior as u64);
        let cut_off = s.near_limit >= 4 && s.near_limit as f64 > escaped as f64 * 1e-3;
        let all_black = s.max_escaped == 0 && s.interior > 0;
        let new = if cut_off || all_black {
            l.saturating_mul(2)
        } else if whole_view && s.max_escaped < l / 3 {
            s.max_escaped.saturating_mul(2)
        } else {
            l
        }
        .clamp(MIN_ITER, MAX_ITER);
        let changed = new != l;
        self.iter_limit = new;
        changed
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
        if self.auto_iter {
            self.iter_drift *= k.recip().powf(ITERS_PER_ZOOM);
            if !(1.0 / 1.05..=1.05).contains(&self.iter_drift) {
                let n = self.iter_limit as f64 * self.iter_drift;
                self.iter_limit = n.round().clamp(MIN_ITER as f64, MAX_ITER as f64) as u32;
                self.iter_drift = 1.0;
            }
        }
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
            std::thread::spawn(move || {
                let _ = tx.send(RefState::compute(c, limbs, n));
            });
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
            palette: self.palette,
            precision,
            color_density: self.color_density,
            reference,
            ref_offset,
            series_n,
            series_a,
        }
    }

    /// Center as decimal strings with enough digits to reproduce this view.
    pub fn center_strings(&self) -> (String, String) {
        let digits = ((-self.scale.log10()).ceil() as usize + 3).max(8);
        (self.center.0.to_decimal(digits), self.center.1.to_decimal(digits))
    }

    /// Window title. `progress` is the refinement fraction (None when done).
    pub fn hud(&self, gpu_ms: Option<f64>, progress: Option<f32>) -> String {
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
        let iters = format!("iters {} {}", self.iter_limit, if self.auto_iter { "auto" } else { "manual" });
        let status = match (progress, gpu_ms) {
            (Some(p), _) => format!("  |  refining {:.0}%", p * 100.0),
            (None, Some(ms)) => format!("  |  {ms:.0} ms"),
            (None, None) => String::new(),
        };
        // The title bar can't show 35 digits usefully; `c` prints them all.
        let (cx, cy) = (self.center.0.to_f64(), self.center.1.to_f64());
        format!(
            "Mandelbrot  |  {cx:+.15}, {cy:+.15}  |  zoom {:.3e}  |  {iters}  |  {mode}{reference}  |  {}{status}",
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

    fn stats(samples: u64, max_escaped: u32, near_limit: u32, interior: u32) -> FrameStats {
        FrameStats { gpu_ms: 0.0, samples, max_escaped, near_limit, interior }
    }

    #[test]
    fn auto_iterations_grow_shrink_and_settle() {
        let mut e = Explorer::new(800, 600);
        e.iter_limit = 1000;
        // Lots of samples escaping just under the limit: cut off, so grow.
        assert!(e.observe(&stats(100_000, 999, 5_000, 20_000), false));
        assert_eq!(e.iter_limit, 2000);
        // Slowest escape far below the limit, but only part of the view: keep.
        assert!(!e.observe(&stats(100_000, 300, 0, 20_000), false));
        // Mostly black, and most of what escaped did so near the limit: the
        // view is under-iterated even though few samples are near the limit.
        e.iter_limit = 2000;
        assert!(e.observe(&stats(100_000, 1990, 50, 99_900), true));
        assert_eq!(e.iter_limit, 4000);
        e.iter_limit = 2000;
        // Same over the whole view: shrink to 2x it.
        assert!(e.observe(&stats(100_000, 300, 0, 20_000), true));
        assert_eq!(e.iter_limit, 600);
        // Same frame again: stable.
        assert!(!e.observe(&stats(100_000, 300, 0, 20_000), true));
        // Manual mode ignores stats.
        e.scale_iter_limit(2.0);
        assert!(!e.observe(&stats(100_000, 1, 0, 0), true));
    }

    #[test]
    fn iterations_track_zoom_depth() {
        let mut e = Explorer::new(800, 600);
        e.iter_limit = 1000;
        // 1e5 zoom in many small steps, like a pinch.
        for _ in 0..100 {
            e.zoom_at_center(1e5f64.powf(0.01));
        }
        let expected = 1000.0 * 1e5f64.powf(ITERS_PER_ZOOM);
        let got = e.iter_limit as f64;
        assert!((got / expected - 1.0).abs() < 0.06, "{got} vs {expected}");
        // Manual mode leaves it alone.
        e.set_iter_limit(500);
        e.zoom_at_center(1e5);
        assert_eq!(e.iter_limit, 500);
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
