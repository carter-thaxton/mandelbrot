//! Platform-independent explorer state: navigation, iteration policy, colors.
//!
//! Nothing in here knows about Metal, AppKit, or even winit. Input is fed in as
//! plain numbers (physical pixels), and the output is a [`FrameParams`] that any
//! renderer backend can consume.

use std::sync::Arc;

use crate::bigfixed::{Fixed, frac_limbs_for_scale};
use crate::perturb::{ReferenceOrbit, compute_reference};

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
    /// Reference orbit at `center` (Perturbation only). Renderers can use
    /// `Arc::ptr_eq` to skip re-uploading an unchanged orbit.
    pub reference: Option<Arc<ReferenceOrbit>>,
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

pub struct Explorer {
    pub center: (Fixed, Fixed),
    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub precision_mode: PrecisionMode,
    pub palette: u32,
    pub iter_multiplier: f64,
    pub color_density: f32,
    drag_anchor: Option<(f64, f64)>,
    pub cursor: (f64, f64),
    /// (center, max_iter) the cached orbit was computed for.
    reference: Option<((Fixed, Fixed), u32, Arc<ReferenceOrbit>)>,
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
            iter_multiplier: 1.0,
            color_density: 1.0,
            drag_anchor: None,
            cursor: (0.0, 0.0),
            reference: None,
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

    /// Iterations grow roughly linearly with zoom depth (in decades).
    pub fn max_iter(&self) -> u32 {
        let decades = self.zoom().max(1.0).log10();
        let base = 200.0 + 250.0 * decades + 15.0 * decades * decades;
        (base * self.iter_multiplier).clamp(32.0, 50_000.0) as u32
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
        self.shift_center(-dx * self.scale, dy * self.scale);
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

    /// The reference orbit for the current view, recomputed only on change.
    fn reference_orbit(&mut self) -> Arc<ReferenceOrbit> {
        let max_iter = self.max_iter();
        if let Some((c, n, orbit)) = &self.reference {
            if *c == self.center && *n == max_iter {
                return orbit.clone();
            }
        }
        let orbit = Arc::new(compute_reference(&self.center.0, &self.center.1, max_iter));
        self.reference = Some((self.center.clone(), max_iter, orbit.clone()));
        orbit
    }

    pub fn frame_params(&mut self) -> FrameParams {
        let precision = self.precision();
        FrameParams {
            center: (self.center.0.to_f64(), self.center.1.to_f64()),
            scale: self.scale,
            width: self.width,
            height: self.height,
            max_iter: self.max_iter(),
            palette: self.palette,
            precision,
            color_density: self.color_density,
            reference: (precision == Precision::Perturbation).then(|| self.reference_orbit()),
        }
    }

    /// Center as decimal strings with enough digits to reproduce this view.
    pub fn center_strings(&self) -> (String, String) {
        let digits = ((-self.scale.log10()).ceil() as usize + 3).max(8);
        (self.center.0.to_decimal(digits), self.center.1.to_decimal(digits))
    }

    pub fn hud(&self, gpu_ms: Option<f64>) -> String {
        let mode = match (self.precision_mode, self.precision()) {
            (PrecisionMode::Auto, Precision::F32) => "auto/f32",
            (PrecisionMode::Auto, Precision::DoubleFloat) => "auto/df64",
            (PrecisionMode::Auto, Precision::Perturbation) => "auto/perturb",
            (PrecisionMode::Forced(Precision::F32), _) => "f32",
            (PrecisionMode::Forced(Precision::DoubleFloat), _) => "df64",
            (PrecisionMode::Forced(Precision::Perturbation), _) => "perturb",
        };
        let reference = match (&self.reference, self.precision()) {
            (Some((_, _, r)), Precision::Perturbation) => format!(
                " (ref {}{} it, {:.0} ms)",
                r.points.len() - 1,
                if r.escaped { " escaped" } else { "" },
                r.compute_ms
            ),
            _ => String::new(),
        };
        let ms = gpu_ms.map_or(String::new(), |ms| format!("  |  gpu {ms:.1} ms"));
        // The title bar can't show 35 digits usefully; `c` prints them all.
        let (cx, cy) = (self.center.0.to_f64(), self.center.1.to_f64());
        format!(
            "Mandelbrot  |  {cx:+.15}, {cy:+.15}  |  zoom {:.3e}  |  iters {}  |  {mode}{reference}  |  {}{ms}",
            self.zoom(),
            self.max_iter(),
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
}
