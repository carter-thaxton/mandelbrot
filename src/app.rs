//! Platform-independent explorer state: navigation, iteration policy, colors.
//!
//! Nothing in here knows about Metal, AppKit, or even winit. Input is fed in as
//! plain numbers (physical pixels), and the output is a [`FrameParams`] that any
//! renderer backend can consume.

/// Which arithmetic the GPU kernel should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Plain 32-bit floats. Fast; pixelates around 1e-5 zoom.
    F32,
    /// Double-float (two f32s, ~48-bit mantissa). ~3-5x slower; good to ~1e-13.
    DoubleFloat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    Auto,
    Forced(Precision),
}

/// Everything a renderer needs to draw one frame.
#[derive(Clone, Copy, Debug)]
pub struct FrameParams {
    pub center: (f64, f64),
    /// Complex-plane units per physical pixel.
    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub max_iter: u32,
    pub palette: u32,
    pub precision: Precision,
    /// Palette cycles per 100 iterations (before the log-ish remap in the shader).
    pub color_density: f32,
}

pub const PALETTE_COUNT: u32 = 6;
pub const PALETTE_NAMES: [&str; PALETTE_COUNT as usize] =
    ["Classic", "Fire", "Ocean", "Neon", "Pastel", "Grayscale"];

const HOME_CENTER: (f64, f64) = (-0.75, 0.0);
/// Complex-plane height visible at zoom 1.
const HOME_SPAN: f64 = 2.6;
/// Beyond this the f64 center itself (and double-float) run out of bits.
const MIN_SCALE: f64 = 1e-15;
/// Switch to double-float once a pixel is smaller than this (Auto mode).
const F32_LIMIT: f64 = 2e-6;

pub struct Explorer {
    pub center: (f64, f64),
    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub precision_mode: PrecisionMode,
    pub palette: u32,
    pub iter_multiplier: f64,
    pub color_density: f32,
    drag_anchor: Option<(f64, f64)>,
    pub cursor: (f64, f64),
}

impl Explorer {
    pub fn new(width: u32, height: u32) -> Self {
        let mut e = Self {
            center: HOME_CENTER,
            scale: 1.0,
            width: width.max(1),
            height: height.max(1),
            precision_mode: PrecisionMode::Auto,
            palette: 0,
            iter_multiplier: 1.0,
            color_density: 1.0,
            drag_anchor: None,
            cursor: (0.0, 0.0),
        };
        e.reset_view();
        e
    }

    fn home_scale(&self) -> f64 {
        HOME_SPAN / self.height as f64
    }

    pub fn reset_view(&mut self) {
        self.center = HOME_CENTER;
        self.scale = self.home_scale();
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
            PrecisionMode::Auto if self.scale < F32_LIMIT => Precision::DoubleFloat,
            PrecisionMode::Auto => Precision::F32,
        }
    }

    pub fn cycle_precision_mode(&mut self) {
        self.precision_mode = match self.precision_mode {
            PrecisionMode::Auto => PrecisionMode::Forced(Precision::F32),
            PrecisionMode::Forced(Precision::F32) => PrecisionMode::Forced(Precision::DoubleFloat),
            PrecisionMode::Forced(Precision::DoubleFloat) => PrecisionMode::Auto,
        };
    }

    /// Iterations grow roughly linearly with zoom depth (in decades).
    pub fn max_iter(&self) -> u32 {
        let decades = self.zoom().max(1.0).log10();
        let base = 200.0 + 250.0 * decades + 15.0 * decades * decades;
        (base * self.iter_multiplier).clamp(32.0, 50_000.0) as u32
    }

    fn pixel_to_complex(&self, px: f64, py: f64) -> (f64, f64) {
        (
            self.center.0 + (px - self.width as f64 / 2.0) * self.scale,
            self.center.1 - (py - self.height as f64 / 2.0) * self.scale,
        )
    }

    /// Zoom by `factor` (>1 zooms in) keeping the point under (px, py) fixed.
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        let anchor = self.pixel_to_complex(px, py);
        let new_scale = (self.scale / factor).clamp(MIN_SCALE, self.home_scale() * 4.0);
        let k = new_scale / self.scale;
        self.center.0 = anchor.0 + (self.center.0 - anchor.0) * k;
        self.center.1 = anchor.1 + (self.center.1 - anchor.1) * k;
        self.scale = new_scale;
    }

    pub fn zoom_at_cursor(&mut self, factor: f64) {
        self.zoom_at(self.cursor.0, self.cursor.1, factor);
    }

    pub fn zoom_at_center(&mut self, factor: f64) {
        self.zoom_at(self.width as f64 / 2.0, self.height as f64 / 2.0, factor);
    }

    /// Move the content by (dx, dy) physical pixels (content follows the finger).
    pub fn pan_pixels(&mut self, dx: f64, dy: f64) {
        self.center.0 -= dx * self.scale;
        self.center.1 += dy * self.scale;
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

    pub fn frame_params(&self) -> FrameParams {
        FrameParams {
            center: self.center,
            scale: self.scale,
            width: self.width,
            height: self.height,
            max_iter: self.max_iter(),
            palette: self.palette,
            precision: self.precision(),
            color_density: self.color_density,
        }
    }

    pub fn hud(&self, gpu_ms: Option<f64>) -> String {
        let mode = match (self.precision_mode, self.precision()) {
            (PrecisionMode::Auto, Precision::F32) => "auto/f32",
            (PrecisionMode::Auto, Precision::DoubleFloat) => "auto/df64",
            (PrecisionMode::Forced(Precision::F32), _) => "f32",
            (PrecisionMode::Forced(Precision::DoubleFloat), _) => "df64",
        };
        let ms = gpu_ms.map_or(String::new(), |ms| format!("  |  {ms:.1} ms"));
        format!(
            "Mandelbrot  |  {:+.15}, {:+.15}  |  zoom {:.3e}  |  iters {}  |  {}  |  {}{}",
            self.center.0,
            self.center.1,
            self.zoom(),
            self.max_iter(),
            mode,
            PALETTE_NAMES[self.palette as usize],
            ms,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut e = Explorer::new(800, 600);
        let before = e.pixel_to_complex(123.0, 456.0);
        e.zoom_at(123.0, 456.0, 7.5);
        let after = e.pixel_to_complex(123.0, 456.0);
        assert!((before.0 - after.0).abs() < 1e-12);
        assert!((before.1 - after.1).abs() < 1e-12);
    }

    #[test]
    fn auto_precision_switches_when_deep() {
        let mut e = Explorer::new(800, 600);
        assert_eq!(e.precision(), Precision::F32);
        e.zoom_at_center(1e6);
        assert_eq!(e.precision(), Precision::DoubleFloat);
    }
}
