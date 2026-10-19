//! Backend-agnostic rendering interface.
//!
//! The app talks only to [`Renderer`]. Each backend (currently Metal) lives in
//! its own submodule and is selected by [`create`]. A future `wgpu` backend
//! would implement the same trait and reuse [`GpuParams`] as its uniform.
//!
//! Renderers own a persistent *canvas* the size of the window, divided into
//! [`TILE`]-pixel square tiles. The app fills it progressively with `draw`,
//! tile batches at a time, and calls `present` to show the canvas whenever it
//! changed, so partial progress is visible. When the view moves, `reproject`
//! resamples the existing canvas into the new view so previous work is
//! reused.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::app::{FrameParams, Precision};
use crate::perturb::{complex_floatexp, frexp};

#[cfg(target_os = "macos")]
mod metal;

pub type Result<T> = std::result::Result<T, String>;

/// Canvas tile edge in pixels. Must match TILE in the shader.
pub const TILE: u32 = 16;

/// What a kernel dispatch produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// Write pixels to the target (and gather statistics).
    Pixels,
    /// Gather statistics only; leave the target untouched.
    StatsOnly,
}

/// Which tiles of the canvas to compute, how coarsely, and what to produce.
#[derive(Clone, Debug)]
pub struct TileBatch {
    /// Top-left pixel of each tile (multiples of [`TILE`]).
    pub tiles: Vec<[u32; 2]>,
    /// Sample every `step` pixels and fill step x step blocks (1 = full res).
    /// A power of two no larger than [`TILE`].
    pub step: u32,
    pub output: Output,
}

impl TileBatch {
    pub fn new(tiles: Vec<[u32; 2]>, step: u32, output: Output) -> Self {
        Self { tiles, step, output }
    }

    /// Every tile of a `width` x `height` canvas.
    pub fn all(width: u32, height: u32, step: u32, output: Output) -> Self {
        let tiles = (0..height.div_ceil(TILE))
            .flat_map(|ty| (0..width.div_ceil(TILE)).map(move |tx| [tx * TILE, ty * TILE]))
            .collect();
        Self::new(tiles, step, output)
    }
}

/// Where the new view lies in the old canvas: new pixel (x, y) shows old
/// canvas position `old_center + (p - new_center) * ratio + shift`, with
/// positions in pixels (texel centers at +0.5).
#[derive(Clone, Copy, Debug)]
pub struct Reprojection {
    /// New pixel size in old pixels (< 1 when zooming in).
    pub ratio: f64,
    /// New view center minus old, in old pixels (+y down).
    pub shift: (f64, f64),
    pub old_size: (u32, u32),
}

/// Measurements from one `draw`/`capture`, used for scheduling and for
/// automatic iteration limits.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub gpu_ms: f64,
    /// Number of pixels actually iterated.
    pub samples: u64,
    /// Highest iteration count among escaped samples.
    pub max_escaped: u32,
    /// Escaped samples that needed more than 3/4 of the iteration limit.
    pub near_limit: u32,
    /// Samples that never escaped (in the set, or under-iterated).
    pub interior: u32,
}

impl FrameStats {
    /// Combine statistics from several draws.
    pub fn merge(&mut self, s: &FrameStats) {
        self.gpu_ms += s.gpu_ms;
        self.samples += s.samples;
        self.max_escaped = self.max_escaped.max(s.max_escaped);
        self.near_limit += s.near_limit;
        self.interior += s.interior;
    }
}

pub trait Renderer {
    /// The drawable surface changed size (physical pixels) or backing scale.
    /// Canvas contents are undefined afterwards.
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64);

    /// Compute `batch` of the canvas (or only its statistics; see [`Output`]).
    /// Nothing appears on screen until [`Renderer::present`].
    fn draw(&mut self, params: &FrameParams, batch: &TileBatch) -> Result<FrameStats>;

    /// Show the current canvas in the window (a no-op without one).
    fn present(&mut self) -> Result<()>;

    /// Read back the current canvas as RGBA8 rows (what `present` shows).
    fn snapshot(&mut self) -> Result<Vec<u8>>;

    /// Resample the canvas contents for a new view (see [`Reprojection`]).
    /// Pixels with no source become black.
    fn reproject(&mut self, xf: &Reprojection) -> Result<()>;

    /// Render offscreen at `params.width` x `params.height`; returns RGBA8 rows.
    fn capture(&mut self, params: &FrameParams, step: u32) -> Result<(Vec<u8>, FrameStats)>;
}

/// Create the platform's renderer for a window.
pub fn create<W: HasWindowHandle + HasDisplayHandle>(window: &W) -> Result<Box<dyn Renderer>> {
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(metal::MetalRenderer::new(Some(window))?))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = window;
        Err("no renderer backend for this platform yet".into())
    }
}

/// Create a renderer with no window, for `capture` only.
pub fn create_headless() -> Result<Box<dyn Renderer>> {
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(metal::MetalRenderer::new(None::<&NoWindow>)?))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("no renderer backend for this platform yet".into())
    }
}

/// Uniform block shared with the GPU kernel. Layout must match `Params` in the
/// shader (float2 fields are 8-byte aligned there). Doubles are split into
/// hi/lo f32 pairs since Apple GPUs (and WGSL) have no f64.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuParams {
    pub center_x: [f32; 2],
    pub center_y: [f32; 2],
    pub scale: [f32; 2],
    pub width: u32,
    pub height: u32,
    pub max_iter: u32,
    pub palette: u32,
    pub precision: u32,
    pub color_density: f32,
    /// Number of reference orbit points bound at buffer(1) (Perturbation).
    pub ref_len: u32,
    /// Scale as floatexp (scale_m * 2^scale_e): below ~1e-38 it has no f32.
    pub scale_m: f32,
    pub scale_e: i32,
    /// Sample spacing (see [`TileBatch::step`]).
    pub step: u32,
    /// View center minus reference point, as complex floatexp (Perturbation).
    pub ref_off_m: [f32; 2],
    pub ref_off_e: i32,
    pub _pad0: [u32; 2],
    /// Series approximation: start at iteration sa_n with dz = A * dc.
    pub sa_n: u32,
    pub sa_a_m: [f32; 2],
    pub sa_a_e: i32,
    /// Nonzero: compute statistics only, write no pixels ([`Output`]).
    pub stats_only: u32,
}

fn split_f64(v: f64) -> [f32; 2] {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    [hi, lo]
}

impl GpuParams {
    pub fn new(p: &FrameParams, step: u32, output: Output) -> Self {
        let (scale_m, scale_e) = frexp(p.scale);
        let (ref_off_m, ref_off_e) = complex_floatexp(p.ref_offset.0, p.ref_offset.1);
        Self {
            center_x: split_f64(p.center.0),
            center_y: split_f64(p.center.1),
            scale: split_f64(p.scale),
            width: p.width,
            height: p.height,
            max_iter: p.max_iter,
            palette: p.palette,
            precision: match p.precision {
                Precision::F32 => 0,
                Precision::DoubleFloat => 1,
                Precision::Perturbation => 2,
            },
            color_density: p.color_density,
            ref_len: p.reference.as_ref().map_or(0, |r| r.points.len() as u32),
            scale_m: scale_m as f32,
            scale_e,
            step: step.clamp(1, TILE),
            ref_off_m,
            ref_off_e,
            _pad0: [0; 2],
            sa_n: p.series_n,
            sa_a_m: p.series_a.0,
            sa_a_e: p.series_a.1,
            stats_only: (output == Output::StatsOnly) as u32,
        }
    }
}

/// Uninhabited placeholder so `create_headless` can name a window type.
#[cfg(target_os = "macos")]
pub(crate) enum NoWindow {}

#[cfg(target_os = "macos")]
impl HasWindowHandle for NoWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        match *self {}
    }
}
