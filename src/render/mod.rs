//! Backend-agnostic rendering interface.
//!
//! The app talks only to [`Renderer`]. Each backend (currently Metal) lives in
//! its own submodule and is selected by [`create`]. A future `wgpu` backend
//! would implement the same trait and reuse [`GpuParams`] as its uniform.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::app::{FrameParams, Precision};
use crate::perturb::frexp;

#[cfg(target_os = "macos")]
mod metal;

pub type Result<T> = std::result::Result<T, String>;

pub trait Renderer {
    /// The drawable surface changed size (physical pixels) or backing scale.
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64);

    /// Draw one frame to the window. Returns GPU time in milliseconds if known.
    fn render(&mut self, params: &FrameParams) -> Result<Option<f64>>;

    /// Render offscreen at `params.width` x `params.height`; returns RGBA8 rows.
    fn capture(&mut self, params: &FrameParams) -> Result<Vec<u8>>;
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
/// shader (including its 8-byte alignment, hence `_pad`). Doubles are split
/// into hi/lo f32 pairs since Apple GPUs (and WGSL) have no f64.
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
    pub _pad: u32,
}

fn split_f64(v: f64) -> [f32; 2] {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    [hi, lo]
}

impl From<&FrameParams> for GpuParams {
    fn from(p: &FrameParams) -> Self {
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
            scale_m: frexp(p.scale).0 as f32,
            scale_e: frexp(p.scale).1,
            _pad: 0,
        }
    }
}

/// Uninhabited placeholder so `create_headless` can name a window type.
#[cfg(target_os = "macos")]
enum NoWindow {}

#[cfg(target_os = "macos")]
impl HasWindowHandle for NoWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        match *self {}
    }
}
