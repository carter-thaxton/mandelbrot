//! Backend-agnostic rendering interface.
//!
//! The app talks only to [`Renderer`]. Each backend lives in its own
//! submodule and is selected by [`create`]: Metal on macOS, wgpu (Vulkan on
//! Linux) everywhere else. Both share the GPU-facing layouts defined here
//! ([`GpuParams`] and friends), which their shaders must match.
//!
//! Renderers own a persistent *canvas* the size of the window, divided into
//! [`TILE`]-pixel square tiles. It holds each pixel's smooth iteration count
//! (mu), not its color. The app fills it progressively with `draw`, tile
//! batches at a time, and calls `present` to color it (with the current
//! [`Colors`]) and show it, so partial progress is visible and color changes
//! never need recomputation. When the view moves, `reproject` resamples the
//! canvas into the new view so previous work is reused.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::app::{Colors, FrameParams, Precision};
use crate::perturb::{complex_floatexp, frexp};

#[cfg(target_os = "macos")]
mod metal;
#[cfg(not(target_os = "macos"))]
mod wgpu;

pub type Result<T> = std::result::Result<T, String>;

/// Name of the platform's backend (what [`create`] and [`create_headless`]
/// make), for anything kept per backend, like recorded fingerprints.
#[cfg(target_os = "macos")]
pub const BACKEND: &str = "metal";
#[cfg(not(target_os = "macos"))]
pub const BACKEND: &str = "wgpu";

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

/// One tile to compute. Layout matches `TileJob` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileJob {
    /// Top-left pixel (multiples of [`TILE`]).
    pub origin: [u32; 2],
}

impl TileJob {
    pub fn new(origin: [u32; 2]) -> Self {
        Self { origin }
    }
}

/// Default for [`TileBatch::rough_mu`]: about a quarter palette cycle at
/// color density 1 (a cycle per 50 iterations).
pub const DEFAULT_ROUGH_MU: f32 = 12.0;

/// Which tiles of the canvas to compute, how coarsely, and what to produce.
#[derive(Clone, Debug)]
pub struct TileBatch {
    pub tiles: Vec<TileJob>,
    /// Sample every `step` pixels and fill step x step blocks (1 = full res).
    /// A power of two no larger than [`TILE`].
    pub step: u32,
    pub output: Output,
    /// Adjacent samples whose smooth iteration counts differ by more than
    /// this count as rough ([`TileStats::rough`]).
    pub rough_mu: f32,
}

impl TileBatch {
    pub fn new(tiles: Vec<TileJob>, step: u32, output: Output) -> Self {
        Self { tiles, step, output, rough_mu: DEFAULT_ROUGH_MU }
    }

    /// Every tile of a `width` x `height` canvas.
    pub fn all(width: u32, height: u32, step: u32, output: Output) -> Self {
        let tiles = (0..height.div_ceil(TILE))
            .flat_map(|ty| (0..width.div_ceil(TILE)).map(move |tx| TileJob::new([tx * TILE, ty * TILE])))
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

/// Measurements for one tile of a draw, at the batch's step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileStats {
    /// Samples inside the canvas (edge tiles may be partial).
    pub samples: u32,
    /// Highest iteration count among escaped samples.
    pub max_escaped: u32,
    /// Sum of the escaped samples' iteration counts. With `interior` times
    /// the iteration limit, what the tile cost.
    pub escaped_iters: u32,
    /// Escaped samples that needed more than 3/4 of the iteration limit.
    pub near_limit: u32,
    /// Samples that never escaped.
    pub interior: u32,
    /// Horizontally or vertically adjacent sample pairs within the tile.
    pub pairs: u32,
    /// Pairs, both escaped, whose smooth iteration counts differ by more
    /// than [`TileBatch::rough_mu`]: noise rather than detail.
    pub rough: u32,
    /// Pairs where exactly one sample escaped (the fringe next to black).
    pub edge: u32,
}

impl TileStats {
    /// Adjacent pairs in a `w` x `h` grid of samples.
    pub fn pairs_in(w: u32, h: u32) -> u32 {
        (w.saturating_sub(1) * h) + (w * h.saturating_sub(1))
    }
}

/// Measurements from one `draw`: totals, and one entry per tile of the
/// batch, in the batch's order.
#[derive(Clone, Debug, Default)]
pub struct BatchStats {
    pub total: FrameStats,
    pub tiles: Vec<TileStats>,
}

impl BatchStats {
    /// Totals merged from the tiles (all but the escape histogram, which
    /// isn't kept per tile).
    pub fn of_tiles(gpu_ms: f64, tiles: Vec<TileStats>) -> Self {
        let mut total = FrameStats { gpu_ms, ..Default::default() };
        for t in &tiles {
            total.samples += t.samples as u64;
            total.max_escaped = total.max_escaped.max(t.max_escaped);
            total.near_limit += t.near_limit;
            total.interior += t.interior;
        }
        Self { total, tiles }
    }
}

/// Bins of [`EscapeHistogram`]: enough for MAX_ITER. Must match the shader.
pub const ESCAPE_BINS: usize = 72;
/// Bins per doubling of the iteration count.
pub const ESCAPE_BINS_PER_OCTAVE: f64 = 4.0;

/// How many escaped samples needed how many iterations: bin b counts
/// counts n with floor(log2(n) * ESCAPE_BINS_PER_OCTAVE) = b (n = 0 in bin 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EscapeHistogram(pub [u32; ESCAPE_BINS]);

impl Default for EscapeHistogram {
    fn default() -> Self {
        Self([0; ESCAPE_BINS])
    }
}

impl EscapeHistogram {
    fn bin(n: u32) -> usize {
        if n == 0 { 0 } else { (((n as f64).log2() * ESCAPE_BINS_PER_OCTAVE) as usize).min(ESCAPE_BINS - 1) }
    }

    /// Count one sample that escaped after `n` iterations.
    pub fn record(&mut self, n: u32) {
        self.0[Self::bin(n)] += 1;
    }

    pub fn merge(&mut self, other: &EscapeHistogram) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a += b;
        }
    }

    pub fn total(&self) -> u64 {
        self.0.iter().map(|&c| c as u64).sum()
    }

    /// An iteration count that at least a fraction `q` of the samples
    /// escaped within: the upper edge of the bin where that fraction is
    /// reached (so at most ~19% above the exact quantile). None if empty.
    pub fn quantile(&self, q: f64) -> Option<u32> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        let wanted = (q * total as f64).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (b, &c) in self.0.iter().enumerate() {
            seen += c as u64;
            if seen >= wanted {
                return Some(((b + 1) as f64 / ESCAPE_BINS_PER_OCTAVE).exp2().ceil() as u32);
            }
        }
        unreachable!("the bins hold `total` samples")
    }
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
    /// Distribution of the escaped samples' iteration counts.
    pub escapes: EscapeHistogram,
}

impl FrameStats {
    /// Combine statistics from several draws.
    pub fn merge(&mut self, s: &FrameStats) {
        self.gpu_ms += s.gpu_ms;
        self.samples += s.samples;
        self.max_escaped = self.max_escaped.max(s.max_escaped);
        self.near_limit += s.near_limit;
        self.interior += s.interior;
        self.escapes.merge(&s.escapes);
    }
}

pub trait Renderer {
    /// The drawable surface changed size (physical pixels) or backing scale.
    /// Canvas contents are undefined afterwards.
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64);

    /// Compute `batch` of the canvas (or only its statistics; see [`Output`]).
    /// Nothing appears on screen until [`Renderer::present`].
    fn draw(&mut self, params: &FrameParams, batch: &TileBatch) -> Result<BatchStats>;

    /// Color the canvas and show it in the window (a no-op without one).
    fn present(&mut self, colors: &Colors) -> Result<()>;

    /// The canvas colored as `present` would show it, as RGBA8 rows.
    fn snapshot(&mut self, colors: &Colors) -> Result<Vec<u8>>;

    /// Resample the canvas contents for a new view (see [`Reprojection`]).
    /// Pixels with no source become black.
    fn reproject(&mut self, xf: &Reprojection) -> Result<()>;

    /// Render offscreen at `params.width` x `params.height`, colored with
    /// `colors`; returns RGBA8 rows.
    fn capture(&mut self, params: &FrameParams, step: u32, colors: &Colors) -> Result<(Vec<u8>, FrameStats)>;
}

/// Create the platform's renderer for a window.
pub fn create<W: HasWindowHandle + HasDisplayHandle>(window: &W) -> Result<Box<dyn Renderer>> {
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(metal::MetalRenderer::new(Some(window))?))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(Box::new(wgpu::WgpuRenderer::new(Some(window))?))
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
        Ok(Box::new(wgpu::WgpuRenderer::new(None::<&NoWindow>)?))
    }
}

/// Uniform block shared with the GPU kernel. Layout must match `Params` in the
/// shader, where float2 fields are 8-byte aligned, which makes the whole
/// struct 8-aligned (hence `align(8)`: Rust's `[f32; 2]` is only 4-aligned).
/// Doubles are split into hi/lo f32 pairs since Apple GPUs (and WGSL) have
/// no f64.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuParams {
    pub center_x: [f32; 2],
    pub center_y: [f32; 2],
    pub scale: [f32; 2],
    pub width: u32,
    pub height: u32,
    /// Nonzero: compute statistics only, write no pixels ([`Output`]).
    pub stats_only: u32,
    pub precision: u32,
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
    /// Series approximation: start at iteration sa_n with dz = A * dc.
    pub sa_n: u32,
    pub sa_a_m: [f32; 2],
    pub sa_a_e: i32,
    /// See [`TileBatch::rough_mu`].
    pub rough_mu: f32,
    pub max_iter: u32,
    /// Always 1.0: a value shader compilers can't know, used by the WGSL
    /// shader to stop them from optimizing df64's error terms away (see
    /// `hide` there; Metal's safe math mode doesn't need it). Also makes the
    /// trailing padding explicit, so the struct can be uploaded as bytes.
    pub one: f32,
}

fn split_f64(v: f64) -> [f32; 2] {
    let hi = v as f32;
    let lo = (v - hi as f64) as f32;
    [hi, lo]
}

impl GpuParams {
    pub fn new(p: &FrameParams, batch: &TileBatch) -> Self {
        let step = batch.step;
        let (scale_m, scale_e) = frexp(p.scale);
        let (ref_off_m, ref_off_e) = complex_floatexp(p.ref_offset.0, p.ref_offset.1);
        Self {
            center_x: split_f64(p.center.0),
            center_y: split_f64(p.center.1),
            scale: split_f64(p.scale),
            width: p.width,
            height: p.height,
            precision: match p.precision {
                Precision::F32 => 0,
                Precision::DoubleFloat => 1,
                Precision::Perturbation => 2,
            },
            ref_len: p.reference.as_ref().map_or(0, |r| r.points.len() as u32),
            scale_m: scale_m as f32,
            scale_e,
            step: step.clamp(1, TILE),
            ref_off_m,
            ref_off_e,
            sa_n: p.series_n,
            sa_a_m: p.series_a.0,
            sa_a_e: p.series_a.1,
            rough_mu: batch.rough_mu,
            max_iter: p.max_iter,
            stats_only: (batch.output == Output::StatsOnly) as u32,
            one: 1.0,
        }
    }
}

/// Layout of `Reprojection` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuReprojection {
    pub ratio: f32,
    pub _pad: f32,
    pub shift: [f32; 2],
    pub old_w: u32,
    pub old_h: u32,
}

impl GpuReprojection {
    pub fn new(xf: &Reprojection) -> Self {
        Self {
            ratio: xf.ratio as f32,
            _pad: 0.0,
            shift: [xf.shift.0 as f32, xf.shift.1 as f32],
            old_w: xf.old_size.0,
            old_h: xf.old_size.1,
        }
    }
}

/// Layout of `Colors` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuColors {
    pub palette: u32,
    pub density: f32,
    pub offset: f32,
    pub _pad: u32,
}

impl GpuColors {
    pub fn new(c: &Colors) -> Self {
        Self { palette: c.palette, density: c.density, offset: c.offset, _pad: 0 }
    }
}

/// Layout of one entry of the kernel's per-tile stats buffer, `TileStats`
/// in the shaders; see [`TileStats`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuTileStats {
    pub max_escaped: u32,
    pub escaped_iters: u32,
    pub near_limit: u32,
    pub interior: u32,
    pub rough: u32,
    pub edge: u32,
    pub _pad: [u32; 2],
}

impl GpuTileStats {
    /// The public stats for a tile with a `w` x `h` grid of samples.
    pub fn to_tile_stats(self, [w, h]: [u32; 2]) -> TileStats {
        TileStats {
            samples: w * h,
            max_escaped: self.max_escaped,
            escaped_iters: self.escaped_iters,
            near_limit: self.near_limit,
            interior: self.interior,
            pairs: TileStats::pairs_in(w, h),
            rough: self.rough,
            edge: self.edge,
        }
    }
}

/// Each tile's grid of samples inside a `width` x `height` canvas at `step`
/// (edge tiles may be partial).
pub(crate) fn sample_grids(tiles: &[TileJob], width: u32, height: u32, step: u32) -> Vec<[u32; 2]> {
    tiles
        .iter()
        .map(|t| {
            let w = TILE.min(width.saturating_sub(t.origin[0])).div_ceil(step);
            let h = TILE.min(height.saturating_sub(t.origin[1])).div_ceil(step);
            [w, h]
        })
        .collect()
}

/// Uninhabited placeholder so `create_headless` can name a window type.
pub(crate) enum NoWindow {}

impl HasWindowHandle for NoWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        match *self {}
    }
}

impl HasDisplayHandle for NoWindow {
    fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        match *self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Pins the uniform's layout to the shader's `Params`, where float2 is
    /// 8-byte aligned: every float2 field must land on a multiple of 8.
    #[test]
    fn gpu_params_layout_matches_the_shader() {
        let float2_fields = [
            offset_of!(GpuParams, center_x),
            offset_of!(GpuParams, center_y),
            offset_of!(GpuParams, scale),
            offset_of!(GpuParams, ref_off_m),
            offset_of!(GpuParams, sa_a_m),
        ];
        assert!(float2_fields.iter().all(|o| o % 8 == 0), "{float2_fields:?}");
        assert_eq!(offset_of!(GpuParams, stats_only), 32);
        assert_eq!(offset_of!(GpuParams, ref_off_m), 56);
        assert_eq!(offset_of!(GpuParams, sa_a_m), 72);
        assert_eq!(offset_of!(GpuParams, rough_mu), 84);
        assert_eq!(offset_of!(GpuParams, max_iter), 88);
        assert_eq!(offset_of!(GpuParams, one), 92);
        assert_eq!(size_of::<GpuParams>(), 96);
    }

    /// The other uniforms and the stats entry are uploaded and read back as
    /// raw bytes, so they must have no implicit padding.
    #[test]
    fn shared_layouts_have_no_hidden_padding() {
        assert_eq!(size_of::<GpuReprojection>(), 24);
        assert_eq!(offset_of!(GpuReprojection, shift), 8);
        assert_eq!(size_of::<GpuColors>(), 16);
        assert_eq!(size_of::<GpuTileStats>(), 32);
    }

    #[test]
    fn batch_totals_merge_the_tiles() {
        let tile = |samples, max_escaped, near_limit, interior| TileStats {
            samples,
            max_escaped,
            near_limit,
            interior,
            ..Default::default()
        };
        let b = BatchStats::of_tiles(2.5, vec![tile(256, 900, 3, 10), tile(64, 1200, 1, 0), tile(256, 0, 0, 256)]);
        let t = b.total;
        assert_eq!((t.gpu_ms, t.samples, t.max_escaped, t.near_limit, t.interior), (2.5, 576, 1200, 4, 266));
        assert_eq!(b.tiles.len(), 3);
    }

    #[test]
    fn escape_quantiles_come_from_bin_edges() {
        let mut h = EscapeHistogram::default();
        assert_eq!(h.quantile(0.5), None);
        // 999 fast samples (~100 iterations) and one slow one (~6000).
        for _ in 0..999 {
            h.record(100);
        }
        h.record(6000);
        assert_eq!(h.total(), 1000);
        // 100 is in the bin [2^6.5, 2^6.75): its upper edge, 108.
        assert_eq!(h.quantile(0.5), Some(108));
        assert_eq!(h.quantile(0.999), Some(108));
        // The slowest sample's bin: 6000 is in [2^12.5, 2^12.75).
        assert_eq!(h.quantile(1.0), Some(6889));
        let mut both = h;
        both.merge(&h);
        assert_eq!(both.total(), 2000);
        assert_eq!(both.quantile(0.999), Some(108));
    }

    #[test]
    fn adjacent_pairs_in_a_grid() {
        assert_eq!(TileStats::pairs_in(16, 16), 2 * 15 * 16);
        assert_eq!(TileStats::pairs_in(3, 2), 7);
        assert_eq!(TileStats::pairs_in(1, 1), 0);
        assert_eq!(TileStats::pairs_in(0, 5), 0);
    }

    /// `TileJob` is read by the shader as { uint2 origin; }.
    #[test]
    fn tile_job_layout_matches_the_shader() {
        assert_eq!(offset_of!(TileJob, origin), 0);
        assert_eq!(size_of::<TileJob>(), 8);
    }
}
