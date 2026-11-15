//! Metal backend: the only module that touches objc2 / Metal / Core Animation.
//!
//! Three compute kernels, no render pass or vertex pipeline:
//! - `mandelbrot` writes each pixel's smooth iteration count (mu) into a
//!   persistent single-channel float canvas;
//! - `colorize` turns mu into colors, straight into the CAMetalLayer's
//!   drawable when presenting (so the layer has `framebufferOnly = false`),
//!   or into an RGBA texture for snapshots and captures;
//! - `reproject` resamples the canvas into a spare texture when the view
//!   moves, and the two are swapped.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::{
    BatchStats, ESCAPE_BINS, EscapeHistogram, FrameStats, GpuColors, GpuParams, GpuReprojection, GpuTileStats, Output,
    Renderer, Reprojection, Result, TILE, TileBatch, sample_grids,
};
use crate::app::{Colors, FrameParams};
use crate::perturb::ReferenceOrbit;

const SHADER_SOURCE: &str = include_str!("mandelbrot.metal");

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Texture = ProtocolObject<dyn MTLTexture>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// The canvas format: one f32 of smooth iteration count per pixel.
const CANVAS_FORMAT: MTLPixelFormat = MTLPixelFormat::R32Float;

pub struct MetalRenderer {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<Pipeline>,
    reproject_pipeline: Retained<Pipeline>,
    colorize_pipeline: Retained<Pipeline>,
    /// None for headless (capture-only) renderers.
    layer: Option<Retained<CAMetalLayer>>,
    /// Persistent mu image the kernel fills progressively; colorized to present.
    canvas: Option<Retained<Texture>>,
    /// Same size as `canvas`: the target of `reproject`, then swapped in.
    spare: Option<Retained<Texture>>,
    /// Reference orbit for perturbation, uploaded only when the Arc changes.
    reference: Option<(Arc<ReferenceOrbit>, Retained<Buffer>)>,
    /// Bound at buffer(1) when there is no reference, to keep validation happy.
    empty_buffer: Retained<Buffer>,
    /// Per-tile kernel statistics (shared memory), grown as batches need.
    tile_stats: Retained<Buffer>,
    /// The batch's escape-count histogram (shared memory, zeroed before
    /// each dispatch).
    escapes: Retained<Buffer>,
}

impl MetalRenderer {
    pub fn new<W: HasWindowHandle>(window: Option<&W>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("failed to create command queue")?;
        let library = build_library(&device)?;
        let pipeline = build_pipeline(&device, &library, "mandelbrot")?;
        let reproject_pipeline = build_pipeline(&device, &library, "reproject")?;
        let colorize_pipeline = build_pipeline(&device, &library, "colorize")?;
        let layer = window.map(|w| attach_layer(w, &device)).transpose()?;
        let empty_buffer = new_buffer(&device, &[0u8; 64])?;
        let tile_stats = new_buffer(&device, &[GpuTileStats::default(); 64])?;
        let escapes = new_buffer(&device, &[0u32; ESCAPE_BINS])?;
        Ok(Self {
            device,
            queue,
            pipeline,
            reproject_pipeline,
            colorize_pipeline,
            layer,
            canvas: None,
            spare: None,
            reference: None,
            empty_buffer,
            tile_stats,
            escapes,
        })
    }

    fn reference_buffer(&mut self, params: &FrameParams) -> Result<Retained<Buffer>> {
        let Some(orbit) = &params.reference else {
            return Ok(self.empty_buffer.clone());
        };
        if let Some((cached, buffer)) = &self.reference {
            if Arc::ptr_eq(cached, orbit) {
                return Ok(buffer.clone());
            }
        }
        let buffer = new_buffer(&self.device, &orbit.points)?;
        self.reference = Some((orbit.clone(), buffer.clone()));
        Ok(buffer)
    }

    /// Encode the Mandelbrot kernel for `batch` of `target`. Returns each
    /// tile's grid of samples inside the canvas (edge tiles may be partial).
    fn encode(
        &mut self,
        cmd: &CommandBuffer,
        target: &Texture,
        params: &FrameParams,
        batch: &TileBatch,
    ) -> Result<Vec<[u32; 2]>> {
        if batch.tiles.is_empty() {
            return Ok(vec![]);
        }
        let gpu = GpuParams::new(params, batch);
        let per_side = (TILE / gpu.step) as usize;
        let reference = self.reference_buffer(params)?;
        let n = batch.tiles.len();
        if self.tile_stats.length() < n * size_of::<GpuTileStats>() {
            let len = n.next_power_of_two() * size_of::<GpuTileStats>();
            self.tile_stats = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to allocate Metal buffer")?;
        }

        // SAFETY: the buffer holds ESCAPE_BINS u32s, and the GPU is idle
        // (every command buffer is waited on).
        unsafe { self.escapes.contents().cast::<[u32; ESCAPE_BINS]>().write([0; ESCAPE_BINS]) };

        let enc = cmd.computeCommandEncoder().ok_or("failed to create compute encoder")?;
        enc.setComputePipelineState(&self.pipeline);
        let tiles = if size_of_val(batch.tiles.as_slice()) > 4096 {
            Some(new_buffer(&self.device, &batch.tiles)?)
        } else {
            None
        };
        // SAFETY: texture 0 / buffers 0-4 match the kernel signature, setBytes
        // copies its data before returning, the kernel reads at most `ref_len`
        // reference points and one tile per grid slice, writes one stats
        // entry per tile (the buffer holds at least n), and only writes
        // inside those tiles of `target`.
        unsafe {
            enc.setTexture_atIndex(Some(target), 0);
            enc.setBytes_length_atIndex(
                NonNull::from(&gpu).cast::<c_void>(),
                size_of::<GpuParams>(),
                0,
            );
            enc.setBuffer_offset_atIndex(Some(&reference), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.tile_stats), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&self.escapes), 0, 4);
            match &tiles {
                Some(buffer) => enc.setBuffer_offset_atIndex(Some(buffer), 0, 3),
                None => enc.setBytes_length_atIndex(
                    NonNull::new(batch.tiles.as_ptr() as *mut c_void).unwrap(),
                    size_of_val(batch.tiles.as_slice()),
                    3,
                ),
            }
        }
        // One slice of the grid per tile; pack several small slices into a
        // threadgroup so coarse steps still fill SIMD groups. A threadgroup
        // then holds whole tiles: at most 256 threads and 64 tiles, the
        // shader's MAX_GROUP_THREADS and MAX_GROUP_TILES.
        let depth = (64 / (per_side * per_side)).clamp(1, n);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: per_side, height: per_side, depth: n },
            MTLSize { width: per_side, height: per_side, depth },
        );
        enc.endEncoding();
        Ok(sample_grids(&batch.tiles, params.width, params.height, gpu.step))
    }

    /// Wait for `cmd` and collect timing plus the kernel's statistics for
    /// tiles with sample grids `grids` (from [`Self::encode`]).
    fn finish(&self, cmd: &CommandBuffer, grids: &[[u32; 2]]) -> Result<BatchStats> {
        cmd.commit();
        // Each dispatch is kept small by the app's scheduler, so waiting here
        // is cheap and keeps the stats readback trivial.
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(format!("GPU error: {}", err.localizedDescription()));
        }
        // SAFETY: the GPU has finished, having written one entry per tile.
        let gpu = unsafe { std::slice::from_raw_parts(self.tile_stats.contents().cast::<GpuTileStats>().as_ptr(), grids.len()) };
        let tiles = gpu.iter().zip(grids).map(|(s, &grid)| s.to_tile_stats(grid)).collect();
        let mut stats = BatchStats::of_tiles((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0, tiles);
        // SAFETY: the GPU has finished writing the histogram.
        stats.total.escapes = EscapeHistogram(unsafe { self.escapes.contents().cast::<[u32; ESCAPE_BINS]>().read() });
        Ok(stats)
    }

    fn new_texture(&self, w: usize, h: usize, format: MTLPixelFormat, shared: bool) -> Result<Retained<Texture>> {
        // SAFETY: plain descriptor construction with valid arguments.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, w, h, false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead);
        desc.setStorageMode(if shared { MTLStorageMode::Shared } else { MTLStorageMode::Private });
        self.device.newTextureWithDescriptor(&desc).ok_or_else(|| "failed to create texture".into())
    }

    /// Encode coloring the mu texture `mu` into `out` (sized as `out`).
    fn encode_colorize(&self, cmd: &CommandBuffer, mu: &Texture, out: &Texture, colors: &Colors) -> Result<()> {
        let gpu = GpuColors::new(colors);
        let enc = cmd.computeCommandEncoder().ok_or("failed to create compute encoder")?;
        enc.setComputePipelineState(&self.colorize_pipeline);
        // SAFETY: textures 0/1 and buffer 0 match the `colorize` kernel, and
        // setBytes copies `gpu` before returning.
        unsafe {
            enc.setTexture_atIndex(Some(mu), 0);
            enc.setTexture_atIndex(Some(out), 1);
            enc.setBytes_length_atIndex(NonNull::from(&gpu).cast::<c_void>(), size_of::<GpuColors>(), 0);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: out.width(), height: out.height(), depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        Ok(())
    }

    /// Color `mu` and read the result back as RGBA8 rows.
    fn colorized_rgba(&self, mu: &Texture, colors: &Colors) -> Result<Vec<u8>> {
        let (w, h) = (mu.width(), mu.height());
        let rgba = self.new_texture(w, h, MTLPixelFormat::RGBA8Unorm, true)?;
        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        self.encode_colorize(&cmd, mu, &rgba, colors)?;
        cmd.commit();
        cmd.waitUntilCompleted();
        let mut pixels = vec![0u8; w * h * 4];
        let region = MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w, height: h, depth: 1 } };
        // SAFETY: `pixels` holds exactly h rows of w*4 bytes, matching the region.
        unsafe {
            rgba.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new(pixels.as_mut_ptr().cast()).unwrap(),
                w * 4,
                region,
                0,
            );
        }
        Ok(pixels)
    }
}

impl Renderer for MetalRenderer {
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        if let Some(layer) = &self.layer {
            layer.setContentsScale(scale_factor);
            layer.setDrawableSize(CGSize { width: width as f64, height: height as f64 });
        }
        let make = || {
            self.new_texture(width.max(1) as usize, height.max(1) as usize, CANVAS_FORMAT, false)
                .map_err(|e| eprintln!("{e}"))
                .ok()
        };
        (self.canvas, self.spare) = (make(), make());
    }

    fn reproject(&mut self, xf: &Reprojection) -> Result<()> {
        let (Some(canvas), Some(spare)) = (self.canvas.clone(), self.spare.clone()) else {
            return Err("reproject() before resize()".into());
        };
        let gpu = GpuReprojection::new(xf);
        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        let enc = cmd.computeCommandEncoder().ok_or("failed to create compute encoder")?;
        enc.setComputePipelineState(&self.reproject_pipeline);
        // SAFETY: textures 0/1 and buffer 0 match the `reproject` kernel, and
        // setBytes copies `gpu` before returning.
        unsafe {
            enc.setTexture_atIndex(Some(&canvas), 0);
            enc.setTexture_atIndex(Some(&spare), 1);
            enc.setBytes_length_atIndex(
                NonNull::from(&gpu).cast::<c_void>(),
                size_of::<GpuReprojection>(),
                0,
            );
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: spare.width(), height: spare.height(), depth: 1 },
            MTLSize { width: 16, height: 16, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        (self.canvas, self.spare) = (Some(spare), Some(canvas));
        Ok(())
    }

    fn draw(&mut self, params: &FrameParams, batch: &TileBatch) -> Result<BatchStats> {
        let canvas = self.canvas.clone().ok_or("draw() before resize()")?;
        // The canvas is only as big as the last resize; never dispatch past it.
        let mut params = params.clone();
        params.width = params.width.min(canvas.width() as u32);
        params.height = params.height.min(canvas.height() as u32);

        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        let grids = self.encode(&cmd, &canvas, &params, batch)?;
        self.finish(&cmd, &grids)
    }

    fn present(&mut self, colors: &Colors) -> Result<()> {
        // Headless renderers (tests, captures) have nowhere to present to.
        let Some(layer) = &self.layer else { return Ok(()) };
        let canvas = self.canvas.clone().ok_or("present() before resize()")?;
        let Some(drawable) = layer.nextDrawable() else {
            return Ok(()); // e.g. window minimized
        };
        // Color the whole canvas: drawables don't keep earlier contents.
        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        self.encode_colorize(&cmd, &canvas, &drawable.texture(), colors)?;
        cmd.presentDrawable(ProtocolObject::from_ref(&*drawable));
        // No wait: later work on this queue is ordered after the coloring.
        cmd.commit();
        Ok(())
    }

    fn snapshot(&mut self, colors: &Colors) -> Result<Vec<u8>> {
        let canvas = self.canvas.clone().ok_or("snapshot() before resize()")?;
        self.colorized_rgba(&canvas, colors)
    }

    fn capture(&mut self, params: &FrameParams, step: u32, colors: &Colors) -> Result<(Vec<u8>, FrameStats)> {
        let (w, h) = (params.width as usize, params.height as usize);
        let mu = self.new_texture(w, h, CANVAS_FORMAT, false)?;
        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        let batch = TileBatch::all(params.width, params.height, step, Output::Pixels);
        let grids = self.encode(&cmd, &mu, params, &batch)?;
        let stats = self.finish(&cmd, &grids)?.total;
        Ok((self.colorized_rgba(&mu, colors)?, stats))
    }
}

/// Copy a slice of plain data into a new shared-memory Metal buffer.
fn new_buffer<T: Copy>(device: &Device, data: &[T]) -> Result<Retained<Buffer>> {
    let len = size_of_val(data).max(1);
    // SAFETY: `data` is valid for `len` bytes (Metal copies it immediately).
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
            len,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| "failed to allocate Metal buffer".into())
}

fn build_library(device: &Device) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let options = MTLCompileOptions::new();
    // df64 arithmetic depends on exact IEEE rounding; fast-math would break it.
    options.setMathMode(MTLMathMode::Safe);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(SHADER_SOURCE), Some(&options))
        .map_err(|e| format!("shader compile failed: {}", e.localizedDescription()))
}

fn build_pipeline(
    device: &Device,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<Pipeline>> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("kernel `{name}` not found"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("pipeline creation failed: {}", e.localizedDescription()))
}

/// Add a CAMetalLayer as an auto-resizing sublayer of the window's NSView.
fn attach_layer<W: HasWindowHandle>(window: &W, device: &Device) -> Result<Retained<CAMetalLayer>> {
    let handle = window.window_handle().map_err(|e| e.to_string())?;
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return Err("expected an AppKit window handle".into());
    };
    // SAFETY: winit guarantees the handle is a live NSView for the window's
    // lifetime, and we are on the main thread (winit runs callbacks there).
    let view: &NSView = unsafe { appkit.ns_view.cast().as_ref() };

    let layer = CAMetalLayer::new();
    layer.setDevice(Some(device));
    layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    layer.setFramebufferOnly(false); // we blit into its drawables
    layer.setAutoresizingMask(
        CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable,
    );

    view.setWantsLayer(true);
    let root = view.layer().ok_or("view has no backing layer")?;
    layer.setFrame(root.bounds());
    root.addSublayer(&layer);
    Ok(layer)
}
