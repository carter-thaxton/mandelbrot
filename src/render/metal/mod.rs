//! Metal backend: the only module that touches objc2 / Metal / Core Animation.
//!
//! A compute kernel writes into a persistent canvas texture, which is blitted
//! to the CAMetalLayer's drawable on every draw (so the layer is created with
//! `framebufferOnly = false`); no render pass or vertex pipeline is needed.
//! A second kernel resamples the canvas into a spare texture when the view
//! moves, and the two are swapped.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLOrigin, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage,
};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::{FrameStats, GpuParams, Renderer, Reprojection, Result, TILE, TileBatch};
use crate::app::FrameParams;
use crate::perturb::ReferenceOrbit;

const SHADER_SOURCE: &str = include_str!("mandelbrot.metal");

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;
type Texture = ProtocolObject<dyn MTLTexture>;
type CommandBuffer = ProtocolObject<dyn MTLCommandBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// Layout of `Reprojection` in the shader.
#[repr(C)]
struct GpuReprojection {
    ratio: f32,
    _pad: f32,
    shift: [f32; 2],
    old_w: u32,
    old_h: u32,
}

/// Layout of the kernel's stats buffer (buffer 2); see `FrameStats`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuStats {
    max_escaped: u32,
    near_limit: u32,
    interior: u32,
    _pad: u32,
}

pub struct MetalRenderer {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<Pipeline>,
    reproject_pipeline: Retained<Pipeline>,
    /// None for headless (capture-only) renderers.
    layer: Option<Retained<CAMetalLayer>>,
    /// Persistent image the kernel fills progressively; presented by blit.
    canvas: Option<Retained<Texture>>,
    /// Same size as `canvas`: the target of `reproject`, then swapped in.
    spare: Option<Retained<Texture>>,
    /// Reference orbit for perturbation, uploaded only when the Arc changes.
    reference: Option<(Arc<ReferenceOrbit>, Retained<Buffer>)>,
    /// Bound at buffer(1) when there is no reference, to keep validation happy.
    empty_buffer: Retained<Buffer>,
    /// Kernel statistics (shared memory, reset before each dispatch).
    stats: Retained<Buffer>,
}

impl MetalRenderer {
    pub fn new<W: HasWindowHandle>(window: Option<&W>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("failed to create command queue")?;
        let library = build_library(&device)?;
        let pipeline = build_pipeline(&device, &library, "mandelbrot")?;
        let reproject_pipeline = build_pipeline(&device, &library, "reproject")?;
        let layer = window.map(|w| attach_layer(w, &device)).transpose()?;
        let empty_buffer = new_buffer(&device, &[0u8; 64])?;
        let stats = new_buffer(&device, &[GpuStats::default()])?;
        Ok(Self {
            device,
            queue,
            pipeline,
            reproject_pipeline,
            layer,
            canvas: None,
            spare: None,
            reference: None,
            empty_buffer,
            stats,
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

    /// Encode the Mandelbrot kernel for `batch` of `target`; returns samples.
    fn encode(
        &mut self,
        cmd: &CommandBuffer,
        target: &Texture,
        params: &FrameParams,
        batch: &TileBatch,
    ) -> Result<u64> {
        if batch.tiles.is_empty() {
            return Ok(0);
        }
        let gpu = GpuParams::new(params, batch.step);
        let per_side = (TILE / gpu.step) as usize;
        let reference = self.reference_buffer(params)?;
        // SAFETY: the buffer was created with room for one GpuStats, and the
        // GPU is idle (every command buffer is waited on).
        unsafe { self.stats.contents().cast::<GpuStats>().write(GpuStats::default()) };

        let enc = cmd.computeCommandEncoder().ok_or("failed to create compute encoder")?;
        enc.setComputePipelineState(&self.pipeline);
        let tiles = if size_of_val(batch.tiles.as_slice()) > 4096 {
            Some(new_buffer(&self.device, &batch.tiles)?)
        } else {
            None
        };
        // SAFETY: texture 0 / buffers 0-3 match the kernel signature, setBytes
        // copies its data before returning, the kernel reads at most `ref_len`
        // reference points and one tile per grid slice, and only writes inside
        // those tiles of `target`.
        unsafe {
            enc.setTexture_atIndex(Some(target), 0);
            enc.setBytes_length_atIndex(
                NonNull::from(&gpu).cast::<c_void>(),
                size_of::<GpuParams>(),
                0,
            );
            enc.setBuffer_offset_atIndex(Some(&reference), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.stats), 0, 2);
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
        // threadgroup so coarse steps still fill SIMD groups.
        let n = batch.tiles.len();
        let depth = (64 / (per_side * per_side)).clamp(1, n);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: per_side, height: per_side, depth: n },
            MTLSize { width: per_side, height: per_side, depth },
        );
        enc.endEncoding();
        // Samples actually inside the canvas (edge tiles may be partial).
        let step = gpu.step;
        let samples = batch
            .tiles
            .iter()
            .map(|t| {
                let w = TILE.min(params.width.saturating_sub(t[0])).div_ceil(step);
                let h = TILE.min(params.height.saturating_sub(t[1])).div_ceil(step);
                w as u64 * h as u64
            })
            .sum();
        Ok(samples)
    }

    /// Wait for `cmd` and collect timing plus the kernel's statistics.
    fn finish(&self, cmd: &CommandBuffer, samples: u64) -> Result<FrameStats> {
        cmd.commit();
        // Each dispatch is kept small by the app's scheduler, so waiting here
        // is cheap and keeps the stats readback trivial.
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(format!("GPU error: {}", err.localizedDescription()));
        }
        // SAFETY: the GPU has finished writing the stats buffer.
        let s = unsafe { self.stats.contents().cast::<GpuStats>().read() };
        Ok(FrameStats {
            gpu_ms: (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0,
            samples,
            max_escaped: s.max_escaped,
            near_limit: s.near_limit,
            interior: s.interior,
        })
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
}

impl Renderer for MetalRenderer {
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        if let Some(layer) = &self.layer {
            layer.setContentsScale(scale_factor);
            layer.setDrawableSize(CGSize { width: width as f64, height: height as f64 });
        }
        let make = || {
            self.new_texture(width.max(1) as usize, height.max(1) as usize, MTLPixelFormat::BGRA8Unorm, false)
                .map_err(|e| eprintln!("{e}"))
                .ok()
        };
        (self.canvas, self.spare) = (make(), make());
    }

    fn reproject(&mut self, xf: &Reprojection) -> Result<()> {
        let (Some(canvas), Some(spare)) = (self.canvas.clone(), self.spare.clone()) else {
            return Err("reproject() before resize()".into());
        };
        let gpu = GpuReprojection {
            ratio: xf.ratio as f32,
            _pad: 0.0,
            shift: [xf.shift.0 as f32, xf.shift.1 as f32],
            old_w: xf.old_size.0,
            old_h: xf.old_size.1,
        };
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

    fn draw(&mut self, params: &FrameParams, batch: &TileBatch) -> Result<FrameStats> {
        let canvas = self.canvas.clone().ok_or("draw() before resize()")?;
        // The canvas is only as big as the last resize; never dispatch past it.
        let mut params = params.clone();
        params.width = params.width.min(canvas.width() as u32);
        params.height = params.height.min(canvas.height() as u32);

        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        let samples = self.encode(&cmd, &canvas, &params, batch)?;

        // Present the whole canvas (drawables don't keep earlier contents).
        // Headless renderers (tests) just keep the canvas.
        if let Some(drawable) = self.layer.as_ref().and_then(|l| l.nextDrawable()) {
            let target = drawable.texture();
            let w = canvas.width().min(target.width());
            let h = canvas.height().min(target.height());
            let blit = cmd.blitCommandEncoder().ok_or("failed to create blit encoder")?;
            let origin = MTLOrigin { x: 0, y: 0, z: 0 };
            // SAFETY: same pixel format, and the copied size fits both textures.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &canvas, 0, 0, origin, MTLSize { width: w, height: h, depth: 1 },
                    &target, 0, 0, origin,
                );
            }
            blit.endEncoding();
            cmd.presentDrawable(ProtocolObject::from_ref(&*drawable));
        }
        self.finish(&cmd, samples)
    }

    fn capture(&mut self, params: &FrameParams, step: u32) -> Result<(Vec<u8>, FrameStats)> {
        let (w, h) = (params.width as usize, params.height as usize);
        let texture = self.new_texture(w, h, MTLPixelFormat::RGBA8Unorm, true)?;
        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        let batch = TileBatch::all(params.width, params.height, step);
        let samples = self.encode(&cmd, &texture, params, &batch)?;
        let stats = self.finish(&cmd, samples)?;

        let mut pixels = vec![0u8; w * h * 4];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: w, height: h, depth: 1 },
        };
        // SAFETY: `pixels` holds exactly h rows of w*4 bytes, matching the region.
        unsafe {
            texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new(pixels.as_mut_ptr().cast()).unwrap(),
                w * 4,
                region,
                0,
            );
        }
        Ok((pixels, stats))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Explorer;
    use crate::progressive::Scheduler;
    use objc2_metal::{MTLBlitCommandEncoder as _, MTLCommandBuffer as _};

    impl MetalRenderer {
        /// Canvas contents as RGBA8 (the canvas itself is BGRA8).
        fn read_canvas(&self) -> Vec<u8> {
            let canvas = self.canvas.as_ref().unwrap();
            let (w, h) = (canvas.width(), canvas.height());
            let shared = self.new_texture(w, h, MTLPixelFormat::BGRA8Unorm, true).unwrap();
            let cmd = self.queue.commandBuffer().unwrap();
            let blit = cmd.blitCommandEncoder().unwrap();
            let origin = MTLOrigin { x: 0, y: 0, z: 0 };
            let size = MTLSize { width: w, height: h, depth: 1 };
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    canvas, 0, 0, origin, size, &shared, 0, 0, origin,
                );
            }
            blit.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            let mut px = vec![0u8; w * h * 4];
            unsafe {
                shared.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    NonNull::new(px.as_mut_ptr().cast()).unwrap(),
                    w * 4,
                    MTLRegion { origin, size },
                    0,
                );
            }
            for p in px.chunks_mut(4) {
                p.swap(0, 2);
            }
            px
        }
    }

    fn differing(a: &[u8], b: &[u8]) -> f64 {
        let bad = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| (0..3).any(|i| x[i].abs_diff(y[i]) > 16)).count();
        bad as f64 / (a.len() / 4) as f64
    }

    fn finish(s: &mut Scheduler, e: &mut Explorer, r: &mut MetalRenderer) {
        for _ in 0..100_000 {
            if !s.step(e, r).unwrap() {
                return;
            }
        }
        panic!("scheduler never finished");
    }

    /// After panning (reusing exact tiles) or zooming (recomputing over a
    /// resampled preview), the finished canvas must equal a fresh render.
    #[test]
    fn reused_canvas_matches_fresh_render() {
        let (w, h) = (256, 160);
        let mut r = MetalRenderer::new(None::<&crate::render::NoWindow>).unwrap();
        r.resize(w, h, 1.0);
        let mut e = Explorer::new(w, h);
        e.set_center_str("0", "1");
        e.zoom_at_center(1e3);
        e.set_iter_limit(2000);
        let mut s = Scheduler::default();
        finish(&mut s, &mut e, &mut r);

        for (what, change) in [
            ("pan", Box::new(|e: &mut Explorer| e.pan_pixels(-37.0, 21.0)) as Box<dyn Fn(&mut Explorer)>),
            ("zoom in", Box::new(|e: &mut Explorer| e.zoom_at(90.0, 50.0, 1.7))),
            ("zoom out", Box::new(|e: &mut Explorer| e.zoom_at(150.0, 100.0, 0.6))),
        ] {
            change(&mut e);
            finish(&mut s, &mut e, &mut r);
            let reused = r.read_canvas();
            let (fresh, _) = r.capture(&e.frame_params(), 1).unwrap();
            let d = differing(&reused, &fresh);
            assert!(d < 0.002, "{what}: {:.3}% of pixels differ", d * 100.0);
        }
    }
}
