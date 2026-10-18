//! Metal backend: the only module that touches objc2 / Metal / Core Animation.
//!
//! A compute kernel writes directly into the CAMetalLayer's drawable texture
//! (so the layer is created with `framebufferOnly = false`); no render pass or
//! vertex pipeline is needed.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::{GpuParams, Renderer, Result};
use crate::app::FrameParams;
use crate::perturb::ReferenceOrbit;

const SHADER_SOURCE: &str = include_str!("mandelbrot.metal");

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = ProtocolObject<dyn MTLBuffer>;

pub struct MetalRenderer {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// None for headless (capture-only) renderers.
    layer: Option<Retained<CAMetalLayer>>,
    /// Reference orbit for perturbation, uploaded only when the Arc changes.
    reference: Option<(Arc<ReferenceOrbit>, Retained<Buffer>)>,
    /// Bound at buffer(1) when there is no reference, to keep validation happy.
    empty_buffer: Retained<Buffer>,
}

impl MetalRenderer {
    pub fn new<W: HasWindowHandle>(window: Option<&W>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("failed to create command queue")?;
        let pipeline = build_pipeline(&device)?;
        let layer = window.map(|w| attach_layer(w, &device)).transpose()?;
        let empty_buffer = new_buffer(&device, &[0u8; 64])?;
        Ok(Self { device, queue, pipeline, layer, reference: None, empty_buffer })
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

    fn encode(
        &mut self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        target: &ProtocolObject<dyn MTLTexture>,
        params: &FrameParams,
    ) -> Result<()> {
        let gpu = GpuParams::from(params);
        let reference = self.reference_buffer(params)?;
        let enc = cmd.computeCommandEncoder().ok_or("failed to create compute encoder")?;
        enc.setComputePipelineState(&self.pipeline);
        // SAFETY: texture 0 / buffers 0 and 1 match the kernel signature,
        // setBytes copies `gpu` before returning, and the kernel reads at most
        // `ref_len` points from the reference buffer.
        unsafe {
            enc.setTexture_atIndex(Some(target), 0);
            enc.setBytes_length_atIndex(
                NonNull::from(&gpu).cast::<c_void>(),
                size_of::<GpuParams>(),
                0,
            );
            enc.setBuffer_offset_atIndex(Some(&reference), 0, 1);
        }
        let w = self.pipeline.threadExecutionWidth();
        let h = (self.pipeline.maxTotalThreadsPerThreadgroup() / w).max(1);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: params.width as usize, height: params.height as usize, depth: 1 },
            MTLSize { width: w, height: h, depth: 1 },
        );
        enc.endEncoding();
        Ok(())
    }
}

impl Renderer for MetalRenderer {
    fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        if let Some(layer) = &self.layer {
            layer.setContentsScale(scale_factor);
            layer.setDrawableSize(CGSize { width: width as f64, height: height as f64 });
        }
    }

    fn render(&mut self, params: &FrameParams) -> Result<Option<f64>> {
        let layer = self.layer.as_ref().ok_or("render() on a headless renderer")?;
        let Some(drawable) = layer.nextDrawable() else {
            return Ok(None); // e.g. window minimized; skip this frame
        };
        let texture = drawable.texture();
        // The layer may lag a resize by a frame; never dispatch past the texture.
        let mut params = params.clone();
        params.width = params.width.min(texture.width() as u32);
        params.height = params.height.min(texture.height() as u32);

        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        self.encode(&cmd, &texture, &params)?;
        cmd.presentDrawable(ProtocolObject::from_ref(&*drawable));
        cmd.commit();
        // Rendering is on demand, so waiting keeps things simple and gives an
        // honest GPU timing for the HUD.
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(format!("GPU error: {}", err.localizedDescription()));
        }
        Ok(Some((cmd.GPUEndTime() - cmd.GPUStartTime()) * 1000.0))
    }

    fn capture(&mut self, params: &FrameParams) -> Result<Vec<u8>> {
        let (w, h) = (params.width as usize, params.height as usize);
        // SAFETY: plain descriptor construction with valid arguments.
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                w,
                h,
                false,
            )
        };
        desc.setUsage(MTLTextureUsage::ShaderWrite);
        desc.setStorageMode(MTLStorageMode::Shared);
        let texture = self.device.newTextureWithDescriptor(&desc).ok_or("failed to create texture")?;

        let cmd = self.queue.commandBuffer().ok_or("failed to create command buffer")?;
        self.encode(&cmd, &texture, params)?;
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(format!("GPU error: {}", err.localizedDescription()));
        }

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
        Ok(pixels)
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

fn build_pipeline(device: &Device) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let options = MTLCompileOptions::new();
    // df64 arithmetic depends on exact IEEE rounding; fast-math would break it.
    options.setMathMode(MTLMathMode::Safe);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SHADER_SOURCE), Some(&options))
        .map_err(|e| format!("shader compile failed: {}", e.localizedDescription()))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("mandelbrot"))
        .ok_or("kernel `mandelbrot` not found")?;
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
    layer.setFramebufferOnly(false); // the compute kernel writes to it
    layer.setAutoresizingMask(
        CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable,
    );

    view.setWantsLayer(true);
    let root = view.layer().ok_or("view has no backing layer")?;
    layer.setFrame(root.bounds());
    root.addSublayer(&layer);
    Ok(layer)
}
