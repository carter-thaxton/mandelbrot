//! Metal backend: the only module that touches objc2 / Metal / Core Animation.
//!
//! A compute kernel writes directly into the CAMetalLayer's drawable texture
//! (so the layer is created with `framebufferOnly = false`); no render pass or
//! vertex pipeline is needed.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

use super::{GpuParams, Renderer, Result};
use crate::app::FrameParams;

const SHADER_SOURCE: &str = include_str!("mandelbrot.metal");

type Device = ProtocolObject<dyn MTLDevice>;

pub struct MetalRenderer {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// None for headless (capture-only) renderers.
    layer: Option<Retained<CAMetalLayer>>,
}

impl MetalRenderer {
    pub fn new<W: HasWindowHandle>(window: Option<&W>) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device.newCommandQueue().ok_or("failed to create command queue")?;
        let pipeline = build_pipeline(&device)?;
        let layer = window.map(|w| attach_layer(w, &device)).transpose()?;
        Ok(Self { device, queue, pipeline, layer })
    }

    fn encode(
        &self,
        cmd: &ProtocolObject<dyn MTLCommandBuffer>,
        target: &ProtocolObject<dyn MTLTexture>,
        params: &FrameParams,
    ) -> Result<()> {
        let gpu = GpuParams::from(params);
        let enc = cmd.computeCommandEncoder().ok_or("failed to create compute encoder")?;
        enc.setComputePipelineState(&self.pipeline);
        // SAFETY: texture index 0 / buffer index 0 match the kernel signature,
        // and setBytes copies `gpu` before returning.
        unsafe {
            enc.setTexture_atIndex(Some(target), 0);
            enc.setBytes_length_atIndex(
                NonNull::from(&gpu).cast::<c_void>(),
                size_of::<GpuParams>(),
                0,
            );
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
        let mut params = *params;
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
