//! wgpu backend (Vulkan on Linux): the only module that touches wgpu.
//!
//! Mirrors the Metal backend, with the shader ported to WGSL:
//! - `mandelbrot` (compute) writes each pixel's smooth iteration count (mu)
//!   into a persistent single-channel float canvas;
//! - `colorize` (a fullscreen-triangle render pass) turns mu into colors,
//!   into the window's surface when presenting, or into an RGBA texture for
//!   snapshots and captures;
//! - `reproject` (compute) resamples the canvas into a spare texture when
//!   the view moves, and the two are swapped.
//!
//! Every submission except `present` is waited on, as with Metal: the app's
//! scheduler keeps dispatches small, and it keeps readback trivial.

use std::sync::Arc;
use std::time::Instant;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use wgpu::util::DeviceExt;

use super::{
    BatchStats, ESCAPE_BINS, EscapeHistogram, FrameStats, GpuColors, GpuParams, GpuReprojection, GpuTileStats, Output,
    Renderer, Reprojection, Result, TILE, TileBatch, sample_grids,
};
use crate::app::{Colors, FrameParams};
use crate::perturb::{RefPoint, ReferenceOrbit};

const SHADER_SOURCE: &str = include_str!("mandelbrot.wgsl");

/// Threads per `mandelbrot` workgroup. Must match GROUP_THREADS in the shader.
const GROUP_THREADS: u32 = 256;
/// Most workgroups per dispatch dimension WebGPU guarantees.
const MAX_GROUPS_PER_DIM: u32 = 65535;

/// The canvas format: one f32 of smooth iteration count per pixel.
const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Snapshots and captures; colorize writes unorm values, as into the surface.
const RGBA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Bytes of the escape histogram.
const ESCAPES_SIZE: u64 = (ESCAPE_BINS * 4) as u64;
/// Readback layout: two GPU timestamps, the escape histogram, then one
/// [`GpuTileStats`] per tile.
const READBACK_ESCAPES: u64 = 16;
const READBACK_TILES: u64 = READBACK_ESCAPES + ESCAPES_SIZE;

/// The window surface and how it's configured.
struct Window {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// Colorize into the surface (its format, viewed without sRGB encoding).
    pipeline: wgpu::RenderPipeline,
}

/// GPU timestamps around each `mandelbrot` dispatch, if the device has them.
struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period: f64,
}

pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    reproject_pipeline: wgpu::ComputePipeline,
    /// Colorize into an [`RGBA_FORMAT`] texture.
    colorize_pipeline: wgpu::RenderPipeline,
    window: Option<Window>,
    timer: Option<Timer>,
    /// Persistent mu image the kernel fills progressively; colorized to present.
    canvas: Option<wgpu::Texture>,
    /// Same size as `canvas`: the target of `reproject`, then swapped in.
    spare: Option<wgpu::Texture>,
    /// Reference orbit for perturbation, uploaded only when the Arc changes.
    reference: Option<(Arc<ReferenceOrbit>, wgpu::Buffer)>,
    /// Bound as the reference orbit when there is none.
    empty_reference: wgpu::Buffer,
    params: wgpu::Buffer,
    colors: wgpu::Buffer,
    reprojection: wgpu::Buffer,
    /// The batch's tile list, grown as batches need.
    tiles: wgpu::Buffer,
    /// Per-tile kernel statistics, grown as batches need.
    tile_stats: wgpu::Buffer,
    /// The batch's escape-count histogram (cleared before each dispatch).
    escapes: wgpu::Buffer,
    /// Where timestamps, the histogram, and tile stats are copied to be read.
    readback: wgpu::Buffer,
}

impl WgpuRenderer {
    pub fn new<W: HasWindowHandle + HasDisplayHandle>(window: Option<&W>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = window
            .map(|w| {
                let target = wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(w.display_handle().map_err(|e| e.to_string())?.as_raw()),
                    raw_window_handle: w.window_handle().map_err(|e| e.to_string())?.as_raw(),
                };
                // SAFETY: the window outlives the renderer (the app drops the
                // renderer first), and with it this surface.
                unsafe { instance.create_surface_unsafe(target) }.map_err(|e| e.to_string())
            })
            .transpose()?;
        let adapter = choose_adapter(&instance, surface.as_ref())?;
        // Long reference orbits need big storage buffers: take whatever the
        // adapter allows.
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mandelbrot"),
            required_features: if timestamps { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() },
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("failed to create device: {e}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = compute("mandelbrot");
        let reproject_pipeline = compute("reproject");
        let colorize_pipeline = build_colorize(&device, &module, RGBA_FORMAT);

        let window = surface
            .map(|surface| {
                let caps = surface.get_capabilities(&adapter);
                let format = *caps.formats.first().ok_or("the surface supports no formats")?;
                // Write colors unencoded, as Metal does into its BGRA8Unorm
                // layer: view an sRGB surface through its linear twin.
                let view_format = format.remove_srgb_suffix();
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format,
                    width: 1,
                    height: 1,
                    present_mode: wgpu::PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
                        wgpu::CompositeAlphaMode::Opaque
                    } else {
                        caps.alpha_modes[0]
                    },
                    view_formats: if view_format != format { vec![view_format] } else { vec![] },
                    color_space: Default::default(),
                };
                let pipeline = build_colorize(&device, &module, view_format);
                Ok::<_, String>(Window { surface, config, pipeline })
            })
            .transpose()?;

        let timer = timestamps.then(|| Timer {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("dispatch time"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            period: queue.get_timestamp_period() as f64,
        });

        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        use wgpu::BufferUsages as U;
        let empty_reference = buffer("empty reference", size_of::<RefPoint>() as u64, U::STORAGE);
        let params = buffer("params", size_of::<GpuParams>() as u64, U::UNIFORM | U::COPY_DST);
        let colors = buffer("colors", size_of::<GpuColors>() as u64, U::UNIFORM | U::COPY_DST);
        let reprojection = buffer("reprojection", 32, U::UNIFORM | U::COPY_DST);
        let tiles = buffer("tiles", 64 * 8, U::STORAGE | U::COPY_DST);
        let tile_stats = buffer("tile stats", 64 * size_of::<GpuTileStats>() as u64, U::STORAGE | U::COPY_SRC);
        let escapes = buffer("escapes", ESCAPES_SIZE, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let readback = buffer("readback", READBACK_TILES + 64 * size_of::<GpuTileStats>() as u64, U::MAP_READ | U::COPY_DST);

        Ok(Self {
            device,
            queue,
            pipeline,
            reproject_pipeline,
            colorize_pipeline,
            window,
            timer,
            canvas: None,
            spare: None,
            reference: None,
            empty_reference,
            params,
            colors,
            reprojection,
            tiles,
            tile_stats,
            escapes,
            readback,
        })
    }

    fn reference_buffer(&mut self, params: &FrameParams) -> wgpu::Buffer {
        let Some(orbit) = &params.reference else {
            return self.empty_reference.clone();
        };
        if let Some((cached, buffer)) = &self.reference
            && Arc::ptr_eq(cached, orbit)
        {
            return buffer.clone();
        }
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reference orbit"),
            contents: as_bytes(&orbit.points),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.reference = Some((orbit.clone(), buffer.clone()));
        buffer
    }

    /// Record the Mandelbrot kernel for `batch` of `target`. Returns each
    /// tile's grid of samples inside the canvas.
    fn encode(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        params: &FrameParams,
        batch: &TileBatch,
    ) -> Vec<[u32; 2]> {
        let n = batch.tiles.len() as u64;
        let gpu = GpuParams::new(params, batch);
        enc.clear_buffer(&self.escapes, 0, None);
        if n == 0 {
            return vec![];
        }
        let per_side = TILE / gpu.step;
        let reference = self.reference_buffer(params);
        let stats_size = n * size_of::<GpuTileStats>() as u64;
        grow(&self.device, &mut self.tiles, n * 8);
        grow(&self.device, &mut self.tile_stats, stats_size);
        grow(&self.device, &mut self.readback, READBACK_TILES + stats_size);
        self.queue.write_buffer(&self.params, 0, as_bytes(&[gpu]));
        self.queue.write_buffer(&self.tiles, 0, as_bytes(&batch.tiles));

        let view = target.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, self.params.as_entire_binding()),
                entry(1, reference.as_entire_binding()),
                entry(2, self.tile_stats.as_entire_binding()),
                // Exactly the batch, so the shader's arrayLength is its tile count.
                entry(3, wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.tiles, offset: 0, size: wgpu::BufferSize::new(n * 8) })),
                entry(4, self.escapes.as_entire_binding()),
                entry(5, wgpu::BindingResource::TextureView(&view)),
            ],
        });
        // Whole tiles per workgroup; spread the workgroups over two
        // dimensions when there are too many for one.
        let groups = (n as u32).div_ceil(GROUP_THREADS / (per_side * per_side));
        let gx = groups.min(MAX_GROUPS_PER_DIM);
        let gy = groups.div_ceil(gx);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mandelbrot"),
                timestamp_writes: self.timer.as_ref().map(|t| wgpu::ComputePassTimestampWrites {
                    query_set: &t.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if let Some(t) = &self.timer {
            enc.resolve_query_set(&t.queries, 0..2, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &self.readback, 0, 16);
        }
        enc.copy_buffer_to_buffer(&self.tile_stats, 0, &self.readback, READBACK_TILES, stats_size);
        sample_grids(&batch.tiles, params.width, params.height, gpu.step)
    }

    /// Submit `enc`, wait for it, and collect timing plus the kernel's
    /// statistics for tiles with sample grids `grids` (from [`Self::encode`]).
    fn finish(&self, mut enc: wgpu::CommandEncoder, grids: &[[u32; 2]]) -> Result<BatchStats> {
        enc.copy_buffer_to_buffer(&self.escapes, 0, &self.readback, READBACK_ESCAPES, ESCAPES_SIZE);
        let start = Instant::now();
        self.queue.submit([enc.finish()]);
        let size = READBACK_TILES + (grids.len() * size_of::<GpuTileStats>()) as u64;
        let bytes = read_buffer(&self.device, &self.readback, size)?;
        let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

        let gpu_ms = match &self.timer {
            Some(t) if !grids.is_empty() => {
                let [begin, end]: [u64; 2] = from_bytes(&bytes[..16]);
                end.wrapping_sub(begin) as f64 * t.period / 1e6
            }
            // No timestamps: the round trip is the next best measure.
            _ => wall_ms,
        };
        let tiles = (0..grids.len())
            .map(|i| {
                let at = READBACK_TILES as usize + i * size_of::<GpuTileStats>();
                let s: GpuTileStats = from_bytes(&bytes[at..at + size_of::<GpuTileStats>()]);
                s.to_tile_stats(grids[i])
            })
            .collect();
        let mut stats = BatchStats::of_tiles(gpu_ms, tiles);
        let escapes = &bytes[READBACK_ESCAPES as usize..READBACK_TILES as usize];
        stats.total.escapes = EscapeHistogram(from_bytes(escapes));
        Ok(stats)
    }

    fn new_texture(&self, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn new_canvas(&self, w: u32, h: u32) -> wgpu::Texture {
        use wgpu::TextureUsages as U;
        self.new_texture("canvas", w, h, CANVAS_FORMAT, U::STORAGE_BINDING | U::TEXTURE_BINDING)
    }

    /// Record coloring the mu texture `mu` into `out` with `pipeline`.
    fn encode_colorize(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        mu: &wgpu::Texture,
        out: &wgpu::TextureView,
        colors: &Colors,
    ) {
        self.queue.write_buffer(&self.colors, 0, as_bytes(&[GpuColors::new(colors)]));
        let mu_view = mu.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("colorize"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[entry(0, self.colors.as_entire_binding()), entry(1, wgpu::BindingResource::TextureView(&mu_view))],
        });
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("colorize"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out,
                depth_slice: None,
                resolve_target: None,
                // Every pixel is drawn, so the old contents don't matter.
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Color `mu` and read the result back as RGBA8 rows.
    fn colorized_rgba(&self, mu: &wgpu::Texture, colors: &Colors) -> Result<Vec<u8>> {
        let (w, h) = (mu.width(), mu.height());
        use wgpu::TextureUsages as U;
        let rgba = self.new_texture("rgba", w, h, RGBA_FORMAT, U::RENDER_ATTACHMENT | U::COPY_SRC);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode_colorize(&mut enc, &self.colorize_pipeline, mu, &rgba.create_view(&Default::default()), colors);

        // Buffer rows must be padded to COPY_BYTES_PER_ROW_ALIGNMENT.
        let row = w * 4;
        let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = padded as u64 * h as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgba readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            rgba.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
            },
            rgba.size(),
        );
        self.queue.submit([enc.finish()]);
        let bytes = read_buffer(&self.device, &buffer, size)?;
        Ok(bytes.chunks(padded as usize).flat_map(|r| &r[..row as usize]).copied().collect())
    }
}

impl Renderer for WgpuRenderer {
    fn resize(&mut self, width: u32, height: u32, _scale_factor: f64) {
        if let Some(win) = &mut self.window
            && width > 0
            && height > 0
        {
            win.config.width = width;
            win.config.height = height;
            win.surface.configure(&self.device, &win.config);
        }
        self.canvas = Some(self.new_canvas(width, height));
        self.spare = Some(self.new_canvas(width, height));
    }

    fn reproject(&mut self, xf: &Reprojection) -> Result<()> {
        let (Some(canvas), Some(spare)) = (self.canvas.take(), self.spare.take()) else {
            return Err("reproject() before resize()".into());
        };
        // The uniform is 24 bytes; the buffer is padded to 32.
        self.queue.write_buffer(&self.reprojection, 0, as_bytes(&[GpuReprojection::new(xf)]));
        let src = canvas.create_view(&Default::default());
        let dst = spare.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reproject"),
            layout: &self.reproject_pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, self.reprojection.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&src)),
                entry(2, wgpu::BindingResource::TextureView(&dst)),
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.reproject_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(spare.width().div_ceil(16), spare.height().div_ceil(16), 1);
        }
        self.queue.submit([enc.finish()]);
        wait(&self.device)?;
        (self.canvas, self.spare) = (Some(spare), Some(canvas));
        Ok(())
    }

    fn draw(&mut self, params: &FrameParams, batch: &TileBatch) -> Result<BatchStats> {
        let canvas = self.canvas.clone().ok_or("draw() before resize()")?;
        // The canvas is only as big as the last resize; never dispatch past it.
        let mut params = params.clone();
        params.width = params.width.min(canvas.width());
        params.height = params.height.min(canvas.height());

        let mut enc = self.device.create_command_encoder(&Default::default());
        let grids = self.encode(&mut enc, &canvas, &params, batch);
        self.finish(enc, &grids)
    }

    fn present(&mut self, colors: &Colors) -> Result<()> {
        // Headless renderers (tests, captures) have nowhere to present to.
        let Some(win) = &self.window else { return Ok(()) };
        let canvas = self.canvas.as_ref().ok_or("present() before resize()")?;
        let frame = match win.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            // Minimized, hidden, or mid-resize: skip this frame.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            wgpu::CurrentSurfaceTexture::Outdated => {
                win.surface.configure(&self.device, &win.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => return Err("the window surface was lost".into()),
            wgpu::CurrentSurfaceTexture::Validation => return Err("surface validation error".into()),
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(win.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode_colorize(&mut enc, &win.pipeline, canvas, &view, colors);
        // No wait: later work on this queue is ordered after the coloring.
        self.queue.submit([enc.finish()]);
        self.queue.present(frame);
        Ok(())
    }

    fn snapshot(&mut self, colors: &Colors) -> Result<Vec<u8>> {
        let canvas = self.canvas.clone().ok_or("snapshot() before resize()")?;
        self.colorized_rgba(&canvas, colors)
    }

    fn capture(&mut self, params: &FrameParams, step: u32, colors: &Colors) -> Result<(Vec<u8>, FrameStats)> {
        let mu = self.new_canvas(params.width, params.height);
        let batch = TileBatch::all(params.width, params.height, step, Output::Pixels);
        let mut enc = self.device.create_command_encoder(&Default::default());
        let grids = self.encode(&mut enc, &mu, params, &batch);
        let stats = self.finish(enc, &grids)?.total;
        Ok((self.colorized_rgba(&mu, colors)?, stats))
    }
}

/// The best adapter that can run compute shaders (and present to `surface`,
/// if any): a real GPU over a software one. `WGPU_ADAPTER_NAME` (a
/// case-insensitive substring) picks one by name instead.
///
/// In a VM, the host GPU may only be passed through as OpenGL without
/// compute shaders (e.g. virgl), leaving only a CPU rasterizer like llvmpipe.
fn choose_adapter(instance: &wgpu::Instance, surface: Option<&wgpu::Surface>) -> Result<wgpu::Adapter> {
    let wanted = std::env::var("WGPU_ADAPTER_NAME").ok().map(|n| n.to_lowercase());
    let rank = |t: wgpu::DeviceType| match t {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    let mut seen = vec![];
    let mut usable = vec![];
    for adapter in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
        let info = adapter.get_info();
        let compute = adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
        let presents = surface.is_none_or(|s| adapter.is_surface_supported(s));
        let named = wanted.as_ref().is_none_or(|n| info.name.to_lowercase().contains(n));
        seen.push(format!("{} ({:?}{})", info.name, info.backend, if compute { "" } else { ", no compute shaders" }));
        if compute && presents && named {
            usable.push(adapter);
        }
    }
    usable.sort_by_key(|a| rank(a.get_info().device_type));
    let adapter = usable.into_iter().next().ok_or_else(|| format!("no usable GPU adapter among: {}", seen.join(", ")))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu && wanted.is_none() && surface.is_some() {
        eprintln!("note: rendering on the CPU with {} ({:?}); no GPU with compute shaders found", info.name, info.backend);
    }
    Ok(adapter)
}

fn build_colorize(device: &wgpu::Device, module: &wgpu::ShaderModule, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("colorize"),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("colorize"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

/// Replace `buffer` with a bigger one (same usage) if it holds under `size` bytes.
fn grow(device: &wgpu::Device, buffer: &mut wgpu::Buffer, size: u64) {
    if buffer.size() < size {
        *buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.next_power_of_two(),
            usage: buffer.usage(),
            mapped_at_creation: false,
        });
    }
}

/// Block until all submitted work is done.
fn wait(device: &wgpu::Device) -> Result<()> {
    device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ()).map_err(|e| format!("GPU error: {e}"))
}

/// Wait for the GPU, then copy out the first `size` bytes of `buffer`.
fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
    let slice = buffer.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    wait(device)?;
    rx.recv().map_err(|e| e.to_string())?.map_err(|e| format!("readback failed: {e}"))?;
    let bytes = slice.get_mapped_range().map_err(|e| e.to_string())?.to_vec();
    buffer.unmap();
    Ok(bytes)
}

/// Marks the plain-old-data types uploaded and read back as raw bytes.
///
/// # Safety
/// Implementors must be `repr(C)` with no padding, and every bit pattern
/// must be a valid value (integers and floats only).
unsafe trait Pod: Copy {}
// SAFETY: all repr(C) structs of 4-byte integers and floats with explicit
// padding fields (their layouts are pinned by tests in render/mod.rs and
// perturb.rs), or arrays of integers.
unsafe impl Pod for GpuParams {}
unsafe impl Pod for GpuColors {}
unsafe impl Pod for GpuReprojection {}
unsafe impl Pod for GpuTileStats {}
unsafe impl Pod for super::TileJob {}
unsafe impl Pod for RefPoint {}
unsafe impl Pod for [u64; 2] {}
unsafe impl Pod for [u32; ESCAPE_BINS] {}

fn as_bytes<T: Pod>(data: &[T]) -> &[u8] {
    // SAFETY: `T: Pod` has no padding, so every byte is initialized.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

fn from_bytes<T: Pod>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    // SAFETY: the length matches and any bit pattern is a valid `T: Pod`.
    unsafe { bytes.as_ptr().cast::<T>().read_unaligned() }
}
