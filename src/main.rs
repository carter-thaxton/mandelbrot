mod app;
mod bigfixed;
mod perturb;
mod progressive;
mod render;

use std::time::{SystemTime, UNIX_EPOCH};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use app::{Explorer, Precision, PrecisionMode};
use progressive::Scheduler;
use render::Renderer;

const HELP: &str = "\
Mandelbrot explorer
  drag / two-finger scroll   pan
  pinch / cmd+scroll / wheel zoom at cursor
  + / -                      zoom in / out (center)
  arrow keys                 pan
  [ / ]                      fewer / more iterations (switches to manual)
  a                          automatic iteration limit on / off
  , / .                      color density down / up
  p / shift+p                next / previous palette
  m                          precision: auto -> f32 -> df64 -> perturb
  r                          reset view
  s                          save screenshot (PNG in current dir)
  c                          print location (full precision) as a --capture command
  esc / q                    quit

Headless: mandelbrot --capture out.png [--size WxH] [--center X,Y] [--zoom Z] [--iters N]
          [--density D] [--precision f32|df|perturb] [--no-skip]";

struct App {
    window: Option<Window>,
    renderer: Option<Box<dyn Renderer>>,
    explorer: Explorer,
    modifiers: ModifiersState,
    scheduler: Scheduler,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            explorer: Explorer::new(1, 1),
            modifiers: ModifiersState::empty(),
            scheduler: Scheduler::default(),
        }
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Something changed: the scheduler works out what to reuse and redo.
    fn invalidate(&mut self) {
        self.request_redraw();
    }

    fn handle_key(&mut self, key: &Key, event_loop: &ActiveEventLoop) {
        let e = &mut self.explorer;
        match key {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::ArrowLeft) => e.pan_fraction(0.1, 0.0),
            Key::Named(NamedKey::ArrowRight) => e.pan_fraction(-0.1, 0.0),
            Key::Named(NamedKey::ArrowUp) => e.pan_fraction(0.0, 0.1),
            Key::Named(NamedKey::ArrowDown) => e.pan_fraction(0.0, -0.1),
            Key::Character(s) => match s.as_str() {
                "q" | "Q" => event_loop.exit(),
                "=" | "+" => e.zoom_at_center(2.0),
                "-" | "_" => e.zoom_at_center(0.5),
                "[" => e.scale_iter_limit(1.0 / 1.5),
                "]" => e.scale_iter_limit(1.5),
                "a" | "A" => e.auto_iter = !e.auto_iter,
                "," | "<" => e.color_density /= 1.25,
                "." | ">" => e.color_density *= 1.25,
                "p" => e.cycle_palette(true),
                "P" => e.cycle_palette(false),
                "m" | "M" => e.cycle_precision_mode(),
                "r" | "R" => {
                    e.reset_view();
                    e.color_density = 1.0;
                }
                "s" | "S" => self.save_screenshot(),
                "c" | "C" => {
                    let (x, y) = e.center_strings();
                    println!("--center {x},{y} --zoom {:.6e}", e.zoom());
                    return;
                }
                _ => return,
            },
            _ => return,
        }
        self.invalidate();
    }

    fn save_screenshot(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else { return };
        let params = self.explorer.frame_params();
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let path = format!("mandelbrot-{secs}.png");
        match renderer.capture(&params, 1).and_then(|(px, _)| write_png(&path, &params, &px)) {
            Ok(()) => println!("saved {path}"),
            Err(err) => eprintln!("screenshot failed: {err}"),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Mandelbrot")
            .with_inner_size(LogicalSize::new(1200.0, 800.0));
        let window = event_loop.create_window(attrs).expect("failed to create window");
        let renderer = render::create(&window).unwrap_or_else(|e| {
            eprintln!("renderer init failed: {e}");
            std::process::exit(1);
        });
        let size = window.inner_size();
        self.explorer = Explorer::new(size.width, size.height);
        self.renderer = Some(renderer);
        self.renderer.as_mut().unwrap().resize(size.width, size.height, window.scale_factor());
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.explorer.resize(size.width, size.height);
                if let (Some(r), Some(w)) = (self.renderer.as_mut(), self.window.as_ref()) {
                    r.resize(size.width, size.height, w.scale_factor());
                }
                self.scheduler.reset();
                self.invalidate();
            }

            WindowEvent::ScaleFactorChanged { .. } => self.invalidate(),

            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.handle_key(&event.logical_key, event_loop);
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.explorer.cursor_moved(position.x, position.y) {
                    self.invalidate();
                }
            }

            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed => self.explorer.begin_drag(),
                    ElementState::Released => self.explorer.end_drag(),
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let zoom_modifier = self.modifiers.super_key() || self.modifiers.control_key();
                match delta {
                    // Trackpad: two-finger scroll pans; with cmd/ctrl it zooms.
                    MouseScrollDelta::PixelDelta(d) if !zoom_modifier => {
                        self.explorer.pan_pixels(d.x, d.y);
                    }
                    MouseScrollDelta::PixelDelta(d) => {
                        self.explorer.zoom_at_cursor((d.y * 0.01).exp());
                    }
                    // Mouse wheel: zoom.
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.explorer.zoom_at_cursor(1.25f64.powf(y as f64));
                    }
                }
                self.invalidate();
            }

            WindowEvent::PinchGesture { delta, .. } => {
                self.explorer.zoom_at_cursor(delta.exp());
                self.invalidate();
            }

            WindowEvent::RedrawRequested => {
                let Some(r) = self.renderer.as_mut() else { return };
                // One budget-sized slice of work per display refresh; input
                // that arrives in between invalidates the rest.
                match self.scheduler.step(&mut self.explorer, r.as_mut()) {
                    Ok(true) => self.request_redraw(),
                    Ok(false) => {}
                    Err(err) => eprintln!("render failed: {err}"),
                }
                if let Some(w) = &self.window {
                    let hud = self.explorer.hud(Some(self.scheduler.work_ms()), self.scheduler.progress());
                    w.set_title(&hud);
                }
            }

            _ => {}
        }
    }
}

fn write_png(path: &str, params: &app::FrameParams, rgba: &[u8]) -> render::Result<()> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), params.width, params.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())
}

/// `--capture out.png [--size WxH] [--center X,Y] [--zoom Z] [--iters N] [--density D] [--precision ...]`
fn run_capture(args: &[String]) -> render::Result<()> {
    let mut path = None;
    let (mut w, mut h) = (1600u32, 1000u32);
    let mut center: Option<(String, String)> = None;
    let mut zoom = 1.0f64;
    let mut precision = PrecisionMode::Auto;
    let mut iters: Option<u32> = None;
    let mut density = 1.0f32;
    let mut series_skip = true;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut val = || it.next().cloned().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--capture" => path = Some(val()?),
            "--size" => {
                let v = val()?;
                let (a, b) = v.split_once('x').ok_or("size must be WxH")?;
                w = a.parse().map_err(|_| "bad width")?;
                h = b.parse().map_err(|_| "bad height")?;
            }
            "--center" => {
                let v = val()?;
                let (a, b) = v.split_once(',').ok_or("center must be X,Y")?;
                center = Some((a.to_string(), b.to_string()));
            }
            "--zoom" => zoom = val()?.parse().map_err(|_| "bad zoom")?,
            "--iters" => iters = Some(val()?.parse().map_err(|_| "bad iters")?),
            "--density" => density = val()?.parse().map_err(|_| "bad density")?,
            "--no-skip" => series_skip = false,
            "--precision" => {
                precision = match val()?.as_str() {
                    "f32" => PrecisionMode::Forced(Precision::F32),
                    "df" | "df64" => PrecisionMode::Forced(Precision::DoubleFloat),
                    "perturb" | "pt" => PrecisionMode::Forced(Precision::Perturbation),
                    _ => PrecisionMode::Auto,
                }
            }
            other => return Err(format!("unknown argument {other}\n\n{HELP}")),
        }
    }
    let path = path.ok_or("missing --capture path")?;

    let mut explorer = Explorer::new(w, h);
    explorer.precision_mode = precision;
    if let Some((x, y)) = center {
        explorer.set_center_str(&x, &y).ok_or("bad center")?;
    }
    explorer.zoom_at_center(zoom);
    explorer.color_density = density;
    explorer.series_skip = series_skip;
    explorer.iter_limit = explorer.estimate_iter();
    if let Some(n) = iters {
        explorer.set_iter_limit(n);
    }

    let mut renderer = render::create_headless()?;
    let start = std::time::Instant::now();
    // Settle the automatic iteration limit on cheap coarse passes, as the
    // interactive preview does.
    for _ in 0..8 {
        let (_, stats) = renderer.capture(&explorer.frame_params(), 4)?;
        if std::env::var_os("MANDEL_DEBUG_ITERS").is_some() {
            eprintln!("settle: limit {} -> {stats:?}", explorer.iter_limit);
        }
        if !explorer.observe(&stats, true) {
            break;
        }
    }
    let settle_ms = start.elapsed().as_secs_f64() * 1000.0;
    let params = explorer.frame_params();
    let (pixels, stats) = renderer.capture(&params, 1)?;
    println!(
        "{}  (settle {settle_ms:.0} ms, gpu {:.1} ms, total {:.1} ms)",
        explorer.hud(None, None),
        stats.gpu_ms,
        start.elapsed().as_secs_f64() * 1000.0
    );
    write_png(&path, &params, &pixels)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{HELP}");
        return;
    }
    if !args.is_empty() {
        if let Err(err) = run_capture(&args) {
            eprintln!("{err}");
            std::process::exit(1);
        }
        return;
    }

    println!("{HELP}");
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.run_app(&mut App::new()).expect("event loop failed");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fraction of pixels whose color differs noticeably between two images.
    fn differing(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len() / 4;
        let bad = a
            .chunks(4)
            .zip(b.chunks(4))
            .filter(|(x, y)| (0..3).any(|i| x[i].abs_diff(y[i]) > 16))
            .count();
        bad as f64 / n as f64
    }

    /// Rendering against a reused, off-center reference (as happens while
    /// panning) must match rendering against a fresh centered one.
    #[test]
    fn gpu_off_center_reference_matches_fresh() {
        let mut renderer = render::create_headless().expect("Metal device");
        for zoom in [1e8, 1e40] {
            let mut panned = Explorer::new(320, 200);
            // c = i: a boundary point with smooth surroundings at every depth,
            // so any offset error shows up as a visibly shifted image.
            panned.set_center_str("0", "1");
            panned.zoom_at_center(zoom);
            panned.set_iter_limit(4000);
            panned.frame_params(); // creates the reference at this center
            panned.pan_pixels(120.0, -70.0);
            let reused = panned.frame_params();
            assert!(reused.ref_offset != (0.0, 0.0), "reference should have been reused");

            let mut fresh = Explorer::new(320, 200);
            fresh.center = panned.center.clone();
            fresh.scale = panned.scale;
            fresh.set_iter_limit(4000);
            let centered = fresh.frame_params();
            // Only the truncation of the center to the needed precision.
            let off = centered.ref_offset.0.abs().max(centered.ref_offset.1.abs());
            assert!(off < centered.scale * 1e-6, "{off:e}");

            let (a, _) = renderer.capture(&reused, 1).unwrap();
            let (b, _) = renderer.capture(&centered, 1).unwrap();
            if let Ok(dir) = std::env::var("MANDEL_TEST_DUMP") {
                write_png(&format!("{dir}/reused_{zoom:e}.png"), &reused, &a).unwrap();
                write_png(&format!("{dir}/fresh_{zoom:e}.png"), &centered, &b).unwrap();
            }
            let d = differing(&a, &b);
            // Different references round differently right at the boundary.
            assert!(d < 0.01, "zoom {zoom}: {:.2}% of pixels differ", d * 100.0);
        }
    }
}
