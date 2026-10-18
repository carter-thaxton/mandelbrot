mod app;
mod bigfixed;
mod perturb;
mod render;

use std::time::{SystemTime, UNIX_EPOCH};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowId};

use app::{Explorer, Precision, PrecisionMode};
use render::Renderer;

const HELP: &str = "\
Mandelbrot explorer
  drag / two-finger scroll   pan
  pinch / cmd+scroll / wheel zoom at cursor
  + / -                      zoom in / out (center)
  arrow keys                 pan
  [ / ]                      fewer / more iterations
  , / .                      color density down / up
  p / shift+p                next / previous palette
  m                          precision: auto -> f32 -> df64 -> perturb
  r                          reset view
  s                          save screenshot (PNG in current dir)
  c                          print location (full precision) as a --capture command
  esc / q                    quit

Headless: mandelbrot --capture out.png [--size WxH] [--center X,Y] [--zoom Z] [--iters N]
          [--density D] [--precision f32|df|perturb]";

struct App {
    window: Option<Window>,
    renderer: Option<Box<dyn Renderer>>,
    explorer: Explorer,
    modifiers: ModifiersState,
    last_gpu_ms: Option<f64>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            explorer: Explorer::new(1, 1),
            modifiers: ModifiersState::empty(),
            last_gpu_ms: None,
        }
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
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
                "[" => e.iter_multiplier /= 1.5,
                "]" => e.iter_multiplier *= 1.5,
                "," | "<" => e.color_density /= 1.25,
                "." | ">" => e.color_density *= 1.25,
                "p" => e.cycle_palette(true),
                "P" => e.cycle_palette(false),
                "m" | "M" => e.cycle_precision_mode(),
                "r" | "R" => {
                    e.reset_view();
                    e.iter_multiplier = 1.0;
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
        self.request_redraw();
    }

    fn save_screenshot(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else { return };
        let params = self.explorer.frame_params();
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let path = format!("mandelbrot-{secs}.png");
        match renderer.capture(&params).and_then(|px| write_png(&path, &params, &px)) {
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
                self.request_redraw();
            }

            WindowEvent::ScaleFactorChanged { .. } => self.request_redraw(),

            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                self.handle_key(&event.logical_key, event_loop);
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.explorer.cursor_moved(position.x, position.y) {
                    self.request_redraw();
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
                self.request_redraw();
            }

            WindowEvent::PinchGesture { delta, .. } => {
                self.explorer.zoom_at_cursor(delta.exp());
                self.request_redraw();
            }

            WindowEvent::RedrawRequested => {
                let params = self.explorer.frame_params();
                if let Some(r) = self.renderer.as_mut() {
                    match r.render(&params) {
                        Ok(Some(ms)) => self.last_gpu_ms = Some(ms),
                        Ok(None) => {}
                        Err(err) => eprintln!("render failed: {err}"),
                    }
                }
                if let Some(w) = &self.window {
                    w.set_title(&self.explorer.hud(self.last_gpu_ms));
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
    let mut iters: Option<f64> = None;
    let mut density = 1.0f32;
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
    if let Some(n) = iters {
        explorer.iter_multiplier = n / explorer.max_iter() as f64;
    }

    let mut renderer = render::create_headless()?;
    let params = explorer.frame_params();
    let start = std::time::Instant::now();
    let pixels = renderer.capture(&params)?;
    println!("{}  ({:.1} ms)", explorer.hud(None), start.elapsed().as_secs_f64() * 1000.0);
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
