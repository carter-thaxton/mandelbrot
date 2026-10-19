use std::time::{Instant, SystemTime, UNIX_EPOCH};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

use mandelbrot::app::{self, Explorer, Precision, PrecisionMode};
use mandelbrot::bookmarks::{Bookmarks, HOME};
use mandelbrot::flight::{self, Flight};
use mandelbrot::progressive::Scheduler;
use mandelbrot::render::{self, Renderer};

const HELP: &str = "\
Mandelbrot explorer
  drag / cmd+two-finger      pan
  two-finger / pinch / wheel zoom at cursor
  = / -                      zoom in / out (center)
  arrow keys                 pan
  0-9                        fly to a saved place (0: home; 1-9 start
                             with deep presets, shallow to deep)
  cmd+1-9                    save the current place
  shift+0-9                  jump there instantly
  + / _                      flight speed up / down (also mid-flight)
  space                      pause / resume a flight
  any navigation, or esc     stop a flight where it is
  [ / ]                      fewer / more iterations (switches to manual)
  a                          automatic iteration limit on / off
  , / .                      color density down / up
  o                          color cycling on / off
  { / }                      color cycling slower / faster
  ; / '                      shift colors back / forward
  p / shift+p                next / previous palette
  m                          precision: auto -> f32 -> df64 -> perturb
  r                          reset view
  s                          save screenshot (PNG in current dir)
  c                          print location (full precision) as a --capture command
  q / cmd+w / cmd+q          quit

Headless: mandelbrot --capture out.png [--size WxH] [--center X,Y] [--zoom Z] [--iters N]
          [--density D] [--precision f32|df|perturb] [--no-skip]";

struct App {
    window: Option<Window>,
    renderer: Option<Box<dyn Renderer>>,
    explorer: Explorer,
    modifiers: ModifiersState,
    scheduler: Scheduler,
    bookmarks: Bookmarks,
    flight: Option<ActiveFlight>,
    /// Cruising speed for flights, zoom decades per second.
    flight_speed: f64,
    /// When color cycling last advanced (None: restart timing).
    last_color_tick: Option<Instant>,
}

struct ActiveFlight {
    flight: Flight,
    slot: u8,
    last_tick: Option<Instant>,
}

/// The bookmark slot of a number-row key.
fn digit_slot(code: KeyCode) -> Option<u8> {
    use KeyCode::*;
    let digits = [Digit0, Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9];
    digits.iter().position(|&d| d == code).map(|i| i as u8)
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            explorer: Explorer::new(1, 1),
            modifiers: ModifiersState::empty(),
            scheduler: Scheduler::default(),
            bookmarks: Bookmarks::with_presets(),
            flight: None,
            flight_speed: flight::DEFAULT_SPEED,
            last_color_tick: None,
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

    /// Number keys: save (cmd), jump (shift), or fly to a place.
    fn handle_slot(&mut self, slot: u8) {
        if self.modifiers.super_key() {
            if self.bookmarks.save(slot, self.explorer.place()) {
                println!("saved place {slot}");
            } else {
                println!("0 is home and can't be changed");
            }
            return;
        }
        let target = if slot == HOME {
            self.explorer.home_place()
        } else if let Some(place) = self.bookmarks.get(slot) {
            place.clone()
        } else {
            println!("nothing saved in {slot} yet (cmd+{slot} saves the current place)");
            return;
        };
        if self.modifiers.shift_key() {
            self.flight = None;
            self.explorer.go_to(&target);
        } else {
            let flight = Flight::new(self.explorer.place(), target, self.flight_speed);
            self.flight = Some(ActiveFlight { flight, slot, last_tick: None });
        }
        self.invalidate();
    }

    fn set_flight_speed(&mut self, factor: f64) {
        self.flight_speed = (self.flight_speed * factor).clamp(0.1, 100.0);
        match &mut self.flight {
            Some(active) => active.flight.speed = self.flight_speed,
            None => println!("flight speed {:.2} decades/s", self.flight_speed),
        }
    }

    fn handle_key(&mut self, event: &KeyEvent, event_loop: &ActiveEventLoop) {
        if let PhysicalKey::Code(code) = event.physical_key
            && let Some(slot) = digit_slot(code)
        {
            self.handle_slot(slot);
            return;
        }
        let key = &event.logical_key;
        // Manual navigation takes over from a flight, leaving the view where it is.
        let navigates = matches!(
            key,
            Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowRight | NamedKey::ArrowUp | NamedKey::ArrowDown)
        ) || matches!(key, Key::Character(c) if matches!(c.as_str(), "=" | "-" | "r" | "R"));
        if navigates {
            self.flight = None;
        }
        let e = &mut self.explorer;
        match key {
            Key::Named(NamedKey::Space) => match &mut self.flight {
                Some(active) if active.flight.is_paused() => active.flight.resume(),
                Some(active) => {
                    active.flight.pause();
                    active.last_tick = None; // so resuming doesn't count the pause as elapsed
                }
                None => return,
            },
            // Esc only ever stops a flight (quitting is q / cmd+w / cmd+q).
            Key::Named(NamedKey::Escape) => {
                if self.flight.take().is_none() {
                    return;
                }
            }
            Key::Named(NamedKey::ArrowLeft) => e.pan_fraction(0.1, 0.0),
            Key::Named(NamedKey::ArrowRight) => e.pan_fraction(-0.1, 0.0),
            Key::Named(NamedKey::ArrowUp) => e.pan_fraction(0.0, 0.1),
            Key::Named(NamedKey::ArrowDown) => e.pan_fraction(0.0, -0.1),
            Key::Character(s) => match s.as_str() {
                "q" | "Q" => event_loop.exit(),
                "w" | "W" if self.modifiers.super_key() => event_loop.exit(),
                "=" => e.zoom_at_center(2.0),
                "-" => e.zoom_at_center(0.5),
                "+" => return self.set_flight_speed(1.5),
                "_" => return self.set_flight_speed(1.0 / 1.5),
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
                    e.color_offset = 0.0;
                }
                "o" | "O" => {
                    e.color_cycling = !e.color_cycling;
                    self.last_color_tick = None;
                }
                "{" => e.scale_color_speed(1.0 / 1.5),
                "}" => e.scale_color_speed(1.5),
                ";" => e.shift_colors(-1.0 / 32.0),
                "'" => e.shift_colors(1.0 / 32.0),
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
        let colors = self.explorer.colors();
        match renderer.capture(&params, 1, &colors).and_then(|(px, _)| write_png(&path, &params, &px)) {
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
                self.handle_key(&event, event_loop);
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.explorer.cursor_moved(position.x, position.y) {
                    self.invalidate();
                }
            }

            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.flight = None; // grabbing the view stops a flight
                        self.explorer.begin_drag();
                    }
                    ElementState::Released => self.explorer.end_drag(),
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.flight = None;
                let pan_modifier = self.modifiers.super_key() || self.modifiers.control_key();
                match delta {
                    // Trackpad: two-finger scroll zooms (one-handed driving,
                    // since dragging already pans); with cmd/ctrl it pans.
                    MouseScrollDelta::PixelDelta(d) if pan_modifier => {
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
                self.flight = None;
                self.explorer.zoom_at_cursor(delta.exp());
                self.invalidate();
            }

            WindowEvent::RedrawRequested => {
                let Some(r) = self.renderer.as_mut() else { return };
                // Color cycling moves the palette by however much time has passed.
                if self.explorer.color_cycling {
                    let now = Instant::now();
                    let dt = self.last_color_tick.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
                    self.last_color_tick = Some(now);
                    self.explorer.advance_colors(dt.min(0.1));
                }
                // A flight moves the view by however much time has passed.
                if let Some(active) = self.flight.as_mut().filter(|a| !a.flight.is_paused()) {
                    let now = Instant::now();
                    let dt = active.last_tick.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
                    active.last_tick = Some(now);
                    let (place, arrived) = active.flight.advance(dt);
                    self.explorer.go_to(&place);
                    if arrived {
                        self.flight = None;
                    }
                }
                // One budget-sized slice of work per display refresh; input
                // that arrives in between invalidates the rest.
                let more = match self.scheduler.step(&mut self.explorer, r.as_mut()) {
                    Ok(more) => more,
                    Err(err) => {
                        eprintln!("render failed: {err}");
                        false
                    }
                };
                let flying = self.flight.as_ref().is_some_and(|a| !a.flight.is_paused());
                if more || flying || self.explorer.color_cycling {
                    self.request_redraw();
                }
                if let Some(w) = &self.window {
                    let mut title = self.explorer.hud(Some(self.scheduler.work_ms()), self.scheduler.progress());
                    if let Some(active) = &self.flight {
                        let dest = match (active.slot, self.bookmarks.name(active.slot)) {
                            (HOME, _) => "home".to_string(),
                            (slot, Some(name)) => format!("{slot} ({name})"),
                            (slot, None) => slot.to_string(),
                        };
                        title += &if active.flight.is_paused() {
                            format!("  |  flight to {dest} paused (space resumes)")
                        } else {
                            format!("  |  flying to {dest} at {:.2} decades/s", active.flight.speed)
                        };
                    }
                    w.set_title(&title);
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
    // interactive preview does: as coarse as possible while still giving
    // enough samples to decide on (small images need a finer step).
    let pixels = w as f64 * h as f64;
    let settle_step = (1..=4).rev().find(|s| pixels / (s * s) as f64 >= app::MIN_DECISION_SAMPLES as f64).unwrap_or(1);
    for _ in 0..8 {
        let (_, stats) = renderer.capture(&explorer.frame_params(), settle_step, &explorer.colors())?;
        if std::env::var_os("MANDEL_DEBUG_ITERS").is_some() {
            eprintln!("settle: limit {} -> {stats:?}", explorer.iter_limit);
        }
        if !explorer.observe(&stats, true) {
            break;
        }
    }
    let settle_ms = start.elapsed().as_secs_f64() * 1000.0;
    let params = explorer.frame_params();
    let (pixels, stats) = renderer.capture(&params, 1, &explorer.colors())?;
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
