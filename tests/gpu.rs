//! End-to-end tests on the real GPU renderer (headless): the explorer's
//! frame parameters, the scheduler's reuse of earlier work, and the kernel
//! must together produce the same pixels as a fresh render.
//!
//! Also holds an ignored measurement, `simulated_deep_zoom`:
//! `cargo test --release --test gpu simulated_deep_zoom -- --ignored --nocapture`

use mandelbrot::app::{Colors, Explorer, PALETTE_COUNT, PALETTE_PERIOD, Place};
use mandelbrot::bookmarks::PRESETS;
use mandelbrot::progressive::Scheduler;
use mandelbrot::render::{self, Output, Renderer, TileBatch};

const SEAHORSE: (&str, &str) = ("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139");

/// A headless renderer with a `width` x `height` canvas.
fn renderer(width: u32, height: u32) -> Box<dyn Renderer> {
    let mut r = render::create_headless().expect("Metal device");
    r.resize(width, height, 1.0);
    r
}

/// Fraction of pixels whose color differs noticeably between two RGBA images.
fn differing(a: &[u8], b: &[u8]) -> f64 {
    let bad = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| (0..3).any(|i| x[i].abs_diff(y[i]) > 16)).count();
    bad as f64 / (a.len() / 4) as f64
}

fn run_to_completion(s: &mut Scheduler, e: &mut Explorer, r: &mut dyn Renderer) {
    for _ in 0..100_000 {
        if !s.step(e, r).unwrap() {
            return;
        }
    }
    panic!("scheduler never finished");
}

/// Rendering against a reused, off-center reference (as happens while
/// panning) must match rendering against a fresh centered one.
#[test]
fn off_center_reference_matches_fresh() {
    let mut r = render::create_headless().expect("Metal device");
    for zoom in [1e8, 1e40] {
        // c = i: a boundary point with smooth surroundings at every depth, so
        // any offset error shows up as a visibly shifted image.
        let mut panned = Explorer::new(320, 200);
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

        let (a, _) = r.capture(&reused, 1, &panned.colors()).unwrap();
        let (b, _) = r.capture(&centered, 1, &fresh.colors()).unwrap();
        let d = differing(&a, &b);
        // Different references round differently right at the boundary.
        assert!(d < 0.01, "zoom {zoom}: {:.2}% of pixels differ", d * 100.0);
    }
}

/// After panning (reusing exact tiles) or zooming (recomputing over a
/// resampled preview), the finished canvas must equal a fresh render.
#[test]
fn reused_canvas_matches_fresh_render() {
    let (w, h) = (256, 160);
    let mut r = renderer(w, h);
    let mut e = Explorer::new(w, h);
    e.set_center_str("0", "1");
    e.zoom_at_center(1e3);
    e.set_iter_limit(2000);
    let mut s = Scheduler::default();
    run_to_completion(&mut s, &mut e, r.as_mut());

    let changes: [(&str, fn(&mut Explorer)); 3] = [
        ("pan", |e| e.pan_pixels(-37.0, 21.0)),
        ("zoom in", |e| e.zoom_at(90.0, 50.0, 1.7)),
        ("zoom out", |e| e.zoom_at(150.0, 100.0, 0.6)),
    ];
    for (what, change) in changes {
        change(&mut e);
        run_to_completion(&mut s, &mut e, r.as_mut());
        let reused = r.snapshot(&e.colors()).unwrap();
        let (fresh, _) = r.capture(&e.frame_params(), 1, &e.colors()).unwrap();
        let d = differing(&reused, &fresh);
        assert!(d < 0.002, "{what}: {:.3}% of pixels differ", d * 100.0);
    }
}

#[test]
fn stats_only_draw_leaves_canvas_untouched() {
    let (w, h) = (128, 96);
    let mut r = renderer(w, h);
    let mut e = Explorer::new(w, h);
    let colors = e.colors();
    r.draw(&e.frame_params(), &TileBatch::all(w, h, 1, Output::Pixels)).unwrap();
    let before = r.snapshot(&colors).unwrap();

    // Far fewer iterations change mu (more of the image goes black), so any
    // write would show.
    e.set_iter_limit(16);
    let p = e.frame_params();
    let stats = r.draw(&p, &TileBatch::all(w, h, 4, Output::StatsOnly)).unwrap();
    assert!(stats.samples > 0 && stats.max_escaped > 0, "{stats:?}");
    assert!(r.snapshot(&colors).unwrap() == before, "stats-only draw wrote pixels");

    r.draw(&p, &TileBatch::all(w, h, 1, Output::Pixels)).unwrap();
    assert!(r.snapshot(&colors).unwrap() != before, "the iteration change should be visible");
}

/// Colors are applied when showing the canvas, not when computing it: the
/// same canvas gives different images for different colors, and the offset
/// is a phase (a whole cycle changes nothing).
#[test]
fn colors_apply_at_present_time() {
    let (w, h) = (160, 120);
    let mut r = renderer(w, h);
    let mut e = Explorer::new(w, h);
    r.draw(&e.frame_params(), &TileBatch::all(w, h, 1, Output::Pixels)).unwrap();
    let base = e.colors();
    let image = r.snapshot(&base).unwrap();

    let other_palette = Colors { palette: base.palette + 1, ..base };
    let denser = Colors { density: base.density * 1.25, ..base };
    let shifted = Colors { offset: 0.3, ..base };
    for colors in [other_palette, denser, shifted] {
        let other = r.snapshot(&colors).unwrap();
        assert!(differing(&image, &other) > 0.2, "{colors:?} looks the same");
        // The set's interior stays black whatever the colors.
        let black = |px: &[u8]| px.chunks(4).map(|p| p[..3] == [0, 0, 0]).collect::<Vec<_>>();
        assert_eq!(black(&image), black(&other), "{colors:?} changed the interior");
    }
    // Matches a fresh capture with the same colors.
    let (fresh, _) = r.capture(&e.frame_params(), 1, &base).unwrap();
    assert_eq!(image, fresh);
}

/// Color cycling wraps its offset at PALETTE_PERIOD, which is only seamless
/// if every palette repeats there (one that didn't would visibly jump).
#[test]
fn every_palette_repeats_after_the_palette_period() {
    let (w, h) = (160, 120);
    let mut r = renderer(w, h);
    let mut e = Explorer::new(w, h);
    r.draw(&e.frame_params(), &TileBatch::all(w, h, 1, Output::Pixels)).unwrap();
    for palette in 0..PALETTE_COUNT {
        let colors = |offset| Colors { palette, density: 1.0, offset };
        let image = r.snapshot(&colors(0.0)).unwrap();
        let wrapped = r.snapshot(&colors(PALETTE_PERIOD)).unwrap();
        assert!(differing(&image, &wrapped) < 0.001, "palette {palette} doesn't repeat after {PALETTE_PERIOD} cycles");
    }
}

/// Fraction of pure black (never-escaped) pixels, and number of distinct colors.
fn black_and_colors(rgba: &[u8]) -> (f64, usize) {
    let px: Vec<&[u8]> = rgba.chunks(4).collect();
    let black = px.iter().filter(|p| p[..3] == [0, 0, 0]).count() as f64 / px.len() as f64;
    let colors = px.iter().map(|p| [p[0], p[1], p[2]]).collect::<std::collections::HashSet<_>>().len();
    (black, colors)
}

/// Every preset bookmark must show real structure: not one flat color (all
/// pixels escaping alike) and not mostly black (pixels wrongly never
/// escaping, as a bug near nuclei once caused).
#[test]
fn presets_render_structure() {
    let mut r = render::create_headless().expect("Metal device");
    for p in &PRESETS {
        let mut e = Explorer::new(240, 160);
        e.go_to(&Place::parse(p.x, p.y, p.span).unwrap());
        e.set_iter_limit(20_000);
        let (px, _) = r.capture(&e.frame_params(), 1, &e.colors()).unwrap();
        let (black, colors) = black_and_colors(&px);
        assert!(colors > 50, "{}: only {colors} colors", p.name);
        assert!(black < 0.5, "{}: {:.0}% black", p.name, black * 100.0);
    }
}

/// Next to a nucleus the reference orbit passes through ~0 every period,
/// where a fast-phase step is just dz² + dc and can underflow f32. Skipping
/// iterations by series approximation and iterating them all must agree.
#[test]
fn series_skip_and_full_iteration_agree_near_a_nucleus() {
    let star = PRESETS.iter().find(|p| p.name == "six-armed star").unwrap();
    let mut r = render::create_headless().expect("Metal device");
    for span in [1e-63, 1e-75] {
        let mut images = vec![];
        for skip in [true, false] {
            let mut e = Explorer::new(240, 160);
            e.go_to(&Place::parse(star.x, star.y, span).unwrap());
            e.set_iter_limit(3000);
            e.series_skip = skip;
            images.push(r.capture(&e.frame_params(), 1, &e.colors()).unwrap().0);
        }
        let (black, _) = black_and_colors(&images[1]);
        assert!(black < 0.5, "span {span:e}: {:.0}% black without the skip", black * 100.0);
        let d = differing(&images[0], &images[1]);
        assert!(d < 0.01, "span {span:e}: {:.2}% of pixels differ", d * 100.0);
    }
}

/// Simulated continuous pinch-zoom through the seahorse valley, printing the
/// iteration limit along the way. A measurement, not a pass/fail test.
#[test]
#[ignore]
fn simulated_deep_zoom() {
    let (w, h) = (1200, 800);
    let mut r = renderer(w, h);
    let mut e = Explorer::new(w, h);
    e.set_center_str(SEAHORSE.0, SEAHORSE.1);
    e.zoom_at_center(1e10);
    e.iter_limit = e.estimate_iter();
    let mut s = Scheduler::default();
    let start = std::time::Instant::now();
    let mut next_report = 10.0;
    while e.zoom() < 1e30 {
        s.step(&mut e, r.as_mut()).unwrap();
        e.zoom_at_center(1.05);
        let decade = e.zoom().log10();
        if decade >= next_report {
            println!("zoom 1e{decade:.0}: iters {:>6}  ({:.1} s)", e.iter_limit, start.elapsed().as_secs_f64());
            next_report += 2.0;
        }
    }
    // Then stop and let it settle.
    let settle = std::time::Instant::now();
    while s.step(&mut e, r.as_mut()).unwrap() {}
    println!("settled at 1e30: iters {} after {:.1} s", e.iter_limit, settle.elapsed().as_secs_f64());
    println!("{}", e.hud(None, None));
}
