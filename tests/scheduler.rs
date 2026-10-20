//! The progressive scheduler driving an `Explorer`, against a mock renderer.
//!
//! These check how the two cooperate: which tiles get computed and when,
//! what reaches the screen, and how GPU statistics steer the explorer's
//! automatic iteration limit. No GPU needed.

mod common;

use std::time::Duration;

use common::{Call, MockRenderer, run_to_completion, stats};
use mandelbrot::app::Explorer;
use mandelbrot::progressive::Scheduler;
use mandelbrot::render::{Output, TILE};

#[test]
fn fresh_view_previews_then_refines_every_tile_once() {
    let mut e = Explorer::new(400, 300);
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    run_to_completion(&mut s, &mut e, &mut r);

    let first_work = r.batches.iter().find(|b| b.output == Output::Pixels).unwrap();
    assert!(first_work.step > 1, "first work should be a coarse preview");
    let mut seen = std::collections::HashMap::new();
    for b in r.batches.iter().filter(|b| b.step == 1 && b.output == Output::Pixels) {
        for t in &b.tiles {
            *seen.entry(t.origin).or_insert(0) += 1;
        }
    }
    let total = (400u32.div_ceil(TILE) * 300u32.div_ceil(TILE)) as usize;
    assert_eq!(seen.len(), total);
    assert!(seen.values().all(|&n| n == 1));
    assert_eq!(s.progress(), None);
}

#[test]
fn whole_pixel_pan_only_computes_revealed_tiles() {
    let mut e = Explorer::new(400, 304); // 25 x 19 tiles
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    run_to_completion(&mut s, &mut e, &mut r);

    let before = r.batches.len();
    e.pan_pixels(-40.0, 0.0); // content moves left: 40 new columns on the right
    run_to_completion(&mut s, &mut e, &mut r);
    let xf = r.reprojections.last().unwrap();
    assert_eq!((xf.ratio, xf.shift), (1.0, (40.0, 0.0)));
    let redone = r.exact_tiles_since(before);
    assert!((3 * 19..=4 * 19).contains(&redone), "recomputed {redone} tiles for a 40 px pan");
}

#[test]
fn zoom_in_refines_over_the_resampled_preview() {
    let mut e = Explorer::new(400, 304);
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    run_to_completion(&mut s, &mut e, &mut r);

    // Every tile has a (blurry) source, so no coarse pass is needed.
    let before = r.batches.len();
    e.zoom_at_center(1.5);
    run_to_completion(&mut s, &mut e, &mut r);
    let work: Vec<_> = r.batches[before..].iter().filter(|b| b.output == Output::Pixels).collect();
    assert!(!work.is_empty() && work.iter().all(|b| b.step == 1));
    assert_eq!(r.exact_tiles_since(before), 25 * 19);
}

#[test]
fn canvas_changes_are_presented_and_stats_only_draws_are_not() {
    let mut e = Explorer::new(400, 304);
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    run_to_completion(&mut s, &mut e, &mut r);
    e.pan_pixels(-40.0, 10.0);
    run_to_completion(&mut s, &mut e, &mut r);
    e.zoom_at_center(1.5);
    run_to_completion(&mut s, &mut e, &mut r);

    let log = &r.log;
    assert!(log.contains(&Call::DrawStats), "expected probes");
    assert!(log.contains(&Call::Reproject));
    for (i, call) in log.iter().enumerate() {
        let next = log.get(i + 1);
        match call {
            // Each pixel draw is shown right away.
            Call::DrawPixels => assert_eq!(next, Some(&Call::Present), "at {i}: {log:?}"),
            // A reprojection reaches the screen (possibly together with that
            // frame's pixel draw) before the view moves again.
            Call::Reproject => {
                let rest = &log[i + 1..];
                let present = rest.iter().position(|c| *c == Call::Present);
                let next_move = rest.iter().position(|c| *c == Call::Reproject);
                assert!(present.is_some_and(|p| next_move.is_none_or(|m| p < m)), "at {i}: {log:?}");
            }
            // Statistics alone change nothing on screen.
            Call::DrawStats => assert_ne!(next, Some(&Call::Present), "at {i}: {log:?}"),
            Call::Present => {}
        }
    }
}

#[test]
fn color_changes_only_re_present() {
    let mut e = Explorer::new(400, 304);
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    run_to_completion(&mut s, &mut e, &mut r);

    let changes: [fn(&mut Explorer); 4] = [
        |e| e.cycle_palette(true),
        |e| e.color_density *= 1.25,
        |e| e.shift_colors(1.0 / 32.0),
        // A frame of color cycling.
        |e| {
            e.color_cycling = true;
            e.advance_colors(1.0 / 60.0);
        },
    ];
    for change in changes {
        change(&mut e);
        let before = r.log.len();
        // One step re-presents, with nothing to compute and nothing left to do.
        assert!(!s.step(&mut e, &mut r).unwrap(), "a color change left work to do");
        assert_eq!(&r.log[before..], &[Call::Present], "expected just a present");
        // And once shown, it isn't presented again.
        assert!(!s.step(&mut e, &mut r).unwrap());
        assert_eq!(r.log.len(), before + 1);
    }
}

#[test]
fn iteration_limit_is_judged_per_sweep_not_per_batch() {
    // Full-resolution batches all look cut off, as tiles along the set's
    // boundary do; coarse ones don't.
    let mut r = MockRenderer::new(1e-3, |p, b, samples| {
        let near_limit = if b.step == 1 { (samples / 50) as u32 } else { 0 };
        stats(samples, p.max_iter - 1, near_limit, 0)
    });
    // Window-sized, so a sweep takes many batches even at the minimum batch size.
    let mut e = Explorer::new(2400, 1600);
    let mut s = Scheduler::default();
    let start = e.iter_limit;
    let mut steps = 0;
    while e.iter_limit == start {
        assert!(s.step(&mut e, &mut r).unwrap());
        steps += 1;
        assert!(steps < 10_000);
    }
    // The limit only grew once the whole view had been refined.
    assert!(r.exact_tiles_since(0) > 0 && steps > 5, "expected several batches, got {steps}");
    assert_eq!(s.progress(), None, "limit changed mid-sweep");
    assert_eq!(e.iter_limit, start * 2);
}

#[test]
fn limit_adapts_during_continuous_motion() {
    // Needs 8000 iterations: below that, half the samples stay black and a
    // quarter escape near the limit.
    const NEEDED: u32 = 8000;
    let mut r = MockRenderer::new(1e-4, |p, _, samples| {
        if p.max_iter < NEEDED {
            stats(samples, p.max_iter - 1, (samples / 4) as u32, (samples / 2) as u32)
        } else {
            stats(samples, NEEDED - 1, 0, 0)
        }
    });
    let mut e = Explorer::new(400, 304);
    let mut s = Scheduler::with_probe_interval(Duration::ZERO);
    let start = e.iter_limit;
    // Pan a little every frame: no sweep ever completes, so only probes can
    // move the limit.
    for _ in 0..40 {
        s.step(&mut e, &mut r).unwrap();
        e.pan_pixels(3.0, 0.0);
    }
    assert!(e.iter_limit >= NEEDED, "limit stuck at {} (started at {start})", e.iter_limit);
    assert!(e.iter_limit < 4 * NEEDED, "overshot to {}", e.iter_limit);
}
