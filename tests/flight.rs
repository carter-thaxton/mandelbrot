//! Flights driving the explorer and scheduler frame by frame, as the app
//! does: the view must arrive exactly, and every frame should reuse the
//! previous one (a reprojection, never a fresh canvas).

mod common;

use common::MockRenderer;
use mandelbrot::app::{Explorer, Place};
use mandelbrot::bookmarks::Bookmarks;
use mandelbrot::flight::Flight;
use mandelbrot::progressive::Scheduler;

const FRAME: f64 = 1.0 / 60.0;

/// True if two places are the same view (Fixed's == also compares precision).
fn same_place(a: &Place, b: &Place) -> bool {
    (&a.center.0 - &b.center.0).to_f64() == 0.0 && (&a.center.1 - &b.center.1).to_f64() == 0.0 && a.span == b.span
}

/// Fly the explorer to `to`, one scheduler step per frame. Returns the
/// number of frames and the widest view along the way.
fn fly(e: &mut Explorer, s: &mut Scheduler, r: &mut MockRenderer, to: &Place, speed: f64) -> (usize, f64) {
    let mut flight = Flight::new(e.place(), to.clone(), speed);
    let mut widest: f64 = 0.0;
    for frame in 1..100_000 {
        let (place, arrived) = flight.advance(FRAME);
        e.go_to(&place);
        widest = widest.max(e.place().span);
        s.step(e, r).unwrap();
        if arrived {
            return (frame, widest);
        }
    }
    panic!("flight never arrived");
}

fn deep_place(x: &str, y: &str, span: f64) -> Place {
    let mut e = Explorer::new(400, 304);
    e.set_center_str(x, y);
    let mut p = e.place();
    p.span = span;
    p
}

#[test]
fn flies_from_home_into_a_deep_bookmark() {
    let mut e = Explorer::new(400, 304);
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    s.step(&mut e, &mut r).unwrap();

    let mut bookmarks = Bookmarks::default();
    let target = deep_place("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 1e-25);
    assert!(bookmarks.save(1, target.clone()));
    let start_iters = e.iter_limit;

    let (frames, widest) = fly(&mut e, &mut s, &mut r, bookmarks.get(1).unwrap(), 10.0);
    assert!(same_place(&e.place(), &target), "did not arrive exactly");
    // Mostly a zoom-in: never wider than home.
    assert!(widest <= e.home_place().span * 1.01, "zoomed out to {widest}");
    // Every frame reprojected the previous one.
    assert!(r.reprojections.len() >= frames - 1, "{} reprojections in {frames} frames", r.reprojections.len());
    // Auto iterations followed the depth.
    assert!(e.iter_limit > start_iters, "iterations stayed at {}", e.iter_limit);
}

#[test]
fn flies_between_deep_places_via_a_shallow_apex() {
    let mut e = Explorer::new(400, 304);
    let (mut s, mut r) = (Scheduler::default(), MockRenderer::settled());
    let a = deep_place("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 1e-60);
    let b = deep_place("0", "1", 1e-60);
    e.go_to(&a);
    s.step(&mut e, &mut r).unwrap();

    let (_, widest) = fly(&mut e, &mut s, &mut r, &b, 10.0);
    assert!(same_place(&e.place(), &b), "did not arrive exactly");
    // The two places are ~1.3 apart: the flight climbs to roughly that size.
    assert!((0.3..10.0).contains(&widest), "apex view {widest}");

    // And home again.
    let home = e.home_place();
    fly(&mut e, &mut s, &mut r, &home, 10.0);
    assert!(same_place(&e.place(), &home));
    assert_eq!(e.zoom(), 1.0);
}
