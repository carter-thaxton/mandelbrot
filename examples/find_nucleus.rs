//! Locate a minibrot's nucleus to high precision (used to author the preset
//! bookmarks).
//!
//! `cargo run --release --example find_nucleus -- <cx> <cy> <radius> [max_period]`
//!
//! Finds the lowest period p with a nucleus (a c whose orbit of 0 returns to
//! 0 after p steps) within about `radius` of (cx, cy), refines it with
//! Newton's method in `Fixed` precision, and prints it with an estimate of
//! the minibrot's size.
//!
//! - Period: the "ball" test. A disk of radius r around c maps, to first
//!   order, to a disk of radius r·|dz_n/dc| around z_n; the first n where
//!   that disk contains 0 is the period of the dominant nucleus nearby.
//! - Newton: c ← c − z_p(c) / z_p'(c), with z in Fixed and the derivative in
//!   f64 (only its relative accuracy matters).
//! - Size: the standard estimate 1 / |β λ²| with λ = Π 2z_i and
//!   β = Σ 1/Π 2z_i over the cycle (see e.g. mu-ency "atom size").

use mandelbrot::bigfixed::{Fixed, frac_limbs_for_scale};

type C = (Fixed, Fixed);

/// z² + c
fn step(z: &C, c: &C) -> C {
    let x2 = &z.0 * &z.0;
    let y2 = &z.1 * &z.1;
    let xy = &z.0 * &z.1;
    (&(&x2 - &y2) + &c.0, &(&xy + &xy) + &c.1)
}

fn f64s(z: &C) -> (f64, f64) {
    (z.0.to_f64(), z.1.to_f64())
}

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// The lowest period whose ball around `c` of radius `r` contains 0.
fn period(c: &C, r: f64, max: u32) -> Option<u32> {
    let frac = c.0.frac_limbs();
    let mut z = (Fixed::zero(frac), Fixed::zero(frac));
    let mut dz = (0.0, 0.0);
    for n in 1..=max {
        let zf = f64s(&z);
        let t = cmul(zf, dz);
        dz = (2.0 * t.0 + 1.0, 2.0 * t.1);
        z = step(&z, c);
        let zf = f64s(&z);
        if zf.0.hypot(zf.1) > 1e10 {
            return None;
        }
        if zf.0.hypot(zf.1) < r * dz.0.hypot(dz.1) {
            return Some(n);
        }
    }
    None
}

/// Newton's method for the period-p nucleus near `c`; None if it doesn't
/// converge to one.
fn nucleus(mut c: C, p: u32) -> Option<C> {
    let frac = c.0.frac_limbs();
    // Steps can grow at first; stop only once they reach the precision floor.
    let floor = 2f64.powi(-64 * frac as i32 + 8);
    for _ in 0..200 {
        let mut z = (Fixed::zero(frac), Fixed::zero(frac));
        let mut dz = (0.0, 0.0);
        for _ in 0..p {
            let t = cmul(f64s(&z), dz);
            dz = (2.0 * t.0 + 1.0, 2.0 * t.1);
            z = step(&z, &c);
        }
        let delta = cdiv(f64s(&z), dz);
        c = (c.0.add_f64(-delta.0), c.1.add_f64(-delta.1));
        if delta.0.hypot(delta.1) <= floor {
            return returns_to_zero(&c, p).then_some(c);
        }
    }
    None
}

/// Whether the orbit of 0 under c is back near 0 after p steps.
fn returns_to_zero(c: &C, p: u32) -> bool {
    let frac = c.0.frac_limbs();
    let mut z = (Fixed::zero(frac), Fixed::zero(frac));
    for _ in 0..p {
        z = step(&z, c);
    }
    let zf = f64s(&z);
    zf.0.hypot(zf.1) < 1e-20
}

/// Estimated minibrot size (radius, plane units) of the period-p nucleus `c`.
fn size(c: &C, p: u32) -> f64 {
    let frac = c.0.frac_limbs();
    let mut z = (Fixed::zero(frac), Fixed::zero(frac));
    let (mut l, mut b) = ((1.0, 0.0), (1.0, 0.0));
    for _ in 1..p {
        z = step(&z, c);
        l = cmul((2.0, 0.0), cmul(f64s(&z), l));
        let inv = cdiv((1.0, 0.0), l);
        b = (b.0 + inv.0, b.1 + inv.1);
    }
    let s = cdiv((1.0, 0.0), cmul(b, cmul(l, l)));
    s.0.hypot(s.1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [cx, cy, r, rest @ ..] = args.as_slice() else {
        eprintln!("usage: find_nucleus <cx> <cy> <radius> [max_period]");
        std::process::exit(1);
    };
    let r: f64 = r.parse().expect("radius");
    let max: u32 = rest.first().map_or(100_000, |m| m.parse().expect("max_period"));
    // Enough precision for minibrots far smaller than the search radius.
    let frac = frac_limbs_for_scale(r * r * 1e-20);
    let c = (Fixed::parse(cx, frac).expect("cx"), Fixed::parse(cy, frac).expect("cy"));

    let Some(p) = period(&c, r, max) else {
        println!("no nucleus found within {r:e} (up to period {max})");
        return;
    };
    let Some(n) = nucleus(c.clone(), p) else {
        println!("period {p}, but Newton did not converge to a nucleus");
        return;
    };
    let s = size(&n, p);
    let moved = (&n.0 - &c.0).to_f64().hypot((&n.1 - &c.1).to_f64());
    let digits = ((-s.log10()).ceil() as usize + 8).max(10);
    println!("period {p}");
    println!("nucleus {},{}", n.0.to_decimal(digits), n.1.to_decimal(digits));
    println!("size {s:.3e}  (moved {moved:.3e} from the start)");
    // Check: z_p returns (almost) to 0, and the orbit stays bounded.
    let frac = n.0.frac_limbs();
    let mut z = (Fixed::zero(frac), Fixed::zero(frac));
    for i in 1..=20 * p {
        z = step(&z, &n);
        let zf = f64s(&z);
        if i == p {
            println!("check: |z_p| = {:.3e}", zf.0.hypot(zf.1));
        }
        if zf.0.hypot(zf.1) > 2.0 {
            println!("check: ESCAPED at iteration {i}");
            return;
        }
    }
    println!("check: bounded for {} iterations", 20 * p);
}
