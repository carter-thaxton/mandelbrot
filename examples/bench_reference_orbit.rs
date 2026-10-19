//! Cost of computing the perturbation reference orbit (the one high-precision
//! orbit every pixel is measured against) at various precisions.
//!
//! `cargo run --release --example bench_reference_orbit`
//!
//! Prints, per fraction-limb count: the full `compute_reference` (Fixed
//! arithmetic + f64 conversions + series coefficients), and loops doing only
//! the Fixed arithmetic of one orbit step, written two ways: with buffer-
//! reusing in-place ops (as `compute_reference` does), and with the allocating
//! `&a op &b` operators.

use std::hint::black_box;
use std::time::Instant;

use mandelbrot::bigfixed::Fixed;
use mandelbrot::perturb::compute_reference;

const ITERS: u32 = 50_000;
const REPEATS: usize = 5;

/// Fastest of REPEATS runs, in ms.
fn best_ms(mut f: impl FnMut()) -> f64 {
    (0..REPEATS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .fold(f64::INFINITY, f64::min)
}

/// The reference loop's arithmetic with reused buffers, nothing else.
fn fixed_in_place(cx: &Fixed, cy: &Fixed, iters: u32) -> Fixed {
    let frac = cx.frac_limbs();
    let (mut x, mut y) = (Fixed::zero(frac), Fixed::zero(frac));
    let (mut x2, mut y2, mut xy) = (Fixed::zero(frac), Fixed::zero(frac), Fixed::zero(frac));
    let mut scratch = Vec::new();
    for _ in 0..iters {
        x.mul_into(&x, &mut x2, &mut scratch);
        y.mul_into(&y, &mut y2, &mut scratch);
        x.mul_into(&y, &mut xy, &mut scratch);
        y.clone_from(&xy);
        y += &xy;
        y += cy;
        x.clone_from(&x2);
        x -= &y2;
        x += cx;
    }
    x
}

/// The same arithmetic with the allocating operators.
fn fixed_operators(cx: &Fixed, cy: &Fixed, iters: u32) -> Fixed {
    let frac = cx.frac_limbs();
    let (mut x, mut y) = (Fixed::zero(frac), Fixed::zero(frac));
    for _ in 0..iters {
        let x2 = &x * &x;
        let y2 = &y * &y;
        let xy = &x * &y;
        y = &(&xy + &xy) + cy;
        x = &(&x2 - &y2) + cx;
    }
    x
}

fn main() {
    println!("{ITERS} iterations, best of {REPEATS}\n");
    println!(
        "{:>6} {:>8} {:>10} {:>16} {:>8} {:>16}",
        "limbs", "~zoom", "full (ms)", "Fixed in-place", "share", "Fixed operators"
    );
    for frac in [2usize, 4, 8, 17] {
        // In the main cardioid: never escapes, so every run is full length.
        let cx = Fixed::from_f64(-0.1, frac);
        let cy = Fixed::from_f64(0.1, frac);
        let full = best_ms(|| {
            let r = compute_reference(&cx, &cy, ITERS);
            assert_eq!(r.points.len(), ITERS as usize + 1);
            black_box(r);
        });
        let in_place = best_ms(|| {
            black_box(fixed_in_place(&cx, &cy, ITERS));
        });
        let operators = best_ms(|| {
            black_box(fixed_operators(&cx, &cy, ITERS));
        });
        // Roughly the depth this precision serves: 64 bits of headroom.
        let zoom_decades = ((frac as f64 * 64.0 - 64.0) * 2f64.log10()).max(0.0);
        println!(
            "{frac:>6} {:>8} {full:>10.1} {in_place:>16.1} {:>7.0}% {operators:>16.1}",
            format!("1e{zoom_decades:.0}"),
            100.0 * in_place / full
        );
    }
}
