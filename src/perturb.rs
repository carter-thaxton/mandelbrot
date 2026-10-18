//! Perturbation theory: the CPU half.
//!
//! Instead of iterating every pixel at high precision, we iterate ONE point,
//! the reference C (the view center), at high precision:
//!
//!     Z_{n+1} = Z_n^2 + C
//!
//! Each pixel c = C + dc then only tracks its small difference dz = z - Z:
//!
//!     dz_{n+1} = 2 Z_n dz_n + dz_n^2 + dc  =  (2 Z_n + dz_n) dz_n + dc
//!
//! dz and dc are tiny, but *relatively* they only need ~7 digits, so f32 on the
//! GPU is enough. The reference only needs to be stored rounded to f32, since
//! it enters the formula multiplied by dz.
//!
//! The GPU half (with "rebasing" to avoid glitches) is `iterate_perturb` in
//! src/render/metal/mandelbrot.metal.

use crate::bigfixed::Fixed;

pub struct ReferenceOrbit {
    /// Z_0 = 0, Z_1 = C, ... rounded to f32. Ends at max_iter, or at the first
    /// Z that escapes (inclusive).
    pub points: Vec<[f32; 2]>,
    /// True if the reference escaped before max_iter (the GPU copes via
    /// rebasing, but an in-set reference is cheaper).
    pub escaped: bool,
    pub compute_ms: f64,
}

/// Same bailout as the kernel (|Z| > 256).
const BAILOUT2: f64 = 65536.0;

pub fn compute_reference(cx: &Fixed, cy: &Fixed, max_iter: u32) -> ReferenceOrbit {
    let start = std::time::Instant::now();
    let frac = cx.frac_limbs().max(cy.frac_limbs());
    let mut x = Fixed::zero(frac);
    let mut y = Fixed::zero(frac);
    let mut points = Vec::with_capacity(max_iter as usize + 1);
    points.push([0.0, 0.0]);
    let mut escaped = false;

    for _ in 0..max_iter {
        let x2 = &x * &x;
        let y2 = &y * &y;
        let xy = &x * &y;
        y = &(&xy + &xy) + cy;
        x = &(&x2 - &y2) + cx;

        let (fx, fy) = (x.to_f64(), y.to_f64());
        points.push([fx as f32, fy as f32]);
        if fx * fx + fy * fy > BAILOUT2 {
            escaped = true;
            break;
        }
    }

    ReferenceOrbit {
        points,
        escaped,
        compute_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_point_runs_full_length() {
        let c = (Fixed::from_f64(-0.1, 2), Fixed::from_f64(0.1, 2));
        let r = compute_reference(&c.0, &c.1, 500);
        assert_eq!(r.points.len(), 501);
        assert!(!r.escaped);
    }

    #[test]
    fn exterior_point_escapes_like_f64() {
        let (cx, cy) = (0.5, 0.5);
        let r = compute_reference(&Fixed::from_f64(cx, 2), &Fixed::from_f64(cy, 2), 1000);
        assert!(r.escaped);
        // Compare against a plain f64 iteration.
        let (mut x, mut y) = (0.0f64, 0.0f64);
        for p in &r.points[1..] {
            (x, y) = (x * x - y * y + cx, 2.0 * x * y + cy);
            assert!((p[0] as f64 - x).abs() <= 1e-6 * x.abs().max(1.0));
            assert!((p[1] as f64 - y).abs() <= 1e-6 * y.abs().max(1.0));
        }
    }
}
