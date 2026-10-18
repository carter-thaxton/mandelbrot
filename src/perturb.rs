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
//! Past ~1e-38, dz and dc are too small for f32's *exponent* range, so the GPU
//! starts each pixel in "floatexp" (f32 mantissa + separate int exponent) and
//! switches to plain f32 once dz has grown. Each reference point therefore
//! also carries a floatexp copy, for when Z itself is tiny.
//!
//! The GPU half (with "rebasing" to avoid glitches) is `iterate_perturb` in
//! src/render/metal/mandelbrot.metal.

use crate::bigfixed::Fixed;

/// Exponent used for floatexp zero; must match FE_ZERO_EXP in the shader.
pub const FE_ZERO_EXP: i32 = -1_000_000;

/// One reference point, laid out to match `RefPoint` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RefPoint {
    /// Z rounded to f32 (fast phase).
    pub z: [f32; 2],
    /// Z = mant * 2^exp, with max(|mant.x|, |mant.y|) in [0.5, 1) (extended phase).
    pub mant: [f32; 2],
    pub exp: i32,
    pub _pad: i32,
}

pub struct ReferenceOrbit {
    /// Z_0 = 0, Z_1 = C, ... Ends at max_iter, or at the first Z that escapes
    /// (inclusive).
    pub points: Vec<RefPoint>,
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
    points.push(ref_point(0.0, 0.0));
    let mut escaped = false;

    for _ in 0..max_iter {
        let x2 = &x * &x;
        let y2 = &y * &y;
        let xy = &x * &y;
        y = &(&xy + &xy) + cy;
        x = &(&x2 - &y2) + cx;

        let (fx, fy) = (x.to_f64(), y.to_f64());
        points.push(ref_point(fx, fy));
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

fn ref_point(x: f64, y: f64) -> RefPoint {
    let (mant, exp) = complex_floatexp(x, y);
    RefPoint { z: [x as f32, y as f32], mant, exp, _pad: 0 }
}

/// x * 2^k without intermediate overflow/underflow of 2^k.
fn ldexp(x: f64, k: i32) -> f64 {
    x * 2f64.powi(k / 2) * 2f64.powi(k - k / 2)
}

/// Split v = m * 2^e with |m| in [0.5, 1) (like C's frexp).
pub fn frexp(v: f64) -> (f64, i32) {
    if v == 0.0 || !v.is_finite() {
        return (v, 0);
    }
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // Subnormal: normalize first.
        let (m, e) = frexp(v * 2f64.powi(64));
        return (m, e - 64);
    }
    (f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52)), exp - 1022)
}

/// Complex floatexp with one shared exponent: (x, y) = mant * 2^exp.
pub fn complex_floatexp(x: f64, y: f64) -> ([f32; 2], i32) {
    let a = x.abs().max(y.abs());
    if a == 0.0 {
        return ([0.0, 0.0], FE_ZERO_EXP);
    }
    let (_, e) = frexp(a);
    ([ldexp(x, -e) as f32, ldexp(y, -e) as f32], e)
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
        for p in r.points[1..].iter().map(|p| p.z) {
            (x, y) = (x * x - y * y + cx, 2.0 * x * y + cy);
            assert!((p[0] as f64 - x).abs() <= 1e-6 * x.abs().max(1.0));
            assert!((p[1] as f64 - y).abs() <= 1e-6 * y.abs().max(1.0));
        }
    }

    #[test]
    fn floatexp_round_trip() {
        for (x, y) in [(1.0, 0.0), (-3.0e-200, 7.5e-201), (0.3, -0.7), (1e-310, 0.0)] {
            let (m, e) = complex_floatexp(x, y);
            let a = m[0].abs().max(m[1].abs());
            assert!((0.5..1.0).contains(&a), "{x} {y}: {m:?}");
            let back = (ldexp(m[0] as f64, e), ldexp(m[1] as f64, e));
            assert!((back.0 - x).abs() <= 1e-7 * x.abs().max(y.abs()));
            assert!((back.1 - y).abs() <= 1e-7 * x.abs().max(y.abs()));
        }
        assert_eq!(complex_floatexp(0.0, 0.0), ([0.0, 0.0], FE_ZERO_EXP));
    }
}
