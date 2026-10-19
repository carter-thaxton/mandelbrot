//! Minimal arbitrary-precision signed fixed-point numbers.
//!
//! Mandelbrot coordinates are always small (|x| < 2^64 by a huge margin), so
//! fixed-point is simpler than a real big-float: one 64-bit integer limb plus
//! `frac` 64-bit fraction limbs. Only what the explorer needs is implemented:
//! add, sub, mul, f64 conversion, and decimal parse/format.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// `value = (-1)^neg * Σ limbs[i] * 2^(64 * (i - frac))`, with limbs stored
/// little-endian: `limbs[0]` is the least significant fraction limb and the
/// last limb is the integer part.
///
/// Invariant: zero is never negative, so the derived `PartialEq` is value
/// equality. Every operation ends with `normalize`.
///
/// `Clone` is implemented by hand so `clone_from` can reuse the allocation;
/// a new field must be added there too (`clone_from` would not fail to compile).
#[derive(Debug, PartialEq, Eq)]
pub struct Fixed {
    neg: bool,
    limbs: Vec<u64>,
}

/// Fraction limbs needed to resolve `scale` (units per pixel) with ~64 bits
/// of headroom for accumulated rounding in the reference orbit.
pub fn frac_limbs_for_scale(scale: f64) -> usize {
    let bits = (-scale.log2()).max(0.0) + 64.0;
    ((bits / 64.0).ceil() as usize).max(2)
}

impl Fixed {
    pub fn zero(frac: usize) -> Self {
        Self { neg: false, limbs: vec![0; frac + 1] }
    }

    pub fn frac_limbs(&self) -> usize {
        self.limbs.len() - 1
    }

    /// Change precision: pads with zero low limbs, or truncates them.
    pub fn with_frac_limbs(&self, frac: usize) -> Self {
        let cur = self.frac_limbs();
        let limbs = match frac.cmp(&cur) {
            Ordering::Equal => self.limbs.clone(),
            Ordering::Greater => {
                let mut v = vec![0; frac - cur];
                v.extend_from_slice(&self.limbs);
                v
            }
            Ordering::Less => self.limbs[cur - frac..].to_vec(),
        };
        Self { neg: self.neg, limbs }.normalized()
    }

    fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    /// Restore the invariant: zero is never negative.
    fn normalize(&mut self) {
        if self.is_zero() {
            self.neg = false;
        }
    }

    /// [`Fixed::normalize`], passing the value through.
    fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    pub fn from_f64(v: f64, frac: usize) -> Self {
        let mut out = Self::zero(frac);
        if v == 0.0 || !v.is_finite() {
            return out;
        }
        // |v| = m * 2^e with integer mantissa m.
        let bits = v.abs().to_bits();
        let exp_bits = ((bits >> 52) & 0x7ff) as i64;
        let (m, e) = if exp_bits == 0 {
            (bits & ((1 << 52) - 1), -1074)
        } else {
            ((bits & ((1 << 52) - 1)) | (1 << 52), exp_bits - 1075)
        };
        // Integer representation is m * 2^(e + 64*frac); place m at that bit.
        let shift = e + 64 * frac as i64;
        if shift < 0 {
            if shift > -64 {
                out.limbs[0] = m >> -shift;
            }
        } else {
            let (idx, off) = ((shift / 64) as usize, (shift % 64) as u32);
            if idx < out.limbs.len() {
                out.limbs[idx] |= m << off;
            }
            if off > 0 && idx + 1 < out.limbs.len() {
                out.limbs[idx + 1] |= m >> (64 - off);
            }
        }
        out.neg = v < 0.0;
        out.normalized()
    }

    pub fn to_f64(&self) -> f64 {
        let frac = self.frac_limbs() as i32;
        let mut acc = 0.0;
        for (i, &l) in self.limbs.iter().enumerate() {
            if l != 0 {
                // Split the power so 2^(-64k) can't underflow before the multiply.
                let e = 64 * (i as i32 - frac);
                acc += l as f64 * 2f64.powi(e / 2) * 2f64.powi(e - e / 2);
            }
        }
        if self.neg { -acc } else { acc }
    }

    pub fn add_f64(&self, v: f64) -> Self {
        self + &Fixed::from_f64(v, self.frac_limbs())
    }

    /// Parse a decimal like "-0.7436438870371587047521915". Falls back to f64
    /// parsing for other forms (e.g. "1e-5").
    pub fn parse(s: &str, frac: usize) -> Option<Self> {
        let s = s.trim();
        let (neg, body) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let (int_str, frac_str) = body.split_once('.').unwrap_or((body, ""));
        let digits_ok = |t: &str| t.bytes().all(|b| b.is_ascii_digit());
        if !digits_ok(int_str) || !digits_ok(frac_str) || (int_str.is_empty() && frac_str.is_empty()) {
            return s.parse::<f64>().ok().map(|v| Self::from_f64(v, frac));
        }
        let mut out = Self::zero(frac);
        // Fraction digits from least significant: acc = (acc + d) / 10.
        for d in frac_str.bytes().rev() {
            out.limbs[frac] = (d - b'0') as u64;
            out.div_small(10);
        }
        out.limbs[frac] = if int_str.is_empty() { 0 } else { int_str.parse().ok()? };
        out.neg = neg;
        Some(out.normalized())
    }

    /// Format with exactly `digits` fractional decimal digits (rounded).
    pub fn to_decimal(&self, digits: usize) -> String {
        let frac = self.frac_limbs();
        // Round half away from zero: add 0.5 * 10^-digits to the magnitude.
        let half = Fixed::parse(&format!("0.{}5", "0".repeat(digits)), frac).unwrap();
        let mut f = Fixed { neg: false, limbs: self.limbs.clone() };
        add_mag_assign(&mut f.limbs, &half.limbs);
        let mut s = String::with_capacity(digits + 24);
        if self.neg {
            s.push('-');
        }
        s.push_str(&f.limbs[frac].to_string());
        s.push('.');
        for _ in 0..digits {
            f.limbs[frac] = 0;
            f.mul_small(10);
            s.push((b'0' + f.limbs[frac] as u8) as char);
        }
        s
    }

    fn div_small(&mut self, d: u64) {
        let mut rem = 0u128;
        for l in self.limbs.iter_mut().rev() {
            let cur = (rem << 64) | *l as u128;
            *l = (cur / d as u128) as u64;
            rem = cur % d as u128;
        }
    }

    fn mul_small(&mut self, k: u64) {
        let mut carry = 0u128;
        for l in self.limbs.iter_mut() {
            let t = *l as u128 * k as u128 + carry;
            *l = t as u64;
            carry = t >> 64;
        }
    }

    /// Bring two operands to a common precision.
    fn aligned<'a>(a: &'a Self, b: &'a Self) -> (Cow<'a, Self>, Cow<'a, Self>) {
        let f = a.frac_limbs().max(b.frac_limbs());
        let wrap = |x: &'a Self| {
            if x.frac_limbs() == f { Cow::Borrowed(x) } else { Cow::Owned(x.with_frac_limbs(f)) }
        };
        (wrap(a), wrap(b))
    }
}

fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    a.iter().rev().cmp(b.iter().rev())
}

/// a += b (equal lengths).
fn add_mag_assign(a: &mut [u64], b: &[u64]) {
    let mut carry = false;
    for (x, &y) in a.iter_mut().zip(b) {
        let (s1, c1) = x.overflowing_add(y);
        let (s2, c2) = s1.overflowing_add(carry as u64);
        *x = s2;
        carry = c1 || c2;
    }
}

/// a -= b, requires |a| >= |b|.
fn sub_mag_assign(a: &mut [u64], b: &[u64]) {
    let mut borrow = false;
    for (x, &y) in a.iter_mut().zip(b) {
        let (d1, b1) = x.overflowing_sub(y);
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        *x = d2;
        borrow = b1 || b2;
    }
}

/// a = b - a, requires |b| > |a|.
fn rsub_mag_assign(a: &mut [u64], b: &[u64]) {
    let mut borrow = false;
    for (x, &y) in a.iter_mut().zip(b) {
        let (d1, b1) = y.overflowing_sub(*x);
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        *x = d2;
        borrow = b1 || b2;
    }
}

impl Fixed {
    /// self += other (or -= when `negate`), in place. The result has the
    /// larger of the two precisions.
    fn add_signed(&mut self, other: &Fixed, negate: bool) {
        if other.frac_limbs() > self.frac_limbs() {
            *self = self.with_frac_limbs(other.frac_limbs());
        }
        let other = if other.frac_limbs() < self.frac_limbs() {
            Cow::Owned(other.with_frac_limbs(self.frac_limbs()))
        } else {
            Cow::Borrowed(other)
        };
        let other_neg = other.neg != negate;
        if self.neg == other_neg {
            add_mag_assign(&mut self.limbs, &other.limbs);
        } else if cmp_mag(&self.limbs, &other.limbs) != Ordering::Less {
            sub_mag_assign(&mut self.limbs, &other.limbs);
        } else {
            rsub_mag_assign(&mut self.limbs, &other.limbs);
            self.neg = other_neg;
        }
        self.normalize();
    }

    /// `out = self * other`, reusing `out`'s and `scratch`'s allocations.
    ///
    /// For hot loops, so it requires equal precision rather than silently
    /// realigning (and allocating) every call; `&a * &b` accepts any mix.
    pub fn mul_into(&self, other: &Fixed, out: &mut Fixed, scratch: &mut Vec<u64>) {
        debug_assert_eq!(self.frac_limbs(), other.frac_limbs(), "mul_into needs equal precision");
        let (a, b) = (self, other);
        let n = a.limbs.len();
        let frac = n - 1;
        // Schoolbook product (2n limbs), then drop `frac` low limbs to rescale.
        scratch.clear();
        scratch.resize(2 * n, 0);
        for (i, &x) in a.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in b.limbs.iter().enumerate() {
                let t = scratch[i + j] as u128 + x as u128 * y as u128 + carry;
                scratch[i + j] = t as u64;
                carry = t >> 64;
            }
            scratch[i + n] = carry as u64;
        }
        out.limbs.clear();
        out.limbs.extend_from_slice(&scratch[frac..frac + n]);
        out.neg = a.neg != b.neg;
        out.normalize();
    }
}

impl Clone for Fixed {
    fn clone(&self) -> Self {
        Self { neg: self.neg, limbs: self.limbs.clone() }
    }

    /// Reuses `self`'s allocation (the derived version would not).
    fn clone_from(&mut self, source: &Self) {
        self.neg = source.neg;
        self.limbs.clone_from(&source.limbs);
    }
}

impl AddAssign<&Fixed> for Fixed {
    fn add_assign(&mut self, other: &Fixed) {
        self.add_signed(other, false);
    }
}

impl SubAssign<&Fixed> for Fixed {
    fn sub_assign(&mut self, other: &Fixed) {
        self.add_signed(other, true);
    }
}

/// Consumes the left operand and reuses its buffer.
impl Add<&Fixed> for Fixed {
    type Output = Fixed;
    fn add(mut self, other: &Fixed) -> Fixed {
        self += other;
        self
    }
}

/// Consumes the left operand and reuses its buffer.
impl Sub<&Fixed> for Fixed {
    type Output = Fixed;
    fn sub(mut self, other: &Fixed) -> Fixed {
        self -= other;
        self
    }
}

impl Add for &Fixed {
    type Output = Fixed;
    fn add(self, other: &Fixed) -> Fixed {
        self.clone() + other
    }
}

impl Sub for &Fixed {
    type Output = Fixed;
    fn sub(self, other: &Fixed) -> Fixed {
        self.clone() - other
    }
}

impl Neg for &Fixed {
    type Output = Fixed;
    fn neg(self) -> Fixed {
        Fixed { neg: !self.neg, limbs: self.limbs.clone() }.normalized()
    }
}

impl Mul for &Fixed {
    type Output = Fixed;
    fn mul(self, other: &Fixed) -> Fixed {
        let (a, b) = Fixed::aligned(self, other);
        // Allocate both buffers at their final size up front.
        let n = a.limbs.len();
        let mut out = Fixed { neg: false, limbs: Vec::with_capacity(n) };
        a.mul_into(&b, &mut out, &mut Vec::with_capacity(2 * n));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_round_trip() {
        for v in [0.0, 1.0, -0.75, 0.1, -1.0e-10, 3.141592653589793, -2.5e-300, 123.456] {
            assert_eq!(Fixed::from_f64(v, 20).to_f64(), v, "{v}");
        }
    }

    #[test]
    fn arithmetic_matches_f64() {
        let vals = [0.5, -0.75, 1.25e-7, -3.0, 0.1];
        for &x in &vals {
            for &y in &vals {
                let (a, b) = (Fixed::from_f64(x, 3), Fixed::from_f64(y, 3));
                assert!(((&a + &b).to_f64() - (x + y)).abs() <= 1e-15 * (x + y).abs().max(1.0));
                assert!(((&a - &b).to_f64() - (x - y)).abs() <= 1e-15 * (x - y).abs().max(1.0));
                assert!(((&a * &b).to_f64() - x * y).abs() <= 1e-15 * (x * y).abs().max(1e-30));
            }
        }
    }

    #[test]
    fn decimal_beyond_f64() {
        let s = "-0.743643887037158704752191506114774";
        let x = Fixed::parse(s, 3).unwrap();
        assert_eq!(x.to_decimal(33), s);
        // A difference far below f64 resolution survives.
        let y = Fixed::parse("-0.743643887037158704752191506114775", 3).unwrap();
        let d = (&x - &y).to_f64();
        assert!((d - 1e-33).abs() < 1e-45, "{d}");
    }

    #[test]
    fn mixed_precision_ops() {
        let a = Fixed::from_f64(1.5, 2);
        let b = Fixed::from_f64(-0.25, 5);
        assert_eq!((&a * &b).to_f64(), -0.375);
        assert_eq!((&a + &b).frac_limbs(), 5);
    }

    /// Values with bits far below f64's reach, so exactness is meaningful.
    fn deep(digits: &str, frac: usize) -> Fixed {
        Fixed::parse(digits, frac).unwrap()
    }

    #[test]
    fn zero_is_never_negative() {
        let a = deep("-0.123456789012345678901234567890123456789", 3);
        let zero = Fixed::zero(3);
        assert_eq!(&a - &a, zero);
        let mut b = a.clone();
        b -= &a;
        assert_eq!(b, zero);
        b = -&a;
        b += &a;
        assert_eq!(b, zero);
        // A negative times zero.
        let mut out = Fixed::zero(3);
        a.mul_into(&zero, &mut out, &mut Vec::new());
        assert_eq!(out, zero);
        // Truncating away every nonzero limb.
        let tiny = Fixed::from_f64(-1e-30, 3);
        assert_eq!(tiny.with_frac_limbs(1), Fixed::zero(1));
    }

    #[test]
    fn add_takes_sign_of_larger_magnitude() {
        // Opposite signs with |other| > |self|: the "reverse subtract" path.
        let cases = [(0.25, -0.75, -0.5), (-0.25, 0.75, 0.5), (0.75, -0.25, 0.5), (-0.75, 0.25, -0.5)];
        for (x, y, sum) in cases {
            let mut a = Fixed::from_f64(x, 2);
            a += &Fixed::from_f64(y, 2);
            assert_eq!(a.to_f64(), sum, "{x} + {y}");
            let mut a = Fixed::from_f64(x, 2);
            a -= &Fixed::from_f64(-y, 2);
            assert_eq!(a.to_f64(), sum, "{x} - {}", -y);
        }
    }

    #[test]
    fn in_place_ops_grow_to_the_larger_precision() {
        let mut a = Fixed::from_f64(1.5, 2);
        let b = deep("0.000000000000000000000000000000000000000001", 5); // 1e-42: needs > 2 limbs
        a += &b;
        assert_eq!(a.frac_limbs(), 5);
        assert_eq!(&(a.clone() - &b), &Fixed::from_f64(1.5, 5));
        // The other way round: the narrower operand is widened, self keeps its precision.
        let mut c = deep("0.5", 5);
        c -= &Fixed::from_f64(0.25, 2);
        assert_eq!((c.frac_limbs(), c.to_f64()), (5, 0.25));
    }

    #[test]
    fn identities_hold_exactly() {
        let vals = [
            deep("-0.743643887037158704752191506114774", 4),
            deep("0.131825904205311970493132056385139", 4),
            deep("1.999999999999999999999999999999999999", 4),
            Fixed::zero(4),
        ];
        let mut scratch = Vec::new();
        for a in &vals {
            for b in &vals {
                // Owned (buffer-reusing) and borrowed forms.
                assert_eq!(&(a.clone() + b) - b, *a);
                assert_eq!(&(a - b) + b, *a);
                assert_eq!(a + b, b + a);
                // mul_into into an output of a different size than the result.
                let mut out = Fixed::zero(9);
                a.mul_into(b, &mut out, &mut scratch);
                assert_eq!(out, a * b);
                assert_eq!(out, b * a);
            }
        }
    }

    #[test]
    fn clone_from_reuses_buffer_of_any_size() {
        let src = deep("-0.743643887037158704752191506114774", 4);
        for frac in [1, 4, 9] {
            let mut dst = Fixed::from_f64(0.5, frac);
            dst.clone_from(&src);
            assert_eq!(dst, src);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "equal precision")]
    fn mul_into_rejects_mixed_precision() {
        let (a, b) = (Fixed::from_f64(0.5, 2), Fixed::from_f64(0.5, 3));
        a.mul_into(&b, &mut Fixed::zero(2), &mut Vec::new());
    }
}
