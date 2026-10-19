//! Smooth flights between places: zoom out as far as needed, pan, and zoom
//! back in.
//!
//! The path is the optimal one from van Wijk & Nuij, "Smooth and efficient
//! zooming and panning" (2003). Treating pan distance `u` and view size `w`
//! as one space, in which moving by one view width costs the same as zooming
//! by a constant factor, the shortest path between two views is a hyperbola,
//! with a closed form in the path length `s`:
//!
//! ```text
//! u(s) = w0/ρ² · sinh(ρs) / cosh(r0 + ρs)      (the paper's tanh form, rewritten
//! w(s) = w0 · cosh(r0) / cosh(r0 + ρs)          so u is accurate near 0)
//! ```
//!
//! ρ trades zooming against panning: the larger it is, the higher the path
//! climbs. Between two equally deep places a distance D apart, the widest
//! view is ρ²/2 · D tall. Close places mostly pan; same-center ones only zoom.
//!
//! Moving at constant speed in `s` looks like constant speed to the eye, so
//! cruising speed is given in zoom decades per second. On top of that, speed
//! follows "gravity": like a ball thrown upward, the flight slows as it
//! climbs, hangs briefly at its widest view, and speeds up again as it
//! descends (see [`APEX_SPEED_FRACTION`]).
//!
//! Precision: at deep zoom the view is far smaller than anything f64 can
//! resolve next to the distance between the places. So each frame is placed
//! relative to whichever endpoint is closer (the path evaluated forward from
//! the start, or backward from the destination, which is the same curve), as
//! a small f64 offset added to that endpoint's exact [`Fixed`] center. Sizes
//! are computed in log space, so `cosh` of the large arguments that deep zooms
//! produce (hundreds) never overflows.

use std::f64::consts::{LN_2, LN_10};

use crate::app::Place;
use crate::bigfixed::Fixed;

// Tuning. These set the feel of a flight.

/// Zoom/pan trade-off: the widest view between two equally deep places is
/// ρ²/2 times their distance (√2, the paper's value, gives just 1x).
const RHO: f64 = 2.0;
/// Speed at the flight's widest view, as a fraction of cruising speed.
pub const APEX_SPEED_FRACTION: f64 = 0.25;
/// Zoom decades below the widest view at which cruising speed is regained
/// (speed² grows linearly with depth, as a falling body's does with height).
pub const APEX_EASE_DECADES: f64 = 1.5;

/// Initial cruising speed, zoom decades per second.
pub const DEFAULT_SPEED: f64 = 1.5;

// Mechanics.

/// Seconds to reach cruising speed, and roughly to stop from it.
const RAMP_SECONDS: f64 = 0.5;
/// Never advance more than this per frame (after a stall, catch up gently).
const MAX_STEP_SECONDS: f64 = 0.1;
/// Below this fraction of cruising speed the ramps don't slow down further,
/// so the flight starts moving at once and actually arrives.
const MIN_SPEED_FRACTION: f64 = 0.05;
/// Places closer than this (relative to their view size) are a pure zoom.
const SAME_CENTER: f64 = 1e-9;

/// ln(cosh(x)) without overflow.
fn ln_cosh(x: f64) -> f64 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p() - LN_2
}

/// ln(sinh(x)) for x > 0, without overflow.
fn ln_sinh(x: f64) -> f64 {
    if x < 1.0 { x.sinh().ln() } else { x + (-(-2.0 * x).exp()).ln_1p() - LN_2 }
}

/// The path seen from one endpoint, as distance travelled from it.
#[derive(Clone, Copy, Debug)]
struct Half {
    ln_w0: f64,
    r0: f64,
}

impl Half {
    /// After path length `s`: (pan distance from this endpoint, ln view size).
    fn at(&self, s: f64) -> (f64, f64) {
        let x = RHO * s;
        let ln_cosh_end = ln_cosh(self.r0 + x);
        let ln_w = self.ln_w0 + ln_cosh(self.r0) - ln_cosh_end;
        let u = if x <= 0.0 { 0.0 } else { (self.ln_w0 - 2.0 * RHO.ln() + ln_sinh(x) - ln_cosh_end).exp() };
        (u, ln_w)
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    /// From the start and from the destination; the same curve.
    Pan { from_start: Half, from_end: Half },
    /// Same center: zoom only, at constant speed in log size.
    Zoom { ln_w0: f64, ln_w1: f64 },
}

pub struct Flight {
    from: Place,
    to: Place,
    /// Unit vector from `from` to `to` (f64 is plenty for a direction).
    dir: (f64, f64),
    /// |to - from| in plane units.
    distance: f64,
    shape: Shape,
    /// Total path length, and how far along it we are.
    length: f64,
    travelled: f64,
    elapsed: f64,
    /// ln of the widest view along the path, and of the current view.
    ln_w_apex: f64,
    ln_w: f64,
    /// Cruising speed in zoom decades per second; may change mid-flight.
    pub speed: f64,
}

impl Flight {
    pub fn new(from: Place, to: Place, speed: f64) -> Self {
        let dx = (&to.center.0 - &from.center.0).to_f64();
        let dy = (&to.center.1 - &from.center.1).to_f64();
        let distance = dx.hypot(dy);
        let (w0, w1) = (from.span, to.span);

        let (shape, length) = if distance <= SAME_CENTER * w0.min(w1) {
            let (ln_w0, ln_w1) = (w0.ln(), w1.ln());
            (Shape::Zoom { ln_w0, ln_w1 }, (ln_w1 - ln_w0).abs() / RHO)
        } else {
            // The paper's b_i = (w1² − w0² ± ρ⁴u²) / (2 w_i ρ² u), written so
            // tiny or huge sizes are never squared (1e-300² underflows).
            let rho2 = RHO * RHO;
            let b0 = (w1 * (w1 / w0) - w0) / (2.0 * rho2 * distance) + rho2 * distance / (2.0 * w0);
            let b1 = (w1 - w0 * (w0 / w1)) / (2.0 * rho2 * distance) - rho2 * distance / (2.0 * w1);
            // r_i = ln(−b_i + √(b_i² + 1)) = −asinh(b_i), without squaring b_i.
            let (r0, r1) = (-b0.asinh(), -b1.asinh());
            let from_start = Half { ln_w0: w0.ln(), r0 };
            // Reversed, the destination's parameter is −r1.
            let from_end = Half { ln_w0: w1.ln(), r0: -r1 };
            (Shape::Pan { from_start, from_end }, (r1 - r0) / RHO)
        };
        // The widest view: where r0 + ρs = 0 if that's on the path (w is then
        // w0·cosh(r0)); otherwise the path only zooms one way, and it's the
        // shallower end.
        let (ln_w0, ln_w1) = (w0.ln(), w1.ln());
        let ln_w_apex = match shape {
            Shape::Pan { from_start, .. } if (0.0..length).contains(&(-from_start.r0 / RHO)) => {
                ln_w0 + ln_cosh(from_start.r0)
            }
            _ => ln_w0.max(ln_w1),
        };
        let dir = if distance > 0.0 { (dx / distance, dy / distance) } else { (0.0, 0.0) };
        Self {
            from,
            to,
            dir,
            distance,
            shape,
            length,
            travelled: 0.0,
            elapsed: 0.0,
            ln_w_apex,
            ln_w: ln_w0,
            speed,
        }
    }

    /// Path length per second at cruising speed: one zoom decade is
    /// ln(10)/ρ of path length.
    fn cruise(&self) -> f64 {
        self.speed * LN_10 / RHO
    }

    /// Seconds a flight takes at constant cruising speed (no ramps).
    pub fn cruise_seconds(&self) -> f64 {
        self.length / self.cruise()
    }

    /// Path speed right now: cruising, limited by the start and arrival
    /// ramps and by gravity near the widest view.
    fn current_speed(&self) -> f64 {
        let cruise = self.cruise();
        // Speed up over RAMP_SECONDS; slow down so a constant deceleration
        // (cruise / RAMP_SECONDS) would stop exactly at the destination.
        let ramp_up = cruise * (self.elapsed / RAMP_SECONDS);
        let ramp_down = (2.0 * cruise / RAMP_SECONDS * (self.length - self.travelled)).max(0.0).sqrt();
        // Gravity: speed² = top² + 2·g·depth, reaching cruise at APEX_EASE_DECADES.
        let top = cruise * APEX_SPEED_FRACTION;
        let g = (cruise * cruise - top * top) / (2.0 * APEX_EASE_DECADES);
        let depth = ((self.ln_w_apex - self.ln_w) / LN_10).max(0.0);
        let gravity = (top * top + 2.0 * g * depth).sqrt();
        ramp_up.min(ramp_down).min(gravity).min(cruise).max(cruise * MIN_SPEED_FRACTION)
    }

    /// Advance by `dt` seconds of wall-clock time; returns the place to show
    /// and whether the flight has arrived.
    pub fn advance(&mut self, dt: f64) -> (Place, bool) {
        let dt = dt.clamp(0.0, MAX_STEP_SECONDS);
        self.elapsed += dt;
        self.travelled += self.current_speed() * dt;
        if self.travelled >= self.length {
            self.travelled = self.length;
            self.ln_w = self.to.span.ln();
            return (self.to.clone(), true);
        }
        let (place, ln_w) = self.at(self.travelled);
        self.ln_w = ln_w;
        (place, false)
    }

    /// The place at path length `s` (0 < s < length), and its ln view size.
    fn at(&self, s: f64) -> (Place, f64) {
        let (offset_from_start, offset_from_end, ln_w) = match self.shape {
            Shape::Pan { from_start, from_end } => {
                let (u_start, ln_w_start) = from_start.at(s);
                let (u_end, ln_w_end) = from_end.at(self.length - s);
                if u_start <= u_end {
                    (Some(u_start), None, ln_w_start)
                } else {
                    (None, Some(u_end), ln_w_end)
                }
            }
            Shape::Zoom { ln_w0, ln_w1 } => {
                let t = s / self.length;
                let ln_w = ln_w0 + (ln_w1 - ln_w0) * t;
                if t <= 0.5 {
                    (Some(self.distance * t), None, ln_w)
                } else {
                    (None, Some(self.distance * (1.0 - t)), ln_w)
                }
            }
        };
        let center = match (offset_from_start, offset_from_end) {
            (Some(u), _) => offset(&self.from.center, self.dir, u),
            (_, Some(u)) => offset(&self.to.center, self.dir, -u),
            _ => unreachable!(),
        };
        (Place { center, span: ln_w.exp() }, ln_w)
    }
}

/// `center + u * dir`, exact to the precision of `center`.
fn offset(center: &(Fixed, Fixed), dir: (f64, f64), u: f64) -> (Fixed, Fixed) {
    (center.0.add_f64(dir.0 * u), center.1.add_f64(dir.1 * u))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn place(x: &str, y: &str, span: f64) -> Place {
        Place { center: (Fixed::parse(x, 8).unwrap(), Fixed::parse(y, 8).unwrap()), span }
    }

    /// Distance between two places' centers, in units of the first's span.
    fn apart(a: &Place, b: &Place) -> f64 {
        let dx = (&a.center.0 - &b.center.0).to_f64();
        let dy = (&a.center.1 - &b.center.1).to_f64();
        dx.hypot(dy) / a.span
    }

    /// Every place along a flight, at 60 frames per second.
    fn fly(from: &Place, to: &Place, speed: f64) -> Vec<Place> {
        let mut f = Flight::new(from.clone(), to.clone(), speed);
        let mut path = vec![from.clone()];
        for _ in 0..1_000_000 {
            let (p, done) = f.advance(1.0 / 60.0);
            path.push(p);
            if done {
                return path;
            }
        }
        panic!("never arrived");
    }

    /// No frame jumps: consecutive frames differ by a small fraction of the
    /// view, in both position and size.
    fn assert_smooth(path: &[Place], max_step: f64) {
        for (i, w) in path.windows(2).enumerate() {
            let (a, b) = (&w[0], &w[1]);
            let pan = apart(a, b);
            let zoom = (b.span / a.span).ln().abs();
            assert!(pan < max_step && zoom < max_step, "frame {i}: pan {pan:.3} views, zoom {zoom:.3} (ln)");
        }
    }

    #[test]
    fn arrives_exactly() {
        let from = place("-0.75", "0", 2.6);
        let to = place("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 1e-25);
        let path = fly(&from, &to, 6.0);
        assert_eq!(path.last().unwrap(), &to);
        assert_smooth(&path, 0.25);
    }

    #[test]
    fn far_apart_deep_places_zoom_out_to_see_both() {
        // Two deep places 1.3 units apart: the path must climb until the view
        // is about that big, and stay exact and smooth at both ends.
        let a = place("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 1e-200);
        let b = place("0", "1", 1e-200);
        // 10 decades/s at 60 fps: about 0.38 (ln) of zoom per frame.
        let path = fly(&a, &b, 10.0);
        let widest = path.iter().map(|p| p.span).fold(0.0, f64::max);
        assert!((0.3..10.0).contains(&widest), "apex view {widest}");
        assert_eq!(path.last().unwrap(), &b);
        assert_smooth(&path, 0.5);
        // Near both ends, the place is resolved far below f64 precision:
        // successive frames approach `b` smoothly, in units of the view.
        let tail = &path[path.len() - 20..];
        let remaining: Vec<f64> = tail.iter().map(|p| apart(p, &b)).collect();
        assert!(remaining.windows(2).all(|w| w[1] <= w[0] + 1e-9), "{remaining:?}");
    }

    #[test]
    fn halves_describe_the_same_curve() {
        // From each end, the two views of the path must agree everywhere:
        // offsets sum to the distance, and sizes match.
        let cases = [
            (place("-0.75", "0", 2.6), place("0", "1", 1e-30)),
            (place("0", "1", 1e-100), place("-1.75", "0", 1e-100)),
            (place("0.25", "0", 1e-3), place("0.2501", "0", 1e-3)),
        ];
        for (from, to) in cases {
            let f = Flight::new(from, to, 1.0);
            let Shape::Pan { from_start, from_end } = f.shape else { panic!("expected a pan") };
            for k in 1..100 {
                let s = f.length * k as f64 / 100.0;
                let (u0, lw0) = from_start.at(s);
                let (u1, lw1) = from_end.at(f.length - s);
                assert!(((u0 + u1) / f.distance - 1.0).abs() < 1e-9, "s {s}: {u0} + {u1} vs {}", f.distance);
                assert!((lw0 - lw1).abs() < 1e-9, "s {s}: ln w {lw0} vs {lw1}");
            }
        }
    }

    /// Two equally deep places 1.3 apart (seahorse valley and c = i).
    fn far_apart() -> (Place, Place) {
        (
            place("-0.743643887037158704752191506114774", "0.131825904205311970493132056385139", 1e-40),
            place("0", "1", 1e-40),
        )
    }

    /// Per frame: (seconds, decades below the widest view, speed / cruise),
    /// with the depth that set that frame's speed (where it started).
    fn profile(from: &Place, to: &Place, speed: f64) -> Vec<(f64, f64, f64)> {
        let mut f = Flight::new(from.clone(), to.clone(), speed);
        let (mut rows, mut t) = (vec![], 0.0);
        loop {
            let (before, depth) = (f.travelled, (f.ln_w_apex - f.ln_w) / LN_10);
            let (_, done) = f.advance(1.0 / 60.0);
            t += 1.0 / 60.0;
            rows.push((t, depth, (f.travelled - before) * 60.0 / f.cruise()));
            if done {
                return rows;
            }
        }
    }

    #[test]
    fn apex_is_rho_squared_over_two_times_the_distance() {
        let (a, b) = far_apart();
        let f = Flight::new(a.clone(), b.clone(), 1.0);
        let expected = RHO * RHO / 2.0 * f.distance;
        let apex = f.ln_w_apex.exp();
        assert!((apex / expected - 1.0).abs() < 1e-6, "apex {apex} vs {expected}");
        // And the flight actually gets there (it lingers, so frames sample it).
        let widest = fly(&a, &b, 4.0).iter().map(|p| p.span).fold(0.0, f64::max);
        assert!(widest / apex > 0.99, "widest frame {widest} vs apex {apex}");
    }

    #[test]
    fn slows_at_the_apex_and_cruises_below_it() {
        let (a, b) = far_apart();
        let rows = profile(&a, &b, 4.0);
        // At the widest view: about the apex fraction of cruising speed.
        let top = rows.iter().min_by(|x, y| x.1.total_cmp(&y.1)).unwrap();
        assert!(top.1 < 0.05, "never reached the apex: {top:?}");
        assert!(top.2 < APEX_SPEED_FRACTION + 0.05, "apex speed {:.2} of cruise", top.2);
        // Well below it (and clear of the start and arrival ramps): cruising.
        let end = rows.last().unwrap().0;
        let deep: Vec<_> = rows.iter().filter(|r| r.1 >= APEX_EASE_DECADES && r.0 > 1.0 && r.0 < end - 1.0).collect();
        assert!(!deep.is_empty() && deep.iter().all(|r| (r.2 - 1.0).abs() < 1e-9), "not cruising below the apex");
        // In between, speed rises with depth.
        let mut easing: Vec<_> = rows.iter().filter(|r| r.1 > 0.1 && r.1 < APEX_EASE_DECADES && r.0 > 1.0 && r.0 < end - 1.0).collect();
        easing.sort_by(|x, y| x.1.total_cmp(&y.1));
        assert!(easing.windows(2).all(|w| w[1].2 >= w[0].2 - 1e-9), "speed not increasing with depth");
    }

    /// Prints a flight's profile, for tuning by eye:
    /// `cargo test --release --lib print_flight_profile -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn print_flight_profile() {
        let (a, b) = far_apart();
        println!("{:>6} {:>18} {:>14}", "secs", "decades below top", "speed/cruise");
        for (i, (t, depth, v)) in profile(&a, &b, DEFAULT_SPEED).into_iter().enumerate() {
            if i % 15 == 0 {
                println!("{t:>6.2} {depth:>18.2} {v:>14.2}");
            }
        }
    }

    #[test]
    fn same_center_only_zooms() {
        let from = place("0", "1", 1e-5);
        let to = place("0", "1", 1e-45);
        let mut f = Flight::new(from.clone(), to.clone(), 4.0);
        // 40 decades at 4 decades/s.
        assert!((f.cruise_seconds() - 10.0).abs() < 1e-9, "{}", f.cruise_seconds());
        while !f.advance(1.0 / 60.0).1 {
            let (p, _) = f.at(f.travelled);
            assert_eq!(apart(&p, &from), 0.0);
        }
    }

    #[test]
    fn flight_time_follows_speed() {
        let from = place("-0.75", "0", 2.6);
        let to = place("0", "1", 1e-30);
        let slow = Flight::new(from.clone(), to.clone(), 1.0).cruise_seconds();
        let fast = Flight::new(from, to, 3.0).cruise_seconds();
        assert!((slow / fast - 3.0).abs() < 1e-9);
        // About one second per decade of zoom, plus a little for the pan.
        assert!((30.0..40.0).contains(&slow), "{slow}");
    }

    #[test]
    fn speed_change_mid_flight_is_smooth() {
        let from = place("-0.75", "0", 2.6);
        let to = place("0", "1", 1e-30);
        let mut f = Flight::new(from.clone(), to.clone(), 2.0);
        let mut path = vec![from];
        for frame in 0..100_000 {
            if frame == 120 {
                f.speed *= 4.0;
            }
            let (p, done) = f.advance(1.0 / 60.0);
            path.push(p);
            if done {
                break;
            }
        }
        assert_eq!(path.last().unwrap(), &to);
        assert_smooth(&path, 0.5);
    }

    #[test]
    fn start_and_end_frames_are_continuous() {
        let from = place("-0.75", "0", 2.6);
        let to = place("0", "1", 1e-10);
        let mut f = Flight::new(from.clone(), to.clone(), 1.5);
        // First frame barely moves.
        let (first, _) = f.advance(1.0 / 60.0);
        assert!(apart(&from, &first) < 0.01 && (first.span / from.span - 1.0).abs() < 0.01);
        // Just before arrival, it's already within a sliver of the target.
        let mut last = first;
        loop {
            let (p, done) = f.advance(1.0 / 60.0);
            if done {
                break;
            }
            last = p;
        }
        assert!(apart(&last, &to) < 0.02 && (last.span / to.span - 1.0).abs() < 0.02, "{}", apart(&last, &to));
    }
}
