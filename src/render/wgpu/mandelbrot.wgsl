// Mandelbrot kernels for the wgpu backend: a WGSL port of
// src/render/metal/mandelbrot.metal, which has the full commentary on the
// arithmetic. Keep the two in step.
//
// Differences from Metal, forced by WGSL:
//   - fixed workgroup size: a workgroup is always GROUP_THREADS threads,
//     packing 256 / (TILE/step)^2 whole tiles;
//   - no SIMD-group reductions: per-tile statistics go straight to
//     workgroup atomics;
//   - `ldexp` is only defined for exponents within f32's range, so
//     floatexp-to-f32 conversion clamps the exponent first;
//   - colorize is a fragment shader (swapchain textures rarely allow storage
//     writes), drawn as one fullscreen triangle.
//
// The df64 routines rely on exact IEEE rounding in a fixed order, which WGSL
// doesn't promise: naga's SPIR-V carries no `NoContraction`, so drivers may
// treat float arithmetic as inexact. Mesa (llvmpipe, at least) simplifies
// (a + b) - a to b, zeroing error terms and silently reducing df64 to f32,
// reorders and fuses the rest, and splits `fma` into a multiply and an add.
// See `hide` for the countermeasure.

// Must match `GpuParams` in src/render/mod.rs.
struct Params {
    center_x: vec2<f32>,   // (hi, lo)
    center_y: vec2<f32>,   // (hi, lo)
    scale: vec2<f32>,      // (hi, lo) complex units per pixel
    width: u32,
    height: u32,
    stats_only: u32,       // nonzero: gather statistics, write no pixels
    prec: u32,             // `precision` is reserved in WGSL
    ref_len: u32,
    scale_m: f32,          // scale as floatexp: scale_m * 2^scale_e
    scale_e: i32,
    step: u32,             // sample every `step` pixels, filling step x step blocks
    ref_off_m: vec2<f32>,  // view center - reference point, as complex floatexp
    ref_off_e: i32,
    sa_n: u32,             // series approximation: start at iteration sa_n...
    sa_a_m: vec2<f32>,     // ...with dz = A * dc, A as complex floatexp
    sa_a_e: i32,
    rough_mu: f32,         // neighbors whose mu differ by more than this are rough
    max_iter: u32,
    one: f32,              // always 1.0 (see `hide`)
}

// Must match `RefPoint` in perturb.rs.
struct RefPoint {
    z: vec2<f32>,  // Z rounded to f32
    m: vec2<f32>,  // Z as floatexp mantissa...
    e: i32,        // ...and exponent
    _pad: i32,
}

// Must match `GpuTileStats` in src/render/mod.rs.
struct TileStats {
    max_escaped: u32,
    escaped_iters: u32,
    near_limit: u32,
    interior: u32,
    rough: u32,
    edge: u32,
    _pad0: u32,
    _pad1: u32,
}

const INSIDE: f32 = -1.0;
const NO_SAMPLE: f32 = -2.0;
const TILE: u32 = 16u; // must match TILE in src/render/mod.rs
const STAT_COUNT: u32 = 6u;
// Must match GROUP_THREADS in src/render/wgpu/mod.rs.
const GROUP_THREADS: u32 = 256u;
// At step 16 each tile is one thread.
const MAX_GROUP_TILES: u32 = 256u;
const ESCAPE_BINS: u32 = 72u;
const ESCAPE_BINS_PER_OCTAVE: f32 = 4.0;
const BAILOUT2: f32 = 65536.0;

// ---------------------------------------------------------------------------
// Double-float arithmetic
// ---------------------------------------------------------------------------

// An optimization barrier: returns x unchanged, by XOR-ing its bits with a
// zero the compiler can't know (derived from the uniform `p.one` = 1.0).
// Float rewrites can't see through integer operations, and every rounded
// step of df64 goes through here, so each is computed exactly as written.
// (Multiplying by `p.one` isn't enough: with reassociation allowed, the
// compiler factors the multiplies out.) Checked against a CPU emulation:
// the results are bit-identical. `p` is declared further down.
fn hide(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ (bitcast<u32>(p.one) ^ 0x3f800000u)); }

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = hide(a + b);
    let e = hide(b - hide(s - a));
    return vec2(s, e);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = hide(a + b);
    let bb = hide(s - a);
    let e = hide(hide(a - hide(s - bb)) + hide(b - bb));
    return vec2(s, e);
}

// Veltkamp split: a = hi + lo exactly, each with at most 12 significant bits.
fn split(a: f32) -> vec2<f32> {
    let c = hide(4097.0 * a);
    let hi = hide(c - hide(c - a));
    return vec2(hi, hide(a - hi));
}

// Dekker's product rather than fma(a, b, -p): Mesa may split an inexact
// fma into a multiply and an add, which makes the error term always 0.
// (The partial products of 12-bit halves are exact.)
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let prod = hide(a * b);
    let x = split(a);
    let y = split(b);
    let e = hide(hide(hide(hide(x.x * y.x - prod) + x.x * y.y) + x.y * y.x) + x.y * y.y);
    return vec2(prod, e);
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    s.y = hide(s.y + t.x);
    s = quick_two_sum(s.x, s.y);
    s.y = hide(s.y + t.y);
    return quick_two_sum(s.x, s.y);
}

fn df_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return df_add(a, -b); }

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var p = two_prod(a.x, b.x);
    p.y = hide(p.y + hide(hide(a.x * b.y) + hide(a.y * b.x)));
    return quick_two_sum(p.x, p.y);
}

fn df_mul_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    var p = two_prod(a.x, b);
    p.y = hide(p.y + hide(a.y * b));
    return quick_two_sum(p.x, p.y);
}

fn df_sqr(a: vec2<f32>) -> vec2<f32> {
    var p = two_prod(a.x, a.x);
    p.y = hide(p.y + hide(2.0 * a.x * a.y));
    return quick_two_sum(p.x, p.y);
}

// ---------------------------------------------------------------------------
// Escape-time iteration. Returns smooth iteration count, or -1 for "inside".
// ---------------------------------------------------------------------------

fn smooth_mu(n: u32, r2: f32) -> f32 {
    return f32(n) + 1.0 - log2(0.5 * log(r2));
}

fn iterate_f32(c: vec2<f32>, max_iter: u32, period_eps: f32) -> f32 {
    let xq = c.x - 0.25;
    let q = xq * xq + c.y * c.y;
    if (q * (q + xq) <= 0.25 * c.y * c.y) { return -1.0; }
    if ((c.x + 1.0) * (c.x + 1.0) + c.y * c.y <= 0.0625) { return -1.0; }

    var x = 0.0; var y = 0.0; var x2 = 0.0; var y2 = 0.0;
    var saved = vec2(0.0);
    var check_at = 16u;
    for (var n = 0u; n < max_iter; n++) {
        y = 2.0 * x * y + c.y;
        x = x2 - y2 + c.x;
        x2 = x * x;
        y2 = y * y;
        let r2 = x2 + y2;
        if (r2 > BAILOUT2) { return smooth_mu(n, r2); }

        if (n > 16u && abs(x - saved.x) + abs(y - saved.y) < period_eps) { return -1.0; }
        if (n == check_at) { saved = vec2(x, y); check_at *= 2u; }
    }
    return -1.0;
}

fn iterate_df(cx: vec2<f32>, cy: vec2<f32>, max_iter: u32, period_eps: f32) -> f32 {
    var x = vec2(0.0); var y = vec2(0.0); var x2 = vec2(0.0); var y2 = vec2(0.0);
    var sx = vec2(0.0); var sy = vec2(0.0);
    var check_at = 16u;
    for (var n = 0u; n < max_iter; n++) {
        let xy = df_mul(x, y);
        y = df_add(2.0 * xy, cy);
        x = df_add(df_sub(x2, y2), cx);
        x2 = df_sqr(x);
        y2 = df_sqr(y);
        let r2 = x2.x + y2.x;
        if (r2 > BAILOUT2) { return smooth_mu(n, r2); }

        if (n > 16u) {
            let d = abs(df_sub(x, sx).x) + abs(df_sub(y, sy).x);
            if (d < period_eps) { return -1.0; }
        }
        if (n == check_at) { sx = x; sy = y; check_at *= 2u; }
    }
    return -1.0;
}

// ---------------------------------------------------------------------------
// Complex floatexp: value = m * 2^e, max(|m.x|, |m.y|) in [0.5, 1).
// ---------------------------------------------------------------------------

const FE_ZERO_EXP: i32 = -1000000; // must match FE_ZERO_EXP in perturb.rs

struct cfe { m: vec2<f32>, e: i32 }

fn cfe_make(m: vec2<f32>, e: i32) -> cfe {
    let a = max(abs(m.x), abs(m.y));
    if (a == 0.0) { return cfe(vec2(0.0), FE_ZERO_EXP); }
    let k = frexp(a).exp;
    // Normal a: -k is within ldexp's defined range.
    return cfe(vec2(ldexp(m.x, -k), ldexp(m.y, -k)), e + k);
}

fn cfe_to_float2(a: cfe) -> vec2<f32> {
    // The mantissa is below 1, so below 2^-160 is zero and 2^127 is already
    // past any bailout; the clamp keeps ldexp's exponent in its defined range.
    let e = clamp(a.e, -160, 127);
    return vec2(ldexp(a.m.x, e), ldexp(a.m.y, e));
}

fn cfe_add(a_in: cfe, b_in: cfe) -> cfe {
    var a = a_in; var b = b_in;
    if (a.e < b.e) { let t = a; a = b; b = t; }
    let d = b.e - a.e;
    if (d < -40) { return a; }
    return cfe_make(a.m + vec2(ldexp(b.m.x, d), ldexp(b.m.y, d)), a.e);
}

fn cfe_mul(a: cfe, b: cfe) -> cfe {
    let m = vec2(a.m.x * b.m.x - a.m.y * b.m.y, a.m.x * b.m.y + a.m.y * b.m.x);
    return cfe_make(m, a.e + b.e);
}

fn cfe_l1_less(a: cfe, b: cfe) -> bool {
    let la = abs(a.m.x) + abs(a.m.y);
    let lb = abs(b.m.x) + abs(b.m.y);
    if (lb == 0.0) { return false; }
    if (la == 0.0) { return true; }
    return ldexp(la, clamp(a.e - b.e, -64, 64)) < lb;
}

// ---------------------------------------------------------------------------
// Perturbation with rebasing (see the Metal shader for the full story).
// ---------------------------------------------------------------------------

const EXTENDED_TO_FAST_EXP: i32 = -100;
const FAST_TO_EXTENDED: f32 = 7.70371978e-34; // 2^-110

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> refs: array<RefPoint>;
@group(0) @binding(2) var<storage, read_write> tile_stats: array<TileStats>;
@group(0) @binding(3) var<storage, read> tiles: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> escapes: array<atomic<u32>, ESCAPE_BINS>;
@group(0) @binding(5) var out: texture_storage_2d<r32float, write>;

fn ref_fe(m: u32) -> cfe { return cfe(refs[m].m, refs[m].e); }

fn iterate_perturb(dc: cfe, ref_len: u32, max_iter: u32, skip: u32, skip_a: cfe) -> f32 {
    let dcf = cfe_to_float2(dc);
    var dzx = cfe(vec2(0.0), FE_ZERO_EXP);
    if (skip > 0u) { dzx = cfe_mul(skip_a, dc); }
    var dz = vec2(0.0);
    var m = skip;
    var n = skip;
    var extended = true;

    while (n < max_iter) {
        if (extended) {
            for (; n < max_iter; n++) {
                let zr = ref_fe(m);
                let t = cfe_add(cfe(zr.m, zr.e + 1), dzx);   // 2Z + dz
                dzx = cfe_add(cfe_mul(t, dzx), dc);
                m++;

                let z = cfe_add(ref_fe(m), dzx);
                let zf = cfe_to_float2(z);
                let r2 = dot(zf, zf);
                if (r2 > BAILOUT2) { return smooth_mu(n, r2); }

                if (cfe_l1_less(z, dzx) || m >= ref_len - 1u) {
                    dzx = z;
                    m = 0u;
                }
                if (dzx.e > EXTENDED_TO_FAST_EXP) {
                    dz = cfe_to_float2(dzx);
                    extended = false;
                    n++;
                    break;
                }
            }
        } else {
            for (; n < max_iter; n++) {
                let t = 2.0 * refs[m].z + dz;
                let next = vec2(t.x * dz.x - t.y * dz.y, t.x * dz.y + t.y * dz.x) + dcf;
                if (abs(next.x) + abs(next.y) < FAST_TO_EXTENDED) {
                    dzx = cfe_make(dz, 0);
                    extended = true;
                    break;
                }
                dz = next;
                m++;

                let z = refs[m].z + dz;
                let r2 = dot(z, z);
                if (r2 > BAILOUT2) { return smooth_mu(n, r2); }

                var dz1 = abs(dz.x) + abs(dz.y);
                if (abs(z.x) + abs(z.y) < dz1 || m >= ref_len - 1u) {
                    dz = z;
                    m = 0u;
                    dz1 = abs(dz.x) + abs(dz.y);
                }
                if (dz1 < FAST_TO_EXTENDED) {
                    dzx = cfe_make(dz, 0);
                    extended = true;
                    n++;
                    break;
                }
            }
        }
    }
    return -1.0;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// Workgroup memory starts zeroed (a WGSL guarantee), so the accumulators
// need no explicit clearing.
var<workgroup> shared_mu: array<f32, GROUP_THREADS>;
var<workgroup> acc: array<atomic<u32>, 1536>; // MAX_GROUP_TILES * STAT_COUNT
var<workgroup> group_escapes: array<atomic<u32>, ESCAPE_BINS>;

@compute @workgroup_size(256)
fn mandelbrot(@builtin(local_invocation_index) li: u32,
              @builtin(workgroup_id) wid: vec3<u32>,
              @builtin(num_workgroups) groups: vec3<u32>) {
    // Each workgroup holds whole tiles: thread `li` is sample `in_tile` of
    // the workgroup's tile `slot`.
    let per_side = TILE / p.step;
    let tile_threads = per_side * per_side;
    let slot = li / tile_threads;
    let in_tile = li % tile_threads;
    let lx = in_tile % per_side;
    let ly = in_tile / per_side;
    let tile = (wid.y * groups.x + wid.x) * (GROUP_THREADS / tile_threads) + slot;
    let have_tile = tile < arrayLength(&tiles);
    var origin = vec2(0u);
    if (have_tile) { origin = tiles[tile]; }

    let max_iter = p.max_iter;
    let x0 = origin.x + lx * p.step;
    let y0 = origin.y + ly * p.step;
    // No early return: every thread must reach the barriers.
    let valid = have_tile && x0 < p.width && y0 < p.height;

    var mu = -1.0;
    if (valid) {
        let half_block = 0.5 * f32(p.step);
        let px = f32(x0) + half_block - 0.5 * f32(p.width);
        let py = 0.5 * f32(p.height) - (f32(y0) + half_block);

        let period_eps = max(p.scale.x * 1e-3, 1e-30);

        if (p.prec == 0u) {
            let c = vec2(p.center_x.x + px * p.scale.x, p.center_y.x + py * p.scale.x);
            mu = iterate_f32(c, max_iter, max(period_eps, 1e-7));
        } else if (p.prec == 1u) {
            let cx = df_add(p.center_x, df_mul_f(p.scale, px));
            let cy = df_add(p.center_y, df_mul_f(p.scale, py));
            mu = iterate_df(cx, cy, max_iter, period_eps);
        } else if (p.ref_len >= 2u) {
            let dc = cfe_add(cfe_make(vec2(px, py) * p.scale_m, p.scale_e), cfe(p.ref_off_m, p.ref_off_e));
            mu = iterate_perturb(dc, p.ref_len, max_iter, p.sa_n, cfe(p.sa_a_m, p.sa_a_e));
        }

        if (p.stats_only == 0u) {
            let value = select(INSIDE, mu, mu >= 0.0);
            let x1 = min(x0 + p.step, p.width);
            let y1 = min(y0 + p.step, p.height);
            for (var y = y0; y < y1; y++) {
                for (var x = x0; x < x1; x++) {
                    textureStore(out, vec2(x, y), vec4(value));
                }
            }
        }
    }

    shared_mu[li] = select(NO_SAMPLE, mu, valid);
    workgroupBarrier();

    let escaped = valid && mu >= 0.0;
    let n = select(0u, u32(mu), escaped);
    var rough = 0u;
    var edge = 0u;
    if (valid) {
        // Right and lower neighbors within the tile.
        var right = NO_SAMPLE;
        if (lx + 1u < per_side) { right = shared_mu[li + 1u]; }
        var below = NO_SAMPLE;
        if (ly + 1u < per_side) { below = shared_mu[li + per_side]; }
        for (var k = 0u; k < 2u; k++) {
            let other = select(below, right, k == 0u);
            if (other == NO_SAMPLE) { continue; }
            let other_escaped = other >= 0.0;
            if (escaped && other_escaped) {
                rough += u32(abs(mu - other) > p.rough_mu);
            } else {
                edge += u32(escaped != other_escaped);
            }
        }
    }
    let base = slot * STAT_COUNT;
    if (escaped) {
        atomicMax(&acc[base], n);
        atomicAdd(&acc[base + 1u], n);
        if (n > max_iter / 4u * 3u) { atomicAdd(&acc[base + 2u], 1u); }
        var bin = 0u;
        if (n != 0u) { bin = min(u32(log2(f32(n)) * ESCAPE_BINS_PER_OCTAVE), ESCAPE_BINS - 1u); }
        atomicAdd(&group_escapes[bin], 1u);
    }
    if (valid && !escaped) { atomicAdd(&acc[base + 3u], 1u); }
    if (rough != 0u) { atomicAdd(&acc[base + 4u], rough); }
    if (edge != 0u) { atomicAdd(&acc[base + 5u], edge); }
    workgroupBarrier();

    if (have_tile && in_tile == 0u) {
        tile_stats[tile] = TileStats(
            atomicLoad(&acc[base]),
            atomicLoad(&acc[base + 1u]),
            atomicLoad(&acc[base + 2u]),
            atomicLoad(&acc[base + 3u]),
            atomicLoad(&acc[base + 4u]),
            atomicLoad(&acc[base + 5u]),
            0u, 0u,
        );
    }
    if (li < ESCAPE_BINS) {
        let count = atomicLoad(&group_escapes[li]);
        if (count != 0u) { atomicAdd(&escapes[li], count); }
    }
}

// ---------------------------------------------------------------------------
// Reprojection
// ---------------------------------------------------------------------------

// Must match `GpuReprojection` in src/render/mod.rs.
struct Reprojection {
    ratio: f32,
    _pad: f32,
    shift: vec2<f32>,
    old_w: u32,
    old_h: u32,
}

@group(0) @binding(0) var<uniform> r: Reprojection;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

fn texel(q: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(src)) - 1;
    return textureLoad(src, clamp(q, vec2(0), last), 0).r;
}

@compute @workgroup_size(16, 16)
fn reproject(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.xy;
    let size = textureDimensions(dst);
    if (gid.x >= size.x || gid.y >= size.y) { return; }

    let old_size = vec2(f32(r.old_w), f32(r.old_h));
    let u = 0.5 * old_size + (vec2<f32>(gid) + 0.5 - 0.5 * vec2<f32>(size)) * r.ratio + r.shift;
    var value = INSIDE;
    if (all(u >= vec2(0.0)) && all(u <= old_size)) {
        let q = u - 0.5;
        let i = vec2<i32>(floor(q));
        let f = q - vec2<f32>(i);
        let a = texel(i);
        let b = texel(i + vec2(1, 0));
        let c = texel(i + vec2(0, 1));
        let d = texel(i + vec2(1, 1));
        if (min(min(a, b), min(c, d)) < 0.0) {
            value = texel(vec2<i32>(floor(u)));
        } else {
            value = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
    }
    textureStore(dst, gid, vec4(value));
}

// ---------------------------------------------------------------------------
// Colorize: a fullscreen triangle whose fragments color the mu canvas.
// ---------------------------------------------------------------------------

// Must match `GpuColors` in src/render/mod.rs.
struct Colors {
    palette: u32,
    density: f32,
    offset: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> colors: Colors;
@group(0) @binding(1) var mu_tex: texture_2d<f32>;

fn cosine_palette(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318530718 * (c * t + d));
}

fn classic_palette(t_in: f32) -> vec3<f32> {
    var pos = array<f32, 6>(0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0);
    var col = array<vec3<f32>, 6>(
        vec3(0.0,   7.0, 100.0) / 255.0,
        vec3(32.0, 107.0, 203.0) / 255.0,
        vec3(237.0, 255.0, 255.0) / 255.0,
        vec3(255.0, 170.0,   0.0) / 255.0,
        vec3(0.0,    2.0,   0.0) / 255.0,
        vec3(0.0,    7.0, 100.0) / 255.0,
    );
    let t = fract(t_in);
    for (var i = 0; i < 5; i++) {
        if (t <= pos[i + 1]) {
            let k = (t - pos[i]) / (pos[i + 1] - pos[i]);
            return mix(col[i], col[i + 1], smoothstep(0.0, 1.0, k));
        }
    }
    return col[5];
}

// Every palette must repeat after PALETTE_PERIOD (10) cycles of t.
fn palette(which: u32, t: f32) -> vec3<f32> {
    switch (which) {
        case 0u: { return classic_palette(t); }
        case 1u: { return cosine_palette(t, vec3(0.5), vec3(0.5),
                                         vec3(1.0, 0.7, 0.4), vec3(0.0, 0.15, 0.20)); }
        case 2u: { return cosine_palette(t, vec3(0.2, 0.4, 0.6), vec3(0.2, 0.3, 0.4),
                                         vec3(1.0), vec3(0.0, 0.1, 0.2)); }
        case 3u: { return cosine_palette(t, vec3(0.5), vec3(0.5),
                                         vec3(2.0, 1.0, 0.0), vec3(0.5, 0.20, 0.25)); }
        case 4u: { return cosine_palette(t, vec3(0.8, 0.5, 0.4), vec3(0.2, 0.4, 0.2),
                                         vec3(2.0, 1.0, 1.0), vec3(0.0, 0.25, 0.25)); }
        default: { return vec3(0.5 + 0.5 * cos(6.28318530718 * t)); }
    }
}

@vertex
fn fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // One triangle covering the viewport: (-1,-1), (3,-1), (-1,3).
    let uv = vec2(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn colorize(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let g = vec2<u32>(pos.xy);
    let size = textureDimensions(mu_tex);
    // The window may briefly be larger than the canvas during a resize.
    var m = INSIDE;
    if (g.x < size.x && g.y < size.y) { m = textureLoad(mu_tex, g, 0).r; }
    var rgb = vec3(0.0);
    if (m >= 0.0) { rgb = palette(colors.palette, m * 0.02 * colors.density + colors.offset); }
    return vec4(rgb, 1.0);
}
