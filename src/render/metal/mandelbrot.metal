// Mandelbrot compute kernel.
//
// Writes straight into the target texture (the CAMetalLayer drawable, or an
// offscreen texture for captures). Three arithmetic paths:
//   precision == 0 : plain f32
//   precision == 1 : double-float ("df64"), each value is an (hi, lo) f32 pair
//   precision == 2 : perturbation against a CPU reference orbit (buffer 1)
//
// The df64 routines rely on exact IEEE rounding, so this library must be
// compiled with MTLMathModeSafe (no fast-math reassociation).

#include <metal_stdlib>
using namespace metal;

// Must match `GpuParams` in src/render/mod.rs.
struct Params {
    float2 center_x;   // (hi, lo)
    float2 center_y;   // (hi, lo)
    float2 scale;      // (hi, lo) complex units per pixel
    uint   width;
    uint   height;
    uint   stats_only; // nonzero: gather statistics, write no pixels
    uint   precision;
    uint   ref_len;
    float  scale_m;    // scale as floatexp: scale_m * 2^scale_e
    int    scale_e;
    uint   step;       // sample every `step` pixels, filling step x step blocks
    float2 ref_off_m;  // view center - reference point, as complex floatexp
    int    ref_off_e;
    uint   sa_n;       // series approximation: start at iteration sa_n...
    float2 sa_a_m;     // ...with dz = A * dc, A as complex floatexp
    int    sa_a_e;
    float  rough_mu;   // neighbors whose mu differ by more than this are rough
};

// One tile to compute. Must match `TileJob` in src/render/mod.rs.
struct TileJob {
    uint2 origin;   // top-left pixel
    uint  max_iter; // iteration limit for this tile (at most the frame's, which the reference covers)
    uint  _pad;
};

/// Canvas value for samples that never escaped (in the set, or not computed).
constant float INSIDE = -1.0;

constant uint TILE = 16; // must match TILE in src/render/mod.rs

// Statistics for one tile of a batch. Must match `GpuTileStats` in
// render/metal/mod.rs.
struct TileStats {
    uint max_escaped;   // highest iteration count among escaped samples
    uint escaped_iters; // sum of the escaped samples' iteration counts
    uint near_limit;    // escaped samples needing > 3/4 of the tile's limit
    uint interior;      // samples that never escaped
    uint rough;         // adjacent pairs, both escaped, mu differing > rough_mu
    uint edge;          // adjacent pairs where exactly one sample escaped
    uint _pad[2];
};
constant uint STAT_COUNT = 6;

// Largest threadgroup: one full-resolution tile (16 x 16), and at most 64
// tiles per threadgroup at coarser steps (see `encode` in metal/mod.rs).
constant uint MAX_GROUP_THREADS = 256;
constant uint MAX_GROUP_TILES = 64;
/// Shared-memory marker for samples outside the canvas.
constant float NO_SAMPLE = -2.0;

constant float BAILOUT2 = 65536.0; // |z| > 256: large radius for smooth coloring

// ---------------------------------------------------------------------------
// Double-float arithmetic (Dekker / Knuth error-free transforms)
// ---------------------------------------------------------------------------

typedef float2 df;

inline df quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return df(s, e);
}

inline df two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float e = (a - (s - bb)) + (b - bb);
    return df(s, e);
}

inline df two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);
    return df(p, e);
}

inline df df_add(df a, df b) {
    df s = two_sum(a.x, b.x);
    df t = two_sum(a.y, b.y);
    s.y += t.x;
    s = quick_two_sum(s.x, s.y);
    s.y += t.y;
    return quick_two_sum(s.x, s.y);
}

inline df df_neg(df a) { return df(-a.x, -a.y); }
inline df df_sub(df a, df b) { return df_add(a, df_neg(b)); }

inline df df_mul(df a, df b) {
    df p = two_prod(a.x, b.x);
    p.y += a.x * b.y + a.y * b.x;
    return quick_two_sum(p.x, p.y);
}

inline df df_mul_f(df a, float b) {
    df p = two_prod(a.x, b);
    p.y += a.y * b;
    return quick_two_sum(p.x, p.y);
}

inline df df_sqr(df a) {
    df p = two_prod(a.x, a.x);
    p.y += 2.0 * a.x * a.y;
    return quick_two_sum(p.x, p.y);
}

// ---------------------------------------------------------------------------
// Escape-time iteration. Returns smooth iteration count, or -1 for "inside".
// ---------------------------------------------------------------------------

inline float smooth_mu(uint n, float r2) {
    return float(n) + 1.0 - log2(0.5 * log(r2));
}

float iterate_f32(float2 c, uint max_iter, float period_eps) {
    // Main cardioid and period-2 bulb: skip the (expensive) interior.
    float xq = c.x - 0.25;
    float q = xq * xq + c.y * c.y;
    if (q * (q + xq) <= 0.25 * c.y * c.y) return -1.0;
    if ((c.x + 1.0) * (c.x + 1.0) + c.y * c.y <= 0.0625) return -1.0;

    float x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0;
    float2 saved = float2(0.0);
    uint check_at = 16;
    for (uint n = 0; n < max_iter; ++n) {
        y = 2.0 * x * y + c.y;
        x = x2 - y2 + c.x;
        x2 = x * x;
        y2 = y * y;
        float r2 = x2 + y2;
        if (r2 > BAILOUT2) return smooth_mu(n, r2);

        // Brent-style periodicity check: settled into a cycle => inside.
        if (n > 16 && abs(x - saved.x) + abs(y - saved.y) < period_eps) return -1.0;
        if (n == check_at) { saved = float2(x, y); check_at *= 2; }
    }
    return -1.0;
}

float iterate_df(df cx, df cy, uint max_iter, float period_eps) {
    df x = df(0.0), y = df(0.0), x2 = df(0.0), y2 = df(0.0);
    df sx = df(0.0), sy = df(0.0);
    uint check_at = 16;
    for (uint n = 0; n < max_iter; ++n) {
        df xy = df_mul(x, y);
        y = df_add(df(2.0 * xy.x, 2.0 * xy.y), cy);
        x = df_add(df_sub(x2, y2), cx);
        x2 = df_sqr(x);
        y2 = df_sqr(y);
        float r2 = x2.x + y2.x;
        if (r2 > BAILOUT2) return smooth_mu(n, r2);

        if (n > 16) {
            float d = abs(df_sub(x, sx).x) + abs(df_sub(y, sy).x);
            if (d < period_eps) return -1.0;
        }
        if (n == check_at) { sx = x; sy = y; check_at *= 2; }
    }
    return -1.0;
}

// ---------------------------------------------------------------------------
// Complex floatexp: value = m * 2^e, one exponent shared by both components,
// max(|m.x|, |m.y|) kept in [0.5, 1). Same 24-bit precision as f32, but the
// exponent range of an int, so values like 1e-200 are fine.
// ---------------------------------------------------------------------------

constant int FE_ZERO_EXP = -1000000; // must match FE_ZERO_EXP in perturb.rs

struct cfe { float2 m; int e; };

inline cfe cfe_make(float2 m, int e) {
    float a = max(abs(m.x), abs(m.y));
    if (a == 0.0) return cfe{float2(0.0), FE_ZERO_EXP};
    int k;
    frexp(a, k);
    return cfe{float2(ldexp(m.x, -k), ldexp(m.y, -k)), e + k};
}

inline float2 cfe_to_float2(cfe a) {
    return float2(ldexp(a.m.x, a.e), ldexp(a.m.y, a.e)); // underflows to 0 if tiny
}

inline cfe cfe_add(cfe a, cfe b) {
    if (a.e < b.e) { cfe t = a; a = b; b = t; }
    int d = b.e - a.e;               // <= 0: shift the smaller operand down
    if (d < -40) return a;           // below a's precision (also catches zero)
    return cfe_make(a.m + float2(ldexp(b.m.x, d), ldexp(b.m.y, d)), a.e);
}

inline cfe cfe_mul(cfe a, cfe b) {
    float2 m = float2(a.m.x * b.m.x - a.m.y * b.m.y, a.m.x * b.m.y + a.m.y * b.m.x);
    return cfe_make(m, a.e + b.e);
}

/// |a| < |b| using L1 norms.
inline bool cfe_l1_less(cfe a, cfe b) {
    float la = abs(a.m.x) + abs(a.m.y);
    float lb = abs(b.m.x) + abs(b.m.y);
    if (lb == 0.0) return false;
    if (la == 0.0) return true;
    return ldexp(la, clamp(a.e - b.e, -64, 64)) < lb;
}

// ---------------------------------------------------------------------------
// Perturbation with rebasing (Zhuoran, 2021).
//
// The reference orbit Z (computed on the CPU at high precision for the view
// center C) is followed by index m; this pixel tracks only dz = z - Z_m, where
// its c = C + dc:
//
//     dz <- (2 Z_m + dz) * dz + dc
//
// Rebasing: if the pixel's actual z = Z_m + dz becomes smaller than dz, the
// reference is a poor guide here (dz would suffer catastrophic cancellation).
// So restart against the *beginning* of the reference: dz = z, m = 0. Z_0 = 0,
// so this is exact. Same when the reference orbit runs out (it escaped, or hit
// max_iter). With rebasing a single reference serves the whole image, with no
// glitch detection or secondary references.
//
// Two phases, because at deep zoom dz starts out far below f32's range:
//   extended: dz in floatexp until |dz| > 2^-100 (typically a few hundred or
//             thousand iterations at very deep zoom; skipped entirely above
//             ~1e30 since dz starts out large enough)
//   fast:     plain f32, as long as |dz| stays above 2^-110
// dc may underflow to 0 in the fast phase. That's harmless while 2Z·dz
// dominates the step, but not where the reference passes through ~0 (a
// nucleus, every period): there the step is just dz² + dc, which can be far
// too small for f32 and would round to exactly 0, leaving the pixel stuck on
// the reference's periodic orbit forever. So a fast step whose result would
// be that small is not taken; the extended phase redoes it with dc intact.
// ---------------------------------------------------------------------------

// Must match `RefPoint` in perturb.rs.
struct RefPoint {
    float2 z;     // Z rounded to f32
    float2 m;     // Z as floatexp mantissa...
    int    e;     // ...and exponent
    int    _pad;
};

constant int   EXTENDED_TO_FAST_EXP = -100;     // |dz| >~ 2^-100
constant float FAST_TO_EXTENDED     = 7.70371978e-34; // 2^-110

float iterate_perturb(cfe dc, device const RefPoint* ref, uint ref_len, uint max_iter,
                      uint skip, cfe skip_a) {
    float2 dcf = cfe_to_float2(dc);
    // Series approximation: the first `skip` iterations are linear in dc,
    // so jump straight to dz = A * dc (the CPU checked the error bound).
    cfe dzx = skip > 0 ? cfe_mul(skip_a, dc) : cfe{float2(0.0), FE_ZERO_EXP};
    float2 dz = float2(0.0);
    uint m = skip;
    uint n = skip;
    bool extended = true;

    while (n < max_iter) {
        if (extended) {
            for (; n < max_iter; ++n) {
                cfe Z = cfe{ref[m].m, ref[m].e};
                cfe t = cfe_add(cfe{Z.m, Z.e + 1}, dzx);   // 2Z + dz
                dzx = cfe_add(cfe_mul(t, dzx), dc);
                ++m;

                cfe z = cfe_add(cfe{ref[m].m, ref[m].e}, dzx);
                float2 zf = cfe_to_float2(z);
                float r2 = dot(zf, zf);
                if (r2 > BAILOUT2) return smooth_mu(n, r2);

                if (cfe_l1_less(z, dzx) || m >= ref_len - 1) {
                    dzx = z;
                    m = 0;
                }
                if (dzx.e > EXTENDED_TO_FAST_EXP) {
                    dz = cfe_to_float2(dzx);
                    extended = false;
                    ++n;
                    break;
                }
            }
        } else {
            for (; n < max_iter; ++n) {
                float2 t = 2.0 * ref[m].z + dz;
                float2 next = float2(t.x * dz.x - t.y * dz.y, t.x * dz.y + t.y * dz.x) + dcf;
                if (abs(next.x) + abs(next.y) < FAST_TO_EXTENDED) {
                    // Too small for f32: take this step in floatexp instead
                    // (dz itself is still a normal f32, so this is exact).
                    dzx = cfe_make(dz, 0);
                    extended = true;
                    break;
                }
                dz = next;
                ++m;

                float2 z = ref[m].z + dz;
                float r2 = dot(z, z);
                if (r2 > BAILOUT2) return smooth_mu(n, r2);

                // L1 norms: squares of tiny deltas would underflow f32.
                float dz1 = abs(dz.x) + abs(dz.y);
                if (abs(z.x) + abs(z.y) < dz1 || m >= ref_len - 1) {
                    dz = z;
                    m = 0;
                    dz1 = abs(dz.x) + abs(dz.y);
                }
                if (dz1 < FAST_TO_EXTENDED) {
                    dzx = cfe_make(dz, 0);
                    extended = true;
                    ++n;
                    break;
                }
            }
        }
    }
    return -1.0;
}

// ---------------------------------------------------------------------------
// Coloring
// ---------------------------------------------------------------------------

// Inigo Quilez cosine palette: a + b * cos(2pi * (c * t + d))
inline float3 cosine_palette(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(6.28318530718 * (c * t + d));
}

// The well-known "Ultra Fractal" style gradient, piecewise smooth.
float3 classic_palette(float t) {
    const float  pos[6] = { 0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0 };
    const float3 col[6] = {
        float3(0.0,   7.0, 100.0) / 255.0,
        float3(32.0, 107.0, 203.0) / 255.0,
        float3(237.0, 255.0, 255.0) / 255.0,
        float3(255.0, 170.0,   0.0) / 255.0,
        float3(0.0,    2.0,   0.0) / 255.0,
        float3(0.0,    7.0, 100.0) / 255.0,
    };
    t = fract(t);
    for (int i = 0; i < 5; ++i) {
        if (t <= pos[i + 1]) {
            float k = (t - pos[i]) / (pos[i + 1] - pos[i]);
            return mix(col[i], col[i + 1], smoothstep(0.0, 1.0, k));
        }
    }
    return col[5];
}

// Every palette must repeat after PALETTE_PERIOD (10) cycles of t, so color
// cycling can wrap its offset seamlessly: cosine frequencies × 10 must be
// whole numbers (Fire's 0.7 and 0.4 are why it's 10, not 1). Checked by the
// `every_palette_repeats_after_the_palette_period` GPU test.
float3 palette(uint which, float t) {
    switch (which) {
        case 0: return classic_palette(t);
        case 1: return cosine_palette(t, float3(0.5), float3(0.5),
                                      float3(1.0, 0.7, 0.4), float3(0.0, 0.15, 0.20));
        case 2: return cosine_palette(t, float3(0.2, 0.4, 0.6), float3(0.2, 0.3, 0.4),
                                      float3(1.0), float3(0.0, 0.1, 0.2));
        case 3: return cosine_palette(t, float3(0.5), float3(0.5),
                                      float3(2.0, 1.0, 0.0), float3(0.5, 0.20, 0.25));
        case 4: return cosine_palette(t, float3(0.8, 0.5, 0.4), float3(0.2, 0.4, 0.2),
                                      float3(2.0, 1.0, 1.0), float3(0.0, 0.25, 0.25));
        default: return float3(0.5 + 0.5 * cos(6.28318530718 * t));
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// Writes each sample's smooth iteration count mu (or INSIDE) to `out`, a
// single-channel float canvas; `colorize` turns mu into colors when the
// canvas is shown, so color changes never require recomputation.
kernel void mandelbrot(texture2d<float, access::write> out [[texture(0)]],
                       constant Params& p                 [[buffer(0)]],
                       device const RefPoint* ref         [[buffer(1)]],
                       device TileStats* tile_stats       [[buffer(2)]],
                       device const TileJob* tiles        [[buffer(3)]],
                       uint3 gid                          [[thread_position_in_grid]],
                       uint3 lid                          [[thread_position_in_threadgroup]],
                       uint simd_width                    [[threads_per_simdgroup]])
{
    // Grid: (TILE/step, TILE/step, tile count). Each thread computes one
    // sample at the center of a step x step block within its tile.
    TileJob job = tiles[gid.z];
    uint max_iter = job.max_iter;
    uint x0 = job.origin.x + gid.x * p.step;
    uint y0 = job.origin.y + gid.y * p.step;
    // No early return: every thread must take part in the SIMD reductions.
    bool valid = gid.x * p.step < TILE && gid.y * p.step < TILE && x0 < p.width && y0 < p.height;

    float mu = -1.0;
    if (valid) {
        // Sample offset from the view center, in pixels; exact in f32.
        float half_block = 0.5 * float(p.step);
        float px = float(x0) + half_block - 0.5 * float(p.width);
        float py = 0.5 * float(p.height) - (float(y0) + half_block);

        // A cycle is "detected" when the orbit returns to well within a pixel.
        float period_eps = max(p.scale.x * 1e-3, 1e-30);

        if (p.precision == 0) {
            float2 c = float2(p.center_x.x + px * p.scale.x,
                              p.center_y.x + py * p.scale.x);
            mu = iterate_f32(c, max_iter, max(period_eps, 1e-7));
        } else if (p.precision == 1) {
            df cx = df_add(p.center_x, df_mul_f(p.scale, px));
            df cy = df_add(p.center_y, df_mul_f(p.scale, py));
            mu = iterate_df(cx, cy, max_iter, period_eps);
        } else {
            // dc = c - C: pixel offset from the view center, plus the view
            // center's offset from the (possibly off-center) reference point.
            cfe dc = cfe_add(cfe_make(float2(px, py) * p.scale_m, p.scale_e),
                             cfe{p.ref_off_m, p.ref_off_e});
            mu = p.ref_len >= 2 ? iterate_perturb(dc, ref, p.ref_len, max_iter,
                                                  p.sa_n, cfe{p.sa_a_m, p.sa_a_e})
                                : -1.0;
        }

        if (!p.stats_only) {
            float value = mu >= 0.0 ? mu : INSIDE;
            uint x1 = min(x0 + p.step, p.width);
            uint y1 = min(y0 + p.step, p.height);
            for (uint y = y0; y < y1; ++y) {
                for (uint x = x0; x < x1; ++x) {
                    out.write(float4(value), uint2(x, y));
                }
            }
        }
    }

    // Per-tile statistics. A threadgroup holds whole tiles (slice lid.z is
    // one tile), so neighbors are exchanged through threadgroup memory and
    // each tile is reduced on its own.
    uint per_side = TILE / p.step;
    uint flat = (lid.z * per_side + lid.y) * per_side + lid.x;
    threadgroup float shared_mu[MAX_GROUP_THREADS];
    threadgroup atomic_uint acc[MAX_GROUP_TILES * STAT_COUNT];
    threadgroup atomic_uint* mine = &acc[lid.z * STAT_COUNT];
    bool leader = lid.x == 0 && lid.y == 0;
    shared_mu[flat] = valid ? mu : NO_SAMPLE;
    if (leader) {
        for (uint k = 0; k < STAT_COUNT; ++k) atomic_store_explicit(&mine[k], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool escaped = valid && mu >= 0.0;
    uint n = escaped ? uint(mu) : 0;
    uint rough = 0, edge = 0;
    if (valid) {
        // Right and lower neighbors within the tile.
        float right = lid.x + 1 < per_side ? shared_mu[flat + 1] : NO_SAMPLE;
        float below = lid.y + 1 < per_side ? shared_mu[flat + per_side] : NO_SAMPLE;
        for (uint k = 0; k < 2; ++k) {
            float other = k == 0 ? right : below;
            if (other == NO_SAMPLE) continue;
            bool other_escaped = other >= 0.0;
            if (escaped && other_escaped) rough += uint(abs(mu - other) > p.rough_mu);
            else edge += uint(escaped != other_escaped);
        }
    }
    uint v[STAT_COUNT] = { n, n, uint(escaped && n > max_iter / 4 * 3), uint(valid && !escaped), rough, edge };
    if (per_side * per_side % simd_width == 0) {
        // Each SIMD group lies within one tile: reduce it first so only one
        // thread per group touches the atomics.
        v[0] = simd_max(v[0]);
        for (uint k = 1; k < STAT_COUNT; ++k) v[k] = simd_sum(v[k]);
        if (simd_is_first()) {
            atomic_fetch_max_explicit(&mine[0], v[0], memory_order_relaxed);
            for (uint k = 1; k < STAT_COUNT; ++k) atomic_fetch_add_explicit(&mine[k], v[k], memory_order_relaxed);
        }
    } else {
        // Small tiles (coarse steps) share SIMD groups: few samples, so
        // per-thread atomics are cheap.
        atomic_fetch_max_explicit(&mine[0], v[0], memory_order_relaxed);
        for (uint k = 1; k < STAT_COUNT; ++k) {
            if (v[k] != 0) atomic_fetch_add_explicit(&mine[k], v[k], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (leader) {
        device TileStats& out_stats = tile_stats[gid.z];
        out_stats.max_escaped = atomic_load_explicit(&mine[0], memory_order_relaxed);
        out_stats.escaped_iters = atomic_load_explicit(&mine[1], memory_order_relaxed);
        out_stats.near_limit = atomic_load_explicit(&mine[2], memory_order_relaxed);
        out_stats.interior = atomic_load_explicit(&mine[3], memory_order_relaxed);
        out_stats.rough = atomic_load_explicit(&mine[4], memory_order_relaxed);
        out_stats.edge = atomic_load_explicit(&mine[5], memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Reprojection: resample the previous canvas into a moved/zoomed view, so the
// old image serves as the preview and only new or stale tiles are recomputed.
// ---------------------------------------------------------------------------

// Must match `GpuReprojection` in render/metal/mod.rs.
struct Reprojection {
    float  ratio;      // new pixel size in old pixels
    float  _pad;
    float2 shift;      // new center - old center, in old pixels (+y down)
    uint   old_w;
    uint   old_h;
};

/// Texel `q` of `src`, clamped to its edges.
inline float texel(texture2d<float, access::read> src, int2 q) {
    int2 last = int2(src.get_width(), src.get_height()) - 1;
    return src.read(uint2(clamp(q, int2(0), last))).r;
}

kernel void reproject(texture2d<float, access::read>  src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      constant Reprojection& r            [[buffer(0)]],
                      uint2 gid                           [[thread_position_in_grid]])
{
    uint2 size = uint2(dst.get_width(), dst.get_height());
    if (gid.x >= size.x || gid.y >= size.y) return;

    float2 old_size = float2(r.old_w, r.old_h);
    float2 u = 0.5 * old_size + (float2(gid) + 0.5 - 0.5 * float2(size)) * r.ratio + r.shift;
    float value = INSIDE; // no source: drawn black until its tile is recomputed
    if (all(u >= 0.0) && all(u <= old_size)) {
        // Bilinear in mu (by hand: float textures needn't support filtering).
        // Whole-pixel pans land exactly on texel centers (f = 0), so they copy
        // the old values unchanged.
        float2 p = u - 0.5;
        int2 i = int2(floor(p));
        float2 f = p - float2(i);
        float a = texel(src, i), b = texel(src, i + int2(1, 0));
        float c = texel(src, i + int2(0, 1)), d = texel(src, i + int2(1, 1));
        if (min(min(a, b), min(c, d)) < 0.0) {
            // Next to the set's interior: blending with INSIDE would invent mu
            // values (stray colors), so take the nearest texel instead.
            value = texel(src, int2(floor(u)));
        } else {
            value = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
    }
    dst.write(float4(value), gid);
}

// ---------------------------------------------------------------------------
// Colorize: mu canvas -> colors, every time the canvas is shown. Cheap (one
// palette lookup per pixel), so colors can change every frame for free.
// ---------------------------------------------------------------------------

// Must match `GpuColors` in render/metal/mod.rs.
struct Colors {
    uint  palette;
    float density;   // palette cycles per 50 iterations
    float offset;    // palette phase, in cycles
    uint  _pad;
};

kernel void colorize(texture2d<float, access::read>  mu  [[texture(0)]],
                     texture2d<float, access::write> out [[texture(1)]],
                     constant Colors& c                  [[buffer(0)]],
                     uint2 gid                           [[thread_position_in_grid]])
{
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    // The window may briefly be larger than the canvas during a resize.
    float m = gid.x < mu.get_width() && gid.y < mu.get_height() ? mu.read(gid).r : INSIDE;
    float3 rgb = m >= 0.0 ? palette(c.palette, m * 0.02 * c.density + c.offset) : float3(0.0);
    out.write(float4(rgb, 1.0), gid);
}
