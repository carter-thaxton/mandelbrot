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
    uint   max_iter;
    uint   palette;
    uint   precision;
    float  color_density;
    uint   ref_len;
    float  scale_m;    // scale as floatexp: scale_m * 2^scale_e
    int    scale_e;
    uint   _pad;
};

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
// dc may underflow to 0 in the fast phase; by then it is ~1e-7 or less of dz,
// i.e. below f32 precision anyway.
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

float iterate_perturb(cfe dc, device const RefPoint* ref, uint ref_len, uint max_iter) {
    float2 dcf = cfe_to_float2(dc);
    cfe dzx = cfe{float2(0.0), FE_ZERO_EXP};
    float2 dz = float2(0.0);
    uint m = 0;
    uint n = 0;
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
                dz = float2(t.x * dz.x - t.y * dz.y, t.x * dz.y + t.y * dz.x) + dcf;
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

kernel void mandelbrot(texture2d<float, access::write> out [[texture(0)]],
                       constant Params& p                 [[buffer(0)]],
                       device const RefPoint* ref         [[buffer(1)]],
                       uint2 gid                          [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;

    // Pixel offset from the view center; exact in f32 for any sane size.
    float px = float(gid.x) + 0.5 - 0.5 * float(p.width);
    float py = 0.5 * float(p.height) - (float(gid.y) + 0.5);

    // A cycle is "detected" when the orbit returns to well within a pixel.
    float period_eps = max(p.scale.x * 1e-3, 1e-30);

    float mu;
    if (p.precision == 0) {
        float2 c = float2(p.center_x.x + px * p.scale.x,
                          p.center_y.x + py * p.scale.x);
        mu = iterate_f32(c, p.max_iter, max(period_eps, 1e-7));
    } else if (p.precision == 1) {
        df cx = df_add(p.center_x, df_mul_f(p.scale, px));
        df cy = df_add(p.center_y, df_mul_f(p.scale, py));
        mu = iterate_df(cx, cy, p.max_iter, period_eps);
    } else {
        // The reference is the view center, so dc is just the pixel offset.
        cfe dc = cfe_make(float2(px, py) * p.scale_m, p.scale_e);
        mu = p.ref_len >= 2 ? iterate_perturb(dc, ref, p.ref_len, p.max_iter) : -1.0;
    }

    float3 rgb = float3(0.0);
    if (mu >= 0.0) {
        float t = mu * 0.02 * p.color_density;
        rgb = palette(p.palette, t);
    }
    out.write(float4(rgb, 1.0), gid);
}
