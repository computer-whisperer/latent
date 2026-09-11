// Colour pass: camera RGB (white balanced, unclamped) -> output space.
// Each output pixel samples the camera RGB at its lens-corrected source
// coordinate per channel (distortion, lateral CA): bilinear at scale 1,
// a scale x scale box average in camera-space linear light otherwise.
// Then highlight handling, the vignetting gain, camera matrix
// (Bradford-adapted to D65), exposure, tone curve, and the display
// primaries matrix (identity for exports, BT.2020 -> sRGB for the
// damascene preview).

struct Uniforms {
    width: u32,            // output size
    height: u32,
    highlight_mode: u32,   // 0 clip, 1 blend
    tone_mode: u32,        // 0 none, 1 sigmoid
    src_width: u32,        // camera RGB size
    src_height: u32,
    scale: u32,            // output pixel = scale x scale source pixels
    pad0: u32,
    // Post-WB clip level per camera channel (the WB multiplier).
    clip: vec4<f32>,
    // Luminance weights in camera WB space (sum to 1).
    ylum_cam: vec4<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    d0: vec4<f32>,
    d1: vec4<f32>,
    d2: vec4<f32>,
    // exposure gain, contrast, peak, hue preservation
    tone: vec4<f32>,
    // sigmoid s, white, 1 / F(white), pad
    tone2: vec4<f32>,
    // distortion, chromatic aberration, vignetting flags; knot count
    lens: vec4<u32>,
    // centre x, y (camera RGB pixels), 1 / half-diagonal, pad
    centre: vec4<f32>,
    // Correction tables over normalised radius: knot positions, then
    // source/output radius (distortion), red and blue extra radius
    // factors (CA), and gain (vignetting). Linear between knots.
    knots: array<vec4<f32>, 4>,
    dist: array<vec4<f32>, 4>,
    ca_r: array<vec4<f32>, 4>,
    ca_b: array<vec4<f32>, 4>,
    vig: array<vec4<f32>, 4>,
    // warning overlay enabled, pad
    warn: vec4<u32>,
    // working space -> the panel's gamut
    g0: vec4<f32>,
    g1: vec4<f32>,
    g2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var cam_rgb: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;

const Y2020: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);

// peak * F(x) / F(white), F(x) = x^c / (x^c + s); constants from
// params::sigmoid_constants so grey stays at grey and white hits peak.
fn sigmoid(x: f32, contrast: f32, peak: f32) -> f32 {
    let xc = pow(max(x, 0.0), contrast);
    let f = xc / (xc + u.tone2.x);
    return min(peak * f * u.tone2.z, peak);
}

fn table(which: u32, i: u32) -> f32 {
    let j = i / 4u;
    let k = i % 4u;
    switch which {
        case 0u: { return u.knots[j][k]; }
        case 1u: { return u.dist[j][k]; }
        case 2u: { return u.ca_r[j][k]; }
        case 3u: { return u.ca_b[j][k]; }
        default: { return u.vig[j][k]; }
    }
}

// Table `which` at normalised radius r: linear between knots, clamped.
fn spline(which: u32, r: f32) -> f32 {
    let nc = u.lens.w;
    if r <= table(0u, 0u) {
        return table(which, 0u);
    }
    for (var i = 1u; i < nc; i++) {
        let x0 = table(0u, i - 1u);
        let x1 = table(0u, i);
        if r <= x1 {
            let t = (r - x0) / max(x1 - x0, 1e-6);
            return mix(table(which, i - 1u), table(which, i), t);
        }
    }
    return table(which, nc - 1u);
}

fn load(p: vec2<i32>, ch: u32) -> f32 {
    let size = vec2<i32>(i32(u.src_width), i32(u.src_height));
    let q = clamp(p, vec2<i32>(0), size - 1);
    return textureLoad(cam_rgb, q, 0)[ch];
}

// Channel `ch` of the camera RGB at source coordinate `src` (pixel
// centres at +0.5): bilinear at scale 1, else the mean of the scale x
// scale block centred there.
fn sample(src: vec2<f32>, ch: u32) -> f32 {
    if u.scale <= 1u {
        let q = src - 0.5;
        let q0 = floor(q);
        let f = q - q0;
        let a = vec2<i32>(q0);
        let top = mix(load(a, ch), load(a + vec2<i32>(1, 0), ch), f.x);
        let bottom = mix(load(a + vec2<i32>(0, 1), ch), load(a + vec2<i32>(1, 1), ch), f.x);
        return mix(top, bottom, f.y);
    }
    let s = i32(u.scale);
    let size = vec2<i32>(i32(u.src_width), i32(u.src_height));
    let b = clamp(vec2<i32>(round(src - f32(s) * 0.5)), vec2<i32>(0), max(size - s, vec2<i32>(0)));
    let e = min(b + s, size);
    var acc = 0.0;
    var n = 0.0;
    for (var y = b.y; y < e.y; y++) {
        for (var x = b.x; x < e.x; x++) {
            acc += load(vec2<i32>(x, y), ch);
            n += 1.0;
        }
    }
    return acc / max(n, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    // This output pixel's centre in camera RGB pixels, and its offset
    // from the image centre.
    let out_c = (vec2<f32>(gid.xy) + 0.5) * f32(u.scale);
    let off = out_c - u.centre.xy;
    let rout = u.centre.z * length(off);
    var f = vec3<f32>(1.0);
    if u.lens.x == 1u {
        f *= spline(1u, rout);
    }
    if u.lens.y == 1u {
        f.r *= spline(2u, rout);
        f.b *= spline(3u, rout);
    }
    // An identity factor samples the pixel exactly (bit-identical to
    // no correction) rather than through the rounding of c + (o - c).
    let src_r = select(u.centre.xy + f.r * off, out_c, f.r == 1.0);
    let src_g = select(u.centre.xy + f.g * off, out_c, f.g == 1.0);
    let src_b = select(u.centre.xy + f.b * off, out_c, f.b == 1.0);
    var v = vec3<f32>(sample(src_r, 0u), sample(src_g, 1u), sample(src_b, 2u));

    if u.highlight_mode == 0u {
        v = min(v, vec3<f32>(1.0));
    } else {
        // Blend reconstruction. A channel at its clip level is a lower
        // bound on the truth; lift it to at least the brightest channel
        // of the pixel. Once a second channel clips the colour is
        // unknowable, so the pixel desaturates toward a neutral at its
        // own luminance, driven by the second-most-clipped channel so the
        // transition is continuous. Fully clipped pixels end up neutral at
        // the highest clip level (red's, ~1.5 stops over diffuse white on
        // a typical daylight balance), which the tone curve rolls off.
        let c = u.clip.xyz;
        let clamped = min(v, c);
        let f = smoothstep(c * 0.98, c, v);
        let m = max(clamped.x, max(clamped.y, clamped.z));
        let lifted = mix(clamped, max(clamped, vec3<f32>(m)), f);
        let f_max = max(f.x, max(f.y, f.z));
        let f_min = min(f.x, min(f.y, f.z));
        let f_mid = f.x + f.y + f.z - f_max - f_min;
        let y = dot(u.ylum_cam.xyz, lifted);
        v = mix(lifted, vec3<f32>(y), f_mid);
    }
    if u.lens.z == 1u {
        v *= spline(4u, rout);
    }

    var w = vec3<f32>(dot(u.m0.xyz, v), dot(u.m1.xyz, v), dot(u.m2.xyz, v)) * u.tone.x;

    if u.tone_mode == 1u {
        let contrast = u.tone.y;
        let peak = u.tone.z;
        let per_channel = vec3<f32>(
            sigmoid(w.r, contrast, peak),
            sigmoid(w.g, contrast, peak),
            sigmoid(w.b, contrast, peak),
        );
        let y = max(dot(Y2020, w), 1e-6);
        let by_luminance = min(w * (sigmoid(y, contrast, peak) / y), vec3<f32>(peak));
        w = mix(per_channel, by_luminance, u.tone.w);
    }

    if u.warn.x == 1u {
        // Zebra over what the panel cannot show: red where a channel
        // sits at the output peak, cyan where the colour leaves the
        // panel's gamut. Markers at diffuse-white brightness.
        let peak = u.tone.z;
        let g = vec3<f32>(dot(u.g0.xyz, w), dot(u.g1.xyz, w), dot(u.g2.xyz, w));
        let clipped = any(w >= vec3<f32>(peak * 0.995));
        let outside = any(g < vec3<f32>(-0.002 * peak)) || any(g > vec3<f32>(peak * 1.005));
        let stripe = ((gid.x + gid.y) / 6u) % 2u == 0u;
        if stripe && clipped {
            w = vec3<f32>(0.9, 0.04, 0.04);
        } else if stripe && outside {
            w = vec3<f32>(0.04, 0.7, 0.9);
        }
    }

    let d = vec3<f32>(dot(u.d0.xyz, w), dot(u.d1.xyz, w), dot(u.d2.xyz, w));
    textureStore(out_tex, p, vec4<f32>(d, 1.0));
}
