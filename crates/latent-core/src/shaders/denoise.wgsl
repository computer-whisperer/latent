// Noise reduction on the white-balanced camera RGB (linear, 1.0 = clip),
// before the colour pass, so the tone curve never amplifies what was
// removed. Two bilateral filters in one pass:
//
// - chroma: the per-pixel chroma (RGB minus luminance) is smoothed over
//   a 13 x 13 window sampled every other pixel, the range kernel on the
//   chroma difference so colour edges stay put;
// - luminance: a 7 x 7 bilateral on the luminance with a range sigma
//   that follows shot noise (∝ sqrt(Y)) scaled by the strength.
//
// The output is luminance' + chroma'.

struct Uniforms {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
    chroma: f32,    // 0..1
    luma: f32,      // 0..1
    pad2: f32,
    pad3: f32,
    ylum: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

fn load(p: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(i32(u.width), i32(u.height));
    return textureLoad(src, clamp(p, vec2<i32>(0), size - 1), 0).rgb;
}

fn lum(v: vec3<f32>) -> f32 {
    return dot(u.ylum.xyz, v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let c = load(p);
    let y0 = lum(c);
    // Chroma as differences from luminance: additive, so a near-black
    // noisy pixel cannot blow up a ratio. Shot noise grows with
    // sqrt(Y), so both range sigmas follow that.
    let chroma0 = c - y0;
    // Luminance can dip below zero after black subtraction; keep the
    // shot-noise estimate real.
    let shot = sqrt(max(y0, 0.0) + 0.002);

    var chroma = chroma0;
    if u.chroma > 0.0 {
        let sr = 0.1 * u.chroma * shot;
        var acc = vec3<f32>(0.0);
        var wsum = 0.0;
        for (var dy = -6; dy <= 6; dy += 2) {
            for (var dx = -6; dx <= 6; dx += 2) {
                let q = load(p + vec2<i32>(dx, dy));
                let cq = q - lum(q);
                let d = cq - chroma0;
                let ws = exp(-f32(dx * dx + dy * dy) / (2.0 * 16.0));
                let wr = exp(-dot(d, d) / (2.0 * sr * sr));
                acc += ws * wr * cq;
                wsum += ws * wr;
            }
        }
        chroma = acc / wsum;
    }

    var y = y0;
    if u.luma > 0.0 {
        let sr = 0.06 * u.luma * shot;
        var acc = 0.0;
        var wsum = 0.0;
        for (var dy = -3; dy <= 3; dy++) {
            for (var dx = -3; dx <= 3; dx++) {
                let yq = lum(load(p + vec2<i32>(dx, dy)));
                let d = yq - y0;
                let ws = exp(-f32(dx * dx + dy * dy) / (2.0 * 2.25));
                let wr = exp(-d * d / (2.0 * sr * sr));
                acc += ws * wr * yq;
                wsum += ws * wr;
            }
        }
        y = acc / wsum;
    }

    textureStore(dst, p, vec4<f32>(y + chroma, 1.0));
}
