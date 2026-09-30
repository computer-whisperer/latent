// Gaussian blur along one axis of the sharpening luminance. The first
// pass (`encode` = 1) reads the RGBA working image and blurs its
// perceptually encoded luminance; the second reads that R32 result.

struct Uniforms {
    width: u32,
    height: u32,
    axis: u32,       // 0 horizontal, 1 vertical
    encode: u32,     // 1: input is RGBA, take encoded luminance
    sigma: f32,      // pixels
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

const Y2020: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);

// Luminance in a gamma 2.2 encoding: halos then read symmetric.
fn encoded(p: vec2<i32>) -> f32 {
    let v = textureLoad(src, p, 0);
    if u.encode == 1u {
        return pow(max(dot(Y2020, v.rgb), 0.0), 1.0 / 2.2);
    }
    return v.r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let r = i32(ceil(3.0 * u.sigma));
    let size = vec2<i32>(i32(u.width), i32(u.height));
    var acc = 0.0;
    var wsum = 0.0;
    for (var i = -r; i <= r; i++) {
        let w = exp(-0.5 * f32(i * i) / (u.sigma * u.sigma));
        var p = vec2<i32>(gid.xy);
        if u.axis == 0u {
            p.x += i;
        } else {
            p.y += i;
        }
        p = clamp(p, vec2<i32>(0), size - 1);
        acc += w * encoded(p);
        wsum += w;
    }
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(acc / wsum, 0.0, 0.0, 1.0));
}
