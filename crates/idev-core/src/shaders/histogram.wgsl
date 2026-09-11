// Histogram of a developed texture: 3 x 256 bins of the sRGB-encoded
// channel values (display-referred, peak = 1.0 -> bin 255), so the
// shape matches what the viewer sees.

struct Uniforms {
    width: u32,
    height: u32,
    // Values are divided by this before encoding (the display peak).
    peak: f32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>, 768>;

fn oetf(v: f32) -> f32 {
    let c = clamp(v, 0.0, 1.0);
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, 12.92 * c, c <= 0.0031308);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let v = textureLoad(img, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb / u.peak;
    let r = u32(oetf(v.r) * 255.0 + 0.5);
    let g = u32(oetf(v.g) * 255.0 + 0.5);
    let b = u32(oetf(v.b) * 255.0 + 0.5);
    atomicAdd(&bins[r], 1u);
    atomicAdd(&bins[256u + g], 1u);
    atomicAdd(&bins[512u + b], 1u);
}
