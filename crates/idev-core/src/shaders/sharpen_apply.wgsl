// Unsharp mask on the encoded luminance: the detail (encoded luminance
// minus its blur) is scaled by `amount`, soft-thresholded, and the
// resulting luminance ratio applied to all three channels, so colour
// is untouched. Channels clamp to the output peak.

struct Uniforms {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
    amount: f32,
    threshold: f32,  // in encoded luminance (0..1)
    peak: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var blurred: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

const Y2020: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let v = textureLoad(src, p, 0);
    let y = max(dot(Y2020, v.rgb), 0.0);
    let l = pow(y, 1.0 / 2.2);
    let b = textureLoad(blurred, p, 0).r;
    var d = l - b;
    // Soft threshold: nothing below it, full strength at twice it.
    if u.threshold > 0.0 {
        d *= smoothstep(u.threshold, 2.0 * u.threshold, abs(d));
    }
    let l2 = max(l + u.amount * d, 0.0);
    let gain = select(1.0, pow(l2 / max(l, 1e-6), 2.2), l > 1e-6);
    let out = min(v.rgb * gain, vec3<f32>(u.peak));
    textureStore(dst, p, vec4<f32>(out, 1.0));
}
