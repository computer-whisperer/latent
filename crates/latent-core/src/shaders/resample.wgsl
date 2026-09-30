// Separable Lanczos-3 resample of an Rgba16Float texture along one
// axis, in whatever light the texture holds (the working image is
// linear). The kernel widens with the downscale factor so it acts as
// the anti-aliasing filter; upscaling uses the unit kernel. Edges
// clamp. Negative lobes are clamped out of the result.

struct Uniforms {
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    axis: u32,     // 0 horizontal, 1 vertical
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;

fn sinc(x: f32) -> f32 {
    if abs(x) < 1e-4 {
        return 1.0;
    }
    let px = PI * x;
    return sin(px) / px;
}

fn lanczos3(x: f32) -> f32 {
    if abs(x) >= 3.0 {
        return 0.0;
    }
    return sinc(x) * sinc(x / 3.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.out_w || gid.y >= u.out_h {
        return;
    }
    let horizontal = u.axis == 0u;
    let in_len = select(u.in_h, u.in_w, horizontal);
    let out_len = select(u.out_h, u.out_w, horizontal);
    let o = select(gid.y, gid.x, horizontal);
    let f = f32(in_len) / f32(out_len);
    let fs = max(f, 1.0);
    let sc = (f32(o) + 0.5) * f - 0.5;
    let lo = i32(floor(sc - 3.0 * fs));
    let hi = i32(ceil(sc + 3.0 * fs));
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = lo; i <= hi; i++) {
        let w = lanczos3((f32(i) - sc) / fs);
        if w == 0.0 {
            continue;
        }
        let j = clamp(i, 0, i32(in_len) - 1);
        let p = select(vec2<i32>(i32(gid.x), j), vec2<i32>(j, i32(gid.y)), horizontal);
        acc += w * textureLoad(src, p, 0);
        wsum += w;
    }
    let v = max(acc / wsum, vec4<f32>(0.0));
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(v.rgb, 1.0));
}
