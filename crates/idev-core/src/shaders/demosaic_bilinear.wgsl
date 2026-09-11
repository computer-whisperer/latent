// Bilinear demosaic of the linear plane over the crop rectangle. Output is
// camera RGB after white balance, unclamped. The fast path; RCD is the
// quality path.
//
// Bilinear on a 2x2 Bayer mosaic: a site keeps its own color; every other
// color is the mean of the same-colored sites in its 3x3 neighbourhood
// (2 for the row/column neighbours, 4 for the cross or the diagonals).

struct Uniforms {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    out_width: u32,
    out_height: u32,
    pad0: u32,
    pad1: u32,
    // CFA color (0 = R, 1 = G, 2 = B) per position (y & 1) * 2 + (x & 1),
    // in source raster coordinates.
    cfa: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var plane: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;

fn load(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.src_width) - 1);
    let cy = clamp(y, 0, i32(u.src_height) - 1);
    return textureLoad(plane, vec2<i32>(cx, cy), 0).r;
}

fn cfa_color(x: i32, y: i32) -> u32 {
    let pos = (u32(y) & 1u) * 2u + (u32(x) & 1u);
    return u.cfa[pos];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.out_width || gid.y >= u.out_height {
        return;
    }
    let sx = i32(gid.x + u.crop_x);
    let sy = i32(gid.y + u.crop_y);
    let own = cfa_color(sx, sy);

    var sum = vec3<f32>(0.0);
    var cnt = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let c = cfa_color(sx + dx, sy + dy);
            let v = load(sx + dx, sy + dy);
            sum[c] += v;
            cnt[c] += 1.0;
        }
    }
    var rgb = sum / max(cnt, vec3<f32>(1.0));
    rgb[own] = load(sx, sy);

    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
