// RCD step 1: directional discrimination and low-pass.
//
// For each site, V_stat sums squared second differences along the column
// over same-colour samples (stride 2, so a flat mosaic gives zero), H_stat
// likewise along the row. vh = V_stat / (V_stat + H_stat): near 0 the
// vertical direction is smooth, near 1 the horizontal is. The low-pass is
// the 3x3 binomial over all colours, the "luminance" the ratio correction
// in step 2 works against.
//
// The plane is clamped at zero on load, in all three steps, as the
// reference implementation does. Black subtraction leaves shadow sites
// negative, and the ratio correction in step 2 divides by the sum of two
// low-pass values; with negatives that sum crosses zero and the factor is
// unbounded (measured up to 10^6 on ISO 10000 frames, and a single site
// 70 DN under black made a 4x4 saturated red blob). Non-negative input
// keeps the factor in [0, 2].

struct Uniforms {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var plane: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rg32float, write>;

fn cfa(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    return max(textureLoad(plane, vec2<i32>(cx, cy), 0).r, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);

    var v_stat = 1e-10;
    var h_stat = 1e-10;
    for (var k = -2; k <= 2; k++) {
        let dv = cfa(x, y + k - 2) - 2.0 * cfa(x, y + k) + cfa(x, y + k + 2);
        let dh = cfa(x + k - 2, y) - 2.0 * cfa(x + k, y) + cfa(x + k + 2, y);
        v_stat += dv * dv;
        h_stat += dh * dh;
    }
    let vh = v_stat / (v_stat + h_stat);

    let lpf = (4.0 * cfa(x, y)
        + 2.0 * (cfa(x, y - 1) + cfa(x, y + 1) + cfa(x - 1, y) + cfa(x + 1, y))
        + cfa(x - 1, y - 1) + cfa(x + 1, y - 1) + cfa(x - 1, y + 1) + cfa(x + 1, y + 1)) / 16.0;

    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(vh, lpf, 0.0, 1.0));
}
