// RCD step 3: red and blue everywhere, as colour differences against the
// full green plane, over the crop rectangle. Output is camera RGB after
// white balance, unclamped.

struct Uniforms {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    out_width: u32,
    out_height: u32,
    pad0: u32,
    pad1: u32,
    cfa_pattern: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var plane: texture_2d<f32>;
@group(0) @binding(2) var green: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;

fn cfa(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.src_width) - 1);
    let cy = clamp(y, 0, i32(u.src_height) - 1);
    return max(textureLoad(plane, vec2<i32>(cx, cy), 0).r, 0.0);
}

fn g(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.src_width) - 1);
    let cy = clamp(y, 0, i32(u.src_height) - 1);
    return textureLoad(green, vec2<i32>(cx, cy), 0).r;
}

fn colour_at(x: i32, y: i32) -> u32 {
    return u.cfa_pattern[(u32(y) & 1u) * 2u + (u32(x) & 1u)];
}

// The colour native at (x±1, y) or (x, y±1), interpolated from that axis.
fn axis_colour(x: i32, y: i32, dx: i32, dy: i32) -> f32 {
    let eps = 1e-5;
    let a1 = cfa(x - dx, y - dy);
    let b1 = cfa(x + dx, y + dy);
    let gc = g(x, y);
    let a_grad = eps + abs(a1 - b1) + abs(a1 - cfa(x - 3 * dx, y - 3 * dy)) + abs(gc - g(x - 2 * dx, y - 2 * dy));
    let b_grad = eps + abs(a1 - b1) + abs(b1 - cfa(x + 3 * dx, y + 3 * dy)) + abs(gc - g(x + 2 * dx, y + 2 * dy));
    let a_est = a1 - g(x - dx, y - dy);
    let b_est = b1 - g(x + dx, y + dy);
    return gc + (b_grad * a_est + a_grad * b_est) / (a_grad + b_grad);
}

// The colour native on the diagonals of an R/B site.
fn diagonal_colour(x: i32, y: i32) -> f32 {
    let eps = 1e-5;
    let gc = g(x, y);
    let nw = cfa(x - 1, y - 1);
    let se = cfa(x + 1, y + 1);
    let ne = cfa(x + 1, y - 1);
    let sw = cfa(x - 1, y + 1);
    let p_grad = eps + abs(nw - se) + abs(nw - cfa(x - 3, y - 3)) + abs(se - cfa(x + 3, y + 3))
        + abs(gc - g(x - 2, y - 2)) + abs(gc - g(x + 2, y + 2));
    let q_grad = eps + abs(ne - sw) + abs(ne - cfa(x + 3, y - 3)) + abs(sw - cfa(x - 3, y + 3))
        + abs(gc - g(x + 2, y - 2)) + abs(gc - g(x - 2, y + 2));
    let p_est = 0.5 * ((nw - g(x - 1, y - 1)) + (se - g(x + 1, y + 1)));
    let q_est = 0.5 * ((ne - g(x + 1, y - 1)) + (sw - g(x - 1, y + 1)));
    return gc + (q_grad * p_est + p_grad * q_est) / (p_grad + q_grad);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.out_width || gid.y >= u.out_height {
        return;
    }
    let x = i32(gid.x + u.crop_x);
    let y = i32(gid.y + u.crop_y);
    let own = colour_at(x, y);
    var rgb = vec3<f32>(0.0);
    rgb[1] = g(x, y);
    if own == 1u {
        let ch = colour_at(x + 1, y);
        let cv = colour_at(x, y + 1);
        rgb[ch] = axis_colour(x, y, 1, 0);
        rgb[cv] = axis_colour(x, y, 0, 1);
    } else {
        rgb[own] = cfa(x, y);
        rgb[2u - own] = diagonal_colour(x, y);
    }
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
