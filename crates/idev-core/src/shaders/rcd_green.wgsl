// RCD step 2: green at red/blue sites.
//
// Each cardinal neighbour's green is ratio-corrected by the low-pass
// slope toward the centre, the two estimates per axis are blended by
// inverse gradient, and the axes are blended by the (locally smoothed)
// directional discrimination.

struct Uniforms {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
    // CFA colour per position (y & 1) * 2 + (x & 1).
    cfa_pattern: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var plane: texture_2d<f32>;
@group(0) @binding(2) var vhlpf: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<r32float, write>;

fn cfa(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    return textureLoad(plane, vec2<i32>(cx, cy), 0).r;
}

fn vh(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    return textureLoad(vhlpf, vec2<i32>(cx, cy), 0).r;
}

fn lpf(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    return textureLoad(vhlpf, vec2<i32>(cx, cy), 0).g;
}

fn colour_at(x: i32, y: i32) -> u32 {
    return u.cfa_pattern[(u32(y) & 1u) * 2u + (u32(x) & 1u)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let centre = cfa(x, y);
    if colour_at(x, y) == 1u {
        textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(centre, 0.0, 0.0, 1.0));
        return;
    }

    let eps = 1e-5;
    let n1 = cfa(x, y - 1);
    let s1 = cfa(x, y + 1);
    let w1 = cfa(x - 1, y);
    let e1 = cfa(x + 1, y);
    let n2 = cfa(x, y - 2);
    let s2 = cfa(x, y + 2);
    let w2 = cfa(x - 2, y);
    let e2 = cfa(x + 2, y);

    let n_grad = eps + abs(n1 - s1) + abs(centre - n2) + abs(n1 - cfa(x, y - 3)) + abs(n2 - cfa(x, y - 4));
    let s_grad = eps + abs(n1 - s1) + abs(centre - s2) + abs(s1 - cfa(x, y + 3)) + abs(s2 - cfa(x, y + 4));
    let w_grad = eps + abs(w1 - e1) + abs(centre - w2) + abs(w1 - cfa(x - 3, y)) + abs(w2 - cfa(x - 4, y));
    let e_grad = eps + abs(w1 - e1) + abs(centre - e2) + abs(e1 - cfa(x + 3, y)) + abs(e2 - cfa(x + 4, y));

    let lc = lpf(x, y);
    let n_est = n1 * (1.0 + (lc - lpf(x, y - 2)) / (eps + lc + lpf(x, y - 2)));
    let s_est = s1 * (1.0 + (lc - lpf(x, y + 2)) / (eps + lc + lpf(x, y + 2)));
    let w_est = w1 * (1.0 + (lc - lpf(x - 2, y)) / (eps + lc + lpf(x - 2, y)));
    let e_est = e1 * (1.0 + (lc - lpf(x + 2, y)) / (eps + lc + lpf(x + 2, y)));

    let v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad);
    let h_est = (e_grad * w_est + w_grad * e_est) / (w_grad + e_grad);

    let vh_c = vh(x, y);
    let vh_n = 0.25 * (vh(x, y - 1) + vh(x, y + 1) + vh(x - 1, y) + vh(x + 1, y));
    let vh_disc = select(vh_c, vh_n, abs(0.5 - vh_c) < abs(0.5 - vh_n));

    let g = vh_disc * h_est + (1.0 - vh_disc) * v_est;
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(g, 0.0, 0.0, 1.0));
}
