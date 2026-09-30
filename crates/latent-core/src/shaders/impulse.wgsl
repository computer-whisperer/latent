// Impulse filter on the linearized plane, before demosaic. The sensor's
// read noise at high gain is heavy-tailed: single sites land 10 to 30
// sigma above or below their neighbours (measured on ISO 1000 to 10000
// frames), at positions that do not repeat between frames, and every
// demosaic turns such a site into a coloured dot. A site is replaced
// when it is further than `k` sigma from its same-colour 5x5
// neighbourhood, with the mean and sigma taken over those 24 samples
// after dropping the two largest and two smallest, so one or two more
// bad sites in the window cannot hide the centre or trigger it.

struct Uniforms {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
    // Threshold in sigmas, and the floor under sigma (plane units).
    k: f32,
    floor: f32,
    pad2: f32,
    pad3: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var plane: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<r32float, write>;

fn load(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(u.width) - 1);
    let cy = clamp(y, 0, i32(u.height) - 1);
    return textureLoad(plane, vec2<i32>(cx, cy), 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let v = load(x, y);

    var sum = 0.0;
    var sumsq = 0.0;
    // Two largest and two smallest of the 24 neighbours.
    var hi1 = -1e30;
    var hi2 = -1e30;
    var lo1 = 1e30;
    var lo2 = 1e30;
    for (var dy = -4; dy <= 4; dy += 2) {
        for (var dx = -4; dx <= 4; dx += 2) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let s = load(x + dx, y + dy);
            sum += s;
            sumsq += s * s;
            if s > hi1 {
                hi2 = hi1;
                hi1 = s;
            } else if s > hi2 {
                hi2 = s;
            }
            if s < lo1 {
                lo2 = lo1;
                lo1 = s;
            } else if s < lo2 {
                lo2 = s;
            }
        }
    }
    let n = 20.0;
    let tsum = sum - hi1 - hi2 - lo1 - lo2;
    let tsumsq = sumsq - hi1 * hi1 - hi2 * hi2 - lo1 * lo1 - lo2 * lo2;
    let mean = tsum / n;
    let sigma = sqrt(max(tsumsq / n - mean * mean, 0.0));
    let limit = u.k * max(sigma, u.floor);
    let out = select(v, mean, abs(v - mean) > limit);
    textureStore(out_tex, vec2<i32>(x, y), vec4<f32>(out, 0.0, 0.0, 1.0));
}
