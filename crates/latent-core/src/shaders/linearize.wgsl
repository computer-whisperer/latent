// Pass 1: raw u16 Bayer values -> linear, black-subtracted, white-balanced
// f32 plane. Per-CFA-position tables so the shader needs no pattern logic.
// No clamping here: the colour pass handles clipped sites with the full
// pixel in hand.

struct Uniforms {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
    // Indexed by (y & 1) * 2 + (x & 1).
    black: vec4<f32>,
    scale: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Raw values packed two per u32, little-endian, row-major.
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var plane: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= u.width || y >= u.height {
        return;
    }
    let idx = y * u.width + x;
    let word = raw[idx >> 1u];
    let v = select(word & 0xffffu, word >> 16u, (idx & 1u) == 1u);
    let pos = (y & 1u) * 2u + (x & 1u);
    let lin = (f32(v) - u.black[pos]) * u.scale[pos];
    textureStore(plane, vec2<i32>(i32(x), i32(y)), vec4<f32>(lin, 0.0, 0.0, 1.0));
}
