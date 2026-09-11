// Colour pass: camera RGB (white balanced, unclamped) -> working space.
// Highlight handling, camera matrix (Bradford-adapted to D65), exposure,
// tone curve.

struct Uniforms {
    width: u32,
    height: u32,
    highlight_mode: u32,   // 0 clip, 1 blend
    tone_mode: u32,        // 0 none, 1 sigmoid
    // Post-WB clip level per camera channel (the WB multiplier).
    clip: vec4<f32>,
    // Luminance weights in camera WB space (sum to 1).
    ylum_cam: vec4<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    // exposure gain, contrast, peak, hue preservation
    tone: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var cam_rgb: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;

const GREY: f32 = 0.18;
const Y2020: vec3<f32> = vec3<f32>(0.2627, 0.6780, 0.0593);

fn sigmoid(x: f32, contrast: f32, peak: f32) -> f32 {
    let xc = pow(max(x, 0.0), contrast);
    let s = pow(GREY, contrast) * (peak / GREY - 1.0);
    return peak * xc / (xc + s);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.width || gid.y >= u.height {
        return;
    }
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    var v = textureLoad(cam_rgb, p, 0).rgb;

    if u.highlight_mode == 0u {
        v = min(v, vec3<f32>(1.0));
    } else {
        // Blend reconstruction. A channel at its clip level is a lower
        // bound on the truth; lift it to at least the brightest channel
        // of the pixel. Once a second channel clips the colour is
        // unknowable, so the pixel desaturates toward a neutral at its
        // own luminance, driven by the second-most-clipped channel so the
        // transition is continuous. Fully clipped pixels end up neutral at
        // the highest clip level (red's, ~1.5 stops over diffuse white on
        // a typical daylight balance), which the tone curve rolls off.
        let c = u.clip.xyz;
        let clamped = min(v, c);
        let f = smoothstep(c * 0.98, c, v);
        let m = max(clamped.x, max(clamped.y, clamped.z));
        let lifted = mix(clamped, max(clamped, vec3<f32>(m)), f);
        let f_max = max(f.x, max(f.y, f.z));
        let f_min = min(f.x, min(f.y, f.z));
        let f_mid = f.x + f.y + f.z - f_max - f_min;
        let y = dot(u.ylum_cam.xyz, lifted);
        v = mix(lifted, vec3<f32>(y), f_mid);
    }

    var w = vec3<f32>(dot(u.m0.xyz, v), dot(u.m1.xyz, v), dot(u.m2.xyz, v)) * u.tone.x;

    if u.tone_mode == 1u {
        let contrast = u.tone.y;
        let peak = u.tone.z;
        let per_channel = vec3<f32>(
            sigmoid(w.r, contrast, peak),
            sigmoid(w.g, contrast, peak),
            sigmoid(w.b, contrast, peak),
        );
        let y = max(dot(Y2020, w), 1e-6);
        let by_luminance = min(w * (sigmoid(y, contrast, peak) / y), vec3<f32>(peak));
        w = mix(per_channel, by_luminance, u.tone.w);
    }

    textureStore(out_tex, p, vec4<f32>(w, 1.0));
}
