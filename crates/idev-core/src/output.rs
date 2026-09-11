//! CPU output transforms from the working space. These will move onto
//! the GPU as the pipeline's final stage; for now they make the working
//! image viewable and exportable.

use crate::WorkingImage;
use crate::color::{apply, srgb_from_bt2020, srgb_oetf};

/// Linear RGB with sRGB primaries, f32, unclipped (scRGB convention,
/// 1.0 = diffuse white). Three floats per pixel.
pub fn scrgb_linear_f32(img: &WorkingImage) -> Vec<f32> {
    let m = srgb_from_bt2020();
    let mut out = Vec::with_capacity(img.width as usize * img.height as usize * 3);
    for px in img.rgba.chunks_exact(4) {
        let v = apply(&m, [px[0].to_f32(), px[1].to_f32(), px[2].to_f32()]);
        out.extend_from_slice(&v);
    }
    out
}

/// Display-referred sRGB, 8-bit, hard-clipped at diffuse white. Three
/// bytes per pixel.
pub fn srgb8(img: &WorkingImage) -> Vec<u8> {
    let m = srgb_from_bt2020();
    let mut out = Vec::with_capacity(img.width as usize * img.height as usize * 3);
    for px in img.rgba.chunks_exact(4) {
        let v = apply(&m, [px[0].to_f32(), px[1].to_f32(), px[2].to_f32()]);
        for c in v {
            out.push((srgb_oetf(c.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8);
        }
    }
    out
}

/// Per-channel maxima and mean luminance over the working image, for
/// quick sanity checks.
pub fn stats(img: &WorkingImage) -> ([f32; 3], f32) {
    let mut max = [f32::MIN; 3];
    let mut lum_sum = 0.0f64;
    let mut n = 0u64;
    for px in img.rgba.chunks_exact(4) {
        let v = [px[0].to_f32(), px[1].to_f32(), px[2].to_f32()];
        for c in 0..3 {
            max[c] = max[c].max(v[c]);
        }
        lum_sum += (0.2627 * v[0] + 0.678 * v[1] + 0.0593 * v[2]) as f64;
        n += 1;
    }
    (
        max,
        if n > 0 {
            (lum_sum / n as f64) as f32
        } else {
            0.0
        },
    )
}
