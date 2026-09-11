//! CPU output transforms from the working space. These will move onto
//! the GPU as the pipeline's final stage; for now they make the working
//! image viewable and exportable.

use rawler::Orientation;

use crate::WorkingImage;
use crate::color::{apply, srgb_from_bt2020, srgb_oetf};

/// Output (x, y) → source (x, y) for one orientation.
type PixelMap = Box<dyn Fn(usize, usize) -> (usize, usize)>;

/// Rotate/flip the working image into viewing orientation.
pub fn oriented(img: &WorkingImage, orientation: Orientation) -> WorkingImage {
    let (w, h) = (img.width as usize, img.height as usize);
    let src = |x: usize, y: usize| &img.rgba[(y * w + x) * 4..(y * w + x) * 4 + 4];
    // (out width, out height, mapping from output (x, y) to source (x, y))
    let (ow, oh, map): (usize, usize, PixelMap) = match orientation {
        Orientation::Normal | Orientation::Unknown => return img.clone(),
        Orientation::HorizontalFlip => (w, h, Box::new(move |x, y| (w - 1 - x, y))),
        Orientation::Rotate180 => (w, h, Box::new(move |x, y| (w - 1 - x, h - 1 - y))),
        Orientation::VerticalFlip => (w, h, Box::new(move |x, y| (x, h - 1 - y))),
        Orientation::Transpose => (h, w, Box::new(move |x, y| (y, x))),
        // Rotate90 = rotate the image 90° clockwise to view it.
        Orientation::Rotate90 => (h, w, Box::new(move |x, y| (y, h - 1 - x))),
        Orientation::Transverse => (h, w, Box::new(move |x, y| (w - 1 - y, h - 1 - x))),
        Orientation::Rotate270 => (h, w, Box::new(move |x, y| (w - 1 - y, x))),
    };
    let mut rgba = Vec::with_capacity(ow * oh * 4);
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = map(x, y);
            rgba.extend_from_slice(src(sx, sy));
        }
    }
    WorkingImage {
        width: ow as u32,
        height: oh as u32,
        rgba,
    }
}

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

/// Reference (diffuse) white for PQ output, per ITU-R BT.2408.
pub const PQ_REFERENCE_NITS: f32 = 203.0;

/// SMPTE ST 2084 PQ OETF on absolute luminance in cd/m².
pub fn pq_oetf(nits: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;
    let y = (nits / 10000.0).clamp(0.0, 1.0).powf(M1);
    ((C1 + C2 * y) / (1.0 + C3 * y)).powf(M2)
}

/// BT.2020 primaries, PQ transfer, 16-bit, with working 1.0 placed at
/// [`PQ_REFERENCE_NITS`]. Three samples per pixel.
pub fn bt2020_pq16(img: &WorkingImage) -> Vec<u16> {
    let mut out = Vec::with_capacity(img.width as usize * img.height as usize * 3);
    for px in img.rgba.chunks_exact(4) {
        for c in &px[..3] {
            let nits = c.to_f32().max(0.0) * PQ_REFERENCE_NITS;
            out.push((pq_oetf(nits) * 65535.0 + 0.5) as u16);
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
        let v: [f32; 3] = [px[0].to_f32(), px[1].to_f32(), px[2].to_f32()];
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

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn pq_anchors() {
        // ST 2084: 10000 nits -> 1.0; 203 nits -> ~0.58 (BT.2408 HDR reference white).
        assert!((pq_oetf(10000.0) - 1.0).abs() < 1e-5);
        assert!((pq_oetf(203.0) - 0.5807).abs() < 0.002);
        assert!(pq_oetf(0.0) < 1e-6);
    }

    #[test]
    fn rotate90_dimensions_and_corner() {
        let mut rgba = vec![f16::from_f32(0.0); 3 * 2 * 4];
        // mark source (2, 0) (top-right of a 3x2) red
        rgba[2 * 4] = f16::from_f32(1.0);
        let img = WorkingImage {
            width: 3,
            height: 2,
            rgba,
        };
        let r = oriented(&img, Orientation::Rotate90);
        assert_eq!((r.width, r.height), (2, 3));
        // rotating 90° clockwise moves top-right to bottom-right: (1, 2)
        assert_eq!(r.rgba[((2 * 2) + 1) * 4].to_f32(), 1.0);
    }
}
