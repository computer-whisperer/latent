//! Output encoders. Each function takes pixels already transformed to
//! the target encoding; the color decisions live in `idev-core::output`.

use std::path::Path;

use anyhow::{Context, Result, ensure};

/// Write linear RGB (sRGB/Rec.709 primaries, the EXR default) as f16.
pub fn write_exr_rgb_f32(path: &Path, width: u32, height: u32, rgb: &[f32]) -> Result<()> {
    ensure!(
        rgb.len() == (width * height * 3) as usize,
        "exr: pixel buffer size mismatch"
    );
    let w = width as usize;
    exr::image::write::write_rgb_file(path, w, height as usize, |x, y| {
        let i = (y * w + x) * 3;
        (
            half::f16::from_f32(rgb[i]),
            half::f16::from_f32(rgb[i + 1]),
            half::f16::from_f32(rgb[i + 2]),
        )
    })
    .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Write 8-bit sRGB RGB as PNG.
pub fn write_png_srgb8(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    ensure!(
        rgb.len() == (width * height * 3) as usize,
        "png: pixel buffer size mismatch"
    );
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = enc.write_header().context("png header")?;
    writer.write_image_data(rgb).context("png data")?;
    Ok(())
}

/// Write a 16-bit binary PGM (P5) of raw sensor values, for cross-checks
/// against other decoders.
pub fn write_pgm16(path: &Path, width: u32, height: u32, data: &[u16]) -> Result<()> {
    use std::io::Write;
    ensure!(
        data.len() == (width * height) as usize,
        "pgm: buffer size mismatch"
    );
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(out, "P5\n{width} {height}\n65535\n")?;
    let mut bytes = Vec::with_capacity(data.len() * 2);
    for v in data {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    out.write_all(&bytes)?;
    Ok(())
}
