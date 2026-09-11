//! Output encoders. Each function takes pixels already transformed to
//! the target encoding; the color decisions live in `idev-core::output`.

mod avif;

use std::path::Path;

use anyhow::{Context, Result, ensure};
pub use avif::{write_avif_bt2020_pq10, write_avif_srgb8};

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

/// Write 8-bit sRGB RGB as PNG. `exif` is a TIFF-structured EXIF block
/// for an `eXIf` chunk.
pub fn write_png_srgb8(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[u8],
    exif: Option<&[u8]>,
) -> Result<()> {
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
    write_exif_chunk(&mut writer, exif)?;
    writer.write_image_data(rgb).context("png data")?;
    Ok(())
}

fn write_exif_chunk<W: std::io::Write>(
    writer: &mut png::Writer<W>,
    exif: Option<&[u8]>,
) -> Result<()> {
    if let Some(exif) = exif {
        writer
            .write_chunk(png::chunk::ChunkType(*b"eXIf"), exif)
            .context("png eXIf")?;
    }
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

/// Write 16-bit RGB tagged as BT.2020 primaries with the PQ transfer
/// (`cICP` 9/16/0/full-range), the HDR PNG browsers and achromat read.
pub fn write_png_bt2020_pq16(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[u16],
    exif: Option<&[u8]>,
) -> Result<()> {
    ensure!(
        rgb.len() == (width * height * 3) as usize,
        "png: pixel buffer size mismatch"
    );
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Sixteen);
    let mut writer = enc.write_header().context("png header")?;
    // Colour primaries BT.2020, transfer ST 2084, matrix identity, full range.
    writer
        .write_chunk(png::chunk::cICP, &[9, 16, 0, 1])
        .context("png cICP")?;
    write_exif_chunk(&mut writer, exif)?;
    let mut bytes = Vec::with_capacity(rgb.len() * 2);
    for v in rgb {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    writer.write_image_data(&bytes).context("png data")?;
    Ok(())
}

/// Write 8-bit sRGB RGB as baseline JPEG. `exif` is a TIFF-structured
/// EXIF block for an APP1 segment (at most 65 527 bytes).
pub fn write_jpeg_srgb8(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[u8],
    quality: u8,
    exif: Option<&[u8]>,
) -> Result<()> {
    ensure!(
        rgb.len() == (width * height * 3) as usize,
        "jpeg: pixel buffer size mismatch"
    );
    let mut jpeg = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, quality);
    enc.encode(rgb, width, height, image::ExtendedColorType::Rgb8)
        .context("jpeg encode")?;
    let bytes = match exif {
        Some(exif) => with_app1_exif(&jpeg, exif)?,
        None => jpeg,
    };
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// Insert an `Exif` APP1 segment after SOI (and after a JFIF APP0 if
/// the encoder wrote one, as the JFIF spec asks).
fn with_app1_exif(jpeg: &[u8], exif: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        exif.len() + 8 <= 0xffff,
        "jpeg: exif block too large for one APP1 segment"
    );
    ensure!(
        jpeg.starts_with(&[0xff, 0xd8]),
        "jpeg: encoder output has no SOI"
    );
    let mut at = 2;
    if jpeg.get(2..4) == Some(&[0xff, 0xe0]) {
        let len = u16::from_be_bytes([jpeg[4], jpeg[5]]) as usize;
        at = 4 + len;
    }
    let mut out = Vec::with_capacity(jpeg.len() + exif.len() + 10);
    out.extend_from_slice(&jpeg[..at]);
    out.extend_from_slice(&[0xff, 0xe1]);
    out.extend_from_slice(&((exif.len() + 8) as u16).to_be_bytes());
    out.extend_from_slice(b"Exif\0\0");
    out.extend_from_slice(exif);
    out.extend_from_slice(&jpeg[at..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app1_goes_after_jfif() {
        let jpeg = [
            0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46, 0xff, 0xda, 0x00,
        ];
        let out = with_app1_exif(&jpeg, b"II*\0").unwrap();
        assert_eq!(&out[..8], &jpeg[..8]);
        assert_eq!(&out[8..22], b"\xff\xe1\x00\x0cExif\0\0II*\0");
        assert_eq!(&out[22..], &jpeg[8..]);

        let bare = [0xff, 0xd8, 0xff, 0xdb];
        let out = with_app1_exif(&bare, b"MM").unwrap();
        assert_eq!(&out[2..4], &[0xff, 0xe1]);
        assert_eq!(&out[out.len() - 2..], &[0xff, 0xdb]);
    }
}
