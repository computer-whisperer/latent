//! AVIF via rav1e (AV1 encoder) and avif-serialize (the ISOBMFF
//! container), both pure Rust. Still picture, 4:4:4, identity matrix
//! (the planes carry G, B, R directly, so no YCbCr rounding), full
//! range, 10-bit samples in both flavours:
//!
//! - SDR: sRGB 8-bit input widened to 10 bits, tagged sRGB / BT.709.
//! - HDR: BT.2020 PQ 16-bit input reduced to 10 bits, tagged BT.2020 /
//!   PQ, i.e. the same cICP `[9, 16, 0, 1]` as the PQ PNG.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use avif_serialize::constants::{ColorPrimaries, MatrixCoefficients, TransferCharacteristics};
use rav1e::prelude::*;
use v_frame::plane::PlaneOffset;

/// rav1e speed preset, 0 (slowest) to 10. 6 is a sensible still-image
/// default; the pure-Rust build (no assembly) is several times slower
/// than a native libaom at the same preset.
const SPEED: u8 = 6;

/// sRGB 8-bit RGB, three bytes per pixel. `quality` 1..=100. `exif` is
/// a TIFF-structured EXIF block, stored as an `Exif` item.
pub fn write_avif_srgb8(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[u8],
    quality: u8,
    exif: Option<&[u8]>,
) -> Result<()> {
    ensure!(
        rgb.len() == (width * height * 3) as usize,
        "avif: pixel buffer size mismatch"
    );
    let planes = rgb
        .chunks_exact(3)
        .map(|px| [widen8(px[1]), widen8(px[2]), widen8(px[0])]);
    let av1 = encode_av1(
        width,
        height,
        quality,
        ColorDescription {
            color_primaries: rav1e::color::ColorPrimaries::BT709,
            transfer_characteristics: rav1e::color::TransferCharacteristics::SRGB,
            matrix_coefficients: rav1e::color::MatrixCoefficients::Identity,
        },
        planes,
    )?;
    write_container(
        path,
        &av1,
        width,
        height,
        ColorPrimaries::Bt709,
        TransferCharacteristics::Srgb,
        exif,
    )
}

/// BT.2020 primaries, PQ transfer, 16-bit samples (as `output::bt2020_pq16`
/// produces), three per pixel. `quality` 1..=100. `exif` as above.
pub fn write_avif_bt2020_pq10(
    path: &Path,
    width: u32,
    height: u32,
    rgb: &[u16],
    quality: u8,
    exif: Option<&[u8]>,
) -> Result<()> {
    ensure!(
        rgb.len() == (width * height * 3) as usize,
        "avif: pixel buffer size mismatch"
    );
    let planes = rgb
        .chunks_exact(3)
        .map(|px| [narrow16(px[1]), narrow16(px[2]), narrow16(px[0])]);
    let av1 = encode_av1(
        width,
        height,
        quality,
        ColorDescription {
            color_primaries: rav1e::color::ColorPrimaries::BT2020,
            transfer_characteristics: rav1e::color::TransferCharacteristics::SMPTE2084,
            matrix_coefficients: rav1e::color::MatrixCoefficients::Identity,
        },
        planes,
    )?;
    write_container(
        path,
        &av1,
        width,
        height,
        ColorPrimaries::Bt2020,
        TransferCharacteristics::Smpte2084,
        exif,
    )
}

/// 8-bit to 10-bit, replicating the top bits so 255 maps to 1023.
fn widen8(v: u8) -> u16 {
    (u16::from(v) << 2) | (u16::from(v) >> 6)
}

/// 16-bit to 10-bit, rounded.
fn narrow16(v: u16) -> u16 {
    ((u32::from(v) + 32) >> 6).min(1023) as u16
}

/// ravif's mapping from a 1..=100 quality to rav1e's 0..=255 quantizer.
fn quantizer(quality: u8) -> usize {
    let q = f32::from(quality.clamp(1, 100)) / 100.0;
    let x = if q >= 0.82 {
        (1.0 - q) * 2.6
    } else if q > 0.25 {
        q.mul_add(-0.5, 1.0 - 0.125)
    } else {
        1.0 - q
    };
    (x * 255.0).round() as usize
}

/// Encode one 10-bit 4:4:4 key frame; returns the OBU stream.
fn encode_av1(
    width: u32,
    height: u32,
    quality: u8,
    color_description: ColorDescription,
    planes: impl Iterator<Item = [u16; 3]>,
) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Tiles let rav1e use the threads; too many tiny tiles cost
    // efficiency, so cap by area (64x64 blocks per tile at least).
    let tiles = threads.min((w * h) / (64 * 64)).max(1);
    let q = quantizer(quality);
    let cfg = Config::new()
        .with_encoder_config(EncoderConfig {
            width: w,
            height: h,
            time_base: Rational::new(1, 1),
            sample_aspect_ratio: Rational::new(1, 1),
            bit_depth: 10,
            chroma_sampling: ChromaSampling::Cs444,
            chroma_sample_position: ChromaSamplePosition::Unknown,
            pixel_range: PixelRange::Full,
            color_description: Some(color_description),
            mastering_display: None,
            content_light: None,
            enable_timing_info: false,
            still_picture: true,
            error_resilient: false,
            switch_frame_interval: 0,
            min_key_frame_interval: 0,
            max_key_frame_interval: 0,
            reservoir_frame_delay: None,
            low_latency: false,
            quantizer: q,
            min_quantizer: q as u8,
            bitrate: 0,
            tune: Tune::Psychovisual,
            tile_cols: 0,
            tile_rows: 0,
            tiles,
            film_grain_params: None,
            level_idx: None,
            speed_settings: SpeedSettings::from_preset(SPEED),
        })
        .with_threads(threads);
    let mut ctx: rav1e::Context<u16> = cfg.new_context().context("rav1e context")?;
    let mut frame = ctx.new_frame();
    {
        let mut it = frame.planes.iter_mut();
        let (mut p0, mut p1, mut p2) = (
            it.next().unwrap().mut_slice(PlaneOffset::default()),
            it.next().unwrap().mut_slice(PlaneOffset::default()),
            it.next().unwrap().mut_slice(PlaneOffset::default()),
        );
        let mut planes = planes;
        for ((r0, r1), r2) in p0
            .rows_iter_mut()
            .zip(p1.rows_iter_mut())
            .zip(p2.rows_iter_mut())
            .take(h)
        {
            for ((a, b), c) in r0[..w].iter_mut().zip(&mut r1[..w]).zip(&mut r2[..w]) {
                let px = planes.next().context("avif: too few pixels")?;
                *a = px[0];
                *b = px[1];
                *c = px[2];
            }
        }
    }
    ctx.send_frame(frame).context("rav1e send_frame")?;
    ctx.flush();
    let mut out = Vec::new();
    loop {
        match ctx.receive_packet() {
            Ok(mut packet) => {
                if packet.frame_type == FrameType::KEY {
                    out.append(&mut packet.data);
                }
            }
            Err(EncoderStatus::Encoded | EncoderStatus::LimitReached) => break,
            Err(e) => return Err(anyhow::anyhow!("rav1e: {e}")),
        }
    }
    Ok(out)
}

fn write_container(
    path: &Path,
    av1: &[u8],
    width: u32,
    height: u32,
    primaries: ColorPrimaries,
    transfer: TransferCharacteristics,
    exif: Option<&[u8]>,
) -> Result<()> {
    let mut aviffy = avif_serialize::Aviffy::new();
    aviffy
        .set_color_primaries(primaries)
        .set_transfer_characteristics(transfer)
        .set_matrix_coefficients(MatrixCoefficients::Rgb)
        .set_full_color_range(true);
    if let Some(exif) = exif {
        aviffy.set_exif(exif.to_vec());
    }
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    aviffy
        .write(std::io::BufWriter::new(file), av1, None, width, height, 10)
        .with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_conversions() {
        assert_eq!(widen8(0), 0);
        assert_eq!(widen8(255), 1023);
        assert_eq!(narrow16(0), 0);
        assert_eq!(narrow16(65535), 1023);
        assert_eq!(narrow16(32768), 512);
    }

    #[test]
    fn quality_maps_monotonically() {
        assert!(quantizer(100) < quantizer(90));
        assert!(quantizer(90) < quantizer(50));
        assert!(quantizer(50) < quantizer(10));
    }
}
