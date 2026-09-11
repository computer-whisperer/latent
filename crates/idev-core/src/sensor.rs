//! Camera raw unpacking via rawler.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rawler::decoders::RawDecodeParams;
use rawler::formats::tiff::GenericTiffReader;
use rawler::formats::tiff::reader::TiffReader;
use rawler::imgop::xyz::Illuminant;
use rawler::rawsource::RawSource;
use rawler::{Orientation, RawImageData};

/// CFA color of one sensor site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfaColor {
    Red = 0,
    Green = 1,
    Blue = 2,
}

/// Axis-aligned pixel rectangle on the sensor raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Camera-to-XYZ characterisation for one illuminant (DNG `ColorMatrix`
/// convention: rows map XYZ to camera RGB).
#[derive(Debug, Clone, Copy)]
pub struct ColorMatrix {
    pub illuminant: Illuminant,
    pub xyz_to_cam: [[f32; 3]; 3],
}

/// The shooting metadata we carry forward into outputs and the UI.
#[derive(Debug, Clone, Default)]
pub struct ExifSummary {
    pub make: String,
    pub model: String,
    pub lens: Option<String>,
    pub exposure_time: Option<(u32, u32)>,
    pub f_number: Option<f32>,
    pub iso: Option<u32>,
    pub focal_length_mm: Option<f32>,
    pub date_time_original: Option<String>,
}

/// One unpacked raw file: the Bayer plane and how to interpret it.
#[derive(Debug, Clone)]
pub struct SensorFrame {
    /// Full raster dimensions of `data`.
    pub width: u32,
    pub height: u32,
    /// 2x2 CFA pattern, row-major: `[top-left, top-right, bottom-left, bottom-right]`.
    pub cfa: [CfaColor; 4],
    /// Black level per CFA position, same order as `cfa`.
    pub black: [f32; 4],
    /// White (clip) level per CFA position, same order as `cfa`.
    pub white: [f32; 4],
    /// The manufacturer's recommended crop.
    pub crop: Rect,
    /// As-shot white balance multipliers, RGB, green normalised to 1.
    pub wb_as_shot: [f32; 3],
    /// Characterisation matrices, one per illuminant the profile knows.
    pub matrices: Vec<ColorMatrix>,
    pub orientation: Orientation,
    pub exif: ExifSummary,
    /// The camera's lens correction tables, when the file carries them.
    pub lens: Option<LensCorrection>,
    /// Raw sensor values, `width * height`, row-major.
    pub data: Vec<u16>,
}

/// Sony's in-body lens correction tables (raw SubIFD tags 0x7037,
/// 0x7035, 0x7032), decoded the way RawTherapee's `lensmetadata.cc`
/// does: `nc` knots at normalised radii `(i + 0.5) / (nc - 1)`, where
/// 1.0 is the half-diagonal of the image, linear between knots and
/// clamped outside.
#[derive(Debug, Clone, PartialEq)]
pub struct LensCorrection {
    pub knots: Vec<f32>,
    /// Source radius over output radius (distortion).
    pub distortion: Vec<f32>,
    /// Additional radius factor for red and blue (lateral chromatic
    /// aberration).
    pub ca_red: Vec<f32>,
    pub ca_blue: Vec<f32>,
    /// Gain that flattens the vignetting.
    pub vignetting: Vec<f32>,
}

const TAG_VIGNETTING_PARAMS: u16 = 0x7032;
const TAG_CA_PARAMS: u16 = 0x7035;
const TAG_DISTORTION_PARAMS: u16 = 0x7037;

/// Read the correction tables, if the file has consistent ones.
fn lens_correction(source: &RawSource) -> Option<LensCorrection> {
    let tiff = GenericTiffReader::new(&mut source.reader(), 0, 0, None, &[]).ok()?;
    let root = tiff.root_ifd();
    let ints = |tag: u16| -> Option<Vec<i32>> {
        let e = root.get_entry_recursive(tag)?;
        (0..e.value.count())
            .map(|i| e.value.get_i32(i).ok().flatten())
            .collect()
    };
    let dist = ints(TAG_DISTORTION_PARAMS)?;
    let ca = ints(TAG_CA_PARAMS)?;
    let vig = ints(TAG_VIGNETTING_PARAMS)?;
    let nc = *dist.first()? as usize;
    if !(2..=16).contains(&nc)
        || ca.first().copied() != Some(2 * nc as i32)
        || vig.first().copied() != Some(nc as i32)
        || dist.len() < 1 + nc
        || ca.len() < 1 + 2 * nc
        || vig.len() < 1 + nc
    {
        tracing::warn!(
            ?dist,
            ?ca,
            ?vig,
            "inconsistent lens correction tables, ignoring"
        );
        return None;
    }
    let knots = (0..nc)
        .map(|i| (i as f32 + 0.5) / (nc as f32 - 1.0))
        .collect();
    Some(LensCorrection {
        knots,
        distortion: dist[1..=nc]
            .iter()
            .map(|&v| v as f32 * 2f32.powi(-14) + 1.0)
            .collect(),
        ca_red: ca[1..=nc]
            .iter()
            .map(|&v| v as f32 * 2f32.powi(-21) + 1.0)
            .collect(),
        ca_blue: ca[1 + nc..=2 * nc]
            .iter()
            .map(|&v| v as f32 * 2f32.powi(-21) + 1.0)
            .collect(),
        vignetting: vig[1..=nc]
            .iter()
            .map(|&v| 1.0 / 2f32.powf(0.5 - 2f32.powf(v as f32 * 2f32.powi(-13) - 1.0)))
            .collect(),
    })
}

impl SensorFrame {
    pub fn cfa_at(&self, x: u32, y: u32) -> CfaColor {
        self.cfa[((y & 1) * 2 + (x & 1)) as usize]
    }

    /// Camera-space neutral implied by the as-shot multipliers.
    pub fn neutral_as_shot(&self) -> [f32; 3] {
        [1.0 / self.wb_as_shot[0], 1.0, 1.0 / self.wb_as_shot[2]]
    }
}

/// The camera's own JPEG rendering embedded in the raw, as stored (not
/// yet rotated by `orientation`).
pub struct Preview {
    pub width: u32,
    pub height: u32,
    /// sRGB-encoded RGB, three bytes per pixel.
    pub rgb: Vec<u8>,
    pub orientation: Orientation,
}

/// Read the embedded JPEG preview without decoding the raw data.
pub fn embedded_preview(path: &Path) -> Result<Preview> {
    let source = RawSource::new(path).with_context(|| format!("open {}", path.display()))?;
    let decoder = rawler::get_decoder(&source)
        .with_context(|| format!("no decoder for {}", path.display()))?;
    let params = RawDecodeParams::default();
    let image = decoder
        .preview_image(&source, &params)
        .with_context(|| format!("preview {}", path.display()))?
        .with_context(|| format!("{}: no embedded preview", path.display()))?;
    let meta = decoder
        .raw_metadata(&source, &params)
        .with_context(|| format!("metadata {}", path.display()))?;
    let orientation = match meta.exif.orientation {
        Some(o @ 1..=8) => Orientation::from_u16(o),
        _ => Orientation::Normal,
    };
    let rgb = image.into_rgb8();
    Ok(Preview {
        width: rgb.width(),
        height: rgb.height(),
        rgb: rgb.into_raw(),
        orientation,
    })
}

pub fn unpack(path: &Path) -> Result<SensorFrame> {
    let source = RawSource::new(path).with_context(|| format!("open {}", path.display()))?;
    let decoder = rawler::get_decoder(&source)
        .with_context(|| format!("no decoder for {}", path.display()))?;
    let params = RawDecodeParams::default();
    let raw = decoder
        .raw_image(&source, &params, false)
        .with_context(|| format!("decode {}", path.display()))?;
    let meta = decoder
        .raw_metadata(&source, &params)
        .with_context(|| format!("metadata {}", path.display()))?;
    let lens = lens_correction(&source);

    if raw.cpp != 1 {
        bail!(
            "{}: {} components per pixel; only Bayer mosaics are supported",
            path.display(),
            raw.cpp
        );
    }
    let cfa = &raw.camera.cfa;
    if cfa.width != 2 || cfa.height != 2 {
        bail!(
            "{}: CFA pattern {}x{} is not a 2x2 Bayer mosaic",
            path.display(),
            cfa.width,
            cfa.height
        );
    }
    let mut pattern = [CfaColor::Green; 4];
    for (i, slot) in pattern.iter_mut().enumerate() {
        *slot = match cfa.color_at(i / 2, i % 2) {
            0 => CfaColor::Red,
            1 => CfaColor::Green,
            2 => CfaColor::Blue,
            other => bail!("{}: CFA color index {other} is not R/G/B", path.display()),
        };
    }

    let data = match raw.data {
        RawImageData::Integer(v) => v,
        RawImageData::Float(_) => bail!("{}: float raw data is not supported yet", path.display()),
    };
    if data.len() != raw.width * raw.height {
        bail!(
            "{}: data length {} != {}x{}",
            path.display(),
            data.len(),
            raw.width,
            raw.height
        );
    }

    let black = raw.blacklevel.as_bayer_array();
    // rawler's white level is per color plane (RGBE); spread it over the
    // CFA positions so both tables index the same way.
    let white_planes = raw.whitelevel.0.clone();
    let mut white = [0.0f32; 4];
    for (i, w) in white.iter_mut().enumerate() {
        let plane = pattern[i] as usize;
        *w = *white_planes
            .get(plane)
            .or_else(|| white_planes.first())
            .context("empty white level")? as f32;
    }

    let crop = raw
        .crop_area
        .or(raw.active_area)
        .map(|r| Rect {
            x: r.p.x as u32,
            y: r.p.y as u32,
            width: r.d.w as u32,
            height: r.d.h as u32,
        })
        .unwrap_or(Rect {
            x: 0,
            y: 0,
            width: raw.width as u32,
            height: raw.height as u32,
        });

    let wb = raw.wb_coeffs;
    let g = if wb[1] > 0.0 { wb[1] } else { 1.0 };
    let wb_as_shot = [wb[0] / g, 1.0, wb[2] / g];

    let mut matrices: Vec<ColorMatrix> = raw
        .color_matrix
        .iter()
        .filter(|(_, m)| m.len() == 9)
        .map(|(ill, m)| ColorMatrix {
            illuminant: *ill,
            xyz_to_cam: [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]],
        })
        .collect();
    matrices.sort_by_key(|m| m.illuminant as u32);
    if matrices.is_empty() {
        bail!(
            "{}: no color matrix for {} {}",
            path.display(),
            raw.clean_make,
            raw.clean_model
        );
    }

    let exif = &meta.exif;
    // rawler's own orientation is Normal for Sony files whose EXIF says
    // otherwise; the EXIF tag is authoritative when present.
    let orientation = match exif.orientation {
        Some(o @ 1..=8) => Orientation::from_u16(o),
        _ => raw.orientation,
    };
    let summary = ExifSummary {
        make: raw.clean_make.clone(),
        model: raw.clean_model.clone(),
        lens: meta.lens.as_ref().map(|l| l.lens_name.clone()),
        exposure_time: exif.exposure_time.map(|r| (r.n, r.d)),
        f_number: exif.fnumber.map(|r| r.as_f32()),
        iso: exif.iso_speed_ratings.map(u32::from),
        focal_length_mm: exif.focal_length.map(|r| r.as_f32()),
        date_time_original: exif.date_time_original.clone(),
    };

    Ok(SensorFrame {
        width: raw.width as u32,
        height: raw.height as u32,
        cfa: pattern,
        black,
        white,
        crop,
        wb_as_shot,
        matrices,
        orientation,
        exif: summary,
        lens,
        data,
    })
}
