//! What a develop starts from: a camera raw, or an already-rendered
//! image brought into the working space (linear BT.2020, 1.0 = reference
//! white) so exposure, tone and output handling apply to both.

use std::path::Path;

use achromat::color::{PrimaryVolume, Tf};
use achromat::decode::{self, DecodedImage, Pixels};
use anyhow::{Context, Result};
use half::f16;
use rawler::Orientation;

use crate::color::{
    Gamut, Mat3, XYZ_FROM_BT2020, apply, invert, mul, pq_eotf, xyz_from_chromaticities,
};
use crate::geometry::{self, Layout};
use crate::metadata::{self, Metadata};
use crate::params::{Geometry, LensParams};
use crate::sensor::{self, ExifSummary, SensorFrame};

/// Camera raws the pipeline unpacks with rawler.
pub const RAW_EXTENSIONS: &[&str] = &[
    "arw", "dng", "cr2", "cr3", "nef", "nrw", "raf", "orf", "rw2", "pef", "srw",
];

/// Rendered images achromat decodes.
pub const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "avif", "jxl", "jxr", "exr", "hdr",
];

fn has_extension(path: &Path, list: &[&str]) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .is_some_and(|e| list.contains(&e.as_str()))
}

pub fn is_raw(path: &Path) -> bool {
    has_extension(path, RAW_EXTENSIONS)
}

pub fn is_supported(path: &Path) -> bool {
    is_raw(path) || has_extension(path, IMAGE_EXTENSIONS)
}

/// A rendered image in the working space.
pub struct ImageSource {
    pub width: u32,
    pub height: u32,
    /// Linear BT.2020 RGBA f16, alpha 1, as stored (see `orientation`).
    pub rgba: Vec<f16>,
    /// The source encoding, for the UI ("PQ BT.2020 16-bit").
    pub encoding: String,
    /// From the embedded EXIF; `Normal` when there is none.
    pub orientation: Orientation,
    pub exif: ExifSummary,
    pub metadata: Metadata,
}

/// Camera RGB reconstructed from a burst (`burst::reconstruct`): the
/// reference frame's metadata (its sensor plane dropped) and an RGBA
/// f16 plane at some multiple of the crop, black-subtracted and
/// normalised to white, before white balance.
pub struct CameraRgb {
    pub frame: SensorFrame,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<f16>,
    /// Mean robustness weight per input frame.
    pub mean_weight: Vec<f32>,
    /// Fraction of output channel values no site reached.
    pub uncovered: f32,
}

pub enum Source {
    /// All boxed: each carries a few hundred bytes of tables and
    /// metadata beside its pixel data.
    Raw(Box<SensorFrame>),
    Image(Box<ImageSource>),
    Rgb(Box<CameraRgb>),
}

impl Source {
    pub fn load(path: &Path) -> Result<Source> {
        if is_raw(path) {
            return Ok(Source::Raw(Box::new(sensor::unpack(path)?)));
        }
        let decoded =
            decode::load_straight(path).with_context(|| format!("decode {}", path.display()))?;
        let mut image = ImageSource::from_decoded(&decoded);
        match metadata::from_image_file(path) {
            Ok(Some(embedded)) => {
                image.orientation = embedded.orientation;
                image.exif = embedded.summary;
                image.metadata = embedded.metadata;
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("{}: EXIF ignored: {e:#}", path.display()),
        }
        Ok(Source::Image(Box::new(image)))
    }

    /// Size of the developed image before orientation.
    pub fn size(&self) -> (u32, u32) {
        match self {
            Source::Raw(f) => (f.crop.width, f.crop.height),
            Source::Image(i) => (i.width, i.height),
            Source::Rgb(r) => (r.width, r.height),
        }
    }

    pub fn orientation(&self) -> Orientation {
        match self {
            Source::Raw(f) => f.orientation,
            Source::Image(i) => i.orientation,
            Source::Rgb(r) => r.frame.orientation,
        }
    }

    pub fn exif(&self) -> &ExifSummary {
        match self {
            Source::Raw(f) => &f.exif,
            Source::Image(i) => &i.exif,
            Source::Rgb(r) => &r.frame.exif,
        }
    }

    pub fn metadata(&self) -> &Metadata {
        match self {
            Source::Raw(f) => &f.metadata,
            Source::Image(i) => &i.metadata,
            Source::Rgb(r) => &r.frame.metadata,
        }
    }

    /// Pixels the lens correction cannot fill on each side, x and y
    /// (`LensCorrection::valid_inset`); zero for rendered images and
    /// files without tables.
    pub fn lens_inset(&self, lens: &LensParams) -> (f32, f32) {
        let (w, h) = self.size();
        match self.as_raw().and_then(|f| f.lens.as_ref()) {
            Some(l) => l.valid_inset(w, h, lens.distortion, lens.chromatic_aberration),
            None => (0.0, 0.0),
        }
    }

    /// The output geometry for these parameters.
    pub fn layout(&self, lens: &LensParams, geo: &Geometry) -> Layout {
        geometry::layout(self.size(), self.orientation(), self.lens_inset(lens), geo)
    }

    /// The sensor frame behind a raw or a reconstruction (whose plane
    /// is empty; its tables and metadata are valid).
    pub fn as_raw(&self) -> Option<&SensorFrame> {
        match self {
            Source::Raw(f) => Some(f),
            Source::Rgb(r) => Some(&r.frame),
            Source::Image(_) => None,
        }
    }
}

impl ImageSource {
    /// Linearize with the declared transfer (PQ anchored at the declared
    /// reference white, 203 cd/m² by default) and rotate the primaries
    /// into BT.2020.
    pub fn from_decoded(img: &DecodedImage) -> ImageSource {
        let enc = &img.encoding;
        let lin = linearizer(enc.tf, enc.luminances.map(|l| l.reference as f32));
        let m = bt2020_from(enc.primaries);
        let n = img.width as usize * img.height as usize;
        let mut rgba = Vec::with_capacity(n * 4);
        let mut push = |r: f32, g: f32, b: f32| {
            let v = match m {
                Some(m) => apply(&m, [r, g, b]),
                None => [r, g, b],
            };
            rgba.extend_from_slice(&[
                f16::from_f32(v[0]),
                f16::from_f32(v[1]),
                f16::from_f32(v[2]),
                f16::ONE,
            ]);
        };
        match &img.pixels {
            Pixels::Rgba8(d) => {
                let lut: Vec<f32> = (0..=255u32).map(|v| lin(v as f32 / 255.0)).collect();
                for px in d.chunks_exact(4) {
                    push(
                        lut[px[0] as usize],
                        lut[px[1] as usize],
                        lut[px[2] as usize],
                    );
                }
            }
            Pixels::Rgba16(d) => {
                for px in d.chunks_exact(4) {
                    push(
                        lin(px[0] as f32 / 65535.0),
                        lin(px[1] as f32 / 65535.0),
                        lin(px[2] as f32 / 65535.0),
                    );
                }
            }
            Pixels::RgbaF16(d) => {
                for px in d.chunks_exact(4) {
                    push(
                        lin(px[0].to_f32()),
                        lin(px[1].to_f32()),
                        lin(px[2].to_f32()),
                    );
                }
            }
        }
        let container = match &img.pixels {
            Pixels::Rgba8(_) => "8-bit",
            Pixels::Rgba16(_) => "16-bit",
            Pixels::RgbaF16(_) => "f16",
        };
        let tf = match enc.tf {
            Tf::Srgb => "sRGB",
            Tf::Gamma22 => "gamma 2.2",
            Tf::Bt1886 => "BT.1886",
            Tf::Pq => "PQ",
            Tf::Linear => "linear",
        };
        let primaries = match enc.primaries.snap_to_named(0.01) {
            PrimaryVolume::Srgb => "sRGB",
            PrimaryVolume::DisplayP3 => "P3",
            PrimaryVolume::Bt2020 => "BT.2020",
            PrimaryVolume::Custom(_) => "custom primaries",
        };
        ImageSource {
            width: img.width,
            height: img.height,
            rgba,
            encoding: format!("{tf} · {primaries} · {container}"),
            orientation: Orientation::Normal,
            exif: ExifSummary::default(),
            metadata: Metadata::default(),
        }
    }
}

/// Code value → linear, 1.0 = reference white.
fn linearizer(tf: Tf, reference_nits: Option<f32>) -> impl Fn(f32) -> f32 {
    let pq_scale = 1.0 / reference_nits.unwrap_or(203.0);
    move |v| match tf {
        Tf::Pq => pq_eotf(v) * pq_scale,
        Tf::Linear => v,
        other => other.eotf(v),
    }
}

/// Source primaries → BT.2020, or `None` when they already are.
fn bt2020_from(p: PrimaryVolume) -> Option<Mat3> {
    let xyz_from = match p.snap_to_named(0.01) {
        PrimaryVolume::Bt2020 => return None,
        PrimaryVolume::Srgb => Gamut::Srgb.xyz_from(),
        PrimaryVolume::DisplayP3 => Gamut::DisplayP3.xyz_from(),
        PrimaryVolume::Custom(c) => xyz_from_chromaticities([
            [c.r.0 as f32, c.r.1 as f32],
            [c.g.0 as f32, c.g.1 as f32],
            [c.b.0 as f32, c.b.1 as f32],
            [c.w.0 as f32, c.w.1 as f32],
        ]),
    };
    let bt2020_from_xyz = invert(&XYZ_FROM_BT2020).expect("BT.2020 matrix is invertible");
    Some(mul(&bt2020_from_xyz, &xyz_from))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_white_stays_white() {
        let m = bt2020_from(PrimaryVolume::Srgb).unwrap();
        let w = apply(&m, [1.0, 1.0, 1.0]);
        for c in w {
            assert!((c - 1.0).abs() < 1e-3, "{w:?}");
        }
    }

    #[test]
    fn pq_reference_white_is_one() {
        let lin = linearizer(Tf::Pq, None);
        // PQ code for 203 nits is ~0.5807
        assert!((lin(0.5807) - 1.0).abs() < 0.01);
    }
}
