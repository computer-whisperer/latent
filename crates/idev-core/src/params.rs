//! Edit state. This struct is the contract between CLI flags, GUI
//! controls, sidecar files, and the pipeline, so it is serialized with
//! defaults for every field and a schema version.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WhiteBalance {
    /// The camera's as-shot multipliers.
    #[default]
    AsShot,
    /// Explicit RGB multipliers, green normalised to 1.
    Multipliers { red: f32, blue: f32 },
    /// A scene white on the daylight/Planckian locus, offset by tint
    /// (Duv x 1000, positive toward magenta).
    Temperature { kelvin: f32, tint: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Demosaic {
    /// Fast, soft; the preview path.
    Bilinear,
    /// Ratio Corrected Demosaicing; the quality path.
    #[default]
    Rcd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Highlights {
    /// Clamp every channel at green's clip level. Clipped areas go
    /// flat white at exactly diffuse white.
    Clip,
    /// A clipped channel is a lower bound; lift it to the brightest
    /// still-valid channel. Fully clipped pixels become a neutral at
    /// the luminance of the per-channel-clamped value, above diffuse
    /// white, so the tone curve has headroom to roll off.
    #[default]
    Blend,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ToneCurve {
    /// Scene-referred output, no curve.
    None,
    /// Grey-anchored sigmoid in linear light: 0.18 maps to 0.18, the
    /// slope at grey scales with `contrast`, and the curve approaches
    /// the output target's peak at scene value `white` (relative to
    /// diffuse white: 1.0 is diffuse white itself, 4.0 two stops above);
    /// anything brighter clips there. `hue_preservation` blends from
    /// per-channel (0) toward a luminance-ratio result (1).
    Sigmoid {
        contrast: f32,
        hue_preservation: f32,
        #[serde(default = "default_white")]
        white: f32,
    },
}

pub const DEFAULT_WHITE: f32 = 4.0;

fn default_white() -> f32 {
    DEFAULT_WHITE
}

impl Default for ToneCurve {
    fn default() -> Self {
        ToneCurve::Sigmoid {
            contrast: 1.6,
            hue_preservation: 0.0,
            white: DEFAULT_WHITE,
        }
    }
}

/// The sigmoid's constants for a target peak: `(contrast, s, white)`
/// such that `f(x) = peak * F(x) / F(white)` with `F(x) = x^c / (x^c + s)`
/// passes through grey and reaches `peak` at `white`.
///
/// Anchoring grey fixes `s` for a given contrast; that solution only
/// exists when the contrast is above `ln(peak/grey) / ln(white/grey)`,
/// so the contrast is raised to that floor when necessary, and `white`
/// is held above the peak (a white below the display's peak is
/// meaningless).
pub fn sigmoid_constants(contrast: f32, white: f32, peak: f32) -> (f32, f32, f32) {
    const GREY: f32 = 0.18;
    let white = white.max(peak * 1.001).max(GREY * 1.5);
    let c_min = (peak / GREY).ln() / (white / GREY).ln();
    let c = contrast.max(c_min * 1.02);
    let r = GREY / peak;
    let a = GREY.powf(c);
    let b = white.powf(c);
    let s = a * b * (1.0 - r) / (r * b - a);
    (c, s.max(1e-6), white)
}

/// Which of the camera's lens correction tables to apply (no-ops when
/// the file has none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LensParams {
    pub distortion: bool,
    pub chromatic_aberration: bool,
    pub vignetting: bool,
}

impl Default for LensParams {
    fn default() -> Self {
        LensParams {
            distortion: true,
            chromatic_aberration: true,
            vignetting: true,
        }
    }
}

impl LensParams {
    pub const NONE: LensParams = LensParams {
        distortion: false,
        chromatic_aberration: false,
        vignetting: false,
    };
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DevelopParams {
    pub schema: u32,
    /// Exposure compensation in stops, applied in linear light after
    /// white balance.
    pub exposure_ev: f32,
    pub white_balance: WhiteBalance,
    pub demosaic: Demosaic,
    pub highlights: Highlights,
    pub tone: ToneCurve,
    pub lens: LensParams,
    pub geometry: Geometry,
    pub sharpen: Sharpen,
}

/// Unsharp mask on luminance at output resolution (after resize).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sharpen {
    /// Strength; 0 disables. 0.5–1.5 is the usual range.
    pub amount: f32,
    /// Gaussian sigma in output pixels.
    pub radius: f32,
    /// Detail below this (in gamma-2.2 luminance, 0..1) is left alone,
    /// so noise in flat areas is not amplified.
    pub threshold: f32,
}

impl Default for Sharpen {
    fn default() -> Self {
        Sharpen {
            amount: 0.0,
            radius: 0.8,
            threshold: 0.0,
        }
    }
}

/// Straighten and crop, applied after the EXIF orientation (see
/// `geometry`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Geometry {
    /// Straighten angle in degrees, positive turns the image clockwise.
    pub angle_deg: f32,
    /// Crop as fractions (x, y, width, height) of the oriented frame;
    /// `None` keeps the largest same-aspect rectangle that stays inside
    /// the rotated content (the whole frame at 0°).
    pub crop: Option<[f32; 4]>,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            angle_deg: 0.0,
            crop: None,
        }
    }
}

impl Default for DevelopParams {
    fn default() -> Self {
        Self {
            schema: SCHEMA_VERSION,
            exposure_ev: 0.0,
            white_balance: WhiteBalance::AsShot,
            demosaic: Demosaic::default(),
            highlights: Highlights::default(),
            tone: ToneCurve::default(),
            lens: LensParams::default(),
            geometry: Geometry::default(),
            sharpen: Sharpen::default(),
        }
    }
}

/// Sidecar extension, appended to the raw's full file name.
pub const SIDECAR_SUFFIX: &str = ".idev.toml";

pub fn sidecar_path(raw: &Path) -> PathBuf {
    let mut name = raw
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(SIDECAR_SUFFIX);
    raw.with_file_name(name)
}

impl DevelopParams {
    pub fn load(path: &Path) -> Result<DevelopParams> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let params: DevelopParams =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        if params.schema > SCHEMA_VERSION {
            anyhow::bail!(
                "{}: schema {} is newer than this build ({SCHEMA_VERSION})",
                path.display(),
                params.schema
            );
        }
        Ok(params)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self).context("serialize params")?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
    }

    /// The sidecar beside `raw` if there is one, else defaults.
    pub fn for_raw(raw: &Path) -> Result<DevelopParams> {
        let side = sidecar_path(raw);
        if side.exists() {
            Self::load(&side)
        } else {
            Ok(Self::default())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f32, c: f32, s: f32, white: f32, peak: f32) -> f32 {
        let big_f = |v: f32| v.powf(c) / (v.powf(c) + s);
        peak * big_f(x) / big_f(white)
    }

    #[test]
    fn sigmoid_hits_grey_and_white() {
        for (contrast, white, peak) in [
            (1.6, 4.0, 1.0),
            (1.6, 4.0, 2.0),
            (1.2, 2.0, 1.0),
            (0.5, 8.0, 1.0),
        ] {
            let (c, s, w) = sigmoid_constants(contrast, white, peak);
            assert!(
                (f(0.18, c, s, w, peak) - 0.18).abs() < 1e-3,
                "grey {contrast} {white} {peak}"
            );
            assert!(
                (f(w, c, s, w, peak) - peak).abs() < 1e-3,
                "white {contrast} {white} {peak}"
            );
            assert!(f(0.05, c, s, w, peak) < f(0.5, c, s, w, peak), "monotone");
        }
    }

    #[test]
    fn old_sidecar_without_white_loads() {
        let text =
            "schema = 1\n[tone]\nkind = \"sigmoid\"\ncontrast = 1.6\nhue_preservation = 0.0\n";
        let p: DevelopParams = toml::from_str(text).unwrap();
        assert_eq!(
            p.tone,
            ToneCurve::Sigmoid {
                contrast: 1.6,
                hue_preservation: 0.0,
                white: DEFAULT_WHITE
            }
        );
    }
}
