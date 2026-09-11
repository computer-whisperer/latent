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
    /// the output target's peak asymptotically. `hue_preservation`
    /// blends from per-channel (0) toward a luminance-ratio result (1).
    Sigmoid {
        contrast: f32,
        hue_preservation: f32,
    },
}

impl Default for ToneCurve {
    fn default() -> Self {
        ToneCurve::Sigmoid {
            contrast: 1.6,
            hue_preservation: 0.0,
        }
    }
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
