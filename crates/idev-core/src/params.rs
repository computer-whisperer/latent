//! Edit state. This struct is the contract between CLI flags, GUI
//! controls, sidecar files, and the pipeline, so it is serialized with
//! defaults for every field and a schema version.

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
    #[default]
    Bilinear,
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
}

impl Default for DevelopParams {
    fn default() -> Self {
        Self {
            schema: SCHEMA_VERSION,
            exposure_ev: 0.0,
            white_balance: WhiteBalance::AsShot,
            demosaic: Demosaic::Bilinear,
        }
    }
}
