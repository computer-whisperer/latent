//! Scene white point: from as-shot multipliers to XYZ and CCT, matrix
//! interpolation between profile illuminants, Bradford adaptation, and
//! the kelvin/tint ↔ chromaticity mapping the white-balance control uses.
//!
//! Method follows the DNG specification's camera-to-XYZ construction:
//! the camera neutral is mapped through the colour matrix interpolated
//! for its own correlated colour temperature (a fixed point found by
//! iteration), the resulting white is adapted to D65 with Bradford, and
//! the working matrix is built so that the neutral lands on working
//! white exactly.

use rawler::imgop::xyz::Illuminant;

use crate::color::{Mat3, XYZ_FROM_BT2020, apply, invert, mul};
use crate::sensor::ColorMatrix;

/// CIE 1931 chromaticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xy {
    pub x: f32,
    pub y: f32,
}

pub const D65: Xy = Xy {
    x: 0.3127,
    y: 0.3290,
};

impl Xy {
    /// XYZ with Y = 1.
    pub fn to_xyz(self) -> [f32; 3] {
        [self.x / self.y, 1.0, (1.0 - self.x - self.y) / self.y]
    }

    pub fn from_xyz(v: [f32; 3]) -> Xy {
        let s = v[0] + v[1] + v[2];
        if s <= 0.0 {
            return D65;
        }
        Xy {
            x: v[0] / s,
            y: v[1] / s,
        }
    }

    /// CIE 1976 u'v'.
    pub fn to_uv(self) -> (f32, f32) {
        let d = -2.0 * self.x + 12.0 * self.y + 3.0;
        (4.0 * self.x / d, 9.0 * self.y / d)
    }

    pub fn from_uv(u: f32, v: f32) -> Xy {
        let d = 6.0 * u - 16.0 * v + 12.0;
        Xy {
            x: 9.0 * u / d,
            y: 4.0 * v / d,
        }
    }
}

/// White balance as the user sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature {
    pub kelvin: f32,
    /// Offset from the locus in CIE 1976 Duv, times 1000. Positive is
    /// toward magenta (below the locus), negative toward green.
    pub tint: f32,
}

/// Chromaticity of the locus at `kelvin`: CIE daylight (D series) from
/// 4000 K up, the Planckian approximation of Kim et al. below.
pub fn locus_xy(kelvin: f32) -> Xy {
    let t = kelvin.clamp(1667.0, 25000.0);
    if t >= 4000.0 {
        let inv = 1000.0 / t;
        let x = if t <= 7000.0 {
            0.244063 + 0.09911 * inv + 2.9678 * inv * inv - 4.6070 * inv * inv * inv
        } else {
            0.237040 + 0.24748 * inv + 1.9018 * inv * inv - 2.0064 * inv * inv * inv
        };
        let y = -3.0 * x * x + 2.870 * x - 0.275;
        Xy { x, y }
    } else {
        let inv = 1000.0 / t;
        let x = -0.2661239 * inv * inv * inv - 0.2343589 * inv * inv + 0.8776956 * inv + 0.179910;
        let y = if t <= 2222.0 {
            -1.1063814 * x * x * x - 1.348_110_2 * x * x + 2.185_558_3 * x - 0.20219683
        } else {
            -0.9549476 * x * x * x - 1.374_185_9 * x * x + 2.091_370_2 * x - 0.167_488_67
        };
        Xy { x, y }
    }
}

/// Chromaticity for a temperature/tint pair: the locus point shifted
/// along its normal in u'v' by `tint / 1000`.
pub fn xy_from_temperature(t: Temperature) -> Xy {
    let k = t.kelvin.clamp(1667.0, 25000.0);
    let (u0, v0) = locus_xy(k).to_uv();
    // Tangent from a small step along the locus.
    let (u1, v1) = locus_xy((k * 1.01).min(25000.0)).to_uv();
    let (mut tu, mut tv) = (u1 - u0, v1 - v0);
    let len = (tu * tu + tv * tv).sqrt().max(1e-9);
    tu /= len;
    tv /= len;
    // Normal pointing toward +v' (green); magenta is the other way.
    let (nu, nv) = (-tv, tu);
    let duv = -t.tint / 1000.0;
    Xy::from_uv(u0 + nu * duv, v0 + nv * duv)
}

/// Temperature/tint for a chromaticity: the locus point nearest in
/// u'v' (a golden-section search in mired) defines the temperature; the signed distance
/// to it is the tint. This makes the function the exact inverse of
/// [`xy_from_temperature`].
pub fn temperature_from_xy(c: Xy) -> Temperature {
    let (u, v) = c.to_uv();
    let dist2 = |mired: f32| {
        let (lu, lv) = locus_xy(1e6 / mired).to_uv();
        (u - lu) * (u - lu) + (v - lv) * (v - lv)
    };
    let (mut lo, mut hi) = (1e6 / 25000.0, 1e6 / 1667.0);
    let phi = 0.618_034;
    let (mut a, mut b) = (hi - phi * (hi - lo), lo + phi * (hi - lo));
    let (mut fa, mut fb) = (dist2(a), dist2(b));
    for _ in 0..60 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - phi * (hi - lo);
            fa = dist2(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + phi * (hi - lo);
            fb = dist2(b);
        }
    }
    let kelvin = 1e6 / (0.5 * (lo + hi));
    let (u0, v0) = locus_xy(kelvin).to_uv();
    let (u1, v1) = locus_xy((kelvin * 1.01).min(25000.0)).to_uv();
    let (mut tu, mut tv) = (u1 - u0, v1 - v0);
    let len = (tu * tu + tv * tv).sqrt().max(1e-9);
    tu /= len;
    tv /= len;
    let (nu, nv) = (-tv, tu);
    let duv = (u - u0) * nu + (v - v0) * nv;
    Temperature {
        kelvin,
        tint: -duv * 1000.0,
    }
}

/// Bradford chromatic adaptation from `src` to `dst` (XYZ whites).
pub fn bradford(src: [f32; 3], dst: [f32; 3]) -> Mat3 {
    const M: Mat3 = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let m_inv = invert(&M).expect("Bradford matrix is invertible");
    let s = apply(&M, src);
    let d = apply(&M, dst);
    let scale = [
        [d[0] / s[0], 0.0, 0.0],
        [0.0, d[1] / s[1], 0.0],
        [0.0, 0.0, d[2] / s[2]],
    ];
    mul(&m_inv, &mul(&scale, &M))
}

/// Nominal CCT of a DNG illuminant, for matrix interpolation.
pub fn illuminant_cct(i: Illuminant) -> Option<f32> {
    Some(match i {
        Illuminant::A | Illuminant::Tungsten | Illuminant::IsoStudioTungsten => 2856.0,
        Illuminant::B => 4874.0,
        Illuminant::C => 6774.0,
        Illuminant::D50 => 5003.0,
        Illuminant::D55 => 5503.0,
        Illuminant::D65 => 6504.0,
        Illuminant::D75 => 7504.0,
        Illuminant::Daylight | Illuminant::Flash | Illuminant::FineWeather => 5500.0,
        Illuminant::CloudyWeather => 6500.0,
        Illuminant::Shade => 7500.0,
        Illuminant::Fluorescent | Illuminant::DaylightFluorescent => 6430.0,
        Illuminant::DaylightWhiteFluorescent => 5000.0,
        Illuminant::CoolWhiteFluorescent => 4230.0,
        Illuminant::WhiteFluorescent => 3450.0,
        _ => return None,
    })
}

/// The profile's XYZ→camera matrix for a scene temperature: linear
/// interpolation in inverse CCT between the two nearest illuminants,
/// clamped at the ends (DNG 1.4 §6 "Mapping Camera Color Space to CIE
/// XYZ Space").
pub fn interpolate_matrix(matrices: &[ColorMatrix], kelvin: f32) -> Mat3 {
    let mut known: Vec<(f32, Mat3)> = matrices
        .iter()
        .filter_map(|m| illuminant_cct(m.illuminant).map(|t| (t, m.xyz_to_cam)))
        .collect();
    if known.is_empty() {
        return matrices.first().map(|m| m.xyz_to_cam).unwrap_or([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
    }
    known.sort_by(|a, b| a.0.total_cmp(&b.0));
    if known.len() == 1 || kelvin <= known[0].0 {
        return known[0].1;
    }
    if kelvin >= known[known.len() - 1].0 {
        return known[known.len() - 1].1;
    }
    let hi = known.iter().position(|(t, _)| *t >= kelvin).unwrap();
    let (t_lo, m_lo) = known[hi - 1];
    let (t_hi, m_hi) = known[hi];
    let w = ((1.0 / kelvin) - (1.0 / t_lo)) / ((1.0 / t_hi) - (1.0 / t_lo));
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m_lo[i][j] * (1.0 - w) + m_hi[i][j] * w;
        }
    }
    out
}

/// Everything the pipeline needs about the scene white.
#[derive(Debug, Clone, Copy)]
pub struct SceneWhite {
    /// Camera-space response to the scene neutral, green normalised.
    pub neutral: [f32; 3],
    pub xyz: [f32; 3],
    pub xy: Xy,
    pub temperature: Temperature,
    /// XYZ→camera interpolated for this white.
    pub xyz_to_cam: Mat3,
}

/// Solve the white for a camera neutral (`1 / wb` multipliers). The
/// matrix depends on the CCT which depends on the matrix; three
/// iterations from D65 converge to well under a kelvin.
pub fn white_from_neutral(matrices: &[ColorMatrix], neutral: [f32; 3]) -> Option<SceneWhite> {
    let mut kelvin = 6504.0;
    let mut xyz = [0.0; 3];
    let mut m = interpolate_matrix(matrices, kelvin);
    for _ in 0..3 {
        m = interpolate_matrix(matrices, kelvin);
        let cam_to_xyz = invert(&m)?;
        xyz = apply(&cam_to_xyz, neutral);
        if xyz[1] <= 0.0 {
            return None;
        }
        kelvin = temperature_from_xy(Xy::from_xyz(xyz)).kelvin;
    }
    let xy = Xy::from_xyz(xyz);
    Some(SceneWhite {
        neutral,
        xyz,
        xy,
        temperature: temperature_from_xy(xy),
        xyz_to_cam: m,
    })
}

/// Solve the white for a chosen chromaticity: find the matrix for its
/// CCT, map XYZ to camera space, and that is the neutral.
pub fn white_from_xy(matrices: &[ColorMatrix], xy: Xy) -> Option<SceneWhite> {
    let temperature = temperature_from_xy(xy);
    let m = interpolate_matrix(matrices, temperature.kelvin);
    let xyz = xy.to_xyz();
    let n = apply(&m, xyz);
    if n[1] <= 0.0 {
        return None;
    }
    let neutral = [n[0] / n[1], 1.0, n[2] / n[1]];
    Some(SceneWhite {
        neutral,
        xyz,
        xy,
        temperature,
        xyz_to_cam: m,
    })
}

impl SceneWhite {
    /// White-balance multipliers, green normalised.
    pub fn multipliers(&self) -> [f32; 3] {
        [1.0 / self.neutral[0], 1.0, 1.0 / self.neutral[2]]
    }

    /// Camera RGB after white balance (neutral = `(1, 1, 1)`) → linear
    /// BT.2020 (D65). Neutrals map to `(1, 1, 1)` exactly; everything
    /// else is Bradford-adapted from the scene white to D65.
    pub fn bt2020_from_camera_wb(&self) -> Option<Mat3> {
        let cam_to_xyz = invert(&self.xyz_to_cam)?;
        let adapt = bradford(self.xyz, D65.to_xyz());
        let bt2020_from_xyz = invert(&XYZ_FROM_BT2020)?;
        let n = self.neutral;
        let diag = [[n[0], 0.0, 0.0], [0.0, n[1], 0.0], [0.0, 0.0, n[2]]];
        let m = mul(&bt2020_from_xyz, &mul(&adapt, &mul(&cam_to_xyz, &diag)));
        // The construction lands the neutral on D65 by design; scale so
        // Y of the neutral is exactly 1 (the profile's absolute scale is
        // arbitrary).
        let w = apply(&m, [1.0, 1.0, 1.0]);
        let y = 0.2627 * w[0] + 0.6780 * w[1] + 0.0593 * w[2];
        if y <= 0.0 {
            return None;
        }
        Some(m.map(|r| r.map(|v| v / y)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a6700() -> Vec<ColorMatrix> {
        vec![
            ColorMatrix {
                illuminant: Illuminant::A,
                xyz_to_cam: [
                    [0.793, -0.406, 0.0417],
                    [-0.374, 1.1119, 0.3015],
                    [-0.0202, 0.0741, 0.6896],
                ],
            },
            ColorMatrix {
                illuminant: Illuminant::D65,
                xyz_to_cam: [
                    [0.6972, -0.2408, -0.06],
                    [-0.433, 1.2101, 0.2515],
                    [-0.0388, 0.1277, 0.5847],
                ],
            },
        ]
    }

    #[test]
    fn locus_roundtrip() {
        for k in [2500.0, 3200.0, 4500.0, 5500.0, 6504.0, 8000.0, 12000.0] {
            let xy = locus_xy(k);
            let t = temperature_from_xy(xy);
            assert!((t.kelvin - k).abs() / k < 0.005, "{k} -> {t:?}");
            assert!(t.tint.abs() < 0.5, "{k} -> {t:?}");
        }
        assert!((locus_xy(6504.0).x - D65.x).abs() < 0.002);
        assert!((locus_xy(6504.0).y - D65.y).abs() < 0.002);
    }

    #[test]
    fn tint_roundtrip() {
        let t = Temperature {
            kelvin: 5000.0,
            tint: 12.0,
        };
        let back = temperature_from_xy(xy_from_temperature(t));
        assert!((back.kelvin - 5000.0).abs() < 25.0, "{back:?}");
        assert!((back.tint - 12.0).abs() < 0.1, "{back:?}");
    }

    #[test]
    fn bradford_identity() {
        let w = D65.to_xyz();
        let m = bradford(w, w);
        for (i, row) in m.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn d65_neutral_solves_to_d65() {
        let mats = a6700();
        let d65 = mats[1].xyz_to_cam;
        let n = apply(&d65, D65.to_xyz());
        let n = [n[0] / n[1], 1.0, n[2] / n[1]];
        let w = white_from_neutral(&mats, n).unwrap();
        assert!(
            (w.temperature.kelvin - 6504.0).abs() < 150.0,
            "{:?}",
            w.temperature
        );
        assert!(
            (w.xy.x - D65.x).abs() < 0.003 && (w.xy.y - D65.y).abs() < 0.003,
            "{:?}",
            w.xy
        );
        let m = w.bt2020_from_camera_wb().unwrap();
        let white = apply(&m, [1.0, 1.0, 1.0]);
        for v in white {
            assert!((v - 1.0).abs() < 1e-3, "{white:?}");
        }
    }

    #[test]
    fn as_shot_multipliers_roundtrip_through_xy() {
        let mats = a6700();
        let wb = [2.8916016f32, 1.0, 1.4013672];
        let w = white_from_neutral(&mats, [1.0 / wb[0], 1.0, 1.0 / wb[2]]).unwrap();
        let back = white_from_xy(&mats, w.xy).unwrap();
        let m = back.multipliers();
        assert!(
            (m[0] - wb[0]).abs() < 0.02 && (m[2] - wb[2]).abs() < 0.02,
            "{m:?} vs {wb:?}"
        );
    }
}
