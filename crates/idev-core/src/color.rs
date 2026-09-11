//! 3x3 color math. Matrices are row-major `[[f32; 3]; 3]` applied as
//! `out = m * v` (row `i` dotted with `v`).

pub type Mat3 = [[f32; 3]; 3];

/// Linear BT.2020 (D65) → CIE XYZ.
pub const XYZ_FROM_BT2020: Mat3 = [
    [0.636_958, 0.144_617, 0.168_881],
    [0.262_700, 0.677_998, 0.059_302],
    [0.000_000, 0.028_073, 1.060_985],
];

/// Linear sRGB/BT.709 (D65) → CIE XYZ.
pub const XYZ_FROM_SRGB: Mat3 = [
    [0.412_391, 0.357_584, 0.180_481],
    [0.212_639, 0.715_169, 0.072_192],
    [0.019_331, 0.119_195, 0.950_532],
];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn apply(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn invert(m: &Mat3) -> Option<Mat3> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (e * i - f * h) * inv,
            (c * h - b * i) * inv,
            (b * f - c * e) * inv,
        ],
        [
            (f * g - d * i) * inv,
            (a * i - c * g) * inv,
            (c * d - a * f) * inv,
        ],
        [
            (d * h - e * g) * inv,
            (b * g - a * h) * inv,
            (a * e - b * d) * inv,
        ],
    ])
}

/// A display's colour gamut, by its primaries (all D65).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gamut {
    Srgb,
    DisplayP3,
    Bt2020,
    AdobeRgb,
}

/// CIE 1931 xy chromaticities of (red, green, blue, white).
type Chromaticities = [[f32; 2]; 4];

const D65_XY: [f32; 2] = [0.3127, 0.3290];

impl Gamut {
    pub fn label(self) -> &'static str {
        match self {
            Gamut::Srgb => "sRGB",
            Gamut::DisplayP3 => "P3",
            Gamut::Bt2020 => "BT.2020",
            Gamut::AdobeRgb => "Adobe RGB",
        }
    }

    fn chromaticities(self) -> Chromaticities {
        match self {
            Gamut::Srgb => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], D65_XY],
            Gamut::DisplayP3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060], D65_XY],
            Gamut::Bt2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046], D65_XY],
            Gamut::AdobeRgb => [[0.64, 0.33], [0.21, 0.71], [0.15, 0.06], D65_XY],
        }
    }

    /// Linear RGB in this gamut → CIE XYZ.
    pub fn xyz_from(self) -> Mat3 {
        xyz_from_chromaticities(self.chromaticities())
    }

    /// Linear BT.2020 → linear RGB in this gamut (negative or above-one
    /// components mean the colour is outside it).
    pub fn from_bt2020(self) -> Mat3 {
        let from_xyz = invert(&self.xyz_from()).expect("gamut matrix is invertible");
        mul(&from_xyz, &XYZ_FROM_BT2020)
    }
}

/// RGB → XYZ for primaries and white given as xy, white scaled to Y = 1.
fn xyz_from_chromaticities(c: Chromaticities) -> Mat3 {
    let xyz = |[x, y]: [f32; 2]| [x / y, 1.0, (1.0 - x - y) / y];
    let [r, g, b] = [xyz(c[0]), xyz(c[1]), xyz(c[2])];
    let w = xyz(c[3]);
    // Columns are the primaries' XYZ, scaled so their sum is the white.
    let p: Mat3 = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
    let scale = apply(&invert(&p).expect("primaries are independent"), w);
    [
        [r[0] * scale[0], g[0] * scale[1], b[0] * scale[2]],
        [r[1] * scale[0], g[1] * scale[1], b[1] * scale[2]],
        [r[2] * scale[0], g[2] * scale[1], b[2] * scale[2]],
    ]
}

/// Linear BT.2020 → linear sRGB primaries (values may go negative or
/// exceed 1; that is the scRGB convention).
pub fn srgb_from_bt2020() -> Mat3 {
    let srgb_from_xyz = invert(&XYZ_FROM_SRGB).expect("sRGB matrix is invertible");
    mul(&srgb_from_xyz, &XYZ_FROM_BT2020)
}

/// sRGB piecewise OETF on a linear value in `[0, 1]`.
pub fn srgb_oetf(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chromaticities_reproduce_the_constants() {
        for (gamut, want) in [
            (Gamut::Srgb, XYZ_FROM_SRGB),
            (Gamut::Bt2020, XYZ_FROM_BT2020),
        ] {
            let got = gamut.xyz_from();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (got[i][j] - want[i][j]).abs() < 2e-4,
                        "{gamut:?}[{i}][{j}] = {} vs {}",
                        got[i][j],
                        want[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_roundtrips() {
        let m = XYZ_FROM_BT2020;
        let inv = invert(&m).unwrap();
        let id = mul(&m, &inv);
        for (i, row) in id.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-5, "id[{i}][{j}] = {v}");
            }
        }
    }
}
