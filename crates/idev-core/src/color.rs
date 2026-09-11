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

/// Camera RGB (white-balanced so that a neutral is `(1, 1, 1)`) → linear
/// BT.2020, from a DNG-style XYZ→camera matrix.
///
/// The dcraw construction: express the camera response to BT.2020
/// primaries, normalise each row so BT.2020 white maps to camera
/// `(1, 1, 1)`, and invert. White balance is then applied to the raw
/// values before this matrix, and neutrals stay neutral by construction.
/// No chromatic adaptation between illuminants yet; the matrix is used
/// as given (D65 preferred by the caller).
pub fn bt2020_from_camera(xyz_to_cam: &Mat3) -> Option<Mat3> {
    let mut cam_from_2020 = mul(xyz_to_cam, &XYZ_FROM_BT2020);
    for row in cam_from_2020.iter_mut() {
        let sum: f32 = row.iter().sum();
        if sum.abs() < 1e-9 {
            return None;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
    invert(&cam_from_2020)
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

    #[test]
    fn camera_white_maps_to_working_white() {
        // The a6700 D65 matrix from rawler's profile.
        let m = [
            [0.6972, -0.2408, -0.06],
            [-0.433, 1.2101, 0.2515],
            [-0.0388, 0.1277, 0.5847],
        ];
        let to_2020 = bt2020_from_camera(&m).unwrap();
        let w = apply(&to_2020, [1.0, 1.0, 1.0]);
        for v in w {
            assert!((v - 1.0).abs() < 1e-4, "white -> {w:?}");
        }
    }
}
