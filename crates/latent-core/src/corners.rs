//! Checkerboard corners as alignment ground truth. A burst of a
//! calibration board gives, in every frame, a few hundred X-corners
//! that can be located to a few hundredths of a pixel; matched through
//! the global shift they score any alignment model directly, which the
//! spectral proxies on night skylines could not (DESIGN.md, step 3
//! correction).
//!
//! Detection runs on the half-resolution quad plane: Gaussian blur,
//! the saddle response `Ixy² − Ixx·Iyy` (positive where the two
//! principal curvatures have opposite sign, i.e. at an X-corner), local
//! maxima above a fraction of the strongest, then a sub-pixel
//! refinement by gradient orthogonality: at an X-corner every
//! neighbouring gradient is perpendicular to the vector from the
//! corner, so the corner is the least-squares point `c` with
//! `Σ gᵢgᵢᵀ (qᵢ − c) = 0`. (A quadratic saddle fit over a window
//! centred on the response peak was tried first; its saddle is pulled
//! toward the window centre by up to 0.15 px.)

use std::cmp::Ordering;

use crate::burst::{self, Gray, Shift};
use crate::sensor::SensorFrame;

/// Blur before the response, in quads. Markers inside a ChArUco
/// board's squares are a few quads wide and wash out at this scale;
/// the squares themselves do not.
const BLUR_SIGMA: f32 = 2.0;
/// Non-maximum suppression half-width, quads.
const NMS: usize = 4;
/// Radius of the refinement window, quads. It must exceed the blur
/// width so the four edges, not just the saddle core, constrain the
/// corner: at 4 quads the estimate stopped 40 % of the way from the
/// response peak to the corner; at 6 it lands within 0.13 px.
const FIT: usize = 6;
/// Refinement passes; each recentres the window on the last estimate.
const REFINE_PASSES: usize = 3;
/// Corners weaker than this fraction of the strongest are dropped.
const RELATIVE_THRESHOLD: f32 = 0.05;
/// A saddle further than this from the response peak is a misfit.
const MAX_REFINE: f32 = 1.5;

/// One corner in sensor pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    pub x: f32,
    pub y: f32,
    pub response: f32,
}

/// Separable Gaussian blur, edges clamped.
fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f32> = (0..=2 * r)
        .map(|i| {
            let d = i as f32 - r as f32;
            (-0.5 * d * d / (sigma * sigma)).exp()
        })
        .collect();
    let norm = 1.0 / kernel.iter().sum::<f32>();
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let mut s = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - r as isize).clamp(0, w as isize - 1) as usize;
                s += kv * row[sx];
            }
            tmp[y * w + x] = s * norm;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let sy = (y as isize + k as isize - r as isize).clamp(0, h as isize - 1) as usize;
                s += kv * tmp[sy * w + x];
            }
            out[y * w + x] = s * norm;
        }
    }
    out
}

/// Gradient-orthogonality refinement of a corner near `(cx, cy)`, as
/// an offset in quads; `None` when the system is degenerate or the
/// estimate leaves the window.
fn refine(plane: &[f32], w: usize, h: usize, cx: usize, cy: usize) -> Option<(f32, f32)> {
    let mut c = (cx as f64, cy as f64);
    for _ in 0..REFINE_PASSES {
        // Σ w G (q − c) = 0 with G = g gᵀ, solved for c as
        // c = (Σ w G)⁻¹ Σ w G q, over a window centred on the estimate.
        let (ox, oy) = (c.0.round() as isize, c.1.round() as isize);
        let mut m = [[0f64; 2]; 2];
        let mut b = [0f64; 2];
        for dy in -(FIT as isize)..=FIT as isize {
            for dx in -(FIT as isize)..=FIT as isize {
                let (x, y) = (ox + dx, oy + dy);
                if x < 1 || y < 1 || x >= w as isize - 1 || y >= h as isize - 1 {
                    return None;
                }
                let (x, y) = (x as usize, y as usize);
                let gx = (plane[y * w + x + 1] - plane[y * w + x - 1]) as f64 * 0.5;
                let gy = (plane[(y + 1) * w + x] - plane[(y - 1) * w + x]) as f64 * 0.5;
                // A disc about the continuous estimate, not the integer
                // quad the samples are gathered around: an asymmetric
                // window biases the solution toward its own centre.
                let (qx, qy) = (x as f64, y as f64);
                let r2 = ((qx - c.0).powi(2) + (qy - c.1).powi(2)) / (FIT * FIT) as f64;
                if r2 > 1.0 {
                    continue;
                }
                let wgt = 1.0;
                m[0][0] += wgt * gx * gx;
                m[0][1] += wgt * gx * gy;
                m[1][1] += wgt * gy * gy;
                b[0] += wgt * (gx * gx * qx + gx * gy * qy);
                b[1] += wgt * (gx * gy * qx + gy * gy * qy);
            }
        }
        m[1][0] = m[0][1];
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < 1e-12 {
            return None;
        }
        c = (
            (m[1][1] * b[0] - m[0][1] * b[1]) / det,
            (-m[1][0] * b[0] + m[0][0] * b[1]) / det,
        );
    }
    let (dx, dy) = ((c.0 - cx as f64) as f32, (c.1 - cy as f64) as f32);
    (dx.abs() <= MAX_REFINE && dy.abs() <= MAX_REFINE).then_some((dx, dy))
}

/// Corners over the crop, strongest first.
pub fn detect(frame: &SensorFrame) -> Vec<Corner> {
    let g: Gray = burst::gray(frame);
    let (w, h) = (g.width, g.height);
    let blurred = gaussian_blur(&g.data, w, h, BLUR_SIGMA);
    let at = |x: usize, y: usize| blurred[y * w + x];
    let mut response = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let ixx = at(x + 1, y) - 2.0 * at(x, y) + at(x - 1, y);
            let iyy = at(x, y + 1) - 2.0 * at(x, y) + at(x, y - 1);
            let ixy =
                (at(x + 1, y + 1) - at(x + 1, y - 1) - at(x - 1, y + 1) + at(x - 1, y - 1)) * 0.25;
            response[y * w + x] = ixy * ixy - ixx * iyy;
        }
    }
    let peak = response.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let threshold = RELATIVE_THRESHOLD * peak;
    let margin = (NMS + FIT + 1).max(1);
    let mut out = Vec::new();
    for y in margin..h - margin {
        for x in margin..w - margin {
            let r = response[y * w + x];
            if r < threshold {
                continue;
            }
            let mut is_max = true;
            'nms: for ny in y - NMS..=y + NMS {
                for nx in x - NMS..=x + NMS {
                    if response[ny * w + nx] > r {
                        is_max = false;
                        break 'nms;
                    }
                }
            }
            if !is_max {
                continue;
            }
            let Some((dx, dy)) = refine(&blurred, w, h, x, y) else {
                continue;
            };
            // Quad (i, j) is centred on sensor (crop + 2i + 0.5).
            out.push(Corner {
                x: frame.crop.x as f32 + 2.0 * (x as f32 + dx) + 0.5,
                y: frame.crop.y as f32 + 2.0 * (y as f32 + dy) + 0.5,
                response: r,
            });
        }
    }
    out.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(Ordering::Equal)
    });
    out
}

/// A reference corner and where it was found in another frame, both
/// in sensor pixels.
#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub p: [f32; 2],
    pub q: [f32; 2],
}

/// Mutual nearest neighbours after displacing the reference corners by
/// the global shift, within `tolerance` sensor pixels.
pub fn matches(reference: &[Corner], frame: &[Corner], shift: Shift, tolerance: f32) -> Vec<Pair> {
    let nearest = |x: f32, y: f32, set: &[Corner]| -> Option<(usize, f32)> {
        set.iter()
            .enumerate()
            .map(|(i, c)| (i, (c.x - x).hypot(c.y - y)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
    };
    let mut out = Vec::new();
    for (i, r) in reference.iter().enumerate() {
        let (px, py) = (r.x + shift.dx, r.y + shift.dy);
        let Some((j, d)) = nearest(px, py, frame) else {
            continue;
        };
        if d > tolerance {
            continue;
        }
        // Mutual: the frame corner's nearest displaced reference is r.
        let back = nearest(frame[j].x - shift.dx, frame[j].y - shift.dy, reference);
        if back.map(|(k, _)| k) != Some(i) {
            continue;
        }
        out.push(Pair {
            p: [r.x, r.y],
            q: [frame[j].x, frame[j].y],
        });
    }
    out
}

/// Gaussian elimination with partial pivoting for a small dense system.
fn solve(mut m: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = m[row][col] / m[col][col];
            let pivot_row = m[col].clone();
            for (mk, pk) in m[row].iter_mut().zip(pivot_row).skip(col) {
                *mk -= f * pk;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0f64; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| m[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / m[i][i];
    }
    Some(x)
}

/// A projective map fitted to the pairs by linear least squares
/// (`h33 = 1`, coordinates normalised to the frame scale). For a planar
/// subject this is the best any global model can do, so its residual
/// is the floor the corner locations set.
pub fn fit_homography(pairs: &[Pair]) -> Option<Homography> {
    if pairs.len() < 8 {
        return None;
    }
    let scale = pairs
        .iter()
        .map(|m| m.p[0].abs().max(m.p[1].abs()))
        .fold(1.0f32, f32::max) as f64;
    let mut ata = vec![vec![0f64; 8]; 8];
    let mut atb = vec![0f64; 8];
    for m in pairs {
        let (x, y) = (m.p[0] as f64 / scale, m.p[1] as f64 / scale);
        let (u, v) = (m.q[0] as f64 / scale, m.q[1] as f64 / scale);
        // u (h31 x + h32 y + 1) = h11 x + h12 y + h13, likewise v.
        let rows = [
            ([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u),
            ([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v),
        ];
        for (r, b) in rows {
            for i in 0..8 {
                atb[i] += r[i] * b;
                for k in 0..8 {
                    ata[i][k] += r[i] * r[k];
                }
            }
        }
    }
    let h = solve(ata, atb)?;
    Some(Homography {
        h: [
            [h[0], h[1], h[2] * scale],
            [h[3], h[4], h[5] * scale],
            [h[6] / scale, h[7] / scale, 1.0],
        ],
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    pub h: [[f64; 3]; 3],
}

impl Homography {
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let (x, y) = (x as f64, y as f64);
        let h = &self.h;
        let w = h[2][0] * x + h[2][1] * y + h[2][2];
        (
            ((h[0][0] * x + h[0][1] * y + h[0][2]) / w) as f32,
            ((h[1][0] * x + h[1][1] * y + h[1][2]) / w) as f32,
        )
    }
}

/// Residual rms after removing each grid cell's mean residual: what a
/// per-tile translation on top of `predict` could reach.
pub fn local_bound(
    pairs: &[Pair],
    predict: impl Fn(f32, f32) -> (f32, f32),
    frame: &SensorFrame,
    cols: usize,
    rows: usize,
) -> f32 {
    let grid = residual_grid(pairs, &predict, frame, cols, rows);
    let c = frame.crop;
    let mut sum = 0f64;
    let mut n = 0usize;
    for m in pairs {
        let (x, y) = predict(m.p[0], m.p[1]);
        let col = (((m.p[0] - c.x as f32) / c.width as f32) * cols as f32).floor();
        let row = (((m.p[1] - c.y as f32) / c.height as f32) * rows as f32).floor();
        if col < 0.0 || row < 0.0 || col >= cols as f32 || row >= rows as f32 {
            continue;
        }
        let Some((mx, my, _)) = grid[row as usize][col as usize] else {
            continue;
        };
        sum += ((x - m.q[0] - mx).powi(2) + (y - m.q[1] - my).powi(2)) as f64;
        n += 1;
    }
    if n == 0 {
        f32::NAN
    } else {
        (sum / n as f64).sqrt() as f32
    }
}

/// Residual statistics of a model over matched pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub n: usize,
    pub rms: f32,
    pub median: f32,
    pub max: f32,
}

pub fn score(pairs: &[Pair], predict: impl Fn(f32, f32) -> (f32, f32)) -> Score {
    let mut r: Vec<f32> = pairs
        .iter()
        .map(|m| {
            let (x, y) = predict(m.p[0], m.p[1]);
            (x - m.q[0]).hypot(y - m.q[1])
        })
        .collect();
    if r.is_empty() {
        return Score {
            n: 0,
            rms: f32::NAN,
            median: f32::NAN,
            max: f32::NAN,
        };
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Score {
        n: r.len(),
        rms: (r.iter().map(|v| v * v).sum::<f32>() / r.len() as f32).sqrt(),
        median: r[r.len() / 2],
        max: r[r.len() - 1],
    }
}

/// Mean residual vector of a model per cell of a `cols` × `rows` grid
/// over the crop, with the pair count; `None` for empty cells.
pub fn residual_grid(
    pairs: &[Pair],
    predict: impl Fn(f32, f32) -> (f32, f32),
    frame: &SensorFrame,
    cols: usize,
    rows: usize,
) -> Vec<Vec<Option<(f32, f32, usize)>>> {
    let c = frame.crop;
    let mut sums = vec![vec![(0f32, 0f32, 0usize); cols]; rows];
    for m in pairs {
        let (x, y) = predict(m.p[0], m.p[1]);
        let col = (((m.p[0] - c.x as f32) / c.width as f32) * cols as f32).floor();
        let row = (((m.p[1] - c.y as f32) / c.height as f32) * rows as f32).floor();
        if col < 0.0 || row < 0.0 || col >= cols as f32 || row >= rows as f32 {
            continue;
        }
        let cell = &mut sums[row as usize][col as usize];
        cell.0 += x - m.q[0];
        cell.1 += y - m.q[1];
        cell.2 += 1;
    }
    sums.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(sx, sy, n)| (n > 0).then(|| (sx / n as f32, sy / n as f32, n)))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::{CfaColor, ExifSummary, Rect};
    use rawler::Orientation;

    /// A checkerboard of `square` px squares, corners at
    /// `(origin + k·square)`, rendered with 4×4 supersampling so
    /// fractional origins are meaningful.
    fn board(width: u32, height: u32, square: f32, origin: (f32, f32)) -> SensorFrame {
        let mut data = vec![0u16; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let mut s = 0.0;
                for sy in 0..4 {
                    for sx in 0..4 {
                        let px = x as f32 + (sx as f32 + 0.5) / 4.0 - 0.5;
                        let py = y as f32 + (sy as f32 + 0.5) / 4.0 - 0.5;
                        let i = ((px - origin.0) / square).floor() as i32;
                        let j = ((py - origin.1) / square).floor() as i32;
                        s += ((i + j) & 1) as f32;
                    }
                }
                data[(y * width + x) as usize] = (512.0 + 4000.0 * s / 16.0) as u16;
            }
        }
        SensorFrame {
            width,
            height,
            cfa: [
                CfaColor::Red,
                CfaColor::Green,
                CfaColor::Green,
                CfaColor::Blue,
            ],
            black: [512.0; 4],
            white: [16383.0; 4],
            crop: Rect {
                x: 4,
                y: 4,
                width: width - 8,
                height: height - 8,
            },
            wb_as_shot: [2.0, 1.0, 1.5],
            matrices: Vec::new(),
            orientation: Orientation::Normal,
            exif: ExifSummary::default(),
            metadata: Default::default(),
            lens: None,
            data,
        }
    }

    #[test]
    fn detects_board_corners_to_a_tenth_of_a_pixel() {
        // A non-integer square puts each corner at a different sub-quad
        // phase, so a centring bias cannot hide.
        let square = 45.6;
        let origin = (17.3, 9.7);
        let f = board(640, 480, square, origin);
        let found = detect(&f);
        // Interior corners (away from the borders the detector skips).
        let mut expected = Vec::new();
        for j in 1..10 {
            for i in 1..13 {
                let (x, y) = (origin.0 + i as f32 * square, origin.1 + j as f32 * square);
                if x > 40.0 && y > 40.0 && x < 600.0 && y < 440.0 {
                    expected.push((x, y));
                }
            }
        }
        let mut errors: Vec<f32> = expected
            .iter()
            .map(|(x, y)| {
                found
                    .iter()
                    .map(|c| (c.x - x).hypot(c.y - y))
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        errors.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "corner errors: n {} median {:.3} p90 {:.3} max {:.3}",
            errors.len(),
            errors[errors.len() / 2],
            errors[errors.len() * 9 / 10],
            errors[errors.len() - 1]
        );
        // Signed bias along x and y.
        let (mut bx, mut by) = (0f32, 0f32);
        for (x, y) in &expected {
            let c = found
                .iter()
                .min_by(|a, b| {
                    (a.x - x)
                        .hypot(a.y - y)
                        .total_cmp(&(b.x - x).hypot(b.y - y))
                })
                .unwrap();
            bx += c.x - x;
            by += c.y - y;
        }
        let (bx, by) = (bx / expected.len() as f32, by / expected.len() as f32);
        eprintln!("mean bias ({bx:.3}, {by:.3})");
        assert!(
            errors[errors.len() - 1] < 0.15,
            "worst corner {} px off",
            errors[errors.len() - 1]
        );
        assert!(bx.abs() < 0.05 && by.abs() < 0.05, "bias ({bx}, {by})");
        // No spurious corners: every detection is near a lattice corner.
        for c in &found {
            let i = ((c.x - origin.0) / square).round();
            let j = ((c.y - origin.1) / square).round();
            let d = (c.x - origin.0 - i * square).hypot(c.y - origin.1 - j * square);
            assert!(d < 0.5, "spurious corner at ({}, {})", c.x, c.y);
        }
    }

    #[test]
    fn matching_and_scoring_a_known_shift() {
        let a = board(640, 480, 45.6, (17.3, 9.7));
        let b = board(640, 480, 45.6, (17.3 + 5.4, 9.7 - 3.1));
        let (ca, cb) = (detect(&a), detect(&b));
        let pairs = matches(
            &ca,
            &cb,
            Shift {
                dx: 5.0,
                dy: -3.0,
                peak: 1.0,
            },
            4.0,
        );
        assert!(pairs.len() >= 60, "{}", pairs.len());
        let exact = score(&pairs, |x, y| (x + 5.4, y - 3.1));
        // Two frames' detection errors add: about 0.17 px on this fixture.
        assert!(exact.rms < 0.2, "{exact:?}");
        let rough = score(&pairs, |x, y| (x + 5.0, y - 3.0));
        assert!(rough.rms > 0.35 && rough.rms < 0.5, "{rough:?}");
    }
}
