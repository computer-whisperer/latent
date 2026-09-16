//! Continuous-drive bursts: grouping frames by their shooting metadata,
//! global alignment on the sensor plane, and a robust merge of the
//! aligned planes into one [`SensorFrame`] that the pipeline develops
//! like any other. DESIGN.md "Burst arc" has the survey behind the
//! choices; the short form:
//!
//! - Night frames hold signal in a few percent of the raster; the rest
//!   is read noise plus a fixed pattern that correlates perfectly with
//!   itself. Alignment therefore votes only where a blurred copy of the
//!   plane rises clear of the noise.
//! - Hand-held motion between frames is a translation to within a pixel
//!   across the frame, so one global shift per frame is fitted.
//! - The merge keeps the CFA phase by rounding shifts to even pixels and
//!   down-weights samples that disagree with the per-pixel median by
//!   more than the noise model allows.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::sensor::{self, ExifSummary, SensorFrame};
use crate::source::CameraRgb;

/// Longest gap between consecutive frames of one burst, on top of the
/// exposure time. The a6700 spaces frames by 160–800 ms.
const MAX_GAP_SECS: f64 = 2.0;
/// Side of the box blur that decides where the plane has signal.
const MASK_BLUR: usize = 7;
/// Signal threshold for the mask, in quad-noise sigmas.
const MASK_SIGMAS: f32 = 6.0;
/// Least fraction of quads that must pass the mask for a usable alignment.
const MIN_COVERAGE: f32 = 0.001;
/// Robustness: a sample this many sigmas of the pair difference from
/// the median has weight e^-½.
const ROBUST_SIGMAS: f32 = 3.0;
/// Frames whose sharpness falls below this fraction of the burst's
/// sharpest are left out of the merge: their smeared lights leak past
/// the robustness gate (on DSC01146–1156 two motion-blurred frames at
/// 0.10 and 0.32 cost the merged lights 21 % of their Laplacian energy
/// and bought 4 % in sky noise).
pub const DROP_SHARPNESS: f32 = 0.5;

// ---------------------------------------------------------------------
// Grouping

/// Split frames into bursts. Frames are ordered by shooting time; a
/// burst continues while the sequence number increments, the gap stays
/// under [`MAX_GAP_SECS`] plus the exposure, and shutter, ISO, aperture
/// and focal length are unchanged. Every frame lands in exactly one
/// group; single shots and frames without metadata are singletons.
pub fn group(summaries: &[Option<&ExifSummary>]) -> Vec<Vec<usize>> {
    let time = |i: usize| summaries[i].and_then(|e| e.timestamp_secs());
    let mut order: Vec<usize> = (0..summaries.len()).collect();
    order.sort_by(|&a, &b| {
        match (time(a), time(b)) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
        .then(a.cmp(&b))
    });
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let extends = match (groups.last(), summaries[i]) {
            (Some(g), Some(cur)) => g
                .last()
                .and_then(|&p| summaries[p])
                .is_some_and(|prev| continues(prev, cur)),
            _ => false,
        };
        match groups.last_mut() {
            Some(g) if extends => g.push(i),
            _ => groups.push(vec![i]),
        }
    }
    groups
}

fn continues(prev: &ExifSummary, cur: &ExifSummary) -> bool {
    let (Some(ps), Some(cs)) = (prev.sequence, cur.sequence) else {
        return false;
    };
    let (Some(pt), Some(ct)) = (prev.timestamp_secs(), cur.timestamp_secs()) else {
        return false;
    };
    let exposure = prev
        .exposure_time
        .map(|(n, d)| n as f64 / d.max(1) as f64)
        .unwrap_or(0.0);
    cs == ps + 1
        && ct - pt <= MAX_GAP_SECS + exposure
        && prev.exposure_time == cur.exposure_time
        && prev.iso == cur.iso
        && prev.f_number == cur.f_number
        && prev.focal_length_mm == cur.focal_length_mm
}

/// The burst `path` leads, itself first, found by walking its
/// neighbours by name in its folder while the burst rule holds (a
/// summary read per neighbour, plus one past each end). `summary` is
/// the file's own. A file that is not a burst's first frame, or not in
/// one, is alone.
pub fn members_of(path: &Path, summary: &ExifSummary) -> Result<Vec<PathBuf>> {
    if summary.sequence.is_none() {
        return Ok(vec![path.to_path_buf()]);
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut siblings: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("list {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| crate::source::is_raw(p))
        .collect();
    siblings.sort();
    let Some(at) = siblings.iter().position(|p| p == path) else {
        return Ok(vec![path.to_path_buf()]);
    };
    // Led by an earlier frame: not a leader.
    if at > 0 && continues(&sensor::summary(&siblings[at - 1])?, summary) {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut members = vec![path.to_path_buf()];
    let mut prev = summary.clone();
    for next in &siblings[at + 1..] {
        let cur = sensor::summary(next)?;
        if !continues(&prev, &cur) {
            break;
        }
        members.push(next.clone());
        prev = cur;
    }
    Ok(members)
}

// ---------------------------------------------------------------------
// Alignment

/// Where a frame's content sits relative to the reference, in sensor
/// pixels: `frame(x + dx, y + dy) ≈ reference(x, y)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shift {
    pub dx: f32,
    pub dy: f32,
    /// Height of the phase-correlation peak, 0..1; below ~0.02 the
    /// match is doubtful.
    pub peak: f32,
}

impl Shift {
    pub const ZERO: Shift = Shift {
        dx: 0.0,
        dy: 0.0,
        peak: 1.0,
    };

    /// Rounded to even sensor pixels, which keeps the CFA phase.
    pub fn even(&self) -> (i32, i32) {
        (
            (self.dx * 0.5).round() as i32 * 2,
            (self.dy * 0.5).round() as i32 * 2,
        )
    }
}

/// Half-resolution plane over the crop: each value is the mean of one
/// 2x2 CFA quad, black subtracted, linear DN.
pub(crate) struct Gray {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) data: Vec<f32>,
}

pub(crate) fn gray(frame: &SensorFrame) -> Gray {
    let c = frame.crop;
    let (w, h) = ((c.width / 2) as usize, (c.height / 2) as usize);
    let stride = frame.width as usize;
    let black = |x: u32, y: u32| frame.black[((y & 1) * 2 + (x & 1)) as usize];
    let mut data = vec![0f32; w * h];
    for qy in 0..h {
        let y = c.y + 2 * qy as u32;
        let row0 = y as usize * stride;
        let row1 = row0 + stride;
        for (qx, out) in data[qy * w..(qy + 1) * w].iter_mut().enumerate() {
            let x = c.x + 2 * qx as u32;
            let xi = x as usize;
            let sum = frame.data[row0 + xi] as f32 - black(x, y) + frame.data[row0 + xi + 1] as f32
                - black(x + 1, y)
                + frame.data[row1 + xi] as f32
                - black(x, y + 1)
                + frame.data[row1 + xi + 1] as f32
                - black(x + 1, y + 1);
            *out = sum * 0.25;
        }
    }
    Gray {
        width: w,
        height: h,
        data,
    }
}

/// How much fine detail a frame holds: mean squared Laplacian of the
/// quad plane over the quads that carry signal (the alignment mask).
/// Frames of one burst share exposure, so their noise contributes the
/// same and the ratio between frames ranks their sharpness: defocus
/// or motion in a frame lowers it. Only comparable within a burst.
pub fn sharpness(frame: &SensorFrame) -> f32 {
    let g = gray(frame);
    let quad_sigma = dark_sigma(frame) * 0.5;
    let blurred = box_blur(&g.data, g.width, g.height);
    let threshold = (MASK_SIGMAS * quad_sigma).max(2.0);
    let (w, h) = (g.width, g.height);
    let mut sum = 0f64;
    let mut n = 0u64;
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = y * w + x;
            if blurred[i] <= threshold {
                continue;
            }
            let d = &g.data;
            let l = 4.0 * d[i] - d[i - 1] - d[i + 1] - d[i - w] - d[i + w];
            sum += (l * l) as f64;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { (sum / n as f64) as f32 }
}

/// Read-noise estimate in DN: RMS of the below-black tail inside the
/// crop. Sites at zero signal scatter symmetrically about black, so the
/// lower half alone gives sigma without a dark frame. Signal in the
/// scene only raises sites above black, which leaves the tail alone.
pub fn dark_sigma(frame: &SensorFrame) -> f32 {
    let c = frame.crop;
    let stride = frame.width as usize;
    let mut sum = 0f64;
    let mut n = 0u64;
    for y in c.y..c.y + c.height {
        let row = &frame.data[y as usize * stride..][c.x as usize..(c.x + c.width) as usize];
        for (i, &v) in row.iter().enumerate() {
            let x = c.x + i as u32;
            let d = v as f32 - frame.black[((y & 1) * 2 + (x & 1)) as usize];
            if d < 0.0 {
                sum += (d as f64) * (d as f64);
                n += 1;
            }
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64).sqrt() as f32
    }
}

/// Separable box blur, `MASK_BLUR` square, edges clamped.
fn box_blur(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let r = MASK_BLUR / 2;
    let norm = 1.0 / MASK_BLUR as f32;
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut tmp[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            // Clamped edges: count the clipped taps at the border value.
            let mut s = row[lo..=hi].iter().sum::<f32>();
            s += (r - (x - lo)) as f32 * row[0] + (r - (hi - x)) as f32 * row[w - 1];
            *o = s * norm;
        }
    }
    let mut out = vec![0f32; w * h];
    let mut col = vec![0f32; h];
    for x in 0..w {
        for (y, c) in col.iter_mut().enumerate() {
            *c = tmp[y * w + x];
        }
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let mut s = col[lo..=hi].iter().sum::<f32>();
            s += (r - (y - lo)) as f32 * col[0] + (r - (hi - y)) as f32 * col[h - 1];
            out[y * w + x] = s * norm;
        }
    }
    out
}

/// The plane the correlator sees: log of the signal where the blurred
/// plane clears the noise, zero elsewhere, mean removed over the mask.
/// Returns the plane and the fraction of quads that passed.
fn masked_log(g: &Gray, quad_sigma: f32) -> (Vec<f32>, f32) {
    let blurred = box_blur(&g.data, g.width, g.height);
    let threshold = (MASK_SIGMAS * quad_sigma).max(2.0);
    let mut out = vec![0f32; g.data.len()];
    let mut count = 0usize;
    let mut sum = 0f64;
    for i in 0..g.data.len() {
        if blurred[i] > threshold {
            let v = g.data[i].max(0.0).ln_1p();
            out[i] = v;
            sum += v as f64;
            count += 1;
        }
    }
    if count > 0 {
        let mean = (sum / count as f64) as f32;
        for (o, &b) in out.iter_mut().zip(&blurred) {
            if b > threshold {
                *o -= mean;
            }
        }
    }
    (out, count as f32 / g.data.len() as f32)
}

/// Run `f` over row bands of `data` on all cores; `f` gets the first
/// row index of its band and the band.
fn par_rows<T: Send>(data: &mut [T], width: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if data.is_empty() {
        return;
    }
    let height = data.len() / width;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(height.max(1));
    let band = height.div_ceil(threads);
    std::thread::scope(|s| {
        for (i, chunk) in data.chunks_mut(band * width).enumerate() {
            let f = &f;
            s.spawn(move || f(i * band, chunk));
        }
    });
}

/// In-place 2D FFT: rows then columns.
fn fft2(data: &mut [Complex<f32>], w: usize, h: usize, inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let row_fft: Arc<dyn Fft<f32>> = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    let col_fft: Arc<dyn Fft<f32>> = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    par_rows(data, w, |_, band| {
        let mut scratch = vec![Complex::default(); row_fft.get_inplace_scratch_len()];
        for row in band.chunks_mut(w) {
            row_fft.process_with_scratch(row, &mut scratch);
        }
    });
    // Columns: transpose, transform rows, transpose back.
    let mut t = vec![Complex::<f32>::default(); w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = data[y * w + x];
        }
    }
    par_rows(&mut t, h, |_, band| {
        let mut scratch = vec![Complex::default(); col_fft.get_inplace_scratch_len()];
        for col in band.chunks_mut(h) {
            col_fft.process_with_scratch(col, &mut scratch);
        }
    });
    for x in 0..w {
        for y in 0..h {
            data[y * w + x] = t[x * h + y];
        }
    }
}

/// Phase correlation of two planes of the same size. Returns the
/// displacement `(dx, dy)` such that `b(x + dx, y + dy) ≈ a(x, y)`,
/// with a parabolic sub-sample refinement, and the peak height.
fn phase_correlate(a: &[f32], b: &[f32], w: usize, h: usize) -> (f32, f32, f32) {
    let to_c =
        |p: &[f32]| -> Vec<Complex<f32>> { p.iter().map(|&v| Complex::new(v, 0.0)).collect() };
    let mut fa = to_c(a);
    let mut fb = to_c(b);
    fft2(&mut fa, w, h, false);
    fft2(&mut fb, w, h, false);
    for (x, y) in fa.iter_mut().zip(&fb) {
        let r = *x * y.conj();
        let n = r.norm();
        *x = if n > 1e-12 { r / n } else { Complex::default() };
    }
    fft2(&mut fa, w, h, true);
    let scale = 1.0 / (w * h) as f32;
    let r: Vec<f32> = fa.iter().map(|c| c.re * scale).collect();
    let (mut best, mut best_v) = (0usize, f32::MIN);
    for (i, &v) in r.iter().enumerate() {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    let (px, py) = (best % w, best / w);
    let at = |x: isize, y: isize| {
        r[(y.rem_euclid(h as isize) as usize) * w + x.rem_euclid(w as isize) as usize]
    };
    let refine = |m: f32, p: f32, q: f32| {
        let d = q - 2.0 * m + p;
        if d.abs() < 1e-12 {
            0.0
        } else {
            (0.5 * (q - p) / d).clamp(-0.5, 0.5)
        }
    };
    let sx = px as f32
        + refine(
            best_v,
            at(px as isize + 1, py as isize),
            at(px as isize - 1, py as isize),
        );
    let sy = py as f32
        + refine(
            best_v,
            at(px as isize, py as isize + 1),
            at(px as isize, py as isize - 1),
        );
    // The peak sits at -shift modulo the plane size when b(x) = a(x - s)
    // is b(x + shift) = a(x); wrap to the signed range and negate.
    let wrap = |v: f32, n: usize| if v > n as f32 * 0.5 { v - n as f32 } else { v };
    (-wrap(sx, w), -wrap(sy, h), best_v)
}

/// Global translation of `frame` relative to `reference`.
pub fn align(reference: &SensorFrame, frame: &SensorFrame) -> Result<Shift> {
    ensure!(
        reference.width == frame.width
            && reference.height == frame.height
            && reference.crop == frame.crop,
        "frames differ in size or crop"
    );
    let ga = gray(reference);
    let gb = gray(frame);
    // Quad means average four sites, so their noise is half a site's.
    let quad_sigma = dark_sigma(reference).max(dark_sigma(frame)) * 0.5;
    let (ma, cov_a) = masked_log(&ga, quad_sigma);
    let (mb, cov_b) = masked_log(&gb, quad_sigma);
    ensure!(
        cov_a >= MIN_COVERAGE && cov_b >= MIN_COVERAGE,
        "too little signal to align: {:.2}% / {:.2}% of quads above the noise",
        cov_a * 100.0,
        cov_b * 100.0
    );
    let (dx, dy, peak) = phase_correlate(&ma, &mb, ga.width, ga.height);
    Ok(Shift {
        dx: 2.0 * dx,
        dy: 2.0 * dy,
        peak,
    })
}

// ---------------------------------------------------------------------
// Sub-pixel alignment: a similarity per frame

/// Reference sensor coordinates to frame sensor coordinates:
/// `p' = a·p + t`. A similarity (rotation and uniform scale) in
/// practice; stored as a general 2×2 so a fuller model can slot in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub a: [[f32; 2]; 2],
    pub t: [f32; 2],
    /// Tiles the coarse fit used.
    pub tiles: u32,
    /// RMS residual of the refinement on the log planes (log DN, over
    /// the reference's signal mask).
    pub residual: f32,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        a: [[1.0, 0.0], [0.0, 1.0]],
        t: [0.0, 0.0],
        tiles: 0,
        residual: 0.0,
    };

    pub fn translation(dx: f32, dy: f32) -> Transform {
        Transform {
            t: [dx, dy],
            ..Transform::IDENTITY
        }
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a[0][0] * x + self.a[0][1] * y + self.t[0],
            self.a[1][0] * x + self.a[1][1] * y + self.t[1],
        )
    }

    /// Rotation in degrees and scale, for reports.
    pub fn rotation_deg(&self) -> f32 {
        self.a[1][0].atan2(self.a[0][0]).to_degrees()
    }

    pub fn scale(&self) -> f32 {
        self.a[0][0].hypot(self.a[1][0])
    }
}

/// Tile side for the local correlations, in quads (half-resolution
/// pixels).
const TILE: usize = 256;
/// Least fraction of a tile's quads above the noise for it to vote.
const TILE_COVERAGE: f32 = 0.02;

/// One tile's measurement: reference point and where it landed in the
/// frame, both in sensor pixels, with the correlation peak as weight.
struct TileMatch {
    p: [f32; 2],
    q: [f32; 2],
    w: f32,
}

/// Fit a similarity from `frame` to `reference` around a global shift
/// estimate (`global`, from [`align`]).
pub fn align_similarity(
    reference: &SensorFrame,
    frame: &SensorFrame,
    global: Shift,
) -> Result<Transform> {
    Ok(align_similarity_stages(reference, frame, global)?.1)
}

/// The coarse tile fit and its refinement, for scoring them separately.
pub fn align_similarity_stages(
    reference: &SensorFrame,
    frame: &SensorFrame,
    global: Shift,
) -> Result<(Transform, Transform)> {
    ensure!(
        reference.width == frame.width
            && reference.height == frame.height
            && reference.crop == frame.crop,
        "frames differ in size or crop"
    );
    let ga = gray(reference);
    let gb = gray(frame);
    let quad_sigma = dark_sigma(reference).max(dark_sigma(frame)) * 0.5;
    let (ma, _) = masked_log(&ga, quad_sigma);
    let (mb, _) = masked_log(&gb, quad_sigma);
    let (gx, gy) = global.even();
    let (qx, qy) = ((gx / 2) as isize, (gy / 2) as isize);
    let crop = reference.crop;
    let mut matches = Vec::new();
    let (tw, th) = (ga.width / TILE, ga.height / TILE);
    // Tiles run in parallel; each is a small FFT.
    let cells: Vec<(usize, usize)> = (0..th)
        .flat_map(|ty| (0..tw).map(move |tx| (tx, ty)))
        .collect();
    let mut results: Vec<Option<TileMatch>> = (0..cells.len()).map(|_| None).collect();
    par_rows(&mut results, 1, |i0, band| {
        for (k, slot) in band.iter_mut().enumerate() {
            let (tx, ty) = cells[i0 + k];
            // The frame's tile sits at the reference tile displaced by
            // the global shift; tiles that would leave either plane are
            // moved inward so both copies fit (they then overlap a
            // neighbour, which is harmless).
            let fit = |o: usize, q: isize, n: usize| -> Option<usize> {
                let lo = (-q).max(0);
                let hi = (n as isize - TILE as isize).min(n as isize - TILE as isize - q);
                (lo <= hi).then(|| (o as isize).clamp(lo, hi) as usize)
            };
            let (Some(x0), Some(y0)) =
                (fit(tx * TILE, qx, ga.width), fit(ty * TILE, qy, ga.height))
            else {
                continue;
            };
            let (bx, by) = ((x0 as isize + qx) as usize, (y0 as isize + qy) as usize);
            let mut a = vec![0f32; TILE * TILE];
            let mut b = vec![0f32; TILE * TILE];
            let (mut ca, mut cb) = (0usize, 0usize);
            for y in 0..TILE {
                for x in 0..TILE {
                    let va = ma[(y0 + y) * ga.width + x0 + x];
                    let vb = mb[(by + y) * gb.width + bx + x];
                    a[y * TILE + x] = va;
                    b[y * TILE + x] = vb;
                    ca += (va != 0.0) as usize;
                    cb += (vb != 0.0) as usize;
                }
            }
            let min = (TILE_COVERAGE * (TILE * TILE) as f32) as usize;
            if ca < min || cb < min {
                continue;
            }
            let (dx, dy, peak) = phase_correlate(&a, &b, TILE, TILE);
            if peak <= 0.0 {
                continue;
            }
            // Tile centre in sensor pixels; the quad at (qx, qy) covers
            // sensor pixels crop + 2q .. crop + 2q + 1.
            let cx = crop.x as f32 + 2.0 * (x0 as f32 + TILE as f32 * 0.5);
            let cy = crop.y as f32 + 2.0 * (y0 as f32 + TILE as f32 * 0.5);
            *slot = Some(TileMatch {
                p: [cx, cy],
                q: [cx + gx as f32 + 2.0 * dx, cy + gy as f32 + 2.0 * dy],
                w: peak,
            });
        }
    });
    matches.extend(results.into_iter().flatten());
    let coarse = fit_similarity(&mut matches, global);
    tracing::debug!(
        ?global,
        tiles = matches.len(),
        coarse = %describe_transform(&coarse, 1.0),
        "similarity: coarse fit"
    );
    for m in &matches {
        tracing::trace!(p = ?m.p, q = ?m.q, w = m.w, "tile");
    }
    let refined = refine_similarity(&ga, &gb, &ma, crop, coarse);
    tracing::debug!(refined = %describe_transform(&refined, 1.0), "similarity: refined");
    Ok((coarse, refined))
}

/// Gauss-Newton refinement of a similarity on the half-resolution log
/// planes: minimise Σ m(x)·(B(W(x)) − A(x))² over the reference's
/// signal mask, warping the frame plane with bilinear sampling. Pixel
/// residuals are reweighted from the third iteration on so moving
/// content and misalignment outliers stop pulling the fit.
fn refine_similarity(
    ga: &Gray,
    gb: &Gray,
    masked_a: &[f32],
    crop: crate::sensor::Rect,
    init: Transform,
) -> Transform {
    const ITERATIONS: usize = 12;
    let (w, h) = (ga.width, ga.height);
    let la: Vec<f32> = ga.data.iter().map(|v| v.max(0.0).ln_1p()).collect();
    let lb: Vec<f32> = gb.data.iter().map(|v| v.max(0.0).ln_1p()).collect();
    // Sensor → quad coordinates: q = (p - crop - 0.5) / 2. Conjugate the
    // transform into quad space.
    let (cx, cy) = (crop.x as f32 + 0.5, crop.y as f32 + 0.5);
    let to_quad = |t: &Transform| -> ([[f32; 2]; 2], [f32; 2]) {
        // p' = A p + t  ⇒  q' = A q + (A c + t - c) / 2
        let (ax, ay) = (
            t.a[0][0] * cx + t.a[0][1] * cy,
            t.a[1][0] * cx + t.a[1][1] * cy,
        );
        (t.a, [(ax + t.t[0] - cx) * 0.5, (ay + t.t[1] - cy) * 0.5])
    };
    let from_quad = |a: [[f32; 2]; 2], tq: [f32; 2]| -> [f32; 2] {
        let (ax, ay) = (a[0][0] * cx + a[0][1] * cy, a[1][0] * cx + a[1][1] * cy);
        [2.0 * tq[0] - ax + cx, 2.0 * tq[1] - ay + cy]
    };
    let (mut a, mut tq) = to_quad(&init);
    // Centre of the mask, for conditioning.
    let (mut mx, mut my, mut mn) = (0f64, 0f64, 0f64);
    for y in 0..h {
        for x in 0..w {
            if masked_a[y * w + x] != 0.0 {
                mx += x as f64;
                my += y as f64;
                mn += 1.0;
            }
        }
    }
    if mn < 16.0 {
        return init;
    }
    let (mx, my) = ((mx / mn) as f32, (my / mn) as f32);
    // The iteration parameterises W(q) = A (q - m) + m + tq about the
    // mask centre m; convert the origin-based initial translation.
    tq = [
        tq[0] + a[0][0] * mx + a[0][1] * my - mx,
        tq[1] + a[1][0] * mx + a[1][1] * my - my,
    ];
    let mut residual_scale = f32::INFINITY;
    let mut last_rms = 0f32;
    // The best state seen once the reweighting has settled, so a step
    // that raises the residual is undone rather than compounded.
    let mut best: Option<(f32, [[f32; 2]; 2], [f32; 2])> = None;
    for it in 0..ITERATIONS {
        // Parameters: (da, db, dtx, dty) on top of the current (a, b, t)
        // with A = [[a, -b], [b, a]] about the mask centre.
        let mut hess = [[0f64; 4]; 4];
        let mut grad = [0f64; 4];
        let mut sum_r2 = 0f64;
        let mut count = 0f64;
        let mut abs_res: Vec<f32> = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let m = masked_a[y * w + x];
                if m == 0.0 {
                    continue;
                }
                let (dx, dy) = (x as f32 - mx, y as f32 - my);
                let (wx, wy) = (
                    a[0][0] * dx + a[0][1] * dy + mx + tq[0],
                    a[1][0] * dx + a[1][1] * dy + my + tq[1],
                );
                let Some((vb, gx, gy)) = sample_plane(&lb, w, h, wx, wy) else {
                    continue;
                };
                let r = vb - la[y * w + x];
                let wgt = if residual_scale.is_finite() {
                    1.0 / (1.0 + (r / residual_scale).powi(2))
                } else {
                    1.0
                };
                abs_res.push(r.abs());
                // dW/d(a,b,tx,ty) = [(dx,dy), (-dy,dx), (1,0), (0,1)]
                let j = [
                    (gx * dx + gy * dy) as f64,
                    (-gx * dy + gy * dx) as f64,
                    gx as f64,
                    gy as f64,
                ];
                for i in 0..4 {
                    grad[i] += wgt as f64 * j[i] * r as f64;
                    for k in 0..4 {
                        hess[i][k] += wgt as f64 * j[i] * j[k];
                    }
                }
                sum_r2 += (wgt * r * r) as f64;
                count += wgt as f64;
            }
        }
        if count < 16.0 {
            break;
        }
        last_rms = (sum_r2 / count).sqrt() as f32;
        if it >= 3 {
            match best {
                Some((rms, ba, btq)) if last_rms > rms * 1.001 => {
                    a = ba;
                    tq = btq;
                    last_rms = rms;
                    break;
                }
                Some((rms, _, _)) if last_rms >= rms => {}
                _ => best = Some((last_rms, a, tq)),
            }
        }
        let Some(delta) = solve4(hess, grad) else {
            break;
        };
        tracing::trace!(
            it,
            rms = last_rms,
            n = count,
            ?delta,
            a = ?a,
            tq = ?tq,
            "refine step"
        );
        a[0][0] -= delta[0] as f32;
        a[1][1] -= delta[0] as f32;
        a[0][1] += delta[1] as f32;
        a[1][0] -= delta[1] as f32;
        tq[0] -= delta[2] as f32;
        tq[1] -= delta[3] as f32;
        if it >= 1 {
            let mut v = abs_res;
            residual_scale = (2.0 * median(&mut v)).max(1e-3);
        }
        let step = delta[2].hypot(delta[3]);
        if step < 1e-3 && delta[0].abs() < 1e-6 && delta[1].abs() < 1e-6 {
            break;
        }
    }
    // Back to origin-based.
    let tq = [
        tq[0] - (a[0][0] * mx + a[0][1] * my - mx),
        tq[1] - (a[1][0] * mx + a[1][1] * my - my),
    ];
    Transform {
        a,
        t: from_quad(a, tq),
        tiles: init.tiles,
        // RMS log-plane residual over the mask, not pixels.
        residual: last_rms,
    }
}

/// Solve a symmetric 4×4 system by Gaussian elimination with partial
/// pivoting; `None` when singular.
fn solve4(mut m: [[f64; 4]; 4], mut b: [f64; 4]) -> Option<[f64; 4]> {
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..4 {
            let f = m[row][col] / m[col][col];
            let pivot_row = m[col];
            for (mk, pk) in m[row].iter_mut().zip(pivot_row).skip(col) {
                *mk -= f * pk;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0f64; 4];
    for i in (0..4).rev() {
        let s: f64 = (i + 1..4).map(|k| m[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / m[i][i];
    }
    Some(x)
}

/// Weighted least squares for `q = [[a, -b], [b, a]]·p + t`, with three
/// rounds of outlier rejection.
fn fit_similarity(matches: &mut Vec<TileMatch>, global: Shift) -> Transform {
    let fallback = Transform::translation(global.dx, global.dy);
    for _round in 0..3 {
        let n = matches.len();
        let t = if n >= 3 {
            solve_similarity(matches)
        } else if n >= 1 {
            let wsum: f32 = matches.iter().map(|m| m.w).sum();
            let dx = matches.iter().map(|m| m.w * (m.q[0] - m.p[0])).sum::<f32>() / wsum;
            let dy = matches.iter().map(|m| m.w * (m.q[1] - m.p[1])).sum::<f32>() / wsum;
            Transform::translation(dx, dy)
        } else {
            return fallback;
        };
        let mut res: Vec<f32> = matches
            .iter()
            .map(|m| {
                let (x, y) = t.apply(m.p[0], m.p[1]);
                (x - m.q[0]).hypot(y - m.q[1])
            })
            .collect();
        let mut sorted = res.clone();
        let med = median(&mut sorted);
        let limit = (3.0 * med).max(1.0);
        let rms = (res.iter().map(|r| r * r).sum::<f32>() / n as f32).sqrt();
        let keep: Vec<bool> = res.iter().map(|&r| r <= limit).collect();
        if keep.iter().all(|&k| k) {
            return Transform {
                tiles: n as u32,
                residual: rms,
                ..t
            };
        }
        let mut i = 0;
        matches.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        res.clear();
    }
    let n = matches.len();
    if n == 0 {
        return fallback;
    }
    let t = if n >= 3 {
        solve_similarity(matches)
    } else {
        let wsum: f32 = matches.iter().map(|m| m.w).sum();
        Transform::translation(
            matches.iter().map(|m| m.w * (m.q[0] - m.p[0])).sum::<f32>() / wsum,
            matches.iter().map(|m| m.w * (m.q[1] - m.p[1])).sum::<f32>() / wsum,
        )
    };
    let rms = (matches
        .iter()
        .map(|m| {
            let (x, y) = t.apply(m.p[0], m.p[1]);
            (x - m.q[0]).powi(2) + (y - m.q[1]).powi(2)
        })
        .sum::<f32>()
        / n as f32)
        .sqrt();
    Transform {
        tiles: n as u32,
        residual: rms,
        ..t
    }
}

/// Normal equations for the four similarity parameters. Coordinates
/// are centred on the weighted mean first, which keeps the system well
/// conditioned for points thousands of pixels from the origin.
fn solve_similarity(matches: &[TileMatch]) -> Transform {
    let wsum: f64 = matches.iter().map(|m| m.w as f64).sum();
    let mean = |f: &dyn Fn(&TileMatch) -> f32| {
        matches
            .iter()
            .map(|m| m.w as f64 * f(m) as f64)
            .sum::<f64>()
            / wsum
    };
    let (px, py) = (mean(&|m| m.p[0]), mean(&|m| m.p[1]));
    let (qx, qy) = (mean(&|m| m.q[0]), mean(&|m| m.q[1]));
    // With centred coordinates the translation drops out and
    // (a, b) minimise Σ w |[[a,-b],[b,a]] p - q|²:
    // a = Σ w (px qx + py qy) / Σ w |p|², b = Σ w (px qy - py qx) / Σ w |p|².
    let (mut num_a, mut num_b, mut den) = (0f64, 0f64, 0f64);
    for m in matches {
        let w = m.w as f64;
        let (x, y) = (m.p[0] as f64 - px, m.p[1] as f64 - py);
        let (u, v) = (m.q[0] as f64 - qx, m.q[1] as f64 - qy);
        num_a += w * (x * u + y * v);
        num_b += w * (x * v - y * u);
        den += w * (x * x + y * y);
    }
    let (a, b) = if den > 1e-9 {
        (num_a / den, num_b / den)
    } else {
        (1.0, 0.0)
    };
    let tx = qx - (a * px - b * py);
    let ty = qy - (b * px + a * py);
    Transform {
        a: [[a as f32, -b as f32], [b as f32, a as f32]],
        t: [tx as f32, ty as f32],
        tiles: matches.len() as u32,
        residual: 0.0,
    }
}

// ---------------------------------------------------------------------
// Local alignment field

/// Tiles with fewer masked quads than this, in either frame, keep the
/// global fit.
const FIELD_MIN_QUADS: usize = 150;
/// A tile correction beyond this many sensor pixels is distrusted.
const FIELD_MAX_CORRECTION: f32 = 12.0;

/// A frame's alignment: the global similarity plus a translation
/// correction per tile, interpolated bilinearly between tile centres,
/// so `frame ≈ T(p) + d(p)`. Tiles without signal carry no correction,
/// which leaves the global fit there. Built by [`align_field`].
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub global: Transform,
    /// Tile-centre x coordinates (sensor px), then y.
    pub nodes_x: Vec<f32>,
    pub nodes_y: Vec<f32>,
    /// Row-major `nodes_y.len() × nodes_x.len()` corrections, sensor px.
    pub d: Vec<[f32; 2]>,
    pub valid: Vec<bool>,
}

impl Field {
    /// The global transform alone.
    pub fn global(t: Transform) -> Field {
        Field {
            global: t,
            nodes_x: Vec::new(),
            nodes_y: Vec::new(),
            d: Vec::new(),
            valid: Vec::new(),
        }
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let (gx, gy) = self.global.apply(x, y);
        if self.d.is_empty() {
            return (gx, gy);
        }
        let (ix, fx) = Self::locate(&self.nodes_x, x);
        let (iy, fy) = Self::locate(&self.nodes_y, y);
        let cols = self.nodes_x.len();
        let at = |i: usize, j: usize| self.d[j * cols + i];
        let (ix1, iy1) = ((ix + 1).min(cols - 1), (iy + 1).min(self.nodes_y.len() - 1));
        // A single node per axis cannot interpolate: constant there.
        let (fx, fy) = (
            if ix1 == ix { 0.0 } else { fx },
            if iy1 == iy { 0.0 } else { fy },
        );
        let top = [
            at(ix, iy)[0] * (1.0 - fx) + at(ix1, iy)[0] * fx,
            at(ix, iy)[1] * (1.0 - fx) + at(ix1, iy)[1] * fx,
        ];
        let bottom = [
            at(ix, iy1)[0] * (1.0 - fx) + at(ix1, iy1)[0] * fx,
            at(ix, iy1)[1] * (1.0 - fx) + at(ix1, iy1)[1] * fx,
        ];
        (
            gx + top[0] * (1.0 - fy) + bottom[0] * fy,
            gy + top[1] * (1.0 - fy) + bottom[1] * fy,
        )
    }

    /// Node interval and fraction for a coordinate. Beyond the outer
    /// nodes the fraction runs past 0..1, i.e. the field extrapolates
    /// linearly from the outer two tiles: the outer half-tiles of the
    /// frame would otherwise hold their neighbour's correction.
    fn locate(nodes: &[f32], v: f32) -> (usize, f32) {
        if nodes.len() < 2 {
            return (0, 0.0);
        }
        let i = nodes.partition_point(|&n| n <= v).clamp(1, nodes.len() - 1) - 1;
        (i, (v - nodes[i]) / (nodes[i + 1] - nodes[i]))
    }

    pub fn valid_tiles(&self) -> usize {
        self.valid.iter().filter(|&&v| v).count()
    }

    /// Mean and largest correction over valid tiles, sensor px.
    pub fn correction_stats(&self) -> (f32, f32) {
        let mut sum = 0f32;
        let mut max = 0f32;
        let mut n = 0;
        for (d, &v) in self.d.iter().zip(&self.valid) {
            if v {
                let m = d[0].hypot(d[1]);
                sum += m;
                max = max.max(m);
                n += 1;
            }
        }
        (if n > 0 { sum / n as f32 } else { 0.0 }, max)
    }
}

/// Gauss-Newton for a translation `d` (quads) over one region of the
/// reference's mask: minimise Σ (B(W(q) + d) − A(q))², `W` the global
/// map in quad space. Returns the correction and the rms before and
/// after, or `None` when the region has too little signal.
fn refine_tile_translation(
    la: &[f32],
    lb: &[f32],
    masked_a: &[f32],
    w: usize,
    h: usize,
    region: (usize, usize, usize, usize),
    warp: &dyn Fn(f32, f32) -> (f32, f32),
) -> Option<([f32; 2], f32, f32)> {
    const ITERATIONS: usize = 10;
    let (x0, y0, x1, y1) = region;
    let mut d = [0f32; 2];
    let mut residual_scale = f32::INFINITY;
    let mut first_rms = None;
    let mut best: Option<(f32, [f32; 2])> = None;
    for it in 0..ITERATIONS {
        let (mut h00, mut h01, mut h11) = (0f64, 0f64, 0f64);
        let (mut g0, mut g1) = (0f64, 0f64);
        let mut sum_r2 = 0f64;
        let mut count = 0f64;
        let mut abs_res = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                if masked_a[y * w + x] == 0.0 {
                    continue;
                }
                let (wx, wy) = warp(x as f32, y as f32);
                let Some((vb, gx, gy)) = sample_plane(lb, w, h, wx + d[0], wy + d[1]) else {
                    continue;
                };
                let r = vb - la[y * w + x];
                let wgt = if residual_scale.is_finite() {
                    1.0 / (1.0 + (r / residual_scale).powi(2))
                } else {
                    1.0
                };
                abs_res.push(r.abs());
                let (gx, gy, r, wgt) = (gx as f64, gy as f64, r as f64, wgt as f64);
                h00 += wgt * gx * gx;
                h01 += wgt * gx * gy;
                h11 += wgt * gy * gy;
                g0 += wgt * gx * r;
                g1 += wgt * gy * r;
                sum_r2 += wgt * r * r;
                count += wgt;
            }
        }
        if abs_res.len() < FIELD_MIN_QUADS {
            return None;
        }
        let rms = (sum_r2 / count).sqrt() as f32;
        if first_rms.is_none() {
            first_rms = Some(rms);
        }
        if it >= 3 {
            match best {
                Some((b, bd)) if rms > b * 1.001 => {
                    d = bd;
                    return Some((d, first_rms.unwrap_or(rms), b));
                }
                Some((b, _)) if rms >= b => {}
                _ => best = Some((rms, d)),
            }
        }
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-12 {
            break;
        }
        let dx = (h11 * g0 - h01 * g1) / det;
        let dy = (-h01 * g0 + h00 * g1) / det;
        d[0] -= dx as f32;
        d[1] -= dy as f32;
        if it >= 1 {
            let mut v = abs_res;
            residual_scale = (2.0 * median(&mut v)).max(1e-3);
        }
        if dx.hypot(dy) < 1e-3 {
            best = Some((rms, d));
            break;
        }
    }
    let (rms, d) = best.unwrap_or((first_rms.unwrap_or(0.0), d));
    Some((d, first_rms.unwrap_or(rms), rms))
}

/// The global similarity plus a per-tile translation correction,
/// tiles of [`TILE`] quads over the crop.
pub fn align_field(
    reference: &SensorFrame,
    frame: &SensorFrame,
    global: Transform,
) -> Result<Field> {
    ensure!(
        reference.width == frame.width
            && reference.height == frame.height
            && reference.crop == frame.crop,
        "frames differ in size or crop"
    );
    let ga = gray(reference);
    let gb = gray(frame);
    let (w, h) = (ga.width, ga.height);
    let quad_sigma = dark_sigma(reference).max(dark_sigma(frame)) * 0.5;
    let (ma, _) = masked_log(&ga, quad_sigma);
    let (mb, _) = masked_log(&gb, quad_sigma);
    let la: Vec<f32> = ga.data.iter().map(|v| v.max(0.0).ln_1p()).collect();
    let lb: Vec<f32> = gb.data.iter().map(|v| v.max(0.0).ln_1p()).collect();
    let crop = reference.crop;
    // Global map in quad space: q = (p - crop - 0.5) / 2.
    let (cx, cy) = (crop.x as f32 + 0.5, crop.y as f32 + 0.5);
    let warp = move |qx: f32, qy: f32| -> (f32, f32) {
        let (px, py) = global.apply(cx + 2.0 * qx, cy + 2.0 * qy);
        ((px - cx) * 0.5, (py - cy) * 0.5)
    };
    let cols = w.div_ceil(TILE);
    let rows = h.div_ceil(TILE);
    let bounds = |n: usize, i: usize| (i * TILE, ((i + 1) * TILE).min(n));
    let nodes_x: Vec<f32> = (0..cols)
        .map(|i| {
            let (a, b) = bounds(w, i);
            cx + 2.0 * ((a + b) as f32 * 0.5 - 0.5)
        })
        .collect();
    let nodes_y: Vec<f32> = (0..rows)
        .map(|j| {
            let (a, b) = bounds(h, j);
            cy + 2.0 * ((a + b) as f32 * 0.5 - 0.5)
        })
        .collect();
    let mut cells: Vec<Option<([f32; 2], f32, f32)>> = (0..cols * rows).map(|_| None).collect();
    par_rows(&mut cells, 1, |i0, band| {
        for (k, slot) in band.iter_mut().enumerate() {
            let idx = i0 + k;
            let (i, j) = (idx % cols, idx / cols);
            let (x0, x1) = bounds(w, i);
            let (y0, y1) = bounds(h, j);
            // The frame must have signal in the tile too (its tile sits
            // within a global shift of the reference's; for shifts
            // small against the tile this is the same region).
            let signal_b = (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .filter(|&(x, y)| mb[y * w + x] != 0.0)
                .count();
            if signal_b < FIELD_MIN_QUADS {
                continue;
            }
            *slot = refine_tile_translation(&la, &lb, &ma, w, h, (x0, y0, x1, y1), &warp);
        }
    });
    let mut d = vec![[0f32; 2]; cols * rows];
    let mut valid = vec![false; cols * rows];
    for (k, cell) in cells.into_iter().enumerate() {
        if let Some((dq, before, after)) = cell {
            let px = [2.0 * dq[0], 2.0 * dq[1]];
            let m = px[0].hypot(px[1]);
            // Keep a correction only where it helped and stayed sane.
            if after <= before && m <= FIELD_MAX_CORRECTION {
                d[k] = px;
                valid[k] = true;
            }
            tracing::trace!(tile = k, ?px, before, after, kept = valid[k], "field tile");
        }
    }
    Ok(Field {
        global,
        nodes_x,
        nodes_y,
        d,
        valid,
    })
}

/// Field of every frame relative to the first, in parallel, from the
/// refined similarities.
pub fn fields_all(frames: &[&SensorFrame], transforms: &[Transform]) -> Result<Vec<Field>> {
    let mut slots: Vec<Option<Result<Field>>> = (0..frames.len()).map(|_| None).collect();
    slots[0] = Some(Ok(Field::global(Transform::IDENTITY)));
    let reference = frames[0];
    std::thread::scope(|s| {
        for ((slot, f), t) in slots.iter_mut().zip(frames).zip(transforms).skip(1) {
            s.spawn(move || {
                *slot = Some(align_field(reference, f, *t));
            });
        }
    });
    slots
        .into_iter()
        .map(|s| s.expect("every slot filled"))
        .collect()
}

/// A frame's field printed for reports.
pub fn describe_field(f: &Field, weight: f32) -> String {
    let (mean, max) = f.correction_stats();
    format!(
        "{}  tiles {}/{} corrected, mean {:.2} max {:.2} px",
        describe_transform(&f.global, weight),
        f.valid_tiles(),
        f.d.len(),
        mean,
        max
    )
}

// ---------------------------------------------------------------------
// Noise model

/// Sensor noise as `σ²(s) = a·s + b`, `s` in DN above black: shot noise
/// (`a` is DN per electron) plus read noise (`b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseModel {
    pub a: f32,
    pub b: f32,
}

impl NoiseModel {
    pub fn sigma(&self, signal: f32) -> f32 {
        (self.a * signal.max(0.0) + self.b).max(0.0).sqrt()
    }
}

/// Fit the noise model from two aligned planes of the same scene. The
/// difference of a pair is binned by the pair mean, which for equal
/// variances is independent of the difference, so the bins are not
/// biased by their own noise. Variance per bin is from the MAD, so
/// moving content and misalignment only cost samples. Bins are
/// weighted by the precision of a variance estimate (`n / var²`) and
/// the fit stops at a quarter of white: the brightest bins are point
/// lights whose residual sub-pixel misalignment swamps the noise, and
/// with plain counts as weights their leverage drives the read-noise
/// intercept negative. `None` when no bin has enough samples.
pub fn noise_model(
    reference: &SensorFrame,
    frame: &SensorFrame,
    shift: (i32, i32),
) -> Option<NoiseModel> {
    const STRIDE: u32 = 5;
    const BINS_PER_OCTAVE: f32 = 4.0;
    const MIN_SAMPLES: usize = 500;
    let c = reference.crop;
    let (w, h) = (reference.width as i32, reference.height as i32);
    let stride = reference.width as usize;
    let site = |x: u32, y: u32| ((y & 1) * 2 + (x & 1)) as usize;
    let mut bins: Vec<(Vec<f32>, f64)> = vec![(Vec::new(), 0.0); 64];
    let mut y = c.y;
    while y < c.y + c.height {
        let mut x = c.x;
        while x < c.x + c.width {
            let (fx, fy) = (x as i32 + shift.0, y as i32 + shift.1);
            if fx >= 0 && fy >= 0 && fx < w && fy < h {
                let s = site(x, y);
                let black = reference.black[s];
                let white = reference.white[s] - black;
                let r = reference.data[y as usize * stride + x as usize] as f32 - black;
                let v = frame.data[fy as usize * stride + fx as usize] as f32 - black;
                let m = 0.5 * (r + v);
                if m < white * 0.25 {
                    let bin = if m < 1.0 {
                        0
                    } else {
                        (1.0 + BINS_PER_OCTAVE * m.log2()).floor() as usize
                    };
                    if let Some((d, sum)) = bins.get_mut(bin.min(63)) {
                        d.push(v - r);
                        *sum += m as f64;
                    }
                }
            }
            x += STRIDE;
        }
        y += STRIDE;
    }
    // Weighted least squares of σ² on the bin means.
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut used = 0;
    for (d, sum) in bins.iter_mut() {
        if d.len() < MIN_SAMPLES {
            continue;
        }
        let mean = *sum / d.len() as f64;
        let med = median(d);
        let mut dev: Vec<f32> = d.iter().map(|v| (v - med).abs()).collect();
        let mad = median(&mut dev) as f64;
        // Pair difference variance is 2σ².
        let var = ((1.4826 * mad).powi(2) * 0.5).max(0.25);
        let wgt = d.len() as f64 / (var * var);
        sw += wgt;
        sx += wgt * mean;
        sy += wgt * var;
        sxx += wgt * mean * mean;
        sxy += wgt * mean * var;
        used += 1;
    }
    if used == 0 {
        return None;
    }
    let det = sw * sxx - sx * sx;
    let (a, b) = if used >= 2 && det.abs() > 1e-9 {
        let a = (sw * sxy - sx * sy) / det;
        let b = (sxx * sy - sx * sxy) / det;
        (a.max(0.0), b.max(0.0))
    } else {
        (0.0, sy / sw)
    };
    Some(NoiseModel {
        a: a as f32,
        b: b as f32,
    })
}

fn median(v: &mut [f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let m = *m;
    if v.len() % 2 == 1 {
        m
    } else {
        let lo = v[..mid].iter().copied().fold(f32::MIN, f32::max);
        0.5 * (lo + m)
    }
}

// ---------------------------------------------------------------------
// Merge

/// A merged burst: the reference frame with the merged plane.
pub struct Merged {
    pub frame: SensorFrame,
    /// Single-frame noise model the robustness weights used.
    pub noise: NoiseModel,
    /// Shifts as applied, even sensor pixels, one per input frame.
    pub offsets: Vec<(i32, i32)>,
    /// Mean robustness weight per input frame over its samples inside
    /// the crop: 1 when a frame agrees everywhere, lower where it was
    /// outvoted.
    pub mean_weight: Vec<f32>,
}

/// Merge aligned frames onto the first one. `shifts[i]` is frame `i`'s
/// shift relative to frame 0 (`shifts[0]` is ignored). The reference's
/// black, white and CFA are taken for all frames.
pub fn merge(frames: &[&SensorFrame], shifts: &[Shift]) -> Result<Merged> {
    ensure!(!frames.is_empty(), "no frames to merge");
    ensure!(frames.len() == shifts.len(), "one shift per frame");
    ensure!(frames.len() <= 64, "at most 64 frames per merge");
    let reference = frames[0];
    for f in frames {
        ensure!(
            f.width == reference.width && f.height == reference.height && f.cfa == reference.cfa,
            "frames differ in raster or CFA"
        );
    }
    let mut offsets: Vec<(i32, i32)> = shifts.iter().map(Shift::even).collect();
    offsets[0] = (0, 0);
    let noise = frames
        .iter()
        .zip(&offsets)
        .skip(1)
        .find_map(|(f, &o)| noise_model(reference, f, o))
        .unwrap_or_else(|| {
            let s = dark_sigma(reference);
            NoiseModel { a: 0.0, b: s * s }
        });

    let (w, h) = (reference.width as i32, reference.height as i32);
    let stride = reference.width as usize;
    let n = frames.len();
    let mut out = vec![0u16; reference.data.len()];
    let sums = std::sync::Mutex::new((vec![0f64; n], vec![0u64; n]));
    par_rows(&mut out, stride, |y0, band| {
        let mut vals: Vec<f32> = Vec::with_capacity(n);
        let mut sorted: Vec<f32> = Vec::with_capacity(n);
        let mut local = vec![0f64; n];
        let mut counted = vec![0u64; n];
        for (dy, row) in band.chunks_mut(stride).enumerate() {
            let y = (y0 + dy) as i32;
            for (x, o) in row.iter_mut().enumerate() {
                let x = x as i32;
                let site = ((y & 1) * 2 + (x & 1)) as usize;
                let black = reference.black[site];
                vals.clear();
                // Frame index per gathered value, for the weight stats.
                let mut who = [0u8; 64];
                for (i, (f, &(ox, oy))) in frames.iter().zip(&offsets).enumerate() {
                    let (fx, fy) = (x + ox, y + oy);
                    if fx >= 0 && fy >= 0 && fx < w && fy < h {
                        who[vals.len()] = i as u8;
                        vals.push(f.data[fy as usize * stride + fx as usize] as f32 - black);
                    }
                }
                if vals.is_empty() {
                    *o = reference.data[y as usize * stride + x as usize];
                    continue;
                }
                sorted.clear();
                sorted.extend_from_slice(&vals);
                let m = median(&mut sorted);
                // Floor at the quantisation step so a model with no
                // read-noise estimate cannot reject everything.
                let sigma_d = noise.sigma(m).max(1.0) * std::f32::consts::SQRT_2;
                let k = 1.0 / (ROBUST_SIGMAS * sigma_d);
                let (mut num, mut den) = (0f32, 0f32);
                let in_crop = x >= reference.crop.x as i32
                    && y >= reference.crop.y as i32
                    && x < (reference.crop.x + reference.crop.width) as i32
                    && y < (reference.crop.y + reference.crop.height) as i32;
                for (j, &v) in vals.iter().enumerate() {
                    let t = (v - m) * k;
                    let wgt = (-0.5 * t * t).exp();
                    num += wgt * v;
                    den += wgt;
                    if in_crop {
                        local[who[j] as usize] += wgt as f64;
                        counted[who[j] as usize] += 1;
                    }
                }
                let merged = if den > 0.0 { num / den } else { m };
                *o = (merged + black).round().clamp(0.0, 65535.0) as u16;
            }
        }
        let mut g = sums.lock().unwrap();
        for (a, b) in g.0.iter_mut().zip(&local) {
            *a += b;
        }
        for (a, b) in g.1.iter_mut().zip(&counted) {
            *a += b;
        }
    });
    let (sums, counted) = sums.into_inner().unwrap();
    let mean_weight = sums
        .iter()
        .zip(&counted)
        .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
        .collect();
    let mut frame = reference.clone();
    frame.data = out;
    Ok(Merged {
        frame,
        noise,
        offsets,
        mean_weight,
    })
}

// ---------------------------------------------------------------------
// Reconstruction: kernel regression onto an output grid

/// Bilinear sample of a plane; `None` within a pixel of the border.
fn bilinear(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> Option<f32> {
    if x < 1.0 || y < 1.0 || x >= (w - 2) as f32 || y >= (h - 2) as f32 {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |i: usize, j: usize| p[j * w + i];
    Some(
        at(x0, y0) * (1.0 - fx) * (1.0 - fy)
            + at(x0 + 1, y0) * fx * (1.0 - fy)
            + at(x0, y0 + 1) * (1.0 - fx) * fy
            + at(x0 + 1, y0 + 1) * fx * fy,
    )
}

/// Bilinear sample of a plane with its gradient, the latter by central
/// differences of bilinear samples one pixel either side of the point
/// itself. (Differencing at the integer floor instead, as an earlier
/// version did, biases every Gauss-Newton step the same way on
/// point-like content, and the refinement then walks away from the
/// coarse fit instead of converging: DESIGN.md, step 3 correction.)
fn sample_plane(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> Option<(f32, f32, f32)> {
    let v = bilinear(p, w, h, x, y)?;
    let gx = (bilinear(p, w, h, x + 1.0, y)? - bilinear(p, w, h, x - 1.0, y)?) * 0.5;
    let gy = (bilinear(p, w, h, x, y + 1.0)? - bilinear(p, w, h, x, y - 1.0)?) * 0.5;
    Some((v, gx, gy))
}

/// Reconstruct camera RGB on the reference crop at `scale`× by kernel
/// regression over every frame's sites. `fields[i]` maps reference
/// sensor coordinates into frame `i` (index 0 is the identity). Each
/// frame contributes the 3×3 sites around the mapped point, weighted by
/// an isotropic Gaussian of `kernel_sigma` sensor pixels and by a
/// robustness weight from the half-resolution planes (see DESIGN.md,
/// step 3). Output values are black-subtracted and normalised to white,
/// before white balance.
pub fn reconstruct(
    frames: &[&SensorFrame],
    fields: &[Field],
    noise: NoiseModel,
    scale: u32,
    kernel_sigma: f32,
) -> Result<CameraRgb> {
    ensure!(!frames.is_empty(), "no frames");
    ensure!(frames.len() == fields.len(), "one field per frame");
    ensure!((1..=4).contains(&scale), "scale must be 1..=4");
    ensure!(kernel_sigma > 0.05, "kernel sigma too small");
    let reference = frames[0];
    for f in frames {
        ensure!(
            f.width == reference.width && f.height == reference.height && f.cfa == reference.cfa,
            "frames differ in raster or CFA"
        );
    }
    let grays: Vec<Gray> = frames.iter().map(|f| gray(f)).collect();
    let crop = reference.crop;
    let (gw, gh) = (grays[0].width, grays[0].height);
    let (out_w, out_h) = (crop.width * scale, crop.height * scale);
    let mut rgba = vec![half::f16::ZERO; (out_w * out_h * 4) as usize];
    let n = frames.len();
    let (raster_w, raster_h) = (reference.width as i32, reference.height as i32);
    let stride = reference.width as usize;
    let inv_2s2 = -1.0 / (2.0 * kernel_sigma * kernel_sigma);
    let stats = std::sync::Mutex::new((vec![0f64; n], 0u64, 0u64));
    par_rows(&mut rgba, out_w as usize * 4, |y0, band| {
        let mut wsum_local = vec![0f64; n];
        let mut pixels = 0u64;
        let mut uncovered = 0u64;
        for (dy, row) in band.chunks_mut(out_w as usize * 4).enumerate() {
            let oy = (y0 + dy) as u32;
            let y = crop.y as f32 + (oy as f32 + 0.5) / scale as f32 - 0.5;
            for (ox, px) in row.chunks_mut(4).enumerate() {
                let x = crop.x as f32 + (ox as f32 + 0.5) / scale as f32 - 0.5;
                let qx = (x - crop.x as f32 - 0.5) * 0.5;
                let qy = (y - crop.y as f32 - 0.5) * 0.5;
                let (g_ref, grad) = match sample_plane(&grays[0].data, gw, gh, qx, qy) {
                    Some((v, gx, gy)) => (v, gx.hypot(gy)),
                    None => (0.0, 0.0),
                };
                // Noise of a quad mean is a quarter of a site's; the
                // difference of two carries twice that; misalignment of
                // 0.25 sensor px (an eighth of a quad) adds along ∇.
                let sigma2 =
                    2.0 * (noise.a * g_ref.max(0.0) + noise.b) * 0.25 + (0.125 * grad).powi(2);
                let k_rob = 1.0 / (ROBUST_SIGMAS * ROBUST_SIGMAS * sigma2.max(0.25));
                let mut acc = [0f32; 3];
                let mut wgt = [0f32; 3];
                for (i, (f, t)) in frames.iter().zip(fields).enumerate() {
                    let (px_f, py_f) = if i == 0 { (x, y) } else { t.apply(x, y) };
                    let w_frame = if i == 0 {
                        1.0
                    } else {
                        let fqx = (px_f - crop.x as f32 - 0.5) * 0.5;
                        let fqy = (py_f - crop.y as f32 - 0.5) * 0.5;
                        match sample_plane(&grays[i].data, gw, gh, fqx, fqy) {
                            Some((g, _, _)) => {
                                let d = g - g_ref;
                                (-0.5 * d * d * k_rob).exp()
                            }
                            None => continue,
                        }
                    };
                    wsum_local[i] += w_frame as f64;
                    let (cx, cy) = (px_f.round() as i32, py_f.round() as i32);
                    for sy in cy - 1..=cy + 1 {
                        if sy < 0 || sy >= raster_h {
                            continue;
                        }
                        for sx in cx - 1..=cx + 1 {
                            if sx < 0 || sx >= raster_w {
                                continue;
                            }
                            let site = ((sy & 1) * 2 + (sx & 1)) as usize;
                            let colour = reference.cfa[site] as usize;
                            let v = (f.data[sy as usize * stride + sx as usize] as f32
                                - reference.black[site])
                                / (reference.white[site] - reference.black[site]);
                            let ddx = sx as f32 - px_f;
                            let ddy = sy as f32 - py_f;
                            let kw = w_frame * ((ddx * ddx + ddy * ddy) * inv_2s2).exp();
                            acc[colour] += kw * v;
                            wgt[colour] += kw;
                        }
                    }
                }
                pixels += 1;
                for c in 0..3 {
                    px[c] = if wgt[c] > 1e-6 {
                        half::f16::from_f32(acc[c] / wgt[c])
                    } else {
                        uncovered += 1;
                        half::f16::ZERO
                    };
                }
                px[3] = half::f16::ONE;
            }
        }
        let mut g = stats.lock().unwrap();
        for (a, b) in g.0.iter_mut().zip(&wsum_local) {
            *a += b;
        }
        g.1 += pixels;
        g.2 += uncovered;
    });
    let (wsum, pixels, uncovered) = stats.into_inner().unwrap();
    let mean_weight = wsum
        .iter()
        .map(|&s| {
            if pixels > 0 {
                (s / pixels as f64) as f32
            } else {
                0.0
            }
        })
        .collect();
    let mut frame = reference.clone();
    frame.data = Vec::new();
    Ok(CameraRgb {
        frame,
        width: out_w,
        height: out_h,
        rgba,
        mean_weight,
        uncovered: uncovered as f32 / (pixels.max(1) * 3) as f32,
    })
}

/// Load the files as one burst, fit a similarity and a local field per
/// frame, and reconstruct at `scale`×.
pub fn reconstruct_files(
    paths: &[impl AsRef<Path>],
    scale: u32,
    kernel_sigma: f32,
) -> Result<(CameraRgb, Vec<Field>)> {
    ensure!(!paths.is_empty(), "no frames");
    let frames = load_all(paths)?;
    let refs: Vec<&SensorFrame> = frames.iter().collect();
    let shifts = align_all(&refs)?;
    let transforms = similarity_all(&refs, &shifts)?;
    let fields = fields_all(&refs, &transforms)?;
    let noise = frames
        .get(1)
        .and_then(|f| noise_model(&frames[0], f, shifts[1].even()))
        .unwrap_or_else(|| {
            let s = dark_sigma(&frames[0]);
            NoiseModel { a: 0.0, b: s * s }
        });
    let out = reconstruct(&refs, &fields, noise, scale, kernel_sigma)?;
    Ok((out, fields))
}

/// Similarity of every frame relative to the first, in parallel, from
/// the global shifts.
pub fn similarity_all(frames: &[&SensorFrame], shifts: &[Shift]) -> Result<Vec<Transform>> {
    let mut slots: Vec<Option<Result<Transform>>> = (0..frames.len()).map(|_| None).collect();
    slots[0] = Some(Ok(Transform::IDENTITY));
    let reference = frames[0];
    std::thread::scope(|s| {
        for ((slot, f), shift) in slots.iter_mut().zip(frames).zip(shifts).skip(1) {
            s.spawn(move || {
                *slot = Some(align_similarity(reference, f, *shift));
            });
        }
    });
    slots
        .into_iter()
        .map(|s| s.expect("every slot filled"))
        .collect()
}

/// A frame's similarity printed for reports.
pub fn describe_transform(t: &Transform, weight: f32) -> String {
    format!(
        "shift ({:+8.2}, {:+8.2}) px  rotation {:+.4}°  scale {:.5}  tiles {:2}  residual {:.3}  weight {:.3}",
        t.t[0],
        t.t[1],
        t.rotation_deg(),
        t.scale(),
        t.tiles,
        t.residual,
        weight
    )
}

/// Load the files as one burst and [`merge_frames`] them.
pub fn merge_files(paths: &[impl AsRef<Path>]) -> Result<(Merged, Report)> {
    ensure!(paths.len() >= 2, "a burst needs at least two frames");
    let frames = load_all(paths)?;
    merge_frames(&frames)
}

/// One input frame's part in a merge ([`Report`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameReport {
    /// Measured shift relative to the reference (zero for it).
    pub shift: Shift,
    /// Even-pixel offset applied; zero when the frame was not used.
    pub offset: (i32, i32),
    /// Mean robustness weight inside the crop; 0 when not used.
    pub weight: f32,
    /// [`sharpness`] relative to the burst's sharpest frame, 0..1.
    pub sharpness: f32,
    pub used: bool,
}

impl FrameReport {
    pub fn describe(&self) -> String {
        if self.used {
            format!(
                "{}  sharp {:.2}",
                describe(&self.shift, self.offset, self.weight),
                self.sharpness
            )
        } else {
            format!(
                "dx {:+7.2} dy {:+7.2} px  peak {:.3}  sharp {:.2}  dropped",
                self.shift.dx, self.shift.dy, self.shift.peak, self.sharpness
            )
        }
    }
}

/// How [`merge_frames`] treated each input, in input order.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Index of the frame the others were merged onto: the sharpest.
    pub reference: usize,
    pub frames: Vec<FrameReport>,
    pub noise: NoiseModel,
}

impl Report {
    pub fn used(&self) -> usize {
        self.frames.iter().filter(|f| f.used).count()
    }

    /// Mean robustness weight of the used frames other than the
    /// reference (whose weight is 1 by construction).
    pub fn mean_weight(&self) -> f32 {
        let others: Vec<f32> = self
            .frames
            .iter()
            .enumerate()
            .filter(|(i, f)| f.used && *i != self.reference)
            .map(|(_, f)| f.weight)
            .collect();
        if others.is_empty() {
            1.0
        } else {
            others.iter().sum::<f32>() / others.len() as f32
        }
    }
}

/// Merge a burst: the sharpest frame is the reference, frames below
/// [`DROP_SHARPNESS`] of it are left out, the rest are aligned to it
/// and merged.
pub fn merge_frames(frames: &[SensorFrame]) -> Result<(Merged, Report)> {
    ensure!(frames.len() >= 2, "a burst needs at least two frames");
    let n = frames.len();
    let sharp = sharpness_all(frames);
    let max = sharp.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
    let rel: Vec<f32> = sharp.iter().map(|s| s / max).collect();
    let reference = (0..n)
        .max_by(|&a, &b| rel[a].partial_cmp(&rel[b]).unwrap_or(Ordering::Equal))
        .expect("at least two frames");
    // Alignment and the low-level merge take frames[0] as reference.
    let order: Vec<usize> = std::iter::once(reference)
        .chain((0..n).filter(|&i| i != reference))
        .collect();
    let refs: Vec<&SensorFrame> = order.iter().map(|&i| &frames[i]).collect();
    let shifts = align_all(&refs)?;
    let used: Vec<bool> = order.iter().map(|&i| rel[i] >= DROP_SHARPNESS).collect();
    let kept: Vec<&SensorFrame> = refs
        .iter()
        .zip(&used)
        .filter(|(_, u)| **u)
        .map(|(f, _)| *f)
        .collect();
    let kept_shifts: Vec<Shift> = shifts
        .iter()
        .zip(&used)
        .filter(|(_, u)| **u)
        .map(|(s, _)| *s)
        .collect();
    let merged = merge(&kept, &kept_shifts)?;
    let mut report = vec![
        FrameReport {
            shift: Shift::ZERO,
            offset: (0, 0),
            weight: 0.0,
            sharpness: 0.0,
            used: false,
        };
        n
    ];
    let mut k = 0;
    for (j, &i) in order.iter().enumerate() {
        let mut r = FrameReport {
            shift: shifts[j],
            offset: (0, 0),
            weight: 0.0,
            sharpness: rel[i],
            used: used[j],
        };
        if used[j] {
            r.offset = merged.offsets[k];
            r.weight = merged.mean_weight[k];
            k += 1;
        }
        report[i] = r;
    }
    let noise = merged.noise;
    Ok((
        merged,
        Report {
            reference,
            frames: report,
            noise,
        },
    ))
}

/// [`sharpness`] of every frame, in parallel.
pub fn sharpness_all(frames: &[SensorFrame]) -> Vec<f32> {
    let mut out = vec![0f32; frames.len()];
    std::thread::scope(|s| {
        for (o, f) in out.iter_mut().zip(frames) {
            s.spawn(move || *o = sharpness(f));
        }
    });
    out
}

/// How a burst's files are named in reports and the develop note:
/// the reference, then the others, then the count.
pub fn source_name(paths: &[impl AsRef<Path>]) -> String {
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let mut names = paths.iter().filter_map(|p| name(p.as_ref()));
    let Some(first) = names.next() else {
        return String::new();
    };
    let rest: Vec<String> = names.collect();
    if rest.is_empty() {
        first
    } else {
        format!("{first} + {} ({} frames)", rest.join(", "), rest.len() + 1)
    }
}

/// Unpack every file, in parallel.
pub fn load_all(paths: &[impl AsRef<Path>]) -> Result<Vec<SensorFrame>> {
    let mut slots: Vec<Option<Result<SensorFrame>>> = (0..paths.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        for (slot, p) in slots.iter_mut().zip(paths) {
            let p = p.as_ref();
            s.spawn(move || {
                *slot = Some(sensor::unpack(p).with_context(|| format!("load {}", p.display())));
            });
        }
    });
    slots
        .into_iter()
        .map(|s| s.expect("every slot filled"))
        .collect()
}

/// Shift of every frame relative to the first, in parallel.
pub fn align_all(frames: &[&SensorFrame]) -> Result<Vec<Shift>> {
    let mut slots: Vec<Option<Result<Shift>>> = (0..frames.len()).map(|_| None).collect();
    slots[0] = Some(Ok(Shift::ZERO));
    let reference = frames[0];
    std::thread::scope(|s| {
        for (slot, f) in slots.iter_mut().zip(frames).skip(1) {
            s.spawn(move || {
                *slot = Some(align(reference, f));
            });
        }
    });
    slots
        .into_iter()
        .map(|s| s.expect("every slot filled"))
        .collect()
}

/// A frame's shift and weight printed for reports.
pub fn describe(shift: &Shift, offset: (i32, i32), weight: f32) -> String {
    format!(
        "dx {:+7.2} dy {:+7.2} px  applied ({:+4}, {:+4})  peak {:.3}  weight {:.3}",
        shift.dx, shift.dy, offset.0, offset.1, shift.peak, weight
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::{CfaColor, Rect};
    use rawler::Orientation;

    fn summary(seq: Option<u32>, secs: u32, ms: u32) -> ExifSummary {
        ExifSummary {
            exposure_time: Some((1, 8)),
            iso: Some(320),
            f_number: Some(2.8),
            focal_length_mm: Some(28.0),
            date_time_original: Some(format!("2026:09:15 19:07:{secs:02}")),
            subsec_ms: Some(ms),
            sequence: seq,
            ..Default::default()
        }
    }

    #[test]
    fn groups_by_sequence_and_time() {
        let s = [
            summary(Some(1), 41, 684),
            summary(Some(2), 41, 944),
            summary(Some(3), 42, 204),
            summary(Some(1), 46, 238), // new burst: counter restarts
            summary(Some(2), 46, 498),
            summary(None, 50, 0), // single
            summary(Some(1), 58, 846),
        ];
        let refs: Vec<Option<&ExifSummary>> = s.iter().map(Some).collect();
        assert_eq!(
            group(&refs),
            vec![vec![0, 1, 2], vec![3, 4], vec![5], vec![6]]
        );
    }

    #[test]
    fn a_long_gap_breaks_a_burst_even_if_the_counter_continues() {
        let s = [summary(Some(1), 0, 0), summary(Some(2), 10, 0)];
        let refs: Vec<Option<&ExifSummary>> = s.iter().map(Some).collect();
        assert_eq!(group(&refs), vec![vec![0], vec![1]]);
    }

    #[test]
    fn groups_follow_time_not_input_order() {
        let s = [summary(Some(2), 41, 944), summary(Some(1), 41, 684)];
        let refs: Vec<Option<&ExifSummary>> = s.iter().map(Some).collect();
        assert_eq!(group(&refs), vec![vec![1, 0]]);
    }

    #[test]
    fn timestamp_epoch_and_subsec() {
        let e = summary(None, 51, 946);
        // 2026-09-15 19:07:51 UTC
        assert_eq!(e.timestamp_secs(), Some(1789499271.946));
        let mut none = e.clone();
        none.date_time_original = None;
        assert_eq!(none.timestamp_secs(), None);
    }

    /// A synthetic frame: sparse bright blobs on a dark noisy plane,
    /// like a night skyline, so the mask has something to pass.
    fn synthetic(width: u32, height: u32, shift: (f32, f32), seed: u32) -> SensorFrame {
        synthetic_blobs(width, height, shift, seed, 50.0)
    }

    /// `falloff` is the blob's exp(-d²/falloff): 50 gives 5 px blobs,
    /// 3 gives point-like lights about a pixel wide.
    fn synthetic_blobs(
        width: u32,
        height: u32,
        shift: (f32, f32),
        seed: u32,
        falloff: f32,
    ) -> SensorFrame {
        let mut rng = seed.wrapping_mul(2654435761).wrapping_add(12345);
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32
        };
        let black = 512.0;
        let mut data = vec![0u16; (width * height) as usize];
        // Fixed blob positions in reference coordinates.
        let blobs: Vec<(f32, f32, f32)> = (0..40)
            .map(|i| {
                let t = i as f32 * 0.618;
                (
                    (t.fract() * width as f32 * 0.8 + width as f32 * 0.1),
                    (((t * 3.7).fract()) * height as f32 * 0.8 + height as f32 * 0.1),
                    200.0 + 800.0 * (t * 7.3).fract(),
                )
            })
            .collect();
        for y in 0..height {
            for x in 0..width {
                // Content of the reference at (x - sx, y - sy) appears here.
                let (rx, ry) = (x as f32 - shift.0, y as f32 - shift.1);
                let mut s = 0.0;
                for &(bx, by, amp) in &blobs {
                    let d2 = (rx - bx).powi(2) + (ry - by).powi(2);
                    if d2 < 400.0 {
                        s += amp * (-d2 / falloff).exp();
                    }
                }
                let noise = (next() + next() + next() - 1.5) * 8.0;
                data[(y * width + x) as usize] = (black + s + noise).round().max(0.0) as u16;
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
            black: [black; 4],
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
    fn align_recovers_a_known_shift() {
        let reference = synthetic(256, 192, (0.0, 0.0), 1);
        let frame = synthetic(256, 192, (6.0, -10.0), 2);
        let s = align(&reference, &frame).unwrap();
        assert!((s.dx - 6.0).abs() < 0.6, "{s:?}");
        assert!((s.dy + 10.0).abs() < 0.6, "{s:?}");
        assert_eq!(s.even(), (6, -10));
        assert!(s.peak > 0.05, "{s:?}");
    }

    #[test]
    fn align_refuses_a_dark_frame() {
        let mut reference = synthetic(128, 96, (0.0, 0.0), 1);
        for v in reference.data.iter_mut() {
            *v = 512;
        }
        assert!(align(&reference, &reference).is_err());
    }

    #[test]
    fn noise_model_recovers_read_noise() {
        // Two frames of the same scene, independent noise, no shift.
        let a = synthetic(256, 192, (0.0, 0.0), 3);
        let b = synthetic(256, 192, (0.0, 0.0), 4);
        let m = noise_model(&a, &b, (0, 0)).unwrap();
        // Sum of three uniforms scaled by 8: variance 3/12 * 64 = 16.
        assert!((m.b - 16.0).abs() < 4.0, "{m:?}");
        assert!(m.a < 0.5, "{m:?}");
    }

    #[test]
    fn merge_frames_drops_the_blurred_frame_and_takes_the_sharpest() {
        let mut frames: Vec<SensorFrame> = (0..4)
            .map(|i| synthetic(256, 192, (2.0 * i as f32, -2.0 * i as f32), 20 + i as u32))
            .collect();
        // Smear frame 0 along rows over same-colour sites.
        let w = frames[0].width as usize;
        let src = frames[0].data.clone();
        for (i, v) in frames[0].data.iter_mut().enumerate() {
            let x = i % w;
            let taps: Vec<u32> = (-3..=3)
                .filter_map(|k| {
                    let xx = x as i64 + 2 * k;
                    (xx >= 0 && (xx as usize) < w).then(|| src[i - x + xx as usize] as u32)
                })
                .collect();
            *v = (taps.iter().sum::<u32>() / taps.len() as u32) as u16;
        }
        let (merged, report) = merge_frames(&frames).unwrap();
        assert_ne!(report.reference, 0, "{report:?}");
        assert!(!report.frames[0].used, "{:?}", report.frames[0]);
        assert!(report.frames[0].sharpness < DROP_SHARPNESS);
        assert_eq!(report.used(), 3);
        assert!((report.frames[report.reference].sharpness - 1.0).abs() < 1e-6);
        assert!(
            report
                .frames
                .iter()
                .enumerate()
                .all(|(i, f)| f.used || i == 0)
        );
        assert_eq!(merged.offsets.len(), 3);
    }

    #[test]
    fn merge_reduces_noise_and_keeps_signal() {
        let frames: Vec<SensorFrame> = (0..8)
            .map(|i| synthetic(256, 192, (2.0 * i as f32, -2.0 * i as f32), 10 + i as u32))
            .collect();
        let refs: Vec<&SensorFrame> = frames.iter().collect();
        let shifts = align_all(&refs).unwrap();
        let merged = merge(&refs, &shifts).unwrap();
        let n = frames.len() as f32;
        // Dark noise falls by about sqrt(N).
        let before = dark_sigma(&frames[0]);
        let after = dark_sigma(&merged.frame);
        assert!(
            after < before / n.sqrt() * 1.4,
            "before {before} after {after}"
        );
        // Signal is preserved: brightest site stays within noise of the reference's.
        let max_ref = *frames[0].data.iter().max().unwrap() as f32;
        let max_merged = *merged.frame.data.iter().max().unwrap() as f32;
        assert!(
            (max_ref - max_merged).abs() < 40.0,
            "{max_ref} vs {max_merged}"
        );
        for w in &merged.mean_weight {
            assert!(*w > 0.9, "{:?}", merged.mean_weight);
        }
    }

    #[test]
    fn merge_outvotes_an_outlier() {
        let mut frames: Vec<SensorFrame> = (0..5)
            .map(|i| synthetic(256, 192, (0.0, 0.0), 20 + i as u32))
            .collect();
        // A hot pixel in the reference, on a dark site.
        let at = 33 * 256 + 20;
        assert!(frames[0].data[at] < 600);
        frames[0].data[at] = 9000;
        let refs: Vec<&SensorFrame> = frames.iter().collect();
        let shifts = vec![Shift::ZERO; 5];
        let merged = merge(&refs, &shifts).unwrap();
        assert!(merged.frame.data[at] < 600, "{}", merged.frame.data[at]);
        assert!(merged.mean_weight[0] > 0.9, "{:?}", merged.mean_weight);
    }

    #[test]
    fn similarity_fit_recovers_rotation_and_shift() {
        let t = Transform {
            a: [[0.9999, -0.0007], [0.0007, 0.9999]],
            t: [12.3, -7.9],
            tiles: 0,
            residual: 0.0,
        };
        let mut matches: Vec<TileMatch> = (0..12)
            .map(|i| {
                let p = [
                    500.0 + 480.0 * (i % 6) as f32,
                    1000.0 + 700.0 * (i / 6) as f32,
                ];
                let (x, y) = t.apply(p[0], p[1]);
                TileMatch {
                    p,
                    q: [x, y],
                    w: 0.05,
                }
            })
            .collect();
        // One bad tile.
        matches[4].q[0] += 30.0;
        let f = fit_similarity(&mut matches, Shift::ZERO);
        assert_eq!(f.tiles, 11);
        assert!((f.rotation_deg() - t.rotation_deg()).abs() < 1e-3, "{f:?}");
        assert!(
            (f.t[0] - t.t[0]).abs() < 0.05 && (f.t[1] - t.t[1]).abs() < 0.05,
            "{f:?}"
        );
        assert!(f.residual < 0.05, "{f:?}");
    }

    #[test]
    fn similarity_on_synthetic_frames() {
        // 1024 x 768 gives 2 x 1 tiles of 256 quads; content across both.
        let reference = synthetic(1024, 768, (0.0, 0.0), 7);
        let frame = synthetic(1024, 768, (5.3, -3.7), 8);
        let g = align(&reference, &frame).unwrap();
        let t = align_similarity(&reference, &frame, g).unwrap();
        let (x, y) = t.apply(300.0, 300.0);
        assert!(
            (x - 305.3).abs() < 0.1 && (y - 296.3).abs() < 0.1,
            "{t:?} global {g:?}"
        );
        assert!(t.rotation_deg().abs() < 0.01, "{t:?}");
    }

    /// A rotated frame: content of the reference at p appears at R p + t.
    fn synthetic_rotated(
        width: u32,
        height: u32,
        deg: f32,
        shift: (f32, f32),
        seed: u32,
    ) -> SensorFrame {
        let mut f = synthetic(width, height, (0.0, 0.0), seed);
        let base = synthetic(width, height, (0.0, 0.0), seed);
        let (s, c) = deg.to_radians().sin_cos();
        let (cx, cy) = (width as f32 * 0.5, height as f32 * 0.5);
        for y in 0..height {
            for x in 0..width {
                // Inverse map: reference point for this frame pixel.
                let (dx, dy) = (x as f32 - shift.0 - cx, y as f32 - shift.1 - cy);
                let (rx, ry) = (c * dx + s * dy + cx, -s * dx + c * dy + cy);
                let (ix, iy) = (rx.round() as i32, ry.round() as i32);
                let v = if ix >= 0 && iy >= 0 && ix < width as i32 && iy < height as i32 {
                    base.data[(iy as u32 * width + ix as u32) as usize]
                } else {
                    512
                };
                f.data[(y * width + x) as usize] = v;
            }
        }
        f
    }

    #[test]
    fn similarity_recovers_a_small_rotation() {
        let reference = synthetic(1024, 768, (0.0, 0.0), 7);
        let frame = synthetic_rotated(1024, 768, 0.3, (4.0, -2.0), 7);
        let g = align(&reference, &frame).unwrap();
        let t = align_similarity(&reference, &frame, g).unwrap();
        assert!((t.rotation_deg() - 0.3).abs() < 0.03, "{t:?}");
        // Nearest-neighbour resampling in the fixture limits accuracy to ~0.5 px.
        let (x, y) = t.apply(512.0, 384.0);
        assert!((x - 516.0).abs() < 0.6 && (y - 382.0).abs() < 0.6, "{t:?}");
    }

    #[test]
    fn reconstruct_single_frame_is_a_soft_demosaic() {
        let f = synthetic(256, 192, (0.0, 0.0), 31);
        let out = reconstruct(
            &[&f],
            &[Field::global(Transform::IDENTITY)],
            NoiseModel { a: 0.0, b: 16.0 },
            1,
            0.7,
        )
        .unwrap();
        assert_eq!((out.width, out.height), (248, 184));
        assert!(out.uncovered == 0.0, "{}", out.uncovered);
        // The brightest blob keeps its level (within the kernel's blur).
        let peak = out
            .rgba
            .chunks(4)
            .map(|p| p[1].to_f32())
            .fold(0.0f32, f32::max);
        let raw_peak = (*f.data.iter().max().unwrap() as f32 - 512.0) / (16383.0 - 512.0);
        assert!(
            peak > raw_peak * 0.5 && peak <= raw_peak * 1.01,
            "{peak} vs {raw_peak}"
        );
    }

    #[test]
    fn reconstruct_at_2x_doubles_the_grid() {
        let frames: Vec<SensorFrame> = (0..3)
            .map(|i| synthetic(256, 192, (1.5 * i as f32, -0.5 * i as f32), 40 + i as u32))
            .collect();
        let refs: Vec<&SensorFrame> = frames.iter().collect();
        let shifts = align_all(&refs).unwrap();
        let transforms = similarity_all(&refs, &shifts).unwrap();
        let fields: Vec<Field> = transforms.into_iter().map(Field::global).collect();
        let out = reconstruct(&refs, &fields, NoiseModel { a: 0.0, b: 16.0 }, 2, 0.7).unwrap();
        assert_eq!((out.width, out.height), (496, 368));
        assert!(out.uncovered == 0.0);
        for w in &out.mean_weight {
            assert!(*w > 0.8, "{:?}", out.mean_weight);
        }
    }

    #[test]
    fn refinement_converges_on_point_like_content() {
        // Sharp lights at fractional positions: the case the night sets
        // present. The refinement must not walk away from the coarse fit.
        let reference = synthetic_blobs(1024, 768, (0.0, 0.0), 51, 3.0);
        let frame = synthetic_blobs(1024, 768, (3.3, -2.6), 52, 3.0);
        let g = align(&reference, &frame).unwrap();
        let (coarse, refined) = align_similarity_stages(&reference, &frame, g).unwrap();
        for (x, y) in [(200.0, 200.0), (800.0, 600.0), (512.0, 384.0)] {
            let (cx, cy) = coarse.apply(x, y);
            let (rx, ry) = refined.apply(x, y);
            assert!(
                (cx - x - 3.3).abs() < 0.5 && (cy - y + 2.6).abs() < 0.5,
                "coarse {coarse:?}"
            );
            assert!(
                (rx - x - 3.3).abs() < 0.15 && (ry - y + 2.6).abs() < 0.15,
                "refined {refined:?} at ({x},{y}) -> ({rx},{ry})"
            );
        }
        assert!(refined.rotation_deg().abs() < 0.01, "{refined:?}");
        assert!((refined.scale() - 1.0).abs() < 2e-4, "{refined:?}");
    }

    const WARP_BLOBS: usize = 160;

    /// Blobs displaced by a position-dependent map: a similarity cannot
    /// fit different scales along x and y, a local field can. The
    /// reference is drawn with the same blob list, unwarped.
    fn synthetic_warped(
        width: u32,
        height: u32,
        seed: u32,
        warp: impl Fn(f32, f32) -> (f32, f32),
    ) -> SensorFrame {
        let mut f = synthetic_blobs(width, height, (0.0, 0.0), seed, 6.0);
        let mut rng = seed.wrapping_mul(2654435761).wrapping_add(12345);
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32
        };
        let blobs: Vec<(f32, f32, f32)> = (0..WARP_BLOBS)
            .map(|i| {
                let t = i as f32 * 0.618;
                (
                    (t.fract() * width as f32 * 0.8 + width as f32 * 0.1),
                    (((t * 3.7).fract()) * height as f32 * 0.8 + height as f32 * 0.1),
                    200.0 + 800.0 * (t * 7.3).fract(),
                )
            })
            .collect();
        for y in 0..height {
            for x in 0..width {
                let mut s = 0.0;
                for &(bx, by, amp) in &blobs {
                    let (wx, wy) = warp(bx, by);
                    let d2 = (x as f32 - wx).powi(2) + (y as f32 - wy).powi(2);
                    if d2 < 400.0 {
                        s += amp * (-d2 / 6.0).exp();
                    }
                }
                let noise = (next() + next() + next() - 1.5) * 8.0;
                f.data[(y * width + x) as usize] = (512.0 + s + noise).round().max(0.0) as u16;
            }
        }
        f
    }

    #[test]
    fn field_recovers_an_anisotropic_warp() {
        let warp = |x: f32, y: f32| (x + 3.0 + 0.004 * (x - 512.0), y - 2.0 - 0.003 * (y - 384.0));
        let reference = synthetic_warped(1024, 768, 61, |x, y| (x, y));
        let frame = synthetic_warped(1024, 768, 62, warp);
        let g = align(&reference, &frame).unwrap();
        let t = align_similarity(&reference, &frame, g).unwrap();
        let field = align_field(&reference, &frame, t).unwrap();
        assert!(field.valid_tiles() >= 2, "{:?}", field.valid);
        // Inside the hull of tile centres the field interpolates; at the
        // frame corners it extrapolates, where node errors get a lever
        // arm, so the bound is looser there.
        let (mut worst_global, mut worst_inside, mut worst_outside) = (0f32, 0f32, 0f32);
        for (x, y, inside) in [
            (300.0, 300.0, true),
            (600.0, 500.0, true),
            (512.0, 384.0, true),
            (150.0, 120.0, false),
            (900.0, 650.0, false),
        ] {
            let (tx, ty) = warp(x, y);
            let (gx, gy) = t.apply(x, y);
            let (fx, fy) = field.apply(x, y);
            worst_global = worst_global.max((gx - tx).hypot(gy - ty));
            let e = (fx - tx).hypot(fy - ty);
            if inside {
                worst_inside = worst_inside.max(e);
            } else {
                worst_outside = worst_outside.max(e);
            }
        }
        assert!(
            worst_global > 0.5,
            "the similarity should not fit this warp: {worst_global}"
        );
        assert!(
            worst_inside < 0.3,
            "field inside {worst_inside}, global {worst_global}"
        );
        assert!(worst_outside < 0.5, "field extrapolated {worst_outside}");
    }

    #[test]
    fn field_falls_back_to_the_global_fit_without_signal() {
        let reference = synthetic_blobs(1024, 768, (0.0, 0.0), 71, 3.0);
        let mut dark = reference.clone();
        for v in dark.data.iter_mut() {
            *v = 512;
        }
        let t = Transform::translation(2.0, -1.0);
        // The frame is dark: no tile can vote, the field is the global map.
        let field = align_field(&reference, &dark, t).unwrap();
        assert_eq!(field.valid_tiles(), 0);
        assert_eq!(field.apply(100.0, 100.0), (102.0, 99.0));
    }
}
