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
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::sensor::{self, ExifSummary, SensorFrame};

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
struct Gray {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

fn gray(frame: &SensorFrame) -> Gray {
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

/// Load the files as one burst, align each to the first and merge.
/// Returns the merge and the measured shifts (index 0 is zero).
pub fn merge_files(paths: &[impl AsRef<Path>]) -> Result<(Merged, Vec<Shift>)> {
    ensure!(paths.len() >= 2, "a burst needs at least two frames");
    let frames = load_all(paths)?;
    let refs: Vec<&SensorFrame> = frames.iter().collect();
    let shifts = align_all(&refs)?;
    let merged = merge(&refs, &shifts)?;
    Ok((merged, shifts))
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
    fn synthetic(width: u32, height: u32, shift: (i32, i32), seed: u32) -> SensorFrame {
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
                let (rx, ry) = (x as f32 - shift.0 as f32, y as f32 - shift.1 as f32);
                let mut s = 0.0;
                for &(bx, by, amp) in &blobs {
                    let d2 = (rx - bx).powi(2) + (ry - by).powi(2);
                    if d2 < 400.0 {
                        s += amp * (-d2 / 50.0).exp();
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
        let reference = synthetic(256, 192, (0, 0), 1);
        let frame = synthetic(256, 192, (6, -10), 2);
        let s = align(&reference, &frame).unwrap();
        assert!((s.dx - 6.0).abs() < 0.6, "{s:?}");
        assert!((s.dy + 10.0).abs() < 0.6, "{s:?}");
        assert_eq!(s.even(), (6, -10));
        assert!(s.peak > 0.05, "{s:?}");
    }

    #[test]
    fn align_refuses_a_dark_frame() {
        let mut reference = synthetic(128, 96, (0, 0), 1);
        for v in reference.data.iter_mut() {
            *v = 512;
        }
        assert!(align(&reference, &reference).is_err());
    }

    #[test]
    fn noise_model_recovers_read_noise() {
        // Two frames of the same scene, independent noise, no shift.
        let a = synthetic(256, 192, (0, 0), 3);
        let b = synthetic(256, 192, (0, 0), 4);
        let m = noise_model(&a, &b, (0, 0)).unwrap();
        // Sum of three uniforms scaled by 8: variance 3/12 * 64 = 16.
        assert!((m.b - 16.0).abs() < 4.0, "{m:?}");
        assert!(m.a < 0.5, "{m:?}");
    }

    #[test]
    fn merge_reduces_noise_and_keeps_signal() {
        let frames: Vec<SensorFrame> = (0..8)
            .map(|i| synthetic(256, 192, (2 * i, -2 * i), 10 + i as u32))
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
            .map(|i| synthetic(256, 192, (0, 0), 20 + i as u32))
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
}
