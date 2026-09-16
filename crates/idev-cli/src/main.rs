use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use idev_core::gpu::{Developer, Gpu, OutputTarget};
use idev_core::params::{
    Demosaic, DevelopParams, Highlights, LensParams, ToneCurve, WhiteBalance, sidecar_path,
};
use idev_core::sensor::{self, ExifSummary};
use idev_core::source::Source;
use idev_core::{WorkingImage, output};
use idev_core::{burst, corners, metadata};

#[derive(Parser)]
#[command(name = "idev", about = "Raw developer")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// scRGB linear f16
    Exr,
    /// sRGB 8-bit, or BT.2020 PQ 16-bit with --hdr
    Png,
    /// sRGB 8-bit
    Jpg,
    /// sRGB 10-bit, or BT.2020 PQ 10-bit with --hdr
    Avif,
}

impl Format {
    fn extension(self) -> &'static str {
        match self {
            Format::Exr => "exr",
            Format::Png => "png",
            Format::Avif => "avif",
            Format::Jpg => "jpg",
        }
    }

    fn from_path(p: &Path) -> Option<Format> {
        match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "exr" => Some(Format::Exr),
            "png" => Some(Format::Png),
            "jpg" | "jpeg" => Some(Format::Jpg),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum LensArg {
    /// Apply the camera's distortion, CA and vignetting tables.
    All,
    None,
}

#[derive(Clone, Copy, ValueEnum)]
enum DemosaicArg {
    Bilinear,
    Rcd,
}

#[derive(Clone, Copy, ValueEnum)]
enum HighlightsArg {
    Clip,
    Blend,
}

#[derive(Clone, Copy, ValueEnum)]
enum ToneArg {
    None,
    Sigmoid,
}

/// Parameter overrides shared by `develop` and `params`.
#[derive(clap::Args, Default)]
struct Overrides {
    /// Read parameters from this file instead of the raw's sidecar.
    #[arg(long)]
    params: Option<PathBuf>,
    /// Exposure compensation in stops.
    #[arg(long)]
    exposure: Option<f32>,
    /// White balance multipliers "R,B" (green = 1).
    #[arg(long)]
    wb: Option<String>,
    /// White balance as "kelvin" or "kelvin,tint".
    #[arg(long)]
    temp: Option<String>,
    /// As-shot white balance (clears a sidecar override).
    #[arg(long)]
    wb_as_shot: bool,
    #[arg(long)]
    demosaic: Option<DemosaicArg>,
    #[arg(long)]
    highlights: Option<HighlightsArg>,
    #[arg(long)]
    tone: Option<ToneArg>,
    /// Sigmoid contrast (slope at middle grey).
    #[arg(long)]
    contrast: Option<f32>,
    /// Sigmoid hue preservation, 0 (per channel) to 1 (luminance ratio).
    #[arg(long)]
    hue_preservation: Option<f32>,
    /// Sigmoid white: scene value (1.0 = diffuse white) that maps to the
    /// output peak.
    #[arg(long)]
    white: Option<f32>,
    /// Lens corrections from the camera's tables.
    #[arg(long)]
    lens: Option<LensArg>,
    /// Straighten: degrees, positive turns the image clockwise.
    #[arg(long, allow_negative_numbers = true)]
    rotate: Option<f32>,
    /// Crop as fractions "x,y,w,h" of the oriented frame, or "auto"
    /// (largest same-aspect rectangle inside the straightened content).
    #[arg(long)]
    crop: Option<String>,
    /// Sharpen: "amount[,radius[,threshold]]" (unsharp mask on
    /// luminance at output resolution; amount 0 disables).
    #[arg(long)]
    sharpen: Option<String>,
    /// Noise reduction: "chroma[,luminance[,impulse]]", each 0 (off)
    /// to 1. Impulse is the hot / cold site filter on the sensor plane.
    #[arg(long)]
    denoise: Option<String>,
}

impl Overrides {
    fn apply(&self, p: &mut DevelopParams) -> Result<()> {
        if let Some(ev) = self.exposure {
            p.exposure_ev = ev;
        }
        if self.wb_as_shot {
            p.white_balance = WhiteBalance::AsShot;
        }
        if let Some(s) = &self.wb {
            let parts = parse_floats(s).context("parse --wb")?;
            let [red, blue] = parts.as_slice() else {
                bail!("--wb expects \"R,B\"")
            };
            p.white_balance = WhiteBalance::Multipliers {
                red: *red,
                blue: *blue,
            };
        }
        if let Some(s) = &self.temp {
            let parts = parse_floats(s).context("parse --temp")?;
            p.white_balance = match parts.as_slice() {
                [k] => WhiteBalance::Temperature {
                    kelvin: *k,
                    tint: 0.0,
                },
                [k, t] => WhiteBalance::Temperature {
                    kelvin: *k,
                    tint: *t,
                },
                _ => bail!("--temp expects \"K\" or \"K,tint\""),
            };
        }
        if let Some(l) = self.lens {
            p.lens = match l {
                LensArg::All => LensParams::default(),
                LensArg::None => LensParams::NONE,
            };
        }
        if let Some(deg) = self.rotate {
            p.geometry.angle_deg = deg;
        }
        if let Some(spec) = &self.denoise {
            let v: Vec<f32> = spec
                .split(',')
                .map(|x| x.trim().parse::<f32>())
                .collect::<Result<_, _>>()
                .with_context(|| {
                    format!("--denoise {spec}: expected chroma[,luminance[,impulse]]")
                })?;
            anyhow::ensure!(
                (1..=3).contains(&v.len()),
                "--denoise {spec}: one to three values"
            );
            p.denoise.chroma = v[0].clamp(0.0, 1.0);
            p.denoise.luminance = v.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            p.denoise.impulse = v.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        }
        if let Some(spec) = &self.sharpen {
            let v: Vec<f32> = spec
                .split(',')
                .map(|x| x.trim().parse::<f32>())
                .collect::<Result<_, _>>()
                .with_context(|| {
                    format!("--sharpen {spec}: expected amount[,radius[,threshold]]")
                })?;
            anyhow::ensure!(
                (1..=3).contains(&v.len()),
                "--sharpen {spec}: one to three values"
            );
            p.sharpen.amount = v[0].max(0.0);
            if let Some(r) = v.get(1) {
                p.sharpen.radius = r.max(0.0);
            }
            if let Some(t) = v.get(2) {
                p.sharpen.threshold = t.max(0.0);
            }
        }
        if let Some(c) = &self.crop {
            p.geometry.crop = if c == "auto" {
                None
            } else {
                let v: Vec<f32> = c
                    .split(',')
                    .map(|x| x.trim().parse::<f32>())
                    .collect::<Result<_, _>>()
                    .with_context(|| format!("--crop {c}: expected x,y,w,h fractions"))?;
                anyhow::ensure!(v.len() == 4, "--crop {c}: expected four values");
                anyhow::ensure!(
                    v[2] > 0.0 && v[3] > 0.0 && v.iter().all(|x| (0.0..=1.0).contains(x)),
                    "--crop {c}: fractions must lie in 0..=1 with positive size"
                );
                Some([v[0], v[1], v[2], v[3]])
            };
        }
        if let Some(d) = self.demosaic {
            p.demosaic = match d {
                DemosaicArg::Bilinear => Demosaic::Bilinear,
                DemosaicArg::Rcd => Demosaic::Rcd,
            };
        }
        if let Some(h) = self.highlights {
            p.highlights = match h {
                HighlightsArg::Clip => Highlights::Clip,
                HighlightsArg::Blend => Highlights::Blend,
            };
        }
        if let Some(t) = self.tone {
            p.tone = match t {
                ToneArg::None => ToneCurve::None,
                ToneArg::Sigmoid => match p.tone {
                    ToneCurve::Sigmoid { .. } => p.tone,
                    ToneCurve::None => ToneCurve::default(),
                },
            };
        }
        if self.contrast.is_some() || self.hue_preservation.is_some() || self.white.is_some() {
            let (mut c, mut h, mut w) = match p.tone {
                ToneCurve::Sigmoid {
                    contrast,
                    hue_preservation,
                    white,
                } => (contrast, hue_preservation, white),
                ToneCurve::None => match ToneCurve::default() {
                    ToneCurve::Sigmoid {
                        contrast,
                        hue_preservation,
                        white,
                    } => (contrast, hue_preservation, white),
                    ToneCurve::None => unreachable!(),
                },
            };
            if let Some(v) = self.contrast {
                c = v;
            }
            if let Some(v) = self.hue_preservation {
                h = v;
            }
            if let Some(v) = self.white {
                w = v;
            }
            p.tone = ToneCurve::Sigmoid {
                contrast: c,
                hue_preservation: h,
                white: w,
            };
        }
        Ok(())
    }

    /// Sidecar (or --params file) plus overrides.
    fn resolve(&self, raw: &Path) -> Result<DevelopParams> {
        let mut p = match &self.params {
            Some(f) => DevelopParams::load(f)?,
            None => DevelopParams::for_raw(raw)?,
        };
        self.apply(&mut p)?;
        Ok(p)
    }
}

fn parse_floats(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|p| p.trim().parse::<f32>().map_err(Into::into))
        .collect()
}

#[derive(Subcommand)]
enum Cmd {
    /// Print what the decoder learned about a raw or image file.
    Inspect {
        file: PathBuf,
        /// Also list every metadata field (EXIF, GPS, makernote).
        #[arg(long)]
        all: bool,
    },
    /// Develop raw files. One input with -o, or any number with --out-dir.
    Develop {
        files: Vec<PathBuf>,
        /// Output file (single input; format from the extension).
        #[arg(short, long, conflicts_with = "out_dir")]
        out: Option<PathBuf>,
        /// Output directory (batch; names follow the inputs).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Output format for --out-dir (default png).
        #[arg(long, value_enum)]
        format: Option<Format>,
        /// HDR output: tone-map for a display of this peak in nits and
        /// write PNG as BT.2020 PQ 16-bit.
        #[arg(long)]
        hdr: Option<f32>,
        /// JPEG / AVIF quality.
        #[arg(long, default_value_t = 92)]
        quality: u8,
        /// Resize so the long edge is this many pixels (Lanczos-3 in
        /// linear light, after crop and straighten).
        #[arg(long)]
        size: Option<u32>,
        /// Overwrite existing outputs.
        #[arg(long)]
        force: bool,
        /// Write the resolved parameters to each raw's sidecar.
        #[arg(long)]
        save_params: bool,
        /// Treat the inputs as one burst: align them to the first,
        /// merge on the sensor plane, develop the result with the first
        /// frame's parameters. Needs -o.
        #[arg(long, requires = "out", conflicts_with = "superres")]
        merge: bool,
        /// Treat the inputs as one burst and reconstruct camera RGB at
        /// this scale (1 or 2) by kernel regression over all frames,
        /// replacing demosaic. Needs -o.
        #[arg(long, requires = "out", value_name = "SCALE")]
        superres: Option<u32>,
        /// Kernel width for --superres, in sensor pixels.
        #[arg(long, default_value_t = 0.7)]
        kernel: f32,
        #[command(flatten)]
        overrides: Overrides,
    },
    /// List the continuous-drive bursts among raw files (or in a
    /// directory), or with --stats align and merge the given frames as
    /// one burst and report shifts, weights and the noise reduction.
    Burst {
        inputs: Vec<PathBuf>,
        /// Align and merge the inputs as one burst and print statistics.
        #[arg(long)]
        stats: bool,
        /// Score the alignment models against checkerboard corners
        /// detected in every frame (the inputs must show a board).
        #[arg(long, conflicts_with = "stats")]
        score: bool,
        /// With --stats: write the merged sensor plane as 16-bit PGM.
        #[arg(long, requires = "stats")]
        pgm: Option<PathBuf>,
    },
    /// Dump the raw sensor plane as 16-bit PGM (for decoder cross-checks).
    DumpBayer {
        file: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Print parameters as TOML: defaults, or a raw's sidecar plus overrides.
    Params {
        file: Option<PathBuf>,
        #[command(flatten)]
        overrides: Overrides,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    match Cli::parse().cmd {
        Cmd::Inspect { file, all } => inspect(&file, all),
        Cmd::Develop {
            files,
            out,
            out_dir,
            format,
            hdr,
            quality,
            size,
            force,
            save_params,
            merge,
            superres,
            kernel,
            overrides,
        } => develop(
            &files,
            out.as_deref(),
            out_dir.as_deref(),
            format,
            hdr,
            quality,
            size,
            force,
            save_params,
            Burst::from_flags(merge, superres, kernel),
            &overrides,
        ),
        Cmd::Burst {
            inputs,
            stats,
            score,
            pgm,
        } => {
            if stats {
                burst_stats(&inputs, pgm.as_deref())
            } else if score {
                burst_score(&inputs)
            } else {
                list_bursts(&inputs)
            }
        }
        Cmd::DumpBayer { file, out } => {
            let Source::Raw(frame) = Source::load(&file)? else {
                bail!("{}: not a camera raw", file.display());
            };
            idev_formats::write_pgm16(&out, frame.width, frame.height, &frame.data)
        }
        Cmd::Params { file, overrides } => {
            let p = match file {
                Some(f) => overrides.resolve(&f)?,
                None => {
                    let mut p = DevelopParams::default();
                    overrides.apply(&mut p)?;
                    p
                }
            };
            print!("{}", toml::to_string_pretty(&p)?);
            Ok(())
        }
    }
}

fn inspect(file: &Path, all: bool) -> Result<()> {
    let t = Instant::now();
    let source = Source::load(file)?;
    println!("{}  ({:.2?})", file.display(), t.elapsed());
    let e = source.exif();
    if !e.make.is_empty() || !e.model.is_empty() {
        println!("camera       {} {}", e.make, e.model);
    }
    if let Some(l) = &e.lens {
        println!("lens         {l}");
    }
    let exp = e
        .exposure_time
        .map(|(n, d)| format!("{n}/{d} s"))
        .unwrap_or_default();
    let fnum = e.f_number.map(|v| format!("f/{v:.1}")).unwrap_or_default();
    let iso = e.iso.map(|v| format!("ISO {v}")).unwrap_or_default();
    let fl = e
        .focal_length_mm
        .map(|v| format!("{v:.0} mm"))
        .unwrap_or_default();
    if !(exp.is_empty() && fnum.is_empty() && iso.is_empty() && fl.is_empty()) {
        println!("exposure     {exp}  {fnum}  {iso}  {fl}");
    }
    if let Some(d) = &e.date_time_original {
        println!("taken        {d}");
    }
    let f = match &source {
        Source::Raw(f) => f,
        Source::Rgb(r) => &r.frame,
        Source::Image(i) => {
            println!("image        {} x {}  {}", i.width, i.height, i.encoding);
            println!("orientation  {:?}", i.orientation);
            let (max, mean) = output::stats(&WorkingImage {
                width: i.width,
                height: i.height,
                rgba: i.rgba.clone(),
            });
            println!("max rgb      {:?}  mean luminance {:.4}", max, mean);
            if all {
                print_metadata(&source);
            }
            return Ok(());
        }
    };
    println!("raster       {} x {}", f.width, f.height);
    println!(
        "crop         {} x {} at ({}, {})",
        f.crop.width, f.crop.height, f.crop.x, f.crop.y
    );
    println!("cfa          {:?}", f.cfa);
    println!("black        {:?}", f.black);
    println!("white        {:?}", f.white);
    println!("wb as shot   {:?}", f.wb_as_shot);
    if let Some(w) = idev_core::whitepoint::white_from_neutral(&f.matrices, f.neutral_as_shot()) {
        println!(
            "white        {:.0} K  tint {:+.1}  xy ({:.4}, {:.4})",
            w.temperature.kelvin, w.temperature.tint, w.xy.x, w.xy.y
        );
    }
    println!("orientation  {:?}", f.orientation);
    match &f.lens {
        Some(l) => {
            let last = |v: &[f32]| v.last().copied().unwrap_or(1.0);
            println!(
                "lens corr    {} knots; corner: distortion {:.4}, CA r {:+.2} px b {:+.2} px, vignetting x{:.3}",
                l.knots.len(),
                last(&l.distortion),
                (last(&l.ca_red) - 1.0) * (f.crop.width as f32).hypot(f.crop.height as f32) * 0.5,
                (last(&l.ca_blue) - 1.0) * (f.crop.width as f32).hypot(f.crop.height as f32) * 0.5,
                last(&l.vignetting)
            );
        }
        None => println!("lens corr    none"),
    }
    for m in &f.matrices {
        println!("xyz_to_cam   {:?}", m.illuminant);
        for row in m.xyz_to_cam {
            println!(
                "             [{:8.4} {:8.4} {:8.4}]",
                row[0], row[1], row[2]
            );
        }
    }
    let (min, max) = f
        .data
        .iter()
        .fold((u16::MAX, 0u16), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    println!("data range   {min} ..= {max}");
    let side = sidecar_path(file);
    if side.exists() {
        println!("sidecar      {}", side.display());
    }
    if all {
        print_metadata(&source);
    }
    Ok(())
}

fn print_metadata(source: &Source) {
    let meta = source.metadata();
    if meta.is_empty() {
        println!("\nno metadata");
        return;
    }
    for group in &meta.groups {
        println!("\n[{}]", group.name);
        for field in &group.fields {
            println!("{:<32} {}", field.name, field.value);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn develop(
    files: &[PathBuf],
    out: Option<&Path>,
    out_dir: Option<&Path>,
    format: Option<Format>,
    hdr: Option<f32>,
    quality: u8,
    size: Option<u32>,
    force: bool,
    save_params: bool,
    burst_mode: Option<Burst>,
    overrides: &Overrides,
) -> Result<()> {
    if files.is_empty() {
        bail!("no input files");
    }
    // Resolve each input's output path and format up front.
    let jobs: Vec<(PathBuf, PathBuf, Format)> = match (out, out_dir) {
        (Some(o), None) => {
            if files.len() != 1 && burst_mode.is_none() {
                bail!(
                    "-o takes exactly one input; use --out-dir for several, --merge or --superres"
                );
            }
            let fmt = format
                .or_else(|| Format::from_path(o))
                .context("cannot tell the output format from the extension; pass --format")?;
            vec![(files[0].clone(), o.to_path_buf(), fmt)]
        }
        (None, Some(dir)) => {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            let fmt = format.unwrap_or(Format::Png);
            files
                .iter()
                .map(|f| {
                    let stem = f.file_stem().map(|s| s.to_os_string()).unwrap_or_default();
                    let mut name = stem;
                    name.push(".");
                    name.push(fmt.extension());
                    (f.clone(), dir.join(name), fmt)
                })
                .collect()
        }
        (None, None) => bail!("pass -o FILE or --out-dir DIR"),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
    };

    let target = match hdr {
        Some(nits) => OutputTarget::hdr_nits(nits),
        None => OutputTarget::SDR,
    };

    let t = Instant::now();
    let gpu = Gpu::headless()?;
    let dev = Developer::new(&gpu);
    tracing::info!("gpu ready in {:.0?}", t.elapsed());

    let mut failures = 0usize;
    for (input, output_path, fmt) in &jobs {
        if output_path.exists() && !force {
            println!(
                "{}  exists, skipped (--force to overwrite)",
                output_path.display()
            );
            continue;
        }
        let load = || -> Result<(Source, String)> {
            let name = format!("{}+{}", file_name(input), files.len() - 1);
            match burst_mode {
                Some(Burst::Merge) => {
                    let (merged, shifts) = burst::merge_files(files)?;
                    for ((f, s), (o, w)) in files
                        .iter()
                        .zip(&shifts)
                        .zip(merged.offsets.iter().zip(&merged.mean_weight))
                    {
                        println!("  {}  {}", f.display(), burst::describe(s, *o, *w));
                    }
                    Ok((Source::Raw(Box::new(merged.frame)), name))
                }
                Some(Burst::Superres { scale, kernel }) => {
                    let (rgb, transforms) = burst::reconstruct_files(files, scale, kernel)?;
                    for ((f, t), w) in files.iter().zip(&transforms).zip(&rgb.mean_weight) {
                        println!("  {}  {}", f.display(), burst::describe_transform(t, *w));
                    }
                    println!(
                        "  reconstructed {}x{} at {scale}x, kernel {kernel} px, {:.3}% uncovered",
                        rgb.width,
                        rgb.height,
                        rgb.uncovered * 100.0
                    );
                    Ok((Source::Rgb(Box::new(rgb)), name))
                }
                None => Ok((Source::load(input)?, file_name(input))),
            }
        };
        match develop_one(
            &gpu,
            &dev,
            input,
            load,
            output_path,
            *fmt,
            target,
            hdr,
            quality,
            size,
            save_params,
            overrides,
        ) {
            Ok(()) => {}
            Err(e) => {
                failures += 1;
                eprintln!("{}: {e:#}", input.display());
            }
        }
    }
    if failures > 0 {
        bail!("{failures} of {} inputs failed", jobs.len());
    }
    Ok(())
}

/// How `develop` combines several inputs into one image.
#[derive(Clone, Copy)]
enum Burst {
    /// Step 2: robust mean on the sensor plane, then the usual demosaic.
    Merge,
    /// Step 3: kernel regression onto the output grid.
    Superres { scale: u32, kernel: f32 },
}

impl Burst {
    fn from_flags(merge: bool, superres: Option<u32>, kernel: f32) -> Option<Burst> {
        match (merge, superres) {
            (true, _) => Some(Burst::Merge),
            (false, Some(scale)) => Some(Burst::Superres { scale, kernel }),
            (false, None) => None,
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Develop one source. `input` names the sidecar and the report line;
/// `load` produces the source (a file, or a merged burst).
#[allow(clippy::too_many_arguments)]
fn develop_one(
    gpu: &Gpu,
    dev: &Developer,
    input: &Path,
    load: impl FnOnce() -> Result<(Source, String)>,
    output_path: &Path,
    fmt: Format,
    target: OutputTarget,
    hdr: Option<f32>,
    quality: u8,
    size: Option<u32>,
    save_params: bool,
    overrides: &Overrides,
) -> Result<()> {
    let params = overrides.resolve(input)?;
    let t0 = Instant::now();
    let (source, name) = load()?;
    let t1 = Instant::now();
    let img: WorkingImage = dev.develop_sized(gpu, &source, &params, target, size)?;
    let t2 = Instant::now();
    let note = metadata::develop_note(&name, &params);
    let exif = metadata::exif_for_output(
        &source.metadata().tags,
        &metadata::Output {
            width: img.width,
            height: img.height,
            srgb: hdr.is_none(),
            note: &note,
        },
    )?;
    encode(&img, output_path, fmt, hdr, quality, &exif)?;
    let t3 = Instant::now();
    if save_params {
        params.save(&sidecar_path(input))?;
    }
    println!(
        "{} -> {}  {}x{}  load {:.0?}  develop {:.0?}  encode {:.0?}",
        input.display(),
        output_path.display(),
        img.width,
        img.height,
        t1 - t0,
        t2 - t1,
        t3 - t2
    );
    Ok(())
}

/// Raw files among the inputs, directories expanded one level, sorted.
fn raw_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for p in inputs {
        if p.is_dir() {
            for entry in std::fs::read_dir(p).with_context(|| format!("read {}", p.display()))? {
                let f = entry?.path();
                if f.is_file() && idev_core::source::is_raw(&f) {
                    files.push(f);
                }
            }
        } else if idev_core::source::is_raw(p) {
            files.push(p.clone());
        } else {
            eprintln!("{}: not a camera raw, skipped", p.display());
        }
    }
    files.sort();
    Ok(files)
}

fn list_bursts(inputs: &[PathBuf]) -> Result<()> {
    let files = raw_inputs(inputs)?;
    if files.is_empty() {
        bail!("no raw files");
    }
    let t = Instant::now();
    let summaries: Vec<Option<ExifSummary>> = files
        .iter()
        .map(|f| match sensor::summary(f) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("{}: {e:#}", f.display());
                None
            }
        })
        .collect();
    let refs: Vec<Option<&ExifSummary>> = summaries.iter().map(|s| s.as_ref()).collect();
    let groups = burst::group(&refs);
    let bursts: Vec<&Vec<usize>> = groups.iter().filter(|g| g.len() > 1).collect();
    println!(
        "{} raw files, {} bursts, {} single frames  ({:.1?})",
        files.len(),
        bursts.len(),
        groups.len() - bursts.len(),
        t.elapsed()
    );
    for g in bursts {
        let first = summaries[g[0]]
            .as_ref()
            .expect("grouped frames have summaries");
        let last = summaries[*g.last().unwrap()]
            .as_ref()
            .expect("grouped frames have summaries");
        let span = match (first.timestamp_secs(), last.timestamp_secs()) {
            (Some(a), Some(b)) => format!("{:.1} s", b - a),
            _ => "?".into(),
        };
        let exposure = first
            .exposure_time
            .map(|(n, d)| format!("{n}/{d} s"))
            .unwrap_or_default();
        println!(
            "{}..{}  {:2} frames  {}  {}  f/{}  ISO {}  {} mm  span {}",
            file_name(&files[g[0]]),
            file_name(&files[*g.last().unwrap()]),
            g.len(),
            first.date().unwrap_or_default(),
            exposure,
            first
                .f_number
                .map(|v| format!("{v:.1}"))
                .unwrap_or_default(),
            first.iso.map(|v| v.to_string()).unwrap_or_default(),
            first
                .focal_length_mm
                .map(|v| format!("{v:.0}"))
                .unwrap_or_default(),
            span
        );
    }
    Ok(())
}

fn burst_stats(inputs: &[PathBuf], pgm: Option<&Path>) -> Result<()> {
    let files = raw_inputs(inputs)?;
    if files.len() < 2 {
        bail!("--stats needs at least two raw frames");
    }
    let t = Instant::now();
    let frames = burst::load_all(&files)?;
    let t_load = t.elapsed();
    let refs: Vec<&idev_core::sensor::SensorFrame> = frames.iter().collect();
    let t = Instant::now();
    let shifts = burst::align_all(&refs)?;
    let t_align = t.elapsed();
    let t = Instant::now();
    let merged = burst::merge(&refs, &shifts)?;
    let t_merge = t.elapsed();
    let n = files.len();
    println!("{n} frames  load {t_load:.1?}  align {t_align:.1?}  merge {t_merge:.1?}");
    for (i, f) in files.iter().enumerate() {
        println!(
            "  {}  {}",
            file_name(f),
            burst::describe(&shifts[i], merged.offsets[i], merged.mean_weight[i])
        );
    }
    let t = Instant::now();
    let transforms = burst::similarity_all(&refs, &shifts)?;
    println!("similarity fits ({:.1?}):", t.elapsed());
    for (f, tr) in files.iter().zip(&transforms) {
        println!("  {}  {}", file_name(f), burst::describe_transform(tr, 1.0));
    }
    let single = merged.noise;
    println!(
        "single-frame noise model  σ² = {:.4}·s + {:.2}   σ(0) {:.2}  σ(100) {:.2}  σ(1000) {:.2} DN",
        single.a,
        single.b,
        single.sigma(0.0),
        single.sigma(100.0),
        single.sigma(1000.0)
    );
    let below = |f: &idev_core::sensor::SensorFrame| {
        let c = f.crop;
        let mut n = 0u64;
        for y in c.y..c.y + c.height {
            let row =
                &f.data[y as usize * f.width as usize..][c.x as usize..(c.x + c.width) as usize];
            for (i, &v) in row.iter().enumerate() {
                if (v as f32) < f.black[((y & 1) * 2 + ((c.x + i as u32) & 1)) as usize] {
                    n += 1;
                }
            }
        }
        n as f32 / (c.width * c.height) as f32
    };
    let before = burst::dark_sigma(&frames[0]);
    let after = burst::dark_sigma(&merged.frame);
    println!(
        "below-black σ  reference {before:.2} ({:.1}% of sites)  merged {after:.2} ({:.1}%)  ratio {:.2}  (√N = {:.2}; only meaningful with a real dark tail)",
        below(&frames[0]) * 100.0,
        below(&merged.frame) * 100.0,
        before / after.max(1e-6),
        (n as f32).sqrt()
    );
    // Halves merged separately: the model between them is the noise of
    // an N/2-frame merge, at every signal level.
    let evens: Vec<&idev_core::sensor::SensorFrame> = refs.iter().step_by(2).copied().collect();
    let odds: Vec<&idev_core::sensor::SensorFrame> =
        refs.iter().skip(1).step_by(2).copied().collect();
    let even_shifts: Vec<burst::Shift> = shifts.iter().step_by(2).copied().collect();
    let mut odd_shifts: Vec<burst::Shift> = shifts.iter().skip(1).step_by(2).copied().collect();
    // Odd frames merge onto frame 1; express their shifts relative to it.
    let base = odd_shifts[0];
    for s in odd_shifts.iter_mut() {
        s.dx -= base.dx;
        s.dy -= base.dy;
    }
    let half_a = burst::merge(&evens, &even_shifts)?;
    let half_b = burst::merge(&odds, &odd_shifts)?;
    // Both halves live on their own reference's grid: frame 1 sits at
    // `base` relative to frame 0.
    if let Some(half) = burst::noise_model(&half_a.frame, &half_b.frame, base.even()) {
        let k = (n as f32 / 2.0).sqrt();
        println!(
            "half-split model (N/2 = {} frames each)  σ² = {:.4}·s + {:.2}   σ(0) {:.2}  σ(100) {:.2}  σ(1000) {:.2} DN",
            evens.len(),
            half.a,
            half.b,
            half.sigma(0.0),
            half.sigma(100.0),
            half.sigma(1000.0)
        );
        println!(
            "single / half ratio  at 0: {:.2}  at 100: {:.2}  at 1000: {:.2}   (√(N/2) = {k:.2})",
            single.sigma(0.0) / half.sigma(0.0).max(1e-6),
            single.sigma(100.0) / half.sigma(100.0).max(1e-6),
            single.sigma(1000.0) / half.sigma(1000.0).max(1e-6)
        );
    } else {
        println!("half-split model: not enough samples");
    }
    if let Some(p) = pgm {
        idev_formats::write_pgm16(
            p,
            merged.frame.width,
            merged.frame.height,
            &merged.frame.data,
        )?;
        println!("merged plane -> {}", p.display());
    }
    Ok(())
}

/// Reference corners, by response, the strong-only score uses.
const STRONG_CORNERS: usize = 120;

/// A named alignment model: reference sensor coordinates to frame.
type Model<'a> = (&'a str, Box<dyn Fn(f32, f32) -> (f32, f32) + 'a>);

/// Alignment models against detected checkerboard corners: the global
/// shift as a translation, the coarse similarity, and its refinement.
fn burst_score(inputs: &[PathBuf]) -> Result<()> {
    let files = raw_inputs(inputs)?;
    if files.len() < 2 {
        bail!("--score needs at least two raw frames");
    }
    let t = Instant::now();
    let frames = burst::load_all(&files)?;
    let refs: Vec<&idev_core::sensor::SensorFrame> = frames.iter().collect();
    let mut corner_sets: Vec<Vec<corners::Corner>> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = refs
            .iter()
            .map(|f| s.spawn(move || corners::detect(f)))
            .collect();
        for h in handles {
            corner_sets.push(h.join().expect("corner detection"));
        }
    });
    let shifts = burst::align_all(&refs)?;
    let mut stages: Vec<Option<Result<(burst::Transform, burst::Transform)>>> =
        (0..refs.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        for ((slot, f), shift) in stages.iter_mut().zip(&refs).zip(&shifts).skip(1) {
            let reference = refs[0];
            s.spawn(move || {
                *slot = Some(burst::align_similarity_stages(reference, f, *shift));
            });
        }
    });
    println!(
        "{} frames, corners per frame: {}  ({:.1?})",
        files.len(),
        corner_sets
            .iter()
            .map(|c| c.len().to_string())
            .collect::<Vec<_>>()
            .join(" "),
        t.elapsed()
    );
    println!("residuals in sensor pixels over matched corners (rms / median / max):");
    let mut totals = [(0f64, 0usize); 5];
    for i in 1..files.len() {
        let pairs = corners::matches(&corner_sets[0], &corner_sets[i], shifts[i], 4.0);
        let (coarse, refined) = match stages[i].take().expect("filled") {
            Ok(v) => v,
            Err(e) => {
                println!("  {}: {e:#}", file_name(&files[i]));
                continue;
            }
        };
        let shift = shifts[i];
        let homography = corners::fit_homography(&pairs);
        let models: [Model; 4] = [
            (
                "translation",
                Box::new(move |x, y| (x + shift.dx, y + shift.dy)),
            ),
            (
                "coarse similarity",
                Box::new(move |x, y| coarse.apply(x, y)),
            ),
            (
                "refined similarity",
                Box::new(move |x, y| refined.apply(x, y)),
            ),
            (
                "homography (in-sample)",
                Box::new(move |x, y| match homography {
                    Some(h) => h.apply(x, y),
                    None => (f32::NAN, f32::NAN),
                }),
            ),
        ];
        println!(
            "  {}  {} pairs  shift ({:+.2}, {:+.2})  refined rotation {:+.4}° scale {:.5}",
            file_name(&files[i]),
            pairs.len(),
            shift.dx,
            shift.dy,
            refined.rotation_deg(),
            refined.scale()
        );
        // The strongest corners are the board's own; weaker ones come
        // from marker cells and are located less well.
        let strong: Vec<corners::Pair> = corners::matches(
            &corner_sets[0][..corner_sets[0].len().min(STRONG_CORNERS)],
            &corner_sets[i],
            shifts[i],
            4.0,
        );
        println!(
            "    {:<22} {:>24}   {:>24}",
            "",
            "all pairs",
            format!("strongest {}", strong.len())
        );
        for (k, (name, predict)) in models.iter().enumerate() {
            let sc = corners::score(&pairs, predict);
            let ss = corners::score(&strong, predict);
            println!(
                "    {name:<22} {:6.3} / {:6.3} / {:6.2}   {:6.3} / {:6.3} / {:6.2}",
                sc.rms, sc.median, sc.max, ss.rms, ss.median, ss.max
            );
            if ss.n > 0 {
                totals[k].0 += ss.rms as f64;
                totals[k].1 += 1;
            }
        }
        let local = corners::local_bound(&strong, |x, y| refined.apply(x, y), refs[0], 6, 4);
        println!(
            "    {:<22} {:>24}   {:6.3} (refined + per-cell translation, 6×4 cells)",
            "local bound", "", local
        );
        if local.is_finite() {
            totals[4].0 += local as f64;
            totals[4].1 += 1;
        }
        let grid = corners::residual_grid(&strong, |x, y| refined.apply(x, y), refs[0], 4, 3);
        println!("    refined residual field (mean dx,dy per cell, 4 columns × 3 rows):");
        for row in grid {
            let cells: Vec<String> = row
                .iter()
                .map(|c| match c {
                    Some((dx, dy, n)) => format!("({dx:+.2},{dy:+.2}) n{n:<3}"),
                    None => "      -          ".to_string(),
                })
                .collect();
            println!("      {}", cells.join("  "));
        }
    }
    println!("mean rms over frames (strongest corners):");
    for (k, name) in [
        "translation",
        "coarse similarity",
        "refined similarity",
        "homography (in-sample)",
        "local bound",
    ]
    .iter()
    .enumerate()
    {
        if totals[k].1 > 0 {
            println!("  {name:<22} {:.3}", totals[k].0 / totals[k].1 as f64);
        }
    }
    Ok(())
}

/// Encode with the EXIF block in every container that has a place for
/// it (EXR has none).
fn encode(
    img: &WorkingImage,
    path: &Path,
    fmt: Format,
    hdr: Option<f32>,
    quality: u8,
    exif: &[u8],
) -> Result<()> {
    let exif = Some(exif);
    match (fmt, hdr) {
        (Format::Exr, _) => {
            let rgb = output::scrgb_linear_f32(img);
            idev_formats::write_exr_rgb_f32(path, img.width, img.height, &rgb)
        }
        (Format::Png, None) => {
            idev_formats::write_png_srgb8(path, img.width, img.height, &output::srgb8(img), exif)
        }
        (Format::Png, Some(_)) => idev_formats::write_png_bt2020_pq16(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(img),
            exif,
        ),
        (Format::Jpg, None) => idev_formats::write_jpeg_srgb8(
            path,
            img.width,
            img.height,
            &output::srgb8(img),
            quality,
            exif,
        ),
        (Format::Jpg, Some(_)) => {
            bail!("JPEG has no HDR encoding; use png, avif or exr with --hdr")
        }
        (Format::Avif, None) => idev_formats::write_avif_srgb8(
            path,
            img.width,
            img.height,
            &output::srgb8(img),
            quality,
            exif,
        ),
        (Format::Avif, Some(_)) => idev_formats::write_avif_bt2020_pq10(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(img),
            quality,
            exif,
        ),
    }
}
