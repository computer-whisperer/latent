use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use idev_core::gpu::{Developer, Gpu, OutputTarget};
use idev_core::params::{
    Demosaic, DevelopParams, Highlights, LensParams, ToneCurve, WhiteBalance, sidecar_path,
};
use idev_core::sensor::SensorFrame;
use idev_core::{WorkingImage, output, sensor};

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
    /// Print what the decoder learned about a raw file.
    Inspect { file: PathBuf },
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
        /// Overwrite existing outputs.
        #[arg(long)]
        force: bool,
        /// Write the resolved parameters to each raw's sidecar.
        #[arg(long)]
        save_params: bool,
        #[command(flatten)]
        overrides: Overrides,
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
        Cmd::Inspect { file } => inspect(&file),
        Cmd::Develop {
            files,
            out,
            out_dir,
            format,
            hdr,
            quality,
            force,
            save_params,
            overrides,
        } => develop(
            &files,
            out.as_deref(),
            out_dir.as_deref(),
            format,
            hdr,
            quality,
            force,
            save_params,
            &overrides,
        ),
        Cmd::DumpBayer { file, out } => {
            let frame = sensor::unpack(&file)?;
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

fn inspect(file: &Path) -> Result<()> {
    let t = Instant::now();
    let f = sensor::unpack(file)?;
    println!("{}  ({:.2?})", file.display(), t.elapsed());
    println!("camera       {} {}", f.exif.make, f.exif.model);
    if let Some(l) = &f.exif.lens {
        println!("lens         {l}");
    }
    let exp = f
        .exif
        .exposure_time
        .map(|(n, d)| format!("{n}/{d} s"))
        .unwrap_or_default();
    let fnum = f
        .exif
        .f_number
        .map(|v| format!("f/{v:.1}"))
        .unwrap_or_default();
    let iso = f.exif.iso.map(|v| format!("ISO {v}")).unwrap_or_default();
    let fl = f
        .exif
        .focal_length_mm
        .map(|v| format!("{v:.0} mm"))
        .unwrap_or_default();
    println!("exposure     {exp}  {fnum}  {iso}  {fl}");
    if let Some(d) = &f.exif.date_time_original {
        println!("taken        {d}");
    }
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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn develop(
    files: &[PathBuf],
    out: Option<&Path>,
    out_dir: Option<&Path>,
    format: Option<Format>,
    hdr: Option<f32>,
    quality: u8,
    force: bool,
    save_params: bool,
    overrides: &Overrides,
) -> Result<()> {
    if files.is_empty() {
        bail!("no input files");
    }
    // Resolve each input's output path and format up front.
    let jobs: Vec<(PathBuf, PathBuf, Format)> = match (out, out_dir) {
        (Some(o), None) => {
            if files.len() != 1 {
                bail!("-o takes exactly one input; use --out-dir for several");
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
        match develop_one(
            &gpu,
            &dev,
            input,
            output_path,
            *fmt,
            target,
            hdr,
            quality,
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

#[allow(clippy::too_many_arguments)]
fn develop_one(
    gpu: &Gpu,
    dev: &Developer,
    input: &Path,
    output_path: &Path,
    fmt: Format,
    target: OutputTarget,
    hdr: Option<f32>,
    quality: u8,
    save_params: bool,
    overrides: &Overrides,
) -> Result<()> {
    let params = overrides.resolve(input)?;
    let t0 = Instant::now();
    let frame: SensorFrame = sensor::unpack(input)?;
    let t1 = Instant::now();
    let img: WorkingImage = dev.develop(gpu, &frame, &params, target)?;
    let t2 = Instant::now();
    let img = output::oriented(&img, frame.orientation);
    encode(&img, output_path, fmt, hdr, quality)?;
    let t3 = Instant::now();
    if save_params {
        params.save(&sidecar_path(input))?;
    }
    println!(
        "{} -> {}  {}x{}  unpack {:.0?}  develop {:.0?}  encode {:.0?}",
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

fn encode(
    img: &WorkingImage,
    path: &Path,
    fmt: Format,
    hdr: Option<f32>,
    quality: u8,
) -> Result<()> {
    match (fmt, hdr) {
        (Format::Exr, _) => {
            let rgb = output::scrgb_linear_f32(img);
            idev_formats::write_exr_rgb_f32(path, img.width, img.height, &rgb)
        }
        (Format::Png, None) => {
            idev_formats::write_png_srgb8(path, img.width, img.height, &output::srgb8(img))
        }
        (Format::Png, Some(_)) => idev_formats::write_png_bt2020_pq16(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(img),
        ),
        (Format::Jpg, None) => idev_formats::write_jpeg_srgb8(
            path,
            img.width,
            img.height,
            &output::srgb8(img),
            quality,
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
        ),
        (Format::Avif, Some(_)) => idev_formats::write_avif_bt2020_pq10(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(img),
            quality,
        ),
    }
}
