use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use idev_core::gpu::{Developer, Gpu};
use idev_core::params::{DevelopParams, WhiteBalance};
use idev_core::{output, sensor};

#[derive(Parser)]
#[command(name = "idev", about = "Raw developer")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print what the decoder learned about a raw file.
    Inspect { file: PathBuf },
    /// Develop a raw file into an output image (.exr or .png by extension).
    Develop {
        file: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        /// Exposure compensation in stops.
        #[arg(long, default_value_t = 0.0)]
        exposure: f32,
        /// White balance multipliers "R,B" (green = 1); default as-shot.
        #[arg(long)]
        wb: Option<String>,
    },
    /// Dump the raw sensor plane as 16-bit PGM (for decoder cross-checks).
    DumpBayer {
        file: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Print the default parameters as TOML.
    Params,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    match Cli::parse().cmd {
        Cmd::Inspect { file } => inspect(&file),
        Cmd::Develop {
            file,
            out,
            exposure,
            wb,
        } => develop(&file, &out, exposure, wb.as_deref()),
        Cmd::DumpBayer { file, out } => {
            let frame = sensor::unpack(&file)?;
            idev_formats::write_pgm16(&out, frame.width, frame.height, &frame.data)
        }
        Cmd::Params => {
            print!("{}", toml::to_string_pretty(&DevelopParams::default())?);
            Ok(())
        }
    }
}

fn inspect(file: &std::path::Path) -> Result<()> {
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
    println!("orientation  {:?}", f.orientation);
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
    Ok(())
}

fn develop(
    file: &std::path::Path,
    out: &std::path::Path,
    exposure: f32,
    wb: Option<&str>,
) -> Result<()> {
    let mut params = DevelopParams {
        exposure_ev: exposure,
        ..Default::default()
    };
    if let Some(s) = wb {
        let parts: Vec<f32> = s
            .split(',')
            .map(|p| p.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .context("parse --wb")?;
        if parts.len() != 2 {
            bail!("--wb expects \"R,B\"");
        }
        params.white_balance = WhiteBalance::Multipliers {
            red: parts[0],
            blue: parts[1],
        };
    }

    let t0 = Instant::now();
    let frame = sensor::unpack(file)?;
    let t1 = Instant::now();
    let gpu = Gpu::headless()?;
    let dev = Developer::new(&gpu);
    let t2 = Instant::now();
    let img = dev.develop(&gpu, &frame, &params)?;
    let t3 = Instant::now();

    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "exr" => {
            let rgb = output::scrgb_linear_f32(&img);
            idev_formats::write_exr_rgb_f32(out, img.width, img.height, &rgb)?;
        }
        "png" => {
            let rgb = output::srgb8(&img);
            idev_formats::write_png_srgb8(out, img.width, img.height, &rgb)?;
        }
        other => bail!("unsupported output extension {other:?} (use .exr or .png)"),
    }
    let t4 = Instant::now();

    let (max, mean) = output::stats(&img);
    println!(
        "{} -> {}  {}x{}  unpack {:.0?}  gpu init {:.0?}  develop {:.0?}  encode {:.0?}",
        file.display(),
        out.display(),
        img.width,
        img.height,
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3
    );
    println!("working max rgb {:?}  mean luminance {:.4}", max, mean);
    Ok(())
}
