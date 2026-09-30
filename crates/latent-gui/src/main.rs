//! latent-gui — browse a folder of raws and develop them interactively.
//!
//! Usage: `latent-gui [DIR | FILE.ARW]` (default: the current directory).

mod app;
mod library;
mod thumbs;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use damascene_core::Rect;
use damascene_core::color::ColorPreferences;
use damascene_winit_wgpu::{HostConfig, run_host_app_with_config};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "latent_gui=info,latent_core=info".into()),
        )
        .init();

    let arg = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let arg = arg
        .canonicalize()
        .with_context(|| format!("{}: not found", arg.display()))?;
    let (dir, file) = if arg.is_dir() {
        (arg, None)
    } else if arg.is_file() {
        let dir = arg.parent().map(PathBuf::from).unwrap_or_default();
        (dir, Some(arg))
    } else {
        bail!("{}: not a file or directory", arg.display());
    };

    let wakeup = app::SharedWakeup::default();
    let app = app::IdevApp::new(dir, file, wakeup.clone())?;
    let config = HostConfig::default()
        .with_app_id("latent-gui")
        .with_color_preferences(ColorPreferences::hdr_extended())
        .with_external_wakeup(move |w| *wakeup.lock().unwrap() = Some(w));
    let viewport = Rect::new(0.0, 0.0, 1600.0, 1000.0);
    run_host_app_with_config("latent", viewport, app, config)
        .map_err(|e| anyhow::anyhow!("host error: {e}"))
}
