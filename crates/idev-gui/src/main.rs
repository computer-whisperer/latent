//! idev-gui — develop one raw file interactively.
//!
//! Usage: `idev-gui FILE.ARW`

mod app;

use std::path::PathBuf;

use anyhow::{Result, bail};
use damascene_core::Rect;
use damascene_core::color::ColorPreferences;
use damascene_winit_wgpu::{HostConfig, run_host_app_with_config};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "idev_gui=info,idev_core=info".into()),
        )
        .init();

    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        bail!("usage: idev-gui FILE.ARW");
    };
    if !path.is_file() {
        bail!("{}: not a file", path.display());
    }

    let wakeup = app::SharedWakeup::default();
    let app = app::DevelopApp::new(path, wakeup.clone())?;
    let config = HostConfig::default()
        .with_app_id("idev-gui")
        .with_color_preferences(ColorPreferences::hdr_extended())
        .with_external_wakeup(move |w| *wakeup.lock().unwrap() = Some(w));
    let viewport = Rect::new(0.0, 0.0, 1600.0, 1000.0);
    run_host_app_with_config("idev", viewport, app, config)
        .map_err(|e| anyhow::anyhow!("host error: {e}"))
}
