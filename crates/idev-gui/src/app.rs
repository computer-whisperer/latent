//! The develop window: one raw, a pan/zoom preview, a control sidebar.
//!
//! GPU work happens in `before_paint` on the host's device: the
//! demosaiced camera RGB stays resident (`Prepared`) and the colour pass
//! re-renders into the preview texture whenever the parameters, the
//! display peak, or the preview scale changed. The preview scale follows
//! the viewport zoom so a fit-to-window preview costs a fraction of a
//! full-resolution pass; at 1:1 the scale is 1. Exports run on their own
//! thread with a headless device so the window never stalls.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use damascene_core::prelude::*;
use damascene_core::scroll;
use damascene_core::surface::{AppTexture, SurfaceAlpha};
use damascene_core::toast::ToastSpec;
use damascene_core::viewport::{FitPolicy, ViewportBehavior, ViewportRequest};
use damascene_core::widgets::number_scrubber::{self, ScrubDrag, ScrubberOpts};
use damascene_core::widgets::tabs;
use damascene_core::{BuildCx, EventCx, KeyChord, UiEvent};
use damascene_winit_wgpu::{Wakeup, WinitWgpuApp};
use idev_core::gpu::{
    Developer, DisplayPrimaries, Gpu, Histogram, HistogramReadback, OutputTarget, Prepared,
};
use idev_core::params::{
    DEFAULT_WHITE, Demosaic, DevelopParams, Highlights, ToneCurve, WhiteBalance, sidecar_path,
};
use idev_core::sensor::{self, SensorFrame};
use idev_core::whitepoint::{self, Temperature};
use idev_core::{WorkingImage, output};

/// The host's redraw handle, filled in once the event loop exists.
pub type SharedWakeup = Arc<Mutex<Option<Wakeup>>>;

fn wake(w: &SharedWakeup) {
    if let Some(w) = w.lock().unwrap().as_ref() {
        w.wake();
    }
}

/// Sidecar writes are coalesced: the writer thread waits this long after
/// the last change before writing, so a slider drag produces one write,
/// not hundreds. (The raw may live on a network mount where each write
/// costs tens of milliseconds; doing that on the UI thread per drag
/// event starved the Wayland socket and the compositor dropped us.)
const SIDECAR_DEBOUNCE: Duration = Duration::from_millis(250);

/// Debounced, latest-wins sidecar writer on its own thread.
struct SidecarWriter {
    tx: Option<Sender<DevelopParams>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SidecarWriter {
    fn spawn(path: PathBuf) -> SidecarWriter {
        let (tx, rx) = channel::<DevelopParams>();
        let thread = std::thread::Builder::new()
            .name("idev-sidecar".into())
            .spawn(move || {
                while let Ok(mut latest) = rx.recv() {
                    // Absorb further changes until the stream goes quiet.
                    loop {
                        match rx.recv_timeout(SIDECAR_DEBOUNCE) {
                            Ok(p) => latest = p,
                            Err(RecvTimeoutError::Timeout) => break,
                            Err(RecvTimeoutError::Disconnected) => {
                                write(&path, &latest);
                                return;
                            }
                        }
                    }
                    write(&path, &latest);
                }
            })
            .expect("spawn sidecar writer");
        fn write(path: &Path, params: &DevelopParams) {
            if let Err(e) = params.save(path) {
                tracing::warn!("sidecar: {e:#}");
            }
        }
        SidecarWriter {
            tx: Some(tx),
            thread: Some(thread),
        }
    }

    fn send(&self, params: &DevelopParams) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(params.clone());
        }
    }
}

impl Drop for SidecarWriter {
    /// Closing the channel makes the thread write whatever is pending
    /// and exit; joining it keeps the last change from being lost on
    /// window close.
    fn drop(&mut self) {
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Peak nits a "HDR PNG" export is mastered for.
const HDR_EXPORT_NITS: f32 = 1000.0;
const SIDEBAR_WIDTH: f32 = 320.0;
const FIT_PADDING: f32 = 8.0;
const TABS_WIDTH: f32 = 170.0;
const HISTOGRAM_HEIGHT: f32 = 90.0;

struct GpuState {
    gpu: Gpu,
    dev: Developer,
    prepared: Option<Prepared>,
    preview: Option<Preview>,
    /// What the preview texture currently shows.
    rendered: Option<(DevelopParams, f32, u32)>,
    /// Histogram of the preview, in flight.
    histogram_pending: Option<HistogramReadback>,
}

struct Preview {
    scale: u32,
    texture: AppTexture,
    view: wgpu::TextureView,
}

enum ExportKind {
    Jpeg,
    Png,
    HdrPng,
    Exr,
}

impl ExportKind {
    fn extension(&self) -> &'static str {
        match self {
            ExportKind::Jpeg => "jpg",
            ExportKind::Png | ExportKind::HdrPng => "png",
            ExportKind::Exr => "exr",
        }
    }
}

pub struct DevelopApp {
    path: PathBuf,
    name: String,
    frame: Option<Arc<SensorFrame>>,
    load_rx: Receiver<Result<SensorFrame>>,
    load_error: Option<String>,
    params: DevelopParams,
    sidecar: SidecarWriter,
    gpu: Option<GpuState>,
    wakeup: SharedWakeup,
    toasts: Vec<ToastSpec>,
    viewport_requests: Vec<ViewportRequest>,
    export_rx: Vec<Receiver<Result<PathBuf>>>,
    exports_running: usize,
    sdr_preview: bool,
    histogram: Option<Histogram>,

    // Read during build (immutable), consumed in before_paint.
    last_zoom: Cell<f32>,
    scale_factor: Cell<f32>,
    display_peak: Cell<f32>,
    hdr_output: Cell<bool>,

    // Control scratch state.
    exposure: Scrub,
    contrast: Scrub,
    hue: Scrub,
    white: Scrub,
    kelvin: Scrub,
    tint: Scrub,
}

/// A number scrubber's text plus drag anchor.
#[derive(Default)]
struct Scrub {
    text: String,
    drag: ScrubDrag,
}

impl Scrub {
    fn set(&mut self, v: f32, decimals: usize) {
        self.text = format!("{v:.decimals$}");
    }
    fn value(&self) -> Option<f32> {
        self.text.trim().parse().ok()
    }
}

impl DevelopApp {
    pub fn new(path: PathBuf, wakeup: SharedWakeup) -> Result<Self> {
        let params = DevelopParams::for_raw(&path)?;
        let (tx, load_rx) = channel();
        {
            let path = path.clone();
            let wakeup = wakeup.clone();
            std::thread::Builder::new()
                .name("idev-unpack".into())
                .spawn(move || {
                    let t = Instant::now();
                    let r = sensor::unpack(&path);
                    tracing::info!("unpacked in {:.0?}", t.elapsed());
                    let _ = tx.send(r);
                    wake(&wakeup);
                })
                .context("spawn unpack thread")?;
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let sidecar = SidecarWriter::spawn(sidecar_path(&path));
        let mut app = Self {
            path,
            name,
            sidecar,
            frame: None,
            load_rx,
            load_error: None,
            params,
            gpu: None,
            wakeup,
            toasts: Vec::new(),
            viewport_requests: Vec::new(),
            export_rx: Vec::new(),
            exports_running: 0,
            sdr_preview: false,
            histogram: None,
            last_zoom: Cell::new(0.2),
            scale_factor: Cell::new(1.0),
            display_peak: Cell::new(1.0),
            hdr_output: Cell::new(false),
            exposure: Scrub::default(),
            contrast: Scrub::default(),
            hue: Scrub::default(),
            white: Scrub::default(),
            kelvin: Scrub::default(),
            tint: Scrub::default(),
        };
        app.seed_controls();
        Ok(app)
    }

    /// Refresh the scrubber texts from the parameters (and the frame's
    /// as-shot white once it is known).
    fn seed_controls(&mut self) {
        self.exposure.set(self.params.exposure_ev, 2);
        let (c, h, w) = match self.params.tone {
            ToneCurve::Sigmoid {
                contrast,
                hue_preservation,
                white,
            } => (contrast, hue_preservation, white),
            ToneCurve::None => (1.6, 0.0, DEFAULT_WHITE),
        };
        self.contrast.set(c, 2);
        self.hue.set(h, 2);
        self.white.set(w, 2);
        let temp = match (self.params.white_balance, &self.frame) {
            (WhiteBalance::Temperature { kelvin, tint }, _) => Some(Temperature { kelvin, tint }),
            (WhiteBalance::AsShot, Some(f)) => {
                whitepoint::white_from_neutral(&f.matrices, f.neutral_as_shot())
                    .map(|w| w.temperature)
            }
            (WhiteBalance::Multipliers { red, blue }, Some(f)) => {
                whitepoint::white_from_neutral(&f.matrices, [1.0 / red, 1.0, 1.0 / blue])
                    .map(|w| w.temperature)
            }
            (_, None) => None,
        };
        if let Some(t) = temp {
            self.kelvin.set(t.kelvin, 0);
            self.tint.set(t.tint, 1);
        }
    }

    fn params_changed(&mut self) {
        self.sidecar.send(&self.params);
        wake(&self.wakeup);
    }

    fn reset_params(&mut self) {
        self.params = DevelopParams::default();
        self.seed_controls();
        self.params_changed();
    }

    fn set_temperature(&mut self) {
        if let (Some(k), Some(t)) = (self.kelvin.value(), self.tint.value()) {
            self.params.white_balance = WhiteBalance::Temperature { kelvin: k, tint: t };
            self.params_changed();
        }
    }

    fn set_tone_numbers(&mut self) {
        if let ToneCurve::Sigmoid { .. } = self.params.tone
            && let (Some(c), Some(h), Some(w)) =
                (self.contrast.value(), self.hue.value(), self.white.value())
        {
            self.params.tone = ToneCurve::Sigmoid {
                contrast: c.max(0.1),
                hue_preservation: h.clamp(0.0, 1.0),
                white: w.max(1.0),
            };
            self.params_changed();
        }
    }

    /// Ask for a save path on a dialog thread, then develop and encode
    /// on a worker with its own headless device.
    fn export(&mut self, kind: ExportKind) {
        let Some(frame) = self.frame.clone() else {
            self.toasts.push(ToastSpec::info("still loading"));
            return;
        };
        let params = self.params.clone();
        let stem = self
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "image".into());
        let dir = self
            .path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let (tx, rx) = channel();
        self.export_rx.push(rx);
        self.exports_running += 1;
        let wakeup = self.wakeup.clone();
        std::thread::Builder::new()
            .name("idev-export".into())
            .spawn(move || {
                let suffix = match kind {
                    ExportKind::HdrPng => ".hdr",
                    _ => "",
                };
                let picked = rfd::FileDialog::new()
                    .set_title("Export")
                    .set_directory(&dir)
                    .set_file_name(format!("{stem}{suffix}.{}", kind.extension()))
                    .add_filter(kind.extension(), &[kind.extension()])
                    .save_file();
                let result = match picked {
                    None => Err(anyhow::anyhow!("cancelled")),
                    Some(path) => export_to(&frame, &params, &kind, &path).map(|()| path),
                };
                let _ = tx.send(result);
                wake(&wakeup);
            })
            .expect("spawn export thread");
    }

    fn poll_background(&mut self) {
        if self.frame.is_none() {
            match self.load_rx.try_recv() {
                Ok(Ok(frame)) => {
                    self.frame = Some(Arc::new(frame));
                    self.seed_controls();
                }
                Ok(Err(e)) => self.load_error = Some(format!("{e:#}")),
                Err(_) => {}
            }
        }
        let mut done = Vec::new();
        for (i, rx) in self.export_rx.iter().enumerate() {
            if let Ok(r) = rx.try_recv() {
                done.push(i);
                self.exports_running = self.exports_running.saturating_sub(1);
                match r {
                    Ok(path) => self
                        .toasts
                        .push(ToastSpec::success(format!("exported {}", path.display()))),
                    Err(e) if e.to_string() == "cancelled" => {}
                    Err(e) => self
                        .toasts
                        .push(ToastSpec::error(format!("export failed: {e:#}"))),
                }
            }
        }
        for i in done.into_iter().rev() {
            self.export_rx.remove(i);
        }
    }

    /// Preview downscale for the current zoom: one texel per physical
    /// pixel when zoomed out, full resolution at or above 1:1.
    fn wanted_scale(&self) -> u32 {
        let px_per_texel = self.last_zoom.get() * self.scale_factor.get();
        if px_per_texel <= 0.0 {
            return 4;
        }
        ((1.0 / px_per_texel).floor() as u32).clamp(1, 16)
    }

    fn peak(&self) -> f32 {
        if self.sdr_preview {
            1.0
        } else {
            self.display_peak.get().max(1.0)
        }
    }

    fn control_panel(&self) -> El {
        let tone_kind = match self.params.tone {
            ToneCurve::None => "none",
            ToneCurve::Sigmoid { .. } => "sigmoid",
        };
        let highlights = match self.params.highlights {
            Highlights::Clip => "clip",
            Highlights::Blend => "blend",
        };
        let demosaic = match self.params.demosaic {
            Demosaic::Bilinear => "bilinear",
            Demosaic::Rcd => "rcd",
        };
        let wb_label = match self.params.white_balance {
            WhiteBalance::AsShot => "as shot",
            WhiteBalance::Multipliers { .. } => "multipliers",
            WhiteBalance::Temperature { .. } => "custom",
        };

        let mut info: Vec<El> = Vec::new();
        if let Some(f) = &self.frame {
            let e = &f.exif;
            info.push(text(format!("{} {}", e.make, e.model)).caption().muted());
            if let Some(l) = &e.lens {
                info.push(text(l.clone()).caption().muted());
            }
            let exp = e
                .exposure_time
                .map(|(n, d)| format!("{n}/{d} s"))
                .unwrap_or_default();
            let fnum = e.f_number.map(|v| format!("f/{v:.1}")).unwrap_or_default();
            let iso = e.iso.map(|v| format!("ISO {v}")).unwrap_or_default();
            info.push(text(format!("{exp}  {fnum}  {iso}")).caption().muted());
            info.push(
                text(format!("{} x {}", f.crop.width, f.crop.height))
                    .caption()
                    .muted(),
            );
        }

        let tabs = |key: &str, current: &str, options: [(&str, &str); 2]| {
            tabs_list(
                key,
                &current,
                options.map(|(v, l)| (v.to_string(), l.to_string())),
            )
            .width(Size::Fixed(TABS_WIDTH))
        };
        sidebar([scroll([column([
            sidebar_group([sidebar_group_label("Histogram"), self.histogram_el()]),
            sidebar_group([
                sidebar_group_label("Tone"),
                field_row(
                    "Exposure (EV)",
                    number_scrubber("exposure", &self.exposure.text),
                ),
                field_row(
                    "Curve",
                    tabs(
                        "tone",
                        tone_kind,
                        [("none", "None"), ("sigmoid", "Sigmoid")],
                    ),
                ),
                field_row("Contrast", number_scrubber("contrast", &self.contrast.text)),
                field_row("Hue preservation", number_scrubber("hue", &self.hue.text)),
                field_row("White", number_scrubber("white", &self.white.text)).tooltip("scene value mapped to the display peak; 1 = diffuse white, 4 = two stops above"),
            ]),
            sidebar_group([
                sidebar_group_label("White balance"),
                field_row(
                    "Temperature (K)",
                    number_scrubber("kelvin", &self.kelvin.text),
                ),
                field_row("Tint", number_scrubber("tint", &self.tint.text)),
                row([
                    text(wb_label).muted(),
                    spacer(),
                    button("As shot").key("wb-as-shot").secondary(),
                ])
                .gap(tokens::SPACE_2),
            ]),
            sidebar_group([
                sidebar_group_label("Pipeline"),
                field_row(
                    "Highlights",
                    tabs(
                        "highlights",
                        highlights,
                        [("clip", "Clip"), ("blend", "Blend")],
                    ),
                ),
                field_row(
                    "Demosaic",
                    tabs(
                        "demosaic",
                        demosaic,
                        [("bilinear", "Bilinear"), ("rcd", "RCD")],
                    ),
                ),
            ]),
            sidebar_group([
                sidebar_group_label("Export"),
                row([
                    button("JPEG").key("export-jpg"),
                    button("PNG").key("export-png"),
                    button("HDR PNG")
                        .key("export-hdr-png")
                        .tooltip("BT.2020 PQ 16-bit, 1000-nit master"),
                    button("EXR")
                        .key("export-exr")
                        .tooltip("scRGB linear f16, current parameters"),
                ])
                .gap(tokens::SPACE_2),
                if self.exports_running > 0 {
                    row([spinner(), text("exporting…").muted()]).gap(tokens::SPACE_2)
                } else {
                    row(Vec::<El>::new())
                },
                button("Reset parameters").key("reset").ghost(),
            ]),
            sidebar_group(info),
        ])
        .gap(tokens::SPACE_4)])
        .scrollbar()])
        .width(Size::Fixed(SIDEBAR_WIDTH))
    }

    /// Per-channel histogram as translucent bars, log-scaled counts.
    fn histogram_el(&self) -> El {
        let Some(h) = &self.histogram else {
            return skeleton().height(Size::Fixed(HISTOGRAM_HEIGHT));
        };
        let bin_px = 2.0;
        let bins = 128usize;
        let per = 256 / bins;
        let max = h
            .bins
            .iter()
            .flat_map(|c| c.chunks(per).map(|ch| ch.iter().sum::<u32>()))
            .max()
            .unwrap_or(1)
            .max(1) as f32;
        let scale = |v: u32| -> f32 {
            if v == 0 {
                0.0
            } else {
                (v as f32).ln_1p() / max.ln_1p() * HISTOGRAM_HEIGHT
            }
        };
        let colours = [
            Color::srgb_u8(242, 77, 77),
            Color::srgb_u8(77, 217, 90),
            Color::srgb_u8(90, 128, 255),
        ];
        let columns: Vec<El> = (0..bins)
            .map(|i| {
                let bars: Vec<El> = (0..3)
                    .map(|c| {
                        let v: u32 = h.bins[c][i * per..(i + 1) * per].iter().sum();
                        column(Vec::<El>::new())
                            .width(Size::Fixed(bin_px))
                            .height(Size::Fixed(scale(v)))
                            .fill(colours[c].with_alpha(0.55))
                    })
                    .collect();
                // Overlay the three bars, bottom-aligned.
                stack(
                    bars.into_iter()
                        .map(|b| column([b]).justify(Justify::End).fill_size())
                        .collect::<Vec<_>>(),
                )
                .width(Size::Fixed(bin_px))
                .height(Size::Fixed(HISTOGRAM_HEIGHT))
            })
            .collect();
        row(columns)
            .height(Size::Fixed(HISTOGRAM_HEIGHT))
            .fill(tokens::MUTED.with_alpha(0.3))
    }

    fn canvas(&self) -> El {
        let Some(frame) = &self.frame else {
            let msg = match &self.load_error {
                Some(e) => text(e.clone()).text_color(tokens::DESTRUCTIVE),
                None => text(format!("Unpacking {}…", self.name)).muted(),
            };
            return column([spinner(), msg])
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .justify(Justify::Center)
                .fill_size();
        };
        let (w, h) = (frame.crop.width as f32, frame.crop.height as f32);
        let content = match self.gpu.as_ref().and_then(|g| g.preview.as_ref()) {
            Some(p) => surface(p.texture.clone())
                .width(Size::Fixed(w))
                .height(Size::Fixed(h))
                .surface_fit(damascene_core::image::ImageFit::Fill)
                .surface_alpha(SurfaceAlpha::Opaque),
            None => column(Vec::<El>::new())
                .width(Size::Fixed(w))
                .height(Size::Fixed(h))
                .fill(tokens::MUTED),
        };
        viewport([content])
            .key("canvas")
            .fit_policy(FitPolicy::Contain {
                padding: FIT_PADDING,
            })
            .min_zoom(0.02)
            .max_zoom(8.0)
            .fill(tokens::CARD)
    }
}

impl App for DevelopApp {
    fn build(&self, cx: &BuildCx) -> El {
        if let Some(diag) = cx.diagnostics() {
            self.scale_factor.set(diag.scale_factor.max(0.1));
            let hdr = diag.hdr_active();
            self.hdr_output.set(hdr);
            let peak = if hdr {
                match &diag.color_management {
                    damascene_core::color::ColorManagementStatus::Available { targets, .. } => {
                        let max = targets
                            .target_max_luminance_nits
                            .or(targets.max_luminance_nits)
                            .unwrap_or(1000.0);
                        let reference = targets.reference_luminance_nits.unwrap_or(203.0);
                        (max / reference).max(1.0)
                    }
                    _ => 1.0,
                }
            } else {
                1.0
            };
            self.display_peak.set(peak);
        }
        let zoom = cx
            .viewport_view("canvas")
            .map_or(self.last_zoom.get(), |v| v.zoom);
        self.last_zoom.set(zoom);
        // The preview texture is rendered in before_paint from the zoom
        // recorded here; if it is stale, ask for one more frame.
        if let Some(p) = self.gpu.as_ref().and_then(|g| g.preview.as_ref())
            && p.scale != self.wanted_scale()
        {
            wake(&self.wakeup);
        }

        let pct = zoom * self.scale_factor.get() * 100.0;
        let readout = if cx.viewport_at_home("canvas").unwrap_or(true) {
            format!("fit · {pct:.0}%")
        } else {
            format!("{pct:.0}%")
        };
        let mode = if self.hdr_output.get() {
            let b = badge(format!("HDR · {:.1}x", self.display_peak.get()));
            if self.sdr_preview {
                b.muted().tooltip("SDR preview (t)")
            } else {
                b.success()
                    .tooltip("tone curve targets the panel headroom; t for SDR preview")
            }
        } else {
            badge("SDR")
        }
        .key("color-mode");

        page([
            toolbar([
                toolbar_title(self.name.clone()),
                spacer(),
                text(readout).muted().tabular_numerals(),
                button("Fit").key("fit").ghost(),
                button("1:1").key("one").ghost(),
                mode,
            ])
            .gap(tokens::SPACE_3)
            .padding(Sides::xy(tokens::SPACE_4, tokens::SPACE_2)),
            row([self.canvas(), self.control_panel()])
                .gap(tokens::SPACE_3)
                .height(Size::Fill(1.0)),
        ])
    }

    fn hotkeys(&self) -> Vec<(KeyChord, String)> {
        vec![
            (KeyChord::vim('f'), "fit".into()),
            (KeyChord::vim('1'), "one".into()),
            (KeyChord::vim('t'), "sdr-preview".into()),
            (KeyChord::vim('e'), "export-jpg".into()),
        ]
    }

    fn on_event(&mut self, event: UiEvent, _cx: &EventCx) {
        if event.is_click_or_activate("fit") || event.is_hotkey("fit") {
            self.viewport_requests.push(ViewportRequest::FitContent {
                key: "canvas".into(),
                padding: FIT_PADDING,
                behavior: ViewportBehavior::Smooth,
            });
            return;
        }
        if event.is_click_or_activate("one") || event.is_hotkey("one") {
            self.viewport_requests.push(ViewportRequest::ResetView {
                key: "canvas".into(),
                behavior: ViewportBehavior::Smooth,
            });
            return;
        }
        if event.is_hotkey("sdr-preview") || event.is_click_or_activate("color-mode") {
            self.sdr_preview = !self.sdr_preview;
            wake(&self.wakeup);
            return;
        }
        if event.is_click_or_activate("export-jpg") || event.is_hotkey("export-jpg") {
            self.export(ExportKind::Jpeg);
            return;
        }
        if event.is_click_or_activate("export-png") {
            self.export(ExportKind::Png);
            return;
        }
        if event.is_click_or_activate("export-hdr-png") {
            self.export(ExportKind::HdrPng);
            return;
        }
        if event.is_click_or_activate("export-exr") {
            self.export(ExportKind::Exr);
            return;
        }
        if event.is_click_or_activate("reset") {
            self.reset_params();
            return;
        }
        if event.is_click_or_activate("wb-as-shot") {
            self.params.white_balance = WhiteBalance::AsShot;
            self.seed_controls();
            self.params_changed();
            return;
        }

        let ev = ScrubberOpts::default()
            .min(-6.0)
            .max(6.0)
            .step(0.05)
            .sensitivity(3.0)
            .decimals(2);
        if number_scrubber::apply_event(
            &mut self.exposure.text,
            &mut self.exposure.drag,
            "exposure",
            &ev,
            &event,
        ) {
            if let Some(v) = self.exposure.value() {
                self.params.exposure_ev = v;
                self.params_changed();
            }
            return;
        }
        let contrast = ScrubberOpts::default()
            .min(0.5)
            .max(4.0)
            .step(0.05)
            .sensitivity(4.0)
            .decimals(2);
        if number_scrubber::apply_event(
            &mut self.contrast.text,
            &mut self.contrast.drag,
            "contrast",
            &contrast,
            &event,
        ) {
            self.set_tone_numbers();
            return;
        }
        let hue = ScrubberOpts::default()
            .min(0.0)
            .max(1.0)
            .step(0.05)
            .sensitivity(4.0)
            .decimals(2);
        if number_scrubber::apply_event(&mut self.hue.text, &mut self.hue.drag, "hue", &hue, &event)
        {
            self.set_tone_numbers();
            return;
        }
        let white = ScrubberOpts::default()
            .min(1.0)
            .max(32.0)
            .step(0.1)
            .sensitivity(3.0)
            .decimals(2);
        if number_scrubber::apply_event(
            &mut self.white.text,
            &mut self.white.drag,
            "white",
            &white,
            &event,
        ) {
            self.set_tone_numbers();
            return;
        }
        let kelvin = ScrubberOpts::default()
            .min(1667.0)
            .max(25000.0)
            .step(50.0)
            .sensitivity(2.0)
            .decimals(0);
        if number_scrubber::apply_event(
            &mut self.kelvin.text,
            &mut self.kelvin.drag,
            "kelvin",
            &kelvin,
            &event,
        ) {
            self.set_temperature();
            return;
        }
        let tint = ScrubberOpts::default()
            .min(-150.0)
            .max(150.0)
            .step(0.5)
            .sensitivity(3.0)
            .decimals(1);
        if number_scrubber::apply_event(
            &mut self.tint.text,
            &mut self.tint.drag,
            "tint",
            &tint,
            &event,
        ) {
            self.set_temperature();
            return;
        }

        let mut tone_kind = String::new();
        if tabs::apply_event(&mut tone_kind, &event, "tone", |s| Some(s.to_string())) {
            self.params.tone = match tone_kind.as_str() {
                "none" => ToneCurve::None,
                _ => match self.params.tone {
                    ToneCurve::Sigmoid { .. } => self.params.tone,
                    ToneCurve::None => ToneCurve::default(),
                },
            };
            self.seed_controls();
            self.params_changed();
            return;
        }
        if tabs::apply_event(
            &mut self.params.highlights,
            &event,
            "highlights",
            |s| match s {
                "clip" => Some(Highlights::Clip),
                "blend" => Some(Highlights::Blend),
                _ => None,
            },
        ) {
            self.params_changed();
            return;
        }
        if tabs::apply_event(&mut self.params.demosaic, &event, "demosaic", |s| match s {
            "bilinear" => Some(Demosaic::Bilinear),
            "rcd" => Some(Demosaic::Rcd),
            _ => None,
        }) {
            self.params_changed();
        }
    }

    fn drain_toasts(&mut self) -> Vec<ToastSpec> {
        std::mem::take(&mut self.toasts)
    }

    fn drain_viewport_requests(&mut self) -> Vec<ViewportRequest> {
        std::mem::take(&mut self.viewport_requests)
    }
}

impl WinitWgpuApp for DevelopApp {
    fn gpu_setup(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let gpu = Gpu::from_device(device.clone(), queue.clone());
        let dev = Developer::new(&gpu);
        self.gpu = Some(GpuState {
            gpu,
            dev,
            prepared: None,
            preview: None,
            rendered: None,
            histogram_pending: None,
        });
    }

    fn before_paint(&mut self, _queue: &wgpu::Queue) {
        self.poll_background();
        let Some(frame) = self.frame.clone() else {
            return;
        };
        let peak = self.peak();
        let scale = self.wanted_scale();
        let params = self.params.clone();
        let Some(g) = self.gpu.as_mut() else { return };

        if g.prepared.as_ref().is_none_or(|p| !p.matches(&params)) {
            let t = Instant::now();
            match g.dev.prepare(&g.gpu, &frame, &params) {
                Ok(p) => {
                    g.prepared = Some(p);
                    g.rendered = None;
                    tracing::info!("prepare submitted in {:.1?}", t.elapsed());
                }
                Err(e) => {
                    self.toasts.push(ToastSpec::error(format!("{e:#}")));
                    return;
                }
            }
        }
        let prepared = g.prepared.as_ref().expect("prepared above");

        if g.preview.as_ref().is_none_or(|p| p.scale != scale) {
            let (w, h) = Developer::scaled_size(prepared, scale);
            let texture = g.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("idev::preview"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let texture = damascene_wgpu::app_texture(Arc::new(texture));
            g.preview = Some(Preview {
                scale,
                texture,
                view,
            });
            g.rendered = None;
            tracing::debug!(w, h, scale, "preview texture");
        }
        let preview = g.preview.as_ref().expect("preview above");

        let want = (params, peak, scale);
        if g.rendered.as_ref() != Some(&want) {
            g.dev.colour(
                &g.gpu,
                prepared,
                &want.0,
                OutputTarget { peak },
                scale,
                DisplayPrimaries::Srgb,
                &preview.view,
            );
            g.rendered = Some(want);
            // One histogram in flight at a time; a newer render just
            // waits for the next frame.
            if g.histogram_pending.is_none() {
                let (w, h) = Developer::scaled_size(prepared, scale);
                g.histogram_pending = Some(g.dev.histogram(&g.gpu, &preview.view, w, h, peak));
            }
        }
        if let Some(pending) = g.histogram_pending.as_mut()
            && let Some(hist) = pending.try_take()
        {
            g.histogram_pending = None;
            self.histogram = Some(hist);
            wake(&self.wakeup);
        }
    }
}

/// Develop at full resolution on a private device and encode.
fn export_to(
    frame: &SensorFrame,
    params: &DevelopParams,
    kind: &ExportKind,
    path: &Path,
) -> Result<()> {
    let t = Instant::now();
    let gpu = Gpu::headless()?;
    let dev = Developer::new(&gpu);
    let target = match kind {
        ExportKind::HdrPng => OutputTarget::hdr_nits(HDR_EXPORT_NITS),
        _ => OutputTarget::SDR,
    };
    let img: WorkingImage = dev.develop(&gpu, frame, params, target)?;
    let img = output::oriented(&img, frame.orientation);
    match kind {
        ExportKind::Jpeg => {
            idev_formats::write_jpeg_srgb8(path, img.width, img.height, &output::srgb8(&img), 92)?
        }
        ExportKind::Png => {
            idev_formats::write_png_srgb8(path, img.width, img.height, &output::srgb8(&img))?
        }
        ExportKind::HdrPng => idev_formats::write_png_bt2020_pq16(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(&img),
        )?,
        ExportKind::Exr => idev_formats::write_exr_rgb_f32(
            path,
            img.width,
            img.height,
            &output::scrgb_linear_f32(&img),
        )?,
    }
    tracing::info!("exported {} in {:.1?}", path.display(), t.elapsed());
    Ok(())
}
