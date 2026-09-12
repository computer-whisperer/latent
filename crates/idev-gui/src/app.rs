//! The window: a folder's thumbnail grid (library mode) and a develop
//! view for one raw (develop mode) with a pan/zoom preview and a control
//! sidebar.
//!
//! GPU work happens in `before_paint` on the host's device: the
//! demosaiced camera RGB of the open raw stays resident (`Prepared`) and
//! the colour pass re-renders into the preview texture whenever the
//! parameters, the display peak, or the preview scale changed. The
//! preview scale follows the viewport zoom so a fit-to-window preview
//! costs a fraction of a full-resolution pass; at 1:1 the scale is 1.
//! Thumbnails and exports run on their own threads with headless
//! devices so the window never stalls.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use damascene_core::color::Primaries;
use damascene_core::event::PointerButton;
use damascene_core::prelude::*;
use damascene_core::scroll::{self, ScrollAlignment, ScrollRequest};
use damascene_core::surface::{AppTexture, SurfaceAlpha};
use damascene_core::toast::ToastSpec;
use damascene_core::viewport::{FitPolicy, ViewportBehavior, ViewportRequest};
use damascene_core::widgets::number_scrubber::{self, ScrubDrag, ScrubberOpts};
use damascene_core::widgets::{switch, tabs, text_input};
use damascene_core::{BuildCx, EventCx, KeyChord, UiEvent, UiEventKind};
use damascene_winit_wgpu::{Wakeup, WinitWgpuApp};
use idev_core::color::Gamut;
use idev_core::geometry;
use idev_core::gpu::{
    Developer, DisplayPrimaries, Gpu, Histogram, HistogramReadback, OutputTarget, Prepared, Render,
};
use idev_core::metadata;
use idev_core::params::{
    DEFAULT_WHITE, Demosaic, DevelopParams, Highlights, ToneCurve, WhiteBalance, sidecar_path,
};
use idev_core::sensor::ExifSummary;
use idev_core::source::{self, Source};
use idev_core::whitepoint::{self, Temperature};
use idev_core::{WorkingImage, output};

use crate::library::{self, Filter, Item, Loaded, Loader, LoaderResults, SortKey, sort_indices};
use crate::thumbs::{CACHE_BUDGET, ThumbCache, ThumbGpu};

/// The host's redraw handle, filled in once the event loop exists.
pub type SharedWakeup = Arc<Mutex<Option<Wakeup>>>;

pub fn wake(w: &SharedWakeup) {
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
    /// window close and lets a requeued thumbnail see the final file.
    fn drop(&mut self) {
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn named(key: NamedKey) -> KeyChord {
    KeyChord::named(LogicalKey::Named(key))
}

/// Peak nits the HDR exports are mastered for.
const HDR_EXPORT_NITS: f32 = 1000.0;
/// JPEG / AVIF quality for GUI exports.
const EXPORT_QUALITY: u8 = 92;
const SIDEBAR_WIDTH: f32 = 320.0;
/// The library filter box.
const FILTER_WIDTH: f32 = 380.0;
const FIT_PADDING: f32 = 8.0;
const TABS_WIDTH: f32 = 170.0;
const HISTOGRAM_HEIGHT: f32 = 90.0;

/// Grid tile geometry (logical px): a 3:2 image area over a caption.
const TILE_W: f32 = 240.0;
const TILE_H: f32 = 160.0;
const CAPTION_H: f32 = 22.0;
const TILE_GAP: f32 = tokens::SPACE_2;
const GRID_PAD: f32 = tokens::SPACE_4;
/// Thumbnail workers. Camera previews are cheap; developed thumbnails
/// serialise on the shared headless device anyway.
const THUMB_WORKERS: usize = 3;

struct GpuState {
    gpu: Gpu,
    dev: Developer,
    prepared: Option<Prepared>,
    preview: Option<Preview>,
    /// What the preview texture currently shows.
    rendered: Option<(DevelopParams, Render)>,
    /// Histogram of the preview, in flight.
    histogram_pending: Option<HistogramReadback>,
}

impl GpuState {
    /// Forget the open raw's resident data (a different file is opening).
    fn clear(&mut self) {
        self.prepared = None;
        self.preview = None;
        self.rendered = None;
        self.histogram_pending = None;
    }
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
    Avif,
    HdrAvif,
    Exr,
}

impl ExportKind {
    fn extension(&self) -> &'static str {
        match self {
            ExportKind::Jpeg => "jpg",
            ExportKind::Png | ExportKind::HdrPng => "png",
            ExportKind::Avif | ExportKind::HdrAvif => "avif",
            ExportKind::Exr => "exr",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Library,
    Develop,
}

/// The folder being browsed and everything the grid knows about it.
struct Library {
    dir: PathBuf,
    files: Vec<PathBuf>,
    names: Vec<String>,
    thumbs: Vec<Option<Image>>,
    developed: Vec<bool>,
    errors: Vec<Option<String>>,
    /// Shooting summary per file, once its thumbnail (cache entry) has
    /// been read.
    exifs: Vec<Option<ExifSummary>>,
    /// Raw + rendered partner by stem (see `library::pairs`).
    pair: Vec<Option<usize>>,
    loader: Loader,
    results: LoaderResults,
    loaded: usize,

    // The view: what the grid shows, in which order.
    sort: SortKey,
    query: String,
    query_sel: Selection,
    /// Hide rendered files that have a raw partner.
    stack: bool,
    /// File indices in display order.
    view: Vec<usize>,
}

impl Library {
    fn open(
        dir: PathBuf,
        files: Vec<PathBuf>,
        wakeup: SharedWakeup,
        cache: Arc<ThumbCache>,
        gpu: Arc<ThumbGpu>,
    ) -> Library {
        let n = files.len();
        let names = files
            .iter()
            .map(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        let (loader, results) = Loader::spawn(files.clone(), THUMB_WORKERS, wakeup, cache, gpu);
        let pair = library::pairs(&files);
        let mut lib = Library {
            dir,
            files,
            names,
            thumbs: vec![None; n],
            developed: vec![false; n],
            errors: vec![None; n],
            exifs: vec![None; n],
            pair,
            loader,
            results,
            loaded: 0,
            sort: SortKey::Name,
            query: String::new(),
            query_sel: Selection::default(),
            stack: true,
            view: Vec::new(),
        };
        lib.refresh_view();
        lib
    }

    /// Whether the view depends on summaries that may still be loading.
    fn view_uses_metadata(&self) -> bool {
        self.sort != SortKey::Name || !self.query.trim().is_empty()
    }

    fn drain(&mut self) -> bool {
        let mut any = false;
        for loaded in self.results.drain() {
            any = true;
            match loaded {
                Loaded::Thumb { index, thumb } => {
                    if self.thumbs[index].is_none() {
                        self.loaded += 1;
                    }
                    self.thumbs[index] = Some(thumb.image);
                    self.developed[index] = thumb.developed;
                    self.exifs[index] = Some(thumb.exif);
                    self.errors[index] = None;
                }
                Loaded::Failed { index, error } => self.errors[index] = Some(error),
            }
        }
        any
    }

    /// Recompute `view` from the stack toggle, the filter and the sort.
    fn refresh_view(&mut self) {
        let items: Vec<Item> = (0..self.files.len())
            .map(|i| Item {
                name: &self.names[i],
                raw: source::is_raw(&self.files[i]),
                exif: self.exifs[i].as_ref(),
            })
            .collect();
        let filter = Filter::parse(&self.query);
        let mut view: Vec<usize> = (0..self.files.len())
            .filter(|&i| !(self.stack && !items[i].raw && self.pair[i].is_some()))
            .filter(|&i| filter.matches(&items[i]))
            .collect();
        sort_indices(&mut view, &items, self.sort);
        self.view = view;
    }

    /// The filter input owns the keyboard while it holds the selection.
    fn filter_focused(&self) -> bool {
        self.query_sel.within("filter").is_some()
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        self.loader.shutdown();
    }
}

/// The raw open in develop mode.
struct Session {
    index: usize,
    path: PathBuf,
    name: String,
    frame: Option<Arc<Source>>,
    load_rx: Receiver<Result<Source>>,
    load_error: Option<String>,
    params: DevelopParams,
    sidecar: SidecarWriter,
    /// Parameters changed since opening: the grid thumbnail is stale.
    edited: bool,
}

impl Session {
    fn open(index: usize, path: PathBuf, wakeup: &SharedWakeup) -> Result<Session> {
        let params = DevelopParams::for_raw(&path)?;
        let (tx, load_rx) = channel();
        {
            let path = path.clone();
            let wakeup = wakeup.clone();
            std::thread::Builder::new()
                .name("idev-unpack".into())
                .spawn(move || {
                    let t = Instant::now();
                    let r = Source::load(&path);
                    tracing::info!("loaded in {:.0?}", t.elapsed());
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
        Ok(Session {
            index,
            path,
            name,
            frame: None,
            load_rx,
            load_error: None,
            params,
            sidecar,
            edited: false,
        })
    }
}

pub struct IdevApp {
    library: Library,
    notice: Option<String>,
    selected: usize,
    cols: Cell<usize>,
    scroll_requests: Vec<ScrollRequest>,
    picker: Option<Receiver<Option<PathBuf>>>,
    cache: Arc<ThumbCache>,
    thumb_gpu: Arc<ThumbGpu>,
    mode: Mode,
    session: Option<Session>,
    histogram: Option<Histogram>,

    gpu: Option<GpuState>,
    wakeup: SharedWakeup,
    toasts: Vec<ToastSpec>,
    viewport_requests: Vec<ViewportRequest>,
    export_rx: Vec<Receiver<Result<PathBuf>>>,
    exports_running: usize,
    sdr_preview: bool,
    /// Zebra overlay over pixels the panel cannot show.
    warn: bool,
    /// Sidebar shows the file's metadata instead of the controls.
    info: bool,

    // Read during build (immutable), consumed in before_paint.
    last_zoom: Cell<f32>,
    scale_factor: Cell<f32>,
    display_peak: Cell<f32>,
    hdr_output: Cell<bool>,
    /// The panel's gamut as the compositor reports it (sRGB when
    /// nothing is reported).
    panel_gamut: Cell<Gamut>,

    // Crop / straighten.
    /// Drawing the crop on the canvas: the full frame shows, plain
    /// drag sets the rectangle, middle-drag pans.
    crop_mode: bool,
    crop_aspect: Aspect,
    /// Anchor of the drag in progress, as view fractions.
    crop_anchor: Option<(f32, f32)>,
    angle: Scrub,
    /// Export long edge in pixels; 0 = full size. Not a develop
    /// parameter: it does not go into the sidecar.
    export_edge: Scrub,
    sharpen_amount: Scrub,
    sharpen_radius: Scrub,
    sharpen_threshold: Scrub,

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

/// Aspect lock for the crop rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Aspect {
    Free,
    R3x2,
    R4x3,
    R1x1,
    R16x9,
}

impl Aspect {
    const ALL: [Aspect; 5] = [
        Aspect::Free,
        Aspect::R3x2,
        Aspect::R4x3,
        Aspect::R1x1,
        Aspect::R16x9,
    ];

    fn key(self) -> &'static str {
        match self {
            Aspect::Free => "free",
            Aspect::R3x2 => "3:2",
            Aspect::R4x3 => "4:3",
            Aspect::R1x1 => "1:1",
            Aspect::R16x9 => "16:9",
        }
    }

    fn label(self) -> &'static str {
        self.key()
    }

    fn parse(s: &str) -> Option<Aspect> {
        Aspect::ALL.into_iter().find(|a| a.key() == s)
    }

    /// Width over height, `None` for free.
    fn ratio(self) -> Option<f32> {
        match self {
            Aspect::Free => None,
            Aspect::R3x2 => Some(3.0 / 2.0),
            Aspect::R4x3 => Some(4.0 / 3.0),
            Aspect::R1x1 => Some(1.0),
            Aspect::R16x9 => Some(16.0 / 9.0),
        }
    }
}

impl Scrub {
    fn set(&mut self, v: f32, decimals: usize) {
        self.text = format!("{v:.decimals$}");
    }
    fn value(&self) -> Option<f32> {
        self.text.trim().parse().ok()
    }
}

impl IdevApp {
    /// Open `dir` in the library; if `file` is given (a raw inside
    /// `dir`), start in develop mode on it.
    pub fn new(dir: PathBuf, file: Option<PathBuf>, wakeup: SharedWakeup) -> Result<Self> {
        let files = library::scan_dir(&dir)?;
        let cache = Arc::new(
            ThumbCache::standard().context("no cache directory (XDG_CACHE_HOME or HOME)")?,
        );
        {
            let cache = cache.clone();
            std::thread::Builder::new()
                .name("idev-thumb-sweep".into())
                .spawn(move || {
                    cache.sweep(CACHE_BUDGET);
                })
                .context("spawn cache sweep")?;
        }
        let thumb_gpu = Arc::new(ThumbGpu::default());
        let selected = file
            .as_ref()
            .and_then(|f| files.iter().position(|p| p == f))
            .unwrap_or(0);
        let library = Library::open(dir, files, wakeup.clone(), cache.clone(), thumb_gpu.clone());
        let mut app = Self {
            library,
            notice: None,
            selected,
            cols: Cell::new(1),
            scroll_requests: Vec::new(),
            picker: None,
            cache,
            thumb_gpu,
            mode: Mode::Library,
            session: None,
            histogram: None,
            gpu: None,
            wakeup,
            toasts: Vec::new(),
            viewport_requests: Vec::new(),
            export_rx: Vec::new(),
            exports_running: 0,
            sdr_preview: false,
            warn: false,
            info: false,
            last_zoom: Cell::new(0.1),
            scale_factor: Cell::new(1.0),
            display_peak: Cell::new(1.0),
            hdr_output: Cell::new(false),
            panel_gamut: Cell::new(Gamut::Srgb),
            crop_mode: false,
            crop_aspect: Aspect::Free,
            crop_anchor: None,
            angle: Scrub::default(),
            export_edge: Scrub {
                text: "0".into(),
                drag: ScrubDrag::default(),
            },
            sharpen_amount: Scrub::default(),
            sharpen_radius: Scrub::default(),
            sharpen_threshold: Scrub::default(),
            exposure: Scrub::default(),
            contrast: Scrub::default(),
            hue: Scrub::default(),
            white: Scrub::default(),
            kelvin: Scrub::default(),
            tint: Scrub::default(),
        };
        if let Some(file) = file {
            if app.library.files.get(selected) == Some(&file) {
                app.open_selected();
            } else {
                anyhow::bail!(
                    "{}: not a raw file in {}",
                    file.display(),
                    app.library.dir.display()
                );
            }
        }
        Ok(app)
    }

    // ----- library -------------------------------------------------------

    fn open_collection(&mut self, dir: PathBuf, files: Vec<PathBuf>) {
        self.close_session();
        self.mode = Mode::Library;
        self.selected = 0;
        self.library = Library::open(
            dir,
            files,
            self.wakeup.clone(),
            self.cache.clone(),
            self.thumb_gpu.clone(),
        );
    }

    /// Launch the system folder picker on its own thread; the dialog
    /// blocks until dismissed while the UI keeps rendering.
    fn open_folder_dialog(&mut self) {
        if self.picker.is_some() {
            return;
        }
        let (tx, rx) = channel();
        self.picker = Some(rx);
        let wakeup = self.wakeup.clone();
        let start = self.library.dir.clone();
        std::thread::Builder::new()
            .name("idev-folder-picker".into())
            .spawn(move || {
                let dialog = rfd::FileDialog::new()
                    .set_title("Open folder")
                    .set_directory(start);
                let _ = tx.send(dialog.pick_folder());
                wake(&wakeup);
            })
            .expect("spawn folder picker");
    }

    fn drain_picker(&mut self) {
        let Some(rx) = &self.picker else { return };
        let picked = match rx.try_recv() {
            Ok(picked) => picked,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        };
        self.picker = None;
        let Some(dir) = picked else { return };
        match library::scan_dir(&dir) {
            Ok(files) => {
                tracing::info!(count = files.len(), dir = %dir.display(), "folder opened");
                self.notice = None;
                self.open_collection(dir, files);
            }
            Err(e) => self.toasts.push(ToastSpec::error(format!("{e:#}"))),
        }
    }

    /// Select by position in the view.
    fn select(&mut self, position: usize) {
        self.selected = position.min(self.library.view.len().saturating_sub(1));
        let cols = self.cols.get().max(1);
        self.scroll_requests.push(ScrollRequest::new(
            "grid",
            self.selected / cols,
            ScrollAlignment::Visible,
        ));
    }

    /// Rebuild the view, keeping the selected file selected when it is
    /// still shown.
    fn refresh_view(&mut self) {
        let current = self.library.view.get(self.selected).copied();
        self.library.refresh_view();
        let position = current.and_then(|f| self.library.view.iter().position(|&i| i == f));
        self.selected = position
            .unwrap_or(0)
            .min(self.library.view.len().saturating_sub(1));
        wake(&self.wakeup);
    }

    /// Enter develop mode on the selected file.
    fn open_selected(&mut self) {
        let Some(&index) = self.library.view.get(self.selected) else {
            return;
        };
        let path = self.library.files[index].clone();
        if self.session.as_ref().is_some_and(|s| s.index == index) {
            self.mode = Mode::Develop;
            return;
        }
        self.close_session();
        match Session::open(index, path, &self.wakeup) {
            Ok(s) => {
                self.session = Some(s);
                self.mode = Mode::Develop;
                self.histogram = None;
                if let Some(g) = self.gpu.as_mut() {
                    g.clear();
                }
                self.viewport_requests.push(ViewportRequest::FitContent {
                    key: "canvas".into(),
                    padding: FIT_PADDING,
                    behavior: ViewportBehavior::Instant,
                });
                self.seed_controls();
            }
            Err(e) => self.toasts.push(ToastSpec::error(format!("{e:#}"))),
        }
    }

    /// Drop the open session, flushing its sidecar, and requeue its
    /// thumbnail if the parameters changed.
    fn close_session(&mut self) {
        if let Some(s) = self.session.take() {
            let (index, edited) = (s.index, s.edited);
            drop(s);
            if edited {
                self.library.loader.invalidate(index);
            }
        }
    }

    fn step(&mut self, delta: isize) {
        let n = self.library.view.len();
        if n == 0 {
            return;
        }
        let i = (self.selected as isize + delta).clamp(0, n as isize - 1) as usize;
        if i != self.selected {
            self.select(i);
            if self.mode == Mode::Develop {
                self.open_selected();
            }
        }
    }

    // ----- develop -------------------------------------------------------

    /// Refresh the scrubber texts from the parameters (and the frame's
    /// as-shot white once it is known).
    fn seed_controls(&mut self) {
        let Some(s) = &self.session else { return };
        self.angle.set(s.params.geometry.angle_deg, 2);
        self.sharpen_amount.set(s.params.sharpen.amount, 2);
        self.sharpen_radius.set(s.params.sharpen.radius, 2);
        self.sharpen_threshold.set(s.params.sharpen.threshold, 3);
        self.exposure.set(s.params.exposure_ev, 2);
        let (c, h, w) = match s.params.tone {
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
        let raw = s.frame.as_deref().and_then(Source::as_raw);
        let temp = match (s.params.white_balance, raw) {
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
        if let Some(s) = self.session.as_mut() {
            s.sidecar.send(&s.params);
            s.edited = true;
        }
        wake(&self.wakeup);
    }

    fn reset_params(&mut self) {
        if let Some(s) = self.session.as_mut() {
            s.params = DevelopParams::default();
        }
        self.seed_controls();
        self.params_changed();
    }

    fn set_temperature(&mut self) {
        if let (Some(k), Some(t), Some(s)) = (
            self.kelvin.value(),
            self.tint.value(),
            self.session.as_mut(),
        ) {
            s.params.white_balance = WhiteBalance::Temperature { kelvin: k, tint: t };
            self.params_changed();
        }
    }

    fn set_tone_numbers(&mut self) {
        let Some(s) = self.session.as_mut() else {
            return;
        };
        if let ToneCurve::Sigmoid { .. } = s.params.tone
            && let (Some(c), Some(h), Some(w)) =
                (self.contrast.value(), self.hue.value(), self.white.value())
        {
            s.params.tone = ToneCurve::Sigmoid {
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
        let Some(s) = &self.session else { return };
        let Some(frame) = s.frame.clone() else {
            self.toasts.push(ToastSpec::info("still loading"));
            return;
        };
        let params = s.params.clone();
        let long_edge = self
            .export_edge
            .value()
            .map(|v| v.round() as u32)
            .filter(|&v| v > 0);
        let stem = s
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "image".into());
        let dir = s.path.parent().map(Path::to_path_buf).unwrap_or_default();
        let name = s.name.clone();
        let (tx, rx) = channel();
        self.export_rx.push(rx);
        self.exports_running += 1;
        let wakeup = self.wakeup.clone();
        std::thread::Builder::new()
            .name("idev-export".into())
            .spawn(move || {
                let suffix = match kind {
                    ExportKind::HdrPng | ExportKind::HdrAvif => ".hdr",
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
                    Some(path) => {
                        export_to(&frame, &name, &params, &kind, long_edge, &path).map(|()| path)
                    }
                };
                let _ = tx.send(result);
                wake(&wakeup);
            })
            .expect("spawn export thread");
    }

    fn poll_background(&mut self) {
        self.drain_picker();
        // Summaries arriving can change a metadata-ordered or filtered view.
        if self.library.drain() && self.library.view_uses_metadata() {
            self.refresh_view();
        }
        let mut loaded = false;
        if let Some(s) = self.session.as_mut()
            && s.frame.is_none()
        {
            match s.load_rx.try_recv() {
                Ok(Ok(frame)) => {
                    s.frame = Some(Arc::new(frame));
                    loaded = true;
                }
                Ok(Err(e)) => s.load_error = Some(format!("{e:#}")),
                Err(_) => {}
            }
        }
        if loaded {
            self.seed_controls();
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

    // ----- build ---------------------------------------------------------

    fn color_mode_badge(&self) -> El {
        let gamut = self.panel_gamut.get().label();
        if self.hdr_output.get() {
            let b = badge(format!("HDR · {:.1}x · {gamut}", self.display_peak.get()));
            if self.sdr_preview {
                b.muted().tooltip("SDR preview (t)")
            } else {
                b.success()
                    .tooltip("tone curve targets the panel headroom; t for SDR preview")
            }
        } else {
            badge(format!("SDR · {gamut}"))
        }
        .key("color-mode")
    }

    fn info_button(&self) -> El {
        let b = button("Info")
            .key("info")
            .tooltip("show the file's metadata in the sidebar; i");
        if self.info { b } else { b.ghost() }
    }

    fn warn_button(&self) -> El {
        let b = button("Warn").key("warn").tooltip(
            "zebra pixels at the output peak (red) or outside the panel's gamut (cyan); w",
        );
        if self.warn { b } else { b.ghost() }
    }

    fn grid(&self, cx: &BuildCx) -> El {
        let lib = &self.library;
        let vw = cx.viewport_width().unwrap_or(1280.0);
        let avail = vw - 2.0 * (tokens::SPACE_4 + GRID_PAD) - 16.0;
        let cols = (((avail + TILE_GAP) / (TILE_W + TILE_GAP)) as usize).max(1);
        self.cols.set(cols);

        // Snapshots for the 'static cell builder; Image clones are Arcs.
        let thumbs = lib.thumbs.clone();
        let developed = lib.developed.clone();
        let errors = lib.errors.clone();
        let names = lib.names.clone();
        let view = lib.view.clone();
        let paired: Vec<bool> = (0..lib.files.len())
            .map(|i| lib.stack && lib.pair[i].is_some() && source::is_raw(&lib.files[i]))
            .collect();
        let selected = lib.view.get(self.selected).copied();
        let ring = cx.theme().palette().ring;
        let loader = lib.loader.clone();

        let grid = virtual_grid(
            lib.view.len(),
            cols,
            TILE_H + CAPTION_H,
            TILE_GAP,
            move |position| {
                let i = view[position];
                let mut tip = names[i].clone();
                let picture = match (&thumbs[i], &errors[i]) {
                    (Some(img), _) => image(img.clone())
                        .image_fit(ImageFit::Contain)
                        .dynamic_range_limit(DynamicRangeLimit::Standard),
                    (None, Some(err)) => {
                        tip = format!("{} — {err}", names[i]);
                        column([icon("alert-circle")])
                            .align(Align::Center)
                            .justify(Justify::Center)
                    }
                    (None, None) => {
                        loader.request(i);
                        skeleton()
                    }
                };
                let mut caption = vec![text(names[i].clone()).caption().muted()];
                if paired[i] {
                    caption.push(text("+JPG").caption().muted());
                }
                if developed[i] {
                    caption.push(spacer());
                    caption.push(
                        column(Vec::<El>::new())
                            .width(Size::Fixed(6.0))
                            .height(Size::Fixed(6.0))
                            .radius(3.0)
                            .fill(tokens::PRIMARY),
                    );
                }
                let mut cell = column([
                    picture
                        .width(Size::Fill(1.0))
                        .height(Size::Fixed(TILE_H - tokens::SPACE_2))
                        .fill(tokens::CARD)
                        .radius(tokens::RADIUS_MD),
                    row(caption)
                        .height(Size::Fixed(CAPTION_H - tokens::SPACE_2))
                        .align(Align::Center)
                        .padding(Sides::x(tokens::SPACE_2)),
                ])
                .gap(tokens::SPACE_2 / 2.0)
                .key(format!("thumb:{i}"))
                .width(Size::Fill(1.0))
                .height(Size::Fixed(TILE_H + CAPTION_H))
                .padding(tokens::SPACE_2 / 2.0)
                .radius(tokens::RADIUS_MD)
                .focusable()
                .tooltip(tip);
                if Some(i) == selected {
                    cell = cell.stroke(ring).stroke_width(2.0);
                }
                cell
            },
        );

        let body = if lib.view.is_empty() {
            let what = if lib.files.is_empty() {
                format!("no image files in {}", lib.dir.display())
            } else {
                "no files match the filter".to_string()
            };
            column([icon("folder"), text(what).muted()])
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .justify(Justify::Center)
                .fill_size()
        } else {
            grid.key("grid").padding(GRID_PAD).fill_size()
        };

        let dir_name = lib
            .dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| lib.dir.display().to_string());
        let count = if lib.view.len() == lib.files.len() {
            format!("{} images · {} ready", lib.files.len(), lib.loaded)
        } else {
            format!(
                "{} of {} images · {} ready",
                lib.view.len(),
                lib.files.len(),
                lib.loaded
            )
        };
        let stack = button("Stack")
            .key("stack")
            .tooltip("hide JPEGs that have a raw of the same name");
        let mut bar = vec![
            toolbar_title(dir_name),
            toolbar_description(count),
            spacer(),
            text_input_with(
                "filter",
                &lib.query,
                &lib.query_sel,
                TextInputOpts::default()
                    .placeholder("filter: iso>1600 f<2.8 mm>=50 date:2026-09-10 lens…"),
            )
            .width(Size::Fixed(FILTER_WIDTH)),
            tabs_list("sort", &lib.sort, SortKey::ALL.map(|k| (k, k.label()))),
            if lib.stack { stack } else { stack.ghost() },
        ];
        if let Some(n) = &self.notice {
            bar.push(badge(n.clone()).warning());
        }
        bar.push(self.color_mode_badge());
        bar.push(
            button_with_icon("folder", "Open folder")
                .key("open-folder")
                .tooltip("browse a different folder (o)"),
        );
        bar.push(text("Enter to develop · arrows to move").caption().muted());

        page([
            toolbar(bar)
                .gap(tokens::SPACE_3)
                .padding(Sides::xy(tokens::SPACE_4, tokens::SPACE_2)),
            card([body]).fill_size(),
        ])
    }

    /// The sidebar in info mode: every metadata group as name/value rows.
    fn metadata_panel(&self, s: &Session) -> El {
        let mut sections: Vec<El> = Vec::new();
        match &s.frame {
            None => sections.push(sidebar_group([text("loading…").muted()])),
            Some(f) if f.metadata().is_empty() => {
                sections.push(sidebar_group([text("no metadata").muted()]))
            }
            Some(f) => {
                for group in &f.metadata().groups {
                    let mut rows: Vec<El> = vec![sidebar_group_label(group.name)];
                    for field in &group.fields {
                        rows.push(
                            row([
                                text(field.name.clone()).caption().muted(),
                                spacer(),
                                text(field.value.clone()).caption(),
                            ])
                            .gap(tokens::SPACE_3),
                        );
                    }
                    sections.push(sidebar_group(rows));
                }
            }
        }
        sidebar([scroll([column(sections).gap(tokens::SPACE_4)]).scrollbar()])
            .width(Size::Fixed(SIDEBAR_WIDTH))
    }

    fn control_panel(&self, s: &Session) -> El {
        if self.info {
            return self.metadata_panel(s);
        }
        let tone_kind = match s.params.tone {
            ToneCurve::None => "none",
            ToneCurve::Sigmoid { .. } => "sigmoid",
        };
        let highlights = match s.params.highlights {
            Highlights::Clip => "clip",
            Highlights::Blend => "blend",
        };
        let demosaic = match s.params.demosaic {
            Demosaic::Bilinear => "bilinear",
            Demosaic::Rcd => "rcd",
        };
        let wb_label = match s.params.white_balance {
            WhiteBalance::AsShot => "as shot",
            WhiteBalance::Multipliers { .. } => "multipliers",
            WhiteBalance::Temperature { .. } => "custom",
        };

        let mut info: Vec<El> = Vec::new();
        if let Some(f) = &s.frame {
            if let Source::Image(i) = f.as_ref() {
                info.push(text(i.encoding.clone()).caption().muted());
            }
            let e = f.exif();
            if !e.make.is_empty() || !e.model.is_empty() {
                info.push(text(format!("{} {}", e.make, e.model)).caption().muted());
            }
            if let Some(l) = &e.lens {
                info.push(text(l.clone()).caption().muted());
            }
            let exp = e
                .exposure_time
                .map(|(n, d)| format!("{n}/{d} s"))
                .unwrap_or_default();
            let fnum = e.f_number.map(|v| format!("f/{v:.1}")).unwrap_or_default();
            let iso = e.iso.map(|v| format!("ISO {v}")).unwrap_or_default();
            if !(exp.is_empty() && fnum.is_empty() && iso.is_empty()) {
                info.push(text(format!("{exp}  {fnum}  {iso}")).caption().muted());
            }
            let (w, h) = f.size();
            info.push(text(format!("{w} x {h}")).caption().muted());
        }
        // Sensor-space controls only apply to raws.
        let is_raw = source::is_raw(&s.path);

        let tabs = |key: &str, current: &str, options: [(&str, &str); 2]| {
            tabs_list(
                key,
                &current,
                options.map(|(v, l)| (v.to_string(), l.to_string())),
            )
            .width(Size::Fixed(TABS_WIDTH))
        };
        let mut sections: Vec<El> = vec![
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
                field_row("White", number_scrubber("white", &self.white.text)).tooltip(
                    "scene value mapped to the display peak; 1 = diffuse white, 4 = two stops above",
                ),
            ]),
        ];
        if is_raw {
            sections.push(sidebar_group([
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
            ]));
            sections.push(sidebar_group([
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
            ]));
            sections.push(sidebar_group(lens_group(s)));
        }
        let crop_button = button("Crop")
            .key("crop")
            .tooltip("draw the crop on the image (drag; middle-drag pans); c");
        sections.push(sidebar_group([
            sidebar_group_label("Geometry"),
            field_row("Straighten (°)", number_scrubber("angle", &self.angle.text)),
            field_row(
                "Crop",
                row([
                    if self.crop_mode {
                        crop_button
                    } else {
                        crop_button.ghost()
                    },
                    button("Reset").key("crop-reset").ghost(),
                ])
                .gap(tokens::SPACE_2),
            ),
            tabs_list(
                "aspect",
                &self.crop_aspect.key(),
                Aspect::ALL.map(|a| (a.key().to_string(), a.label().to_string())),
            )
            .width(Size::Fill(1.0)),
        ]));
        sections.push(sidebar_group([
            sidebar_group_label("Detail"),
            field_row(
                "Sharpen",
                number_scrubber("sharpen-amount", &self.sharpen_amount.text),
            ),
            field_row(
                "Radius (px)",
                number_scrubber("sharpen-radius", &self.sharpen_radius.text),
            ),
            field_row(
                "Threshold",
                number_scrubber("sharpen-threshold", &self.sharpen_threshold.text),
            ),
        ]));
        sections.extend([
            sidebar_group([
                sidebar_group_label("Export"),
                row([
                    button("JPEG").key("export-jpg"),
                    button("PNG").key("export-png"),
                    button("AVIF").key("export-avif"),
                    button("EXR")
                        .key("export-exr")
                        .tooltip("scRGB linear f16, current parameters"),
                ])
                .gap(tokens::SPACE_2),
                row([
                    button("HDR PNG")
                        .key("export-hdr-png")
                        .tooltip("BT.2020 PQ 16-bit, 1000-nit master"),
                    button("HDR AVIF")
                        .key("export-hdr-avif")
                        .tooltip("BT.2020 PQ 10-bit, 1000-nit master"),
                ])
                .gap(tokens::SPACE_2),
                if self.exports_running > 0 {
                    row([spinner(), text("exporting…").muted()]).gap(tokens::SPACE_2)
                } else {
                    row(Vec::<El>::new())
                },
                field_row(
                    "Long edge (px, 0 = full)",
                    number_scrubber("export-edge", &self.export_edge.text),
                ),
                button("Reset parameters").key("reset").ghost(),
            ]),
            sidebar_group(info),
        ]);
        sidebar([scroll([column(sections).gap(tokens::SPACE_4)]).scrollbar()])
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

    fn canvas(&self, s: &Session) -> El {
        let Some(frame) = &s.frame else {
            let msg = match &s.load_error {
                Some(e) => text(e.clone()).text_color(tokens::DESTRUCTIVE),
                None => text(format!("Unpacking {}…", s.name)).muted(),
            };
            return column([spinner(), msg])
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .justify(Justify::Center)
                .fill_size();
        };
        let shown = geometry::layout(
            frame.size(),
            frame.orientation(),
            &self.preview_params(s).geometry,
        );
        let (w, h) = (shown.width as f32, shown.height as f32);
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
        let content = content.key("image");
        let content = if self.crop_mode {
            // The crop outline over the full frame, in content pixels.
            // The dimming outside it is done by the colour pass
            // (`Render::dim_outside`): translucent fills over an app
            // surface do not composite darker in damascene 0.8.
            let real = geometry::layout(frame.size(), frame.orientation(), &s.params.geometry);
            let [cx, cy, cw, ch] = real.crop;
            let gap = || column(Vec::<El>::new());
            let overlay = column([
                gap().width(Size::Fill(1.0)).height(Size::Fixed(cy)),
                row([
                    gap().width(Size::Fixed(cx)).height(Size::Fill(1.0)),
                    gap()
                        .width(Size::Fixed(cw))
                        .height(Size::Fill(1.0))
                        .stroke(tokens::PRIMARY)
                        .stroke_width(2.0),
                    gap().width(Size::Fill(1.0)).height(Size::Fill(1.0)),
                ])
                .height(Size::Fixed(ch)),
                gap().width(Size::Fill(1.0)).height(Size::Fill(1.0)),
            ])
            .width(Size::Fixed(w))
            .height(Size::Fixed(h));
            stack([content, overlay])
                .width(Size::Fixed(w))
                .height(Size::Fixed(h))
        } else {
            content
        };
        let pan = if self.crop_mode {
            PointerButton::Middle
        } else {
            PointerButton::Primary
        };
        viewport([content])
            .key("canvas")
            .fit_policy(FitPolicy::Contain {
                padding: FIT_PADDING,
            })
            .pan_button(pan)
            .min_zoom(0.02)
            .max_zoom(8.0)
            .fill(tokens::CARD)
    }

    /// The parameters the preview renders with: in crop mode the whole
    /// straightened frame, so the crop can be drawn over it.
    fn preview_params(&self, s: &Session) -> DevelopParams {
        let mut p = s.params.clone();
        if self.crop_mode {
            p.geometry.crop = Some([0.0, 0.0, 1.0, 1.0]);
        }
        p
    }

    /// A pointer position as fractions of the image node's rect.
    fn image_fraction(event: &UiEvent) -> Option<(f32, f32)> {
        let (px, py) = event.pointer?;
        let r = event.target.as_ref()?.rect;
        if r.w <= 0.0 || r.h <= 0.0 {
            return None;
        }
        Some((
            ((px - r.x) / r.w).clamp(0.0, 1.0),
            ((py - r.y) / r.h).clamp(0.0, 1.0),
        ))
    }

    /// The crop rectangle from the drag anchor to `to`, honouring the
    /// aspect lock (in view pixels, so `view` matters).
    fn crop_from_drag(&self, anchor: (f32, f32), to: (f32, f32), view: (u32, u32)) -> [f32; 4] {
        let (vw, vh) = (view.0 as f32, view.1 as f32);
        let mut w = (to.0 - anchor.0).abs().max(0.02);
        let mut h = (to.1 - anchor.1).abs().max(0.02);
        if let Some(r) = self.crop_aspect.ratio() {
            // Keep the longer drag axis, derive the other from the ratio.
            let h_for_w = w * vw / (r * vh);
            if h_for_w <= h {
                h = h_for_w;
            } else {
                w = h * vh * r / vw;
            }
        }
        let x = if to.0 >= anchor.0 {
            anchor.0
        } else {
            anchor.0 - w
        };
        let y = if to.1 >= anchor.1 {
            anchor.1
        } else {
            anchor.1 - h
        };
        let x = x.clamp(0.0, 1.0 - w.min(1.0));
        let y = y.clamp(0.0, 1.0 - h.min(1.0));
        [x, y, w.min(1.0 - x), h.min(1.0 - y)]
    }

    fn develop(&self, cx: &BuildCx, s: &Session) -> El {
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
        let position = format!("{} / {}", s.index + 1, self.library.files.len());

        page([
            toolbar([
                button_with_icon("layout-dashboard", "Library")
                    .key("back")
                    .ghost()
                    .tooltip("back to the folder (Esc)"),
                toolbar_title(s.name.clone()),
                toolbar_description(position),
                spacer(),
                text(readout).muted().tabular_numerals(),
                button("Fit").key("fit").ghost(),
                button("1:1").key("one").ghost(),
                self.info_button(),
                self.warn_button(),
                self.color_mode_badge(),
            ])
            .gap(tokens::SPACE_3)
            .padding(Sides::xy(tokens::SPACE_4, tokens::SPACE_2)),
            row([self.canvas(s), self.control_panel(s)])
                .gap(tokens::SPACE_3)
                .height(Size::Fill(1.0)),
        ])
    }

    fn on_develop_event(&mut self, event: &UiEvent) {
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
        if event.is_click_or_activate("warn") || event.is_hotkey("warn") {
            self.warn = !self.warn;
            wake(&self.wakeup);
            return;
        }
        if event.is_click_or_activate("info") || event.is_hotkey("info") {
            self.info = !self.info;
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
        if event.is_click_or_activate("export-avif") {
            self.export(ExportKind::Avif);
            return;
        }
        if event.is_click_or_activate("export-hdr-avif") {
            self.export(ExportKind::HdrAvif);
            return;
        }
        if event.is_click_or_activate("reset") {
            self.reset_params();
            return;
        }
        if event.is_click_or_activate("crop") || event.is_hotkey("crop") {
            self.crop_mode = !self.crop_mode;
            self.crop_anchor = None;
            wake(&self.wakeup);
            return;
        }
        if event.is_click_or_activate("crop-reset") {
            if let Some(s) = self.session.as_mut() {
                s.params.geometry = idev_core::params::Geometry::default();
            }
            self.seed_controls();
            self.params_changed();
            return;
        }
        let mut aspect = self.crop_aspect;
        if tabs::apply_event(&mut aspect, event, "aspect", Aspect::parse) {
            self.crop_aspect = aspect;
            // Refit an existing crop to the new ratio about its centre.
            if let Some(r) = aspect.ratio()
                && let Some(s) = self.session.as_mut()
                && let (Some(frame), Some([x, y, w, h])) = (&s.frame, s.params.geometry.crop)
            {
                let view = geometry::oriented_size(frame.size(), frame.orientation());
                let (vw, vh) = (view.0 as f32, view.1 as f32);
                let mut nw = w;
                let mut nh = w * vw / (r * vh);
                if nh > 1.0 {
                    nh = 1.0;
                    nw = nh * vh * r / vw;
                }
                let nx = (x + w * 0.5 - nw * 0.5).clamp(0.0, 1.0 - nw);
                let ny = (y + h * 0.5 - nh * 0.5).clamp(0.0, 1.0 - nh);
                s.params.geometry.crop = Some([nx, ny, nw, nh]);
                self.params_changed();
            }
            wake(&self.wakeup);
            return;
        }
        if self.crop_mode && event.is_route("image") {
            match event.kind {
                UiEventKind::PointerDown => {
                    self.crop_anchor = Self::image_fraction(event);
                    return;
                }
                UiEventKind::Drag => {
                    if let (Some(anchor), Some(to)) =
                        (self.crop_anchor, Self::image_fraction(event))
                        && let Some(s) = self.session.as_mut()
                        && let Some(frame) = &s.frame
                    {
                        let view = geometry::oriented_size(frame.size(), frame.orientation());
                        let rect = self.crop_from_drag(anchor, to, view);
                        if let Some(s) = self.session.as_mut() {
                            s.params.geometry.crop = Some(rect);
                        }
                        self.params_changed();
                    }
                    return;
                }
                UiEventKind::PointerUp => {
                    self.crop_anchor = None;
                    return;
                }
                _ => {}
            }
        }
        let edge = ScrubberOpts::default()
            .min(0.0)
            .max(20000.0)
            .step(100.0)
            .sensitivity(20.0)
            .decimals(0);
        if number_scrubber::apply_event(
            &mut self.export_edge.text,
            &mut self.export_edge.drag,
            "export-edge",
            &edge,
            event,
        ) {
            wake(&self.wakeup);
            return;
        }
        let amount = ScrubberOpts::default()
            .min(0.0)
            .max(3.0)
            .step(0.05)
            .sensitivity(4.0)
            .decimals(2);
        if number_scrubber::apply_event(
            &mut self.sharpen_amount.text,
            &mut self.sharpen_amount.drag,
            "sharpen-amount",
            &amount,
            event,
        ) {
            if let (Some(v), Some(s)) = (self.sharpen_amount.value(), self.session.as_mut()) {
                s.params.sharpen.amount = v.clamp(0.0, 3.0);
                self.params_changed();
            }
            return;
        }
        let radius = ScrubberOpts::default()
            .min(0.3)
            .max(3.0)
            .step(0.05)
            .sensitivity(4.0)
            .decimals(2);
        if number_scrubber::apply_event(
            &mut self.sharpen_radius.text,
            &mut self.sharpen_radius.drag,
            "sharpen-radius",
            &radius,
            event,
        ) {
            if let (Some(v), Some(s)) = (self.sharpen_radius.value(), self.session.as_mut()) {
                s.params.sharpen.radius = v.clamp(0.3, 3.0);
                self.params_changed();
            }
            return;
        }
        let threshold = ScrubberOpts::default()
            .min(0.0)
            .max(0.1)
            .step(0.002)
            .sensitivity(6.0)
            .decimals(3);
        if number_scrubber::apply_event(
            &mut self.sharpen_threshold.text,
            &mut self.sharpen_threshold.drag,
            "sharpen-threshold",
            &threshold,
            event,
        ) {
            if let (Some(v), Some(s)) = (self.sharpen_threshold.value(), self.session.as_mut()) {
                s.params.sharpen.threshold = v.clamp(0.0, 0.1);
                self.params_changed();
            }
            return;
        }
        let angle = ScrubberOpts::default()
            .min(-45.0)
            .max(45.0)
            .step(0.1)
            .sensitivity(4.0)
            .decimals(2);
        if number_scrubber::apply_event(
            &mut self.angle.text,
            &mut self.angle.drag,
            "angle",
            &angle,
            event,
        ) {
            if let (Some(v), Some(s)) = (self.angle.value(), self.session.as_mut()) {
                s.params.geometry.angle_deg = v.clamp(-45.0, 45.0);
                self.params_changed();
            }
            return;
        }
        if event.is_click_or_activate("wb-as-shot") {
            if let Some(s) = self.session.as_mut() {
                s.params.white_balance = WhiteBalance::AsShot;
            }
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
            event,
        ) {
            if let (Some(v), Some(s)) = (self.exposure.value(), self.session.as_mut()) {
                s.params.exposure_ev = v;
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
            event,
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
        if number_scrubber::apply_event(&mut self.hue.text, &mut self.hue.drag, "hue", &hue, event)
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
            event,
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
            event,
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
            event,
        ) {
            self.set_temperature();
            return;
        }

        let Some(s) = self.session.as_mut() else {
            return;
        };
        if switch::apply_event(&mut s.params.lens.distortion, event, "lens-distortion")
            || switch::apply_event(&mut s.params.lens.chromatic_aberration, event, "lens-ca")
            || switch::apply_event(&mut s.params.lens.vignetting, event, "lens-vignetting")
        {
            self.params_changed();
            return;
        }
        let mut tone_kind = String::new();
        if tabs::apply_event(&mut tone_kind, event, "tone", |s| Some(s.to_string())) {
            s.params.tone = match tone_kind.as_str() {
                "none" => ToneCurve::None,
                _ => match s.params.tone {
                    ToneCurve::Sigmoid { .. } => s.params.tone,
                    ToneCurve::None => ToneCurve::default(),
                },
            };
            self.seed_controls();
            self.params_changed();
            return;
        }
        if tabs::apply_event(&mut s.params.highlights, event, "highlights", |s| match s {
            "clip" => Some(Highlights::Clip),
            "blend" => Some(Highlights::Blend),
            _ => None,
        }) {
            self.params_changed();
            return;
        }
        if tabs::apply_event(&mut s.params.demosaic, event, "demosaic", |s| match s {
            "bilinear" => Some(Demosaic::Bilinear),
            "rcd" => Some(Demosaic::Rcd),
            _ => None,
        }) {
            self.params_changed();
        }
    }

    fn on_library_event(&mut self, event: &UiEvent) {
        // View controls first: the filter box owns keys while focused.
        let lib = &mut self.library;
        if text_input::apply_event(&mut lib.query, &mut lib.query_sel, event, "filter") {
            self.refresh_view();
            return;
        }
        if event.kind == UiEventKind::Escape && lib.filter_focused() {
            lib.query_sel = Selection::default();
            return;
        }
        if event.kind == UiEventKind::PointerDown && !event.is_route("filter") {
            lib.query_sel = Selection::default();
        }
        if tabs::apply_event(&mut lib.sort, event, "sort", SortKey::parse) {
            self.refresh_view();
            return;
        }
        if event.is_click_or_activate("stack") {
            lib.stack = !lib.stack;
            self.refresh_view();
            return;
        }
        if let Some(key) = event.target_key()
            && let Some(i) = key
                .strip_prefix("thumb:")
                .and_then(|s| s.parse::<usize>().ok())
            && let Some(position) = self.library.view.iter().position(|&f| f == i)
        {
            if event.kind == UiEventKind::Click && event.click_count >= 2 {
                self.select(position);
                self.open_selected();
                return;
            }
            if event.is_click_or_activate(key) {
                self.select(position);
                return;
            }
        }
        let last = self.library.view.len().saturating_sub(1);
        let cols = self.cols.get().max(1) as isize;
        if event.is_hotkey("open") {
            self.open_selected();
        } else if event.is_hotkey("up") {
            self.step(-cols);
        } else if event.is_hotkey("down") {
            self.step(cols);
        } else if event.is_hotkey("home") {
            self.select(0);
        } else if event.is_hotkey("end") {
            self.select(last);
        }
    }
}

impl App for IdevApp {
    fn before_build(&mut self) {
        self.poll_background();
    }

    fn build(&self, cx: &BuildCx) -> El {
        if let Some(diag) = cx.diagnostics() {
            self.scale_factor.set(diag.scale_factor.max(0.1));
            let hdr = diag.hdr_active();
            self.hdr_output.set(hdr);
            let mut peak = 1.0;
            let mut gamut = Gamut::Srgb;
            if let damascene_core::color::ColorManagementStatus::Available { targets, .. } =
                &diag.color_management
            {
                if hdr {
                    let max = targets
                        .target_max_luminance_nits
                        .or(targets.max_luminance_nits)
                        .unwrap_or(1000.0);
                    let reference = targets.reference_luminance_nits.unwrap_or(203.0);
                    peak = (max / reference).max(1.0);
                }
                gamut = match targets.preferred_primaries {
                    Some(Primaries::DisplayP3) => Gamut::DisplayP3,
                    Some(Primaries::Bt2020) => Gamut::Bt2020,
                    Some(Primaries::AdobeRgb) => Gamut::AdobeRgb,
                    Some(Primaries::Srgb) | None => Gamut::Srgb,
                };
            }
            self.display_peak.set(peak);
            self.panel_gamut.set(gamut);
        }
        match (self.mode, &self.session) {
            (Mode::Develop, Some(s)) => self.develop(cx, s),
            _ => self.grid(cx),
        }
    }

    fn selection(&self) -> Selection {
        self.library.query_sel.clone()
    }

    fn hotkeys(&self) -> Vec<(KeyChord, String)> {
        // Hotkeys beat text editing in the runtime, so none while the
        // filter box is being typed into.
        if self.mode == Mode::Library && self.library.filter_focused() {
            return Vec::new();
        }
        let mut keys = vec![
            (KeyChord::vim('t'), "sdr-preview".into()),
            (KeyChord::vim('o'), "open-folder".into()),
            (named(NamedKey::ArrowLeft), "left".into()),
            (named(NamedKey::ArrowRight), "right".into()),
            (KeyChord::vim('h'), "left".into()),
            (KeyChord::vim('l'), "right".into()),
        ];
        match self.mode {
            Mode::Library => keys.extend([
                (named(NamedKey::ArrowUp), "up".into()),
                (named(NamedKey::ArrowDown), "down".into()),
                (KeyChord::vim('j'), "down".into()),
                (KeyChord::vim('k'), "up".into()),
                (named(NamedKey::Home), "home".into()),
                (named(NamedKey::End), "end".into()),
                (named(NamedKey::Enter), "open".into()),
            ]),
            Mode::Develop => keys.extend([
                (KeyChord::vim('f'), "fit".into()),
                (KeyChord::vim('1'), "one".into()),
                (KeyChord::vim('e'), "export-jpg".into()),
                (KeyChord::vim('w'), "warn".into()),
                (KeyChord::vim('i'), "info".into()),
                (KeyChord::vim('c'), "crop".into()),
            ]),
        }
        keys
    }

    fn on_event(&mut self, event: UiEvent, _cx: &EventCx) {
        if event.is_click_or_activate("open-folder") || event.is_hotkey("open-folder") {
            self.open_folder_dialog();
            return;
        }
        if event.is_hotkey("sdr-preview") || event.is_click_or_activate("color-mode") {
            self.sdr_preview = !self.sdr_preview;
            wake(&self.wakeup);
            return;
        }
        if event.is_hotkey("left") {
            self.step(-1);
            return;
        }
        if event.is_hotkey("right") {
            self.step(1);
            return;
        }
        match self.mode {
            Mode::Develop => {
                if event.kind == UiEventKind::Escape || event.is_click_or_activate("back") {
                    self.mode = Mode::Library;
                    self.close_session();
                    self.select(self.selected);
                    return;
                }
                self.on_develop_event(&event);
            }
            Mode::Library => self.on_library_event(&event),
        }
    }

    fn drain_toasts(&mut self) -> Vec<ToastSpec> {
        std::mem::take(&mut self.toasts)
    }

    fn drain_viewport_requests(&mut self) -> Vec<ViewportRequest> {
        std::mem::take(&mut self.viewport_requests)
    }

    fn drain_scroll_requests(&mut self) -> Vec<scroll::ScrollRequest> {
        std::mem::take(&mut self.scroll_requests)
    }
}

impl WinitWgpuApp for IdevApp {
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
        if self.mode != Mode::Develop {
            return;
        }
        let Some(s) = &self.session else { return };
        let Some(frame) = s.frame.clone() else {
            return;
        };
        let params = self.preview_params(s);
        let dim_outside = self
            .crop_mode
            .then(|| geometry::layout(frame.size(), frame.orientation(), &s.params.geometry).crop);
        let render = Render {
            target: OutputTarget { peak: self.peak() },
            scale: self.wanted_scale(),
            display: DisplayPrimaries::Srgb,
            warn: self.warn.then(|| self.panel_gamut.get()),
            dim_outside,
        };
        let scale = render.scale;
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
            let (w, h) = Developer::output_size(prepared, &params, scale);
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

        let want = (params, render);
        if g.rendered.as_ref() != Some(&want) {
            g.dev
                .render(&g.gpu, prepared, &want.0, render, &preview.view);
            let (w, h) = Developer::output_size(prepared, &want.0, scale);
            g.rendered = Some(want);
            // One histogram in flight at a time; a newer render just
            // waits for the next frame. The zebra overlay would pollute
            // it, so the last clean histogram stays while warning.
            if g.histogram_pending.is_none() && render.warn.is_none() {
                g.histogram_pending =
                    Some(
                        g.dev
                            .histogram(&g.gpu, &preview.view, w, h, render.target.peak),
                    );
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

/// The lens sidebar group: three switches, or a note when the file has
/// no correction tables.
fn lens_group(s: &Session) -> Vec<El> {
    let mut items = vec![sidebar_group_label("Lens")];
    match s
        .frame
        .as_ref()
        .map(|f| f.as_raw().is_some_and(|r| r.lens.is_some()))
    {
        Some(true) => {
            let l = s.params.lens;
            items.push(field_row(
                "Distortion",
                switch("lens-distortion", l.distortion),
            ));
            items.push(field_row(
                "Chromatic aberration",
                switch("lens-ca", l.chromatic_aberration),
            ));
            items.push(field_row(
                "Vignetting",
                switch("lens-vignetting", l.vignetting),
            ));
        }
        Some(false) => items.push(text("no correction tables in this file").caption().muted()),
        None => items.push(skeleton()),
    }
    items
}

/// Develop at full resolution on a private device and encode.
fn export_to(
    frame: &Source,
    source_name: &str,
    params: &DevelopParams,
    kind: &ExportKind,
    long_edge: Option<u32>,
    path: &Path,
) -> Result<()> {
    let t = Instant::now();
    let gpu = Gpu::headless()?;
    let dev = Developer::new(&gpu);
    let target = match kind {
        ExportKind::HdrPng | ExportKind::HdrAvif => OutputTarget::hdr_nits(HDR_EXPORT_NITS),
        _ => OutputTarget::SDR,
    };
    // Oriented, straightened and cropped by the colour pass itself.
    let img: WorkingImage = dev.develop_sized(&gpu, frame, params, target, long_edge)?;
    let note = metadata::develop_note(source_name, params);
    let exif = metadata::exif_for_output(
        &frame.metadata().tags,
        &metadata::Output {
            width: img.width,
            height: img.height,
            srgb: matches!(kind, ExportKind::Jpeg | ExportKind::Png | ExportKind::Avif),
            note: &note,
        },
    )?;
    let exif = Some(exif.as_slice());
    match kind {
        ExportKind::Jpeg => idev_formats::write_jpeg_srgb8(
            path,
            img.width,
            img.height,
            &output::srgb8(&img),
            EXPORT_QUALITY,
            exif,
        )?,
        ExportKind::Png => {
            idev_formats::write_png_srgb8(path, img.width, img.height, &output::srgb8(&img), exif)?
        }
        ExportKind::HdrPng => idev_formats::write_png_bt2020_pq16(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(&img),
            exif,
        )?,
        ExportKind::Avif => idev_formats::write_avif_srgb8(
            path,
            img.width,
            img.height,
            &output::srgb8(&img),
            EXPORT_QUALITY,
            exif,
        )?,
        ExportKind::HdrAvif => idev_formats::write_avif_bt2020_pq10(
            path,
            img.width,
            img.height,
            &output::bt2020_pq16(&img),
            EXPORT_QUALITY,
            exif,
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
