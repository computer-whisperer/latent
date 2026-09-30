//! The folder being browsed: its raw files and the worker pool that
//! thumbnails them off the UI thread.
//!
//! Lifted from damascene-gallery's loader with one job kind. Two queues:
//! files the grid realized (hot, most recent first) ahead of a sweep over
//! the folder in file order. Each result pokes the host's wakeup so the
//! idle event loop renders a frame.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use latent_core::sensor::ExifSummary;
use latent_core::source::is_raw;

use crate::app::{SharedWakeup, wake};
use crate::thumbs::{Thumb, ThumbCache, ThumbGpu};

/// Image files directly inside `dir` (not recursive), sorted by name:
/// camera raws and the rendered formats achromat decodes.
pub fn scan_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?
    {
        let p = entry?.path();
        if p.is_file() && latent_core::source::is_supported(&p) {
            files.push(p);
        }
    }
    files.sort();
    Ok(files)
}

pub enum Loaded {
    Thumb { index: usize, thumb: Thumb },
    Failed { index: usize, error: String },
}

struct Queue {
    hot: VecDeque<usize>,
    sweep: VecDeque<usize>,
    /// Queued or in progress, to dedupe requests.
    pending: HashSet<usize>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    cond: Condvar,
    results: Mutex<Sender<Loaded>>,
    wakeup: SharedWakeup,
    files: Vec<PathBuf>,
    cache: Arc<ThumbCache>,
    gpu: Arc<ThumbGpu>,
}

/// Handle owned by the app; clones go into the grid's cell builder.
#[derive(Clone)]
pub struct Loader {
    shared: Arc<Shared>,
}

pub struct LoaderResults {
    rx: Receiver<Loaded>,
}

impl LoaderResults {
    pub fn drain(&self) -> Vec<Loaded> {
        self.rx.try_iter().collect()
    }
}

impl Loader {
    /// Spawn `workers` threads over `files`; the whole folder is queued
    /// for the sweep immediately.
    pub fn spawn(
        files: Vec<PathBuf>,
        workers: usize,
        wakeup: SharedWakeup,
        cache: Arc<ThumbCache>,
        gpu: Arc<ThumbGpu>,
    ) -> (Loader, LoaderResults) {
        let (tx, rx) = channel();
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                hot: VecDeque::new(),
                sweep: (0..files.len()).collect(),
                pending: (0..files.len()).collect(),
                shutdown: false,
            }),
            cond: Condvar::new(),
            results: Mutex::new(tx),
            wakeup,
            files,
            cache,
            gpu,
        });
        for n in 0..workers {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("latent-thumb-{n}"))
                .spawn(move || worker(shared))
                .expect("spawning thumbnail worker");
        }
        (Loader { shared }, LoaderResults { rx })
    }

    /// Clear the queues and wake every worker so it exits. Jobs already
    /// running finish and post into the dropped channel, harmlessly.
    pub fn shutdown(&self) {
        let mut q = self.shared.queue.lock().unwrap();
        q.shutdown = true;
        q.hot.clear();
        q.sweep.clear();
        self.shared.cond.notify_all();
    }

    /// Jump a file to the front of the queue (the grid realized it).
    /// No-op if done, already hot, or in progress.
    pub fn request(&self, index: usize) {
        if index >= self.shared.files.len() {
            return;
        }
        let mut q = self.shared.queue.lock().unwrap();
        if q.pending.contains(&index) {
            if let Some(pos) = q.sweep.iter().position(|&i| i == index) {
                q.sweep.remove(pos);
                q.hot.push_front(index);
                self.shared.cond.notify_one();
            }
            return;
        }
        q.pending.insert(index);
        q.hot.push_front(index);
        self.shared.cond.notify_one();
    }

    /// Re-thumbnail a file whose sidecar changed. Unlike [`Self::request`]
    /// this queues even when a result already exists.
    pub fn invalidate(&self, index: usize) {
        if index >= self.shared.files.len() {
            return;
        }
        let mut q = self.shared.queue.lock().unwrap();
        if q.pending.insert(index) {
            q.hot.push_front(index);
            self.shared.cond.notify_one();
        }
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let index = {
            let mut q = shared.queue.lock().unwrap();
            loop {
                if q.shutdown {
                    return;
                }
                if let Some(i) = q.hot.pop_front().or_else(|| q.sweep.pop_front()) {
                    break i;
                }
                q = shared.cond.wait(q).unwrap();
            }
        };
        let path = &shared.files[index];
        let result = match shared.cache.thumbnail(path, &shared.gpu) {
            Ok(thumb) => Loaded::Thumb { index, thumb },
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %format!("{e:#}"), "thumbnail failed");
                Loaded::Failed {
                    index,
                    error: format!("{e:#}"),
                }
            }
        };
        shared.queue.lock().unwrap().pending.remove(&index);
        if shared.results.lock().unwrap().send(result).is_err() {
            return;
        }
        wake(&shared.wakeup);
    }
}

// ---------------------------------------------------------------------
// View: sort, filter, stack

/// What the grid is ordered by. Files without the key sort last; ties
/// break by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortKey {
    #[default]
    Name,
    Date,
    Iso,
    Focal,
    Lens,
}

impl SortKey {
    pub const ALL: [SortKey; 5] = [
        SortKey::Name,
        SortKey::Date,
        SortKey::Iso,
        SortKey::Focal,
        SortKey::Lens,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SortKey::Name => "Name",
            SortKey::Date => "Date",
            SortKey::Iso => "ISO",
            SortKey::Focal => "Focal",
            SortKey::Lens => "Lens",
        }
    }

    pub fn parse(s: &str) -> Option<SortKey> {
        SortKey::ALL.into_iter().find(|k| k.label() == s)
    }
}

impl std::fmt::Display for SortKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// One file as the view sees it.
pub struct Item<'a> {
    pub name: &'a str,
    pub raw: bool,
    pub exif: Option<&'a ExifSummary>,
}

/// A parsed filter query: whitespace-separated terms, all of which
/// must hold.
///
/// - `iso>=1600`, `iso<800`, `f<2.8`, `f>=4`, `mm>=50`, `mm<24`
///   (also `=`, `>`, `<=`);
/// - `date:2026-09-10` — prefix of the shooting date;
/// - `raw` / `jpg` — file kind;
/// - anything else — case-insensitive substring of the file name,
///   lens or camera model.
///
/// Terms that need metadata reject files whose summary has not loaded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Filter {
    terms: Vec<Term>,
}

#[derive(Debug, Clone, PartialEq)]
enum Term {
    Number { field: Field, op: Op, value: f32 },
    Date(String),
    Raw(bool),
    Text(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Field {
    Iso,
    Aperture,
    Focal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

impl Filter {
    pub fn parse(query: &str) -> Filter {
        let terms = query.split_whitespace().map(Term::parse).collect();
        Filter { terms }
    }

    pub fn matches(&self, item: &Item) -> bool {
        self.terms.iter().all(|t| t.matches(item))
    }
}

impl Term {
    fn parse(word: &str) -> Term {
        let lower = word.to_lowercase();
        if let Some(date) = lower.strip_prefix("date:") {
            return Term::Date(date.replace(':', "-"));
        }
        match lower.as_str() {
            "raw" => return Term::Raw(true),
            "jpg" | "jpeg" | "image" => return Term::Raw(false),
            _ => {}
        }
        for (prefix, field) in [
            ("iso", Field::Iso),
            ("f", Field::Aperture),
            ("mm", Field::Focal),
        ] {
            if let Some(rest) = lower.strip_prefix(prefix) {
                let (op, num) = if let Some(n) = rest.strip_prefix(">=") {
                    (Op::Ge, n)
                } else if let Some(n) = rest.strip_prefix("<=") {
                    (Op::Le, n)
                } else if let Some(n) = rest.strip_prefix('>') {
                    (Op::Gt, n)
                } else if let Some(n) = rest.strip_prefix('<') {
                    (Op::Lt, n)
                } else if let Some(n) = rest.strip_prefix('=') {
                    (Op::Eq, n)
                } else {
                    continue;
                };
                if let Ok(value) = num.parse::<f32>() {
                    return Term::Number { field, op, value };
                }
            }
        }
        Term::Text(lower)
    }

    fn matches(&self, item: &Item) -> bool {
        match self {
            Term::Raw(raw) => item.raw == *raw,
            Term::Text(t) => {
                if item.name.to_lowercase().contains(t) {
                    return true;
                }
                item.exif.is_some_and(|e| {
                    e.lens
                        .as_deref()
                        .is_some_and(|l| l.to_lowercase().contains(t))
                        || e.model.to_lowercase().contains(t)
                        || e.make.to_lowercase().contains(t)
                })
            }
            Term::Date(prefix) => item
                .exif
                .and_then(|e| e.date())
                .is_some_and(|d| d.starts_with(prefix)),
            Term::Number { field, op, value } => {
                let Some(e) = item.exif else { return false };
                let actual = match field {
                    Field::Iso => e.iso.map(|v| v as f32),
                    Field::Aperture => e.f_number,
                    Field::Focal => e.focal_length_mm,
                };
                let Some(actual) = actual else { return false };
                match op {
                    Op::Lt => actual < *value,
                    Op::Le => actual <= *value,
                    Op::Gt => actual > *value,
                    Op::Ge => actual >= *value,
                    Op::Eq => (actual - value).abs() < 0.05,
                }
            }
        }
    }
}

/// Sort `indices` (into `items`) by `key`, files without the key last,
/// ties by name.
pub fn sort_indices(indices: &mut [usize], items: &[Item], key: SortKey) {
    let name = |i: usize| items[i].name;
    match key {
        SortKey::Name => indices.sort_by(|&a, &b| name(a).cmp(name(b))),
        SortKey::Date => sort_by_option(indices, items, |e| e.date()),
        SortKey::Iso => sort_by_option(indices, items, |e| e.iso),
        SortKey::Focal => sort_by_option(indices, items, |e| e.focal_length_mm.map(Ordered)),
        SortKey::Lens => sort_by_option(indices, items, |e| e.lens.clone()),
    }
}

fn sort_by_option<K: Ord>(
    indices: &mut [usize],
    items: &[Item],
    k: impl Fn(&ExifSummary) -> Option<K>,
) {
    indices.sort_by(|&a, &b| {
        let ka = items[a].exif.and_then(&k);
        let kb = items[b].exif.and_then(&k);
        match (ka, kb) {
            (Some(x), Some(y)) => x.cmp(&y).then_with(|| items[a].name.cmp(items[b].name)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => items[a].name.cmp(items[b].name),
        }
    });
}

/// A float with a total order for sorting (NaN sorts last).
#[derive(PartialEq)]
struct Ordered(f32);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Raw + rendered pairs by file stem (case-insensitive): for each file,
/// the index of its partner of the other kind, if any.
pub fn pairs(files: &[PathBuf]) -> Vec<Option<usize>> {
    let stem = |p: &PathBuf| {
        p.file_stem()
            .map(|s| s.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    };
    let mut raws: HashMap<String, usize> = HashMap::new();
    for (i, p) in files.iter().enumerate() {
        if is_raw(p) {
            raws.entry(stem(p)).or_insert(i);
        }
    }
    let mut out = vec![None; files.len()];
    for (i, p) in files.iter().enumerate() {
        if is_raw(p) {
            continue;
        }
        if let Some(&r) = raws.get(&stem(p)) {
            out[i] = Some(r);
            if out[r].is_none() {
                out[r] = Some(i);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exif(iso: u32, f: f32, mm: f32, lens: &str, date: &str) -> ExifSummary {
        ExifSummary {
            make: "SONY".into(),
            model: "ILCE-6700".into(),
            lens: Some(lens.into()),
            f_number: Some(f),
            iso: Some(iso),
            focal_length_mm: Some(mm),
            date_time_original: Some(date.into()),
            ..Default::default()
        }
    }

    #[test]
    fn filter_terms() {
        let e = exif(3200, 3.2, 50.0, "FE 50mm F1.8", "2026:09:10 13:22:37");
        let item = Item {
            name: "DSC00039.ARW",
            raw: true,
            exif: Some(&e),
        };
        let bare = Item {
            name: "DSC00039.ARW",
            raw: true,
            exif: None,
        };
        let yes = |q: &str| assert!(Filter::parse(q).matches(&item), "{q}");
        let no = |q: &str| assert!(!Filter::parse(q).matches(&item), "{q}");
        yes("iso>=3200");
        yes("iso>1600 f<4");
        no("iso>3200");
        yes("f=3.2");
        yes("mm>=50 mm<=50");
        yes("date:2026-09-10");
        yes("date:2026-09");
        no("date:2026-09-11");
        yes("raw");
        no("jpg");
        yes("fe 50mm");
        yes("dsc00039");
        yes("ilce");
        no("nikon");
        yes("");
        assert!(!Filter::parse("iso>100").matches(&bare));
        assert!(Filter::parse("dsc").matches(&bare));
    }

    #[test]
    fn sorting_puts_unknown_last() {
        let a = exif(100, 1.8, 50.0, "FE 50mm F1.8", "2026:09:10 13:00:00");
        let b = exif(6400, 5.6, 16.0, "E 16-50mm", "2026:09:09 08:00:00");
        let items = vec![
            Item {
                name: "c.ARW",
                raw: true,
                exif: None,
            },
            Item {
                name: "b.ARW",
                raw: true,
                exif: Some(&a),
            },
            Item {
                name: "a.ARW",
                raw: true,
                exif: Some(&b),
            },
        ];
        let mut idx = vec![0, 1, 2];
        sort_indices(&mut idx, &items, SortKey::Name);
        assert_eq!(idx, [2, 1, 0]);
        sort_indices(&mut idx, &items, SortKey::Iso);
        assert_eq!(idx, [1, 2, 0]);
        sort_indices(&mut idx, &items, SortKey::Date);
        assert_eq!(idx, [2, 1, 0]);
        sort_indices(&mut idx, &items, SortKey::Focal);
        assert_eq!(idx, [2, 1, 0]);
        sort_indices(&mut idx, &items, SortKey::Lens);
        assert_eq!(idx, [2, 1, 0]);
    }

    #[test]
    fn pairs_by_stem() {
        let files: Vec<PathBuf> = ["DSC00039.ARW", "DSC00039.JPG", "DSC00040.JPG", "x.png"]
            .iter()
            .map(PathBuf::from)
            .collect();
        assert_eq!(pairs(&files), vec![Some(1), Some(0), None, None]);
    }
}
