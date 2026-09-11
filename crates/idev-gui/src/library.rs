//! The folder being browsed: its raw files and the worker pool that
//! thumbnails them off the UI thread.
//!
//! Lifted from damascene-gallery's loader with one job kind. Two queues:
//! files the grid realized (hot, most recent first) ahead of a sweep over
//! the folder in file order. Each result pokes the host's wakeup so the
//! idle event loop renders a frame.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};

use crate::app::{SharedWakeup, wake};
use crate::thumbs::{Thumb, ThumbCache, ThumbGpu};

/// Raw extensions worth listing. The pipeline supports 2x2 Bayer
/// mosaics that rawler decodes; anything else fails at unpack and shows
/// as an error tile.
pub const EXTENSIONS: &[&str] = &[
    "arw", "dng", "cr2", "cr3", "nef", "nrw", "raf", "orf", "rw2", "pef", "srw",
];

/// Raw files directly inside `dir` (not recursive), sorted by name.
pub fn scan_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?
    {
        let p = entry?.path();
        let ext = p
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if p.is_file() && EXTENSIONS.contains(&ext.as_str()) {
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
                .name(format!("idev-thumb-{n}"))
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
