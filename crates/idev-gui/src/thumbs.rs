//! Grid thumbnails: produced from the raw's embedded camera JPEG, or
//! from a scaled develop once the raw has a sidecar, and cached on the
//! local disk.
//!
//! The cache follows prism-explorer's `explorer-thumbs`: entries are
//! keyed by provenance (source path, mtime, size — here also the
//! sidecar's mtime and size — plus edge and format version) so a touched
//! source or edited sidecar simply never hits its old entry; entries are
//! written to a temp file and renamed into place; anything that fails to
//! parse is deleted and treated as a miss. Unlike the explorer, entries
//! are sRGB RGBA8: developed thumbnails are rendered for the SDR target,
//! so there is nothing above diffuse white to preserve.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use anyhow::{Context, Result};
use damascene_core::image::Image;
use idev_core::WorkingImage;
use idev_core::color::srgb_oetf;
use idev_core::gpu::{Developer, Gpu, OutputTarget};
use idev_core::output;
use idev_core::params::{DevelopParams, sidecar_path};
use idev_core::sensor::{self, ExifSummary};
use idev_core::source::{self, Source};

/// Long edge of a thumbnail in pixels. Tiles are ~240 logical px wide,
/// so this covers a 2x display without upscaling.
pub const THUMB_EDGE: u32 = 512;

/// Bumped whenever the entry encoding or the rendering changes; part of
/// the key, so old entries age out through the sweep.
// 2: rendered images honour their EXIF orientation.
// 3: the EXIF summary rides along for library sorting and filtering.
const FORMAT_VERSION: u32 = 4;
const MAGIC: [u8; 4] = *b"IDTH";
const HEADER_LEN: usize = 4 + 4 + 4 + 4 + 1 + 4;
const MAX_DIM: u32 = 16_384;

/// Bytes the cache may hold before the sweep evicts least recently used
/// entries.
pub const CACHE_BUDGET: u64 = 2 << 30;

/// A thumbnail ready for the grid.
pub struct Thumb {
    pub image: Image,
    /// Rendered from the sidecar (true) or the camera's preview (false).
    pub developed: bool,
    pub exif: ExifSummary,
}

/// The headless device developed thumbnails render on. One full
/// resolution `prepare` holds ~0.5 GB of textures, so workers take turns.
#[derive(Default)]
pub struct ThumbGpu(Mutex<Option<(Gpu, Developer)>>);

pub struct ThumbCache {
    root: PathBuf,
}

/// What a [`ThumbCache::sweep`] found and did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SweepStats {
    pub files: u64,
    pub bytes: u64,
    pub removed_files: u64,
    pub removed_bytes: u64,
}

impl ThumbCache {
    pub fn new(root: PathBuf) -> Self {
        ThumbCache { root }
    }

    /// `$XDG_CACHE_HOME/idev/thumbs` (or `~/.cache/…`): the local cache
    /// directory, never the (possibly network) folder being browsed.
    pub fn standard() -> Option<Self> {
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(Self::new(base.join("idev/thumbs")))
    }

    /// The thumbnail for `raw`: a cache hit if neither the raw nor its
    /// sidecar changed, otherwise produced, stored, and returned. Worker
    /// thread; blocks on IO, decode, and (with a sidecar) the GPU.
    pub fn thumbnail(&self, raw: &Path, gpu: &ThumbGpu) -> Result<Thumb> {
        let meta = fs::metadata(raw).with_context(|| format!("stat {}", raw.display()))?;
        let sidecar = sidecar_path(raw);
        let sidecar_meta = fs::metadata(&sidecar).ok();
        let entry = self.entry_path(raw, &meta, sidecar_meta.as_ref());
        if let Some(thumb) = self.load_entry(&entry) {
            return Ok(thumb);
        }
        let thumb = if sidecar_meta.is_some() {
            developed(raw, &sidecar, gpu)?
        } else if source::is_raw(raw) {
            camera_preview(raw)?
        } else {
            image_preview(raw)?
        };
        if let Err(e) = self.store_entry(&entry, &thumb) {
            tracing::warn!(entry = %entry.display(), error = %format!("{e:#}"), "thumb store failed");
        }
        Ok(thumb)
    }

    fn entry_path(
        &self,
        raw: &Path,
        meta: &fs::Metadata,
        sidecar: Option<&fs::Metadata>,
    ) -> PathBuf {
        let mut h = Fnv1a::new();
        h.write(&FORMAT_VERSION.to_le_bytes());
        h.write(&THUMB_EDGE.to_le_bytes());
        h.write(raw.as_os_str().as_encoded_bytes());
        h.write_meta(meta);
        match sidecar {
            Some(m) => h.write_meta(m),
            None => h.write(&[0]),
        }
        let hex = format!("{:016x}", h.finish());
        self.root.join(&hex[..2]).join(format!("{hex}.thumb"))
    }

    /// Read and validate one entry. A mismatch deletes the file and is a
    /// miss. A hit refreshes the entry's mtime so the sweep's LRU sees
    /// recent use (atime is unreliable under `relatime`).
    fn load_entry(&self, entry: &Path) -> Option<Thumb> {
        let data = fs::read(entry).ok()?;
        match parse_entry(&data) {
            Some(thumb) => {
                if let Ok(f) = fs::File::open(entry) {
                    let _ = f.set_modified(SystemTime::now());
                }
                Some(thumb)
            }
            None => {
                tracing::warn!(entry = %entry.display(), "corrupt thumb entry, removing");
                let _ = fs::remove_file(entry);
                None
            }
        }
    }

    fn store_entry(&self, entry: &Path, thumb: &Thumb) -> Result<()> {
        let data = encode_entry(thumb);
        let dir = entry.parent().expect("entry paths have a parent");
        fs::create_dir_all(dir)?;
        let tmp = entry.with_extension(format!("tmp.{}", std::process::id()));
        fs::write(&tmp, &data)?;
        fs::rename(&tmp, entry)?;
        Ok(())
    }

    /// Delete least recently used entries until the cache fits
    /// `budget_bytes`; also reaps temp files orphaned by a crash.
    pub fn sweep(&self, budget_bytes: u64) -> SweepStats {
        let mut stats = SweepStats::default();
        let mut entries: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
        let Ok(fanout) = fs::read_dir(&self.root) else {
            return stats;
        };
        for dir in fanout.flatten() {
            let Ok(files) = fs::read_dir(dir.path()) else {
                continue;
            };
            for f in files.flatten() {
                let path = f.path();
                let Ok(meta) = f.metadata() else { continue };
                if !meta.is_file() {
                    continue;
                }
                if path.extension().is_some_and(|e| e != "thumb") {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                stats.files += 1;
                stats.bytes += meta.len();
                entries.push((path, meta.len(), mtime));
            }
        }
        if stats.bytes <= budget_bytes {
            return stats;
        }
        entries.sort_by_key(|&(_, _, mtime)| mtime);
        let mut remaining = stats.bytes;
        for (path, len, _) in entries {
            if remaining <= budget_bytes {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                remaining -= len;
                stats.removed_files += 1;
                stats.removed_bytes += len;
            }
        }
        tracing::info!(
            files = stats.files,
            mib = stats.bytes >> 20,
            removed = stats.removed_files,
            "thumb cache sweep"
        );
        stats
    }
}

/// The camera's embedded JPEG, box-downscaled in linear light and
/// rotated into viewing orientation.
fn camera_preview(raw: &Path) -> Result<Thumb> {
    let p = sensor::embedded_preview(raw)?;
    let (w, h, rgb) = downscale_rgb8(p.width, p.height, &p.rgb, THUMB_EDGE);
    let (w, h, rgb) = output::orient(w, h, 3, &rgb, p.orientation);
    Ok(Thumb {
        image: Image::from_rgb8(w, h, rgb),
        developed: false,
        exif: p.exif,
    })
}

/// A rendered image decoded into the working space, downscaled in
/// linear light, SDR-clipped.
fn image_preview(path: &Path) -> Result<Thumb> {
    let Source::Image(i) = Source::load(path)? else {
        anyhow::bail!("{}: expected a rendered image", path.display());
    };
    let img = output::downscaled(
        &WorkingImage {
            width: i.width,
            height: i.height,
            rgba: i.rgba,
        },
        THUMB_EDGE,
    );
    let img = output::oriented(&img, i.orientation);
    Ok(Thumb {
        image: Image::from_rgb8(img.width, img.height, output::srgb8(&img)),
        developed: false,
        exif: i.exif,
    })
}

/// Develop with the sidecar's parameters at the integer scale that
/// lands the long edge at or under the thumbnail edge.
fn developed(raw: &Path, sidecar: &Path, gpu: &ThumbGpu) -> Result<Thumb> {
    let params = DevelopParams::load(sidecar)?;
    let source = Source::load(raw)?;
    let (w, h) = source.size();
    let scale = w.max(h).div_ceil(THUMB_EDGE).max(1);
    let mut guard = gpu.0.lock().unwrap();
    if guard.is_none() {
        let g = Gpu::headless().context("thumbnail device")?;
        let d = Developer::new(&g);
        *guard = Some((g, d));
    }
    let (g, dev) = guard.as_ref().expect("initialised above");
    let prepared = dev.prepare(g, &source, &params)?;
    let img = dev.develop_prepared(g, &prepared, &params, OutputTarget::SDR, scale)?;
    drop(guard);
    Ok(Thumb {
        image: Image::from_rgb8(img.width, img.height, output::srgb8(&img)),
        developed: true,
        exif: source.exif().clone(),
    })
}

/// Integer box downscale of sRGB-encoded RGB8 so the long edge is at
/// most `edge`, averaging in linear light.
fn downscale_rgb8(width: u32, height: u32, rgb: &[u8], edge: u32) -> (u32, u32, Vec<u8>) {
    let f = width.max(height).div_ceil(edge).max(1) as usize;
    if f == 1 {
        return (width, height, rgb.to_vec());
    }
    let (w, h) = (width as usize, height as usize);
    let (ow, oh) = (w / f, h / f);
    let lut: Vec<f32> = (0..256).map(|v| srgb_eotf(v as f32 / 255.0)).collect();
    let mut out = Vec::with_capacity(ow * oh * 3);
    let norm = 1.0 / (f * f) as f32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0.0f32; 3];
            for y in oy * f..(oy + 1) * f {
                let row = &rgb[(y * w + ox * f) * 3..(y * w + (ox + 1) * f) * 3];
                for px in row.chunks_exact(3) {
                    acc[0] += lut[px[0] as usize];
                    acc[1] += lut[px[1] as usize];
                    acc[2] += lut[px[2] as usize];
                }
            }
            for c in acc {
                out.push((srgb_oetf(c * norm) * 255.0 + 0.5) as u8);
            }
        }
    }
    (ow as u32, oh as u32, out)
}

fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Entry layout, little-endian:
///
/// ```text
/// magic "IDTH" · version u32 · width u32 · height u32 · developed u8
/// · exif_len u32 · exif: TOML of `ExifSummary`
/// payload: width × height × 4 bytes, sRGB RGBA8
/// ```
fn encode_entry(thumb: &Thumb) -> Vec<u8> {
    let img = &thumb.image;
    let pixels = img.pixels();
    let exif = thumb.exif.to_toml();
    let mut out = Vec::with_capacity(HEADER_LEN + exif.len() + pixels.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&img.width().to_le_bytes());
    out.extend_from_slice(&img.height().to_le_bytes());
    out.push(thumb.developed as u8);
    out.extend_from_slice(&(exif.len() as u32).to_le_bytes());
    out.extend_from_slice(exif.as_bytes());
    out.extend_from_slice(pixels);
    out
}

fn parse_entry(data: &[u8]) -> Option<Thumb> {
    let header = data.get(..HEADER_LEN)?;
    if header[..4] != MAGIC {
        return None;
    }
    let u32_at = |i: usize| u32::from_le_bytes(header[i..i + 4].try_into().unwrap());
    if u32_at(4) != FORMAT_VERSION {
        return None;
    }
    let (width, height) = (u32_at(8), u32_at(12));
    if !(1..=MAX_DIM).contains(&width) || !(1..=MAX_DIM).contains(&height) {
        return None;
    }
    let developed = match header[16] {
        0 => false,
        1 => true,
        _ => return None,
    };
    let exif_len = u32_at(17) as usize;
    let exif_text = data.get(HEADER_LEN..HEADER_LEN + exif_len)?;
    let exif = ExifSummary::from_toml(std::str::from_utf8(exif_text).ok()?)?;
    let payload = &data[HEADER_LEN + exif_len..];
    if payload.len() != (width as usize) * (height as usize) * 4 {
        return None;
    }
    Some(Thumb {
        image: Image::from_rgba8(width, height, payload.to_vec()),
        developed,
        exif,
    })
}

/// FNV-1a 64: stable across runs and platforms, unlike `DefaultHasher`.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Fnv1a(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn write_meta(&mut self, meta: &fs::Metadata) {
        if let Some(d) = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(SystemTime::UNIX_EPOCH).ok())
        {
            self.write(&d.as_nanos().to_le_bytes());
        }
        self.write(&meta.len().to_le_bytes());
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_round_trip() {
        let thumb = Thumb {
            image: Image::from_rgba8(3, 2, (0..24).collect()),
            developed: true,
            exif: ExifSummary {
                make: "SONY".into(),
                iso: Some(3200),
                ..Default::default()
            },
        };
        let back = parse_entry(&encode_entry(&thumb)).unwrap();
        assert!(back.developed);
        assert_eq!(back.exif, thumb.exif);
        assert_eq!(back.image.width(), 3);
        assert_eq!(back.image.pixels(), thumb.image.pixels());
        assert!(parse_entry(&encode_entry(&thumb)[..20]).is_none());
    }

    #[test]
    fn downscale_averages_in_linear_light() {
        // 2x2 of black and white averages to linear 0.5 -> sRGB ~188.
        let rgb = [0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0];
        let (w, h, out) = downscale_rgb8(2, 2, &rgb, 1);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, [188, 188, 188]);
    }
}
