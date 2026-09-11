//! Human-readable shooting metadata: the standard EXIF directories
//! (IFD0, Exif, GPS) plus the Sony makernote, read from a raw's TIFF
//! structure or from the EXIF block embedded in a rendered image.
//!
//! Everything is rendered to strings at load time, grouped for display.
//! Known tags get a name and a domain formatter; the rest of IFD0, the
//! Exif IFD and the makernote land in an "Other" group as small numeric
//! dumps so nothing on disk is silently invisible. Bulky and structural
//! tags (offsets, previews, tone tables, enciphered Sony blocks) are
//! skipped.
//!
//! Sony decode tables follow ExifTool's Sony.pm; the sample values in the
//! tests were cross-checked against exiv2.

use std::io::{Cursor, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use rawler::Orientation;
use rawler::formats::tiff::ifd::OffsetMode;
use rawler::formats::tiff::reader::TiffReader;
use rawler::formats::tiff::{GenericTiffReader, IFD, Value};
use rawler::tags::{ExifGpsTag, ExifTag};

use crate::sensor::ExifSummary;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Group {
    pub name: &'static str,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metadata {
    pub groups: Vec<Group>,
}

impl Metadata {
    /// Metadata from a parsed TIFF structure (a raw file, or the EXIF
    /// block of a rendered image). `reader` is the same byte source the
    /// TIFF was parsed from; the makernote is read through it.
    pub fn from_tiff<R: Read + Seek>(tiff: &GenericTiffReader, reader: &mut R) -> Metadata {
        let root = tiff.root_ifd();
        let exif = root.get_sub_ifd(ExifTag::ExifOffset);
        let gps = root.get_sub_ifd(ExifTag::GPSInfo);
        let makernote = exif.and_then(|e| {
            e.parse_makernote(reader, OffsetMode::Absolute, &[])
                .ok()
                .flatten()
        });

        let mut out = Metadata::default();
        let mut other = Vec::new();
        let dirs: Vec<&IFD> = [Some(root), exif].into_iter().flatten().collect();
        for group in STANDARD_GROUPS {
            let mut fields = Vec::new();
            for spec in group.specs {
                let Some(entry) = dirs.iter().find_map(|d| d.get_entry(spec.tag)) else {
                    continue;
                };
                if let Some(value) = spec.fmt.render(&entry.value) {
                    fields.push(Field {
                        name: spec.name.into(),
                        value,
                    });
                }
            }
            out.push(group.name, fields);
        }
        for dir in &dirs {
            for (tag, entry) in dir.entries() {
                if STANDARD_GROUPS
                    .iter()
                    .any(|g| g.specs.iter().any(|s| s.tag == *tag))
                    || SKIP_STANDARD.contains(tag)
                {
                    continue;
                }
                let name = match ExifTag::n(*tag) {
                    Some(t) => format!("{t:?}"),
                    None => format!("0x{tag:04x}"),
                };
                if let Some(value) = Fmt::Raw.render(&entry.value) {
                    other.push(Field { name, value });
                }
            }
        }
        if let Some(gps) = gps {
            out.push("GPS", gps_fields(gps));
        }
        if let Some(mn) = &makernote {
            let mut fields = Vec::new();
            for spec in SONY_SPECS {
                if let Some(entry) = mn.get_entry(spec.tag)
                    && let Some(value) = spec.fmt.render(&entry.value)
                {
                    fields.push(Field {
                        name: spec.name.into(),
                        value,
                    });
                }
            }
            out.push("Sony", fields);
            for (tag, entry) in mn.entries() {
                if SONY_SPECS.iter().any(|s| s.tag == *tag)
                    || entry.value.count() > 8
                    || SKIP_SONY.contains(tag)
                    || (0x9000..=0x9fff).contains(tag)
                {
                    continue;
                }
                if let Some(value) = Fmt::Raw.render(&entry.value) {
                    other.push(Field {
                        name: format!("Sony 0x{tag:04x}"),
                        value,
                    });
                }
            }
        }
        out.push("Other", other);
        out
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Value of one field, for lookups and tests.
    pub fn get(&self, group: &str, name: &str) -> Option<&str> {
        self.groups
            .iter()
            .find(|g| g.name == group)?
            .fields
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.value.as_str())
    }

    fn push(&mut self, name: &'static str, fields: Vec<Field>) {
        if !fields.is_empty() {
            self.groups.push(Group { name, fields });
        }
    }
}

/// What a rendered image's EXIF block yields.
pub struct Embedded {
    pub summary: ExifSummary,
    pub orientation: Orientation,
    pub metadata: Metadata,
}

/// Read the EXIF block of a rendered image file (JPEG APP1, PNG `eXIf`,
/// WebP `EXIF`). `None` when the file carries none.
pub fn from_image_file(path: &Path) -> Result<Option<Embedded>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let Some(block) = exif_block(&bytes) else {
        return Ok(None);
    };
    let mut cursor = Cursor::new(block);
    let tiff = GenericTiffReader::new(&mut cursor, 0, 0, None, &[ExifTag::GPSInfo as u16])
        .with_context(|| format!("{}: EXIF block", path.display()))?;
    let root = tiff.root_ifd();
    let exif =
        rawler::exif::Exif::new(root).with_context(|| format!("{}: EXIF tags", path.display()))?;
    let ascii = |tag: ExifTag| {
        root.get_entry(tag)
            .and_then(|e| text(&e.value))
            .unwrap_or_default()
    };
    let summary = ExifSummary::from_exif(&exif, ascii(ExifTag::Make), ascii(ExifTag::Model));
    let orientation = match exif.orientation {
        Some(o @ 1..=8) => Orientation::from_u16(o),
        _ => Orientation::Normal,
    };
    Ok(Some(Embedded {
        summary,
        orientation,
        metadata: Metadata::from_tiff(&tiff, &mut cursor),
    }))
}

/// The TIFF-structured EXIF payload inside a JPEG, PNG or WebP file.
pub fn exif_block(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.starts_with(&[0xff, 0xd8]) {
        return jpeg_app1_exif(bytes);
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return png_chunk(bytes, b"eXIf");
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return riff_chunk(&bytes[12..], b"EXIF");
    }
    None
}

fn jpeg_app1_exif(bytes: &[u8]) -> Option<&[u8]> {
    let mut pos = 2;
    while pos + 4 <= bytes.len() {
        if bytes[pos] != 0xff {
            return None;
        }
        let marker = bytes[pos + 1];
        if marker == 0xd8 || (0xd0..=0xd7).contains(&marker) || marker == 0xff {
            pos += if marker == 0xff { 1 } else { 2 };
            continue;
        }
        if marker == 0xda || marker == 0xd9 {
            return None;
        }
        let len = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]) as usize;
        let body = bytes.get(pos + 4..pos + 2 + len)?;
        if marker == 0xe1 && body.starts_with(b"Exif\0\0") {
            return Some(&body[6..]);
        }
        pos += 2 + len;
    }
    None
}

fn png_chunk<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 8;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let name = &bytes[pos + 4..pos + 8];
        if name == kind {
            return bytes.get(pos + 8..pos + 8 + len);
        }
        if name == b"IEND" {
            return None;
        }
        pos += 12 + len;
    }
    None
}

fn riff_chunk<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 8 <= bytes.len() {
        let name = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        if name == kind {
            return bytes.get(pos + 8..pos + 8 + len);
        }
        pos += 8 + len + (len & 1);
    }
    None
}

// ---------------------------------------------------------------------
// Value access

fn ints(v: &Value) -> Vec<i64> {
    match v {
        Value::Byte(d) | Value::Undefined(d) => d.iter().map(|&x| i64::from(x)).collect(),
        Value::SByte(d) => d.iter().map(|&x| i64::from(x)).collect(),
        Value::Short(d) => d.iter().map(|&x| i64::from(x)).collect(),
        Value::SShort(d) => d.iter().map(|&x| i64::from(x)).collect(),
        Value::Long(d) => d.iter().map(|&x| i64::from(x)).collect(),
        Value::SLong(d) => d.iter().map(|&x| i64::from(x)).collect(),
        _ => Vec::new(),
    }
}

fn first_int(v: &Value) -> Option<i64> {
    ints(v).first().copied()
}

/// Rationals as (numerator, denominator); integers as (n, 1).
fn ratios(v: &Value) -> Vec<(i64, i64)> {
    match v {
        Value::Rational(d) => d.iter().map(|r| (i64::from(r.n), i64::from(r.d))).collect(),
        Value::SRational(d) => d.iter().map(|r| (i64::from(r.n), i64::from(r.d))).collect(),
        Value::Float(d) => d
            .iter()
            .map(|&x| ((x * 1000.0).round() as i64, 1000))
            .collect(),
        Value::Double(d) => d
            .iter()
            .map(|&x| ((x * 1000.0).round() as i64, 1000))
            .collect(),
        _ => ints(v).into_iter().map(|n| (n, 1)).collect(),
    }
}

fn first_f64(v: &Value) -> Option<f64> {
    let (n, d) = ratios(v).first().copied()?;
    (d != 0).then(|| n as f64 / d as f64)
}

fn text(v: &Value) -> Option<String> {
    let s = match v {
        Value::Ascii(a) => a.strings().join(" "),
        Value::Undefined(d) | Value::Byte(d) => {
            let end = d.iter().position(|&b| b == 0).unwrap_or(d.len());
            let s = &d[..end];
            if !s.iter().all(|b| (0x20..0x7f).contains(b)) {
                return None;
            }
            String::from_utf8_lossy(s).into_owned()
        }
        _ => return None,
    };
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// A decimal with trailing zeros trimmed, at most `places` places.
fn decimal(x: f64, places: usize) -> String {
    let s = format!("{x:.places$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Two-digit BCD byte.
fn bcd(b: i64) -> i64 {
    (b >> 4) * 10 + (b & 0xf)
}

// ---------------------------------------------------------------------
// Formatters

type EnumTable = &'static [(i64, &'static str)];

#[derive(Clone, Copy)]
enum Fmt {
    /// ASCII (or printable undefined) as-is.
    Text,
    /// First integer.
    Int,
    /// All integers, space separated.
    Ints,
    /// First rational as a short decimal.
    Decimal,
    /// Exposure time: `1/250 s` or `2.5 s`.
    Exposure,
    /// `f/1.8`.
    Aperture,
    /// `50 mm`.
    Millimetres,
    /// Signed EV with sign: `+0.3 EV`.
    Ev,
    /// Four undefined ASCII digits `0232` as `2.32`.
    Version,
    /// Integer looked up in a table, else the number.
    Enum(EnumTable),
    /// APEX aperture value `Av` as `f/N`.
    ApexAperture,
    /// APEX shutter value `Tv` as an exposure time.
    ApexShutter,
    /// EXIF LensSpecification: four rationals.
    LensSpec,
    /// EXIF Flash bit field.
    Flash,
    /// Whatever the value is, compactly (unknown tags).
    Raw,
    // Sony makernote specifics.
    SonyLensSpec,
    SonyAutoHdr,
    SonyColorTemperature,
    SonyWbShift,
    SonyFrameSize,
    SonyFocusLocation,
    SonyFileFormat,
    SonyFlashLevel,
    SonySequence,
    SonyQuality2,
    SonyLowPass,
    SonyAfPoints,
    /// Bytes as little-endian u16 pairs: `W × H`.
    SonyU16Size,
    /// Six bytes, all zero when unused.
    SonyPixelShift,
    SonyLensId,
    SonyModelId,
}

impl Fmt {
    fn render(self, v: &Value) -> Option<String> {
        Some(match self {
            Fmt::Text => text(v)?,
            Fmt::Int => first_int(v)?.to_string(),
            Fmt::Ints => {
                let all = ints(v);
                if all.is_empty() {
                    return None;
                }
                all.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            Fmt::Decimal => decimal(first_f64(v)?, 3),
            Fmt::Exposure => {
                let (n, d) = ratios(v).first().copied()?;
                if n == 0 || d == 0 {
                    return None;
                }
                let t = n as f64 / d as f64;
                if t < 0.3 {
                    format!("1/{} s", decimal(1.0 / t, 1))
                } else {
                    format!("{} s", decimal(t, 2))
                }
            }
            Fmt::Aperture => format!("f/{}", decimal(first_f64(v)?, 1)),
            Fmt::Millimetres => format!("{} mm", decimal(first_f64(v)?, 1)),
            Fmt::Ev => format!("{:+} EV", decimal(first_f64(v)?, 2)),
            Fmt::Version => {
                let s = text(v)?;
                let s = s.trim_start_matches('0');
                if s.len() >= 2 {
                    format!("{}.{}", &s[..s.len() - 2], &s[s.len() - 2..])
                } else {
                    format!("0.{s:0>2}")
                }
            }
            Fmt::Enum(table) => {
                let x = first_int(v)?;
                match table.iter().find(|(k, _)| *k == x) {
                    Some((_, name)) => (*name).to_string(),
                    None => x.to_string(),
                }
            }
            Fmt::ApexAperture => format!("f/{}", decimal(2f64.powf(first_f64(v)? / 2.0), 1)),
            Fmt::ApexShutter => {
                let t = 2f64.powf(-first_f64(v)?);
                if t < 0.3 {
                    format!("1/{} s", decimal(1.0 / t, 0))
                } else {
                    format!("{} s", decimal(t, 2))
                }
            }
            Fmt::LensSpec => {
                let r: Vec<f64> = ratios(v)
                    .iter()
                    .map(|&(n, d)| if d == 0 { 0.0 } else { n as f64 / d as f64 })
                    .collect();
                if r.len() < 4 {
                    return None;
                }
                lens_spec(r[0], r[1], r[2], r[3])?
            }
            Fmt::Flash => {
                let x = first_int(v)?;
                let mut parts = vec![if x & 1 != 0 { "Fired" } else { "Did not fire" }];
                match (x >> 3) & 3 {
                    1 => parts.push("compulsory"),
                    2 => parts.push("suppressed"),
                    3 => parts.push("auto"),
                    _ => {}
                }
                match (x >> 1) & 3 {
                    2 => parts.push("return not detected"),
                    3 => parts.push("return detected"),
                    _ => {}
                }
                if x & 0x20 != 0 {
                    parts.push("no flash function");
                }
                if x & 0x40 != 0 {
                    parts.push("red-eye reduction");
                }
                parts.join(", ")
            }
            Fmt::Raw => {
                if let Some(s) = text(v) {
                    return Some(s);
                }
                let r = ratios(v);
                if r.is_empty() {
                    return None;
                }
                let shown: Vec<String> = r
                    .iter()
                    .take(8)
                    .map(|&(n, d)| {
                        if d == 1 {
                            n.to_string()
                        } else {
                            format!("{n}/{d}")
                        }
                    })
                    .collect();
                let mut s = shown.join(" ");
                if r.len() > 8 {
                    s.push_str(&format!(" … ({} values)", r.len()));
                }
                s
            }
            Fmt::SonyLensSpec => {
                let b = ints(v);
                if b.len() < 8 {
                    return None;
                }
                let f = |hi: i64, lo: i64| (bcd(hi) * 100 + bcd(lo)) as f64;
                lens_spec(
                    f(b[1], b[2]),
                    f(b[3], b[4]),
                    bcd(b[5]) as f64 / 10.0,
                    bcd(b[6]) as f64 / 10.0,
                )?
            }
            Fmt::SonyAutoHdr => {
                let x = first_int(v)?;
                let (hi, lo) = (x >> 16, x & 0xffff);
                if hi == 0 {
                    "Off".into()
                } else if lo == 1 {
                    "Auto".into()
                } else {
                    format!("{}.{} EV", lo >> 4, if lo & 0xf == 6 { 5 } else { 0 })
                }
            }
            Fmt::SonyColorTemperature => match first_int(v)? {
                0 => "Auto".into(),
                0xffff_ffff => "n/a".into(),
                k => format!("{k} K"),
            },
            Fmt::SonyWbShift => {
                let all = ints(v);
                format!("A/B: {}, G/M: {}", all.first()?, all.get(1)?)
            }
            Fmt::SonyFrameSize => {
                // Height, width.
                let all = ints(v);
                format!("{} × {}", all.get(1)?, all.first()?)
            }
            Fmt::SonyFocusLocation => {
                let all = ints(v);
                if all.len() < 4 {
                    return None;
                }
                format!("{}, {} of {} × {}", all[2], all[3], all[0], all[1])
            }
            Fmt::SonyFileFormat => {
                let b = ints(v);
                if b.len() < 4 {
                    return None;
                }
                match (b[0], b[1], b[2], b[3]) {
                    (0, 0, 0, 2) => "JPEG".into(),
                    (1, 0, 0, 0) => "SR2".into(),
                    (2, 0, 0, 0) => "ARW 1.0".into(),
                    (3, m, p, 0) => match p {
                        0 => format!("ARW 2.{m}"),
                        p => format!("ARW 2.{m}.{p}"),
                    },
                    (major, m, p, _) => match p {
                        0 => format!("ARW {major}.{m}"),
                        p => format!("ARW {major}.{m}.{p}"),
                    },
                }
            }
            Fmt::SonyFlashLevel => match first_int(v)? {
                -32768 => "Low".into(),
                0 => "Normal".into(),
                128 => "n/a".into(),
                32767 => "High".into(),
                x => format!("{x:+}/3"),
            },
            Fmt::SonySequence => match first_int(v)? {
                0 => "Single".into(),
                65535 => "n/a".into(),
                x => x.to_string(),
            },
            Fmt::SonyQuality2 => {
                let all = ints(v);
                match (all.first()?, all.get(1)?) {
                    (0, 0) => "n/a".into(),
                    (0, _) => "JPEG / HEIF".into(),
                    (_, 0) => "RAW".into(),
                    _ => "RAW + JPEG / HEIF".into(),
                }
            }
            Fmt::SonyLowPass => {
                let all = ints(v);
                match (all.first()?, all.get(1)?) {
                    (65535, _) => "n/a".into(),
                    (_, 0) => "Off".into(),
                    (_, 1) => "Standard".into(),
                    (_, 2) => "High".into(),
                    (a, b) => format!("{a} {b}"),
                }
            }
            Fmt::SonyAfPoints => {
                let bits = ints(v);
                let set: Vec<String> = bits
                    .iter()
                    .enumerate()
                    .flat_map(|(i, b)| {
                        (0..8)
                            .filter(move |k| b & (1 << k) != 0)
                            .map(move |k| (i * 8 + k).to_string())
                    })
                    .collect();
                if set.is_empty() {
                    "None".into()
                } else {
                    set.join(" ")
                }
            }
            Fmt::SonyU16Size => {
                let b = ints(v);
                if b.len() < 4 {
                    return None;
                }
                format!("{} × {}", b[0] | (b[1] << 8), b[2] | (b[3] << 8))
            }
            Fmt::SonyPixelShift => {
                let b = ints(v);
                if b.iter().all(|&x| x == 0) {
                    "None".into()
                } else {
                    b.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            }
            Fmt::SonyLensId => match first_int(v)? {
                65535 => "E-mount or other".into(),
                x => x.to_string(),
            },
            Fmt::SonyModelId => match first_int(v)? {
                394 => "ILCE-6700".into(),
                x => x.to_string(),
            },
        })
    }
}

/// `50mm f/1.8`, `16-50mm f/3.5-5.6`.
fn lens_spec(f_min: f64, f_max: f64, a_min: f64, a_max: f64) -> Option<String> {
    if f_min <= 0.0 {
        return None;
    }
    let focal = if f_max > f_min {
        format!("{}-{}mm", decimal(f_min, 1), decimal(f_max, 1))
    } else {
        format!("{}mm", decimal(f_min, 1))
    };
    let aperture = if a_min <= 0.0 {
        String::new()
    } else if a_max > a_min {
        format!(" f/{}-{}", decimal(a_min, 1), decimal(a_max, 1))
    } else {
        format!(" f/{}", decimal(a_min, 1))
    };
    Some(format!("{focal}{aperture}"))
}

// ---------------------------------------------------------------------
// Standard EXIF

struct Spec {
    tag: u16,
    name: &'static str,
    fmt: Fmt,
}

const fn spec(tag: u16, name: &'static str, fmt: Fmt) -> Spec {
    Spec { tag, name, fmt }
}

struct GroupSpec {
    name: &'static str,
    specs: &'static [Spec],
}

const EXPOSURE_PROGRAM: EnumTable = &[
    (0, "Not defined"),
    (1, "Manual"),
    (2, "Program AE"),
    (3, "Aperture priority"),
    (4, "Shutter priority"),
    (5, "Creative"),
    (6, "Action"),
    (7, "Portrait"),
    (8, "Landscape"),
];
const METERING_MODE: EnumTable = &[
    (0, "Unknown"),
    (1, "Average"),
    (2, "Center-weighted average"),
    (3, "Spot"),
    (4, "Multi-spot"),
    (5, "Multi-segment"),
    (6, "Partial"),
    (255, "Other"),
];
const LIGHT_SOURCE: EnumTable = &[
    (0, "Unknown"),
    (1, "Daylight"),
    (2, "Fluorescent"),
    (3, "Tungsten"),
    (4, "Flash"),
    (9, "Fine weather"),
    (10, "Cloudy"),
    (11, "Shade"),
    (12, "Daylight fluorescent"),
    (13, "Day white fluorescent"),
    (14, "Cool white fluorescent"),
    (15, "White fluorescent"),
    (17, "Standard light A"),
    (18, "Standard light B"),
    (19, "Standard light C"),
    (20, "D55"),
    (21, "D65"),
    (22, "D75"),
    (23, "D50"),
    (24, "ISO studio tungsten"),
    (255, "Other"),
];
const EXPOSURE_MODE: EnumTable = &[(0, "Auto"), (1, "Manual"), (2, "Auto bracket")];
const WHITE_BALANCE: EnumTable = &[(0, "Auto"), (1, "Manual")];
const SCENE_CAPTURE: EnumTable = &[
    (0, "Standard"),
    (1, "Landscape"),
    (2, "Portrait"),
    (3, "Night scene"),
];
const LOW_NORMAL_HIGH: EnumTable = &[(0, "Normal"), (1, "Low"), (2, "High")];
const SOFT_NORMAL_HARD: EnumTable = &[(0, "Normal"), (1, "Soft"), (2, "Hard")];
const CUSTOM_RENDERED: EnumTable = &[(0, "Normal"), (1, "Custom")];
const COLOR_SPACE: EnumTable = &[(1, "sRGB"), (2, "Adobe RGB"), (65535, "Uncalibrated")];
const ORIENTATION: EnumTable = &[
    (1, "Normal"),
    (2, "Mirror horizontal"),
    (3, "Rotate 180"),
    (4, "Mirror vertical"),
    (5, "Mirror horizontal, rotate 270 CW"),
    (6, "Rotate 90 CW"),
    (7, "Mirror horizontal, rotate 90 CW"),
    (8, "Rotate 270 CW"),
];
const SENSITIVITY_TYPE: EnumTable = &[
    (0, "Unknown"),
    (1, "Standard output sensitivity"),
    (2, "Recommended exposure index"),
    (3, "ISO speed"),
    (4, "SOS and REI"),
    (5, "SOS and ISO speed"),
    (6, "REI and ISO speed"),
    (7, "SOS, REI and ISO speed"),
];
const RESOLUTION_UNIT: EnumTable = &[(1, "none"), (2, "inch"), (3, "cm")];
const FILE_SOURCE: EnumTable = &[
    (1, "Film scanner"),
    (2, "Reflection print scanner"),
    (3, "Digital still camera"),
];
const SCENE_TYPE: EnumTable = &[(1, "Directly photographed")];
const COMPOSITE_IMAGE: EnumTable = &[
    (0, "Unknown"),
    (1, "Not a composite"),
    (2, "General composite"),
    (3, "Composite captured while shooting"),
];

const STANDARD_GROUPS: &[GroupSpec] = &[
    GroupSpec {
        name: "Camera",
        specs: &[
            spec(0x010f, "Make", Fmt::Text),
            spec(0x0110, "Model", Fmt::Text),
            spec(0xa431, "Serial number", Fmt::Text),
            spec(0x0131, "Software", Fmt::Text),
            spec(0xa430, "Owner", Fmt::Text),
            spec(0x013b, "Artist", Fmt::Text),
            spec(0x8298, "Copyright", Fmt::Text),
            spec(0x010e, "Description", Fmt::Text),
            spec(0x9286, "User comment", Fmt::Text),
        ],
    },
    GroupSpec {
        name: "Exposure",
        specs: &[
            spec(0x829a, "Exposure time", Fmt::Exposure),
            spec(0x829d, "Aperture", Fmt::Aperture),
            spec(0x8827, "ISO", Fmt::Int),
            spec(0x8830, "Sensitivity type", Fmt::Enum(SENSITIVITY_TYPE)),
            spec(0x8832, "Recommended exposure index", Fmt::Int),
            spec(0x8833, "ISO speed", Fmt::Int),
            spec(0x8822, "Program", Fmt::Enum(EXPOSURE_PROGRAM)),
            spec(0xa402, "Exposure mode", Fmt::Enum(EXPOSURE_MODE)),
            spec(0x9204, "Exposure bias", Fmt::Ev),
            spec(0x9207, "Metering", Fmt::Enum(METERING_MODE)),
            spec(0x9203, "Brightness value", Fmt::Decimal),
            spec(0x9201, "Shutter speed value", Fmt::ApexShutter),
            spec(0x9202, "Aperture value", Fmt::ApexAperture),
            spec(0x9205, "Max aperture", Fmt::ApexAperture),
            spec(0x9209, "Flash", Fmt::Flash),
            spec(0x9208, "Light source", Fmt::Enum(LIGHT_SOURCE)),
            spec(0xa403, "White balance", Fmt::Enum(WHITE_BALANCE)),
            spec(0xa406, "Scene capture", Fmt::Enum(SCENE_CAPTURE)),
            spec(0xa408, "Contrast", Fmt::Enum(SOFT_NORMAL_HARD)),
            spec(0xa409, "Saturation", Fmt::Enum(LOW_NORMAL_HIGH)),
            spec(0xa40a, "Sharpness", Fmt::Enum(SOFT_NORMAL_HARD)),
            spec(0xa404, "Digital zoom", Fmt::Decimal),
            spec(0xa401, "Rendering", Fmt::Enum(CUSTOM_RENDERED)),
            spec(0x9206, "Subject distance", Fmt::Decimal),
        ],
    },
    GroupSpec {
        name: "Lens",
        specs: &[
            spec(0xa434, "Lens", Fmt::Text),
            spec(0xa433, "Lens make", Fmt::Text),
            spec(0xa432, "Lens spec", Fmt::LensSpec),
            spec(0xa435, "Lens serial", Fmt::Text),
            spec(0x920a, "Focal length", Fmt::Millimetres),
            spec(0xa405, "Focal length (35 mm)", Fmt::Millimetres),
        ],
    },
    GroupSpec {
        name: "Time",
        specs: &[
            spec(0x9003, "Taken", Fmt::Text),
            spec(0x9291, "Taken (subsec)", Fmt::Text),
            spec(0x9011, "Taken (offset)", Fmt::Text),
            spec(0x9004, "Digitized", Fmt::Text),
            spec(0x0132, "Modified", Fmt::Text),
            spec(0x9010, "Modified (offset)", Fmt::Text),
        ],
    },
    GroupSpec {
        name: "Image",
        specs: &[
            spec(0x0112, "Orientation", Fmt::Enum(ORIENTATION)),
            spec(0xa002, "Width", Fmt::Int),
            spec(0xa003, "Height", Fmt::Int),
            spec(0xa001, "Colour space", Fmt::Enum(COLOR_SPACE)),
            spec(0x011a, "X resolution", Fmt::Decimal),
            spec(0x011b, "Y resolution", Fmt::Decimal),
            spec(0x0128, "Resolution unit", Fmt::Enum(RESOLUTION_UNIT)),
            spec(0x9000, "EXIF version", Fmt::Version),
            spec(0xa000, "Flashpix version", Fmt::Version),
            spec(0xa300, "File source", Fmt::Enum(FILE_SOURCE)),
            spec(0xa301, "Scene type", Fmt::Enum(SCENE_TYPE)),
            spec(0xa460, "Composite", Fmt::Enum(COMPOSITE_IMAGE)),
            spec(0x9211, "Image number", Fmt::Int),
        ],
    },
];

/// Structural, bulky or duplicated tags left out of "Other".
const SKIP_STANDARD: &[u16] = &[
    0x00fe, 0x0100, 0x0101, 0x0102, 0x0103, 0x0106, 0x0111, 0x0115, 0x0116, 0x0117, 0x011c, 0x014a,
    0x0201, 0x0202, 0x0213, 0x02bc, 0x4746, 0x4749, 0x8769, 0x8825, 0x9101, 0x9102, 0x927c, 0x9290,
    0x9292, 0x9012, 0xa005, 0xc4a5, 0xc634, 0xc6f2, 0x828d, 0x828e, 0x7010, 0x7031, 0x7032, 0x7034,
    0x7035, 0x7036, 0x7037, 0x7300, 0x7302, 0x7303, 0x7310, 0x7313, 0x7314, 0x7316, 0x7317, 0x7321,
    0x74c7, 0x74c8, 0xea1c,
];

// ---------------------------------------------------------------------
// GPS

fn gps_fields(gps: &IFD) -> Vec<Field> {
    let get = |tag: ExifGpsTag| gps.get_entry(tag).map(|e| &e.value);
    let mut out = Vec::new();
    let mut push = |name: &str, value: Option<String>| {
        if let Some(value) = value {
            out.push(Field {
                name: name.into(),
                value,
            });
        }
    };
    let coord = |value: Option<&Value>, r: Option<&Value>, pos: char| -> Option<String> {
        let r = r.and_then(text)?;
        let parts = ratios(value?);
        if parts.len() < 3 {
            return None;
        }
        let dms: Vec<f64> = parts
            .iter()
            .map(|&(n, d)| if d == 0 { 0.0 } else { n as f64 / d as f64 })
            .collect();
        let deg = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0;
        let sign = if r.starts_with(pos) { 1.0 } else { -1.0 };
        Some(format!("{:.6}", sign * deg))
    };
    push(
        "Latitude",
        coord(
            get(ExifGpsTag::GPSLatitude),
            get(ExifGpsTag::GPSLatitudeRef),
            'N',
        ),
    );
    push(
        "Longitude",
        coord(
            get(ExifGpsTag::GPSLongitude),
            get(ExifGpsTag::GPSLongitudeRef),
            'E',
        ),
    );
    push(
        "Altitude",
        get(ExifGpsTag::GPSAltitude).and_then(first_f64).map(|a| {
            let below = get(ExifGpsTag::GPSAltitudeRef).and_then(first_int) == Some(1);
            format!(
                "{} m{}",
                decimal(a, 1),
                if below { " below sea level" } else { "" }
            )
        }),
    );
    push(
        "Time (UTC)",
        get(ExifGpsTag::GPSTimeStamp).map(ratios).and_then(|t| {
            let f = |i: usize| {
                t.get(i)
                    .map(|&(n, d)| if d == 0 { 0.0 } else { n as f64 / d as f64 })
            };
            Some(format!(
                "{:02}:{:02}:{:02}",
                f(0)? as u32,
                f(1)? as u32,
                f(2)? as u32
            ))
        }),
    );
    push("Date (UTC)", get(ExifGpsTag::GPSDateStamp).and_then(text));
    push(
        "Status",
        get(ExifGpsTag::GPSStatus)
            .and_then(text)
            .map(|s| match s.as_str() {
                "A" => "Measurement in progress".into(),
                "V" => "Measurement interrupted".into(),
                _ => s,
            }),
    );
    push(
        "Measure mode",
        get(ExifGpsTag::GPSMeasureMode)
            .and_then(text)
            .map(|s| match s.as_str() {
                "2" => "2-dimensional".into(),
                "3" => "3-dimensional".into(),
                _ => s,
            }),
    );
    push("Satellites", get(ExifGpsTag::GPSSatellites).and_then(text));
    push(
        "DOP",
        get(ExifGpsTag::GPSDOP)
            .and_then(first_f64)
            .map(|x| decimal(x, 2)),
    );
    push(
        "Speed",
        get(ExifGpsTag::GPSSpeed).and_then(first_f64).map(|s| {
            let unit = match get(ExifGpsTag::GPSSpeedRef).and_then(text).as_deref() {
                Some("K") => "km/h",
                Some("M") => "mph",
                Some("N") => "knots",
                _ => "",
            };
            format!("{} {unit}", decimal(s, 1))
        }),
    );
    let direction = |value: ExifGpsTag, reference: ExifGpsTag| -> Option<String> {
        let d = get(value).and_then(first_f64)?;
        let kind = match get(reference).and_then(text).as_deref() {
            Some("T") => " true",
            Some("M") => " magnetic",
            _ => "",
        };
        Some(format!("{}°{kind}", decimal(d, 1)))
    };
    push(
        "Track",
        direction(ExifGpsTag::GPSTrack, ExifGpsTag::GPSTrackRef),
    );
    push(
        "Image direction",
        direction(ExifGpsTag::GPSImgDirection, ExifGpsTag::GPSImgDirectionRef),
    );
    push("Datum", get(ExifGpsTag::GPSMapDatum).and_then(text));
    push(
        "Differential",
        get(ExifGpsTag::GPSDifferential)
            .and_then(first_int)
            .map(|d| {
                match d {
                    0 => "Without correction",
                    1 => "With correction",
                    _ => "Unknown",
                }
                .into()
            }),
    );
    push(
        "Positioning error",
        get(ExifGpsTag::GPSHPositioningError)
            .and_then(first_f64)
            .map(|x| format!("{} m", decimal(x, 1))),
    );
    out
}

// ---------------------------------------------------------------------
// Sony makernote (tables after ExifTool's Sony.pm)

/// Binary blocks (Tag2010, ShotInfo, PrintIM, Minolta makernote) and,
/// by range above, the enciphered 0x94xx/0x9050 tags.
const SKIP_SONY: &[u16] = &[0x0e00, 0x2010, 0x3000, 0xb028];

const SONY_ON_OFF: EnumTable = &[(0, "Off"), (1, "On"), (0xffff_ffff, "n/a")];
const SONY_QUALITY: EnumTable = &[
    (0, "RAW"),
    (1, "Super fine"),
    (2, "Fine"),
    (3, "Standard"),
    (4, "Economy"),
    (5, "Extra fine"),
    (6, "RAW + JPEG / HEIF"),
    (7, "Compressed RAW"),
    (8, "Compressed RAW + JPEG"),
    (9, "Light"),
    (0xffff_ffff, "n/a"),
];
const SONY_WHITE_BALANCE: EnumTable = &[
    (0x00, "Auto"),
    (0x01, "Colour temperature / filter"),
    (0x10, "Daylight"),
    (0x20, "Cloudy"),
    (0x30, "Fluorescent"),
    (0x40, "Tungsten"),
    (0x50, "Flash"),
    (0x100, "Custom"),
    (0x200, "Underwater"),
];
const SONY_LONG_EXPOSURE_NR: EnumTable = &[
    (0, "Off"),
    (1, "On (unused)"),
    (0x10001, "On (dark subtracted)"),
    (0xffff_0000, "n/a"),
];
const SONY_HIGH_ISO_NR: EnumTable = &[
    (0, "Off"),
    (1, "Low"),
    (2, "Normal"),
    (3, "High"),
    (256, "Auto"),
    (65535, "n/a"),
];
const SONY_MULTI_FRAME_NR: EnumTable = &[(0, "Off"), (1, "On"), (255, "None"), (65535, "n/a")];
const SONY_PICTURE_EFFECT: EnumTable = &[
    (0, "Off"),
    (1, "Toy camera"),
    (2, "Pop colour"),
    (3, "Posterization"),
    (4, "Posterization B/W"),
    (5, "Retro photo"),
    (6, "Soft high key"),
    (7, "Partial colour (red)"),
    (8, "Partial colour (green)"),
    (9, "Partial colour (blue)"),
    (10, "Partial colour (yellow)"),
    (13, "High contrast monochrome"),
    (16, "Toy camera (normal)"),
    (17, "Toy camera (cool)"),
    (18, "Toy camera (warm)"),
    (19, "Toy camera (green)"),
    (20, "Toy camera (magenta)"),
    (32, "Soft focus (low)"),
    (33, "Soft focus (mid)"),
    (34, "Soft focus (high)"),
    (48, "Miniature (auto)"),
    (49, "Miniature (top)"),
    (50, "Miniature (middle horizontal)"),
    (51, "Miniature (bottom)"),
    (52, "Miniature (left)"),
    (53, "Miniature (middle vertical)"),
    (54, "Miniature (right)"),
    (64, "HDR painting (low)"),
    (65, "HDR painting (mid)"),
    (66, "HDR painting (high)"),
    (80, "Rich-tone monochrome"),
    (97, "Water colour"),
    (98, "Water colour 2"),
    (112, "Illustration (low)"),
    (113, "Illustration (mid)"),
    (114, "Illustration (high)"),
];
const SONY_SOFT_SKIN: EnumTable = &[(0, "Off"), (1, "Low"), (2, "Mid"), (3, "High")];
const SONY_CORRECTION: EnumTable = &[(0, "Off"), (2, "Auto"), (0xffff_ffff, "n/a")];
const SONY_YES_NO: EnumTable = &[(0, "No"), (1, "Yes")];
const SONY_FLASH_ACTION: EnumTable = &[
    (0, "Did not fire"),
    (1, "Fired"),
    (2, "External flash fired"),
    (3, "Wireless controlled flash fired"),
];
const SONY_FOCUS_MODE: EnumTable = &[
    (0, "Manual"),
    (2, "AF-S"),
    (3, "AF-C"),
    (4, "AF-A"),
    (6, "DMF"),
];
const SONY_AF_AREA_MODE: EnumTable = &[
    (0, "Wide"),
    (4, "Flexible spot"),
    (8, "Zone"),
    (9, "Center"),
    (11, "Flexible spot"),
    (12, "Expanded flexible spot"),
    (13, "Custom"),
];
const SONY_AF_POINT: EnumTable = &[(0, "n/a")];
const SONY_AF_TRACKING: EnumTable = &[(0, "Off"), (1, "Face tracking"), (2, "Lock-on AF")];
const SONY_NR_EFFECT: EnumTable = &[(0, "Normal"), (1, "High")];
const SONY_RAW_TYPE: EnumTable = &[
    (0, "Compressed"),
    (1, "Uncompressed"),
    (2, "Lossless compressed"),
];
const SONY_AWB_PRIORITY: EnumTable = &[(0, "Standard"), (1, "Ambience"), (2, "White")];
const SONY_METERING: EnumTable = &[
    (0x100, "Multi-segment"),
    (0x200, "Center-weighted average"),
    (0x301, "Spot (standard)"),
    (0x302, "Spot (large)"),
    (0x400, "Average"),
    (0x500, "Highlight"),
];
const SONY_JPEG_HEIF: EnumTable = &[(0, "JPEG"), (1, "HEIF")];
const SONY_SCENE_MODE: EnumTable = &[
    (0, "Standard"),
    (1, "Portrait"),
    (2, "Text"),
    (3, "Night scene"),
    (4, "Sunset"),
    (5, "Sports"),
    (6, "Landscape"),
    (7, "Night portrait"),
    (8, "Macro"),
    (9, "Super macro"),
    (16, "Auto"),
    (17, "Night view / portrait"),
    (18, "Sweep panorama"),
    (19, "Handheld night shot"),
    (20, "Anti motion blur"),
    (21, "Cont. priority AE"),
    (22, "Auto+"),
    (23, "3D sweep panorama"),
    (24, "Superior auto"),
    (25, "High sensitivity"),
    (26, "Fireworks"),
    (27, "Food"),
    (28, "Pet"),
    (33, "HDR"),
    (0xffff, "n/a"),
];
const SONY_ZONE_MATCHING: EnumTable = &[(0, "ISO setting used"), (1, "High key"), (2, "Low key")];
const SONY_DRO: EnumTable = &[
    (0, "Off"),
    (1, "Standard"),
    (2, "Advanced auto"),
    (3, "Auto"),
    (8, "Advanced Lv1"),
    (9, "Advanced Lv2"),
    (10, "Advanced Lv3"),
    (11, "Advanced Lv4"),
    (12, "Advanced Lv5"),
    (16, "Lv1"),
    (17, "Lv2"),
    (18, "Lv3"),
    (19, "Lv4"),
    (20, "Lv5"),
    (0xffff_ffff, "n/a"),
];
const SONY_COLOR_MODE: EnumTable = &[
    (0, "Standard"),
    (1, "Vivid"),
    (2, "Portrait"),
    (3, "Landscape"),
    (4, "Sunset"),
    (5, "Night view / portrait"),
    (6, "B&W"),
    (7, "Adobe RGB"),
    (12, "Neutral"),
    (13, "Clear"),
    (14, "Deep"),
    (15, "Light"),
    (16, "Autumn leaves"),
    (17, "Sepia"),
    (100, "Neutral"),
    (101, "Clear"),
    (102, "Deep"),
    (103, "Light"),
    (104, "Night view"),
    (105, "Autumn leaves"),
    (0xffff_ffff, "n/a"),
];
const SONY_EXPOSURE_MODE: EnumTable = &[
    (0, "Program AE"),
    (1, "Portrait"),
    (2, "Beach"),
    (3, "Sports"),
    (4, "Snow"),
    (5, "Landscape"),
    (6, "Auto"),
    (7, "Aperture priority AE"),
    (8, "Shutter priority AE"),
    (9, "Night scene / twilight"),
    (10, "Hi-speed shutter"),
    (11, "Twilight portrait"),
    (12, "Soft snap / portrait"),
    (13, "Fireworks"),
    (14, "Smile shutter"),
    (15, "Manual"),
    (18, "High sensitivity"),
    (19, "Macro"),
    (20, "Advanced sports shooting"),
    (29, "Underwater"),
    (33, "Food"),
    (34, "Gourmet"),
    (35, "Pet"),
    (37, "Handheld night shot"),
    (38, "Anti motion blur"),
    (39, "Sweep panorama"),
    (43, "Intelligent auto"),
    (0xffff, "n/a"),
];
const SONY_RELEASE_MODE: EnumTable = &[
    (0, "Normal"),
    (2, "Continuous"),
    (5, "Exposure bracketing"),
    (6, "White balance bracketing"),
    (8, "DRO bracketing"),
    (65535, "n/a"),
];
const SONY_ANTI_BLUR: EnumTable = &[
    (0, "Off"),
    (1, "On (continuous)"),
    (2, "On (shooting)"),
    (65535, "n/a"),
];
const SONY_DRO2: EnumTable = &[(0, "Off"), (1, "Standard"), (2, "Plus"), (65535, "n/a")];
const SONY_HIGH_ISO_NR2: EnumTable = &[
    (0, "Normal"),
    (1, "High"),
    (2, "Low"),
    (3, "Off"),
    (65535, "n/a"),
];
const SONY_INTELLIGENT_AUTO: EnumTable = &[(0, "Off"), (1, "On"), (2, "Advanced")];
const SONY_WHITE_BALANCE2: EnumTable = &[
    (0, "Auto"),
    (4, "Custom"),
    (5, "Daylight"),
    (6, "Cloudy"),
    (7, "Cool white fluorescent"),
    (8, "Day white fluorescent"),
    (9, "Daylight fluorescent"),
    (10, "Incandescent"),
    (14, "Warm white fluorescent"),
    (15, "Flash"),
    (17, "Underwater 1"),
    (18, "Underwater 2"),
];
const SONY_TELECONVERTER: EnumTable = &[(0, "None")];

const SONY_SPECS: &[Spec] = &[
    spec(0xb001, "Model ID", Fmt::SonyModelId),
    spec(0xb000, "File format", Fmt::SonyFileFormat),
    spec(0x2029, "RAW type", Fmt::Enum(SONY_RAW_TYPE)),
    spec(0x0102, "Quality", Fmt::Enum(SONY_QUALITY)),
    spec(0x202e, "Quality (2)", Fmt::SonyQuality2),
    spec(0x2039, "JPEG / HEIF", Fmt::Enum(SONY_JPEG_HEIF)),
    spec(0xb02b, "Full image size", Fmt::SonyFrameSize),
    spec(0xb02c, "Preview size", Fmt::SonyFrameSize),
    spec(0xb041, "Exposure mode", Fmt::Enum(SONY_EXPOSURE_MODE)),
    spec(0x202c, "Metering", Fmt::Enum(SONY_METERING)),
    spec(0x202d, "Exposure standard adjustment", Fmt::Ev),
    spec(0xb049, "Release mode", Fmt::Enum(SONY_RELEASE_MODE)),
    spec(0xb04a, "Sequence number", Fmt::SonySequence),
    spec(0x201a, "Electronic front curtain", Fmt::Enum(SONY_ON_OFF)),
    spec(0x201b, "Focus mode", Fmt::Enum(SONY_FOCUS_MODE)),
    spec(0x201c, "AF area mode", Fmt::Enum(SONY_AF_AREA_MODE)),
    spec(0x201d, "Flexible spot position", Fmt::Ints),
    spec(0x201e, "AF point selected", Fmt::Enum(SONY_AF_POINT)),
    spec(0x2020, "AF points used", Fmt::SonyAfPoints),
    spec(0x2021, "AF tracking", Fmt::Enum(SONY_AF_TRACKING)),
    spec(0x2027, "Focus location", Fmt::SonyFocusLocation),
    spec(0x2037, "Focus frame size", Fmt::SonyU16Size),
    spec(0x202f, "Pixel shift", Fmt::SonyPixelShift),
    spec(0xb027, "Lens ID", Fmt::SonyLensId),
    spec(0xb02a, "Lens spec", Fmt::SonyLensSpec),
    spec(0x0105, "Teleconverter", Fmt::Enum(SONY_TELECONVERTER)),
    spec(0xb026, "Image stabilization", Fmt::Enum(SONY_ON_OFF)),
    spec(0xb04b, "Anti-blur", Fmt::Enum(SONY_ANTI_BLUR)),
    spec(0x0115, "White balance", Fmt::Enum(SONY_WHITE_BALANCE)),
    spec(0xb054, "White balance (2)", Fmt::Enum(SONY_WHITE_BALANCE2)),
    spec(0x0112, "White balance fine tune", Fmt::Int),
    spec(0x2014, "WB shift", Fmt::SonyWbShift),
    spec(0x2026, "WB shift (precise)", Fmt::SonyWbShift),
    spec(0xb021, "Colour temperature", Fmt::SonyColorTemperature),
    spec(0xb022, "Colour compensation filter", Fmt::Int),
    spec(0x202b, "AWB priority", Fmt::Enum(SONY_AWB_PRIORITY)),
    spec(0xb020, "Creative style", Fmt::Text),
    spec(0xb029, "Colour mode", Fmt::Enum(SONY_COLOR_MODE)),
    spec(0xb023, "Scene mode", Fmt::Enum(SONY_SCENE_MODE)),
    spec(0xb052, "Intelligent auto", Fmt::Enum(SONY_INTELLIGENT_AUTO)),
    spec(0x2004, "Contrast", Fmt::Int),
    spec(0x2005, "Saturation", Fmt::Int),
    spec(0x2006, "Sharpness", Fmt::Int),
    spec(0x2035, "Sharpness range", Fmt::Int),
    spec(0x2007, "Brightness", Fmt::Int),
    spec(0x2032, "Shadows", Fmt::Int),
    spec(0x2033, "Highlights", Fmt::Int),
    spec(0x2034, "Fade", Fmt::Int),
    spec(0x2036, "Clarity", Fmt::Int),
    spec(0xb025, "Dynamic range optimizer", Fmt::Enum(SONY_DRO)),
    spec(0xb04f, "Dynamic range optimizer (2)", Fmt::Enum(SONY_DRO2)),
    spec(0xb024, "Zone matching", Fmt::Enum(SONY_ZONE_MATCHING)),
    spec(0x200a, "Auto HDR", Fmt::SonyAutoHdr),
    spec(0x2008, "Long exposure NR", Fmt::Enum(SONY_LONG_EXPOSURE_NR)),
    spec(0x2009, "High ISO NR", Fmt::Enum(SONY_HIGH_ISO_NR)),
    spec(0xb050, "High ISO NR (2)", Fmt::Enum(SONY_HIGH_ISO_NR2)),
    spec(0x200b, "Multi-frame NR", Fmt::Enum(SONY_MULTI_FRAME_NR)),
    spec(0x2023, "Multi-frame NR effect", Fmt::Enum(SONY_NR_EFFECT)),
    spec(0x200e, "Picture effect", Fmt::Enum(SONY_PICTURE_EFFECT)),
    spec(0x200f, "Soft skin effect", Fmt::Enum(SONY_SOFT_SKIN)),
    spec(0x2016, "Auto portrait framed", Fmt::Enum(SONY_YES_NO)),
    spec(0x2011, "Vignetting correction", Fmt::Enum(SONY_CORRECTION)),
    spec(
        0x2012,
        "Chromatic aberration correction",
        Fmt::Enum(SONY_CORRECTION),
    ),
    spec(0x2013, "Distortion correction", Fmt::Enum(SONY_CORRECTION)),
    spec(0x2028, "Variable low-pass filter", Fmt::SonyLowPass),
    spec(0x2017, "Flash action", Fmt::Enum(SONY_FLASH_ACTION)),
    spec(0xb048, "Flash level", Fmt::SonyFlashLevel),
    spec(0x0104, "Flash exposure compensation", Fmt::Ev),
    spec(0x2002, "Rating", Fmt::Int),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn render(fmt: Fmt, v: Value) -> Option<String> {
        fmt.render(&v)
    }

    #[test]
    fn exposure_and_aperture() {
        use rawler::formats::tiff::Rational;
        assert_eq!(
            render(Fmt::Exposure, Value::Rational(vec![Rational::new(1, 80)])).as_deref(),
            Some("1/80 s")
        );
        assert_eq!(
            render(Fmt::Exposure, Value::Rational(vec![Rational::new(25, 10)])).as_deref(),
            Some("2.5 s")
        );
        assert_eq!(
            render(Fmt::Aperture, Value::Rational(vec![Rational::new(32, 10)])).as_deref(),
            Some("f/3.2")
        );
        assert_eq!(
            render(
                Fmt::Ev,
                Value::SRational(vec![rawler::formats::tiff::SRational::new(-3, 10)])
            )
            .as_deref(),
            Some("-0.3 EV")
        );
        // Av 1.695 → f/1.8; Tv 6.32 → 1/80 s.
        assert_eq!(
            render(
                Fmt::ApexAperture,
                Value::Rational(vec![Rational::new(1695, 1000)])
            )
            .as_deref(),
            Some("f/1.8")
        );
        assert_eq!(
            render(
                Fmt::ApexShutter,
                Value::SRational(vec![rawler::formats::tiff::SRational::new(632, 100)])
            )
            .as_deref(),
            Some("1/80 s")
        );
    }

    #[test]
    fn versions_and_enums() {
        assert_eq!(
            render(Fmt::Version, Value::Undefined(b"0232".to_vec())).as_deref(),
            Some("2.32")
        );
        assert_eq!(
            render(Fmt::Flash, Value::Short(vec![16])).as_deref(),
            Some("Did not fire, suppressed")
        );
        assert_eq!(
            render(Fmt::Flash, Value::Short(vec![0x19])).as_deref(),
            Some("Fired, auto")
        );
        assert_eq!(
            render(Fmt::Enum(ORIENTATION), Value::Short(vec![8])).as_deref(),
            Some("Rotate 270 CW")
        );
        assert_eq!(
            render(Fmt::Enum(ORIENTATION), Value::Short(vec![42])).as_deref(),
            Some("42")
        );
    }

    #[test]
    fn sony_formats() {
        // FE 50mm F1.8 as the a6700 writes it.
        assert_eq!(
            render(
                Fmt::SonyLensSpec,
                Value::Byte(vec![2, 0, 0x50, 0, 0, 0x18, 0, 0])
            )
            .as_deref(),
            Some("50mm f/1.8")
        );
        assert_eq!(
            render(
                Fmt::SonyLensSpec,
                Value::Byte(vec![2, 0, 0x16, 0, 0x50, 0x35, 0x56, 0])
            )
            .as_deref(),
            Some("16-50mm f/3.5-5.6")
        );
        assert_eq!(
            render(Fmt::SonyFileFormat, Value::Byte(vec![4, 0, 1, 0])).as_deref(),
            Some("ARW 4.0.1")
        );
        assert_eq!(
            render(Fmt::SonyFileFormat, Value::Byte(vec![3, 3, 0, 0])).as_deref(),
            Some("ARW 2.3")
        );
        assert_eq!(
            render(Fmt::SonyFrameSize, Value::Long(vec![4128, 6192])).as_deref(),
            Some("6192 × 4128")
        );
        assert_eq!(
            render(Fmt::SonyAutoHdr, Value::Long(vec![0x10001])).as_deref(),
            Some("Auto")
        );
        assert_eq!(
            render(Fmt::SonyAutoHdr, Value::Long(vec![0x10016])).as_deref(),
            Some("1.5 EV")
        );
        assert_eq!(
            render(Fmt::SonyFlashLevel, Value::SShort(vec![-3])).as_deref(),
            Some("-3/3")
        );
        assert_eq!(
            render(Fmt::SonyAfPoints, Value::Byte(vec![0, 5])).as_deref(),
            Some("8 10")
        );
        assert_eq!(
            render(
                Fmt::SonyU16Size,
                Value::Undefined(vec![135, 0, 138, 0, 1, 1])
            )
            .as_deref(),
            Some("135 × 138")
        );
        assert_eq!(
            render(Fmt::SonyPixelShift, Value::Undefined(vec![0; 6])).as_deref(),
            Some("None")
        );
    }

    #[test]
    fn exif_block_containers() {
        // JPEG: SOI, APP0 (skipped), APP1 Exif.
        let mut jpeg = vec![0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0xaa, 0xbb];
        jpeg.extend([0xff, 0xe1, 0x00, 0x0a]);
        jpeg.extend(b"Exif\0\0II");
        jpeg.extend([0xff, 0xda]);
        assert_eq!(exif_block(&jpeg), Some(&b"II"[..]));

        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend([0, 0, 0, 1]);
        png.extend(b"IHDR");
        png.extend([7, 0, 0, 0, 0]);
        png.extend([0, 0, 0, 2]);
        png.extend(b"eXIf");
        png.extend(b"MM");
        png.extend([0, 0, 0, 0]);
        assert_eq!(exif_block(&png), Some(&b"MM"[..]));

        let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
        webp.extend(b"VP8 ");
        webp.extend([1, 0, 0, 0, 9, 0]);
        webp.extend(b"EXIF");
        webp.extend([2, 0, 0, 0]);
        webp.extend(b"II");
        assert_eq!(exif_block(&webp), Some(&b"II"[..]));

        assert_eq!(exif_block(b"nothing"), None);
    }
}
