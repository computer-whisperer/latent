# image_developer — design (draft, 2026-09-10)

Status: planning. Nothing built. This file is the ground truth for the
plan; the conversation that produced it is not.

## Goal

Develop Sony a6700 raw files (ARW) into downstream formats, HDR and SDR,
with the same pipeline reachable from a CLI (batch, scripted) and a GUI
(interactive adjustment, browsing). Display is color-managed through
damascene on prism.

## Inputs (verified against the card dump)

Source: `~/workspace/playground/camera_control/downloads/slot1/DCIM/100MSDCF`
(camsync pulls these; production inbox is `/ceph/christian/Photos/inbox`).

| Fact | Value |
|---|---|
| Camera | Sony ILCE-6700, firmware 2.00 |
| Files | 239 ARW + 277 JPG (239 RAW+JPEG pairs, 38 JPEG-only) |
| Sensor raster | 6272 x 4168, RGGB, 14-bit, white level 16383 |
| Default crop | 6192 x 4128 at (26, 20) |
| Compression | "Sony ARW Compressed": StripByteCounts = 6272*4168 = 1 byte/pixel, i.e. Sony's lossy ARW2 block scheme. The a6700 also offers Lossless Compressed; worth switching the camera before shooting anything that matters. |
| Embedded | 6192x4128 JPEG preview; Sony lens-correction params (distortion, vignetting, CA) as SubIFD tags |
| Sidecar JPEG | sRGB, Creative Style Standard |

## Ecosystem facts that shape the design (from the surveys)

- **achromat** (3.1k lines): decode stack for PNG/JPEG/WebP/AVIF/JXL/JXR/EXR/HDR
  with `ColorEncoding {tf, primaries, luminances}`, ICC→parametric CMS,
  `bt2390_eetf`, `LuminanceControl`, and a bridge to damascene `Image`.
  No camera raw, no encoders. Its `RawImage`/`RawPixels` types mean
  "straight-alpha before color resolution", not camera raw.
- **damascene 0.8.0** (crates.io, wgpu 30, WGSL): `Image` in
  Rgba8/16/F16/F32 with `ColorSpace {primaries, transfer, reference_luminance_nits}`,
  normalised to scRGB-linear f16 working space; `DynamicRangeLimit` +
  BT.2390 remaster in the image shader; `viewport()` with
  `FitPolicy::Contain` documented as the photo-viewer default;
  `AppTexture`/`surface()` with `Rgba16Float` for app-owned GPU textures;
  `WinitWgpuApp::gpu_setup`/`before_paint` for apps that own GPU work;
  sliders, number scrubber, numeric input, grid, virtual_list, menubar,
  resize_handle, file drag-and-drop. **No compute pipelines anywhere, no
  histogram/curve widgets, no encoders, no file picker.** Copy the
  `[profile.dev.package.*]` opt-level block into our workspace root.
- **prism**: standard `wp_color_management_v1` v2; no ICC on the wire,
  no HLG; TFs sRGB/gamma22/BT.1886/PQ/ext-linear; primaries sRGB/P3/
  BT.2020/custom. A damascene app never tags the surface itself; Mesa
  WSI tags the `Rgba16Float` swapchain as scRGB. Reference-white
  anchoring per output (80 nits SDR, 203 nits HDR). Decode/encode
  hard-clip today; the measured per-output 3D LUT does the gamut
  projection. `prism-ipc` can hand out the measured LUT/gamut mesh
  (soft-proof against the real panel, later).
- **damascene-gallery** (1.2k lines, damascene 0.4.4, last commit
  2026-06-28): three-tier decode worker pool, virtualized thumbnail grid
  (`ConstrainedHigh`), viewer with LRU (`NoLimit`, `t` for SDR preview),
  HDR badge, rfd folder picker. No pan/zoom. Mine it; do not extend it.
- **prism-explorer**: `explorer-thumbs` (467 lines) is an on-disk
  linear-f16 thumbnail cache tagged with primaries + reference luminance.
  Reuse for the library view.
- **tristim**: precedent for the crate split (`-gather` shared core,
  `-cli`, `-gui`), edition 2024, MIT/Apache.

## Raw decoding: rawler (decision proposed)

`rawler 0.8.0` (dnglab, LGPL-2.1, pure Rust): has an `ILCE-6700` camera
profile with Adobe-style ColorMatrix for illuminants A and D65, ARW2
decoder, as-shot WB, black/white levels, crop, orientation, EXIF, lens
DB, DNG writer. Does not parse Sony's embedded lens-correction tags
(we read those ourselves from the TIFF SubIFD; exiv2 shows them).

Alternative: libraw 0.22.2 via FFI (also LGPL). Installed system-wide;
`dcraw_emu`/`unprocessed_raw` decode these files in 0.15 s (unpack) /
0.56 s (full dcraw pipeline, 24 threads). Keep as an external
cross-check for the decode stage (compare unpacked Bayer planes and
matrices), not as a dependency.

Both are LGPL; our crates stay MIT/Apache and link them as normal deps.
Flagged, not a blocker for a personal tool.

## Processing: one GPU pipeline serving CLI and GUI (decision proposed)

The CLI and GUI must produce bit-identical output from the same
parameters, so there is exactly one pipeline. It runs as wgpu 30 compute
passes (same wgpu as damascene, so the GUI can inlay damascene on the
same device and hand it the result as an `AppTexture Rgba16Float`
without a copy). The CLI creates a headless wgpu device. A 26 MP frame
at f32 is 400 MB per RGBA plane; keep intermediates at RGBA16Float or
single-channel where possible, and process at a preview scale for
interactive sliders (full-res only on export and 1:1 zoom).

Rejected: CPU/rayon pipeline. Fine for the CLI (~100s of ms per pass on
24 threads) but interactive editing at 26 MP needs the GPU, and a hybrid
means two implementations to keep identical.

Pipeline stages, v1 in order:

1. Unpack (rawler, CPU) → u16 Bayer plane + `SensorInfo` (CFA, black
   per channel, white, crop, WB coeffs, matrices, orientation, EXIF).
2. Linearize: subtract black, scale by white → f32 [0,1], upload.
3. White balance (as-shot, or temperature/tint over the two matrices
   via CCT interpolation, DNG-style).
4. Demosaic: bilinear first (proof of pipeline), then RCD (good
   quality, simple, GPU-friendly). AMaZE later if RCD shows artifacts.
5. Camera RGB → XYZ (D65) → working space: linear BT.2020, D65,
   reference white = scene diffuse white. Stored as RGBA16Float.
6. Exposure, highlight reconstruction (clip-aware, in camera space
   before matrix, so this is really stage 4.5), tone curve (parametric,
   later a spline editor), contrast/saturation. All in linear working
   space.
7. Output transform, chosen per export target:
   - SDR: display-referred, sRGB or Display-P3, gamma per format.
   - HDR: PQ/BT.2020 with mastering metadata (peak from a
     user-chosen master nits, default 1000), or scRGB linear f16.
   Preview in the GUI is always the working-space f16 texture; damascene
   does the remaster to the live output (`DynamicRangeLimit`).
8. Encode (CPU): see formats.

Later stages, not v1: lens corrections from the Sony tags (distortion,
vignetting, CA), noise reduction, sharpening, crop/rotate, local edits,
DCP-style camera profiles, soft-proof against prism's measured LUT.

## Output formats

v1: EXR (scRGB linear f16 master; `exr` crate), 16-bit PNG (SDR now;
HDR with cICP PQ/BT.2020 once we confirm `png 0.18` writes cICP),
JPEG (SDR; `image`). Next: AVIF 10-bit PQ via `rav1e` (`ravif` is
8-bit-oriented, verify), JPEG XL via libjxl (`jpegxl-rs` is GPL-3; use
`zune-jpegxl` for lossless only, or shell out to `cjxl` which is
installed), linear DNG via rawler's writer. TIFF 16-bit if anything
needs it.

## Edit state: sidecars

Non-destructive. Per-raw `DSC00039.ARW.<ext>` sidecar (TOML) holding the
full `DevelopParams` plus a schema version. CLI: `develop DSC00039.ARW
--out x.exr` uses the sidecar if present, flags override; `--print
-params` dumps defaults. GUI writes sidecars on change. Params are the
API boundary between GUI, CLI, and pipeline, so they get a stable
serialized form from day one.

## Workspace layout (tristim pattern)

```
Cargo.toml              workspace, edition 2024, MIT OR Apache-2.0, damascene's
                        [profile.dev.package.*] opt-level block copied in
crates/<name>-core/     rawler unpack, SensorInfo, DevelopParams, wgpu compute
                        pipeline (headless-capable), output transforms
crates/<name>-formats/  encoders (EXR/PNG/JPEG/…), one fn per target
crates/<name>-cli/      clap: inspect, develop, batch, params
crates/<name>-gui/      damascene WinitWgpuApp: library grid (lifted from
                        damascene-gallery + explorer-thumbs), develop view
                        (viewport over AppTexture, control panel, histogram)
```

Name TBD (project dir is `image_developer`; the ecosystem favors
optics/color names). Sidecar extension follows the name.

Not in achromat: the pipeline has parameters and GPU deps, achromat is a
decode stack. But `<name>-core` exposes `develop_default(path) ->
achromat::DecodedImage` so achromat (feature-gated) or the gallery can
later open raws with a sensible default rendering.

## Milestone 1 results (2026-09-10)

Pipeline proof done: rawler unpack → headless wgpu (linearize + WB +
highlight clamp → bilinear demosaic + camera→BT.2020 matrix) → EXR/PNG.

- rawler's Bayer plane vs libraw's `unprocessed_raw` on DSC00039.ARW:
  same 6272x4168 raster, max abs difference 8 on a 16k scale, typical
  difference 1 (ARW2 curve rounding). Decode is trustworthy.
- rawler's a6700 profile: black 512, white 15360; decoded values reach
  16634. libraw reaches 16628. Sites above the white level clamp to 1.0
  after WB, so the profile's white only matters for the clamp threshold.
  Revisit when doing highlight reconstruction.
- Release timings on one 26 MP frame: unpack 75 ms, GPU init 45 ms,
  develop 115 ms (both passes plus readback), EXR encode 0.6 s, PNG
  encode 3 s (default zlib level; not tuned).
- Without the highlight clamp, clipped neutrals rendered magenta (WB
  gains 2.89 R / 1.40 B on clipped sites). The clamp is the dcraw
  "clip" behaviour; real reconstruction is milestone 2.
- Against the camera JPEG the render is darker and cooler: no tone
  curve yet, and the D65 matrix is used unadapted regardless of the
  shot's white point. Both are milestone 2 items.

## Milestone 2 arc (started 2026-09-10)

Pipeline after this arc: linearize (no clamp) → demosaic (RCD, bilinear
kept as the fast path) → colour pass (highlight reconstruction, camera
matrix with Bradford adaptation, exposure, tone curve) → readback →
CPU output transform (orientation, primaries, transfer) → encoder.

- **White point.** As-shot WB multipliers give the camera neutral;
  its XYZ comes from the inverse colour matrix, CCT from McCamy, the
  matrix is interpolated between the profile's illuminants by inverse
  CCT (DNG method, 3 iterations). The camera→working matrix is
  `XYZ→BT.2020 · Bradford(white→D65) · inv(CM) · diag(neutral)`,
  applied after WB, so neutrals hit working white exactly and
  everything else is Bradford-adapted rather than sensor-space
  von Kries. `WhiteBalance::Temperature { kelvin, tint }` maps through
  the daylight/Planckian locus to a white and back to multipliers.
- **Demosaic: RCD**, reimplemented from the published algorithm
  (directional discrimination from squared second differences on
  same-colour samples, ratio-corrected green from a 3x3 low-pass,
  colour-difference R/B). Three GPU passes: VH+LPF, green, RB.
- **Highlights: blend**, per pixel after demosaic. Each channel clamps
  at its own post-WB clip level (a clipped value is a lower bound), a
  clipped channel is lifted to at least the brightest still-valid
  channel, and once a second channel clips the pixel desaturates
  toward a neutral at its own luminance (weight = the second-largest
  clip factor, so the transition is continuous); fully clipped pixels
  are neutral at ~1.5x diffuse white on this camera. 2% soft ramp
  under the clip. Rejected: desaturating by the
  fraction of clipped channels; green clips first on this sensor and
  that left near-white walls pink.
- **Tone curve: sigmoid** in linear light, grey-anchored (0.18→0.18),
  `contrast` sets the slope at grey, `skew` splits toe/shoulder
  exponents, `peak` (relative to display white) comes from the output
  target: 1.0 for SDR, master_nits/203 for PQ. Per-channel by default
  with a `hue_preservation` blend toward the luminance-ratio result.
  `ToneCurve::None` keeps the image scene-referred (EXR master).
- **Formats.** JPEG (sRGB, quality param), PQ PNG (16-bit, BT.2020
  primaries, PQ, `cICP` chunk, 203 nits = diffuse white), EXR and
  sRGB PNG as before. Orientation applied on the CPU before encode.
- **Sidecars** `<raw>.idev.toml` = `DevelopParams`; CLI flags
  override; `--save-params` writes back. **Batch**: many inputs, one
  GPU, one output directory, skip-existing.

## Milestone 2 results (2026-09-10)

Everything in the arc landed; the CLI is usable for real batches.

- Tone: the sigmoid default (contrast 1.6, per-channel) puts the
  calibration-board frame visually on top of the camera JPEG; the
  high-ISO cat frame is close but a touch flatter in the shadows.
  `skew` was dropped from the design: separate toe/shoulder exponents
  put a slope kink at grey. A finite white point (input that maps to
  peak, instead of the asymptote) is the likely next tone control.
- Highlights: the first "desaturate by fraction clipped" rule left
  green-clipped walls pink; the lift-to-brightest-channel rule with
  desaturation by the second-largest clip factor is continuous and
  neutral. Fully clipped regions land at ~0.95 after the SDR sigmoid,
  faintly textured by the lift; a white-point control would finish
  them off.
- Orientation: rawler reports Normal for Sony portrait files; the
  EXIF tag is authoritative.
- RCD costs nothing measurable: the whole develop is ~115 ms at 26 MP
  including readback, so the GPU is not the bottleneck. Encoders are:
  JPEG q92 1 s, 8-bit PNG 3 s, 16-bit PQ PNG far longer (zlib on
  noisy 16-bit data; not tuned). EXR 0.7 s.
- Sidecars work as `<raw>.idev.toml`; `--save-params` writes the
  resolved parameters, later runs pick them up, flags override.

## Milestone 3 arc (started 2026-09-10): single-image develop window

Decisions: preview-scale pipeline for slider drags, full resolution at
1:1 and on export; single-image window first, library view later.

- **Crate `idev-gui`**, binary `idev-gui <raw>`. damascene
  `WinitWgpuApp` host: `gpu_setup` wraps the host device in `Gpu`,
  `before_paint` reruns whatever is dirty.
- **Pipeline split for interactivity.** The full-resolution linearize
  plane and demosaiced camera RGB stay resident on the GPU and are
  recomputed only when white balance or the demosaic choice changes.
  The colour pass (highlights, matrix, exposure, tone) is what sliders
  touch; it renders into the display texture and takes an integer
  `scale` that box-averages scale×scale camera-RGB pixels, so a
  fit-to-window preview costs a few ms and 1:1 uses scale 1 at full
  resolution. Averaging happens in camera-space linear light before
  highlight reconstruction, which is the right place for it.
- **Display texture** is an `AppTexture` (`Rgba16Float`, opaque) shown
  through `surface()` inside a `viewport()` with `FitPolicy::Contain`.
  damascene's working space is scRGB linear, so the colour pass gets a
  `display_primaries` switch: identity (BT.2020, exports) or the
  BT.2020→sRGB matrix (GUI). The tone curve peak follows the host's
  negotiated headroom: 1.0 on SDR, the reported headroom on HDR.
- **Controls**: exposure, temperature, tint, contrast, hue
  preservation, highlights, demosaic; export (JPEG / PNG / PQ PNG /
  EXR via an rfd save dialog, full-resolution readback through the
  existing `develop`), reset. Sidecar written on every change.
- **Not in this arc**: histogram, library grid, noise reduction.

## Milestone 3 results (2026-09-10)

The single-image window works on prism's HDR output. Verified by
screenshot: the preview renders through `surface()` inside a
`viewport()`, the badge reports the negotiated headroom (2.0x on the
400-nit-advertised output), and the sidecar round-trips.

- Preview scale converges within two frames (the zoom is read in
  `build`, consumed in `before_paint`; a mismatch wakes one more
  frame). At fit the preview is ~1240 px wide; 1:1 allocates a
  full-resolution texture.
- `prepare` (linearize + RCD) submits in ~30 ms; the colour pass per
  slider event is a few ms. Not measured precisely yet.
- Exports run on a thread with a private headless device, so the
  window never blocks on readback or encode.
- Not done: histogram, library view, keyboard focus polish on the
  scrubbers, a white-point tone control. Interaction (drag, tabs,
  export dialog) was not exercised by me; it is wired through
  damascene's documented `apply_event` folds.

## Chunk after milestone 3 (2026-09-11): white point, histogram

- **Sigmoid white point.** `ToneCurve::Sigmoid` gained `white` (scene
  value, relative to diffuse white, that maps to the output peak;
  default 4.0). Curve is `peak · F(x) / F(white)` with
  `F(x) = x^c / (x^c + s)`; `s` solves the grey anchor for the chosen
  contrast, which only exists above a contrast floor of
  `ln(peak/grey) / ln(white/grey)`, so contrast is raised to it when
  needed. Old sidecars load with the default. At white 2.9 (this
  camera's fully-clipped level) clipped regions reach exact white.
- **Histogram.** Compute pass over the preview texture: 3 × 256 bins
  of sRGB-encoded values after division by the display peak, atomics
  into a storage buffer, copied to a staging buffer and mapped
  asynchronously; the UI polls without blocking, one in flight at a
  time. Drawn as 128 overlaid translucent bars per channel, log
  counts, plain Els (damascene's plot API was more surface than the
  widget needed).
- GUI sidecar writes are debounced 250 ms on a writer thread after
  synchronous writes on a Ceph mount (20–70 ms each) starved the
  Wayland socket during drags and the compositor dropped the client.

## Library arc (started 2026-09-11): folder grid + develop as a mode

Decision (proposed in the chunk report, taken as ratified by "let's
continue with the library view"): grid thumbnails come from the
embedded camera JPEG when a raw has no sidecar, and from a scaled
develop with the sidecar's parameters once one exists. The embedded
preview is instant (1616×1080 JPEG, ~200 KB, in every a6700 ARW) and
matches the camera's rendering; the developed thumbnail shows the edit.

- **Entry point**: `idev-gui [PATH]`. A directory opens the library;
  a file opens the develop view on it with its folder as the library;
  no argument means the current directory. `o` picks another folder.
- **App structure**: one `IdevApp` with `Mode::{Library, Develop}`.
  The GPU state (device, developer, preview texture) lives on the app;
  the per-file state (frame, params, sidecar writer, histogram) is a
  `Session` swapped on open. Enter / double-click opens the selected
  file, Escape returns to the grid, Left/Right in develop mode step
  through the folder. Dropping a session joins its sidecar writer, so
  the thumbnail requeued on return sees the final sidecar.
- **Thumbnails**: `idev-gui::thumbs` is the explorer-thumbs design
  (content-addressed by path + mtime + size, temp-and-rename, corrupt
  = miss, LRU byte sweep) with two changes: the key also covers the
  sidecar's mtime + size, and entries are sRGB RGBA8 at a 512 px long
  edge rather than linear f16 (developed thumbs are rendered for the
  SDR target, so nothing above diffuse white is lost). Cache root is
  `$XDG_CACHE_HOME/idev/thumbs`.
- **Producing a thumbnail**: no sidecar → rawler's `preview_image`
  (the JPEGInterchangeFormat blob), box-downscaled in linear light and
  rotated by the EXIF orientation. Sidecar → `unpack` + `prepare` +
  the colour pass at the integer scale that lands the long edge at
  or under 512 px, read back, oriented, encoded sRGB8. Developed
  thumbs go through one headless device shared by the workers behind
  a mutex (a full-resolution `prepare` holds ~0.5 GB of textures).
- **Loader**: the gallery's pool, reduced to one job kind: a hot queue
  (rows the grid realized) ahead of a sweep over the folder in file
  order; `invalidate(index)` requeues one file after its session
  closes. Results are drained in `before_build`.
- **Grid**: damascene `virtual_grid` keyed `"grid"`, cells are the
  thumbnail (`ImageFit::Contain` on a card background) over a caption
  with the file name and a dot when a sidecar exists. Column count
  from the build-time viewport width; selection ring; `ScrollRequest`
  keeps the selection visible on keyboard navigation.
- **Not in this arc**: ratings/filters, recursive scan, thumbnail
  rendering at the display's HDR peak, gallery retirement.

## Library arc results (2026-09-11)

Verified by screenshot on the 239-file card folder: the grid fills
within the first seconds (embedded previews; three workers), the
selection ring and captions render, a raw given a sidecar with +1 EV
shows the brighter developed thumbnail and the dot marker, and
`idev-gui FILE` opens in develop mode (unpack 83 ms, prepare 68 ms on
the host device). The thumbnail cache holds ~420 KB per entry (101 MB
for the folder). The headless thumbnail device is created lazily on
the first developed thumbnail. Not exercised by me: keyboard
navigation, Escape back to the grid with the thumbnail requeue, the
folder picker (no input injection on this desktop).

## Milestone 4 arc (started 2026-09-11): polish

Survey results that shape the order:

- **Lens corrections.** Every a6700 ARW carries Sony's in-body tables
  in the raw SubIFD: `0x7037` DistortionCorrParams, `0x7035`
  ChromaticAberrationCorrParams, `0x7032` VignettingCorrParams. First
  value is the knot count `nc` (11 on the FE 50 F1.8; CA carries `2nc`
  values, red then blue). RawTherapee's `lensmetadata.cc` decodes them:
  knots at normalised radius `(i + 0.5) / (nc − 1)` with 1.0 at the
  half-diagonal, linear between; distortion factor `v·2⁻¹⁴ + 1`
  (source radius over output radius), CA factor `v·2⁻²¹ + 1` per
  channel, vignetting gain `1 / 2^(0.5 − 2^(v·2⁻¹³ − 1))`. On the
  sample frames: −0.26 % radius at the corner, CA ±1 px, up to +7 %
  gain at f/3.2. Applied in the colour pass: per-output-pixel source
  coordinate per channel (bilinear at scale 1, block average at
  preview scales), vignetting gain after highlight reconstruction so
  clip detection stays on the sensor's levels. `DevelopParams.lens`
  holds three booleans, default on (old sidecars pick them up).
  Validation: the checkerboard frames against the camera JPEG, which
  has the same corrections applied.
- **Export formats.** `ravif` 0.13 encodes AVIF from 8-bit input only
  (10-bit internally), so SDR AVIF is a few lines; HDR AVIF (10-bit PQ
  BT.2020 with cICP) needs `rav1e` + `avif-serialize` directly.
  `jpegxl-rs` 0.15 binds the system libjxl 0.12 and takes f32 input;
  its colour-encoding surface needs checking for PQ/BT.2020.
- **Soft-proof.** prism-ipc has `Request::Lut3d { output }` (the
  effective 33³ calibration LUT: panel *command* per PQ-shaped
  BT.2020 input, domain nits or drive) and `Request::GamutMesh` (the
  measured reachable-colour boundary). A soft-proof against the LUT is
  therefore a gamut/peak warning: where the command saturates, the
  panel cannot show the colour. Semantics to agree with the user
  before building.
- **Gallery.** damascene-gallery views JXR/JXL/AVIF/EXR/PNG through
  achromat; idev's library only lists raws. Retiring it needs a
  non-raw path in the library (achromat decode → thumbnail → viewer).
  Proposal: keep it until then.

Order: lens corrections first (this chunk), then the format and
soft-proof questions go back to the user.

## Lens correction results (2026-09-11)

- Uncorrected renders are byte-identical to before the sampling
  rewrite (the identity path samples exactly); develop time unchanged
  at ~106 ms full-resolution.
- **Vignetting confirmed** against the camera JPEG of DSC00049 (the
  camera applies shading compensation; tag 0x7031 = 272): after
  matching tone curves on the centre, the JPEG's corners sit 4–5 %
  above the uncorrected render and within 1–3 % of the corrected one;
  the table predicts ×1.045 at that radius.
- **Distortion not validated**: every card frame was shot with the
  camera's distortion compensation off (tag 0x7036 = 0), so the JPEGs
  match the uncorrected geometry to 0.00 px in all corners. Our
  correction moves the corners 6 px inward on the FE 50 F1.8, per
  RawTherapee's decoding. Needs a frame shot with "Distortion Comp."
  on, or a straight-edge subject at the frame edge.
- **CA not validated**: the shift is ~1 px, and every measurement on
  demosaiced output is blind to it (colour-difference demosaicing
  aligns R/B fine structure to G; phase correlation on the mosaic
  planes is unreliable). Visual check at ISO 2500 was inconclusive.
  Needs a low-ISO frame with a high-contrast edge near a corner,
  toggled at 1:1 in the GUI.
- GUI: a "Lens" sidebar group with three switches; not exercised.

## AVIF export results (2026-09-11)

- `idev-formats::avif`: rav1e (pure Rust, no assembly, `threading`)
  plus avif-serialize. Still picture, 4:4:4, identity matrix (planes
  carry G, B, R), full range, 10-bit in both flavours. SDR widens
  sRGB 8-bit; HDR narrows the PQ 16-bit samples. ravif's quality →
  quantizer mapping; speed preset 6.
- ffprobe reads the tags as intended: `gbrp10le`, `pc` range,
  `gbr`, sRGB / BT.709 for SDR, `smpte2084` / `bt2020` for HDR.
- Round trip through ffmpeg's decoder at quality 92: SDR 43.8 dB
  PSNR against the PNG (mean 1.3 / 255, max 13); HDR 44.9 dB against
  the PQ PNG in 16-bit code values.
- Cost on the 25 MP frame: 27 s per file on 16 threads (JPEG: 1 s,
  PQ PNG: 5 s); SDR 18 MB, HDR 12 MB. Speed preset and quality are
  the knobs if that is too slow.
- `--format avif` (with `--hdr` for PQ) on the CLI; "AVIF" and "HDR
  AVIF" buttons in the GUI export group (not exercised).

## Warning overlay results (2026-09-11)

- `Render { target, scale, display, warn: Option<Gamut> }` replaces
  the colour pass's loose arguments. With `warn`, the shader zebra-
  marks (6 px diagonal stripes) pixels with a channel at the output
  peak in red and pixels outside the panel's gamut in cyan, after the
  tone curve and before the display matrix. `color::Gamut` derives
  the BT.2020 → panel matrix from chromaticities (sRGB, Display P3,
  BT.2020, Adobe RGB; a test checks the derivation against the
  constants).
- The panel's gamut comes from the compositor's preferred primaries
  in damascene's colour-management diagnostics, sRGB when nothing is
  reported; the badge shows it ("HDR · 2.0x · BT.2020"). prism
  advertises BT.2020 on the HDR output, so the gamut half is vacuous
  there and the peak half is what fires. `w` or the Warn button
  toggles it; the histogram keeps its last clean reading while the
  overlay is on.
- Verified by screenshot with a sidecar at white 2.0: the clipped
  window and sill show the red zebra; the rest is untouched.

## Rendered images as sources (2026-09-11)

Option 2 from the chunk report, ratified by the user: a rendered
image enters the working space and goes through the same colour pass.

- `source::Source::{Raw(Box<SensorFrame>), Image(ImageSource)}`;
  `Source::load` dispatches on extension (rawler for raws, achromat's
  `load_straight` for PNG/JPEG/WebP/AVIF/JXL/JXR/EXR/HDR). achromat is
  consumed without its `damascene` feature (it pins damascene-core
  0.4, ours is 0.8) and converted here: transfer linearized (PQ
  anchored at the declared reference or 203 cd/m², linear passed
  through, the rest by their EOTF), primaries rotated into BT.2020
  through `color::Gamut` or the custom chromaticities, stored as
  linear BT.2020 f16 in viewing orientation.
- `Developer::prepare` uploads an image source straight into the
  camera-RGB texture with identity matrices; `Prepared::is_image`
  turns off highlight handling (shader mode 2) and lens tables, and
  `matches` ignores white balance and demosaic. The GUI hides the
  White balance, Pipeline and Lens groups for non-raws; the info
  block shows the source encoding.
- Verified: a PNG through the pipeline with the tone curve off comes
  back with zero error in all 4.8 M samples; the camera JPEG differs
  by up to 4 codes, which is zune-jpeg vs libjpeg IDCT, not the
  pipeline (the PNG test isolates that). A scene-referred EXR of a
  raw, developed again, matches the direct develop within 1 code.
  Screenshots: the card folder lists 516 images with JPEG thumbnails,
  and a JPEG opens in the develop view with the reduced sidebar and a
  live histogram. (Correction 2026-09-11: this entry originally claimed
  portrait JPEGs were oriented; they were not, since achromat applies
  no EXIF orientation and `Source::orientation()` returned `Normal`
  for images. Fixed by the metadata arc below.)
- Cost: JPEG load 330 ms for 25 MP (decode plus conversion); the
  thumbnail sweep decodes each JPEG at full size, so a mixed folder
  fills slower than a raw-only one (embedded previews are ~30 ms).
- Dependencies achromat brings that are not pure Rust: dav1d for
  AVIF decode, and the vendored jpegxr C library (bindgen). rav1d is
  the replacement for the former when the time comes.
- Not done: raw + JPEG pairs from the same shot show as two entries;
  a stacking rule (same stem) is an obvious next step. Gallery
  retirement can proceed once the user has used this for a while.

## Metadata arc, chunk 1: readout (2026-09-11)

Survey: we carried eight fields per raw; an a6700 ARW holds 205 tags
under exiv2 (133 standard, 72 Sony makernote), and rawler's `Exif`
struct parses ~45 of the standard ones but skips the makernote.
Exports carry no metadata at all (chunk 2). Non-raw sources got no
metadata because achromat returns pixels only.

Built:

- `idev-core::metadata` — `Metadata { groups: [Group { name, fields:
  [Field { name, value }] }] }`, rendered to strings at load time.
  Groups: Camera, Exposure, Lens, Time, Image, GPS, Sony, Other.
  Known tags have a name and a domain formatter (exposure as `1/80 s`,
  APEX values as f-numbers and times, EXIF and Sony enums decoded, GPS
  folded to signed decimal degrees, Sony lens spec from its BCD bytes);
  every other IFD0 / Exif / makernote entry lands in Other as a short
  numeric dump unless it is structural (offsets, previews, tone
  tables) or enciphered (Sony 0x9xxx, Tag2010, ShotInfo). Sony tables
  follow ExifTool's Sony.pm.
- The raw path parses the TIFF once for lens tables and metadata (the
  GPS IFD has to be requested from rawler explicitly). The image path
  finds the EXIF block itself (JPEG APP1, PNG `eXIf`, WebP `EXIF`) and
  runs the same reader over it, and now also fills `ExifSummary` and
  the orientation for images, so `Source::exif()` is total and
  portrait JPEGs rotate in thumbnails and exports.
- CLI: `inspect --all` prints every group; `inspect` prints the camera
  summary for images too. GUI: `i` or the Info button swaps the
  sidebar for the metadata groups as name/value rows.

Verified (DSC00039.ARW and its JPEG, now under the library at
`2026/2026-09-10`; the camera_control download folder is gone):

- Field-for-field against exiv2 on the ARW: every standard tag exiv2
  lists is either shown or deliberately skipped (offsets, blank
  strings, ComponentsConfiguration and the like); all 72 Sony tags
  exiv2 decodes are shown with matching values, plus 24 unknown small
  ones in Other. Two formatter bugs found by that comparison
  (FocusFrameSize bytes read as shorts, APEX MaxApertureValue shown
  raw) were fixed.
- The JPEG path yields the identical group list to the ARW, down to
  the Sony makernote (one unknown tag differs in value, as expected).
- DSC00035.JPG (a portrait shot) now reports `Rotate270`.

Not done / caveats:

- Sony model ID table only knows the a6700; other bodies show the
  number. LensID shows "E-mount or other" for 65535; the EXIF lens
  name covers the rest.
- Enciphered Sony blocks (shutter count lives in 0x9050) are skipped;
  ExifTool's substitution cipher is documented and a candidate.
- AVIF, JPEG XL, JPEG XR, EXR and Radiance HDR sources get no
  metadata yet (no EXIF block extraction for those containers).
- The develop canvas draws raws and images in stored orientation;
  only thumbnails and exports rotate. Pre-existing; a candidate for
  the crop/straighten tool since both touch the same geometry.

## Metadata arc, chunk 2: metadata into exports (2026-09-11)

Built: `metadata::Tags` keeps the parsed IFD0 / Exif / GPS entries on
every `Source` (structural and bulky tags dropped at load), and
`metadata::exif_for_output` writes a TIFF block from them with
rawler's writer: IFD0 whitelist (make, model, description, artist,
copyright, resolution, modify date) plus `Orientation` = 1, `Software`
= "idev x.y.z", pointers to the Exif and GPS IFDs; the Exif IFD copied
whole except MakerNote, the source's colour space and dimensions, and
the JPEG-encoding tags, with the output's `PixelX/YDimension`,
`ColorSpace` (sRGB, or Uncalibrated for the PQ outputs where cICP is
the truth) and a `UserComment` (`metadata::develop_note`: source name
and the parameter TOML); GPS IFD copied whole. Containers: JPEG APP1
spliced after the JFIF APP0 (image's encoder has no EXIF hook), PNG
`eXIf` before IDAT, AVIF `Exif` item via avif-serialize's `set_exif`.
EXR gets nothing. Both CLI and GUI exports pass the block.

Verified on DSC00039.ARW developed five ways (JPEG, PNG, PQ PNG, AVIF,
PQ AVIF): exiv2 reads 59 tags from the JPEG and both PNGs with the
expected values (Software, orientation top-left, output dimensions,
sRGB vs Uncalibrated, exposure, lens, GPS, the user comment). exiv2
reads nothing from the AVIFs even with BMFF enabled, but libheif does:
`heif-info` lists the 1354-byte Exif item, `heif-convert` carries it
into a JPEG from which exiv2 reads the same 59 tags, and a direct iloc
parse shows the item as the 4-byte offset prefix plus the TIFF block.
The block round-trips through our own reader (`inspect --all` on the
output). A portrait raw (DSC00063) exports as 4128 × 6192 pixels with
orientation 1. The block is ~1.3 KB, far under the 64 KB APP1 limit.

Not done: EXIF in EXR (no standard slot; string attributes would be
the analogue), XMP, ICC profiles (the PNG has sRGB/cICP chunks and the
AVIF its colr box, which is what readers use), and reading EXIF back
out of AVIF/HEIF sources.

## Metadata arc, chunk 3: library sort, filter, stack (2026-09-11)

Built:

- `ExifSummary` is serde-serializable and rides in every thumbnail
  cache entry (format v3: a TOML blob between the header and the
  pixels). `sensor::embedded_preview` now fills it for raws; image
  previews take it from `Source`. The library keeps one per file as
  thumbnails arrive, so sorting and filtering need no second pass over
  the files and cost nothing on a warm cache.
- `library::{SortKey, Filter, Item, sort_indices, pairs}`: sort by
  name, date, ISO, focal length or lens (missing key sorts last, ties
  by name); a filter query of ANDed terms (`iso>=1600`, `f<2.8`,
  `mm>=50`, `date:2026-09-10`, `raw`/`jpg`, free text over name, lens,
  make and model); raw + rendered pairs by case-insensitive stem. Unit
  tests cover all three.
- The app keeps a `view: Vec<usize>` of file indices; the grid, the
  selection, stepping and Enter all go through it, so left/right in
  the develop view follow the sorted, filtered order. The toolbar has
  a text input (damascene `text_input`), sort tabs and a Stack button;
  the view is rebuilt on every change and, when it depends on
  metadata, after each batch of thumbnails, keeping the selected file
  selected. The runtime matches hotkeys before text editing, so the
  app's `hotkeys()` returns nothing while the filter box owns the
  selection; Escape or a click elsewhere releases it.
- Fix found on the way: `image_preview` never applied the EXIF
  orientation, so portrait JPEG thumbnails were still landscape after
  chunk 1 (only the developed-thumbnail path and exports rotated).
  Now it does; the cache version bump regenerates them.

Verified: unit tests for the filter grammar, the sort order and the
pairing; clippy and the full suite clean. No screenshot this time: the
user was working on the desktop and the test window did not land on
the captured outputs, so the toolbar layout is unreviewed visually.

Not done: burst / sequence grouping (Sony SequenceNumber lives in the
makernote, not in `ExifSummary`); persisting the sort/filter across
folders; a date range syntax beyond prefix matching.

## Processing arc, step 1: crop and straighten (2026-09-11)

Design: EXIF orientation, the straighten angle and the crop are one
affine map from output pixels to camera RGB pixels (`geometry.rs`),
applied in the colour shader before the lens sampling, so the image is
resampled exactly once and the CPU orientation step after the develop
is gone (CLI, GUI export, developed thumbnails). Frames: sensor →
oriented → view (turned about the centre, same size) → output (the
crop). `Geometry { angle_deg, crop: Option<[x, y, w, h]> }` in the
sidecar; `None` is the largest same-aspect rectangle inside the turned
content, `k = min(W / (W cos θ + H sin θ), H / (W sin θ + H cos θ))`.
Identity geometry maps each output pixel to exactly its own sensor
coordinate, so the uncorrected render stays bit-identical.

GUI: Geometry group with a Straighten scrubber, Crop toggle (`c`),
aspect tabs (free, 3:2, 4:3, 1:1, 16:9) and Reset. Crop mode renders
the full frame (crop forced to the whole view) with the colour pass
dimming everything outside the real crop three stops
(`Render::dim_outside`), and draws the outline as a stroked box in a
`stack` over the surface; plain drag on the image sets the rectangle
(anchor from `PointerDown`, `Drag` with the target rect for
fractions), the viewport pans on middle-drag meanwhile.

Verified: the geometry map agrees with `output::orient` for all eight
orientations (unit test over every pixel of a 5×3 image); a sample raw
with identity geometry produces a PNG with zero pixel difference to
the pre-change export, and the portrait raw's JPEG pixels are
identical to the CPU-oriented one; `--rotate 3` gives 5749×3832 with
no empty corners, `--crop 0.25,0.25,0.5,0.5` 3096×2064; screenshots
show the crop mode with the outline and the dimmed surround on a
rotated frame.

Found on the way: translucent fills composited over an app surface
lighten instead of darken in damascene (tried a token scrim, a plain
sRGB black with alpha and a linear black with alpha; an opaque fill
composites correctly). Filed on the damascene repo; the shader
dimming is the workaround.

Not done: dragging the edges/corners of an existing crop (a drag
always starts a new rectangle); a straighten guide (grid or a
two-point horizon line); the crop is not clamped to the turned
content, so a rectangle pulled past the edge shows black.

## Processing arc, step 2: resize on export (2026-09-11)

A GPU separable Lanczos-3 pass (`resample.wgsl`) on the developed
working image, in linear light, before readback and output encoding:
the kernel widens with the downscale factor so it is the
anti-aliasing filter; negative lobes are clamped out. Not a develop
parameter: the CLI's `--size N` and the GUI's Export long-edge
scrubber (0 = full). `Developer::develop_sized`; the preview keeps its
integer box-mean path.

Verified: 25 MP → 1000 × 667 in 40 ms of GPU work (the PNG encode is
the cost); against ffmpeg's Lanczos scale of the full-size PNG the
result is 48.3 dB PSNR with a max difference of 87 codes at edges,
the expected gap between filtering in linear light and in sRGB.
Upscaling is allowed (unit kernel).

## Processing arc, step 3: sharpening (2026-09-11)

Unsharp mask on luminance, defined at output resolution: two Gaussian
passes over the gamma-2.2-encoded BT.2020 luminance of the developed
image, then `l' = l + amount · detail` with a soft threshold
(`smoothstep(t, 2t, |detail|)`), applied as a luminance ratio to all
three channels so colour is untouched, clamped to the output peak.
Runs after the colour pass (and after the export resize), on the
preview with sigma divided by the downscale. `Sharpen { amount,
radius, threshold }` in the sidecar; `--sharpen a[,r[,t]]`; Detail
group in the GUI. `Developer::render` = colour + sharpen, used by the
preview and thumbnails; `develop_sized` sharpens after the resample.

Verified on a 2000 px export: amount 1 / radius 0.8 triples the
Laplacian variance (119 → 360) and touches 32 % of pixels by more than
one code; threshold 0.02 keeps the edge gain (var 192) while touching
1.4 % of pixels; amount 2 / radius 2 is visibly strong (var 907). A
crop of the cat's face shows crisp whiskers without dark or light
halos at these settings. GPU cost at 2000 px: within the 46 ms
develop time. Amount 0 takes the direct path and is bit-identical.

Cost note: with sharpening on, `render` allocates a scratch texture
per frame (25 MP × 8 bytes at 1:1); fine so far, a cached scratch in
the app's GpuState is the fix if scrubbing at 1:1 stutters.

## Processing arc, step 4: noise reduction (2026-09-11)

One GPU pass (`denoise.wgsl`) over the white-balanced camera RGB,
before the colour pass so the tone curve never amplifies what was
removed; cached per `Denoise` on the `Prepared` (a `RefCell` slot the
colour pass fills when the parameters change, so scrubbing exposure
does not re-denoise). Chroma is the difference from luminance (ratios
were tried first and blow up on near-black pixels): a 13 × 13 window
sampled every other pixel, spatial sigma 4, range kernel on the
chroma-difference vector. Luminance: a 7 × 7 bilateral, spatial sigma
1.5. Both range sigmas follow shot noise, `strength · k · sqrt(max(Y,
0) + 0.002)` (k = 0.1 chroma, 0.06 luminance); the `max` matters,
negative luminance after black subtraction made the sigma NaN and the
pixel garbage. `Denoise { chroma, luminance }` in the sidecar,
`--denoise c[,l]`, a Noise group in the GUI.

Verified on the ISO 3200 sample exported at 2000 px, in the flattest
mid-tone 48 px block, high-pass (sigma 2) standard deviations:

| denoise | luma HF | chroma HF | global Laplacian var |
|---|---|---|---|
| off | 1.78 | 5.88 | 109 |
| chroma 0.5 | 1.66 | 3.21 | 103 |
| chroma 1 | 1.74 | 2.16 | 107 |
| luminance 0.5 | 1.59 | 5.89 | 92 |
| luminance 1 | 1.51 | 5.89 | 78 |
| both 1 | 1.23 | 2.09 | 64 |

Chroma smoothing leaves luminance alone and vice versa; luminance 1
takes ~30 % off the global Laplacian variance, part noise and part
fine texture (fur softens a little at 1:1). A near-zero strength is
within one code of the untouched render. A 1:1 crop of the shadowed
pillow at chroma 1 / luminance 0.7 loses the grain and keeps the
print. Cost: within the develop time at 25 MP (denoise runs once per
parameter change, ~10 ms).

Not done: a noise profile per ISO (the k constants are guesses that
fit this sensor at ISO 3200), and a proper non-local or wavelet
luminance denoiser; the bilateral is a first version.

## Clipboard (2026-09-13)

Ctrl+C in the develop view opens a modal (damascene `modal` over the
page via `overlays`) with the encoding, PNG / JPEG / HDR PNG as tabs,
and a long-edge scrubber seeded from the Export group; the description
line shows the output size after crop and resize. Enter or Copy
develops on a worker with the same `develop_for` the file exports use
(headless device, current parameters including crop and straighten,
EXIF block) and encodes in memory (`encode_*` variants of the PNG and
JPEG writers in `idev-formats`; the file writers now wrap them). The
bytes go on the Wayland clipboard through `wl-clipboard-rs`
(data-control protocol, the same crate damascene's text copy uses):
one MIME type, the chosen one, served by a thread the crate keeps
until another client takes the selection. Hotkeys collapse to Enter
while the dialog is open; Esc, Cancel and the scrim close it.

Verified with a temporary hook that copied on load and pasted back
through the compositor: 13.3 MB `image/png`, 3096 × 2477 (the
sidecar's crop), EXIF present (Software, UserComment, dimensions),
pixels identical to the CLI's PNG export of the same file. Screenshot
of the dialog checked.

Not done: AVIF and EXR are not offered (few paste targets read them);
only the chosen type is offered, so a JPEG copy will not paste into an
application that reads `image/png` only; no `text/uri-list` for the
source file.

## Low-light review (2026-09-15): RCD input clamp

Green and red speckle in dark areas, worse under RCD than bilinear, and
4x4 saturated red blobs on DSC01036 under RCD only.

- **Cause, ours.** RCD's green step ratio-corrects each neighbour by
  `1 + (lc - lpf2) / (eps + lc + lpf2)`. The reference implementations
  clamp the input to non-negative first, which bounds the factor to
  [0, 2]; we fed the unclamped plane, where black subtraction leaves a
  third of shadow sites negative, so the denominator crosses zero. A
  CPU replica of the shader arithmetic on DSC01024 (ISO 10000) put
  3.1 % of R/B sites outside [0, 2], 58 k over 10x, worst 3·10^5; on
  DSC01036 (ISO 1000) a single red site 73 DN under black gave a
  factor of −14160, a green estimate of −250 k DN, and through the
  colour-difference RB step a red blob at every neighbour within two
  sites. Fixed by clamping the plane at zero in all three RCD loads
  (reference behaviour; the shadow bias is ~0.4 sigma, under a code).
  Strong dark-area outliers (>80 codes over the 5x5 median) on
  DSC01024 went from R 6464 / G 3293 / B 3532 to R 2118 / G 35 / B 88;
  bilinear sits at R 2149 / G 157 / B 75. DSC01036's blobs are gone.
- **Cause, the sensor.** The red outliers that remain in every path are
  single red CFA sites 400–800 DN above their same-colour median (noise
  core ~28 DN at ISO 10000), at positions that do not repeat between
  frames; the plane has a matching low tail (2088 sites at exactly 0
  inside the crop). Heavy-tailed read noise, red first because of the
  2.9x red multiplier. A genuine hot-pixel population exists but is
  small (~1000 sites, 2–3x chance overlap between frames). Remedy, not
  built: an impulse filter on the linearized plane before demosaic
  (same-colour 3x3 median replacement past k sigma).
- **Checked and fine.** rawler's lossless decode is bit-identical to
  libraw over all 30.7 M sites (the lossless raster is 6656 x 4608,
  zero-padded outside the crop); black 512; 14-bit data in these
  frames (12-bit applies to continuous drive and electronic shutter);
  the colour pass is linear in negatives and the sigmoid clamps. The
  linearize scale uses the profile white 15360 while lossless files
  declare 16383, so clip detection is 7 % early on them; harmless as
  long as both use the same number.
- **Not applicable now.** Sony's lossy Compressed RAW (files before
  mid 2026-09-10) quantizes 14 of every 16 same-colour pixels to steps
  of up to 64 DN when a bright pixel shares the block; the camera has
  been on Lossless Compressed since.

## Impulse filter (2026-09-15)

`impulse.wgsl`, one pass on the linearized plane between linearize and
demosaic, on when `Denoise.impulse` > 0 (default 0; `--denoise
c,l,i`; an Impulse scrubber in the Noise group). It is part of
`prepare`, so `Prepared::matches` compares it. Per site: the 24
same-colour neighbours of the 5x5 window, the two largest and two
smallest dropped, mean and sigma over the remaining 20; the site is
replaced by that mean when it is more than `k` sigma away, `k = 12 −
8 · strength` (0.5 → 8 sigma, 1 → 4). Sigma is floored at one 14-bit
code so flat regions cannot fire. No median, so no sorting network;
the trim handles a second bad site in the window.

Verified (RCD after the clamp, strong dark-area outliers over 80 codes
above the 5x5 median):

| DSC01024, ISO 10000 | R | G | B |
|---|---|---|---|
| impulse 0 | 2118 | 35 | 88 |
| impulse 0.5 | 207 | 23 | 8 |
| impulse 1.0 | 64 | 28 | 10 |

Footprint on DSC00744 (ISO 200, no push): 0.5 changes 0.007 % of
output pixels by more than 2 codes, 1.0 changes 1.07 % (4 sigma on a
20-sample estimate fires on ordinary noise and fine texture, which is
why 1.0 is the eager end). DSC01036 at +4 EV: 0.5 touches 0.75 %.
Cost: within the develop time (~125 ms at 25 MP including the pass).
Default stays 0 because every existing sidecar deserialises the
default; turning it on by default would change their renders.

## Burst arc (2026-09-15)

Survey of the night sets (2026-09-15, 239 frames, a6700, Lossless
Compressed, ISO 320 throughout, 20 continuous-drive bursts of 4–16
frames at 1/20–1.6 s): median signal 5–10 DN above black on a 14848 DN
range, 1–2 % of each frame above 30 DN. Phase correlation on the
log green plane: consecutive frames move 6–15 px with non-integer
fractions (stabilisation on), drift across a 13-frame burst reaches
74 px, rotation ~0.04° (1 px across 1500 px). Unmasked correlation on
dark tiles locks onto fixed-pattern noise and reports exactly 0.0 —
the aligner must vote only where there is signal. Gain: Photons to
Photos shows the a6700 ISO-invariant from 100 to 4000 with one small
step at 5000; the "second base at ISO 320" folklore is not in the data.

Reference pipeline: HDR+ (Hasinoff 2016) and handheld multi-frame
super-resolution (Wronski 2019): merge on the Bayer plane before
demosaic, coarse-to-fine alignment, robust merge weighted by a
shot+read noise model. Staged here:

1. Burst grouping from metadata.
2. Global signal-masked alignment, even-pixel quantised, robust
   Bayer-plane mean → one `SensorFrame`, developed as usual. (This
   section.)
3. Sub-pixel alignment and kernel-regression reconstruction replacing
   demosaic (1x), then 2x. Not started.

### Step 2 design (target, written before implementation)

- `sensor::ExifSummary` gains `sequence: Option<u32>` (Sony makernote
  0xb04a, `None` for single-shot) and `subsec_ms: Option<u32>`
  (`SubSecTimeOriginal`), so the thumbnail cache can group without
  reopening files (cache format 3 → 4). `Metadata::from_tiff` exposes
  the sequence number; `embedded_preview` parses the TIFF structure
  for it (a few ms beside the JPEG decode).
- `burst::group(&[Option<&ExifSummary>]) -> Vec<Vec<usize>>`: sort by
  timestamp, continue a burst while the sequence number increments,
  the gap is under 2 s plus the exposure time, and shutter, ISO,
  aperture and focal length are unchanged. Frames without a sequence
  number are singleton groups.
- `burst::align(reference, frame) -> Shift {dx, dy, peak}` in sensor
  pixels, `frame(x + dx, y + dy) ≈ reference(x, y)`: quad-mean gray of
  the crop (half resolution), black-subtracted, log1p; mask = 7×7 box
  blur of the linear gray above 6× the quad noise (from the below-black
  tail), fail if under 0.1 % of quads pass; phase correlation (rustfft)
  of the masked, mean-removed log gray, parabolic sub-pixel peak.
  Global translation only: the measured rotation is under one merge
  grid step at the frame edge, and with 2 of 12 tiles holding content
  a local field would be undefined almost everywhere.
- `burst::noise_model(a, b, shift) -> σ²(s) = a·s + b` (DN above
  black) from the difference of two aligned planes: bin by the pair
  mean (independent of the difference for equal-variance noise), MAD
  variance per bin, weighted least squares over bins.
- `burst::merge(frames, shifts) -> Merged {frame, noise, weights}`:
  shifts rounded to even sensor pixels (CFA phase preserved); per
  pixel the median of the aligned values anchors a Gaussian robustness
  weight `exp(-½ (d / 3σ_d)²)`, σ_d² = 2σ²(median), applied to every
  frame including the reference (so the reference's own hot pixels
  are outvoted); out-of-bounds samples get weight 0. Output is the
  reference frame's `SensorFrame` with the merged plane. CPU, row
  bands over `std::thread::scope`: this is a load-time step like
  unpack, not an interactive stage.
- CLI: `idev burst FILES|DIR` lists groups; `idev burst --stats
  FILES` aligns and merges the given frames as one burst and prints
  shifts, weights, the noise model, the below-black σ before and
  after, and a half-split check (evens vs odds merged separately,
  model between them; expected σ ratio √(N/2)). `--pgm` dumps the
  merged plane. `idev develop --merge FILES -o OUT` merges and
  develops with the first frame's sidecar.
- GUI: the library's Stack toggle also collapses a burst onto its
  first frame with an "×N" badge. Merging in the GUI waits for the
  merge to be worth caching (step 3).

### Step 2 results (2026-09-15)

Built as designed: `burst.rs` (grouping, `align`, `noise_model`,
`merge`, parallel loaders), `ExifSummary::{subsec_ms, sequence}` with
the thumbnail cache at format 4, `idev burst` / `idev burst --stats`
/ `idev develop --merge`, and the library's Stack toggle collapsing
bursts with an "×N" caption. rustfft is the one new dependency; the
merge and the FFT rows run over `std::thread::scope` bands.

- Grouping: 239 frames of the first session and ~300 of the second
  fall into the same bursts the hand-made table found; `idev burst`
  on a folder takes ~270 ms per file (rawler metadata plus the
  makernote parse, on ceph).
- Alignment on the real bursts: shifts drift smoothly (4 → 35 px over
  a 7-frame 0.6 s set; 4 → 43 px over a 6-frame 1/8 s set), fractions
  non-integer, correlation peaks 0.014–0.085. Mean robustness weight
  0.966–0.973 for every frame: nothing is being rejected wholesale.
- Timing, 7 frames: load 1.2 s, align 0.8 s, merge 0.3 s.
- Noise, 7-frame 0.6 s set (DSC01126–1132), half-split check (evens
  vs odds merged separately, model between the halves, expected
  ratio √(N/2) = 1.87): σ(100 DN) 12.3 → 6.7 (1.83), σ(1000) 38.7 →
  21.2 (1.83). 11-frame set (DSC01146–1156, expected 2.35): 2.42 at
  100 and 1000 DN. 6-frame 1/8 s set (DSC01233–1238, expected 1.73):
  1.98 / 2.03; below-black σ (2.6 % of sites below black, so
  meaningful there) 3.49 → 1.51 against √6 = 2.45.
- Single-frame model at ISO 320: σ² ≈ 1.3–1.5·s + 2.1–2.5 DN². The
  pair method cancels fixed-pattern noise, so its read-noise floor
  (~1.5 DN) sits below Photons to Photos' single-frame 5.9 DN, which
  includes the pattern; a hand-held stack dithers the pattern
  against the scene, which is why the below-black σ falls with N.
  Lens-cap frames will separate the two terms.
- Where the numbers deviate: the single-pair `a` on sharp sets is
  inflated by residual sub-pixel misalignment at point lights (the
  ratio then exceeds √(N/2)); at zero signal the merged planes are
  quantised to u16, so the half-split read-noise term is not
  resolvable below ~2 DN. Step 3 should keep the merged plane in
  float.
- The 0.6 s 17 mm sets are soft (defocus or intra-frame motion): the
  merge cleans the sky and water and leaves the lights as they were,
  no ghosting. They are noise tests, not resolution tests.
- Not done: developed-image comparison beyond a visual check;
  merging in the GUI; a GUI screenshot of the "×N" caption (the
  build and tests pass, the layout is unreviewed).

### Lens-cap series (2026-09-15, DSC01616–1643): the gain switch is real

1/200 s with the cap on, ISO 100–32000, pairs up to ISO 800. One
green site over the crop, std of the core (±6 MAD, so the integer
quantisation of the MAD does not bite), temporal noise from the pair
difference / √2:

| ISO | σ total DN | σ/ISO ×100 | σ temporal DN | FPN DN |
|---|---|---|---|---|
| 100 | 1.45 | 1.45 | 1.42 | 0.3 |
| 200 | 2.57 | 1.29 | 2.49 | 0.6 |
| 400 | 2.07 | 0.52 | 2.16 | 0.0 |
| 500 | 2.72 | 0.54 | 2.62 | 0.7 |
| 640 | 3.29 | 0.51 | 3.36 | 0.0 |
| 800 | 4.23 | 0.53 | 4.25 | 0.0 |
| 1600 | 8.11 | 0.51 | | |
| 3200 | 15.4 | 0.48 | | |
| 6400 | 29.5 | 0.46 | | |
| 12800 | 52.0 | 0.41 | | |
| 16000 | 56.5 | 0.35 | | |
| 32000 | 105 | 0.33 | | |

Input-referred read noise drops 2.5× between ISO 200 and 400, so
the a6700's second conversion gain engages between them (the series
skipped 250 and 320; the forum figure of 320 is consistent). Above
that it declines slowly to 12800, then a second smaller step at
16000 coincides with hot pixels vanishing (0.1 % → 0.002 % of sites
beyond 10σ), i.e. in-camera raw noise reduction. Fixed pattern is
under 1 DN at every ISO. This contradicts the Photons to Photos
curve read earlier in the day (smooth through 320) and confirms the
user's recollection; the burst pair model's ~1.5 DN at ISO 320 on
the night frames matches the interpolated lens-cap value. Shooting
consequence: ISO 320 sits exactly at the switch, which is the right
choice for these bursts; ISO 100–200 would cost 2.5× in read noise.

### Step 3 design (target, written before implementation)

Wronski 2019 in its simplest workable form, on the CPU first: the
method's parameters (kernel width, robustness) are unsettled, and a
CPU pass over 26 MP × N frames is seconds, which is enough to measure
whether resolution is recovered on these sets. The GPU port follows
once the parameters are settled (decision 2 stands for the interactive
pipeline; this is a load-time step like unpack until then).

- **Alignment model: a similarity per frame** (`burst::Transform`,
  reference sensor coords → frame sensor coords, `p' = A·p + t`).
  The measured motion is a translation plus ~0.04° of roll; a
  similarity is the physical model and, unlike a full affine, is
  constrained by content that lies along the horizon (rotation and
  scale from the horizontal extent). Fitted from tile
  phase-correlations: 256-quad (512 px) tiles of the masked log gray,
  each pre-shifted by the frame's global even shift, tiles with under
  2 % coverage skipped, peak as the weight, weighted least squares in
  (a, b, tx, ty), three rounds dropping tiles beyond max(1 px, 3×
  median residual). Under three tiles: translation only from the
  tile mean; none: the global shift. Reports tiles used and residual
  RMS.
- **Reconstruction** (`burst::reconstruct(frames, transforms, noise,
  scale, kernel_sigma) -> CameraRgb`): output grid is the reference
  crop at `scale`×, pixel centres mapped to reference sensor
  coordinates, then through each frame's transform. Each frame
  contributes the 3×3 sites around the mapped point, weighted by an
  isotropic Gaussian in sensor pixels (`kernel_sigma`, default 0.7)
  times a per-frame robustness weight; sums per CFA colour give the
  output RGB, black-subtracted and normalised to white, before white
  balance. Isotropic first; the structure-tensor kernel of the paper
  is a later refinement if edges show it is needed.
- **Robustness** from the half-resolution quad-gray planes: the
  frame's gray at the mapped point (bilinear) against the
  reference's, normalised by √2 × the quad noise from the model plus
  a misalignment term (|∇gray| × 0.25 px), Gaussian weight at 3σ as
  in step 2. The reference has weight 1.
- **Into the pipeline**: `Source::Rgb(CameraRgb)` carries the
  reference `SensorFrame` (metadata, plane dropped) and an RGBA f16
  plane; `Developer::prepare` uploads it as the camera-RGB texture
  after applying the white-balance multipliers, sharing the colour
  setup with the raw path. Highlights, lens tables (normalised
  radius, so scale-free), geometry and everything downstream are
  unchanged. 2× of the a6700 crop is 12384 × 8256, inside the 16384
  texture limit the headless device already requests.
- **CLI**: `idev develop --superres SCALE [--kernel SIGMA] FILES -o
  OUT`; `idev burst --stats` also prints each frame's similarity fit.
- **Evaluation**: on a sharp 28 mm 1/8 s skyline set, the width of
  isolated point lights in (a) the single frame's RCD develop
  upscaled 2× against (b) the 2× reconstruction, and at 1× the
  reconstruction against RCD of the step-2 merge; timing; memory.

### Step 3 results (2026-09-15)

Built as designed, with one addition the tests forced: the parabolic
sub-pixel peak of the phase correlation was 0.6 px off on synthetic
data, so `align_similarity` finishes with a Gauss-Newton refinement
of the four similarity parameters on the half-resolution log planes
over the reference's signal mask (Cauchy reweighting from the third
iteration). Synthetic checks: a (5.3, −3.7) px shift recovered within
0.1 px, a 0.3° roll within 0.03°. `burst::reconstruct`,
`Source::Rgb(CameraRgb)` with `Developer::prepare_rgb` (the colour
setup is now shared with the raw path), `idev develop --superres
SCALE [--kernel σ]`, `idev burst --stats` prints the fits.

- Fits on real bursts: 9–16 tiles per frame, rolls of −0.18° to
  +0.32° between frames, scale within 0.3 %; robustness weights
  0.94–0.96; no uncovered output values at 1× or 2×.
- Timing, 6 frames, 24 threads, debug profile with idev-core at
  opt-level 2: 1× 10.5 s wall (load + align + reconstruct 5 s,
  develop 0.5 s, PNG 5 s); 2× 27 s (8 s, 2 s, 16 s). The 2× plane is
  816 MB f16, doubled once for the white-balance upload.
- Resolution evidence, 28 mm 1/8 s skyline (DSC01233–1238), energy
  of the horizontal spectrum above the single-frame Nyquist
  (0.25–0.5 cycles per 2× pixel) in a horizon band of content versus
  a sky band of noise, on the developed 8-bit output:

  | version | horizon high band | sky high band |
  |---|---|---|
  | single RCD, Lanczos 2× | 1.08e9 | 2.9e8 |
  | 1-frame reconstruction, σ 0.7 | 3.75e9 | 3.8e8 |
  | 6-frame reconstruction, σ 0.7 | 5.43e9 | 0.83e8 |
  | 6-frame reconstruction, σ 0.5 | 9.51e9 | 1.9e8 |

  Against the 1-frame reconstruction (same estimator, same
  nonlinearity) the 6-frame one has 1.45× the high-band energy where
  there is content and 4.6× less where there is only noise, so the
  gain is content. Visually the pier and lamp posts are finer and
  cleaner at 2×; σ 0.5 is crisper with more grain. Point-light
  widths are identical in every version (σ ≈ 2.2 sensor px): the
  lights are optically limited and cannot show resolution.
- **Correction (2026-09-16, red-fringe investigation): the
  resolution evidence above is invalid.** The refined transforms are
  systematically 3–6 px off on the real sets (measured at 60 lights
  across the frame, the same offset in every frame); the reconstruction
  was the reference frame plus partially rejected ghosts of the other
  frames, and ghost edges are what raised the high band. Cause: the
  Gauss-Newton gradient in `refine_similarity::sample` is the central
  difference at the integer floor of the sample position, not at the
  position itself; on point-like content that biases every step the
  same way, the loop never converges, and twelve iterations of ~0.2
  quads each accumulate the offset (the coarse tile fit was right to
  0.6 px). Verified by replicating the iteration in Python: the
  floor-based gradient reproduces the Rust steps exactly, a gradient
  centred on the sample point converges within 0.3 px of the measured
  light positions. Fixed in the entry below.
- Not evidence: the 37 mm 1/60 s set (DSC01468–1475) is too dark;
  there the 1-frame reconstruction has more high-band energy than
  the 8-frame one, i.e. the band is noise. A hand-held daytime burst
  of a detailed static subject is the real resolution test and has
  not been shot.
- Seen and not fixed: at a saturated light the 2× reconstruction
  shows a red fringe the RCD path does not. Candidates: lateral CA
  that colour-difference demosaicing hides (the lens CA tables still
  apply downstream), or per-colour kernels straddling the clip.
  Needs a look at the camera-RGB plane before the colour pass.
- Not done: the GPU port; the structure-tensor kernel; merging or
  reconstruction from the GUI; the merged plane in float (step 2).

### Refinement fix and the corner scorer (2026-09-16)

- **Fix.** `sample_plane` now takes its gradient by central differences
  of bilinear samples one pixel either side of the point; the
  refinement uses it (the reconstruction's robustness term did too and
  is unaffected in practice). The loop keeps the best state once the
  reweighting has settled and stops when the residual rises. A new
  test with point-like content at fractional positions reproduced the
  drift on the old code (0.2 px on a 1024-px frame) and passes now.
  On the night pair DSC01233/1234 the refined similarity lands within
  0.3 px of the measured light positions (rotation −0.12°, independent
  tile estimate −0.11°), where it was 5 px off.
- **Scorer** (`corners.rs`, `idev burst --score`): X-corners on the
  half-resolution plane by the saddle response of a σ = 2 quad blur,
  refined by gradient orthogonality over a 6-quad disc centred on the
  running estimate (the disc must exceed the blur width: at 4 quads
  the estimate stopped 40 % of the way to the corner; a quadratic
  saddle fit was tried first and biased 0.15 px toward the window
  centre). Synthetic board with corners at every sub-quad phase: 0.08
  px median, 0.13 px worst, no bias; pairwise 0.17 px. Corners are
  matched as mutual nearest neighbours through the global shift.
  Reported models: the global shift, coarse and refined similarity,
  an in-sample homography (the best any global model can do on a
  plane) and a local bound (refined plus per-cell translation over
  6×4 cells). The strongest 120 reference corners are scored
  separately because the ChArUco markers add weaker, worse-located
  corners.
- **Results, board bursts (1/200 s, ISO 320, 18 mm).** Second burst
  (DSC01654–1670, shifts 1–8 px), strongest corners, rms px:
  translation 0.5–1.4, coarse similarity 0.6–0.8, refined similarity
  0.32–0.47, homography 0.29–0.39, local bound 0.27–0.37. First burst
  (DSC01644–1653, shifts growing to 56 px): refined similarity
  0.34 → 2.1 px with the shift, coarse about the same, homography
  0.25–0.41, local bound 0.21–0.93.
- **What that says.** The refinement now improves on the coarse fit
  where the model fits at all. A global similarity is model-limited
  on a tilted plane once the camera has moved a few tens of pixels:
  the homography is up to eight times better. The residual floor on
  real frames is about 0.3 px for every model, above the 0.13 px
  synthetic floor; the likely cause is the markers intruding into
  the 12-px refinement window, not tested. Lens distortion (4 % at
  the corner at 18 mm) also makes even rigid motion non-projective on
  the raw plane. Per-cell translation over 1000-px cells is not
  enough at 56 px of motion (0.9 px), so the local field must be
  finer than that.
- **Next**: a per-tile alignment field (512-px tiles, refined by the
  same Gauss-Newton per tile, initialised from the global fit), then
  the reconstruction re-measured on the board and the room set.

### Local alignment field (2026-09-16)

`burst::Field`: the refined global similarity plus a translation
correction per 512-px tile, fitted by the same Gauss-Newton on the log
planes (translation only, Cauchy reweighting, best-state guard), kept
only where it lowered the tile's residual and stayed under 12 px,
interpolated bilinearly between tile centres and extrapolated linearly
past the outer ones. Tiles with under 150 masked quads in either frame
keep the global fit, which is what the night skylines get outside the
horizon. The reconstruction now takes fields; `idev burst --score`
reports the field as a model.

Kill test, board bursts, strongest corners, rms px:

| model | second burst, 1–8 px motion | first burst, 14–93 px motion |
|---|---|---|
| refined similarity | 0.32–0.47 | 0.34–2.08 |
| homography (in-sample) | 0.24–0.39 | 0.25–0.41 |
| local field | 0.22–0.38 | 0.21–0.29 |
| local bound (per-cell translation) | 0.23–0.37 | 0.21–0.93 |

The field reaches the corner-location floor on both bursts and beats
the in-sample homography on the first, i.e. it absorbs the lens
distortion a projective model cannot. Passed. Synthetic check: an
anisotropic warp (0.4 % along x, −0.3 % along y) that a similarity
misses by 1.7 px is recovered to 0.3 px inside the tile hull and
0.5 px at the extrapolated corners.

Reconstruction on the board (DSC01654–1663, 2×, kernel 0.5 px,
developed with +2 EV), over the board's centre:

| version | edge width 10–90 %, 2× px | flat-square noise, 8-bit codes |
|---|---|---|
| single RCD, Lanczos 2× | 6.70 | 2.3 |
| 1-frame reconstruction | 7.10 | 2.2 |
| 10-frame reconstruction, field | 7.05 | 1.2–1.4 |

Noise halves (the 8-bit output floors the measured ratio below √10)
and the 10-frame edge is clean where the single-frame versions show
colour speckle and a faint fringe; edge width does not move. The
single frame's native edge is 3.5 px 10–90 %: at 18 mm f/2.8 the lens
delivers no energy near Nyquist, so there is nothing aliased for
multi-frame sampling to recover, on this set or the room set shot
the same way. The kernel costs ~0.4 px (2×) of width against RCD;
0.35 px would narrow it at the price of grain.

What a resolution test needs: a lens stopped to its sharpest
(f/5.6–8), a subject with detail at or beyond Nyquist (distant text,
fabric, a printed chart at a few metres) so the single frame's edges
are 1–1.5 px, hand-held for the dither. The pipeline is ready to
measure it: `--score` for the alignment, edge widths and flat noise
on the develop.

Carried notes: the corner scorer's 0.3 px real-data floor (markers in
the window, untested); the robustness allowance scales with the
gradient and let 5 px ghosts through before the fix, now moot with a
correct field but still generous.

### Resolution test attempts (2026-09-16)

Three more sets, none a resolution test yet, each for a different
reason; and one instrument that finally gives a yes/no answer.

- **Room at f/8 and f/16, 1/160 s, ISO 320 (DSC01687–1722).** f/16:
  green median 4 DN, pure noise. f/8: median 14 DN, a quarter of the
  frame above 100 DN, single-frame edges 2.5–4.5 px. Stopping down
  removed the light; the room holds no detail near Nyquist. The field
  corrected ~105 of 117 tiles with 1–2 px mean and up to 10 px
  (parallax between shelves and wall), so the field is earning its
  place in rooms. The step-2 merge through RCD denoised better than
  the 2× kernel regression (0.9 vs 1.3 codes): with no resolution to
  gain, the merge is the better tool. The f/8 frame's sharpest edges
  (2.5 px) beat f/2.8's (3.2 px): the lens does sharpen stopped down.
- **Mesh chair, 17 frames, 1/100 s, f/5, ISO 3200 (DSC01737–1753).**
  Exposure right at last (green median 257 DN, p90 2440, no clipping)
  but the black mesh sits at 16/255 in the develop. Weave pitch
  5.9–6.2 px. The 17-frame reconstruction at kernel 0.35 shows a
  fine dot lattice the single frames and the 17-frame merge do not,
  and its spectrum holds a forest of peaks above 0.16 cycles per 2×
  px at ~12× local background. That looked like super-resolution.
  It is not: **the half-split coherence test** (odd and even frames
  reconstructed separately onto the same reference, normalised
  cross-spectrum per band, CFA-locked frequencies excluded) gives
  0.94 in the weave band and 0.10–0.16 above 0.2 cycles per 2× px,
  against 0.10 on a flat wall. The high band is not reproducible
  between halves, i.e. noise plus sampling artefacts: texture
  multiplied by the uneven union of 17 shifted CFA lattices produces
  sidebands that a narrow kernel passes, and the per-pixel
  normalisation removes the gain modulation but not the bandwidth
  modulation. A single-frame kernel regression also carries a
  texture-locked peak at the CFA frequency (82× background in the
  mesh, 1× on the wall), 3× weaker with 17 frames.
- **What this means.** (1) A narrow-kernel multi-frame reconstruction
  can manufacture convincing fine texture; no visual or single-image
  spectral metric catches it, the split does. The coherence test
  should gate any resolution claim and belongs in the CLI. (2) At low
  SNR the reconstruction's high band is untrustworthy; the robust
  merge plus RCD is the safer default, and kernel width should widen
  with noise or with sample sparsity (Wronski's adaptive kernel is
  about this, not only about edges). (3) The shoot still owed:
  daylight, f/5.6, ISO ≤ 800, a textured subject at a quarter of
  white or brighter, single-frame edges 1–1.5 px; then the split
  coherence above 0.25 cycles per 2× px is the pass/fail.
- Oddity noted, not chased: the two merged halves cohere at only 0.38
  in the weave band where the reconstructions reach 0.94.

## Milestones

1. **Pipeline proof (CLI only).** Workspace, rawler unpack of an a6700
   ARW, headless wgpu: linearize → WB → bilinear → matrix → EXR out.
   Cross-check the Bayer plane and matrices against libraw's
   `unprocessed_raw`/`dcraw_emu -4`. Look at the EXR in damascene-gallery.
2. **Real development.** RCD demosaic, highlight reconstruction, exposure
   and tone curve, PQ PNG + JPEG export, sidecars, batch over a
   directory. Compare against the camera JPEG and RawTherapee's default.
3. **GUI.** Inlay damascene on the pipeline's device; develop view with
   viewport + sliders + histogram; sidecar round-trip; library grid with
   the explorer-thumbs cache; HDR/SDR badge and `t` preview toggle.
4. **Polish.** Lens corrections from Sony tags, AVIF/JXL, soft-proof
   against prism's LUT, decide the fate of damascene-gallery.

## Decisions (ratified 2026-09-10)

1. rawler, not libraw FFI. libraw stays an external cross-check.
2. GPU-only pipeline (wgpu compute). Rejected CPU/hybrid: two
   implementations to keep identical, and 26 MP sliders need the GPU.
3. Working space: linear BT.2020, D65, f16. Rejected scRGB-linear:
   negative values complicate every stage in between.
4. Name: open. Crates use the placeholder prefix `idev` until chosen;
   the playground directory is a placeholder too. Rename is mechanical.
5. damascene-gallery will be retired once the GUI covers browsing; this
   project is meant to grow into the user's image processing system.
