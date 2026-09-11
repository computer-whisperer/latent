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
