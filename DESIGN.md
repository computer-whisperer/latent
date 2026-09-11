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
