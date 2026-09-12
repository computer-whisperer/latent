# image_developer (working name)

Raw developer for the prism/damascene HDR ecosystem: Sony ARW in, HDR
and SDR images out, one GPU pipeline behind both a CLI and (later) a
damascene GUI. `DESIGN.md` is the plan and the decision record.

```
idev inspect DSC00039.ARW                                 # raws and rendered images alike
idev inspect DSC00039.ARW --all                           # plus every EXIF, GPS and Sony makernote field
idev develop DSC00039.ARW -o out.jpg                      # sRGB JPEG, sidecar params if present
idev develop DSC00039.ARW -o out.png --hdr 1000           # BT.2020 PQ 16-bit PNG for a 1000-nit master
idev develop DSC00039.ARW -o out.avif --hdr 1000          # BT.2020 PQ 10-bit AVIF (pure-Rust rav1e; slow)
idev develop DSC00039.ARW -o out.exr --tone none          # scene-referred scRGB linear f16
idev develop DSC00039.ARW -o out.jpg --exposure 0.5 --temp 5200,4 --contrast 1.8 --white 2.9 --save-params
idev develop DSC00039.ARW -o out.jpg --rotate 1.5 --crop 0.1,0.1,0.8,0.6   # straighten (deg, cw) and crop (fractions)
idev develop DSC00039.ARW -o out.jpg --size 3000                         # long edge 3000 px (Lanczos-3, linear light)
idev develop *.ARW --out-dir developed --format jpg       # batch; skips existing outputs
idev params DSC00039.ARW                                  # resolved params (sidecar + defaults)
idev dump-bayer DSC00039.ARW -o raw.pgm                   # decoder cross-checks
```

Build with `cargo build --release`; the binary is `target/release/idev`.
Parameters live in `<raw>.idev.toml` beside each raw; flags override,
`--save-params` writes back. Every JPEG, PNG and AVIF output carries an
EXIF block: the source's camera, exposure, lens, time and GPS tags,
orientation reset (pixels are rotated), the output size and colour
space, `Software` = idev, and a `UserComment` holding the source name
and the parameters as TOML, so a render is reproducible from the
output alone. EXR has no EXIF slot and carries none. The camera's lens correction tables
(distortion, lateral CA, vignetting; Sony SubIFD tags) are applied by
default; `--lens none` disables them and `inspect` reports them.

GUI: `cargo run --release -p idev-gui -- DIR` opens a folder of images
(camera raws, plus PNG, JPEG, WebP, AVIF, JPEG XL, JPEG XR, EXR and
Radiance HDR through achromat) as a thumbnail grid (`idev-gui FILE.ARW` opens straight into that file;
no argument means the current directory). Thumbnails come from the
camera's embedded JPEG until a raw has a sidecar, then from a scaled
develop with the sidecar's parameters (marked with a dot); they are
cached under `~/.cache/idev/thumbs`. The toolbar has a filter box
(terms are ANDed: `iso>=1600`, `f<2.8`, `mm>=50`, `date:2026-09-10`,
`raw` / `jpg`, or any word matched against the file name, lens and
camera), sort tabs (name, date, ISO, focal length, lens; files without
the key sort last) and a Stack toggle that hides JPEGs which have a raw
of the same name (the raw's tile says `+JPG`). Metadata for sorting and
filtering rides along in the thumbnail cache, so a metadata-ordered
view settles as thumbnails arrive. Enter or a double click opens the
develop view (damascene, HDR-aware on prism): pan/zoom preview, a
per-channel histogram of the preview, scrubbers for exposure, contrast,
hue preservation, white, temperature and tint, tabs for tone curve,
highlights and demosaic, switches for the lens corrections, export
buttons; the sidecar is written
(debounced) on every change. Keys: arrows or `hjkl` move (left/right
also step files in the develop view), Enter open, Esc back to the grid,
`o` open another folder, `f` fit, `1` 1:1, `t` SDR preview, `w` zebra
warning for pixels at the panel's peak (red) or outside its gamut
(cyan), `i` metadata in the sidebar (every EXIF, GPS and Sony makernote
field the file carries), `c` crop mode, `e` export JPEG. The Geometry
group has a Straighten scrubber (degrees, positive clockwise; the crop
defaults to the largest same-aspect rectangle inside the turned
content) and Crop: in crop mode the whole straightened frame shows
with the outside of the crop dimmed, a plain drag draws the rectangle
(middle-drag pans), the aspect tabs lock its ratio, Reset clears both.
The Export group's long-edge scrubber resizes exports (0 = full size).

Crates:

- `idev-core` — rawler unpack into a `SensorFrame`, achromat decode into
  the working space (`Source`), metadata readout (EXIF, GPS, Sony
  makernote; also from the EXIF block of JPEG, PNG and WebP files),
  color matrices, `DevelopParams`, the wgpu compute pipeline, CPU
  output transforms.
- `idev-formats` — encoders (EXR, PNG, JPEG, AVIF, PGM).
- `idev-cli` — the `idev` binary.
- `idev-gui` — the `idev-gui` window: library grid, thumbnail cache,
  develop view.

`idev` is a placeholder prefix until the project is named.
