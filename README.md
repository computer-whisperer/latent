# latent

Raw developer for the prism/damascene HDR ecosystem: camera raws (built
and tested on Sony ARW) and rendered images in, HDR and SDR images out,
one wgpu compute pipeline behind both a CLI and a damascene GUI. Named
for the latent image: captured, but invisible until developed.
`DESIGN.md` is the design and decision record.

```
latent inspect DSC00039.ARW                                 # raws and rendered images alike
latent inspect DSC00039.ARW --all                           # plus every EXIF, GPS and Sony makernote field
latent develop DSC00039.ARW -o out.jpg                      # sRGB JPEG, sidecar params if present
latent develop DSC00039.ARW -o out.png --hdr 1000           # BT.2020 PQ 16-bit PNG for a 1000-nit master
latent develop DSC00039.ARW -o out.avif --hdr 1000          # BT.2020 PQ 10-bit AVIF (pure-Rust rav1e; slow)
latent develop DSC00039.ARW -o out.exr --tone none          # scene-referred scRGB linear f16
latent develop DSC00039.ARW -o out.jpg --exposure 0.5 --temp 5200,4 --contrast 1.8 --white 2.9 --save-params
latent develop DSC00039.ARW -o out.jpg --rotate 1.5 --crop 0.1,0.1,0.8,0.6   # straighten (deg, cw) and crop (fractions)
latent develop DSC00039.ARW -o out.jpg --size 3000                         # long edge 3000 px (Lanczos-3, linear light)
latent develop DSC00039.ARW -o out.jpg --sharpen 1,0.8,0.01                # unsharp mask: amount, radius px, threshold
latent develop DSC00039.ARW -o out.jpg --denoise 1,0.5                     # noise reduction: chroma, luminance (0..1)
latent develop *.ARW --out-dir developed --format jpg       # batch; skips existing outputs
latent burst DIR                                           # list continuous-drive bursts
latent develop --merge DSC01126.ARW -o merged.jpg          # merge the burst this frame leads (or list the frames)
latent develop --superres 2 DSC01*.ARW -o sr.png            # burst super-resolution by kernel regression
latent params DSC00039.ARW                                  # resolved params (sidecar + defaults)
latent dump-bayer DSC00039.ARW -o raw.pgm                   # decoder cross-checks
```

Build with `cargo build --release`; the binary is `target/release/latent`.
Parameters live in `<raw>.latent.toml` beside each raw; flags override,
`--save-params` writes back. Every JPEG, PNG and AVIF output carries an
EXIF block: the source's camera, exposure, lens, time and GPS tags,
orientation reset (pixels are rotated), the output size and colour
space, `Software` = latent, and a `UserComment` holding the source name
and the parameters as TOML, so a render is reproducible from the
output alone. EXR has no EXIF slot and carries none. The camera's lens correction tables
(distortion, lateral CA, vignetting; Sony SubIFD tags) are applied by
default; `--lens none` disables them and `inspect` reports them.

GUI: `cargo run --release -p latent-gui -- DIR` opens a folder of images
(camera raws, plus PNG, JPEG, WebP, AVIF, JPEG XL, JPEG XR, EXR and
Radiance HDR through achromat) as a thumbnail grid (`latent-gui FILE.ARW` opens straight into that file;
no argument means the current directory). Thumbnails come from the
camera's embedded JPEG until a raw has a sidecar, then from a scaled
develop with the sidecar's parameters (marked with a dot); they are
cached under `~/.cache/latent/thumbs`. The toolbar has a filter box
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
field the file carries), `c` crop mode, `e` export JPEG, Ctrl+C copy
the developed image to the clipboard (a dialog picks PNG, JPEG or HDR
PNG and a long edge; Enter copies). The Geometry
group has a Straighten scrubber (degrees, positive clockwise; the crop
defaults to the largest same-aspect rectangle inside the turned
content) and Crop: in crop mode the whole straightened frame shows
with the outside of the crop dimmed, a plain drag draws the rectangle
(middle-drag pans), the aspect tabs lock its ratio, Reset clears both.
The Export group's long-edge scrubber resizes exports (0 = full size).
The Detail group is an unsharp mask on luminance at output resolution
(amount, radius, threshold); in the preview its radius shrinks with the
downscale, so judge it at 1:1. The Noise group runs bilateral chroma
and luminance smoothing on the demosaiced camera RGB, before the tone
curve; strengths 0 to 1.

Bursts: continuous-drive sequences are merged on the sensor plane
before demosaic, for roughly √N less noise. The sharpest frame is the
reference, frames under half its sharpness are left out, and the crop
is inset to where every frame covers. In the GUI a burst's leader shows
`×N` in the grid and a Burst group with a Merge switch (per image, in
the sidecar); exports and the clipboard use whatever the view shows.

Crates:

- `latent-core` — rawler unpack into a `SensorFrame`, achromat decode into
  the working space (`Source`), metadata readout (EXIF, GPS, Sony
  makernote; also from the EXIF block of JPEG, PNG and WebP files),
  color matrices, `DevelopParams`, the wgpu compute pipeline, CPU
  output transforms.
- `latent-formats` — encoders (EXR, PNG, JPEG, AVIF, PGM).
- `latent-cli` — the `latent` binary.
- `latent-gui` — the `latent-gui` window: library grid, thumbnail cache,
  develop view.

