# image_developer (working name)

Raw developer for the prism/damascene HDR ecosystem: Sony ARW in, HDR
and SDR images out, one GPU pipeline behind both a CLI and (later) a
damascene GUI. `DESIGN.md` is the plan and the decision record.

```
idev inspect DSC00039.ARW
idev develop DSC00039.ARW -o out.jpg                      # sRGB JPEG, sidecar params if present
idev develop DSC00039.ARW -o out.png --hdr 1000           # BT.2020 PQ 16-bit PNG for a 1000-nit master
idev develop DSC00039.ARW -o out.exr --tone none          # scene-referred scRGB linear f16
idev develop DSC00039.ARW -o out.jpg --exposure 0.5 --temp 5200,4 --contrast 1.8 --white 2.9 --save-params
idev develop *.ARW --out-dir developed --format jpg       # batch; skips existing outputs
idev params DSC00039.ARW                                  # resolved params (sidecar + defaults)
idev dump-bayer DSC00039.ARW -o raw.pgm                   # decoder cross-checks
```

Build with `cargo build --release`; the binary is `target/release/idev`.
Parameters live in `<raw>.idev.toml` beside each raw; flags override,
`--save-params` writes back. The camera's lens correction tables
(distortion, lateral CA, vignetting; Sony SubIFD tags) are applied by
default; `--lens none` disables them and `inspect` reports them.

GUI: `cargo run --release -p idev-gui -- DIR` opens a folder of raws as
a thumbnail grid (`idev-gui FILE.ARW` opens straight into that file;
no argument means the current directory). Thumbnails come from the
camera's embedded JPEG until a raw has a sidecar, then from a scaled
develop with the sidecar's parameters (marked with a dot); they are
cached under `~/.cache/idev/thumbs`. Enter or a double click opens the
develop view (damascene, HDR-aware on prism): pan/zoom preview, a
per-channel histogram of the preview, scrubbers for exposure, contrast,
hue preservation, white, temperature and tint, tabs for tone curve,
highlights and demosaic, switches for the lens corrections, export
buttons; the sidecar is written
(debounced) on every change. Keys: arrows or `hjkl` move (left/right
also step files in the develop view), Enter open, Esc back to the grid,
`o` open another folder, `f` fit, `1` 1:1, `t` SDR preview, `e` export
JPEG.

Crates:

- `idev-core` — rawler unpack into a `SensorFrame`, color matrices,
  `DevelopParams`, the wgpu compute pipeline, CPU output transforms.
- `idev-formats` — encoders (EXR, PNG, PGM).
- `idev-cli` — the `idev` binary.
- `idev-gui` — the `idev-gui` window: library grid, thumbnail cache,
  develop view.

`idev` is a placeholder prefix until the project is named.
