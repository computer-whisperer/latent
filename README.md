# image_developer (working name)

Raw developer for the prism/damascene HDR ecosystem: Sony ARW in, HDR
and SDR images out, one GPU pipeline behind both a CLI and (later) a
damascene GUI. `DESIGN.md` is the plan and the decision record.

```
idev inspect DSC00039.ARW
idev develop DSC00039.ARW -o out.jpg                      # sRGB JPEG, sidecar params if present
idev develop DSC00039.ARW -o out.png --hdr 1000           # BT.2020 PQ 16-bit PNG for a 1000-nit master
idev develop DSC00039.ARW -o out.exr --tone none          # scene-referred scRGB linear f16
idev develop DSC00039.ARW -o out.jpg --exposure 0.5 --temp 5200,4 --contrast 1.8 --save-params
idev develop *.ARW --out-dir developed --format jpg       # batch; skips existing outputs
idev params DSC00039.ARW                                  # resolved params (sidecar + defaults)
idev dump-bayer DSC00039.ARW -o raw.pgm                   # decoder cross-checks
```

Build with `cargo build --release`; the binary is `target/release/idev`.
Parameters live in `<raw>.idev.toml` beside each raw; flags override,
`--save-params` writes back.

Crates:

- `idev-core` — rawler unpack into a `SensorFrame`, color matrices,
  `DevelopParams`, the wgpu compute pipeline, CPU output transforms.
- `idev-formats` — encoders (EXR, PNG, PGM).
- `idev-cli` — the `idev` binary.

`idev` is a placeholder prefix until the project is named.
