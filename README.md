# image_developer (working name)

Raw developer for the prism/damascene HDR ecosystem: Sony ARW in, HDR
and SDR images out, one GPU pipeline behind both a CLI and (later) a
damascene GUI. `DESIGN.md` is the plan and the decision record.

```
cargo run --release -p idev-cli -- inspect  DSC00039.ARW
cargo run --release -p idev-cli -- develop  DSC00039.ARW -o out.exr   # scRGB linear f16
cargo run --release -p idev-cli -- develop  DSC00039.ARW -o out.png   # sRGB 8-bit, clipped
cargo run --release -p idev-cli -- develop  DSC00039.ARW -o out.png --exposure 1.0 --wb 2.5,1.5
cargo run --release -p idev-cli -- dump-bayer DSC00039.ARW -o raw.pgm  # decoder cross-checks
cargo run --release -p idev-cli -- params                              # default params as TOML
```

Crates:

- `idev-core` — rawler unpack into a `SensorFrame`, color matrices,
  `DevelopParams`, the wgpu compute pipeline, CPU output transforms.
- `idev-formats` — encoders (EXR, PNG, PGM).
- `idev-cli` — the `idev` binary.

`idev` is a placeholder prefix until the project is named.
