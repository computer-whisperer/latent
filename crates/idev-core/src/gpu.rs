//! The wgpu compute pipeline. One implementation serves the CLI (headless
//! device) and the GUI (device shared with damascene).
//!
//! Passes, all 16x16 workgroups:
//!
//! 1. `linearize`: raw u16 → f32 plane, black/white levels and white
//!    balance applied, unclamped.
//! 2. demosaic → camera RGB over the crop. Either `bilinear` (one pass)
//!    or RCD (`rcd_vh` discrimination + low-pass, `rcd_green`, `rcd_rb`).
//! 3. `colour`: optional box downscale, highlights, camera→BT.2020
//!    matrix (Bradford-adapted), exposure, tone curve, display primaries.
//!
//! Passes 1–2 are [`Developer::prepare`] and stay resident as a
//! [`Prepared`]; pass 3 is [`Developer::colour`], cheap enough to rerun
//! per slider event. [`Developer::develop`] chains both with a readback.

use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::WorkingImage;
use rawler::Orientation;

use crate::color::{Gamut, Mat3, XYZ_FROM_BT2020, srgb_from_bt2020};
use crate::geometry::{self, Layout};
use crate::params::{
    DEFAULT_WHITE, Demosaic, DevelopParams, Highlights, ToneCurve, WhiteBalance, sigmoid_constants,
};
use crate::sensor::{LensCorrection, SensorFrame};
use crate::source::{ImageSource, Source};
use crate::whitepoint::{self, Temperature, xy_from_temperature};

/// A device and queue the pipeline runs on.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Create a headless device on the first high-performance adapter.
    pub fn headless() -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .context("no wgpu adapter")?;
        let info = adapter.get_info();
        tracing::info!(name = %info.name, backend = ?info.backend, "gpu adapter");
        // Sensors outrun the 8192 default texture limit; take what the
        // hardware offers.
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("idev"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .context("request wgpu device")?;
        Ok(Gpu { device, queue })
    }

    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Gpu {
        Gpu { device, queue }
    }
}

/// What the developed image is destined for; the tone curve targets it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutputTarget {
    /// Display peak relative to diffuse white (1.0 for SDR; nits / 203
    /// for HDR).
    pub peak: f32,
}

impl OutputTarget {
    pub const SDR: OutputTarget = OutputTarget { peak: 1.0 };

    pub fn hdr_nits(peak_nits: f32) -> OutputTarget {
        OutputTarget {
            peak: peak_nits / 203.0,
        }
    }
}

// ---- uniform blocks (WGSL std140-compatible layouts) ----

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearizeUniforms {
    width: u32,
    height: u32,
    _pad: [u32; 2],
    black: [f32; 4],
    scale: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneUniforms {
    width: u32,
    height: u32,
    _pad: [u32; 2],
    cfa: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CropUniforms {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    out_width: u32,
    out_height: u32,
    _pad: [u32; 2],
    cfa: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HistogramUniforms {
    width: u32,
    height: u32,
    peak: f32,
    _pad: u32,
}

/// 3 x 256 bins over sRGB-encoded channel values.
#[derive(Debug, Clone)]
pub struct Histogram {
    pub bins: [[u32; 256]; 3],
    pub pixels: u32,
}

/// A histogram in flight on the GPU. Poll with [`Self::try_take`] from
/// the UI thread; it never blocks.
pub struct HistogramReadback {
    staging: wgpu::Buffer,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    mapped: bool,
    pixels: u32,
}

impl HistogramReadback {
    /// The histogram once the GPU has finished and the buffer mapped;
    /// `None` while still pending. Consumes the readback on success.
    pub fn try_take(&mut self) -> Option<Histogram> {
        if !self.mapped {
            match self.rx.try_recv() {
                Ok(Ok(())) => self.mapped = true,
                Ok(Err(e)) => {
                    tracing::warn!("histogram map failed: {e:?}");
                    return None;
                }
                Err(_) => return None,
            }
        }
        let view = self.staging.slice(..).get_mapped_range().ok()?;
        let words: &[u32] = bytemuck::cast_slice(&view);
        let mut bins = [[0u32; 256]; 3];
        for c in 0..3 {
            bins[c].copy_from_slice(&words[c * 256..(c + 1) * 256]);
        }
        drop(view);
        self.staging.unmap();
        self.mapped = false;
        Some(Histogram {
            bins,
            pixels: self.pixels,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColourUniforms {
    width: u32,
    height: u32,
    highlight_mode: u32,
    tone_mode: u32,
    src_width: u32,
    src_height: u32,
    scale: u32,
    _pad: u32,
    clip: [f32; 4],
    ylum_cam: [f32; 4],
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
    d0: [f32; 4],
    d1: [f32; 4],
    d2: [f32; 4],
    tone: [f32; 4],
    tone2: [f32; 4],
    /// distortion, chromatic aberration, vignetting flags; knot count
    lens: [u32; 4],
    /// centre x, y (camera RGB pixels), 1 / half-diagonal, pad
    centre: [f32; 4],
    knots: [[f32; 4]; 4],
    dist: [[f32; 4]; 4],
    ca_r: [[f32; 4]; 4],
    ca_b: [[f32; 4]; 4],
    vig: [[f32; 4]; 4],
    /// warning overlay enabled, pad
    warn: [u32; 4],
    /// working space → the panel's gamut, for the overlay
    g0: [f32; 4],
    g1: [f32; 4],
    g2: [f32; 4],
    /// view → sensor: m00, m01, m10, m11
    geo_m: [f32; 4],
    /// view → sensor: tx, ty; crop origin x, y (view pixels)
    geo_t: [f32; 4],
    /// dim outside this view rectangle: x, y, w, h (w = 0: off)
    dim: [f32; 4],
}

/// Spread a correction table over the uniform's 16 slots, repeating
/// the last value (the spline clamps beyond the last knot anyway).
fn lens_table(values: &[f32]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    let last = values.last().copied().unwrap_or(1.0);
    for i in 0..16 {
        out[i / 4][i % 4] = values.get(i).copied().unwrap_or(last);
    }
    out
}

/// Primaries the colour pass writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayPrimaries {
    /// The working space, linear BT.2020 (exports).
    Bt2020,
    /// Linear sRGB primaries, scRGB convention (damascene's working space).
    Srgb,
}

/// How the colour pass renders: for what output, at what downscale,
/// in which primaries, and whether to mark what the panel cannot show.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Render {
    pub target: OutputTarget,
    pub scale: u32,
    pub display: DisplayPrimaries,
    /// Zebra-mark pixels at the target peak (red) and outside this
    /// gamut (cyan). A preview aid; never for exports.
    pub warn: Option<Gamut>,
    /// Dim everything outside this rectangle (view pixels: x, y, w, h)
    /// — the crop being drawn over the full frame. A preview aid.
    pub dim_outside: Option<[f32; 4]>,
}

/// The expensive, parameter-stable half of a develop: the linearized
/// plane and demosaiced camera RGB, resident on the GPU. Rebuilt only
/// when white balance or the demosaic choice changes.
pub struct Prepared {
    pub width: u32,
    pub height: u32,
    pub white_balance: WhiteBalance,
    pub demosaic: Demosaic,
    /// A rendered image: no sensor-space steps (highlights, lens).
    is_image: bool,
    /// EXIF orientation, folded into the output geometry.
    orientation: Orientation,
    wb: [f32; 3],
    to_working: Mat3,
    ylum_cam: [f32; 3],
    lens: Option<LensCorrection>,
    cam_rgb: Tex,
}

impl Prepared {
    /// Whether `params` would need a fresh [`Developer::prepare`].
    pub fn matches(&self, params: &DevelopParams) -> bool {
        self.is_image
            || (self.white_balance == params.white_balance && self.demosaic == params.demosaic)
    }
}

// ---- pipelines ----

/// One compute pass: pipeline plus its bind group layout.
struct Pass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// What a binding slot holds, for building layouts tersely.
#[derive(Clone, Copy)]
enum Slot {
    Uniform,
    StorageBuffer,
    StorageBufferRw,
    Texture,
    StorageTexture(wgpu::TextureFormat),
}

impl Pass {
    fn new(device: &wgpu::Device, label: &str, wgsl: &str, slots: &[Slot]) -> Pass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = slots
            .iter()
            .enumerate()
            .map(|(i, slot)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match slot {
                    Slot::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Slot::StorageBuffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Slot::StorageBufferRw => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Slot::Texture => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    Slot::StorageTexture(format) => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: *format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Pass { pipeline, layout }
    }

    fn bind(
        &self,
        device: &wgpu::Device,
        resources: &[wgpu::BindingResource<'_>],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r.clone(),
            })
            .collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        })
    }

    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(WG), height.div_ceil(WG), 1);
    }
}

const WG: u32 = 16;

/// Compiled pipelines. Build once per device, develop many frames.
pub struct Developer {
    linearize: Pass,
    bilinear: Pass,
    rcd_vh: Pass,
    rcd_green: Pass,
    rcd_rb: Pass,
    colour: Pass,
    histogram: Pass,
}

impl Developer {
    pub fn new(gpu: &Gpu) -> Developer {
        use Slot::*;
        use wgpu::TextureFormat::{R32Float, Rg32Float, Rgba16Float};
        let d = &gpu.device;
        Developer {
            linearize: Pass::new(
                d,
                "idev::linearize",
                include_str!("shaders/linearize.wgsl"),
                &[Uniform, StorageBuffer, StorageTexture(R32Float)],
            ),
            bilinear: Pass::new(
                d,
                "idev::bilinear",
                include_str!("shaders/demosaic_bilinear.wgsl"),
                &[Uniform, Texture, StorageTexture(Rgba16Float)],
            ),
            rcd_vh: Pass::new(
                d,
                "idev::rcd_vh",
                include_str!("shaders/rcd_vh.wgsl"),
                &[Uniform, Texture, StorageTexture(Rg32Float)],
            ),
            rcd_green: Pass::new(
                d,
                "idev::rcd_green",
                include_str!("shaders/rcd_green.wgsl"),
                &[Uniform, Texture, Texture, StorageTexture(R32Float)],
            ),
            rcd_rb: Pass::new(
                d,
                "idev::rcd_rb",
                include_str!("shaders/rcd_rb.wgsl"),
                &[Uniform, Texture, Texture, StorageTexture(Rgba16Float)],
            ),
            colour: Pass::new(
                d,
                "idev::colour",
                include_str!("shaders/colour.wgsl"),
                &[Uniform, Texture, StorageTexture(Rgba16Float)],
            ),
            histogram: Pass::new(
                d,
                "idev::histogram",
                include_str!("shaders/histogram.wgsl"),
                &[Uniform, Texture, StorageBufferRw],
            ),
        }
    }

    /// Bring `source` onto the GPU for `params`: a raw is linearized and
    /// demosaiced for its white balance and demosaic choice, a rendered
    /// image is uploaded as is.
    pub fn prepare(&self, gpu: &Gpu, source: &Source, params: &DevelopParams) -> Result<Prepared> {
        match source {
            Source::Raw(frame) => self.prepare_raw(gpu, frame, params),
            Source::Image(img) => Ok(Self::prepare_image(gpu, img, params)),
        }
    }

    fn prepare_image(gpu: &Gpu, img: &ImageSource, params: &DevelopParams) -> Prepared {
        let size = wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        };
        let cam_rgb = texture(
            &gpu.device,
            "idev::cam_rgb",
            size,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &cam_rgb.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&img.rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(img.width * 8),
                rows_per_image: Some(img.height),
            },
            size,
        );
        Prepared {
            width: img.width,
            height: img.height,
            white_balance: params.white_balance,
            demosaic: params.demosaic,
            is_image: true,
            orientation: img.orientation,
            wb: [1.0; 3],
            to_working: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ylum_cam: XYZ_FROM_BT2020[1],
            lens: None,
            cam_rgb,
        }
    }

    fn prepare_raw(
        &self,
        gpu: &Gpu,
        frame: &SensorFrame,
        params: &DevelopParams,
    ) -> Result<Prepared> {
        let device = &gpu.device;
        let queue = &gpu.queue;

        let white = match params.white_balance {
            WhiteBalance::AsShot => {
                whitepoint::white_from_neutral(&frame.matrices, frame.neutral_as_shot())
            }
            WhiteBalance::Multipliers { red, blue } => {
                whitepoint::white_from_neutral(&frame.matrices, [1.0 / red, 1.0, 1.0 / blue])
            }
            WhiteBalance::Temperature { kelvin, tint } => whitepoint::white_from_xy(
                &frame.matrices,
                xy_from_temperature(Temperature { kelvin, tint }),
            ),
        }
        .context("white balance does not resolve to a valid white")?;
        let wb = white.multipliers();
        tracing::debug!(?wb, temperature = ?white.temperature, "white balance");
        let mut black = [0.0f32; 4];
        let mut scale = [0.0f32; 4];
        for pos in 0..4 {
            let range = frame.white[pos] - frame.black[pos];
            if range <= 0.0 {
                return Err(anyhow!(
                    "CFA position {pos}: white {} <= black {}",
                    frame.white[pos],
                    frame.black[pos]
                ));
            }
            black[pos] = frame.black[pos];
            scale[pos] = wb[frame.cfa[pos] as usize] / range;
        }
        let to_working: Mat3 = white
            .bt2020_from_camera_wb()
            .context("camera matrix is singular")?;
        // Luminance weights in camera WB space: Y row of XYZ←BT.2020
        // composed with camera→BT.2020.
        let ylum_cam = {
            let y2020 = XYZ_FROM_BT2020[1];
            let mut w = [0.0f32; 3];
            for (j, wj) in w.iter_mut().enumerate() {
                *wj = (0..3).map(|i| y2020[i] * to_working[i][j]).sum();
            }
            w
        };
        let cfa = frame.cfa.map(|c| c as u32);
        let crop = frame.crop;
        let (out_w, out_h) = (crop.width, crop.height);

        let raw_words: &[u8] = bytemuck::cast_slice(&frame.data);
        // Storage buffers need 4-byte size alignment; an odd pixel count
        // would leave a dangling u16.
        let padded;
        let raw_bytes: &[u8] = if raw_words.len().is_multiple_of(4) {
            raw_words
        } else {
            let mut v = raw_words.to_vec();
            v.resize(v.len() + 2, 0);
            padded = v;
            &padded
        };
        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idev::raw"),
            size: raw_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&raw_buf, 0, raw_bytes);

        let full = wgpu::Extent3d {
            width: frame.width,
            height: frame.height,
            depth_or_array_layers: 1,
        };
        let cropped = wgpu::Extent3d {
            width: out_w,
            height: out_h,
            depth_or_array_layers: 1,
        };
        let intermediate =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let plane = texture(
            device,
            "idev::plane",
            full,
            wgpu::TextureFormat::R32Float,
            intermediate,
        );
        let cam_rgb = texture(
            device,
            "idev::cam_rgb",
            cropped,
            wgpu::TextureFormat::Rgba16Float,
            intermediate,
        );

        let lin_u = uniform_buffer(
            device,
            queue,
            &LinearizeUniforms {
                width: frame.width,
                height: frame.height,
                _pad: [0; 2],
                black,
                scale,
            },
        );
        let plane_u = uniform_buffer(
            device,
            queue,
            &PlaneUniforms {
                width: frame.width,
                height: frame.height,
                _pad: [0; 2],
                cfa,
            },
        );
        let crop_u = uniform_buffer(
            device,
            queue,
            &CropUniforms {
                src_width: frame.width,
                src_height: frame.height,
                crop_x: crop.x,
                crop_y: crop.y,
                out_width: out_w,
                out_height: out_h,
                _pad: [0; 2],
                cfa,
            },
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("idev::prepare"),
        });
        let bg = self.linearize.bind(
            device,
            &[
                lin_u.as_entire_binding(),
                raw_buf.as_entire_binding(),
                tex(&plane),
            ],
        );
        self.linearize
            .dispatch(&mut encoder, &bg, frame.width, frame.height);

        match params.demosaic {
            Demosaic::Bilinear => {
                let bg = self.bilinear.bind(
                    device,
                    &[crop_u.as_entire_binding(), tex(&plane), tex(&cam_rgb)],
                );
                self.bilinear.dispatch(&mut encoder, &bg, out_w, out_h);
            }
            Demosaic::Rcd => {
                let vhlpf = texture(
                    device,
                    "idev::rcd_vhlpf",
                    full,
                    wgpu::TextureFormat::Rg32Float,
                    intermediate,
                );
                let green = texture(
                    device,
                    "idev::rcd_green",
                    full,
                    wgpu::TextureFormat::R32Float,
                    intermediate,
                );
                let bg = self.rcd_vh.bind(
                    device,
                    &[plane_u.as_entire_binding(), tex(&plane), tex(&vhlpf)],
                );
                self.rcd_vh
                    .dispatch(&mut encoder, &bg, frame.width, frame.height);
                let bg = self.rcd_green.bind(
                    device,
                    &[
                        plane_u.as_entire_binding(),
                        tex(&plane),
                        tex(&vhlpf),
                        tex(&green),
                    ],
                );
                self.rcd_green
                    .dispatch(&mut encoder, &bg, frame.width, frame.height);
                let bg = self.rcd_rb.bind(
                    device,
                    &[
                        crop_u.as_entire_binding(),
                        tex(&plane),
                        tex(&green),
                        tex(&cam_rgb),
                    ],
                );
                self.rcd_rb.dispatch(&mut encoder, &bg, out_w, out_h);
            }
        }
        queue.submit([encoder.finish()]);

        Ok(Prepared {
            width: out_w,
            height: out_h,
            white_balance: params.white_balance,
            demosaic: params.demosaic,
            is_image: false,
            orientation: frame.orientation,
            wb,
            to_working,
            ylum_cam,
            lens: frame.lens.clone(),
            cam_rgb,
        })
    }

    /// Output size of the colour pass for a downscale factor.
    /// The output geometry (orientation, straighten, crop) for `params`.
    pub fn layout(prepared: &Prepared, params: &DevelopParams) -> Layout {
        geometry::layout(
            (prepared.width, prepared.height),
            prepared.orientation,
            &params.geometry,
        )
    }

    /// Output size for `params` at an integer downscale.
    pub fn output_size(prepared: &Prepared, params: &DevelopParams, scale: u32) -> (u32, u32) {
        Self::layout(prepared, params).scaled(scale)
    }

    /// Run the colour pass into `out` (an `Rgba16Float` storage texture
    /// of [`Self::output_size`] for `render.scale`) and submit it.
    pub fn colour(
        &self,
        gpu: &Gpu,
        prepared: &Prepared,
        params: &DevelopParams,
        render: Render,
        out: &wgpu::TextureView,
    ) {
        let device = &gpu.device;
        let queue = &gpu.queue;
        let Render {
            target,
            scale,
            display,
            warn,
            dim_outside,
        } = render;
        let layout = Self::layout(prepared, params);
        let (out_w, out_h) = layout.scaled(scale);
        let gain = 2f32.powf(params.exposure_ev);
        let (tone_mode, contrast, hue_preservation, white) = match params.tone {
            ToneCurve::None => (0u32, 1.0, 0.0, DEFAULT_WHITE),
            ToneCurve::Sigmoid {
                contrast,
                hue_preservation,
                white,
            } => (1u32, contrast, hue_preservation, white),
        };
        let (contrast, sig_s, white) = sigmoid_constants(contrast, white, target.peak);
        let f_white = white.powf(contrast) / (white.powf(contrast) + sig_s);
        let highlight_mode = match (prepared.is_image, params.highlights) {
            (true, _) => 2u32,
            (false, Highlights::Clip) => 0u32,
            (false, Highlights::Blend) => 1u32,
        };
        let d: Mat3 = match display {
            DisplayPrimaries::Bt2020 => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            DisplayPrimaries::Srgb => srgb_from_bt2020(),
        };
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        let m = &prepared.to_working;
        let g: Mat3 = warn.map(Gamut::from_bt2020).unwrap_or([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let (cx, cy) = (prepared.width as f32 * 0.5, prepared.height as f32 * 0.5);
        let rf = 1.0 / (cx * cx + cy * cy).sqrt();
        let (lens_flags, knots, dist, ca_r, ca_b, vig) = match &prepared.lens {
            Some(l) => (
                [
                    params.lens.distortion as u32,
                    params.lens.chromatic_aberration as u32,
                    params.lens.vignetting as u32,
                    l.knots.len() as u32,
                ],
                lens_table(&l.knots),
                lens_table(&l.distortion),
                lens_table(&l.ca_red),
                lens_table(&l.ca_blue),
                lens_table(&l.vignetting),
            ),
            None => (
                [0, 0, 0, 1],
                lens_table(&[1.0]),
                lens_table(&[1.0]),
                lens_table(&[1.0]),
                lens_table(&[1.0]),
                lens_table(&[1.0]),
            ),
        };
        let colour_u = uniform_buffer(
            device,
            queue,
            &ColourUniforms {
                width: out_w,
                height: out_h,
                highlight_mode,
                tone_mode,
                src_width: prepared.width,
                src_height: prepared.height,
                scale: scale.max(1),
                _pad: 0,
                clip: [prepared.wb[0], prepared.wb[1], prepared.wb[2], 0.0],
                ylum_cam: [
                    prepared.ylum_cam[0],
                    prepared.ylum_cam[1],
                    prepared.ylum_cam[2],
                    0.0,
                ],
                m0: row(m[0]),
                m1: row(m[1]),
                m2: row(m[2]),
                d0: row(d[0]),
                d1: row(d[1]),
                d2: row(d[2]),
                tone: [gain, contrast, target.peak, hue_preservation],
                tone2: [sig_s, white, 1.0 / f_white, 0.0],
                lens: lens_flags,
                centre: [cx, cy, rf, 0.0],
                knots,
                dist,
                ca_r,
                ca_b,
                vig,
                warn: [warn.is_some() as u32, 0, 0, 0],
                g0: row(g[0]),
                g1: row(g[1]),
                g2: row(g[2]),
                geo_m: [
                    layout.m[0][0],
                    layout.m[0][1],
                    layout.m[1][0],
                    layout.m[1][1],
                ],
                geo_t: [layout.t[0], layout.t[1], layout.crop[0], layout.crop[1]],
                dim: dim_outside.unwrap_or([0.0; 4]),
            },
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("idev::colour"),
        });
        let bg = self.colour.bind(
            device,
            &[
                colour_u.as_entire_binding(),
                tex(&prepared.cam_rgb),
                wgpu::BindingResource::TextureView(out),
            ],
        );
        self.colour.dispatch(&mut encoder, &bg, out_w, out_h);
        queue.submit([encoder.finish()]);
    }

    /// Submit a histogram of a developed texture (values divided by
    /// `peak` before sRGB encoding) and start its readback.
    pub fn histogram(
        &self,
        gpu: &Gpu,
        image: &wgpu::TextureView,
        width: u32,
        height: u32,
        peak: f32,
    ) -> HistogramReadback {
        let device = &gpu.device;
        let queue = &gpu.queue;
        let size = (3 * 256 * std::mem::size_of::<u32>()) as u64;
        let bins = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idev::histogram::bins"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idev::histogram::staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let u = uniform_buffer(
            device,
            queue,
            &HistogramUniforms {
                width,
                height,
                peak: peak.max(1e-3),
                _pad: 0,
            },
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("idev::histogram"),
        });
        encoder.clear_buffer(&bins, 0, None);
        let bg = self.histogram.bind(
            device,
            &[
                u.as_entire_binding(),
                wgpu::BindingResource::TextureView(image),
                bins.as_entire_binding(),
            ],
        );
        self.histogram.dispatch(&mut encoder, &bg, width, height);
        encoder.copy_buffer_to_buffer(&bins, 0, &staging, 0, size);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        HistogramReadback {
            staging,
            rx,
            mapped: false,
            pixels: width * height,
        }
    }

    /// Full-resolution develop with CPU readback (the CLI path).
    pub fn develop(
        &self,
        gpu: &Gpu,
        source: &Source,
        params: &DevelopParams,
        target: OutputTarget,
    ) -> Result<WorkingImage> {
        let prepared = self.prepare(gpu, source, params)?;
        self.develop_prepared(gpu, &prepared, params, target, 1)
    }

    /// Colour pass over an existing [`Prepared`] at an integer
    /// downscale (see [`Self::colour`]), read back to the CPU.
    pub fn develop_prepared(
        &self,
        gpu: &Gpu,
        prepared: &Prepared,
        params: &DevelopParams,
        target: OutputTarget,
        scale: u32,
    ) -> Result<WorkingImage> {
        let device = &gpu.device;
        let (out_w, out_h) = Self::output_size(prepared, params, scale);
        let working = texture(
            device,
            "idev::working",
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        self.colour(
            gpu,
            prepared,
            params,
            Render {
                target,
                scale,
                display: DisplayPrimaries::Bt2020,
                warn: None,
                dim_outside: None,
            },
            &working.view,
        );
        let rgba = read_back_rgba16f(gpu, &working.texture, out_w, out_h)?;
        Ok(WorkingImage {
            width: out_w,
            height: out_h,
            rgba,
        })
    }
}

/// Copy an `Rgba16Float` texture to the CPU.
fn read_back_rgba16f(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<f16>> {
    let device = &gpu.device;
    let queue = &gpu.queue;
    let bytes_per_px = 8u32;
    let unpadded = width * bytes_per_px;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = unpadded.div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("idev::readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("idev::readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("poll device")?;
    rx.recv()
        .context("map callback dropped")?
        .context("map readback")?;
    let mapped = slice.get_mapped_range().context("mapped range")?;
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height as usize {
        let start = y * padded_row as usize;
        let row = &mapped[start..start + unpadded as usize];
        rgba.extend_from_slice(bytemuck::cast_slice::<u8, f16>(row));
    }
    drop(mapped);
    readback.unmap();
    Ok(rgba)
}

struct Tex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Tex {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Tex { texture, view }
}

fn tex(t: &Tex) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(&t.view)
}

fn uniform_buffer<T: Pod>(device: &wgpu::Device, queue: &wgpu::Queue, value: &T) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("idev::uniforms"),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::bytes_of(value));
    buf
}
