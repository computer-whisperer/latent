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
use crate::color::{Mat3, XYZ_FROM_BT2020, srgb_from_bt2020};
use crate::params::{Demosaic, DevelopParams, Highlights, ToneCurve, WhiteBalance};
use crate::sensor::SensorFrame;
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
}

/// Primaries the colour pass writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayPrimaries {
    /// The working space, linear BT.2020 (exports).
    Bt2020,
    /// Linear sRGB primaries, scRGB convention (damascene's working space).
    Srgb,
}

/// The expensive, parameter-stable half of a develop: the linearized
/// plane and demosaiced camera RGB, resident on the GPU. Rebuilt only
/// when white balance or the demosaic choice changes.
pub struct Prepared {
    pub width: u32,
    pub height: u32,
    pub white_balance: WhiteBalance,
    pub demosaic: Demosaic,
    wb: [f32; 3],
    to_working: Mat3,
    ylum_cam: [f32; 3],
    cam_rgb: Tex,
}

impl Prepared {
    /// Whether `params` would need a fresh [`Developer::prepare`].
    pub fn matches(&self, params: &DevelopParams) -> bool {
        self.white_balance == params.white_balance && self.demosaic == params.demosaic
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
        }
    }

    /// Linearize and demosaic `frame` for the white balance and demosaic
    /// in `params`; the result stays on the GPU.
    pub fn prepare(
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
            wb,
            to_working,
            ylum_cam,
            cam_rgb,
        })
    }

    /// Output size of the colour pass for a downscale factor.
    pub fn scaled_size(prepared: &Prepared, scale: u32) -> (u32, u32) {
        let s = scale.max(1);
        (prepared.width.div_ceil(s), prepared.height.div_ceil(s))
    }

    /// Run the colour pass into `out` (an `Rgba16Float` storage texture
    /// of [`Self::scaled_size`]) and submit it.
    #[allow(clippy::too_many_arguments)]
    pub fn colour(
        &self,
        gpu: &Gpu,
        prepared: &Prepared,
        params: &DevelopParams,
        target: OutputTarget,
        scale: u32,
        display: DisplayPrimaries,
        out: &wgpu::TextureView,
    ) {
        let device = &gpu.device;
        let queue = &gpu.queue;
        let (out_w, out_h) = Self::scaled_size(prepared, scale);
        let gain = 2f32.powf(params.exposure_ev);
        let (tone_mode, contrast, hue_preservation) = match params.tone {
            ToneCurve::None => (0u32, 1.0, 0.0),
            ToneCurve::Sigmoid {
                contrast,
                hue_preservation,
            } => (1u32, contrast, hue_preservation),
        };
        let highlight_mode = match params.highlights {
            Highlights::Clip => 0u32,
            Highlights::Blend => 1u32,
        };
        let d: Mat3 = match display {
            DisplayPrimaries::Bt2020 => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            DisplayPrimaries::Srgb => srgb_from_bt2020(),
        };
        let row = |r: [f32; 3]| [r[0], r[1], r[2], 0.0];
        let m = &prepared.to_working;
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

    /// Full-resolution develop with CPU readback (the CLI path).
    pub fn develop(
        &self,
        gpu: &Gpu,
        frame: &SensorFrame,
        params: &DevelopParams,
        target: OutputTarget,
    ) -> Result<WorkingImage> {
        let prepared = self.prepare(gpu, frame, params)?;
        self.develop_prepared(gpu, &prepared, params, target)
    }

    /// Full-resolution colour pass over an existing [`Prepared`], read
    /// back to the CPU.
    pub fn develop_prepared(
        &self,
        gpu: &Gpu,
        prepared: &Prepared,
        params: &DevelopParams,
        target: OutputTarget,
    ) -> Result<WorkingImage> {
        let device = &gpu.device;
        let (out_w, out_h) = (prepared.width, prepared.height);
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
            target,
            1,
            DisplayPrimaries::Bt2020,
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
