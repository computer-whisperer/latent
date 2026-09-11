//! The wgpu compute pipeline. One implementation serves the CLI (headless
//! device) and the GUI (device shared with damascene).

use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::WorkingImage;
use crate::color::Mat3;
use crate::params::{DevelopParams, WhiteBalance};
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
struct DemosaicUniforms {
    src_width: u32,
    src_height: u32,
    crop_x: u32,
    crop_y: u32,
    out_width: u32,
    out_height: u32,
    _pad: [u32; 2],
    cfa: [u32; 4],
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
}

/// Compiled pipelines. Build once per device, develop many frames.
pub struct Developer {
    linearize: wgpu::ComputePipeline,
    linearize_layout: wgpu::BindGroupLayout,
    demosaic: wgpu::ComputePipeline,
    demosaic_layout: wgpu::BindGroupLayout,
}

const WG: u32 = 16;

impl Developer {
    pub fn new(gpu: &Gpu) -> Developer {
        let device = &gpu.device;
        let lin_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("idev::linearize"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/linearize.wgsl").into()),
        });
        let dem_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("idev::demosaic_bilinear"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/demosaic_bilinear.wgsl").into()),
        });

        let linearize_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("idev::linearize"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_texture_entry(2, wgpu::TextureFormat::R32Float),
            ],
        });
        let demosaic_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("idev::demosaic"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_texture_entry(2, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        let linearize = make_pipeline(device, "idev::linearize", &linearize_layout, &lin_module);
        let demosaic = make_pipeline(device, "idev::demosaic", &demosaic_layout, &dem_module);
        Developer {
            linearize,
            linearize_layout,
            demosaic,
            demosaic_layout,
        }
    }

    /// Run the pipeline and read the working image back to the CPU.
    pub fn develop(
        &self,
        gpu: &Gpu,
        frame: &SensorFrame,
        params: &DevelopParams,
    ) -> Result<WorkingImage> {
        let device = &gpu.device;
        let queue = &gpu.queue;

        // --- CPU-side parameter resolution ---
        let white = match params.white_balance {
            WhiteBalance::AsShot => whitepoint::white_from_neutral(
                &frame.matrices,
                [1.0 / frame.wb_as_shot[0], 1.0, 1.0 / frame.wb_as_shot[2]],
            ),
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
        let gain = 2f32.powf(params.exposure_ev);
        let row = |r: [f32; 3]| [r[0] * gain, r[1] * gain, r[2] * gain, 0.0];

        let crop = frame.crop;
        let out_w = crop.width;
        let out_h = crop.height;

        // --- resources ---
        let raw_words: &[u8] = bytemuck::cast_slice(&frame.data);
        // Storage buffers need 4-byte size alignment; an odd pixel count
        // would leave a dangling u16. Pad by copying only in that case.
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

        let plane = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("idev::plane"),
            size: wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let plane_view = plane.create_view(&Default::default());

        let out_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("idev::working"),
            size: wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&Default::default());

        let lin_uniforms = LinearizeUniforms {
            width: frame.width,
            height: frame.height,
            _pad: [0; 2],
            black,
            scale,
        };
        let lin_ubuf = uniform_buffer(device, queue, "idev::linearize::uniforms", &lin_uniforms);
        let dem_uniforms = DemosaicUniforms {
            src_width: frame.width,
            src_height: frame.height,
            crop_x: crop.x,
            crop_y: crop.y,
            out_width: out_w,
            out_height: out_h,
            _pad: [0; 2],
            cfa: frame.cfa.map(|c| c as u32),
            m0: row(to_working[0]),
            m1: row(to_working[1]),
            m2: row(to_working[2]),
        };
        let dem_ubuf = uniform_buffer(device, queue, "idev::demosaic::uniforms", &dem_uniforms);

        let lin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("idev::linearize"),
            layout: &self.linearize_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lin_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: raw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&plane_view),
                },
            ],
        });
        let dem_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("idev::demosaic"),
            layout: &self.demosaic_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dem_ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&plane_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&out_view),
                },
            ],
        });

        // --- readback buffer ---
        let bytes_per_px = 8u32;
        let unpadded = out_w * bytes_per_px;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded.div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("idev::readback"),
            size: (padded_row * out_h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // --- encode ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("idev::develop"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("idev::linearize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.linearize);
            pass.set_bind_group(0, &lin_bg, &[]);
            pass.dispatch_workgroups(frame.width.div_ceil(WG), frame.height.div_ceil(WG), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("idev::demosaic"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.demosaic);
            pass.set_bind_group(0, &dem_bg, &[]);
            pass.dispatch_workgroups(out_w.div_ceil(WG), out_h.div_ceil(WG), 1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(out_h),
                },
            },
            wgpu::Extent3d {
                width: out_w,
                height: out_h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        // --- map ---
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
        let mut rgba = Vec::with_capacity((out_w * out_h * 4) as usize);
        for y in 0..out_h as usize {
            let start = y * padded_row as usize;
            let row = &mapped[start..start + unpadded as usize];
            rgba.extend_from_slice(bytemuck::cast_slice::<u8, f16>(row));
        }
        drop(mapped);
        readback.unmap();

        Ok(WorkingImage {
            width: out_w,
            height: out_h,
            rgba,
        })
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn uniform_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    value: &T,
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::bytes_of(value));
    buf
}
