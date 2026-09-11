//! Raw development pipeline.
//!
//! - [`sensor`] — unpack a camera raw file into a [`sensor::SensorFrame`]:
//!   the Bayer plane plus everything needed to interpret it.
//! - [`color`] — matrices between camera space, XYZ, and the working
//!   space (linear BT.2020, D65).
//! - [`params`] — [`params::DevelopParams`], the serialized edit state
//!   shared by CLI, GUI, and sidecars.
//! - [`gpu`] — the wgpu compute pipeline that turns a frame plus params
//!   into a [`WorkingImage`]. Headless for the CLI; the GUI hands in its
//!   own device.
//! - [`output`] — CPU output transforms from the working space to
//!   export encodings.

pub mod color;
pub mod gpu;
pub mod output;
pub mod params;
pub mod sensor;
pub mod source;
pub mod whitepoint;

/// A developed frame in the working space: linear BT.2020, D65, RGBA
/// f16, 1.0 = scene diffuse white, row-major, top-left origin.
#[derive(Debug, Clone)]
pub struct WorkingImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<half::f16>,
}
