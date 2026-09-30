//! Output geometry: EXIF orientation, straightening and the crop, folded
//! into one affine map from output pixels to camera RGB pixels so the
//! colour pass resamples once.
//!
//! Frames, all in pixel units with pixel centres at +0.5:
//!
//! - sensor: the camera RGB texture, `W × H`, as stored;
//! - oriented: the sensor rotated/flipped for viewing, `Wo × Ho`;
//! - view: the oriented frame turned by the straighten angle about its
//!   centre, same size, content rotates within it (corners empty);
//! - output: the crop rectangle of the view, `width × height`.
//!
//! `Geometry::crop` is given as fractions of the view frame. `None`
//! means the largest rectangle of the view's aspect that stays inside
//! the rotated content (the whole frame at 0°), where the content is
//! the sensor frame less `inset` on every side: what the lens
//! correction can fill without sampling past the edge
//! (`LensCorrection::valid_inset`), or the whole frame.

use rawler::Orientation;

use crate::params::Geometry;

/// The resolved mapping for one source and one `Geometry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    /// Oriented (= view) frame size.
    pub view: (u32, u32),
    /// Crop in view pixels: x, y, w, h.
    pub crop: [f32; 4],
    /// Output size at scale 1.
    pub width: u32,
    pub height: u32,
    /// sensor = m · view + t.
    pub m: [[f32; 2]; 2],
    pub t: [f32; 2],
}

impl Layout {
    /// Output size at an integer downscale.
    pub fn scaled(&self, scale: u32) -> (u32, u32) {
        let s = scale.max(1);
        (self.width.div_ceil(s), self.height.div_ceil(s))
    }

    /// Sensor coordinate of a view coordinate.
    pub fn source(&self, v: (f32, f32)) -> (f32, f32) {
        (
            self.m[0][0] * v.0 + self.m[0][1] * v.1 + self.t[0],
            self.m[1][0] * v.0 + self.m[1][1] * v.1 + self.t[1],
        )
    }
}

/// The frame size after EXIF orientation.
pub fn oriented_size(sensor: (u32, u32), orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Transpose
        | Orientation::Rotate90
        | Orientation::Transverse
        | Orientation::Rotate270 => (sensor.1, sensor.0),
        _ => sensor,
    }
}

/// Oriented → sensor as an affine map (continuous coordinates; the
/// same maps as `output::orient` takes per pixel).
fn unorient(sensor: (u32, u32), orientation: Orientation) -> ([[f32; 2]; 2], [f32; 2]) {
    let (w, h) = (sensor.0 as f32, sensor.1 as f32);
    match orientation {
        Orientation::Normal | Orientation::Unknown => ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
        Orientation::HorizontalFlip => ([[-1.0, 0.0], [0.0, 1.0]], [w, 0.0]),
        Orientation::Rotate180 => ([[-1.0, 0.0], [0.0, -1.0]], [w, h]),
        Orientation::VerticalFlip => ([[1.0, 0.0], [0.0, -1.0]], [0.0, h]),
        Orientation::Transpose => ([[0.0, 1.0], [1.0, 0.0]], [0.0, 0.0]),
        Orientation::Rotate90 => ([[0.0, 1.0], [-1.0, 0.0]], [0.0, h]),
        Orientation::Transverse => ([[0.0, -1.0], [-1.0, 0.0]], [w, h]),
        Orientation::Rotate270 => ([[0.0, -1.0], [1.0, 0.0]], [w, 0.0]),
    }
}

/// Largest rectangle of the view's own aspect, centred, that stays
/// inside the content after rotation by `angle_deg`; as fractions.
/// The content is the view less `inset` on every side (view axes).
pub fn auto_crop(view: (u32, u32), inset: (f32, f32), angle_deg: f32) -> [f32; 4] {
    let (w, h) = (view.0 as f32, view.1 as f32);
    let (cw, ch) = ((w - 2.0 * inset.0).max(1.0), (h - 2.0 * inset.1).max(1.0));
    let a = angle_deg.to_radians().abs();
    let (s, c) = a.sin_cos();
    let k = (cw / (w * c + h * s))
        .min(ch / (w * s + h * c))
        .clamp(0.0, 1.0);
    [(1.0 - k) * 0.5, (1.0 - k) * 0.5, k, k]
}

/// `inset` is in sensor pixels per side (x, y), before orientation.
pub fn layout(
    sensor: (u32, u32),
    orientation: Orientation,
    inset: (f32, f32),
    geo: &Geometry,
) -> Layout {
    let view = oriented_size(sensor, orientation);
    let (vw, vh) = (view.0 as f32, view.1 as f32);
    let view_inset = if view == sensor {
        inset
    } else {
        (inset.1, inset.0)
    };
    let frac = geo
        .crop
        .unwrap_or_else(|| auto_crop(view, view_inset, geo.angle_deg));
    let crop = [frac[0] * vw, frac[1] * vh, frac[2] * vw, frac[3] * vh];
    let width = (crop[2].round() as u32).max(1);
    let height = (crop[3].round() as u32).max(1);

    // view → oriented: p = c + R(-θ)(v - c); R(θ) turns the content
    // clockwise on screen (y down) for a positive angle.
    let (s, c) = geo.angle_deg.to_radians().sin_cos();
    let r = [[c, s], [-s, c]];
    let centre = [vw * 0.5, vh * 0.5];
    let r_t = [
        centre[0] - (r[0][0] * centre[0] + r[0][1] * centre[1]),
        centre[1] - (r[1][0] * centre[0] + r[1][1] * centre[1]),
    ];
    // oriented → sensor.
    let (a, b) = unorient(sensor, orientation);
    let m = [
        [
            a[0][0] * r[0][0] + a[0][1] * r[1][0],
            a[0][0] * r[0][1] + a[0][1] * r[1][1],
        ],
        [
            a[1][0] * r[0][0] + a[1][1] * r[1][0],
            a[1][0] * r[0][1] + a[1][1] * r[1][1],
        ],
    ];
    let t = [
        a[0][0] * r_t[0] + a[0][1] * r_t[1] + b[0],
        a[1][0] * r_t[0] + a[1][1] * r_t[1] + b[1],
    ];
    Layout {
        view,
        crop,
        width,
        height,
        m,
        t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::orient;

    const ALL: [Orientation; 8] = [
        Orientation::Normal,
        Orientation::HorizontalFlip,
        Orientation::Rotate180,
        Orientation::VerticalFlip,
        Orientation::Transpose,
        Orientation::Rotate90,
        Orientation::Transverse,
        Orientation::Rotate270,
    ];

    #[test]
    fn identity_is_exact() {
        let l = layout(
            (6192, 4128),
            Orientation::Normal,
            (0.0, 0.0),
            &Geometry::default(),
        );
        assert_eq!((l.width, l.height), (6192, 4128));
        assert_eq!(l.m, [[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(l.t, [0.0, 0.0]);
        assert_eq!(l.crop, [0.0, 0.0, 6192.0, 4128.0]);
        assert_eq!(l.scaled(4), (1548, 1032));
    }

    /// The affine map agrees with the per-pixel `output::orient` for
    /// every orientation: each output pixel centre lands in the source
    /// pixel that `orient` copied there.
    #[test]
    fn matches_cpu_orient() {
        let (w, h) = (5u32, 3u32);
        let pixels: Vec<u32> = (0..w * h).collect();
        for o in ALL {
            let (ow, oh, out) = orient(w, h, 1, &pixels, o);
            let l = layout((w, h), o, (0.0, 0.0), &Geometry::default());
            assert_eq!((l.width, l.height), (ow, oh), "{o:?}");
            for y in 0..oh {
                for x in 0..ow {
                    let (sx, sy) = l.source((x as f32 + 0.5, y as f32 + 0.5));
                    let src = sy.floor() as u32 * w + sx.floor() as u32;
                    assert_eq!(src, out[(y * ow + x) as usize], "{o:?} at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn positive_angle_turns_clockwise() {
        // With the view turned 90° clockwise, the right edge of the view
        // shows what was the top of the image.
        let geo = Geometry {
            angle_deg: 90.0,
            crop: Some([0.0, 0.0, 1.0, 1.0]),
        };
        let l = layout((10, 10), Orientation::Normal, (0.0, 0.0), &geo);
        let (sx, sy) = l.source((9.5, 5.0));
        assert!(
            (sx - 5.0).abs() < 1e-4 && (sy - 0.5).abs() < 1e-4,
            "{sx} {sy}"
        );
    }

    #[test]
    fn inset_shrinks_the_default_crop_only() {
        // 30 px off each side in x, 10 in y, on a 3:2 frame: the crop
        // keeps the aspect, so x binds (2940/3000 = 0.98).
        let l = layout(
            (3000, 2000),
            Orientation::Normal,
            (30.0, 10.0),
            &Geometry::default(),
        );
        assert_eq!((l.width, l.height), (2940, 1960));
        assert!(
            (l.crop[0] - 30.0).abs() < 1e-3 && (l.crop[1] - 20.0).abs() < 1e-3,
            "{:?}",
            l.crop
        );
        // Rotated 90°: the insets swap with the axes.
        let r = layout(
            (3000, 2000),
            Orientation::Rotate90,
            (30.0, 10.0),
            &Geometry::default(),
        );
        assert_eq!((r.width, r.height), (1960, 2940));
        // An explicit crop is the user's: untouched.
        let geo = Geometry {
            angle_deg: 0.0,
            crop: Some([0.0, 0.0, 1.0, 1.0]),
        };
        let e = layout((3000, 2000), Orientation::Normal, (30.0, 10.0), &geo);
        assert_eq!((e.width, e.height), (3000, 2000));
    }

    #[test]
    fn auto_crop_fits_rotated_content() {
        assert_eq!(
            auto_crop((3000, 2000), (0.0, 0.0), 0.0),
            [0.0, 0.0, 1.0, 1.0]
        );
        let c = auto_crop((3000, 2000), (0.0, 0.0), 5.0);
        // The corner of the crop, rotated back into the content, stays inside.
        let geo = Geometry {
            angle_deg: 5.0,
            crop: None,
        };
        let l = layout((3000, 2000), Orientation::Normal, (0.0, 0.0), &geo);
        for (vx, vy) in [
            (l.crop[0], l.crop[1]),
            (l.crop[0] + l.crop[2], l.crop[1]),
            (l.crop[0], l.crop[1] + l.crop[3]),
            (l.crop[0] + l.crop[2], l.crop[1] + l.crop[3]),
        ] {
            let (sx, sy) = l.source((vx, vy));
            assert!(
                (-0.01..=3000.01).contains(&sx) && (-0.01..=2000.01).contains(&sy),
                "{sx} {sy}"
            );
        }
        assert!(c[2] < 1.0 && c[2] > 0.8);
        // A slightly larger crop would poke out.
        let bigger = Geometry {
            angle_deg: 5.0,
            crop: Some([c[0] - 0.01, c[1] - 0.01, c[2] + 0.02, c[3] + 0.02]),
        };
        let l2 = layout((3000, 2000), Orientation::Normal, (0.0, 0.0), &bigger);
        let outside = [
            (l2.crop[0], l2.crop[1]),
            (l2.crop[0] + l2.crop[2], l2.crop[1]),
            (l2.crop[0], l2.crop[1] + l2.crop[3]),
            (l2.crop[0] + l2.crop[2], l2.crop[1] + l2.crop[3]),
        ]
        .into_iter()
        .map(|v| l2.source(v))
        .any(|(sx, sy)| !(0.0..=3000.0).contains(&sx) || !(0.0..=2000.0).contains(&sy));
        assert!(outside);
    }
}
