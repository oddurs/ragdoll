//! **`SIMD-0048`**: Fused L1 Compositional Video Filters: `crop` (`--auto-crop`), `pad`,
//! `rotate` (`90`/`180`/`270`/`hflip`/`vflip`), SIMD alpha-blended `overlay` (`RGBA8` over
//! `YUV420p`/`RGB24`), and perceptual Oklab HDR-to-SDR `tonemap` (`BT.2020 PQ`/`HLG` $\to$ `BT.709`).

#![allow(clippy::needless_range_loop, clippy::excessive_precision)]

use ragdoll_core::{
    ColorPrimaries, ColorRange, Colorimetry, MatrixCoefficients, PixelFormat, RagdollError, Result,
    TransferCharacteristics, VideoFrame,
};

use crate::color::convert_frame;

/// Rectangular crop region `(width, height, x, y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub width: usize,
    pub height: usize,
    pub x: usize,
    pub y: usize,
}

impl CropRect {
    /// Parses `"W:H:X:Y"` or `"W:H"` (centered when `X:Y` omitted).
    pub fn parse_spec(spec: &str, in_w: usize, in_h: usize) -> Result<Self> {
        let parts: Vec<&str> = spec.split(':').collect();
        match parts.as_slice() {
            [w_s, h_s] => {
                let w: usize = w_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid crop width in '{spec}'"))
                })?;
                let h: usize = h_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid crop height in '{spec}'"))
                })?;
                let x = in_w.saturating_sub(w) / 2;
                let y = in_h.saturating_sub(h) / 2;
                Ok(Self {
                    width: w,
                    height: h,
                    x,
                    y,
                })
            }
            [w_s, h_s, x_s, y_s] => {
                let w: usize = w_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid crop width in '{spec}'"))
                })?;
                let h: usize = h_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid crop height in '{spec}'"))
                })?;
                let x: usize = x_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid crop x in '{spec}'"))
                })?;
                let y: usize = y_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid crop y in '{spec}'"))
                })?;
                Ok(Self {
                    width: w,
                    height: h,
                    x,
                    y,
                })
            }
            _ => Err(RagdollError::InvalidHeader(format!(
                "expected crop spec 'W:H:X:Y' or 'W:H', got '{spec}'"
            ))),
        }
    }
}

/// Detects letterbox (top/bottom) and pillarbox (left/right) black bars using `luma_threshold`
/// (e.g. `20` for limited-range BT.709 `Y=16` black), returning an even-aligned [`CropRect`].
pub fn detect_auto_crop(frame: &VideoFrame, luma_threshold: u8) -> CropRect {
    let w = frame.width;
    let h = frame.height;
    if w < 4 || h < 4 {
        return CropRect {
            width: w,
            height: h,
            x: 0,
            y: 0,
        };
    }

    let row_is_black = |y: usize| -> bool {
        match frame.format {
            PixelFormat::Rgb24 => {
                let row = frame.plane(0).row(y);
                for x in 0..w {
                    let r = row[x * 3];
                    let g = row[x * 3 + 1];
                    let b = row[x * 3 + 2];
                    if r > luma_threshold || g > luma_threshold || b > luma_threshold {
                        return false;
                    }
                }
                true
            }
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
                let row = frame.plane(0).row(y);
                for x in 0..w {
                    let c0 = row[x * 4];
                    let c1 = row[x * 4 + 1];
                    let c2 = row[x * 4 + 2];
                    if c0 > luma_threshold || c1 > luma_threshold || c2 > luma_threshold {
                        return false;
                    }
                }
                true
            }
            _ => {
                let row = &frame.plane(0).row(y)[..w];
                row.iter().all(|&yv| yv <= luma_threshold)
            }
        }
    };

    let col_is_black = |x: usize, top: usize, bottom: usize| -> bool {
        match frame.format {
            PixelFormat::Rgb24 => {
                for y in top..bottom {
                    let row = frame.plane(0).row(y);
                    if row[x * 3] > luma_threshold
                        || row[x * 3 + 1] > luma_threshold
                        || row[x * 3 + 2] > luma_threshold
                    {
                        return false;
                    }
                }
                true
            }
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
                for y in top..bottom {
                    let row = frame.plane(0).row(y);
                    if row[x * 4] > luma_threshold
                        || row[x * 4 + 1] > luma_threshold
                        || row[x * 4 + 2] > luma_threshold
                    {
                        return false;
                    }
                }
                true
            }
            _ => {
                for y in top..bottom {
                    if frame.plane(0).row(y)[x] > luma_threshold {
                        return false;
                    }
                }
                true
            }
        }
    };

    let mut top = 0usize;
    while top + 2 < h && row_is_black(top) {
        top += 1;
    }
    let mut bottom = h;
    while bottom > top + 2 && row_is_black(bottom - 1) {
        bottom -= 1;
    }

    let mut left = 0usize;
    while left + 2 < w && col_is_black(left, top, bottom) {
        left += 1;
    }
    let mut right = w;
    while right > left + 2 && col_is_black(right - 1, top, bottom) {
        right -= 1;
    }

    // Align to even boundaries so 4:2:0 chroma subsampling stays phase-aligned
    let aligned_x = (left + 1) & !1;
    let aligned_y = (top + 1) & !1;
    let aligned_right = right & !1;
    let aligned_bottom = bottom & !1;

    let crop_w = aligned_right.saturating_sub(aligned_x).max(2);
    let crop_h = aligned_bottom.saturating_sub(aligned_y).max(2);

    CropRect {
        width: crop_w.min(w),
        height: crop_h.min(h),
        x: aligned_x.min(w.saturating_sub(crop_w)),
        y: aligned_y.min(h.saturating_sub(crop_h)),
    }
}

/// Crops `frame` to `rect`, enforcing `2x2` even alignment on chroma-subsampled formats (`Yuv420p`).
pub fn crop_frame(frame: &VideoFrame, rect: CropRect) -> Result<VideoFrame> {
    let subsampled = matches!(
        frame.format,
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Nv12
    );
    let x = if subsampled { rect.x & !1 } else { rect.x };
    let y = if subsampled { rect.y & !1 } else { rect.y };
    let w = if subsampled {
        rect.width & !1
    } else {
        rect.width
    };
    let h = if subsampled {
        rect.height & !1
    } else {
        rect.height
    };

    if w == 0 || h == 0 || x + w > frame.width || y + h > frame.height {
        return Err(RagdollError::InvalidHeader(format!(
            "crop rect {}x{}+{}+{} exceeds frame {}x{}",
            w, h, x, y, frame.width, frame.height
        )));
    }

    let mut out = VideoFrame::new(w, h, frame.format);
    out.pts = frame.pts;
    out.duration = frame.duration;
    out.colorimetry = frame.colorimetry;
    out.keyframe = frame.keyframe;

    match frame.format {
        PixelFormat::Yuv420p => {
            for py in 0..h {
                out.plane_mut(0).row_mut(py)[..w]
                    .copy_from_slice(&frame.plane(0).row(y + py)[x..x + w]);
            }
            let cw = w / 2;
            let ch = h / 2;
            let cx = x / 2;
            let cy = y / 2;
            for p in 1..=2 {
                for py in 0..ch {
                    out.plane_mut(p).row_mut(py)[..cw]
                        .copy_from_slice(&frame.plane(p).row(cy + py)[cx..cx + cw]);
                }
            }
        }
        PixelFormat::Rgb24 => {
            for py in 0..h {
                out.plane_mut(0).row_mut(py)[..w * 3]
                    .copy_from_slice(&frame.plane(0).row(y + py)[x * 3..(x + w) * 3]);
            }
        }
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
            for py in 0..h {
                out.plane_mut(0).row_mut(py)[..w * 4]
                    .copy_from_slice(&frame.plane(0).row(y + py)[x * 4..(x + w) * 4]);
            }
        }
        PixelFormat::Gray8 => {
            for py in 0..h {
                out.plane_mut(0).row_mut(py)[..w]
                    .copy_from_slice(&frame.plane(0).row(y + py)[x..x + w]);
            }
        }
        _ => {
            let rgb = convert_frame(frame, PixelFormat::Rgb24, frame.colorimetry);
            let cropped_rgb = crop_frame(&rgb, rect)?;
            return Ok(convert_frame(&cropped_rgb, frame.format, frame.colorimetry));
        }
    }

    Ok(out)
}

/// Padding specification `W:H:X:Y[:HEX_COLOR]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadSpec {
    pub width: usize,
    pub height: usize,
    pub x: usize,
    pub y: usize,
    pub color_rgb: [u8; 3],
}

impl PadSpec {
    /// Parses `"W:H:X:Y"`, `"W:H:X:Y:#RRGGBB"`, or `"W:H"` (centered black pad when `X:Y` omitted).
    pub fn parse_spec(spec: &str, in_w: usize, in_h: usize) -> Result<Self> {
        let parts: Vec<&str> = spec.split(':').collect();
        match parts.as_slice() {
            [w_s, h_s] => {
                let w: usize = w_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid pad width in '{spec}'"))
                })?;
                let h: usize = h_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid pad height in '{spec}'"))
                })?;
                let x = (w.saturating_sub(in_w) / 2) & !1;
                let y = (h.saturating_sub(in_h) / 2) & !1;
                Ok(Self {
                    width: w,
                    height: h,
                    x,
                    y,
                    color_rgb: [0, 0, 0],
                })
            }
            [w_s, h_s, x_s, y_s] => {
                let w: usize = w_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid pad width in '{spec}'"))
                })?;
                let h: usize = h_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid pad height in '{spec}'"))
                })?;
                let x: usize = x_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid pad x in '{spec}'"))
                })?;
                let y: usize = y_s.parse().map_err(|_| {
                    RagdollError::InvalidHeader(format!("invalid pad y in '{spec}'"))
                })?;
                Ok(Self {
                    width: w,
                    height: h,
                    x,
                    y,
                    color_rgb: [0, 0, 0],
                })
            }
            [w_s, h_s, x_s, y_s, col_s] => {
                let mut base = Self::parse_spec(&format!("{w_s}:{h_s}:{x_s}:{y_s}"), in_w, in_h)?;
                let hex = col_s.trim_start_matches('#').trim_start_matches("0x");
                if hex.len() == 6
                    && let (Ok(r), Ok(g), Ok(b)) = (
                        u8::from_str_radix(&hex[0..2], 16),
                        u8::from_str_radix(&hex[2..4], 16),
                        u8::from_str_radix(&hex[4..6], 16),
                    )
                {
                    base.color_rgb = [r, g, b];
                }
                Ok(base)
            }
            _ => Err(RagdollError::InvalidHeader(format!(
                "expected pad spec 'W:H:X:Y[:COLOR]' or 'W:H', got '{spec}'"
            ))),
        }
    }
}

#[inline]
fn rgb_to_yuv_limited_bt709(rgb: [u8; 3]) -> (u8, u8, u8) {
    let r = rgb[0] as i32;
    let g = rgb[1] as i32;
    let b = rgb[2] as i32;
    let y = ((47 * r + 157 * g + 16 * b + 128) >> 8) + 16;
    let u = ((-26 * r - 87 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 102 * g - 10 * b + 128) >> 8) + 128;
    (
        y.clamp(16, 235) as u8,
        u.clamp(16, 240) as u8,
        v.clamp(16, 240) as u8,
    )
}

/// Pads `frame` onto a larger canvas specified by [`PadSpec`].
pub fn pad_frame(frame: &VideoFrame, spec: PadSpec) -> Result<VideoFrame> {
    let subsampled = matches!(
        frame.format,
        PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Nv12
    );
    let x = if subsampled { spec.x & !1 } else { spec.x };
    let y = if subsampled { spec.y & !1 } else { spec.y };
    let w = if subsampled {
        spec.width & !1
    } else {
        spec.width
    };
    let h = if subsampled {
        spec.height & !1
    } else {
        spec.height
    };

    if x + frame.width > w || y + frame.height > h {
        return Err(RagdollError::InvalidHeader(format!(
            "pad canvas {}x{} cannot fit {}x{} at +{}+{}",
            w, h, frame.width, frame.height, x, y
        )));
    }

    let mut out = VideoFrame::new(w, h, frame.format);
    out.pts = frame.pts;
    out.duration = frame.duration;
    out.colorimetry = frame.colorimetry;
    out.keyframe = frame.keyframe;

    match frame.format {
        PixelFormat::Yuv420p => {
            let (yv, uv, vv) = rgb_to_yuv_limited_bt709(spec.color_rgb);
            for py in 0..h {
                out.plane_mut(0).row_mut(py)[..w].fill(yv);
            }
            let cw = w / 2;
            let ch = h / 2;
            for py in 0..ch {
                out.plane_mut(1).row_mut(py)[..cw].fill(uv);
                out.plane_mut(2).row_mut(py)[..cw].fill(vv);
            }
            for py in 0..frame.height {
                out.plane_mut(0).row_mut(y + py)[x..x + frame.width]
                    .copy_from_slice(&frame.plane(0).row(py)[..frame.width]);
            }
            let src_cw = frame.width / 2;
            let src_ch = frame.height / 2;
            let cx = x / 2;
            let cy = y / 2;
            for p in 1..=2 {
                for py in 0..src_ch {
                    out.plane_mut(p).row_mut(cy + py)[cx..cx + src_cw]
                        .copy_from_slice(&frame.plane(p).row(py)[..src_cw]);
                }
            }
        }
        PixelFormat::Rgb24 => {
            for py in 0..h {
                let row = out.plane_mut(0).row_mut(py);
                for px in 0..w {
                    row[px * 3] = spec.color_rgb[0];
                    row[px * 3 + 1] = spec.color_rgb[1];
                    row[px * 3 + 2] = spec.color_rgb[2];
                }
            }
            for py in 0..frame.height {
                out.plane_mut(0).row_mut(y + py)[x * 3..(x + frame.width) * 3]
                    .copy_from_slice(&frame.plane(0).row(py)[..frame.width * 3]);
            }
        }
        PixelFormat::Rgba8 => {
            for py in 0..h {
                let row = out.plane_mut(0).row_mut(py);
                for px in 0..w {
                    row[px * 4] = spec.color_rgb[0];
                    row[px * 4 + 1] = spec.color_rgb[1];
                    row[px * 4 + 2] = spec.color_rgb[2];
                    row[px * 4 + 3] = 255;
                }
            }
            for py in 0..frame.height {
                out.plane_mut(0).row_mut(y + py)[x * 4..(x + frame.width) * 4]
                    .copy_from_slice(&frame.plane(0).row(py)[..frame.width * 4]);
            }
        }
        PixelFormat::Gray8 => {
            let (yv, _, _) = rgb_to_yuv_limited_bt709(spec.color_rgb);
            for py in 0..h {
                out.plane_mut(0).row_mut(py)[..w].fill(yv);
            }
            for py in 0..frame.height {
                out.plane_mut(0).row_mut(y + py)[x..x + frame.width]
                    .copy_from_slice(&frame.plane(0).row(py)[..frame.width]);
            }
        }
        _ => {
            let rgb = convert_frame(frame, PixelFormat::Rgb24, frame.colorimetry);
            let padded_rgb = pad_frame(&rgb, spec)?;
            return Ok(convert_frame(&padded_rgb, frame.format, frame.colorimetry));
        }
    }

    Ok(out)
}

/// Orthogonal rotation and reflection modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationMode {
    Rotate90,
    Rotate180,
    Rotate270,
    HFlip,
    VFlip,
}

impl RotationMode {
    pub fn parse_spec(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "90" | "cw" | "90cw" => Ok(Self::Rotate90),
            "180" => Ok(Self::Rotate180),
            "270" | "ccw" | "90ccw" => Ok(Self::Rotate270),
            "hflip" | "mirror" => Ok(Self::HFlip),
            "vflip" | "flip" => Ok(Self::VFlip),
            other => Err(RagdollError::InvalidHeader(format!(
                "unsupported rotation mode '{other}' (expected 90, 180, 270, hflip, or vflip)"
            ))),
        }
    }
}

fn rotate_plane_bytes(
    src: &ragdoll_core::Plane,
    dst: &mut ragdoll_core::Plane,
    w: usize,
    h: usize,
    bpp: usize,
    mode: RotationMode,
) {
    match mode {
        RotationMode::VFlip => {
            let row_bytes = w * bpp;
            for y in 0..h {
                dst.row_mut(h - 1 - y)[..row_bytes].copy_from_slice(&src.row(y)[..row_bytes]);
            }
        }
        RotationMode::HFlip => {
            for y in 0..h {
                let srow = src.row(y);
                let drow = dst.row_mut(y);
                for x in 0..w {
                    let dx = w - 1 - x;
                    drow[dx * bpp..(dx + 1) * bpp].copy_from_slice(&srow[x * bpp..(x + 1) * bpp]);
                }
            }
        }
        RotationMode::Rotate180 => {
            for y in 0..h {
                let srow = src.row(y);
                let drow = dst.row_mut(h - 1 - y);
                for x in 0..w {
                    let dx = w - 1 - x;
                    drow[dx * bpp..(dx + 1) * bpp].copy_from_slice(&srow[x * bpp..(x + 1) * bpp]);
                }
            }
        }
        RotationMode::Rotate90 => {
            // (x, y) in WxH -> (h - 1 - y, x) in HxW
            for y in 0..h {
                let srow = src.row(y);
                let dx = h - 1 - y;
                for x in 0..w {
                    let drow = dst.row_mut(x);
                    drow[dx * bpp..(dx + 1) * bpp].copy_from_slice(&srow[x * bpp..(x + 1) * bpp]);
                }
            }
        }
        RotationMode::Rotate270 => {
            // (x, y) in WxH -> (y, w - 1 - x) in HxW
            for y in 0..h {
                let srow = src.row(y);
                let dx = y;
                for x in 0..w {
                    let dy = w - 1 - x;
                    let drow = dst.row_mut(dy);
                    drow[dx * bpp..(dx + 1) * bpp].copy_from_slice(&srow[x * bpp..(x + 1) * bpp]);
                }
            }
        }
    }
}

/// Rotates or flips `frame` according to [`RotationMode`] (`90`, `180`, `270`, `hflip`, `vflip`).
pub fn rotate_frame(frame: &VideoFrame, mode: RotationMode) -> VideoFrame {
    let (out_w, out_h) = match mode {
        RotationMode::Rotate90 | RotationMode::Rotate270 => (frame.height, frame.width),
        RotationMode::Rotate180 | RotationMode::HFlip | RotationMode::VFlip => {
            (frame.width, frame.height)
        }
    };
    let mut out = VideoFrame::new(out_w, out_h, frame.format);
    out.pts = frame.pts;
    out.duration = frame.duration;
    out.colorimetry = frame.colorimetry;
    out.keyframe = frame.keyframe;

    match frame.format {
        PixelFormat::Yuv420p => {
            rotate_plane_bytes(
                frame.plane(0),
                out.plane_mut(0),
                frame.width,
                frame.height,
                1,
                mode,
            );
            let cw = frame.width.div_ceil(2);
            let ch = frame.height.div_ceil(2);
            rotate_plane_bytes(frame.plane(1), out.plane_mut(1), cw, ch, 1, mode);
            rotate_plane_bytes(frame.plane(2), out.plane_mut(2), cw, ch, 1, mode);
        }
        PixelFormat::Rgb24 => {
            rotate_plane_bytes(
                frame.plane(0),
                out.plane_mut(0),
                frame.width,
                frame.height,
                3,
                mode,
            );
        }
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
            rotate_plane_bytes(
                frame.plane(0),
                out.plane_mut(0),
                frame.width,
                frame.height,
                4,
                mode,
            );
        }
        PixelFormat::Gray8 => {
            rotate_plane_bytes(
                frame.plane(0),
                out.plane_mut(0),
                frame.width,
                frame.height,
                1,
                mode,
            );
        }
        _ => {
            let rgb = convert_frame(frame, PixelFormat::Rgb24, frame.colorimetry);
            let rot_rgb = rotate_frame(&rgb, mode);
            return convert_frame(&rot_rgb, frame.format, frame.colorimetry);
        }
    }

    out
}

/// Exact division by 255 for `t` in `0..=65025`: `((t + 128) + ((t + 128) >> 8)) >> 8 == (t + 127) / 255`.
#[inline(always)]
fn div255_round(t: u16) -> u8 {
    let x = t + 128;
    ((x + (x >> 8)) >> 8) as u8
}

/// Scalar reference row kernel for alpha-blending `width` pixels of `Rgba8` (`src_rgba`) over `Rgb24` (`dst_rgb`).
pub fn overlay_rgba_row_onto_rgb24_scalar(dst_rgb: &mut [u8], src_rgba: &[u8], width: usize) {
    for x in 0..width {
        let sr = src_rgba[x * 4] as u16;
        let sg = src_rgba[x * 4 + 1] as u16;
        let sb = src_rgba[x * 4 + 2] as u16;
        let sa = src_rgba[x * 4 + 3] as u16;
        let inv_a = 255 - sa;

        let dr = dst_rgb[x * 3] as u16;
        let dg = dst_rgb[x * 3 + 1] as u16;
        let db = dst_rgb[x * 3 + 2] as u16;

        dst_rgb[x * 3] = div255_round(sr * sa + dr * inv_a);
        dst_rgb[x * 3 + 1] = div255_round(sg * sa + dg * inv_a);
        dst_rgb[x * 3 + 2] = div255_round(sb * sa + db * inv_a);
    }
}

/// SIMD-accelerated (`aarch64` NEON `vld4_u8` / `vld3_u8` / `vmull_u8` / `vmlal_u8`) row kernel for
/// alpha-blending `width` pixels of `Rgba8` (`src_rgba`) over `Rgb24` (`dst_rgb`), matching
/// [`overlay_rgba_row_onto_rgb24_scalar`] bit-for-bit across all alpha values (`0..=255`) and odd widths.
pub fn overlay_rgba_row_onto_rgb24(dst_rgb: &mut [u8], src_rgba: &[u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut x = 0usize;
        let v255 = vdup_n_u8(255);
        let v128 = vdupq_n_u16(128);
        while x + 8 <= width {
            let s = vld4_u8(src_rgba.as_ptr().add(x * 4));
            let mut d = vld3_u8(dst_rgb.as_ptr().add(x * 3));
            let sa = s.3;
            let inv_a = vsub_u8(v255, sa);

            // R
            let tr = vmlal_u8(vmull_u8(s.0, sa), d.0, inv_a);
            let xr = vaddq_u16(tr, v128);
            d.0 = vshrn_n_u16(vaddq_u16(xr, vshrq_n_u16(xr, 8)), 8);

            // G
            let tg = vmlal_u8(vmull_u8(s.1, sa), d.1, inv_a);
            let xg = vaddq_u16(tg, v128);
            d.1 = vshrn_n_u16(vaddq_u16(xg, vshrq_n_u16(xg, 8)), 8);

            // B
            let tb = vmlal_u8(vmull_u8(s.2, sa), d.2, inv_a);
            let xb = vaddq_u16(tb, v128);
            d.2 = vshrn_n_u16(vaddq_u16(xb, vshrq_n_u16(xb, 8)), 8);

            vst3_u8(dst_rgb.as_mut_ptr().add(x * 3), d);
            x += 8;
        }
        if x < width {
            overlay_rgba_row_onto_rgb24_scalar(
                &mut dst_rgb[x * 3..],
                &src_rgba[x * 4..],
                width - x,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        overlay_rgba_row_onto_rgb24_scalar(dst_rgb, src_rgba, width);
    }
}

/// Scalar reference row kernel for alpha-blending `width` samples of planar luma/chroma (`src`, `alpha`) onto `dst`.
pub fn overlay_plane_row_u8_scalar(dst: &mut [u8], src: &[u8], alpha: &[u8], width: usize) {
    for x in 0..width {
        let s = src[x] as u16;
        let a = alpha[x] as u16;
        let d = dst[x] as u16;
        dst[x] = div255_round(s * a + d * (255 - a));
    }
}

/// SIMD-accelerated (`aarch64` NEON) planar alpha-blending row kernel matching [`overlay_plane_row_u8_scalar`] bit-for-bit.
pub fn overlay_plane_row_u8(dst: &mut [u8], src: &[u8], alpha: &[u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut x = 0usize;
        let v255 = vdup_n_u8(255);
        let v128 = vdupq_n_u16(128);
        while x + 8 <= width {
            let vs = vld1_u8(src.as_ptr().add(x));
            let va = vld1_u8(alpha.as_ptr().add(x));
            let vd = vld1_u8(dst.as_ptr().add(x));
            let inv_a = vsub_u8(v255, va);
            let t = vmlal_u8(vmull_u8(vs, va), vd, inv_a);
            let x16 = vaddq_u16(t, v128);
            let out = vshrn_n_u16(vaddq_u16(x16, vshrq_n_u16(x16, 8)), 8);
            vst1_u8(dst.as_mut_ptr().add(x), out);
            x += 8;
        }
        if x < width {
            overlay_plane_row_u8_scalar(&mut dst[x..], &src[x..], &alpha[x..], width - x);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        overlay_plane_row_u8_scalar(dst, src, alpha, width);
    }
}

/// Alpha-blends `overlay` (`Rgba8` or converted to `Rgba8`) onto `base` (`Yuv420p` or `Rgb24`) at `(x_offset, y_offset)`.
pub fn overlay_rgba_onto_frame(
    base: &mut VideoFrame,
    overlay: &VideoFrame,
    x_offset: usize,
    y_offset: usize,
) {
    overlay_rgba_onto_frame_impl(base, overlay, x_offset, y_offset, true);
}

/// Scalar reference for [`overlay_rgba_onto_frame`].
pub fn overlay_rgba_onto_frame_scalar(
    base: &mut VideoFrame,
    overlay: &VideoFrame,
    x_offset: usize,
    y_offset: usize,
) {
    overlay_rgba_onto_frame_impl(base, overlay, x_offset, y_offset, false);
}

fn overlay_rgba_onto_frame_impl(
    base: &mut VideoFrame,
    overlay: &VideoFrame,
    x_offset: usize,
    y_offset: usize,
    use_simd: bool,
) {
    if x_offset >= base.width || y_offset >= base.height {
        return;
    }
    let converted_ov;
    let ov_rgba = if overlay.format == PixelFormat::Rgba8 {
        overlay
    } else {
        converted_ov = convert_frame(overlay, PixelFormat::Rgba8, overlay.colorimetry);
        &converted_ov
    };

    let blend_w = ov_rgba.width.min(base.width - x_offset);
    let blend_h = ov_rgba.height.min(base.height - y_offset);
    if blend_w == 0 || blend_h == 0 {
        return;
    }

    match base.format {
        PixelFormat::Rgb24 => {
            for y in 0..blend_h {
                let dst_row = &mut base.plane_mut(0).row_mut(y_offset + y)
                    [x_offset * 3..(x_offset + blend_w) * 3];
                let src_row = &ov_rgba.plane(0).row(y)[..blend_w * 4];
                if use_simd {
                    overlay_rgba_row_onto_rgb24(dst_row, src_row, blend_w);
                } else {
                    overlay_rgba_row_onto_rgb24_scalar(dst_row, src_row, blend_w);
                }
            }
        }
        PixelFormat::Yuv420p => {
            let x0 = x_offset & !1;
            let y0 = y_offset & !1;
            let bw = blend_w.min(base.width - x0);
            let bh = blend_h.min(base.height - y0);
            let mut y_src = vec![16u8; bw];
            let mut a_src = vec![0u8; bw];

            for y in 0..bh {
                let ov_row = ov_rgba.plane(0).row(y);
                for x in 0..bw {
                    let r = ov_row[x * 4];
                    let g = ov_row[x * 4 + 1];
                    let b = ov_row[x * 4 + 2];
                    let (yv, _, _) = rgb_to_yuv_limited_bt709([r, g, b]);
                    y_src[x] = yv;
                    a_src[x] = ov_row[x * 4 + 3];
                }
                let dst_y = &mut base.plane_mut(0).row_mut(y0 + y)[x0..x0 + bw];
                if use_simd {
                    overlay_plane_row_u8(dst_y, &y_src, &a_src, bw);
                } else {
                    overlay_plane_row_u8_scalar(dst_y, &y_src, &a_src, bw);
                }
            }

            let cw = bw / 2;
            let ch = bh / 2;
            let cx0 = x0 / 2;
            let cy0 = y0 / 2;
            let mut u_src = vec![128u8; cw];
            let mut v_src = vec![128u8; cw];
            let mut ca_src = vec![0u8; cw];
            for cy in 0..ch {
                let row0 = ov_rgba.plane(0).row(cy * 2);
                let row1 = ov_rgba.plane(0).row((cy * 2 + 1).min(bh - 1));
                for cx in 0..cw {
                    let x = cx * 2;
                    let r = ((row0[x * 4] as u16
                        + row0[(x + 1) * 4] as u16
                        + row1[x * 4] as u16
                        + row1[(x + 1) * 4] as u16
                        + 2)
                        >> 2) as u8;
                    let g = ((row0[x * 4 + 1] as u16
                        + row0[(x + 1) * 4 + 1] as u16
                        + row1[x * 4 + 1] as u16
                        + row1[(x + 1) * 4 + 1] as u16
                        + 2)
                        >> 2) as u8;
                    let b = ((row0[x * 4 + 2] as u16
                        + row0[(x + 1) * 4 + 2] as u16
                        + row1[x * 4 + 2] as u16
                        + row1[(x + 1) * 4 + 2] as u16
                        + 2)
                        >> 2) as u8;
                    let a = ((row0[x * 4 + 3] as u16
                        + row0[(x + 1) * 4 + 3] as u16
                        + row1[x * 4 + 3] as u16
                        + row1[(x + 1) * 4 + 3] as u16
                        + 2)
                        >> 2) as u8;
                    let (_, uv, vv) = rgb_to_yuv_limited_bt709([r, g, b]);
                    u_src[cx] = uv;
                    v_src[cx] = vv;
                    ca_src[cx] = a;
                }
                let dst_u = &mut base.plane_mut(1).row_mut(cy0 + cy)[cx0..cx0 + cw];
                if use_simd {
                    overlay_plane_row_u8(dst_u, &u_src, &ca_src, cw);
                } else {
                    overlay_plane_row_u8_scalar(dst_u, &u_src, &ca_src, cw);
                }
                let dst_v = &mut base.plane_mut(2).row_mut(cy0 + cy)[cx0..cx0 + cw];
                if use_simd {
                    overlay_plane_row_u8(dst_v, &v_src, &ca_src, cw);
                } else {
                    overlay_plane_row_u8_scalar(dst_v, &v_src, &ca_src, cw);
                }
            }
        }
        other => {
            let mut rgb = convert_frame(base, PixelFormat::Rgb24, base.colorimetry);
            overlay_rgba_onto_frame_impl(&mut rgb, ov_rgba, x_offset, y_offset, use_simd);
            *base = convert_frame(&rgb, other, base.colorimetry);
        }
    }
}

/// SMPTE ST 2084 (PQ) EOTF: converts non-linear PQ signal $E' \in [0, 1]$ to normalized linear light
/// where $1.0 = 203\text{ cd/m}^2$ SDR reference white (and $49.26 = 10{,}000\text{ cd/m}^2$).
#[inline]
pub fn pq_eotf_normalized_203nits(e_prime: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;

    let ep = e_prime.clamp(0.0, 1.0);
    let ep_inv_m2 = ep.powf(1.0 / M2);
    let num = (ep_inv_m2 - C1).max(0.0);
    let den = (C2 - C3 * ep_inv_m2).max(1e-6);
    let y_10000 = (num / den).powf(1.0 / M1);
    // Scale 10,000 nits peak to 203 nits ITU-R BT.2408 SDR reference white
    y_10000 * (10_000.0 / 203.0)
}

/// ARIB STD-B67 (HLG) inverse OETF + system gamma ($\gamma = 1.2$), normalized so diffuse white = $1.0$.
#[inline]
pub fn hlg_eotf_normalized(e_prime: f32) -> f32 {
    const A: f32 = 0.17883277;
    const B: f32 = 1.0 - 4.0 * A;
    const C: f32 = 0.5599107; // 0.5 - A * ln(4 * A)
    let e = e_prime.clamp(0.0, 1.0);
    let scene = if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - C) / A).exp() + B) / 12.0
    };
    (scene * 4.0).powf(1.2)
}

#[inline]
fn srgb_oetf_u8(lin: f32) -> u8 {
    let v = lin.clamp(0.0, 1.0);
    let encoded = if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Scalar reference row kernel for perceptual Oklab HDR-to-SDR tone-mapping from linear BT.2020
/// RGB triplets (`hdr_bt2020_rgb`, normalized to $1.0 = 203\text{ nits}$) into BT.709 `Rgb24`.
pub fn tonemap_linear_bt2020_row_oklab_scalar(
    hdr_bt2020_rgb: &[f32],
    dst_rgb24: &mut [u8],
    width: usize,
) {
    for x in 0..width {
        let r2020 = hdr_bt2020_rgb[x * 3].max(0.0);
        let g2020 = hdr_bt2020_rgb[x * 3 + 1].max(0.0);
        let b2020 = hdr_bt2020_rgb[x * 3 + 2].max(0.0);

        // 1. Linear BT.2020 -> Oklab LMS cone response (direct BT.2020 -> M1 * XYZ_2020)
        let l = 0.6167597 * r2020 + 0.3601880 * g2020 + 0.0230523 * b2020;
        let m = 0.2651317 * r2020 + 0.6358516 * g2020 + 0.0990167 * b2020;
        let s = 0.1001279 * r2020 + 0.2038784 * g2020 + 0.6959937 * b2020;

        let l_ = l.max(0.0).cbrt();
        let m_ = m.max(0.0).cbrt();
        let s_ = s.max(0.0).cbrt();

        let ok_l = 0.21045426 * l_ + 0.7936178 * m_ - 0.004072047 * s_;
        let ok_a = 1.9779985 * l_ - 2.4285922 * m_ + 0.4505937 * s_;
        let ok_b = 0.025904037 * l_ + 0.78277177 * m_ - 0.80867577 * s_;

        // 2. Perceptual Oklab tone-mapping curve on L with strict hue-angle preservation
        //    Peak L_max ~= (10000 / 203)^(1/3) = 3.6657
        let l_src = ok_l.max(1e-6);
        const L_WHITE: f32 = 3.6657;
        let l_mapped = (l_src * (1.0 + l_src / (L_WHITE * L_WHITE))) / (1.0 + l_src);
        let chroma_scale =
            (l_mapped / l_src).clamp(0.0, 1.0) * (1.0 - 0.25 * (l_mapped.clamp(0.0, 1.0)).powi(2));
        let a_mapped = ok_a * chroma_scale;
        let b_mapped = ok_b * chroma_scale;

        // 3. Inverse Oklab -> Linear BT.709 / sRGB
        let lm = l_mapped + 0.39633778 * a_mapped + 0.21580376 * b_mapped;
        let mm = l_mapped - 0.105561346 * a_mapped - 0.06385417 * b_mapped;
        let sm = l_mapped - 0.08948418 * a_mapped - 1.2914855 * b_mapped;

        let l_lin = lm * lm * lm;
        let m_lin = mm * mm * mm;
        let s_lin = sm * sm * sm;

        let r709 = 4.0767417 * l_lin - 3.3077116 * m_lin + 0.23096993 * s_lin;
        let g709 = -1.268438 * l_lin + 2.6097574 * m_lin - 0.34131938 * s_lin;
        let b709 = -0.0041960863 * l_lin - 0.7034186 * m_lin + 1.7076147 * s_lin;

        dst_rgb24[x * 3] = srgb_oetf_u8(r709);
        dst_rgb24[x * 3 + 1] = srgb_oetf_u8(g709);
        dst_rgb24[x * 3 + 2] = srgb_oetf_u8(b709);
    }
}

/// SIMD-accelerated (`aarch64` NEON `float32x4_t`) perceptual Oklab HDR-to-SDR row kernel.
pub fn tonemap_linear_bt2020_row_oklab(hdr_bt2020_rgb: &[f32], dst_rgb24: &mut [u8], width: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut x = 0usize;
        let v_zero = vdupq_n_f32(0.0);
        let v_eps = vdupq_n_f32(1e-6);
        let v_one = vdupq_n_f32(1.0);
        let v_inv_w2 = vdupq_n_f32(1.0 / (3.6657 * 3.6657));
        let v_025 = vdupq_n_f32(0.25);

        while x + 4 <= width {
            let rgb = vld3q_f32(hdr_bt2020_rgb.as_ptr().add(x * 3));
            let r = vmaxq_f32(rgb.0, v_zero);
            let g = vmaxq_f32(rgb.1, v_zero);
            let b = vmaxq_f32(rgb.2, v_zero);

            let l = vmlaq_n_f32(
                vmlaq_n_f32(vmulq_n_f32(r, 0.6167597), g, 0.3601880),
                b,
                0.0230523,
            );
            let m = vmlaq_n_f32(
                vmlaq_n_f32(vmulq_n_f32(r, 0.2651317), g, 0.6358516),
                b,
                0.0990167,
            );
            let s = vmlaq_n_f32(
                vmlaq_n_f32(vmulq_n_f32(r, 0.1001279), g, 0.2038784),
                b,
                0.6959937,
            );

            let mut la = [0.0f32; 4];
            let mut ma = [0.0f32; 4];
            let mut sa = [0.0f32; 4];
            vst1q_f32(la.as_mut_ptr(), vmaxq_f32(l, v_zero));
            vst1q_f32(ma.as_mut_ptr(), vmaxq_f32(m, v_zero));
            vst1q_f32(sa.as_mut_ptr(), vmaxq_f32(s, v_zero));
            for i in 0..4 {
                la[i] = la[i].cbrt();
                ma[i] = ma[i].cbrt();
                sa[i] = sa[i].cbrt();
            }
            let l_ = vld1q_f32(la.as_ptr());
            let m_ = vld1q_f32(ma.as_ptr());
            let s_ = vld1q_f32(sa.as_ptr());

            let ok_l = vmlsq_n_f32(
                vmlaq_n_f32(vmulq_n_f32(l_, 0.21045426), m_, 0.7936178),
                s_,
                0.004072047,
            );
            let ok_a = vmlaq_n_f32(
                vmlsq_n_f32(vmulq_n_f32(l_, 1.9779985), m_, 2.4285922),
                s_,
                0.4505937,
            );
            let ok_b = vmlsq_n_f32(
                vmlaq_n_f32(vmulq_n_f32(l_, 0.025904037), m_, 0.78277177),
                s_,
                0.80867577,
            );

            let l_src = vmaxq_f32(ok_l, v_eps);
            let num = vmulq_f32(l_src, vmlaq_f32(v_one, l_src, v_inv_w2));
            let den = vaddq_f32(v_one, l_src);
            let l_mapped = vdivq_f32(num, den);
            let l_clamped = vminq_f32(vmaxq_f32(l_mapped, v_zero), v_one);
            let ratio = vminq_f32(vmaxq_f32(vdivq_f32(l_mapped, l_src), v_zero), v_one);
            let desat = vsubq_f32(v_one, vmulq_f32(v_025, vmulq_f32(l_clamped, l_clamped)));
            let c_scale = vmulq_f32(ratio, desat);
            let a_mapped = vmulq_f32(ok_a, c_scale);
            let b_mapped = vmulq_f32(ok_b, c_scale);

            let lm = vmlaq_n_f32(
                vmlaq_n_f32(l_mapped, a_mapped, 0.39633778),
                b_mapped,
                0.21580376,
            );
            let mm = vmlsq_n_f32(
                vmlsq_n_f32(l_mapped, a_mapped, 0.105561346),
                b_mapped,
                0.06385417,
            );
            let sm = vmlsq_n_f32(
                vmlsq_n_f32(l_mapped, a_mapped, 0.08948418),
                b_mapped,
                1.2914855,
            );

            let l_lin = vmulq_f32(vmulq_f32(lm, lm), lm);
            let m_lin = vmulq_f32(vmulq_f32(mm, mm), mm);
            let s_lin = vmulq_f32(vmulq_f32(sm, sm), sm);

            let r709 = vmlaq_n_f32(
                vmlsq_n_f32(vmulq_n_f32(l_lin, 4.0767417), m_lin, 3.3077116),
                s_lin,
                0.23096993,
            );
            let g709 = vmlsq_n_f32(
                vmlaq_n_f32(vmulq_n_f32(l_lin, -1.268438), m_lin, 2.6097574),
                s_lin,
                0.34131938,
            );
            let b709 = vmlaq_n_f32(
                vmlsq_n_f32(vmulq_n_f32(l_lin, -0.0041960863), m_lin, 0.7034186),
                s_lin,
                1.7076147,
            );

            let mut ra = [0.0f32; 4];
            let mut ga = [0.0f32; 4];
            let mut ba = [0.0f32; 4];
            vst1q_f32(ra.as_mut_ptr(), r709);
            vst1q_f32(ga.as_mut_ptr(), g709);
            vst1q_f32(ba.as_mut_ptr(), b709);
            for i in 0..4 {
                dst_rgb24[(x + i) * 3] = srgb_oetf_u8(ra[i]);
                dst_rgb24[(x + i) * 3 + 1] = srgb_oetf_u8(ga[i]);
                dst_rgb24[(x + i) * 3 + 2] = srgb_oetf_u8(ba[i]);
            }
            x += 4;
        }
        if x < width {
            tonemap_linear_bt2020_row_oklab_scalar(
                &hdr_bt2020_rgb[x * 3..],
                &mut dst_rgb24[x * 3..],
                width - x,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        tonemap_linear_bt2020_row_oklab_scalar(hdr_bt2020_rgb, dst_rgb24, width);
    }
}

/// Converts an HDR `BT.2020 PQ` (`SMPTE2084`) or `HLG` (`ARIB-STD-B67`) [`VideoFrame`]
/// (`Yuv420p10le`, `Yuv420p`, or `Rgb24`) to a `BT.709` SDR [`VideoFrame`] (`Yuv420p` or `Rgb24`)
/// via perceptual Oklab luminance compression and gamut mapping.
pub fn tonemap_hdr_to_sdr_oklab(frame: &VideoFrame, out_format: PixelFormat) -> VideoFrame {
    tonemap_hdr_to_sdr_oklab_impl(frame, out_format, true)
}

/// Scalar reference for [`tonemap_hdr_to_sdr_oklab`].
pub fn tonemap_hdr_to_sdr_oklab_scalar(frame: &VideoFrame, out_format: PixelFormat) -> VideoFrame {
    tonemap_hdr_to_sdr_oklab_impl(frame, out_format, false)
}

fn tonemap_hdr_to_sdr_oklab_impl(
    frame: &VideoFrame,
    out_format: PixelFormat,
    use_simd: bool,
) -> VideoFrame {
    let w = frame.width;
    let h = frame.height;
    let is_hlg = frame.colorimetry.transfer == TransferCharacteristics::ArbStdB67HLG;
    let eotf = |e: f32| -> f32 {
        if is_hlg {
            hlg_eotf_normalized(e)
        } else {
            pq_eotf_normalized_203nits(e)
        }
    };

    let mut sdr_rgb = VideoFrame::new(w, h, PixelFormat::Rgb24);
    sdr_rgb.pts = frame.pts;
    sdr_rgb.duration = frame.duration;
    sdr_rgb.keyframe = frame.keyframe;
    sdr_rgb.colorimetry = Colorimetry {
        primaries: ColorPrimaries::BT709,
        transfer: TransferCharacteristics::BT709,
        matrix: MatrixCoefficients::BT709,
        range: ColorRange::Limited,
        chroma_location: frame.colorimetry.chroma_location,
    };

    let mut hdr_row = vec![0.0f32; w * 3];

    for y in 0..h {
        match frame.format {
            PixelFormat::Yuv420p10le => {
                let y_row = frame.plane(0).row(y);
                let u_row = frame.plane(1).row(y / 2);
                let v_row = frame.plane(2).row(y / 2);
                for x in 0..w {
                    let y10 = u16::from_le_bytes([y_row[x * 2], y_row[x * 2 + 1]]).min(1023) as f32;
                    let cx = x / 2;
                    let u10 =
                        u16::from_le_bytes([u_row[cx * 2], u_row[cx * 2 + 1]]).min(1023) as f32;
                    let v10 =
                        u16::from_le_bytes([v_row[cx * 2], v_row[cx * 2 + 1]]).min(1023) as f32;

                    let yn = ((y10 - 64.0) / 876.0).clamp(0.0, 1.0);
                    let cb = ((u10 - 512.0) / 896.0).clamp(-0.5, 0.5);
                    let cr = ((v10 - 512.0) / 896.0).clamp(-0.5, 0.5);

                    let r_p = (yn + 1.4746 * cr).clamp(0.0, 1.0);
                    let g_p = (yn - 0.164553 * cb - 0.571353 * cr).clamp(0.0, 1.0);
                    let b_p = (yn + 1.8814 * cb).clamp(0.0, 1.0);

                    hdr_row[x * 3] = eotf(r_p);
                    hdr_row[x * 3 + 1] = eotf(g_p);
                    hdr_row[x * 3 + 2] = eotf(b_p);
                }
            }
            PixelFormat::Rgb24 => {
                let row = frame.plane(0).row(y);
                for x in 0..w {
                    hdr_row[x * 3] = eotf(row[x * 3] as f32 / 255.0);
                    hdr_row[x * 3 + 1] = eotf(row[x * 3 + 1] as f32 / 255.0);
                    hdr_row[x * 3 + 2] = eotf(row[x * 3 + 2] as f32 / 255.0);
                }
            }
            _ => {
                let y_row = frame.plane(0).row(y);
                let has_chroma = frame.planes.len() >= 3;
                for x in 0..w {
                    let y8 = y_row[x] as f32;
                    let (u8v, v8v) = if has_chroma {
                        (
                            frame.plane(1).row(y / 2)[x / 2] as f32,
                            frame.plane(2).row(y / 2)[x / 2] as f32,
                        )
                    } else {
                        (128.0, 128.0)
                    };
                    let yn = ((y8 - 16.0) / 219.0).clamp(0.0, 1.0);
                    let cb = ((u8v - 128.0) / 224.0).clamp(-0.5, 0.5);
                    let cr = ((v8v - 128.0) / 224.0).clamp(-0.5, 0.5);

                    let r_p = (yn + 1.4746 * cr).clamp(0.0, 1.0);
                    let g_p = (yn - 0.164553 * cb - 0.571353 * cr).clamp(0.0, 1.0);
                    let b_p = (yn + 1.8814 * cb).clamp(0.0, 1.0);

                    hdr_row[x * 3] = eotf(r_p);
                    hdr_row[x * 3 + 1] = eotf(g_p);
                    hdr_row[x * 3 + 2] = eotf(b_p);
                }
            }
        }

        let dst_row = &mut sdr_rgb.plane_mut(0).row_mut(y)[..w * 3];
        if use_simd {
            tonemap_linear_bt2020_row_oklab(&hdr_row, dst_row, w);
        } else {
            tonemap_linear_bt2020_row_oklab_scalar(&hdr_row, dst_row, w);
        }
    }

    if out_format == PixelFormat::Rgb24 {
        sdr_rgb
    } else {
        let mut converted = convert_frame(&sdr_rgb, out_format, sdr_rgb.colorimetry);
        converted.pts = frame.pts;
        converted.duration = frame.duration;
        converted.keyframe = frame.keyframe;
        converted
    }
}
