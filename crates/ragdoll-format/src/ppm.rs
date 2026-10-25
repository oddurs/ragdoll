use crate::io::VectoredWriter;
use ragdoll_core::{PixelFormat, RagdollError, Result, VideoFrame};
use std::io::Write;

/// Parses a binary Netpbm `P6` (`Rgb24`) or `P5` (`Gray8`) image into a 64-byte-aligned [`VideoFrame`].
///
/// Handles arbitrary whitespace and `#` comment lines anywhere in the header token stream,
/// matching the Netpbm specification and `libavcodec/pnm.c`.
pub fn read_pnm_image(data: &[u8]) -> Result<VideoFrame> {
    if data.len() < 7 || data[0] != b'P' || (data[1] != b'6' && data[1] != b'5') {
        return Err(RagdollError::InvalidHeader(
            "expected Netpbm 'P6' (PPM) or 'P5' (PGM) magic".to_string(),
        ));
    }
    let is_ppm = data[1] == b'6';
    let mut pos = 2usize;

    let next_token = |pos: &mut usize| -> Option<&[u8]> {
        while *pos < data.len() {
            let b = data[*pos];
            if b == b'#' {
                while *pos < data.len() && data[*pos] != b'\n' {
                    *pos += 1;
                }
            } else if b.is_ascii_whitespace() {
                *pos += 1;
            } else {
                break;
            }
        }
        if *pos >= data.len() {
            return None;
        }
        let start = *pos;
        while *pos < data.len() && !data[*pos].is_ascii_whitespace() && data[*pos] != b'#' {
            *pos += 1;
        }
        Some(&data[start..*pos])
    };

    let parse_u32 = |tok: &[u8]| -> Result<u32> {
        std::str::from_utf8(tok)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| RagdollError::InvalidHeader("invalid integer in PNM header".to_string()))
    };

    let w_tok = next_token(&mut pos)
        .ok_or_else(|| RagdollError::InvalidHeader("missing width in PNM header".to_string()))?;
    let h_tok = next_token(&mut pos)
        .ok_or_else(|| RagdollError::InvalidHeader("missing height in PNM header".to_string()))?;
    let maxval_tok = next_token(&mut pos)
        .ok_or_else(|| RagdollError::InvalidHeader("missing maxval in PNM header".to_string()))?;

    let width = parse_u32(w_tok)? as usize;
    let height = parse_u32(h_tok)? as usize;
    let maxval = parse_u32(maxval_tok)?;

    if width == 0 || height == 0 || maxval != 255 {
        return Err(RagdollError::InvalidHeader(format!(
            "unsupported PNM dimensions {width}x{height} or maxval {maxval} (expected 255)"
        )));
    }

    // Exactly one whitespace byte separates maxval from raw pixel payload
    if pos >= data.len() || !data[pos].is_ascii_whitespace() {
        return Err(RagdollError::InvalidHeader(
            "missing single whitespace delimiter after PNM maxval".to_string(),
        ));
    }
    pos += 1;

    let format = if is_ppm {
        PixelFormat::Rgb24
    } else {
        PixelFormat::Gray8
    };
    let row_bytes = format.plane_row_bytes(width, 0);
    let needed = row_bytes * height;
    if pos + needed > data.len() {
        return Err(RagdollError::CorruptBitstream(
            "truncated pixel payload in PNM image",
        ));
    }

    let mut frame = VideoFrame::new(width, height, format);
    let plane = &mut frame.planes[0];
    let raw = &data[pos..pos + needed];
    for y in 0..height {
        plane
            .row_mut(y)
            .copy_from_slice(&raw[y * row_bytes..(y + 1) * row_bytes]);
    }
    Ok(frame)
}

/// Writes an `Rgb24` or `Gray8` [`VideoFrame`] as a binary Netpbm `P6` (PPM) or `P5` (PGM) image.
pub fn write_pnm_image<W: Write>(writer: W, frame: &VideoFrame) -> Result<u64> {
    let magic = match frame.format {
        PixelFormat::Rgb24 => "P6",
        PixelFormat::Gray8 => "P5",
        other => {
            return Err(RagdollError::Unsupported(format!(
                "write_pnm_image supports Rgb24 (P6) and Gray8 (P5), got {other}"
            )));
        }
    };
    let mut vw = VectoredWriter::new(writer);
    let hdr = format!("{}\n{} {}\n255\n", magic, frame.width, frame.height);
    vw.write_all_counted(hdr.as_bytes())?;

    let plane = &frame.planes[0];
    let row_bytes = plane.row_bytes();
    if row_bytes == plane.stride() {
        vw.write_all_counted(&plane.buffer().as_slice()[..row_bytes * frame.height])?;
    } else {
        for y in 0..frame.height {
            vw.write_all_counted(plane.row(y))?;
        }
    }
    vw.flush()?;
    Ok(vw.bytes_written())
}

/// Composes a multi-frame `Rgb24` contact-sheet thumbnail grid with configurable columns,
/// cell padding (`padding_px`), and background color (`bg_rgb`).
pub fn compose_contact_sheet_rgb24(
    frames: &[&VideoFrame],
    cols: usize,
    padding_px: usize,
    bg_rgb: [u8; 3],
) -> Result<VideoFrame> {
    if frames.is_empty() || cols == 0 {
        return Err(RagdollError::InvalidHeader(
            "compose_contact_sheet_rgb24 requires at least 1 frame and cols >= 1".to_string(),
        ));
    }
    let tile_w = frames[0].width;
    let tile_h = frames[0].height;
    for f in frames {
        if f.format != PixelFormat::Rgb24 {
            return Err(RagdollError::Unsupported(
                "compose_contact_sheet_rgb24 requires Rgb24 input frames".to_string(),
            ));
        }
        if f.width != tile_w || f.height != tile_h {
            return Err(RagdollError::DimensionMismatch {
                expected_w: tile_w,
                expected_h: tile_h,
                actual_w: f.width,
                actual_h: f.height,
            });
        }
    }

    let rows = frames.len().div_ceil(cols);
    let out_w = cols * tile_w + (cols + 1) * padding_px;
    let out_h = rows * tile_h + (rows + 1) * padding_px;

    let mut canvas = VideoFrame::new(out_w, out_h, PixelFormat::Rgb24);
    let out_plane = &mut canvas.planes[0];

    // Fill background color
    for y in 0..out_h {
        let row = out_plane.row_mut(y);
        for px in row.as_chunks_mut::<3>().0 {
            *px = bg_rgb;
        }
    }

    // Blit each tile into its grid slot
    for (idx, frame) in frames.iter().enumerate() {
        let r = idx / cols;
        let c = idx % cols;
        let dst_x0 = padding_px + c * (tile_w + padding_px);
        let dst_y0 = padding_px + r * (tile_h + padding_px);
        let src_plane = &frame.planes[0];
        let tile_row_bytes = tile_w * 3;
        let dst_byte_x0 = dst_x0 * 3;

        for ty in 0..tile_h {
            let src_row = src_plane.row(ty);
            let dst_row = out_plane.row_mut(dst_y0 + ty);
            dst_row[dst_byte_x0..dst_byte_x0 + tile_row_bytes].copy_from_slice(src_row);
        }
    }

    Ok(canvas)
}
