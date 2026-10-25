use std::hint::black_box;
use std::time::Instant;

use ragdoll_core::{
    BitReader, BitWriter, FramePool, PixelFormat, Rational64, Timestamp, crc16_flac, crc32_mpeg2,
    escape_rbsp, optimal_plane_stride, unescape_rbsp, unescape_rbsp_scalar,
};
use ragdoll_simd::{
    SimdLevel, find_annex_b_start_code, find_annex_b_start_code_scalar, forward_dct4x4,
    forward_dct4x4_scalar, inverse_idct4x4, inverse_idct4x4_scalar, sad_plane_u8,
    sad_plane_u8_scalar, satd_8x8_u8, sse_plane_u8, sse_plane_u8_scalar, ssim_4x4xn_u8,
    ssim_4x4xn_u8_scalar, ssim_plane_8bit, ssim_to_db,
};

fn main() {
    let simd = SimdLevel::detect();
    println!("================================================================================");
    println!("  RAGDOLL v0.1.0 — M1 Foundation & SIMD Kernel Live Benchmark / Self-Test");
    println!("  Detected Hardware SIMD : {}", simd.name());
    println!("================================================================================\n");

    // 1. FramePool + Cache-Set Stride Check
    let pool = FramePool::with_memory_budget(8, 128 * 1024 * 1024);
    let mut frame_a = pool.acquire_scoped(3840, 2160, PixelFormat::Yuv420p);
    let mut frame_b = pool.acquire_scoped(3840, 2160, PixelFormat::Yuv420p);
    frame_a.extend_edges();
    frame_b.extend_edges();

    println!(
        "[Arena] 4K UHD (3840x2160) YUV420p Frame: Y stride = {}B (64B aligned = {}), 2048px de-aliased stride = {}B",
        frame_a.plane(0).stride(),
        (frame_a.plane(0).as_ptr() as usize).is_multiple_of(64),
        optimal_plane_stride(2048),
    );

    // Populate synthetic 4K luma planes (3840 x 2160 = 8,294,400 pixels)
    let width = 3840usize;
    let height = 2160usize;
    let stride = frame_a.plane(0).stride();
    {
        let ya = frame_a.plane_mut(0);
        for y in 0..height {
            let row = ya.row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                *p = ((x * 19 + y * 31 + 17) & 0xFF) as u8;
            }
        }
        let yb = frame_b.plane_mut(0);
        for y in 0..height {
            let row = yb.row_mut(y);
            for (x, p) in row.iter_mut().enumerate() {
                *p = ((x * 19 + y * 31 + 20) & 0xFF) as u8;
            }
        }
    }

    let ya_slice = frame_a.plane(0).data();
    let yb_slice = frame_b.plane(0).data();
    let pixels_per_frame = (width * height) as f64;

    println!(
        "\nKernel (4K 3840x2160 Luma / 16 MB Stream)      Scalar         NEON SIMD      Speedup   Check"
    );
    println!(
        "----------------------------------------------------------------------------------------------"
    );

    // 2. 4K Plane SAD
    let iters = 40;
    let t0 = Instant::now();
    let mut sad_ref = 0u64;
    for _ in 0..iters {
        sad_ref = black_box(sad_plane_u8_scalar(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sad_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sad_simd = 0u64;
    for _ in 0..iters {
        sad_simd = black_box(sad_plane_u8(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sad_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sad_ref, sad_simd);
    println!(
        "4K Plane SAD (sum |a - b|)                  {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT ({sad_simd})",
        (pixels_per_frame * iters as f64) / dt_sad_scalar / 1e9,
        (pixels_per_frame * iters as f64) / dt_sad_simd / 1e9,
        dt_sad_scalar / dt_sad_simd
    );

    // 3. 4K Plane SSE
    let t0 = Instant::now();
    let mut sse_ref = 0u64;
    for _ in 0..iters {
        sse_ref = black_box(sse_plane_u8_scalar(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sse_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sse_simd = 0u64;
    for _ in 0..iters {
        sse_simd = black_box(sse_plane_u8(
            black_box(ya_slice),
            stride,
            black_box(yb_slice),
            stride,
            width,
            height,
        ));
    }
    let dt_sse_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sse_ref, sse_simd);
    println!(
        "4K Plane SSE (sum (a - b)^2)                {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT ({sse_simd})",
        (pixels_per_frame * iters as f64) / dt_sse_scalar / 1e9,
        (pixels_per_frame * iters as f64) / dt_sse_simd / 1e9,
        dt_sse_scalar / dt_sse_simd
    );

    // 4. 4K SSIM 4x4 Sub-Block Kernel (vf_ssim.c algorithm)
    let mut sums_ref = vec![[0i32; 4]; width / 4];
    let mut sums_simd = vec![[0i32; 4]; width / 4];
    let t0 = Instant::now();
    for _ in 0..iters {
        for by in 0..(height / 4) {
            let off = by * 4 * stride;
            ssim_4x4xn_u8_scalar(
                black_box(&ya_slice[off..]),
                stride,
                black_box(&yb_slice[off..]),
                stride,
                &mut sums_ref,
            );
        }
    }
    let dt_ssim_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    for _ in 0..iters {
        for by in 0..(height / 4) {
            let off = by * 4 * stride;
            ssim_4x4xn_u8(
                black_box(&ya_slice[off..]),
                stride,
                black_box(&yb_slice[off..]),
                stride,
                &mut sums_simd,
            );
        }
    }
    let dt_ssim_simd = t1.elapsed().as_secs_f64();
    assert_eq!(sums_ref, sums_simd);
    let ssim_val = ssim_plane_8bit(ya_slice, stride, yb_slice, stride, width, height);
    println!(
        "4K Plane SSIM 4x4-Reuse (vf_ssim)           {:>8.2} Gpx/s   {:>8.2} Gpx/s   {:>5.2}x    EXACT ({:.5} / {:.2} dB)",
        (pixels_per_frame * iters as f64) / dt_ssim_scalar / 1e9,
        (pixels_per_frame * iters as f64) / dt_ssim_simd / 1e9,
        dt_ssim_scalar / dt_ssim_simd,
        ssim_val,
        ssim_to_db(ssim_val)
    );

    // 5. 4x4 Forward + Inverse DCT (in-register vtrn1/vtrn2 transpose + vrshr_n_s16)
    let block = [
        14i16, -22, 7, 3, -9, 31, -15, 4, 18, -6, -11, 2, -3, 8, -5, 1,
    ];
    let dct_iters = 2_000_000usize;
    let t0 = Instant::now();
    let mut out_scalar = [0i16; 16];
    for _ in 0..dct_iters {
        let c = forward_dct4x4_scalar(black_box(&block));
        out_scalar = black_box(inverse_idct4x4_scalar(black_box(&c)));
    }
    let dt_dct_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut out_simd = [0i16; 16];
    for _ in 0..dct_iters {
        let c = forward_dct4x4(black_box(&block));
        out_simd = black_box(inverse_idct4x4(black_box(&c)));
    }
    let dt_dct_simd = t1.elapsed().as_secs_f64();
    assert_eq!(out_scalar, out_simd);
    println!(
        "H.264 4x4 DCT+IDCT (vtrn1/vtrn2 + vrshr)    {:>8.2} Mblk/s  {:>8.2} Mblk/s  {:>5.2}x    EXACT",
        (dct_iters as f64) / dt_dct_scalar / 1e6,
        (dct_iters as f64) / dt_dct_simd / 1e6,
        dt_dct_scalar / dt_dct_simd
    );

    // 6. Annex-B Start-Code Scanner (with periodic isolated 0x00 bytes)
    let stream_len = 16 * 1024 * 1024;
    let mut annex_b = vec![0x5Au8; stream_len];
    for i in (19..stream_len - 16).step_by(23) {
        annex_b[i] = 0x00; // isolated single zero bytes
    }
    annex_b[stream_len - 8] = 0x00;
    annex_b[stream_len - 7] = 0x00;
    annex_b[stream_len - 6] = 0x00;
    annex_b[stream_len - 5] = 0x01;

    let scan_iters = 40usize;
    let t0 = Instant::now();
    let mut pos_ref = None;
    for _ in 0..scan_iters {
        pos_ref = black_box(find_annex_b_start_code_scalar(black_box(&annex_b)));
    }
    let dt_scan_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut pos_simd = None;
    for _ in 0..scan_iters {
        pos_simd = black_box(find_annex_b_start_code(black_box(&annex_b)));
    }
    let dt_scan_simd = t1.elapsed().as_secs_f64();
    assert_eq!(pos_ref, pos_simd);
    println!(
        "Annex-B Start-Code Scan (isolated 0x00s)    {:>8.2} GB/s    {:>8.2} GB/s    {:>5.2}x    EXACT",
        (stream_len * scan_iters) as f64 / dt_scan_scalar / 1e9,
        (stream_len * scan_iters) as f64 / dt_scan_simd / 1e9,
        dt_scan_scalar / dt_scan_simd
    );

    // 7. RBSP Emulation-Prevention Unescaping
    let mut rbsp_raw = vec![0x3Cu8; 4 * 1024 * 1024];
    for i in (0..rbsp_raw.len() - 4).step_by(512) {
        rbsp_raw[i] = 0x00;
        rbsp_raw[i + 1] = 0x00;
        rbsp_raw[i + 2] = 0x01;
    }
    let escaped = escape_rbsp(&rbsp_raw);
    let rbsp_iters = 40usize;
    let t0 = Instant::now();
    let mut u_ref = Vec::new();
    for _ in 0..rbsp_iters {
        u_ref = black_box(unescape_rbsp_scalar(black_box(&escaped)));
    }
    let dt_rbsp_scalar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut u_simd = Vec::new();
    for _ in 0..rbsp_iters {
        u_simd = black_box(unescape_rbsp(black_box(&escaped)));
    }
    let dt_rbsp_simd = t1.elapsed().as_secs_f64();
    assert_eq!(u_ref, u_simd);
    assert_eq!(u_simd, rbsp_raw);
    println!(
        "H.264/HEVC RBSP Unescape (vextq_u8 pair)    {:>8.2} GB/s    {:>8.2} GB/s    {:>5.2}x    EXACT",
        (escaped.len() * rbsp_iters) as f64 / dt_rbsp_scalar / 1e9,
        (escaped.len() * rbsp_iters) as f64 / dt_rbsp_simd / 1e9,
        dt_rbsp_scalar / dt_rbsp_simd
    );

    // 8. Slicing-by-8 CRC-16 (FLAC) & CRC-32 (MPEG-TS), plus CLZ Bitstream & 128-bit Timestamp check
    let t_crc16 = Instant::now();
    let mut c16 = 0u16;
    for _ in 0..20 {
        c16 ^= black_box(crc16_flac(black_box(&annex_b)));
    }
    let crc16_gbps = (annex_b.len() * 20) as f64 / t_crc16.elapsed().as_secs_f64() / 1e9;

    let t_crc32 = Instant::now();
    let mut c32 = 0u32;
    for _ in 0..20 {
        c32 ^= black_box(crc32_mpeg2(black_box(&annex_b)));
    }
    let crc32_gbps = (annex_b.len() * 20) as f64 / t_crc32.elapsed().as_secs_f64() / 1e9;

    let mut bw = BitWriter::with_capacity(1024);
    for v in 0..256u64 {
        bw.write_ue(v);
        bw.write_rice_signed((v as i64) - 128, 4);
    }
    bw.write_rbsp_trailing_bits();
    let bits_buf = bw.finish();
    let mut br = BitReader::new(&bits_buf);
    for v in 0..256u64 {
        assert_eq!(br.read_ue().unwrap(), v);
        assert_eq!(br.read_rice_signed(4).unwrap(), (v as i64) - 128);
    }
    assert!(!br.more_rbsp_data());

    let ts_48h = Timestamp::new(48 * 3600 * 90_000, Rational64::TIMEBASE_MPEG_90K);
    let ts_ns = ts_48h.rescale_to(Rational64::TIMEBASE_NANOS);
    let satd8 = satd_8x8_u8(ya_slice, stride, yb_slice, stride);

    println!(
        "----------------------------------------------------------------------------------------------"
    );
    println!(
        "[CRC Slicing-by-8] FLAC CRC-16: {:.2} GB/s | MPEG-TS CRC-32: {:.2} GB/s (crc16=0x{:04X}, crc32=0x{:08X})",
        crc16_gbps, crc32_gbps, c16, c32
    );
    println!(
        "[CLZ Bitstream]    512 Exp-Golomb ue(v) + Signed Rice(k=4) + RBSP stop-bit verified in {} bytes",
        bits_buf.len()
    );
    println!(
        "[128-bit Time]     48h MPEG-TS 90kHz -> Nanoseconds exact: {} ({} ns), 8x8 SATD = {}",
        ts_ns.format_hms(),
        ts_ns.ticks,
        satd8
    );
}
