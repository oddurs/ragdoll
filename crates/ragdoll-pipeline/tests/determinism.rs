//! `transcode` output does not depend on the thread count or the power mode (`0061`).
//!
//! The input is shorter than 15 frames per worker, where the GOP length used to follow the
//! worker count. One test function, because the power mode is process-wide.

use ragdoll_pipeline::{
    H264Profile, PowerMode, TranscodeOptions, set_global_power_mode, synth_media,
    transcode_file_with_options,
};

#[test]
fn transcode_output_is_independent_of_threads_and_power_mode() {
    let dir = std::env::temp_dir().join(format!("ragdoll_determinism_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("in.mp4");
    synth_media(&input, 320, 180, 30, 1.5).unwrap();

    // (container, scale, profile)
    let variants = [
        ("mp4", None, H264Profile::High),
        ("mkv", Some((160, 90)), H264Profile::Baseline),
    ];
    for (ext, scale_dims, h264_profile) in variants {
        let mut outputs = Vec::new();
        for (mode, threads) in [
            (PowerMode::Performance, 1),
            (PowerMode::Performance, 2),
            (PowerMode::Performance, 4),
            (PowerMode::Efficiency, 0),
            (PowerMode::Performance, 0),
        ] {
            set_global_power_mode(mode);
            let out = dir.join(format!("out_{mode:?}_{threads}.{ext}"));
            let opts = TranscodeOptions {
                qp: 28,
                threads,
                scale_dims,
                h264_profile,
                ..Default::default()
            };
            transcode_file_with_options(&input, &out, &opts).unwrap();
            outputs.push((
                format!("{mode:?}, threads {threads}"),
                std::fs::read(&out).unwrap(),
            ));
        }
        set_global_power_mode(PowerMode::Performance);
        let (first_name, first) = &outputs[0];
        for (name, bytes) in &outputs[1..] {
            assert!(
                bytes == first,
                "{ext}: output for {name} differs from {first_name}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
