//! Peak RSS of `transcode` and `metrics` does not grow with the input length (BUG-0072).
//!
//! Each measurement runs in a child process (this test binary, re-run with only `rss_child`
//! selected), because peak RSS only ever grows within a process. Linux only: the child reads its
//! own `VmHWM`. Both inputs are long enough to fill the decode batches and encode waves of a
//! 4 MiB budget, so their peaks should differ by no more than the encoded output and allocator
//! slack. Decoding the whole input first, as
//! `transcode` used to, adds about 85 KB per 320x180 frame: 15 MB more for the longer input.

use ragdoll_pipeline::{
    TranscodeOptions, evaluate_media_metrics_with_budget, synth_media, transcode_file_with_options,
};
use std::path::Path;
use std::process::Command;

const CHILD_ENV: &str = "RAGDOLL_BOUNDED_MEMORY_CHILD";

/// Child side: runs `$CHILD_ENV` = "transcode|input|output" or "metrics|input|input" and prints
/// its peak RSS. A no-op when the variable is unset, so it passes in normal runs.
#[test]
fn rss_child() {
    let Ok(spec) = std::env::var(CHILD_ENV) else {
        return;
    };
    let parts: Vec<&str> = spec.split('|').collect();
    match parts[0] {
        "transcode" => {
            let opts = TranscodeOptions {
                qp: 30,
                threads: 2,
                memory_budget_bytes: Some(4 << 20),
                ..Default::default()
            };
            transcode_file_with_options(parts[1], parts[2], &opts).unwrap();
        }
        _ => {
            evaluate_media_metrics_with_budget(parts[1], parts[2], 4 << 20).unwrap();
        }
    }
    // `VmHWM`, not `getrusage`: on Linux `ru_maxrss` keeps the parent's peak across `exec`.
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kib: u64 = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.split_whitespace().next()?.parse().ok())
        .unwrap();
    println!("PEAK_RSS={}", kib * 1024);
}

fn peak_rss(verb: &str, a: &Path, b: &Path) -> u64 {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "rss_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, format!("{verb}|{}|{}", a.display(), b.display()))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
        .lines()
        .find_map(|l| l.split_once("PEAK_RSS=").map(|(_, v)| v))
        .and_then(|v| v.split_whitespace().next()?.parse().ok())
        .unwrap_or_else(|| panic!("no PEAK_RSS line in: {stdout}"))
}

#[cfg(target_os = "linux")]
#[test]
fn peak_rss_does_not_grow_with_input_length() {
    let dir = std::env::temp_dir().join(format!("ragdoll_bounded_memory_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // (case, short peak, long peak)
    let mut results = Vec::new();
    for ext in ["mp4", "y4m"] {
        let mut peaks = Vec::new();
        for (name, secs) in [("short", 2.0), ("long", 8.0)] {
            let input = dir.join(format!("{name}.{ext}"));
            synth_media(&input, 320, 180, 30, secs).unwrap();
            let output = dir.join(format!("{name}_{ext}_out.mp4"));
            peaks.push(peak_rss("transcode", &input, &output));
            if ext == "mp4" {
                peaks.push(peak_rss("metrics", &input, &input));
            }
        }
        if ext == "mp4" {
            results.push(("transcode mp4", peaks[0], peaks[2]));
            results.push(("metrics mp4", peaks[1], peaks[3]));
        } else {
            results.push(("transcode y4m", peaks[0], peaks[1]));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    const SLACK: u64 = 6 << 20;
    for (case, short, long) in results {
        assert!(
            long <= short + SLACK,
            "{case}: peak RSS grew with input length: {} MiB for 2 s, {} MiB for 8 s",
            short >> 20,
            long >> 20
        );
    }
}
