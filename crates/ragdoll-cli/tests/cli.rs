//! The `ragdoll` command-line contract (`0061`): golden snapshots of text output, a schema for
//! every `--json` output, and the exit status of every subcommand: 0 on success, 1 when the
//! command fails, 2 for a usage error.
//!
//! Commands run in a scratch directory with relative paths. Snapshots in `tests/golden/` mask
//! timings, rates and machine-chosen thread counts; byte counts and quality numbers stay, so a
//! snapshot also notices when an output changes. After an intended change, rewrite them with
//! `RAGDOLL_BLESS=1 cargo test -p ragdoll-cli --test cli`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

/// A scratch directory holding `v.mp4` (160x90, 1 s) and `a.wav` (1 s), made once.
fn fixture() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("ragdoll_cli_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for args in [
            &[
                "synth",
                "v.mp4",
                "--resolution",
                "160x90",
                "--duration",
                "1",
            ][..],
            &["synth", "a.wav", "--duration", "1"][..],
        ] {
            let out = run_in(&dir, args);
            assert!(out.status.success(), "fixture {args:?}: {}", stderr(&out));
        }
        dir
    })
}

fn run_in(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ragdoll"))
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap()
}

fn run(args: &[&str]) -> Output {
    run_in(fixture(), args)
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

// ---------------------------------------------------------------------------------------------
// Text snapshots

/// Masks what varies between runs and machines: numbers followed by a time or rate unit,
/// machine-chosen thread counts, and the SIMD / CPU lines of `--help` and `--version`.
fn normalize(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        let line = if line.starts_with("  Hardware SIMD :") {
            "  Hardware SIMD : <simd>".to_string()
        } else if line.starts_with("  CPU Topology  :") {
            "  CPU Topology  : <cpu>".to_string()
        } else if line.starts_with("ragdoll ") && line.contains('(') {
            format!("{}(<build>)", &line[..line.find('(').unwrap()])
        } else {
            mask_line(line)
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn mask_line(line: &str) -> String {
    const UNITS: [&str; 4] = [" ms", " µs", " fps", " GB/s"];
    let b = line.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < line.len() {
        if line[i..].starts_with("threads=") {
            let end = i + 8 + line[i + 8..].bytes().take_while(u8::is_ascii_digit).count();
            out.push_str("threads=<n>");
            i = end;
            continue;
        }
        let starts_number = b[i].is_ascii_digit() && (i == 0 || !b[i - 1].is_ascii_alphanumeric());
        if starts_number {
            let len = line[i..]
                .bytes()
                .take_while(|c| c.is_ascii_digit() || *c == b'.')
                .count();
            if UNITS.iter().any(|u| line[i + len..].starts_with(u)) {
                out.push_str("<n>");
                i += len;
                continue;
            }
            out.push_str(&line[i..i + len]);
            i += len;
            continue;
        }
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Text-output commands, in order: earlier ones write files later ones read.
const TEXT_CASES: &[(&str, &[&str])] = &[
    ("help", &["--help"]),
    ("version", &["--version"]),
    ("probe", &["probe", "v.mp4"]),
    ("probe_gop_map", &["probe", "v.mp4", "--gop-map"]),
    ("remux", &["remux", "v.mp4", "text_remux.mkv"]),
    (
        "transcode",
        &[
            "transcode",
            "v.mp4",
            "text_t.mp4",
            "--qp",
            "30",
            "--threads",
            "2",
        ],
    ),
    ("transcode_audio", &["transcode", "a.wav", "text_a.flac"]),
    (
        "cut",
        &[
            "cut",
            "v.mp4",
            "text_cut.mp4",
            "--from",
            "0.2",
            "--to",
            "0.8",
        ],
    ),
    ("concat", &["concat", "v.mp4", "v.mp4", "-o", "text_cc.mp4"]),
    ("package", &["package", "v.mp4", "-o", "text_pkg"]),
    (
        "extract",
        &["extract", "v.mp4", "-o", "text_e.ppm", "--at", "0.5"],
    ),
    ("metrics", &["metrics", "v.mp4", "text_t.mp4"]),
    (
        "synth",
        &[
            "synth",
            "text_s.y4m",
            "--resolution",
            "64x36",
            "--duration",
            "0.2",
        ],
    ),
];

#[test]
fn text_output_matches_golden_snapshots() {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let bless = std::env::var("RAGDOLL_BLESS").is_ok_and(|v| v == "1");
    let mut failures = Vec::new();
    for (name, args) in TEXT_CASES {
        let out = run(args);
        assert!(out.status.success(), "{name}: {}", stderr(&out));
        let got = normalize(&stdout(&out));
        let path = golden_dir.join(format!("{name}.txt"));
        if bless {
            std::fs::create_dir_all(&golden_dir).unwrap();
            std::fs::write(&path, &got).unwrap();
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(want) if want == got => {}
            Ok(want) => failures.push(format!("{name}: expected\n{want}\ngot\n{got}")),
            Err(_) => failures.push(format!("{name}: no snapshot at {}", path.display())),
        }
    }
    assert!(
        failures.is_empty(),
        "{} snapshot(s) differ (RAGDOLL_BLESS=1 rewrites them):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

// ---------------------------------------------------------------------------------------------
// JSON schema

#[derive(Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }
        Ok(v)
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    self.ws();
                    let Json::Str(k) = self.string()? else {
                        unreachable!()
                    };
                    self.eat(b':')?;
                    let v = self.value()?;
                    if m.insert(k.clone(), v).is_some() {
                        return Err(format!("duplicate key {k}"));
                    }
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'"') => self.string(),
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Json::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Json::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Json::Null)
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                self.i += 1;
                while self.i < self.s.len()
                    && (self.s[self.i].is_ascii_digit() || b".eE+-".contains(&self.s[self.i]))
                {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[start..self.i]).unwrap();
                if t.contains(['.', 'e', 'E']) {
                    t.parse().map(Json::Num).map_err(|e| format!("{t}: {e}"))
                } else {
                    t.parse().map(Json::Int).map_err(|e| format!("{t}: {e}"))
                }
            }
            _ => Err(format!("unexpected input at byte {}", self.i)),
        }
    }

    fn string(&mut self) -> Result<Json, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            match self.s.get(self.i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(Json::Str(out));
                }
                Some(b'\\') => {
                    let esc = *self.s.get(self.i + 1).ok_or("bad escape")?;
                    self.i += 2;
                    match esc {
                        b'"' | b'\\' | b'/' => out.push(esc as char),
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.s[self.i..self.i + 4]).unwrap();
                            let cp = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            self.i += 4;
                        }
                        _ => return Err(format!("bad escape \\{}", esc as char)),
                    }
                }
                Some(_) => {
                    let rest = std::str::from_utf8(&self.s[self.i..]).map_err(|e| e.to_string())?;
                    let ch = rest.chars().next().unwrap();
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
            }
        }
    }
}

/// The shape a JSON value must have. Objects must have exactly the listed keys.
enum T {
    Str,
    Int,
    Num,
    Bool,
    Nullable(Box<T>),
    Arr(Box<T>),
    Obj(Vec<(&'static str, T)>),
    OneOf(Vec<T>),
}

fn check(v: &Json, t: &T, path: &str) -> Result<(), String> {
    match (t, v) {
        (T::Str, Json::Str(_)) | (T::Int, Json::Int(_)) | (T::Bool, Json::Bool(_)) => Ok(()),
        // A float field may print without a fractional part.
        (T::Num, Json::Num(_) | Json::Int(_)) => Ok(()),
        (T::Nullable(_), Json::Null) => Ok(()),
        (T::Nullable(inner), _) => check(v, inner, path),
        (T::Arr(inner), Json::Arr(items)) => items
            .iter()
            .enumerate()
            .try_for_each(|(i, item)| check(item, inner, &format!("{path}[{i}]"))),
        (T::Obj(fields), Json::Obj(m)) => {
            for (k, ft) in fields {
                let item = m
                    .get(*k)
                    .ok_or_else(|| format!("{path}: missing key {k}"))?;
                check(item, ft, &format!("{path}.{k}"))?;
            }
            match m.keys().find(|k| !fields.iter().any(|(f, _)| f == k)) {
                Some(extra) => Err(format!("{path}: unexpected key {extra}")),
                None => Ok(()),
            }
        }
        (T::OneOf(options), _) => {
            let errors: Vec<String> = options
                .iter()
                .filter_map(|o| check(v, o, path).err())
                .collect();
            if errors.len() < options.len() {
                Ok(())
            } else {
                Err(format!(
                    "{path}: matches no variant ({})",
                    errors.join("; ")
                ))
            }
        }
        _ => Err(format!("{path}: wrong type ({v:?})")),
    }
}

/// `PipelineTelemetry` as printed by remux, transcode, extract and synth.
fn telemetry() -> T {
    T::Obj(vec![
        ("operation", T::Str),
        ("input", T::Str),
        ("outputs", T::Arr(Box::new(T::Str))),
        ("frames_processed", T::Int),
        ("audio_samples_processed", T::Int),
        ("gops_processed", T::Int),
        ("threads_used", T::Int),
        ("bytes_in", T::Int),
        ("bytes_out", T::Int),
        ("elapsed_ms", T::Num),
        ("fps", T::Num),
        ("bandwidth_gbps", T::Num),
        ("compression_ratio", T::Num),
    ])
}

fn probe_schema() -> T {
    let track = T::OneOf(vec![
        T::Obj(vec![
            ("index", T::Int),
            ("type", T::Str),
            ("codec", T::Str),
            ("width", T::Int),
            ("height", T::Int),
            ("fps", T::Num),
            ("pixel_format", T::Str),
            ("samples", T::Int),
        ]),
        T::Obj(vec![
            ("index", T::Int),
            ("type", T::Str),
            ("codec", T::Str),
            ("sample_rate", T::Int),
            ("channels", T::Int),
            ("sample_format", T::Str),
            ("samples", T::Int),
        ]),
        T::Obj(vec![
            ("index", T::Int),
            ("type", T::Str),
            ("codec", T::Str),
            ("samples", T::Int),
        ]),
    ]);
    T::Obj(vec![
        ("path", T::Str),
        ("container", T::Str),
        ("file_size", T::Int),
        ("probe_time_us", T::Num),
        ("gop_count", T::Int),
        ("h264_profile", T::Nullable(Box::new(T::Str))),
        ("tracks", T::Arr(Box::new(track))),
    ])
}

fn metrics_schema() -> T {
    let mut fields = vec![("frames", T::Int), ("elapsed_ms", T::Num), ("fps", T::Num)];
    for k in [
        "psnr_y",
        "psnr_u",
        "psnr_v",
        "psnr_avg",
        "psnr_yuv_611",
        "ssim_y",
        "ssim_avg",
        "ssim_db",
        "ms_ssim",
        "vmaf_lite",
    ] {
        fields.push((k, T::Num));
    }
    T::Obj(fields)
}

#[test]
fn json_output_matches_schema() {
    let cases: Vec<(&str, Vec<&str>, T)> = vec![
        ("probe", vec!["probe", "v.mp4", "--json"], probe_schema()),
        (
            "probe wav",
            vec!["probe", "a.wav", "--json"],
            probe_schema(),
        ),
        (
            "remux",
            vec!["remux", "v.mp4", "json_remux.mkv", "--json"],
            telemetry(),
        ),
        (
            "transcode",
            vec!["transcode", "v.mp4", "json_t.mp4", "--qp", "30", "--json"],
            telemetry(),
        ),
        (
            "transcode audio",
            vec!["transcode", "a.wav", "json_a.flac", "--json"],
            telemetry(),
        ),
        (
            "cut",
            vec!["cut", "v.mp4", "json_cut.mp4", "--from", "0.2", "--json"],
            T::Obj(vec![
                ("operation", T::Str),
                ("output_bytes", T::Int),
                ("total_video_packets", T::Int),
                ("copied_video_packets", T::Int),
                ("reencoded_video_frames", T::Int),
                ("audio_packets", T::Int),
                ("elapsed_ms", T::Num),
            ]),
        ),
        (
            "concat",
            vec!["concat", "v.mp4", "v.mp4", "-o", "json_cc.mp4", "--json"],
            T::Obj(vec![
                ("operation", T::Str),
                ("segments_merged", T::Int),
                ("video_packets", T::Int),
                ("audio_packets", T::Int),
                ("reencoded", T::Bool),
                ("output_bytes", T::Int),
                ("elapsed_ms", T::Num),
            ]),
        ),
        (
            "package",
            vec!["package", "v.mp4", "-o", "json_pkg", "--json"],
            T::Obj(vec![
                ("operation", T::Str),
                ("renditions", T::Int),
                ("segments_per_rendition", T::Int),
                ("master_m3u8", T::Str),
                ("mpd", T::Str),
                ("total_bytes", T::Int),
                ("elapsed_ms", T::Num),
            ]),
        ),
        (
            "extract",
            vec!["extract", "v.mp4", "-o", "json_e.ppm", "--json"],
            telemetry(),
        ),
        (
            "metrics",
            vec!["metrics", "v.mp4", "v.mp4", "--json"],
            metrics_schema(),
        ),
        (
            "synth",
            vec![
                "synth",
                "json_s.y4m",
                "--resolution",
                "64x36",
                "--duration",
                "0.2",
                "--json",
            ],
            telemetry(),
        ),
    ];
    for (name, args, schema) in cases {
        let out = run(&args);
        assert!(out.status.success(), "{name}: {}", stderr(&out));
        let text = stdout(&out);
        let json = Parser::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
        check(&json, &schema, name).unwrap_or_else(|e| panic!("{e}\n{text}"));
    }
}

// ---------------------------------------------------------------------------------------------
// Exit status

#[test]
fn exit_status_of_every_subcommand() {
    fixture();
    // (args, expected status)
    let cases: &[(&[&str], i32)] = &[
        (&["--help"], 0),
        (&["--version"], 0),
        (&["no-such-subcommand"], 2),
        (&["probe", "v.mp4"], 0),
        (&["probe"], 2),
        (&["probe", "v.mp4", "--bogus"], 2),
        (&["probe", "missing.mp4"], 1),
        (&["remux", "v.mp4", "exit_r.mkv"], 0),
        (&["remux", "v.mp4"], 2),
        (&["remux", "missing.mp4", "exit_r2.mkv"], 1),
        (&["transcode", "v.mp4", "exit_t.mp4", "--qp", "40"], 0),
        (&["transcode", "v.mp4"], 2),
        (&["transcode", "missing.mp4", "exit_t2.mp4"], 1),
        (&["cut", "v.mp4", "exit_c.mp4", "--to", "0.5"], 0),
        (&["cut", "v.mp4"], 2),
        (&["cut", "missing.mp4", "exit_c2.mp4"], 1),
        (&["concat", "v.mp4", "v.mp4", "-o", "exit_cc.mp4"], 0),
        (&["concat", "v.mp4"], 2),
        (&["concat", "missing.mp4", "v.mp4", "-o", "exit_cc2.mp4"], 1),
        (&["package", "v.mp4", "-o", "exit_pkg"], 0),
        (&["package"], 2),
        (&["package", "missing.mp4", "-o", "exit_pkg2"], 1),
        (&["extract", "v.mp4", "-o", "exit_e.ppm"], 0),
        (&["extract"], 2),
        (&["extract", "missing.mp4", "-o", "exit_e2.ppm"], 1),
        (&["metrics", "v.mp4", "v.mp4"], 0),
        (&["metrics", "v.mp4"], 2),
        (&["metrics", "v.mp4", "missing.mp4"], 1),
        (&["synth", "exit_s.y4m", "--duration", "0.1"], 0),
        (&["synth"], 2),
        (&["synth", "no-such-dir/s.y4m", "--duration", "0.1"], 1),
        (&["bench", "--bogus"], 2),
        (&["self-test", "extra"], 2),
    ];
    let mut failures = Vec::new();
    for (args, want) in cases {
        let out = run(args);
        let got = out.status.code().unwrap_or(-1);
        if got != *want {
            failures.push(format!(
                "{args:?}: exit {got}, expected {want}: {}",
                stderr(&out)
            ));
        } else if *want == 2 && !stderr(&out).contains("ragdoll --help") {
            failures.push(format!("{args:?}: usage error without a usage hint"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `self-test` runs the SIMD-versus-scalar kernel checks on 4K frames: about a minute in a debug
/// build, so CI runs it on its own (`cargo test -p ragdoll-cli -- --ignored`).
#[test]
#[ignore]
fn self_test_passes() {
    let out = run(&["self-test"]);
    assert!(out.status.success(), "{}{}", stdout(&out), stderr(&out));
}
