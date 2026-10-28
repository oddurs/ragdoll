//! `qa`: Ragdoll's validation tooling (`0062`), kept out of the library workspace so its needs
//! never reach the library's `Cargo.lock`.
//!
//! `qa corpus` manages the media corpus listed in `corpus/manifest.toml`, in four tiers:
//!
//! 1. synthesized media, made by `ragdoll synth` and kept in git under `corpus/synth/`;
//! 2. conformance suites, downloaded by URL;
//! 3. a real-world zoo of files from other encoders and muxers, downloaded by URL;
//! 4. regression files kept in git forever, at paths relative to the repository root.
//!
//! Every entry pins a SHA-256 and a size. Tiers 2 and 3 download into `corpus/cache/` (or
//! `$RAGDOLL_QA_CACHE`), which CI caches by the manifest's hash.
//!
//! ```text
//! qa corpus list   [--tier N]
//! qa corpus fetch  [--tier N]   download what is missing, verifying each file
//! qa corpus verify [--tier N]   fail on any missing file, size or hash mismatch
//! qa corpus synth               regenerate tier 1 with ragdoll and check it is unchanged
//! ```

mod manifest;
mod sha256;

use manifest::{Entry, Manifest};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

fn qa_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cache_dir() -> PathBuf {
    std::env::var_os("RAGDOLL_QA_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| qa_dir().join("corpus/cache"))
}

/// Where an entry lives on disk.
fn location(e: &Entry) -> PathBuf {
    match e.tier {
        1 => qa_dir().join("corpus/synth").join(&e.name),
        4 => qa_dir().join("..").join(&e.name),
        _ => cache_dir().join(&e.name),
    }
}

/// Checks a file against its entry: `Err` describes what is wrong.
fn check(e: &Entry, path: &Path) -> Result<(), String> {
    let meta = std::fs::metadata(path).map_err(|_| "missing".to_string())?;
    if meta.len() != e.size {
        return Err(format!("size {} (expected {})", meta.len(), e.size));
    }
    let got = sha256::file_hex(path).map_err(|err| err.to_string())?;
    if got != e.sha256 {
        return Err(format!("sha256 {got} (expected {})", e.sha256));
    }
    Ok(())
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: qa corpus list|fetch|verify [--tier N]\n       qa corpus synth\n\n\
         The conformance, interop, fuzz, rd, perf and report tools are tracked as cairn items \
         0063 to 0069."
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("corpus") || args.len() < 2 {
        return usage();
    }
    let tier = match args.get(2).map(String::as_str) {
        None => None,
        Some("--tier") => match args.get(3).and_then(|t| t.parse::<u8>().ok()) {
            Some(t @ 1..=4) => Some(t),
            _ => return usage(),
        },
        Some(_) => return usage(),
    };
    let manifest_path = qa_dir().join("corpus/manifest.toml");
    let manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("{}: {err}", manifest_path.display());
            return ExitCode::FAILURE;
        }
    };
    let entries: Vec<&Entry> = manifest
        .entries
        .iter()
        .filter(|e| tier.is_none_or(|t| e.tier == t))
        .collect();
    let ok = match args[1].as_str() {
        "list" => {
            for e in &entries {
                let state = match check(e, &location(e)) {
                    Ok(()) => "ok".to_string(),
                    Err(why) => why,
                };
                println!("tier {} {:<44} {:>10} B  {state}", e.tier, e.name, e.size);
            }
            true
        }
        "fetch" => fetch(&entries),
        "verify" => verify(&entries),
        "synth" if tier.is_none() => synth(&manifest.entries),
        _ => return usage(),
    };
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn verify(entries: &[&Entry]) -> bool {
    let mut bad = 0;
    for e in entries {
        if let Err(why) = check(e, &location(e)) {
            println!("FAIL tier {} {}: {why}", e.tier, e.name);
            bad += 1;
        }
    }
    println!(
        "{} of {} files verified",
        entries.len() - bad,
        entries.len()
    );
    bad == 0
}

fn fetch(entries: &[&Entry]) -> bool {
    let mut bad = 0;
    for e in entries {
        let Some(url) = &e.url else { continue };
        let dest = location(e);
        if check(e, &dest).is_ok() {
            continue;
        }
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let tmp = dest.with_extension("part");
        println!("fetch {url}");
        let ran = Command::new("curl")
            .args(["-sSfL", "--retry", "3", "--max-time", "1200", "-o"])
            .arg(&tmp)
            .arg(url)
            .status();
        let result = match ran {
            Ok(s) if s.success() => check(e, &tmp),
            Ok(s) => Err(format!("curl exited with {s}")),
            Err(err) => Err(format!("cannot run curl: {err}")),
        };
        match result {
            Ok(()) => std::fs::rename(&tmp, &dest).unwrap(),
            Err(why) => {
                let _ = std::fs::remove_file(&tmp);
                println!("FAIL tier {} {}: {why}", e.tier, e.name);
                bad += 1;
            }
        }
    }
    bad == 0
}

/// Regenerates every tier-1 file in place, then verifies them: a mismatch means `synth` (or a
/// codec under it) changed its output.
fn synth(all: &[Entry]) -> bool {
    let tier1: Vec<&Entry> = all.iter().filter(|e| e.tier == 1).collect();
    for e in &tier1 {
        let Some(s) = e.synth else { continue };
        let path = location(e);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        if let Err(err) = ragdoll_pipeline::synth_media(&path, s.width, s.height, s.fps, s.seconds)
        {
            println!("FAIL tier 1 {}: synth failed: {err}", e.name);
            return false;
        }
    }
    verify(&tier1)
}
