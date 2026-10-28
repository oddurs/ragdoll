//! `corpus/manifest.toml`: a sequence of `[[file]]` tables with string and integer values,
//! parsed here without a TOML crate.
//!
//! ```toml
//! [[file]]
//! name = "h264/BA1_Sony_D.jsv"   # path in the tier's directory (tier 4: from the repo root)
//! tier = 2
//! url = "https://..."             # tiers 2 and 3
//! size = 55537
//! sha256 = "90c8..."
//! synth = "160x90 30fps 1.0s"     # tier 1: what `ragdoll synth` makes
//! ```

use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct Synth {
    pub width: usize,
    pub height: usize,
    pub fps: i64,
    pub seconds: f64,
}

#[derive(Debug)]
pub struct Entry {
    pub name: String,
    pub tier: u8,
    pub url: Option<String>,
    pub size: u64,
    pub sha256: String,
    pub synth: Option<Synth>,
}

pub struct Manifest {
    pub entries: Vec<Entry>,
}

#[derive(Default)]
struct Fields {
    name: Option<String>,
    tier: Option<u8>,
    url: Option<String>,
    size: Option<u64>,
    sha256: Option<String>,
    synth: Option<Synth>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut entries = Vec::new();
        let mut cur: Option<(usize, Fields)> = None;
        for (i, raw) in text.lines().enumerate() {
            let line_no = i + 1;
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if line == "[[file]]" {
                if let Some((start, f)) = cur.take() {
                    entries.push(finish(f).map_err(|e| format!("entry at line {start}: {e}"))?);
                }
                cur = Some((line_no, Fields::default()));
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {line_no}: expected key = value"))?;
            let (key, value) = (key.trim(), value.trim());
            let Some((_, f)) = cur.as_mut() else {
                return Err(format!("line {line_no}: {key} outside a [[file]] table"));
            };
            let err = |what: &str| format!("line {line_no}: {key}: {what}");
            match key {
                "name" => f.name = Some(string(value).ok_or_else(|| err("expected a string"))?),
                "url" => f.url = Some(string(value).ok_or_else(|| err("expected a string"))?),
                "sha256" => {
                    let s = string(value).ok_or_else(|| err("expected a string"))?;
                    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
                        return Err(err("expected 64 hex digits"));
                    }
                    f.sha256 = Some(s.to_ascii_lowercase());
                }
                "tier" => f.tier = Some(value.parse().map_err(|_| err("expected 1 to 4"))?),
                "size" => f.size = Some(value.parse().map_err(|_| err("expected bytes"))?),
                "synth" => {
                    let s = string(value).ok_or_else(|| err("expected a string"))?;
                    f.synth = Some(parse_synth(&s).ok_or_else(|| err("expected \"WxH Nfps Ss\""))?);
                }
                _ => return Err(err("unknown key")),
            }
        }
        if let Some((start, f)) = cur.take() {
            entries.push(finish(f).map_err(|e| format!("entry at line {start}: {e}"))?);
        }
        let mut names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        names.sort_unstable();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(format!("duplicate name {}", w[0]));
        }
        Ok(Self { entries })
    }
}

fn finish(f: Fields) -> Result<Entry, String> {
    let tier = f.tier.ok_or("missing tier")?;
    let e = Entry {
        name: f.name.ok_or("missing name")?,
        tier,
        url: f.url,
        size: f.size.ok_or("missing size")?,
        sha256: f.sha256.ok_or("missing sha256")?,
        synth: f.synth,
    };
    match tier {
        1 if e.synth.is_none() => Err("tier 1 needs synth".into()),
        2 | 3 if e.url.is_none() => Err(format!("tier {tier} needs url")),
        1 | 4 if e.url.is_some() => Err(format!("tier {tier} files live in git, not at a url")),
        1..=4 => Ok(e),
        _ => Err("tier must be 1 to 4".into()),
    }
}

fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_str = !in_str,
            '#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

fn string(v: &str) -> Option<String> {
    let inner = v.strip_prefix('"')?.strip_suffix('"')?;
    (!inner.contains(['"', '\\'])).then(|| inner.to_string())
}

fn parse_synth(s: &str) -> Option<Synth> {
    let mut parts = s.split_whitespace();
    let (w, h) = parts.next()?.split_once('x')?;
    let fps = parts.next()?.strip_suffix("fps")?.parse().ok()?;
    let seconds = parts.next()?.strip_suffix('s')?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(Synth {
        width: w.parse().ok()?,
        height: h.parse().ok()?,
        fps,
        seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn parses_entries_and_rejects_bad_ones() {
        let ok = format!(
            "# corpus\n[[file]]\nname = \"a.mp4\"\ntier = 1\nsize = 0\nsha256 = \"{HASH}\"\n\
             synth = \"160x90 30fps 1.0s\"\n\n[[file]]\nname = \"b.264\" # comment\ntier = 2\n\
             url = \"https://example.com/b.264#x\"\nsize = 5\nsha256 = \"{HASH}\"\n"
        );
        let m = Manifest::parse(&ok).unwrap();
        assert_eq!(m.entries.len(), 2);
        assert_eq!(m.entries[0].synth.unwrap().width, 160);
        assert_eq!(
            m.entries[1].url.as_deref(),
            Some("https://example.com/b.264#x")
        );

        for bad in [
            format!("[[file]]\nname = \"a\"\ntier = 2\nsize = 1\nsha256 = \"{HASH}\"\n"),
            format!(
                "[[file]]\nname = \"a\"\ntier = 4\nsize = 1\nsha256 = \"{HASH}\"\nurl = \"u\"\n"
            ),
            "[[file]]\nname = \"a\"\ntier = 4\nsize = 1\nsha256 = \"abc\"\n".to_string(),
            format!(
                "[[file]]\nname = \"a\"\ntier = 4\nsize = 1\nsha256 = \"{HASH}\"\ncolor = \"x\"\n"
            ),
            format!(
                "[[file]]\nname = \"a\"\ntier = 4\nsize = 1\nsha256 = \"{HASH}\"\n\
                 [[file]]\nname = \"a\"\ntier = 4\nsize = 1\nsha256 = \"{HASH}\"\n"
            ),
        ] {
            assert!(Manifest::parse(&bad).is_err(), "accepted:\n{bad}");
        }
    }
}
