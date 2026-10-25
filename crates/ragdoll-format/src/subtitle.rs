//! # `ragdoll-format::subtitle`
//!
//! Pure-Rust **WebVTT (`.vtt`)** and **SubRip (`.srt`)** subtitle demuxers, muxers,
//! and ISO/IEC 14496-30 (`wvtt` / `vttc` / `payl`) MP4 sample box framing (`0037`).

use ragdoll_core::{
    AlignedBuffer, CodecId, PacketFlags, PacketSlice, RagdollError, Rational64, Result, Timestamp,
    TrackInfo, TrackType, rescale_q,
};
use std::io::Write;

/// A single timed subtitle cue with millisecond precision (`timebase = 1/1000`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubtitleCue {
    /// Start timestamp in milliseconds.
    pub start_ms: i64,
    /// End timestamp in milliseconds (`duration_ms = end_ms - start_ms`).
    pub end_ms: i64,
    /// Optional cue identifier (e.g. `"1"` or `"intro"`).
    pub id: Option<String>,
    /// Optional WebVTT cue settings (e.g. `"line:85% align:center"`).
    pub settings: String,
    /// UTF-8 cue text (may span multiple lines separated by `\n`).
    pub payload: String,
}

impl SubtitleCue {
    #[inline]
    pub const fn duration_ms(&self) -> i64 {
        let d = self.end_ms - self.start_ms;
        if d > 0 { d } else { 1 }
    }
}

/// Parses a WebVTT or SubRip timestamp string (`HH:MM:SS.mmm`, `HH:MM:SS,mmm`, or `MM:SS.mmm`)
/// into milliseconds.
pub fn parse_subtitle_timestamp_ms(s: &str) -> Option<i64> {
    let trimmed = s.trim();
    let (hms_part, ms_part) = trimmed
        .split_once('.')
        .or_else(|| trimmed.split_once(','))?;
    let ms_digits: String = ms_part
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .take(3)
        .collect();
    if ms_digits.is_empty() {
        return None;
    }
    let mut ms = ms_digits.parse::<i64>().ok()?;
    for _ in ms_digits.len()..3 {
        ms *= 10;
    }

    let mut parts = hms_part.split(':');
    let p0 = parts.next()?.parse::<i64>().ok()?;
    let p1 = parts.next()?.parse::<i64>().ok()?;
    let total_sec = if let Some(p2_str) = parts.next() {
        let p2 = p2_str.parse::<i64>().ok()?;
        p0 * 3600 + p1 * 60 + p2
    } else {
        p0 * 60 + p1
    };
    Some(total_sec * 1000 + ms)
}

/// Formats a millisecond timestamp as `HH:MM:SS.mmm` (`sep = '.'` for WebVTT)
/// or `HH:MM:SS,mmm` (`sep = ','` for SubRip).
pub fn format_subtitle_timestamp_ms(ms: i64, sep: char) -> String {
    let clamped = ms.max(0);
    let total_sec = clamped / 1000;
    let millis = clamped % 1000;
    let hours = total_sec / 3600;
    let mins = (total_sec % 3600) / 60;
    let secs = total_sec % 60;
    format!("{hours:02}:{mins:02}:{secs:02}{sep}{millis:03}")
}

/// Returns `true` if `data` starts with a `WEBVTT` magic header (allowing optional UTF-8 BOM).
pub fn is_webvtt_bytes(data: &[u8]) -> bool {
    let stripped = data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data);
    stripped.starts_with(b"WEBVTT")
        && (stripped.len() == 6 || matches!(stripped[6], b'\n' | b'\r' | b' ' | b'\t'))
}

/// Returns `true` if `data` looks like a SubRip (`.srt`) document.
pub fn is_srt_bytes(data: &[u8]) -> bool {
    let stripped = data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data);
    let head_len = stripped.len().min(512);
    let Ok(head) = std::str::from_utf8(&stripped[..head_len]) else {
        return false;
    };
    let mut lines = head.lines().map(str::trim).filter(|l| !l.is_empty());
    let Some(first) = lines.next() else {
        return false;
    };
    let timing_line = if first.contains("-->") {
        first
    } else {
        let Some(second) = lines.next() else {
            return false;
        };
        second
    };
    if let Some((lhs, rhs)) = timing_line.split_once("-->") {
        parse_subtitle_timestamp_ms(lhs).is_some()
            && parse_subtitle_timestamp_ms(rhs.split_whitespace().next().unwrap_or("")).is_some()
    } else {
        false
    }
}

/// Parses a standalone WebVTT (`.vtt`) or SubRip (`.srt`) document into a `Vec<SubtitleCue>`.
pub fn parse_subtitles(data: &[u8]) -> Result<(CodecId, Vec<SubtitleCue>)> {
    let stripped = data.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(data);
    let text = std::str::from_utf8(stripped)
        .map_err(|_| RagdollError::InvalidSyntax("subtitle stream is not valid UTF-8"))?;
    let codec_id = if is_webvtt_bytes(stripped) {
        CodecId::WebVtt
    } else if is_srt_bytes(stripped) {
        CodecId::SubRip
    } else {
        return Err(RagdollError::InvalidHeader(
            "unrecognized subtitle format (expected WEBVTT or SRT)".into(),
        ));
    };

    let normalized = text.replace("\r\n", "\n");
    let mut cues = Vec::new();

    for block in normalized.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("WEBVTT")
            || trimmed.starts_with("NOTE")
            || trimmed.starts_with("STYLE")
            || trimmed.starts_with("REGION")
        {
            continue;
        }
        let mut lines = trimmed.lines();
        let Some(first_line) = lines.next() else {
            continue;
        };
        let (id, timing_line) = if first_line.contains("-->") {
            (None, first_line)
        } else {
            let Some(second_line) = lines.next() else {
                continue;
            };
            if !second_line.contains("-->") {
                continue;
            }
            (Some(first_line.trim().to_string()), second_line)
        };

        let Some((lhs, rhs_full)) = timing_line.split_once("-->") else {
            continue;
        };
        let mut rhs_tokens = rhs_full.trim().splitn(2, char::is_whitespace);
        let rhs_ts = rhs_tokens.next().unwrap_or("");
        let settings = rhs_tokens.next().unwrap_or("").trim().to_string();

        let (Some(start_ms), Some(end_ms)) = (
            parse_subtitle_timestamp_ms(lhs),
            parse_subtitle_timestamp_ms(rhs_ts),
        ) else {
            continue;
        };

        let payload_lines: Vec<&str> = lines.collect();
        let payload = payload_lines.join("\n");
        cues.push(SubtitleCue {
            start_ms,
            end_ms: end_ms.max(start_ms + 1),
            id,
            settings,
            payload,
        });
    }

    Ok((codec_id, cues))
}

/// Wraps a subtitle cue payload and optional settings into an ISO/IEC 14496-30 `wvtt`
/// MP4 sample (`vttc` CueBox containing optional `sttg` + `payl` CuePayloadBox).
pub fn wrap_mp4_wvtt_sample(payload_utf8: &[u8], settings: &str) -> Vec<u8> {
    // If already wrapped in a `vttc` or `vtte` box, return as-is
    if payload_utf8.len() >= 8 && (&payload_utf8[4..8] == b"vttc" || &payload_utf8[4..8] == b"vtte")
    {
        return payload_utf8.to_vec();
    }
    let mut vttc_body = Vec::with_capacity(16 + settings.len() + payload_utf8.len());
    if !settings.is_empty() {
        let sttg_len = (8 + settings.len()) as u32;
        vttc_body.extend_from_slice(&sttg_len.to_be_bytes());
        vttc_body.extend_from_slice(b"sttg");
        vttc_body.extend_from_slice(settings.as_bytes());
    }
    let payl_len = (8 + payload_utf8.len()) as u32;
    vttc_body.extend_from_slice(&payl_len.to_be_bytes());
    vttc_body.extend_from_slice(b"payl");
    vttc_body.extend_from_slice(payload_utf8);

    let total_len = (8 + vttc_body.len()) as u32;
    let mut out = Vec::with_capacity(total_len as usize);
    out.extend_from_slice(&total_len.to_be_bytes());
    out.extend_from_slice(b"vttc");
    out.extend_from_slice(&vttc_body);
    out
}

/// Unwraps an ISO/IEC 14496-30 `wvtt` MP4 sample (`vttc` -> `payl` / `sttg`) into `(payload_utf8, settings)`.
/// If `sample` is raw UTF-8 (e.g. from Matroska `S_TEXT/UTF8` or `S_TEXT/WEBVTT`), returns it directly.
pub fn unwrap_mp4_wvtt_sample(sample: &[u8]) -> (String, String) {
    if sample.len() >= 8 && &sample[4..8] == b"vttc" {
        let vttc_len = (u32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]]) as usize)
            .min(sample.len());
        let mut pos = 8usize;
        let mut text = String::new();
        let mut settings = String::new();
        while pos + 8 <= vttc_len {
            let sz = u32::from_be_bytes([
                sample[pos],
                sample[pos + 1],
                sample[pos + 2],
                sample[pos + 3],
            ]) as usize;
            if sz < 8 || pos + sz > vttc_len {
                break;
            }
            let tag = &sample[pos + 4..pos + 8];
            let body = &sample[pos + 8..pos + sz];
            if tag == b"payl" {
                text = String::from_utf8_lossy(body).into_owned();
            } else if tag == b"sttg" {
                settings = String::from_utf8_lossy(body).into_owned();
            }
            pos += sz;
        }
        return (text, settings);
    }
    if sample.len() >= 8 && &sample[4..8] == b"vtte" {
        return (String::new(), String::new());
    }
    (String::from_utf8_lossy(sample).into_owned(), String::new())
}

/// Standalone `.vtt` / `.srt` subtitle demuxer.
pub struct SubtitleDemuxer {
    track: TrackInfo,
    cues: Vec<SubtitleCue>,
    cursor: usize,
}

impl SubtitleDemuxer {
    pub fn open(data: &[u8]) -> Result<Self> {
        let (codec_id, cues) = parse_subtitles(data)?;
        let duration_ms = cues.last().map(|c| c.end_ms).unwrap_or(0);
        let extra_data = if codec_id == CodecId::WebVtt {
            b"WEBVTT\n".to_vec()
        } else {
            Vec::new()
        };
        let track = TrackInfo {
            index: 0,
            track_type: TrackType::Subtitle,
            codec_id,
            timebase: Rational64::new(1, 1000),
            duration_ticks: Some(duration_ms),
            extra_data,
            video: None,
            audio: None,
            sample_count: cues.len(),
            total_bytes: data.len() as u64,
        };
        Ok(Self {
            track,
            cues,
            cursor: 0,
        })
    }

    #[inline]
    pub const fn track(&self) -> &TrackInfo {
        &self.track
    }

    #[inline]
    pub fn cues(&self) -> &[SubtitleCue] {
        &self.cues
    }

    pub fn next_packet(&mut self) -> Result<Option<PacketSlice<'static>>> {
        if self.cursor >= self.cues.len() {
            return Ok(None);
        }
        let cue = &self.cues[self.cursor];
        self.cursor += 1;
        let tb = Rational64::new(1, 1000);
        let ts = Timestamp::new(cue.start_ms, tb);
        Ok(Some(PacketSlice::from_owned(
            AlignedBuffer::from_slice(cue.payload.as_bytes()),
            0,
            ts,
            ts,
            cue.duration_ms(),
            PacketFlags::KEYFRAME,
        )))
    }
}

/// Standalone `.vtt` (WebVTT) and `.srt` (SubRip) subtitle muxer.
pub struct SubtitleMuxer {
    codec_id: CodecId,
    cues: Vec<SubtitleCue>,
}

impl SubtitleMuxer {
    #[inline]
    pub const fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            cues: Vec::new(),
        }
    }

    pub fn add_cue(&mut self, cue: SubtitleCue) {
        self.cues.push(cue);
    }

    pub fn write_packet(&mut self, pkt: &PacketSlice<'_>) {
        let ms_tb = Rational64::new(1, 1000);
        let start_ms = pkt.pts.rescale_to(ms_tb).ticks.max(0);
        let dur_ms = rescale_q(pkt.duration.max(1), pkt.pts.timebase, ms_tb).max(1);
        let (payload, settings) = unwrap_mp4_wvtt_sample(pkt.data());
        if !payload.is_empty() {
            self.cues.push(SubtitleCue {
                start_ms,
                end_ms: start_ms + dur_ms,
                id: None,
                settings,
                payload,
            });
        }
    }

    pub fn finish<W: Write>(self, mut writer: W) -> Result<u64> {
        let mut out = String::new();
        if self.codec_id == CodecId::WebVtt {
            out.push_str("WEBVTT\n\n");
            for cue in &self.cues {
                if let Some(id) = &cue.id {
                    out.push_str(id);
                    out.push('\n');
                }
                let start = format_subtitle_timestamp_ms(cue.start_ms, '.');
                let end = format_subtitle_timestamp_ms(cue.end_ms, '.');
                if cue.settings.is_empty() {
                    out.push_str(&format!("{start} --> {end}\n"));
                } else {
                    out.push_str(&format!("{start} --> {end} {}\n", cue.settings));
                }
                out.push_str(&cue.payload);
                out.push_str("\n\n");
            }
        } else {
            for (idx, cue) in self.cues.iter().enumerate() {
                let num = idx + 1;
                let start = format_subtitle_timestamp_ms(cue.start_ms, ',');
                let end = format_subtitle_timestamp_ms(cue.end_ms, ',');
                out.push_str(&format!("{num}\n{start} --> {end}\n{}\n\n", cue.payload));
            }
        }
        writer.write_all(out.as_bytes())?;
        writer.flush()?;
        Ok(out.len() as u64)
    }
}
