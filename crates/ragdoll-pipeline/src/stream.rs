//! Bounded-memory video decoding (BUG-0072).
//!
//! [`VideoFrameStream`] decodes a container a batch at a time, in presentation order, so callers
//! hold one batch of decoded frames instead of the whole input. A batch is sized by the memory
//! budget: at least one frame, and for MP4 at least one GOP. The frames, their order and their
//! PTS are exactly what [`crate::decode_video_bytes`] returns for the same input.

use super::*;
use std::path::PathBuf;

/// A conformant H.264 stream outputs no picture after more than 16 pictures that follow it in
/// decoding order (`max_num_reorder_frames <= 16`), so holding 17 pictures and releasing the one
/// with the lowest POC gives the same order as sorting the whole run by POC.
const H264_REORDER_WINDOW: usize = 17;

/// Pictures released per batch by the sequential H.264 sources.
const SEQ_BATCH_FRAMES: usize = 32;

/// Video frames decoded a batch at a time, in presentation order.
pub struct VideoFrameStream<'a> {
    fps: Rational64,
    frame_count: usize,
    frame_bytes: usize,
    batch_bytes: usize,
    threads: usize,
    /// Next presentation index, for H.264 sources, whose PTS is renumbered across the stream.
    next_index: i64,
    src: Source<'a>,
    /// The mapping behind `data`, whose consumed raw (Y4M) frames are released as the stream
    /// moves on, and how far it has been released.
    mapping: Option<&'a MmapSource>,
    released: usize,
}

enum Source<'a> {
    Done,
    Single(Box<VideoFrame>),
    ImageSeq {
        files: Vec<PathBuf>,
        pos: usize,
    },
    Y4m(Box<(Y4mDemuxer<'a>, VideoTrackParams)>),
    Rgdi {
        pkts: Vec<PacketSlice<'a>>,
        pos: usize,
    },
    /// MP4: one decoder per IDR-started GOP, a wave of whole GOPs at a time, decoded in parallel.
    /// A GOP larger than the batch budget on its own is decoded sequentially instead.
    H264Gops {
        extra: Vec<u8>,
        hdr: Option<Vec<u8>>,
        gops: Vec<Vec<PacketSlice<'a>>>,
        pos: usize,
        large: Option<Box<SeqH264<'a>>>,
    },
    /// Matroska and MPEG-TS: one decoder over the whole stream.
    Seq(Box<SeqH264<'a>>),
}

enum Step {
    End,
    Frames(Vec<VideoFrame>),
    /// H.264 pictures, whose PTS is renumbered to the running presentation index.
    H264(Vec<VideoFrame>),
    /// Frames copied out of the input up to the given address, which may be released.
    Consumed(Vec<VideoFrame>, usize),
}

type AuSource<'a> = Box<dyn FnMut() -> Result<Option<Vec<u8>>> + 'a>;

/// One H.264 decoder fed access units in decoding order, decoding frames on `threads` threads.
/// Pictures leave in presentation order: sorted by POC within each run that starts at a
/// keyframe, as [`sort_keyframe_runs`] sorts a whole decoded GOP.
struct SeqH264<'a> {
    next_au: AuSource<'a>,
    dec: H264Decoder,
    window: Vec<VideoFrame>,
    ended: bool,
}

impl<'a> SeqH264<'a> {
    fn new(hdr: Option<&[u8]>, next_au: AuSource<'a>, threads: usize) -> Self {
        let mut dec = H264Decoder::new().with_threads(crate::h264_frame_threads(threads));
        if let Some(h) = hdr {
            let _ = dec.set_extradata(h);
        }
        Self {
            next_au,
            dec,
            window: Vec::new(),
            ended: false,
        }
    }

    /// Decodes until `want` pictures are released or the input ends.
    fn pull(&mut self, want: usize) -> Result<Vec<VideoFrame>> {
        let mut out = Vec::new();
        while out.len() < want && !self.ended {
            match (self.next_au)()? {
                Some(au) => {
                    if let Some(f) = self.dec.decode_annex_b_packet(&au)? {
                        self.take(f, &mut out);
                    }
                }
                None => {
                    // Frame threads still hold the last pictures.
                    for f in self.dec.flush()? {
                        self.take(f, &mut out);
                    }
                    self.ended = true;
                    release_sorted(&mut self.window, &mut out);
                }
            }
        }
        Ok(out)
    }
}

impl SeqH264<'_> {
    /// Adds decoded picture `f` (in decoding order) to the reorder window, releasing to `out`.
    fn take(&mut self, f: VideoFrame, out: &mut Vec<VideoFrame>) {
        if f.keyframe {
            release_sorted(&mut self.window, out);
        }
        self.window.push(f);
        if self.window.len() >= H264_REORDER_WINDOW {
            let mut best = 0;
            for (i, w) in self.window.iter().enumerate() {
                if w.pts.ticks < self.window[best].pts.ticks {
                    best = i;
                }
            }
            out.push(self.window.remove(best));
        }
    }
}

fn release_sorted(window: &mut Vec<VideoFrame>, out: &mut Vec<VideoFrame>) {
    window.sort_by_key(|f| f.pts.ticks);
    out.append(window);
}

/// Puts decode-order `H264Decoder` frames into presentation order. The decoder stamps PTS from
/// `pic_order_cnt`, which restarts at every IDR, so it only orders frames within one GOP: sorting
/// across GOPs interleaves them (BUG-0060). Sorts each run that starts at a keyframe by POC; the
/// stream then renumbers PTS across the whole stream.
fn sort_keyframe_runs(frames: &mut [VideoFrame]) {
    let mut start = 0;
    for i in 1..=frames.len() {
        if i == frames.len() || frames[i].keyframe {
            frames[start..i].sort_by_key(|f| f.pts.ticks);
            start = i;
        }
    }
}

fn yuv420_bytes(w: usize, h: usize) -> usize {
    (w * h * 3 / 2).max(4096)
}

impl<'a> VideoFrameStream<'a> {
    fn empty(budget_bytes: usize, threads: usize) -> Self {
        Self {
            fps: Rational64::FPS_30,
            frame_count: 0,
            frame_bytes: 4096,
            batch_bytes: budget_bytes.max(1),
            threads: resolve_thread_count(threads),
            next_index: 0,
            src: Source::Done,
            mapping: None,
            released: 0,
        }
    }

    /// Opens the video track of a container (or a single image or RGDI frame) held in memory.
    /// `budget_bytes` bounds each batch of decoded frames; `threads` is as for
    /// [`resolve_thread_count`].
    pub fn open(data: &'a [u8], budget_bytes: usize, threads: usize) -> Result<Self> {
        let mut s = Self::empty(budget_bytes, threads);
        if data.starts_with(b"RGDI") {
            s.single(RagdollIntraDecoder::new().decode_frame(data)?);
            return Ok(s);
        }
        let fmt = detect_container_format(data)
            .ok_or_else(|| RagdollError::InvalidHeader("unrecognized video container".into()))?;
        match fmt {
            ContainerFormat::Y4m => s.open_y4m(Y4mDemuxer::open(data)?)?,
            ContainerFormat::Ppm | ContainerFormat::Pgm => s.single_image(read_pnm_image(data)?),
            ContainerFormat::Png => s.single_image(read_png_image(data)?),
            ContainerFormat::Jpeg => s.single_image(read_jpeg_image(data)?),
            ContainerFormat::Mp4 => s.open_mp4(data)?,
            ContainerFormat::Matroska | ContainerFormat::WebM => s.open_mkv(data)?,
            ContainerFormat::MpegTs => s.open_ts(data)?,
            ContainerFormat::Wav
            | ContainerFormat::Flac
            | ContainerFormat::Ogg
            | ContainerFormat::WebVtt
            | ContainerFormat::SubRip => {
                return Err(RagdollError::Unsupported(
                    "audio/subtitle container has no video stream".into(),
                ));
            }
        }
        Ok(s)
    }

    /// Opens a numbered image sequence (`frame_%04d.png`) at 30 fps, decoding a batch of files at
    /// a time.
    pub fn open_image_sequence(pattern: &str, budget_bytes: usize, threads: usize) -> Result<Self> {
        let files = expand_image_sequence_inputs(pattern)?;
        let mut s = Self::empty(budget_bytes, threads);
        s.frame_count = files.len();
        s.src = Source::ImageSeq { files, pos: 0 };
        Ok(s)
    }

    /// [`Self::open`] on a mapped file. Raw Y4M frames already returned are released from the
    /// process (`MmapSource::release_range`), so peak RSS does not grow with the file.
    pub fn open_mapped(
        source: &'a MmapSource,
        budget_bytes: usize,
        threads: usize,
    ) -> Result<Self> {
        let data = source.as_slice();
        if detect_container_format(data) != Some(ContainerFormat::Y4m) {
            return Self::open(data, budget_bytes, threads);
        }
        let mut s = Self::empty(budget_bytes, threads);
        s.open_y4m(Y4mDemuxer::open_mapped(source)?)?;
        s.mapping = Some(source);
        Ok(s)
    }

    /// The track frame rate (30 fps for images and image sequences).
    pub fn frame_rate(&self) -> Rational64 {
        self.fps
    }

    /// Frames the stream will produce: one per video sample or packet in the container index.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    fn single(&mut self, f: VideoFrame) {
        self.frame_count = 1;
        self.src = Source::Single(Box::new(f));
    }

    fn single_image(&mut self, f: VideoFrame) {
        let yuv = if f.format == PixelFormat::Yuv420p {
            f
        } else {
            convert_frame(&f, PixelFormat::Yuv420p, f.colorimetry)
        };
        self.single(yuv);
    }

    fn open_y4m(&mut self, demux: Y4mDemuxer<'a>) -> Result<()> {
        let vp = demux
            .track()
            .video
            .clone()
            .ok_or_else(|| RagdollError::InvalidData("Y4M missing video params".into()))?;
        self.fps = vp.frame_rate;
        self.frame_count = demux.frame_count();
        self.frame_bytes = yuv420_bytes(vp.width as usize, vp.height as usize);
        self.src = Source::Y4m(Box::new((demux, vp)));
        Ok(())
    }

    fn open_mp4(&mut self, data: &'a [u8]) -> Result<()> {
        let mut demux = Mp4Demuxer::open(data)?;
        let v_track = demux
            .tracks()
            .iter()
            .find(|t| t.info.track_type == TrackType::Video)
            .ok_or_else(|| RagdollError::InvalidData("MP4 has no video track".into()))?
            .info
            .clone();
        if let Some(v) = &v_track.video {
            self.fps = v.frame_rate;
            self.frame_bytes = yuv420_bytes(v.width as usize, v.height as usize);
        }
        let mut pkts = Vec::new();
        while let Some(pkt) = demux.next_packet()? {
            if pkt.track_index == v_track.index {
                pkts.push(pkt);
            }
        }
        if pkts.first().is_some_and(|p| p.data().starts_with(b"RGDI")) {
            self.frame_count = pkts.len();
            self.src = Source::Rgdi { pkts, pos: 0 };
            return Ok(());
        }
        if v_track.codec_id != CodecId::H264 {
            return Ok(());
        }
        self.frame_count = pkts.len();
        let extra = v_track.extra_data;
        let hdr = (extra.len() >= 8).then(|| avcc_config_to_annex_b(&extra));
        let mut gops: Vec<Vec<PacketSlice<'a>>> = Vec::new();
        for pkt in pkts {
            // Each GOP gets its own decoder, so only split where the decoder can start: container
            // sync flags on non-IDR pictures would leave it without references.
            let starts_gop = pkt.is_keyframe() && avcc_has_idr(&extra, pkt.data());
            if starts_gop || gops.is_empty() {
                gops.push(vec![pkt]);
            } else if let Some(last) = gops.last_mut() {
                last.push(pkt);
            }
        }
        self.src = Source::H264Gops {
            extra,
            hdr,
            gops,
            pos: 0,
            large: None,
        };
        Ok(())
    }

    fn open_mkv(&mut self, data: &'a [u8]) -> Result<()> {
        let mut demux = MkvDemuxer::open(data)?;
        let v_track = demux
            .tracks()
            .iter()
            .find(|t| t.track_type == TrackType::Video)
            .ok_or_else(|| RagdollError::InvalidData("MKV has no video track".into()))?
            .clone();
        if let Some(v) = &v_track.video {
            self.fps = v.frame_rate;
            self.frame_bytes = yuv420_bytes(v.width as usize, v.height as usize);
        }
        let mut counter = MkvDemuxer::open(data)?;
        while let Some(pkt) = counter.next_packet()? {
            self.frame_count += (pkt.track_index == v_track.index) as usize;
        }
        let hdr = (v_track.codec_id == CodecId::H264 && v_track.extra_data.len() >= 8)
            .then(|| avcc_config_to_annex_b(&v_track.extra_data));
        let next_au: AuSource<'a> = Box::new(move || {
            while let Some(pkt) = demux.next_packet()? {
                if pkt.track_index == v_track.index {
                    return Ok(Some(h264_avcc_to_annex_b_au(
                        &v_track.extra_data,
                        pkt.data(),
                        pkt.is_keyframe(),
                    )));
                }
            }
            Ok(None)
        });
        self.src = Source::Seq(Box::new(SeqH264::new(
            hdr.as_deref(),
            next_au,
            self.threads,
        )));
        Ok(())
    }

    fn open_ts(&mut self, data: &'a [u8]) -> Result<()> {
        let mut counter = MpegTsDemuxer::open(data)?;
        let video: Vec<u32> = counter
            .tracks()
            .iter()
            .filter(|t| t.track_type == TrackType::Video)
            .map(|t| t.index)
            .collect();
        if let Some(v) = counter
            .tracks()
            .iter()
            .find(|t| t.track_type == TrackType::Video)
            .and_then(|t| t.video.as_ref())
        {
            self.frame_bytes = yuv420_bytes(v.width as usize, v.height as usize);
        }
        while let Some(pkt) = counter.next_packet()? {
            self.frame_count += video.contains(&pkt.track_index) as usize;
        }
        // Every packet goes to the decoder; audio payloads carry no slices and yield no picture.
        let mut demux = MpegTsDemuxer::open(data)?;
        let next_au: AuSource<'a> =
            Box::new(move || Ok(demux.next_packet()?.map(|p| p.data().to_vec())));
        self.src = Source::Seq(Box::new(SeqH264::new(None, next_au, self.threads)));
        Ok(())
    }

    /// Returns the next batch of frames in presentation order, or `None` at the end.
    pub fn next_batch(&mut self) -> Result<Option<Vec<VideoFrame>>> {
        loop {
            let step = step(
                &mut self.src,
                self.threads,
                self.batch_bytes,
                self.frame_bytes,
                self.fps,
            )?;
            match step {
                Step::End => {
                    self.src = Source::Done;
                    return Ok(None);
                }
                Step::Frames(f) if !f.is_empty() => return Ok(Some(f)),
                Step::Consumed(f, end_addr) => {
                    if let Some(m) = self.mapping {
                        let end = end_addr.saturating_sub(m.as_slice().as_ptr() as usize);
                        m.release_range(self.released, end.saturating_sub(self.released));
                        self.released = end;
                    }
                    if !f.is_empty() {
                        return Ok(Some(f));
                    }
                }
                Step::H264(mut f) if !f.is_empty() => {
                    for frame in &mut f {
                        frame.pts = Timestamp::new(self.next_index, frame.pts.timebase);
                        self.next_index += 1;
                    }
                    return Ok(Some(f));
                }
                _ => {}
            }
        }
    }

    /// Decodes everything that is left into one vector.
    pub fn collect_all(mut self) -> Result<Vec<VideoFrame>> {
        let mut all = Vec::with_capacity(self.frame_count);
        while let Some(b) = self.next_batch()? {
            all.extend(b);
        }
        Ok(all)
    }
}

fn step<'a>(
    src: &mut Source<'a>,
    threads: usize,
    batch_bytes: usize,
    frame_bytes: usize,
    fps: Rational64,
) -> Result<Step> {
    let batch_frames = (batch_bytes / frame_bytes.max(1)).max(1);
    Ok(match src {
        Source::Done => Step::End,
        Source::Single(_) => match std::mem::replace(src, Source::Done) {
            Source::Single(f) => Step::Frames(vec![*f]),
            _ => unreachable!(),
        },
        Source::ImageSeq { files, pos } => {
            if *pos >= files.len() {
                return Ok(Step::End);
            }
            let start = *pos;
            let n = (threads * 4).max(1).min(files.len() - start);
            *pos += n;
            let decoded = parallel_gop_map(&files[start..start + n], threads, |i, path| {
                decode_image_file(path).map(|img| {
                    let mut yuv = if img.format == PixelFormat::Yuv420p {
                        img
                    } else {
                        convert_frame(&img, PixelFormat::Yuv420p, img.colorimetry)
                    };
                    yuv.pts = Timestamp::new((start + i) as i64, fps.recip());
                    yuv
                })
            });
            Step::Frames(decoded.into_iter().collect::<Result<Vec<_>>>()?)
        }
        Source::Y4m(y4m) => {
            let (demux, vp) = &mut **y4m;
            let mut pkts = Vec::with_capacity(batch_frames.min(1 << 12));
            while pkts.len() < batch_frames {
                match demux.next_packet()? {
                    Some(p) => pkts.push(p),
                    None => break,
                }
            }
            if pkts.is_empty() {
                return Ok(Step::End);
            }
            let vp = &*vp;
            let frames = parallel_gop_map(&pkts, threads, |_, pkt| y4m_packet_to_frame(vp, pkt));
            let last = pkts[pkts.len() - 1].data();
            Step::Consumed(frames, last.as_ptr() as usize + last.len())
        }
        Source::Rgdi { pkts, pos } => {
            if *pos >= pkts.len() {
                return Ok(Step::End);
            }
            let dec = RagdollIntraDecoder::new();
            let end = (*pos + batch_frames).min(pkts.len());
            let mut out = Vec::with_capacity(end - *pos);
            for pkt in &pkts[*pos..end] {
                out.push(dec.decode_frame(pkt.data())?);
            }
            *pos = end;
            Step::Frames(out)
        }
        Source::Seq(seq) => {
            let out = seq.pull(SEQ_BATCH_FRAMES)?;
            if out.is_empty() && seq.ended {
                Step::End
            } else {
                Step::H264(out)
            }
        }
        Source::H264Gops {
            extra,
            hdr,
            gops,
            pos,
            large,
        } => {
            if let Some(seq) = large {
                let out = seq.pull(SEQ_BATCH_FRAMES)?;
                if seq.ended {
                    *large = None;
                }
                return Ok(Step::H264(out));
            }
            if *pos >= gops.len() {
                return Ok(Step::End);
            }
            if gops[*pos].len() > batch_frames {
                let gop = std::mem::take(&mut gops[*pos]);
                *pos += 1;
                let extra = extra.clone();
                let mut pkts = gop.into_iter();
                let next_au: AuSource<'a> = Box::new(move || {
                    Ok(pkts
                        .next()
                        .map(|p| h264_avcc_to_annex_b_au(&extra, p.data(), p.is_keyframe())))
                });
                *large = Some(Box::new(SeqH264::new(hdr.as_deref(), next_au, threads)));
                return Ok(Step::H264(Vec::new()));
            }
            // As many whole GOPs as fit the batch budget, at least one.
            let mut end = *pos;
            let mut frames = 0;
            while end < gops.len() && (end == *pos || frames + gops[end].len() <= batch_frames) {
                frames += gops[end].len();
                end += 1;
            }
            let (extra, hdr) = (&*extra, hdr.as_deref());
            let results: Vec<Result<Vec<VideoFrame>>> =
                parallel_gop_map(&gops[*pos..end], threads, |_, gop_pkts| {
                    let mut dec = H264Decoder::new();
                    if let Some(h) = hdr {
                        let _ = dec.set_extradata(h);
                    }
                    let mut gf = Vec::with_capacity(gop_pkts.len());
                    for pkt in gop_pkts {
                        let au = h264_avcc_to_annex_b_au(extra, pkt.data(), pkt.is_keyframe());
                        if let Some(f) = dec.decode_annex_b_packet(&au)? {
                            gf.push(f);
                        }
                    }
                    Ok(gf)
                });
            // Release the decoded GOPs' packet lists as we go.
            for g in &mut gops[*pos..end] {
                *g = Vec::new();
            }
            *pos = end;
            let mut out = Vec::with_capacity(frames);
            for r in results {
                out.extend(r?);
            }
            sort_keyframe_runs(&mut out);
            Step::H264(out)
        }
    })
}

pub(crate) fn y4m_packet_to_frame(vp: &VideoTrackParams, pkt: &PacketSlice<'_>) -> VideoFrame {
    let mut dst = VideoFrame::new(vp.width as usize, vp.height as usize, vp.pixel_format);
    dst.pts = pkt.pts;
    dst.colorimetry = vp.colorimetry;
    let raw = pkt.data();
    let mut src_off = 0usize;
    for plane in &mut dst.planes {
        let row_bytes = plane.row_bytes();
        let h = plane.height();
        if plane.stride() == row_bytes {
            let total = row_bytes * h;
            plane.data_mut()[..total].copy_from_slice(&raw[src_off..src_off + total]);
            src_off += total;
        } else {
            for y in 0..h {
                plane
                    .row_mut(y)
                    .copy_from_slice(&raw[src_off..src_off + row_bytes]);
                src_off += row_bytes;
            }
        }
    }
    dst
}

/// The `transcode` filter chain of [`apply_video_filters`] (tonemap, crop, pad, rotate, overlay,
/// then `--fps` resampling), applied a batch at a time to a stream of `input_count` frames.
pub struct VideoFilterChain {
    opts: TranscodeOptions,
    crop_rect: Option<CropRect>,
    overlay: Option<VideoFrame>,
    spatial: bool,
    /// `(source fps, target fps, output frame count)` when `--fps` changes the rate.
    resample: Option<(f64, f64, usize)>,
    input_count: usize,
    fps: Rational64,
    in_index: usize,
    out_index: usize,
    last: Option<VideoFrame>,
}

impl VideoFilterChain {
    /// Prepares the chain from the stream's first decoded frame (`--auto-crop` and `--crop`
    /// measure it), its frame rate and its frame count.
    pub fn new(
        first: &VideoFrame,
        fps: Rational64,
        input_count: usize,
        opts: &TranscodeOptions,
    ) -> Result<Self> {
        let auto_rect = opts.auto_crop.then(|| detect_auto_crop(first, 20));
        let overlay = match opts.overlay_path {
            Some(ref p) => Some(decode_image_file(p)?),
            None => None,
        };
        let crop_rect = match opts.crop_spec {
            Some(ref spec) => Some(CropRect::parse_spec(spec, first.width, first.height)?),
            None => auto_rect,
        };
        let spatial = opts.tonemap
            || auto_rect.is_some()
            || opts.crop_spec.is_some()
            || opts.pad_spec.is_some()
            || opts.rotate_mode.is_some()
            || overlay.is_some();
        let mut out_fps = fps;
        let mut resample = None;
        if let Some(dst) = opts.target_fps
            && dst.num > 0
            && dst.den > 0
            && (dst.num * fps.den != fps.num * dst.den)
        {
            let src_f = fps.as_f64().max(1.0);
            let dst_f = dst.as_f64().max(1.0);
            let out_count = ((input_count as f64) * dst_f / src_f).round().max(1.0) as usize;
            resample = Some((src_f, dst_f, out_count));
            out_fps = dst;
        }
        Ok(Self {
            opts: opts.clone(),
            crop_rect,
            overlay,
            spatial,
            resample,
            input_count,
            fps: out_fps,
            in_index: 0,
            out_index: 0,
            last: None,
        })
    }

    /// The output frame rate.
    pub fn frame_rate(&self) -> Rational64 {
        self.fps
    }

    /// Frames the chain will output for the whole stream.
    pub fn output_count(&self) -> usize {
        self.resample.map_or(self.input_count, |r| r.2)
    }

    /// Filters the next batch of input frames.
    pub fn apply(&mut self, frames: Vec<VideoFrame>, threads: usize) -> Vec<VideoFrame> {
        let frames = if self.spatial {
            let (opts, crop_rect, overlay) = (&self.opts, self.crop_rect, self.overlay.as_ref());
            parallel_gop_map(&frames, threads, |_, f| {
                let mut cur = if opts.tonemap {
                    tonemap_hdr_to_sdr_oklab(f, PixelFormat::Yuv420p)
                } else {
                    f.clone()
                };
                if let Some(r) = crop_rect
                    && let Ok(c) = crop_frame(&cur, r)
                {
                    cur = c;
                }
                if let Some(ref p_spec) = opts.pad_spec
                    && let Ok(ps) = PadSpec::parse_spec(p_spec, cur.width, cur.height)
                    && let Ok(p) = pad_frame(&cur, ps)
                {
                    cur = p;
                }
                if let Some(rm) = opts.rotate_mode {
                    cur = rotate_frame(&cur, rm);
                }
                if let Some(ov) = overlay {
                    overlay_rgba_onto_frame(&mut cur, ov, opts.overlay_xy.0, opts.overlay_xy.1);
                }
                cur
            })
        } else {
            frames
        };
        let Some((src_f, dst_f, out_count)) = self.resample else {
            self.in_index += frames.len();
            return frames;
        };
        let last_in = self.input_count.max(1) - 1;
        let src_of =
            |out_i: usize| (((out_i as f64) * src_f / dst_f).floor() as usize).min(last_in);
        let mut out = Vec::new();
        for f in frames {
            while self.out_index < out_count && src_of(self.out_index) == self.in_index {
                let mut c = f.clone();
                c.pts = Timestamp::new(self.out_index as i64, self.fps.recip());
                out.push(c);
                self.out_index += 1;
            }
            self.in_index += 1;
            self.last = Some(f);
        }
        out
    }

    /// Outputs what `--fps` still owes when the stream ended before `input_count` frames.
    pub fn finish(&mut self) -> Vec<VideoFrame> {
        let mut out = Vec::new();
        if let (Some((_, _, out_count)), Some(last)) = (self.resample, self.last.as_ref()) {
            while self.out_index < out_count {
                let mut c = last.clone();
                c.pts = Timestamp::new(self.out_index as i64, self.fps.recip());
                out.push(c);
                self.out_index += 1;
            }
        }
        out
    }
}

/// A [`VideoFrameStream`] run through a [`VideoFilterChain`]: non-empty batches of filtered
/// frames, in order.
pub(crate) struct FilteredFrames<'a> {
    stream: VideoFrameStream<'a>,
    pub(crate) filters: VideoFilterChain,
    first: Option<Vec<VideoFrame>>,
    threads: usize,
    finished: bool,
}

impl<'a> FilteredFrames<'a> {
    /// Opens the chain on the stream's first frame. `None` when the stream has no frames.
    pub(crate) fn new(
        mut stream: VideoFrameStream<'a>,
        opts: &TranscodeOptions,
        threads: usize,
    ) -> Result<Option<Self>> {
        let Some(first) = stream.next_batch()? else {
            return Ok(None);
        };
        let filters =
            VideoFilterChain::new(&first[0], stream.frame_rate(), stream.frame_count(), opts)?;
        Ok(Some(Self {
            stream,
            filters,
            first: Some(first),
            threads,
            finished: false,
        }))
    }

    pub(crate) fn next(&mut self) -> Result<Option<Vec<VideoFrame>>> {
        loop {
            if self.finished {
                return Ok(None);
            }
            let raw = match self.first.take() {
                Some(b) => Some(b),
                None => self.stream.next_batch()?,
            };
            let out = match raw {
                Some(b) => self.filters.apply(b, self.threads),
                None => {
                    self.finished = true;
                    self.filters.finish()
                }
            };
            if !out.is_empty() {
                return Ok(Some(out));
            }
        }
    }
}
