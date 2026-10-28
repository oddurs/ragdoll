//! H.264 decoder core: parameter sets, picture assembly, reference management and output.

mod cabac_se;
mod deblock;
mod dpb;
mod intra;
mod mb;
mod picture;
mod ps;
mod recon;
mod slice;
mod tables;

use std::sync::Arc;

use ragdoll_core::{
    PictureType, PixelFormat, RagdollError, Rational64, Result, Timestamp, VideoFrame,
};

use deblock::SliceDeblock;
use dpb::{Dpb, PocState, compute_poc};
use mb::{RefPic, SliceCtx};
use picture::{PicState, Picture, RefKind};
use ps::{Pps, Sps};
use recon::Dequant;
use slice::{Mmco, SliceHeader, SliceType};

struct CurPic {
    st: PicState,
    first: SliceHeader,
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    dq: Dequant,
    top: i32,
    bottom: i32,
    frame_num_offset: i32,
    poc_lsb_msb: i32,
    slices: Vec<SliceDeblock>,
    pict_type: PictureType,
}

/// A finished picture waiting to be returned, with its output-order key.
struct Output {
    frame: VideoFrame,
    key: (u32, i32),
}

/// The decoder state behind [`crate::h264::H264Decoder`].
pub struct Core {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    dpb: Dpb,
    poc: PocState,
    prev_ref_frame_num: u32,
    cur: Option<CurPic>,
    next_id: u32,
    period: u32,
    out: Vec<Output>,
}

impl Default for Core {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Core {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Core")
            .field("refs", &self.dpb.pics.len())
            .finish()
    }
}

impl Clone for Core {
    /// Decoder state is not cloneable mid-stream; a clone starts fresh with the same parameter sets.
    fn clone(&self) -> Self {
        let mut c = Self::new();
        c.sps = self.sps.clone();
        c.pps = self.pps.clone();
        c
    }
}

impl Core {
    pub fn new() -> Self {
        Self {
            sps: vec![None; 32],
            pps: vec![None; 256],
            dpb: Dpb::new(),
            poc: PocState::default(),
            prev_ref_frame_num: 0,
            cur: None,
            next_id: 0,
            period: 0,
            out: Vec::new(),
        }
    }

    /// Handles one NAL unit (header byte plus unescaped RBSP).
    pub fn decode_nal(&mut self, nal_unit_type: u8, nal_ref_idc: u8, rbsp: &[u8]) -> Result<()> {
        match nal_unit_type {
            1 | 5 => self.decode_slice_nal(nal_unit_type, nal_ref_idc, rbsp),
            7 => {
                let sps = Sps::parse(rbsp)?;
                let id = sps.id as usize;
                self.sps[id] = Some(Arc::new(sps));
                Ok(())
            }
            8 => {
                let sps_table: Vec<Option<Sps>> =
                    self.sps.iter().map(|s| s.as_deref().cloned()).collect();
                let pps = Pps::parse(rbsp, &sps_table)?;
                let id = pps.id as usize;
                self.pps[id] = Some(Arc::new(pps));
                Ok(())
            }
            // Access unit delimiter, end of sequence/stream: the current picture is complete.
            9..=11 => {
                self.finish_picture();
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn decode_slice_nal(&mut self, nal_unit_type: u8, nal_ref_idc: u8, rbsp: &[u8]) -> Result<()> {
        let pps_id = SliceHeader::peek_pps_id(rbsp)? as usize;
        let pps = self
            .pps
            .get(pps_id)
            .cloned()
            .flatten()
            .ok_or(RagdollError::InvalidSyntax(
                "H.264 slice references an unknown PPS",
            ))?;
        let sps = self.sps[pps.sps_id as usize]
            .clone()
            .ok_or(RagdollError::InvalidSyntax(
                "H.264 PPS references an unknown SPS",
            ))?;
        if !sps.frame_mbs_only {
            return Err(RagdollError::Unsupported(
                "H.264 interlaced coding is not supported yet".into(),
            ));
        }
        if sps.chroma_array_type() > 1 || sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(RagdollError::Unsupported(
                "H.264 streams other than 8-bit 4:2:0 and 4:0:0 are not supported".into(),
            ));
        }
        let sh = SliceHeader::parse(rbsp, nal_unit_type, nal_ref_idc, &sps, &pps)?;
        if sh.redundant_pic_cnt > 0 {
            return Ok(());
        }
        if let Some(cur) = &self.cur
            && (sh.starts_new_picture(&cur.first, &cur.sps) || !Arc::ptr_eq(&cur.sps, &sps))
        {
            self.finish_picture();
        }
        if self.cur.is_none() {
            self.start_picture(&sh, sps.clone(), pps.clone())?;
        }
        let cur = self.cur.as_mut().expect("picture started");
        let max_frame_num = 1u32 << cur.sps.log2_max_frame_num;
        let cur_poc = cur.top.min(cur.bottom);
        let lists = self.dpb.build_lists(&sh, cur_poc, max_frame_num)?;
        let refs = lists.map(|l| {
            l.into_iter()
                .map(|i| {
                    let p = &self.dpb.pics[i];
                    RefPic {
                        id: p.id,
                        poc: p.poc,
                        long: p.is_long(),
                        planes: p.planes.clone(),
                        motion: p.motion.clone(),
                    }
                })
                .collect()
        });
        let slice_num = cur.slices.len() as u32;
        cur.slices.push(SliceDeblock {
            idc: sh.disable_deblocking_filter_idc,
            alpha_offset: sh.alpha_offset,
            beta_offset: sh.beta_offset,
        });
        match sh.slice_type {
            SliceType::B => cur.pict_type = PictureType::B,
            SliceType::P | SliceType::Sp if cur.pict_type != PictureType::B => {
                cur.pict_type = PictureType::P
            }
            _ => {}
        }
        let cx = SliceCtx {
            sps: &cur.sps,
            pps: &cur.pps,
            sh: &sh,
            slice_num,
            refs,
            cur_poc,
            dq: &cur.dq,
        };
        mb::decode_slice(&cx, &mut cur.st, rbsp)
    }

    fn start_picture(&mut self, sh: &SliceHeader, sps: Arc<Sps>, pps: Arc<Pps>) -> Result<()> {
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        if sh.idr {
            self.period += 1;
        } else if sh.frame_num != self.prev_ref_frame_num
            && sh.frame_num != (self.prev_ref_frame_num + 1) % max_frame_num
        {
            self.fill_frame_num_gap(sh, &sps);
        }
        let (top, bottom, frame_num_offset) = compute_poc(&self.poc, &sps, sh, false);
        self.dpb.update_frame_num_wrap(sh.frame_num, max_frame_num);
        let ws = ps::weight_scales(&sps, &pps);
        let mb_w = sps.width_mbs as usize;
        let mb_h = sps.height_mbs() as usize;
        let pict_type = if sh.slice_type.is_intra() {
            PictureType::I
        } else {
            PictureType::P
        };
        self.cur = Some(CurPic {
            st: PicState::new(mb_w, mb_h),
            first: sh.clone(),
            sps,
            pps,
            dq: Dequant::new(&ws),
            top,
            bottom,
            frame_num_offset,
            poc_lsb_msb: top - sh.poc_lsb as i32,
            slices: Vec::new(),
            pict_type,
        });
        Ok(())
    }

    /// Inserts "non-existing" frames for skipped `frame_num` values (8.2.5.2).
    fn fill_frame_num_gap(&mut self, sh: &SliceHeader, sps: &Sps) {
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        let template = self
            .dpb
            .pics
            .iter()
            .filter(|p| p.is_short())
            .max_by_key(|p| p.frame_num_wrap)
            .cloned();
        let Some(template) = template else {
            return;
        };
        let mut fnum = (self.prev_ref_frame_num + 1) % max_frame_num;
        let mut guard = 0;
        while fnum != sh.frame_num && guard < max_frame_num {
            guard += 1;
            self.dpb.update_frame_num_wrap(fnum, max_frame_num);
            self.dpb.sliding_window(sps.max_num_ref_frames);
            let mut fake = sh.clone();
            fake.frame_num = fnum;
            fake.idr = false;
            fake.nal_ref_idc = 1;
            fake.delta_poc = [0, 0];
            let (top, bottom, offset) = compute_poc(&self.poc, sps, &fake, false);
            let id = self.alloc_id();
            self.dpb.pics.push(Picture {
                id,
                planes: template.planes.clone(),
                width: template.width,
                height: template.height,
                poc: top.min(bottom),
                top_poc: top,
                bottom_poc: bottom,
                frame_num: fnum,
                frame_num_wrap: fnum as i32,
                long_term_frame_idx: 0,
                kind: RefKind::Short,
                motion: None,
            });
            self.poc.prev_frame_num = fnum;
            self.poc.prev_frame_num_offset = offset;
            self.prev_ref_frame_num = fnum;
            fnum = (fnum + 1) % max_frame_num;
        }
    }

    fn alloc_id(&mut self) -> u32 {
        self.next_id = self.next_id.wrapping_add(1) % (u32::MAX - 1);
        self.next_id
    }

    /// Completes the current picture: deblocking, reference marking and output.
    pub fn finish_picture(&mut self) {
        let Some(mut cur) = self.cur.take() else {
            return;
        };
        let has_chroma = cur.sps.chroma_array_type() != 0;
        deblock::deblock_picture(&mut cur.st, &cur.slices, has_chroma);
        let sh = &cur.first;
        let sps = cur.sps.clone();
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        let mut top = cur.top;
        let mut bottom = cur.bottom;
        let has_mmco5 = sh
            .mmcos
            .as_ref()
            .is_some_and(|ops| ops.contains(&Mmco::Reset));
        let id = self.alloc_id();
        let motion = Arc::new(cur.st.take_motion());
        let planes = Arc::new(std::mem::take(&mut cur.st.planes));
        let mut pic = Picture {
            id,
            planes,
            width: cur.st.width,
            height: cur.st.height,
            poc: top.min(bottom),
            top_poc: top,
            bottom_poc: bottom,
            frame_num: sh.frame_num,
            frame_num_wrap: sh.frame_num as i32,
            long_term_frame_idx: 0,
            kind: RefKind::Short,
            motion: Some(motion),
        };

        if sh.nal_ref_idc != 0 {
            if sh.idr {
                self.dpb.clear();
                if sh.long_term_reference {
                    pic.kind = RefKind::Long;
                    self.dpb.max_long_term_frame_idx = Some(0);
                }
            } else if let Some(ops) = &sh.mmcos {
                self.dpb.update_frame_num_wrap(sh.frame_num, max_frame_num);
                if !self.dpb.apply_mmcos(ops, sh.frame_num, &mut pic) {
                    pic.kind = RefKind::Short;
                }
            } else {
                self.dpb.update_frame_num_wrap(sh.frame_num, max_frame_num);
                self.dpb.sliding_window(sps.max_num_ref_frames);
            }
        }
        if has_mmco5 {
            let temp = top.min(bottom);
            top -= temp;
            bottom -= temp;
            pic.top_poc = top;
            pic.bottom_poc = bottom;
            pic.poc = top.min(bottom);
            pic.frame_num = 0;
            pic.frame_num_wrap = 0;
            self.period += 1;
        }

        // POC state for the next picture.
        self.poc.prev_frame_num = if has_mmco5 { 0 } else { sh.frame_num };
        self.poc.prev_frame_num_offset = if has_mmco5 { 0 } else { cur.frame_num_offset };
        if sh.nal_ref_idc != 0 {
            if has_mmco5 {
                self.poc.prev_poc_msb = 0;
                self.poc.prev_poc_lsb = top;
            } else {
                self.poc.prev_poc_msb = cur.poc_lsb_msb;
                self.poc.prev_poc_lsb = sh.poc_lsb as i32;
            }
            self.prev_ref_frame_num = if has_mmco5 { 0 } else { sh.frame_num };
        }

        let frame = make_output(&pic, &sps, sh.idr, cur.pict_type, self.period);
        self.out.push(Output {
            frame,
            key: (self.period, pic.poc),
        });
        if sh.nal_ref_idc != 0 && pic.kind != RefKind::Unused {
            self.dpb.pics.push(pic);
        }
    }

    /// Removes and returns the oldest finished frame in decode order.
    pub fn pop_decoded(&mut self) -> Option<VideoFrame> {
        if self.out.is_empty() {
            None
        } else {
            Some(self.out.remove(0).frame)
        }
    }

    /// Returns every finished frame in output (presentation) order.
    pub fn drain_output_order(&mut self) -> Vec<VideoFrame> {
        let mut out = std::mem::take(&mut self.out);
        out.sort_by_key(|o| o.key);
        out.into_iter().map(|o| o.frame).collect()
    }

    /// The most recently activated SPS, for stream-level metadata.
    pub fn fps(&self) -> Option<Rational64> {
        self.sps.iter().flatten().find_map(|s| s.fps)
    }
}

fn make_output(
    pic: &Picture,
    sps: &Sps,
    idr: bool,
    pict_type: PictureType,
    period: u32,
) -> VideoFrame {
    let (x0, y0, w, h) = sps.crop_rect();
    let (x0, y0, w, h) = (x0 as usize, y0 as usize, w as usize, h as usize);
    let mut f = VideoFrame::new(w, h, PixelFormat::Yuv420p);
    let pw = pic.width;
    for y in 0..h {
        let src = &pic.planes[0][(y0 + y) * pw + x0..(y0 + y) * pw + x0 + w];
        f.plane_mut(0).row_mut(y)[..w].copy_from_slice(src);
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    for c in 0..2 {
        for y in 0..ch {
            let row = f.plane_mut(1 + c).row_mut(y);
            if sps.chroma_array_type() == 0 {
                row[..cw].fill(128);
            } else {
                let o = (y0 / 2 + y) * (pw / 2) + x0 / 2;
                row[..cw].copy_from_slice(&pic.planes[1 + c][o..o + cw]);
            }
        }
    }
    f.colorimetry = sps.colorimetry;
    f.keyframe = idr;
    f.pict_type = pict_type;
    // Order key within the stream: POC within the current IDR / MMCO5 period. Callers that need
    // presentation order sort by it (see `ragdoll_pipeline::order_h264_frames`).
    let fps = sps.fps.unwrap_or(Rational64::FPS_30);
    let tb = Rational64::new(fps.den, fps.num * 2);
    f.pts = Timestamp::new(((period as i64) << 32) + pic.poc as i64, tb);
    f
}
