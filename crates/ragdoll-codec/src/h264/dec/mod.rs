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
use dpb::{Dpb, PocState, RefSel, compute_poc};
use mb::{RefPic, SliceCtx};
use picture::{FrameStore, PicState, RefKind, Structure};
use ps::{Pps, Sps};
use recon::Dequant;
use slice::{Mmco, SliceHeader, SliceType};

struct CurPic {
    st: PicState,
    first: SliceHeader,
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    dq: Dequant,
    structure: Structure,
    top: i32,
    bottom: i32,
    frame_num_offset: i32,
    poc_msb: i32,
    slices: Vec<SliceDeblock>,
    pict_type: PictureType,
    store_id: u32,
    second_field: bool,
}

impl CurPic {
    /// POC of the picture being decoded: the frame POC or the field's own POC.
    fn poc(&self) -> i32 {
        match self.structure {
            Structure::Frame => self.top.min(self.bottom),
            Structure::Field(0) => self.top,
            Structure::Field(_) => self.bottom,
        }
    }
}

/// A first field waiting for its second field.
struct Pending {
    id: u32,
    parity: u8,
    frame_num: u32,
    is_ref: bool,
    /// The store itself when the field is not a reference (reference fields live in the DPB).
    store: Option<FrameStore>,
    crop_sps: Arc<Sps>,
    idr: bool,
    pict_type: PictureType,
    period: u32,
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
    pending: Option<Pending>,
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
            pending: None,
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
        if sps.chroma_array_type() > 1 || sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(RagdollError::Unsupported(
                "H.264 streams other than 8-bit 4:2:0 and 4:0:0 are not supported".into(),
            ));
        }
        let sh = SliceHeader::parse(rbsp, nal_unit_type, nal_ref_idc, &sps, &pps)?;
        if sps.mbaff && !sh.field_pic {
            return Err(RagdollError::Unsupported(
                "H.264 MBAFF (macroblock-adaptive frame/field) is not supported yet".into(),
            ));
        }
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
        let cur_poc = cur.poc();
        let lists = self.dpb.build_lists(
            &sh,
            cur.structure,
            cur_poc,
            max_frame_num,
            Some(cur.store_id),
        )?;
        let refs = lists.map(|l| {
            l.into_iter()
                .map(|sel| ref_pic(&self.dpb.pics[sel.store], sel))
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
        let structure = if sh.field_pic {
            Structure::Field(sh.bottom_field as u8)
        } else {
            Structure::Frame
        };
        let pairs_with_pending = match (structure, &self.pending) {
            (Structure::Field(p), Some(pd)) => {
                pd.parity != p
                    && pd.frame_num == sh.frame_num
                    && !sh.idr
                    && pd.is_ref == (sh.nal_ref_idc != 0)
            }
            _ => false,
        };
        if !pairs_with_pending {
            self.flush_pending();
        }
        let store_id = match &self.pending {
            Some(pd) if pairs_with_pending => pd.id,
            _ => {
                if sh.idr {
                    self.period += 1;
                } else if sh.frame_num != self.prev_ref_frame_num
                    && sh.frame_num != (self.prev_ref_frame_num + 1) % max_frame_num
                {
                    self.fill_frame_num_gap(sh, &sps);
                }
                self.alloc_id()
            }
        };
        let (top, bottom, frame_num_offset, poc_msb) = compute_poc(&self.poc, &sps, sh);
        self.dpb.update_frame_num_wrap(sh.frame_num, max_frame_num);
        let ws = ps::weight_scales(&sps, &pps);
        let mb_w = sps.width_mbs as usize;
        let mb_h = if sh.field_pic {
            sps.height_map_units as usize
        } else {
            sps.height_mbs() as usize
        };
        let pict_type = if sh.slice_type.is_intra() {
            PictureType::I
        } else {
            PictureType::P
        };
        self.cur = Some(CurPic {
            st: PicState::new(structure, mb_w, mb_h),
            first: sh.clone(),
            sps,
            pps,
            dq: Dequant::new(&ws),
            structure,
            top,
            bottom,
            frame_num_offset,
            poc_msb,
            slices: Vec::new(),
            pict_type,
            store_id,
            second_field: pairs_with_pending,
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
            .filter(|p| p.any(RefKind::Short))
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
            self.dpb.sliding_window(sps.max_num_ref_frames, None);
            self.dpb.drop_unused(None);
            let mut fake = sh.clone();
            fake.frame_num = fnum;
            fake.idr = false;
            fake.nal_ref_idc = 1;
            fake.field_pic = false;
            fake.delta_poc = [0, 0];
            let (top, bottom, offset, _) = compute_poc(&self.poc, sps, &fake);
            let id = self.alloc_id();
            let mut store = template.clone();
            store.id = id;
            store.field_poc = [top, bottom];
            store.decoded = [true, true];
            store.kind = [RefKind::Short; 2];
            store.frame_num = fnum;
            store.frame_num_wrap = fnum as i32;
            store.motion = Arc::new(picture::Motion::new(
                sps.width_mbs as usize,
                sps.height_mbs() as usize,
            ));
            self.dpb.pics.push(store);
            self.poc.prev_frame_num = fnum;
            self.poc.prev_frame_num_offset = offset;
            self.prev_ref_frame_num = fnum;
            fnum = (fnum + 1) % max_frame_num;
        }
    }

    fn alloc_id(&mut self) -> u32 {
        // Reference ids are `id * 4 + {0, 1, 2}`; keep them below `NO_PIC`.
        self.next_id = (self.next_id + 1) % (u32::MAX / 4 - 1);
        self.next_id
    }

    /// Outputs a first field that never got its second field.
    fn flush_pending(&mut self) {
        let Some(pd) = self.pending.take() else {
            return;
        };
        let store = match pd.store {
            Some(s) => Some(s),
            None => self.dpb.pics.iter().find(|p| p.id == pd.id).cloned(),
        };
        if let Some(store) = store {
            let frame = make_output(&store, &pd.crop_sps, pd.idr, pd.pict_type, pd.period);
            self.out.push(Output {
                frame,
                key: (pd.period, store.poc()),
            });
        }
    }

    /// Completes the current picture: deblocking, reference marking and output.
    pub fn finish_picture(&mut self) {
        let Some(mut cur) = self.cur.take() else {
            return;
        };
        let has_chroma = cur.sps.chroma_array_type() != 0;
        deblock::deblock_picture(&mut cur.st, &cur.slices, has_chroma);
        let sh = cur.first.clone();
        let sps = cur.sps.clone();
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        let is_ref = sh.nal_ref_idc != 0;
        let has_mmco5 = sh
            .mmcos
            .as_ref()
            .is_some_and(|ops| ops.contains(&Mmco::Reset));
        let parity = match cur.structure {
            Structure::Frame => None,
            Structure::Field(p) => Some(p),
        };

        // The frame store this picture belongs to.
        let pending_store = if cur.second_field {
            self.pending.as_mut().and_then(|p| p.store.take())
        } else {
            None
        };
        let mut store = match pending_store {
            Some(s) => s,
            None => match self.dpb.pics.iter().position(|p| p.id == cur.store_id) {
                Some(i) if cur.second_field => self.dpb.pics.swap_remove(i),
                _ => FrameStore::new(
                    cur.store_id,
                    cur.st.width,
                    cur.st.height * if parity.is_some() { 2 } else { 1 },
                    cur.st.mb_w,
                    cur.st.mb_h * if parity.is_some() { 2 } else { 1 },
                ),
            },
        };
        cur.st.write_into(&mut store);
        store.frame_num = sh.frame_num;
        match parity {
            None => {
                store.field_poc = [cur.top, cur.bottom];
                store.decoded = [true, true];
            }
            Some(p) => {
                store.field_poc[p as usize] = cur.poc();
                store.decoded[p as usize] = true;
            }
        }

        // Reference marking (8.2.5).
        let mut cur_long = false;
        if is_ref {
            if sh.idr {
                self.dpb.clear();
                if sh.long_term_reference {
                    self.dpb.max_long_term_frame_idx = Some(0);
                    store.long_term_frame_idx = 0;
                    cur_long = true;
                } else {
                    self.dpb.max_long_term_frame_idx = None;
                }
            } else {
                self.dpb.update_frame_num_wrap(sh.frame_num, max_frame_num);
                store.frame_num_wrap = sh.frame_num as i32;
                if let Some(ops) = &sh.mmcos {
                    // MMCOs see the current store (with the current picture not yet marked).
                    let id = store.id;
                    self.dpb.pics.push(store);
                    cur_long = self.dpb.apply_mmcos(ops, sh.frame_num, cur.structure, id);
                    let i = self
                        .dpb
                        .pics
                        .iter()
                        .position(|p| p.id == id)
                        .expect("current store");
                    store = self.dpb.pics.swap_remove(i);
                } else {
                    let first_field_is_ref = cur.second_field && store.is_reference();
                    if !first_field_is_ref {
                        self.dpb
                            .sliding_window(sps.max_num_ref_frames, Some(store.id));
                    }
                }
            }
            let kind = if cur_long {
                RefKind::Long
            } else {
                RefKind::Short
            };
            match parity {
                None => store.kind = [kind; 2],
                Some(p) => store.kind[p as usize] = kind,
            }
        }
        if has_mmco5 {
            match parity {
                None => {
                    let temp = cur.top.min(cur.bottom);
                    store.field_poc = [cur.top - temp, cur.bottom - temp];
                }
                Some(p) => store.field_poc[p as usize] = 0,
            }
            store.frame_num = 0;
            store.frame_num_wrap = 0;
            self.period += 1;
        }

        // POC state for the next picture.
        self.poc.prev_frame_num = if has_mmco5 { 0 } else { sh.frame_num };
        self.poc.prev_frame_num_offset = if has_mmco5 { 0 } else { cur.frame_num_offset };
        if is_ref {
            if has_mmco5 {
                self.poc.prev_poc_msb = 0;
                self.poc.prev_poc_lsb = if parity == Some(1) {
                    0
                } else {
                    store.field_poc[0]
                };
            } else {
                self.poc.prev_poc_msb = cur.poc_msb;
                self.poc.prev_poc_lsb = sh.poc_lsb as i32;
            }
            self.prev_ref_frame_num = if has_mmco5 { 0 } else { sh.frame_num };
        }

        // Store and output.
        let complete = parity.is_none() || cur.second_field;
        let (idr, pict_type, period) = match (&self.pending, cur.second_field) {
            (Some(pd), true) => (pd.idr, pd.pict_type, pd.period),
            _ => (sh.idr, cur.pict_type, self.period),
        };
        if complete {
            self.pending = None;
            let frame = make_output(&store, &sps, idr, pict_type, period);
            self.out.push(Output {
                frame,
                key: (period, store.poc()),
            });
        } else {
            self.pending = Some(Pending {
                id: store.id,
                parity: parity.unwrap_or(0),
                frame_num: sh.frame_num,
                is_ref,
                store: None,
                crop_sps: sps.clone(),
                idr,
                pict_type,
                period,
            });
        }
        if store.is_reference() {
            self.dpb.pics.push(store);
        } else if !complete && let Some(pd) = &mut self.pending {
            pd.store = Some(store);
        }
        let keep = self.pending.as_ref().map(|p| p.id);
        self.dpb.drop_unused(keep);
    }

    /// Removes and returns the oldest finished frame in decode order.
    pub fn pop_decoded(&mut self) -> Option<VideoFrame> {
        if self.out.is_empty() {
            None
        } else {
            Some(self.out.remove(0).frame)
        }
    }

    /// Completes any pending field and returns every finished frame in output (presentation) order.
    pub fn drain_output_order(&mut self) -> Vec<VideoFrame> {
        self.flush_pending();
        let mut out = std::mem::take(&mut self.out);
        out.sort_by_key(|o| o.key);
        out.into_iter().map(|o| o.frame).collect()
    }

    /// The frame rate signalled by any SPS, for stream-level metadata.
    pub fn fps(&self) -> Option<Rational64> {
        self.sps.iter().flatten().find_map(|s| s.fps)
    }
}

/// The view of a frame store a reference list entry selects.
fn ref_pic(store: &FrameStore, sel: RefSel) -> RefPic {
    match sel.parity {
        None => RefPic {
            id: store.frame_ref_id(),
            poc: store.poc(),
            long: store.kind == [RefKind::Long; 2],
            planes: store.planes.clone(),
            motion: Some(store.motion.clone()),
            parity: None,
            field_poc: store.field_poc,
        },
        Some(p) => RefPic {
            id: store.field_ref_id(p),
            poc: store.field_poc[p as usize],
            long: store.kind[p as usize] == RefKind::Long,
            planes: store.field_view(p),
            motion: Some(store.motion.clone()),
            parity: Some(p),
            field_poc: store.field_poc,
        },
    }
}

fn make_output(
    store: &FrameStore,
    sps: &Sps,
    idr: bool,
    pict_type: PictureType,
    period: u32,
) -> VideoFrame {
    let (x0, y0, w, h) = sps.crop_rect();
    let (x0, y0, w, h) = (x0 as usize, y0 as usize, w as usize, h as usize);
    let mut f = VideoFrame::new(w, h, PixelFormat::Yuv420p);
    let pw = store.width;
    for y in 0..h {
        let src = &store.planes[0][(y0 + y) * pw + x0..(y0 + y) * pw + x0 + w];
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
                row[..cw].copy_from_slice(&store.planes[1 + c][o..o + cw]);
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
    f.pts = Timestamp::new(((period as i64) << 32) + store.poc() as i64, tb);
    f
}
