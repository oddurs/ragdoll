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
mod threads;

use std::sync::Arc;

use ragdoll_core::{
    AlignedBuffer, CACHE_LINE_ALIGN, PictureType, PixelFormat, Plane, RagdollError, Rational64,
    Result, Timestamp, VideoFrame,
};

use deblock::SliceDeblock;
use dpb::{Dpb, PocState, RefSel, compute_poc};
use mb::{RefPic, SliceCtx};
use picture::{FrameStore, LiveMotion, Motion, PicPlanes, PicState, PicSync, RefKind, Structure};
use ps::{BASELINE, Pps, Sps};
use recon::Dequant;
use slice::{Mmco, SliceHeader, SliceType};
use threads::{JobDone, PicJob, SliceJob, Workers};

struct CurPic {
    /// The state of a picture decoded here, slice by slice; `None` for one decoded on a
    /// worker (see `job`).
    st: Option<PicState>,
    mb_w: usize,
    mb_h: usize,
    mbaff: bool,
    first: SliceHeader,
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    dq: Arc<Dequant>,
    structure: Structure,
    top: i32,
    bottom: i32,
    frame_num_offset: i32,
    poc_msb: i32,
    pict_type: PictureType,
    store_id: u32,
    second_field: bool,
    /// For a frame picture decoded on a worker: what it needs, collected until it is complete.
    job: Option<JobBuild>,
}

/// A frame picture to decode on a worker, as its slices arrive.
struct JobBuild {
    slices: Vec<SliceJob>,
    deblock: Vec<SliceDeblock>,
    planes: PicPlanes,
    fresh: bool,
    sync: Arc<PicSync>,
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
    frame: OutFrame,
    key: (u32, i32),
}

enum OutFrame {
    /// The frame, or the error its picture's slices met on a worker.
    Done(Result<VideoFrame>),
    /// A picture a worker is decoding, made a frame once it is complete.
    Decoding {
        store: FrameStore,
        sps: Arc<Sps>,
        idr: bool,
        pict_type: PictureType,
        period: u32,
        done: Arc<JobDone>,
    },
}

/// Frame threading state (see [`threads`]).
struct Mt {
    workers: Workers,
    /// Motion of completed pictures, reused once no reference holds it.
    motions: Vec<Arc<Motion>>,
}

/// Motion buffers [`Mt::motions`] keeps.
const MOTION_POOL: usize = 16;

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
    /// The last picture's state, whose buffers the next picture reuses.
    spare: Option<PicState>,
    /// Sample buffers of earlier pictures, reused once nothing reads them.
    planes: Vec<Arc<[AlignedBuffer; 3]>>,
    mt: Option<Mt>,
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
        c.set_threads(self.mt.as_ref().map_or(1, |m| m.workers.threads()));
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
            spare: None,
            planes: Vec::new(),
            mt: None,
        }
    }

    /// Decodes frame pictures on `threads` worker threads when it is above 1 (see [`threads`]);
    /// output is the same as with one. Pictures are then returned a few calls after the access
    /// unit that completes them, the rest by [`Self::flush`].
    pub fn set_threads(&mut self, threads: usize) {
        self.wait_all();
        self.mt = (threads > 1)
            .then(|| Workers::new(threads))
            .flatten()
            .map(|workers| Mt {
                workers,
                motions: Vec::new(),
            });
    }

    /// Waits until every picture decoding on a worker is complete.
    fn wait_all(&self) {
        for o in &self.out {
            if let OutFrame::Decoding { store, .. } = &o.frame
                && let Some(sync) = &store.sync
            {
                sync.wait(PicSync::DONE - 1);
            }
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
        // Field macroblocks of an MBAFF frame use field lists derived from the frame lists
        // (8.2.4.2.5): each frame becomes its field of the MB's parity followed by the other field.
        let field_refs = cur.mbaff.then(|| {
            [0u8, 1].map(|parity| {
                [&lists[0], &lists[1]].map(|l| {
                    l.iter()
                        .flat_map(|sel| [parity, parity ^ 1].map(|p| (sel.store, p)))
                        .map(|(store, p)| {
                            let sel = RefSel {
                                store,
                                parity: Some(p),
                            };
                            ref_pic(&self.dpb.pics[store], sel)
                        })
                        .collect()
                })
            })
        });
        let refs = lists.map(|l| {
            l.into_iter()
                .map(|sel| ref_pic(&self.dpb.pics[sel.store], sel))
                .collect()
        });
        let deblock = match (&mut cur.st, &mut cur.job) {
            (Some(st), _) => &mut st.rows.slices,
            (None, Some(job)) => &mut job.deblock,
            (None, None) => unreachable!("a picture is decoded here or on a worker"),
        };
        let slice_num = deblock.len() as u32;
        deblock.push(SliceDeblock {
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
        if let Some(job) = &mut cur.job {
            job.slices.push(SliceJob {
                sh,
                refs,
                slice_num,
                rbsp: rbsp.to_vec(),
            });
            return Ok(());
        }
        let cx = SliceCtx {
            sps: &cur.sps,
            pps: &cur.pps,
            sh: &sh,
            slice_num,
            refs,
            field_refs,
            cur_poc,
            field_poc: [cur.top, cur.bottom],
            dq: &cur.dq,
        };
        let st = cur.st.as_mut().expect("picture decoded here");
        mb::decode_slice(&cx, st, rbsp)
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
        // Progressive 4:2:0 frames decode on workers; the others (fields, MBAFF, monochrome)
        // here, once every earlier picture is complete.
        let threaded = self.mt.is_some()
            && structure == Structure::Frame
            && !sps.mbaff
            && sps.chroma_array_type() == 1;
        if !threaded {
            self.wait_all();
        }
        let mbaff = sps.mbaff && !sh.field_pic;
        let (st, job) = if threaded {
            // The worker's own state decodes into these planes.
            let (planes, fresh) = PicPlanes::take(&mut self.planes, mb_w * mb_h * 256);
            let job = JobBuild {
                slices: Vec::new(),
                deblock: Vec::new(),
                planes,
                fresh,
                sync: Arc::new(PicSync::default()),
            };
            (None, Some(job))
        } else {
            let spare = self.spare.take();
            let mut st = PicState::recycle(spare, structure, mb_w, mb_h, mbaff, &mut self.planes);
            // Monochrome pictures never write chroma samples.
            if st.recycled && sps.chroma_array_type() == 0 {
                st.planes[1].fill(128);
                st.planes[2].fill(128);
            }
            st.start_rows(sps.chroma_array_type() != 0, None);
            (Some(st), None)
        };
        self.cur = Some(CurPic {
            st,
            mb_w,
            mb_h,
            mbaff,
            first: sh.clone(),
            sps,
            pps,
            dq: Arc::new(Dequant::new(&ws)),
            structure,
            top,
            bottom,
            frame_num_offset,
            poc_msb,
            pict_type,
            store_id,
            second_field: pairs_with_pending,
            job,
        });
        Ok(())
    }

    /// Inserts "non-existing" frames for skipped `frame_num` values (8.2.5.2).
    fn fill_frame_num_gap(&mut self, sh: &SliceHeader, sps: &Sps) {
        // The non-existing frames copy a reference's samples: let it complete first.
        self.wait_all();
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
            store.sync = None;
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
                frame: OutFrame::Done(Ok(frame)),
                key: (pd.period, store.poc()),
            });
        }
    }

    /// Completes the current picture: deblocking, reference marking and output.
    pub fn finish_picture(&mut self) {
        let Some(mut cur) = self.cur.take() else {
            return;
        };
        let job = cur.job.take();
        if let Some(st) = &mut cur.st {
            st.finish_rows();
        }
        let (mb_w, mb_h) = (cur.mb_w, cur.mb_h);
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
        let (w, h) = (mb_w * 16, mb_h * 16);
        let mut store = if let Some(job) = &job {
            // The worker decodes into the store's samples.
            FrameStore::decoding(cur.store_id, w, h, job.planes.shared(), job.sync.clone())
        } else {
            let mut store = match pending_store {
                Some(s) => s,
                None => match self.dpb.pics.iter().position(|p| p.id == cur.store_id) {
                    Some(i) if cur.second_field => self.dpb.pics.swap_remove(i),
                    _ => self.dpb.new_store(
                        cur.store_id,
                        w,
                        h * if parity.is_some() { 2 } else { 1 },
                        mb_w,
                        mb_h * if parity.is_some() { 2 } else { 1 },
                        parity.is_some(),
                    ),
                },
            };
            let st = cur.st.as_mut().expect("picture decoded here");
            st.write_into(&mut store);
            store
        };
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
        let done = job.as_ref().map(|_| Arc::new(JobDone::default()));
        if complete {
            self.pending = None;
            let frame = match &done {
                Some(done) => OutFrame::Decoding {
                    store: store.clone(),
                    sps: sps.clone(),
                    idr,
                    pict_type,
                    period,
                    done: done.clone(),
                },
                None => OutFrame::Done(Ok(make_output(&store, &sps, idr, pict_type, period))),
            };
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
        } else {
            self.dpb.release(store);
        }
        let keep = self.pending.as_ref().map(|p| p.id);
        self.dpb.drop_unused(keep);
        match (job, done, &mut self.mt) {
            (Some(job), Some(done), Some(_)) => {
                self.throttle();
                let mt = self.mt.as_mut().expect("frame threading");
                // Baseline has no B slices: no direct prediction reads the motion (see
                // `Dec::colocated_mb`), which then stays in the worker's own buffers.
                if is_ref && cur.sps.profile_idc != BASELINE {
                    // Its motion, filled as rows are decoded, for direct prediction in later
                    // pictures; slice `i`'s macroblocks use reference table `i`.
                    let n = mb_w * mb_h * 16;
                    let mut motion = mt
                        .motions
                        .iter()
                        .position(|m| Arc::strong_count(m) == 1 && m.blocks.len() == n)
                        .and_then(|i| Arc::try_unwrap(mt.motions.swap_remove(i)).ok())
                        .unwrap_or_else(|| Motion::new(mb_w, mb_h));
                    motion.ref_tables.clear();
                    motion
                        .ref_tables
                        .extend(job.slices.iter().map(|s| mb::ref_ids(&s.refs)));
                    job.sync.set_live(LiveMotion::new(motion, mb_w));
                }
                mt.workers.submit(PicJob {
                    mb_w,
                    mb_h,
                    planes: job.planes,
                    fresh: job.fresh,
                    deblock: job.deblock,
                    sps: cur.sps,
                    pps: cur.pps,
                    dq: cur.dq,
                    cur_poc: cur.top.min(cur.bottom),
                    field_poc: [cur.top, cur.bottom],
                    slices: job.slices,
                    sync: job.sync,
                    done,
                });
            }
            _ => self.spare = cur.st,
        }
    }

    /// Bounds the pictures in flight to one more than there are workers: completes the oldest
    /// ones (which the others are likely waiting on) before another is submitted.
    fn throttle(&mut self) {
        let limit = self.mt.as_ref().map_or(0, |m| m.workers.threads() * 2 + 1);
        loop {
            let mut decoding = self
                .out
                .iter()
                .enumerate()
                .filter(|(_, o)| matches!(o.frame, OutFrame::Decoding { .. }));
            let Some((oldest, _)) = decoding.next() else {
                return;
            };
            if decoding.count() + 1 < limit {
                return;
            }
            self.complete_output(oldest);
        }
    }

    /// Waits for the picture of output entry `i` if it is being decoded, and makes it a frame
    /// (its state is kept for reuse).
    fn complete_output(&mut self, i: usize) {
        let o = &mut self.out[i];
        if matches!(o.frame, OutFrame::Done(_)) {
            return;
        }
        let OutFrame::Decoding {
            store,
            sps,
            idr,
            pict_type,
            period,
            done,
        } = std::mem::replace(
            &mut o.frame,
            OutFrame::Done(Err(RagdollError::BitstreamEndOfData)),
        )
        else {
            unreachable!()
        };
        let sync = store.sync.as_ref().expect("decoding store has progress");
        sync.wait(PicSync::DONE - 1);
        let res = done.take().expect("complete job left its result");
        o.frame = OutFrame::Done(res.map(|()| make_output(&store, &sps, idr, pict_type, period)));
        if let Some(mt) = &mut self.mt
            && let Some(m) = sync.done_motion()
        {
            mt.motions.push(m);
            if mt.motions.len() > MOTION_POOL {
                mt.motions.remove(0);
            }
        }
    }

    /// Removes and returns the oldest finished frame in decode order. With frame threading, a
    /// picture still being decoded is waited for when `wait` is set; otherwise there is no
    /// frame yet.
    pub fn pop_decoded(&mut self, wait: bool) -> Result<Option<VideoFrame>> {
        let Some(first) = self.out.first() else {
            return Ok(None);
        };
        if let OutFrame::Decoding { store, .. } = &first.frame {
            if !wait && store.pending_sync().is_some() {
                return Ok(None);
            }
            self.complete_output(0);
        }
        let OutFrame::Done(frame) = self.out.remove(0).frame else {
            unreachable!("completed above")
        };
        frame.map(Some)
    }

    /// Completes the current picture and any pending field, waits for pictures being decoded
    /// and returns every frame not yet returned, in decode order.
    pub fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        self.finish_picture();
        self.flush_pending();
        let mut frames = Vec::new();
        while let Some(f) = self.pop_decoded(true)? {
            frames.push(f);
        }
        Ok(frames)
    }

    /// Completes any pending field and returns every finished frame in output (presentation) order.
    pub fn drain_output_order(&mut self) -> Result<Vec<VideoFrame>> {
        self.flush_pending();
        for i in 0..self.out.len() {
            self.complete_output(i);
        }
        let mut out = std::mem::take(&mut self.out);
        out.sort_by_key(|o| o.key);
        out.into_iter()
            .map(|o| match o.frame {
                OutFrame::Done(f) => f,
                OutFrame::Decoding { .. } => unreachable!("completed above"),
            })
            .collect()
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
            motion: store.current_motion(),
            parity: None,
            field_poc: store.field_poc,
            sync: store.pending_sync().cloned(),
        },
        Some(p) => RefPic {
            id: store.field_ref_id(p),
            poc: store.field_poc[p as usize],
            long: store.kind[p as usize] == RefKind::Long,
            planes: store.field_view(p),
            motion: store.current_motion(),
            parity: Some(p),
            field_poc: store.field_poc,
            sync: None,
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
    let pw = store.width;
    let cw = w.div_ceil(2);
    // Without a left or top crop, and with rows on 64-byte boundaries, the frame shares the
    // picture's samples instead of copying them (they are copied on write instead).
    let share = x0 == 0
        && y0 == 0
        && sps.chroma_array_type() == 1
        && pw.is_multiple_of(2 * CACHE_LINE_ALIGN);
    let grey = vec![128u8; if sps.chroma_array_type() == 0 { cw } else { 0 }];
    let mut f = if share {
        let fmt = PixelFormat::Yuv420p;
        let planes = (0..3)
            .map(|p| {
                let (w, h) = fmt.plane_dimensions(w, h, p);
                let stride = if p == 0 { pw } else { pw / 2 };
                Plane::shared(store.planes.clone(), p, w, h, 1, stride)
            })
            .collect();
        VideoFrame::from_planes(w, h, fmt, planes)
    } else {
        VideoFrame::from_rows(w, h, PixelFormat::Yuv420p, |p, y| {
            if p == 0 {
                let o = (y0 + y) * pw + x0;
                &store.planes[0][o..o + w]
            } else if sps.chroma_array_type() == 0 {
                &grey
            } else {
                let o = (y0 / 2 + y) * (pw / 2) + x0 / 2;
                &store.planes[p][o..o + cw]
            }
        })
    };
    f.colorimetry = sps.colorimetry;
    f.keyframe = idr;
    f.pict_type = pict_type;
    // Order key within the stream: POC within the current IDR / MMCO5 period. Callers that need
    // presentation order sort by it (see `ragdoll_pipeline::VideoFrameStream`).
    let fps = sps.fps.unwrap_or(Rational64::FPS_30);
    let tb = Rational64::new(fps.den, fps.num * 2);
    f.pts = Timestamp::new(((period as i64) << 32) + store.poc() as i64, tb);
    f
}
