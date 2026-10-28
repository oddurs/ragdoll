//! Picture order count, reference picture marking and reference list construction (ITU-T H.264 8.2).

use ragdoll_core::{RagdollError, Result};

use super::picture::{FrameStore, RefKind, Structure};
use super::ps::Sps;
use super::slice::{Mmco, RefMod, SliceHeader, SliceType};

/// State carried between pictures for POC derivation (8.2.1).
#[derive(Clone, Debug, Default)]
pub struct PocState {
    pub prev_poc_msb: i32,
    pub prev_poc_lsb: i32,
    pub prev_frame_num_offset: i32,
    pub prev_frame_num: u32,
}

/// Computes `(TopFieldOrderCnt, BottomFieldOrderCnt, FrameNumOffset, PicOrderCntMsb)`. For a field
/// picture only the entry of its own parity is meaningful.
pub fn compute_poc(st: &PocState, sps: &Sps, sh: &SliceHeader) -> (i32, i32, i32, i32) {
    let max_frame_num = 1i32 << sps.log2_max_frame_num;
    let frame_num = sh.frame_num as i32;
    let frame_num_offset = if sh.idr {
        0
    } else if st.prev_frame_num as i32 > frame_num {
        st.prev_frame_num_offset + max_frame_num
    } else {
        st.prev_frame_num_offset
    };
    match sps.poc_type {
        0 => {
            let (prev_msb, prev_lsb) = if sh.idr {
                (0, 0)
            } else {
                (st.prev_poc_msb, st.prev_poc_lsb)
            };
            let max_lsb = 1i32 << sps.log2_max_poc_lsb;
            let lsb = sh.poc_lsb as i32;
            let msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            };
            let top = msb + lsb;
            let bottom = if sh.field_pic {
                msb + lsb
            } else {
                top + sh.delta_poc_bottom
            };
            (top, bottom, frame_num_offset, msb)
        }
        1 => {
            let cycle_len = sps.offset_for_ref_frame.len() as i32;
            let mut abs_frame_num = if cycle_len != 0 {
                frame_num_offset + frame_num
            } else {
                0
            };
            if sh.nal_ref_idc == 0 && abs_frame_num > 0 {
                abs_frame_num -= 1;
            }
            let mut expected = 0i32;
            if abs_frame_num > 0 {
                let delta_cycle: i32 = sps.offset_for_ref_frame.iter().sum();
                let cycle_cnt = (abs_frame_num - 1) / cycle_len;
                let in_cycle = (abs_frame_num - 1) % cycle_len;
                expected = cycle_cnt * delta_cycle;
                for i in 0..=in_cycle as usize {
                    expected += sps.offset_for_ref_frame[i];
                }
            }
            if sh.nal_ref_idc == 0 {
                expected += sps.offset_for_non_ref_pic;
            }
            if !sh.field_pic {
                let top = expected + sh.delta_poc[0];
                (
                    top,
                    top + sps.offset_for_top_to_bottom_field + sh.delta_poc[1],
                    frame_num_offset,
                    0,
                )
            } else if !sh.bottom_field {
                let top = expected + sh.delta_poc[0];
                (top, top, frame_num_offset, 0)
            } else {
                let bottom = expected + sps.offset_for_top_to_bottom_field + sh.delta_poc[0];
                (bottom, bottom, frame_num_offset, 0)
            }
        }
        _ => {
            let temp = if sh.idr {
                0
            } else if sh.nal_ref_idc == 0 {
                2 * (frame_num_offset + frame_num) - 1
            } else {
                2 * (frame_num_offset + frame_num)
            };
            (temp, temp, frame_num_offset, 0)
        }
    }
}

/// One entry of a reference picture list: a frame store, as a frame or as one of its fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefSel {
    pub store: usize,
    pub parity: Option<u8>,
}

/// The decoded picture buffer: frame stores holding at least one reference field.
pub struct Dpb {
    pub pics: Vec<FrameStore>,
    /// `MaxLongTermFrameIdx`, `None` meaning "no long-term frame indices".
    pub max_long_term_frame_idx: Option<u32>,
}

impl Dpb {
    pub fn new() -> Self {
        Self {
            pics: Vec::new(),
            max_long_term_frame_idx: None,
        }
    }

    pub fn clear(&mut self) {
        self.pics.clear();
        self.max_long_term_frame_idx = None;
    }

    /// Removes frame stores with no reference field, except the store with id `keep`.
    pub fn drop_unused(&mut self, keep: Option<u32>) {
        self.pics.retain(|p| p.is_reference() || Some(p.id) == keep);
    }

    /// Updates `FrameNumWrap` of short-term references relative to the current `frame_num` (8.2.4.1).
    pub fn update_frame_num_wrap(&mut self, cur_frame_num: u32, max_frame_num: u32) {
        for p in &mut self.pics {
            if p.any(RefKind::Short) {
                p.frame_num_wrap = if p.frame_num > cur_frame_num {
                    p.frame_num as i32 - max_frame_num as i32
                } else {
                    p.frame_num as i32
                };
            }
        }
    }

    /// Sliding window marking (8.2.5.3), applied before marking a new short-term reference.
    /// The store with id `exclude` (the current picture) neither counts nor is removed.
    pub fn sliding_window(&mut self, max_num_ref_frames: u32, exclude: Option<u32>) {
        let num_short = self
            .pics
            .iter()
            .filter(|p| p.any(RefKind::Short) && Some(p.id) != exclude)
            .count();
        let num_long = self
            .pics
            .iter()
            .filter(|p| p.any(RefKind::Long) && Some(p.id) != exclude)
            .count();
        if num_short > 0
            && num_short + num_long >= max_num_ref_frames.max(1) as usize
            && let Some(oldest) = self
                .pics
                .iter_mut()
                .filter(|p| p.any(RefKind::Short) && Some(p.id) != exclude)
                .min_by_key(|p| p.frame_num_wrap)
        {
            for k in &mut oldest.kind {
                if *k == RefKind::Short {
                    *k = RefKind::Unused;
                }
            }
        }
    }

    /// The short-term picture (frame, or field for field decoding) with `PicNum == pic_num`.
    fn short_by_pic_num(&self, pic_num: i32, st: Structure) -> Option<(usize, Option<u8>)> {
        match st {
            Structure::Frame => self
                .pics
                .iter()
                .position(|p| p.kind == [RefKind::Short; 2] && p.frame_num_wrap == pic_num)
                .map(|i| (i, None)),
            Structure::Field(cur) => {
                for (i, p) in self.pics.iter().enumerate() {
                    for par in 0..2u8 {
                        let num = 2 * p.frame_num_wrap + i32::from(par == cur);
                        if p.kind[par as usize] == RefKind::Short && num == pic_num {
                            return Some((i, Some(par)));
                        }
                    }
                }
                None
            }
        }
    }

    /// The long-term picture with `LongTermPicNum == num`.
    fn long_by_pic_num(&self, num: u32, st: Structure) -> Option<(usize, Option<u8>)> {
        match st {
            Structure::Frame => self
                .pics
                .iter()
                .position(|p| p.kind == [RefKind::Long; 2] && p.long_term_frame_idx == num)
                .map(|i| (i, None)),
            Structure::Field(cur) => {
                for (i, p) in self.pics.iter().enumerate() {
                    for par in 0..2u8 {
                        let n = 2 * p.long_term_frame_idx + u32::from(par == cur);
                        if p.kind[par as usize] == RefKind::Long && n == num {
                            return Some((i, Some(par)));
                        }
                    }
                }
                None
            }
        }
    }

    fn set_kind(&mut self, i: usize, parity: Option<u8>, k: RefKind) {
        match parity {
            None => self.pics[i].kind = [k; 2],
            Some(p) => self.pics[i].kind[p as usize] = k,
        }
    }

    /// Unmarks long-term references using `idx`, except fields of the store with id `keep`.
    fn free_long_term_idx(&mut self, idx: u32, keep: Option<u32>) {
        for p in &mut self.pics {
            if Some(p.id) != keep && p.long_term_frame_idx == idx {
                for k in &mut p.kind {
                    if *k == RefKind::Long {
                        *k = RefKind::Unused;
                    }
                }
            }
        }
    }

    /// Applies memory management control operations (8.2.5.4) for the current picture, whose store
    /// (id `cur_id`) is already in the DPB. Returns `true` if the current picture was marked long-term.
    pub fn apply_mmcos(
        &mut self,
        ops: &[Mmco],
        cur_frame_num: u32,
        st: Structure,
        cur_id: u32,
    ) -> bool {
        let curr_pic_num = match st {
            Structure::Frame => cur_frame_num as i32,
            Structure::Field(_) => 2 * cur_frame_num as i32 + 1,
        };
        let mut cur_long = false;
        for op in ops {
            match *op {
                Mmco::UnmarkShort {
                    diff_pic_nums_minus1,
                } => {
                    let num = curr_pic_num - (diff_pic_nums_minus1 as i32 + 1);
                    if let Some((i, par)) = self.short_by_pic_num(num, st) {
                        self.set_kind(i, par, RefKind::Unused);
                    }
                }
                Mmco::UnmarkLong { long_term_pic_num } => {
                    if let Some((i, par)) = self.long_by_pic_num(long_term_pic_num, st) {
                        self.set_kind(i, par, RefKind::Unused);
                    }
                }
                Mmco::ShortToLong {
                    diff_pic_nums_minus1,
                    long_term_frame_idx,
                } => {
                    let num = curr_pic_num - (diff_pic_nums_minus1 as i32 + 1);
                    if let Some((i, par)) = self.short_by_pic_num(num, st) {
                        // A long-term index may be shared only by the two fields of one frame.
                        let p = &self.pics[i];
                        let keep = (par.is_some()
                            && p.any(RefKind::Long)
                            && p.long_term_frame_idx == long_term_frame_idx)
                            .then_some(p.id);
                        self.free_long_term_idx(long_term_frame_idx, keep);
                        self.set_kind(i, par, RefKind::Long);
                        self.pics[i].long_term_frame_idx = long_term_frame_idx;
                    }
                }
                Mmco::MaxLongTermIdx { max_plus1 } => {
                    self.max_long_term_frame_idx = max_plus1.checked_sub(1);
                    let max = self.max_long_term_frame_idx;
                    for p in &mut self.pics {
                        if max.is_none_or(|m| p.long_term_frame_idx > m) {
                            for k in &mut p.kind {
                                if *k == RefKind::Long {
                                    *k = RefKind::Unused;
                                }
                            }
                        }
                    }
                }
                Mmco::Reset => {
                    for p in &mut self.pics {
                        if p.id != cur_id {
                            p.kind = [RefKind::Unused; 2];
                        }
                    }
                    self.max_long_term_frame_idx = None;
                }
                Mmco::CurrentToLong {
                    long_term_frame_idx,
                } => {
                    self.free_long_term_idx(long_term_frame_idx, Some(cur_id));
                    if let Some(i) = self.pics.iter().position(|p| p.id == cur_id) {
                        let par = match st {
                            Structure::Frame => None,
                            Structure::Field(p) => Some(p),
                        };
                        self.set_kind(i, par, RefKind::Long);
                        self.pics[i].long_term_frame_idx = long_term_frame_idx;
                    }
                    cur_long = true;
                }
            }
        }
        cur_long
    }

    /// Builds `RefPicList0` / `RefPicList1` for a slice (8.2.4.2, 8.2.4.3). `cur_id` is the store
    /// of the current picture; when decoding a second field its first field is a reference.
    pub fn build_lists(
        &self,
        sh: &SliceHeader,
        st: Structure,
        cur_poc: i32,
        max_frame_num: u32,
        cur_id: Option<u32>,
    ) -> Result<[Vec<RefSel>; 2]> {
        let mut lists: [Vec<RefSel>; 2] = [Vec::new(), Vec::new()];
        if sh.slice_type.is_intra() {
            return Ok(lists);
        }
        let candidates = |k: RefKind| -> Vec<usize> {
            (0..self.pics.len())
                .filter(|&i| {
                    let p = &self.pics[i];
                    match st {
                        Structure::Frame => p.kind == [k; 2] && Some(p.id) != cur_id,
                        Structure::Field(_) => p.any(k),
                    }
                })
                .collect()
        };
        let short = candidates(RefKind::Short);
        let mut long = candidates(RefKind::Long);
        long.sort_by_key(|&i| self.pics[i].long_term_frame_idx);

        let is_b = sh.slice_type == SliceType::B;
        let mut init: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        if !is_b {
            let mut s = short;
            s.sort_by_key(|&i| std::cmp::Reverse(self.pics[i].frame_num_wrap));
            init[0] = s;
        } else {
            // Field decoding puts entries with POC equal to the current field's before it.
            let le = matches!(st, Structure::Field(_));
            let poc = |i: usize| self.pics[i].poc();
            let mut before: Vec<usize> = short
                .iter()
                .copied()
                .filter(|&i| {
                    if le {
                        poc(i) <= cur_poc
                    } else {
                        poc(i) < cur_poc
                    }
                })
                .collect();
            let mut after: Vec<usize> = short
                .iter()
                .copied()
                .filter(|&i| poc(i) > cur_poc)
                .collect();
            before.sort_by_key(|&i| std::cmp::Reverse(poc(i)));
            after.sort_by_key(|&i| poc(i));
            init[0] = before.iter().chain(after.iter()).copied().collect();
            init[1] = after.iter().chain(before.iter()).copied().collect();
        }

        let mut full: [Vec<RefSel>; 2] = [Vec::new(), Vec::new()];
        for list in 0..if is_b { 2 } else { 1 } {
            full[list] = match st {
                Structure::Frame => init[list]
                    .iter()
                    .chain(long.iter())
                    .map(|&i| RefSel {
                        store: i,
                        parity: None,
                    })
                    .collect(),
                Structure::Field(cur) => {
                    let mut v = self.alternate_fields(&init[list], RefKind::Short, cur);
                    v.extend(self.alternate_fields(&long, RefKind::Long, cur));
                    v
                }
            };
        }
        if is_b && full[1].len() > 1 && full[1] == full[0] {
            full[1].swap(0, 1);
        }

        for list in 0..2 {
            let n = sh.num_ref_idx_active[list] as usize;
            if n == 0 {
                continue;
            }
            let mut l: Vec<Option<RefSel>> = full[list].iter().copied().map(Some).collect();
            l.resize(n.max(l.len()), None);
            l.truncate(n);
            self.modify_list(&mut l, &sh.ref_mods[list], sh.frame_num, max_frame_num, st)?;
            // Missing entries point at the first available reference so damaged streams still decode.
            let fallback = l
                .iter()
                .flatten()
                .copied()
                .next()
                .or_else(|| full[list].first().copied());
            let mut out = Vec::with_capacity(n);
            for e in l {
                match e.or(fallback) {
                    Some(r) => out.push(r),
                    None => {
                        return Err(RagdollError::InvalidData(
                            "H.264 slice references pictures but the DPB is empty".into(),
                        ));
                    }
                }
            }
            lists[list] = out;
        }
        Ok(lists)
    }

    /// Field list from an ordered frame list: alternate parities starting with `cur` (8.2.4.2.5).
    fn alternate_fields(&self, frames: &[usize], k: RefKind, cur: u8) -> Vec<RefSel> {
        let mut out = Vec::new();
        let (mut i, mut j) = (0usize, 0usize);
        let has = |idx: usize, par: u8| self.pics[frames[idx]].kind[par as usize] == k;
        loop {
            while i < frames.len() && !has(i, cur) {
                i += 1;
            }
            while j < frames.len() && !has(j, cur ^ 1) {
                j += 1;
            }
            if i >= frames.len() && j >= frames.len() {
                break;
            }
            if i < frames.len() {
                out.push(RefSel {
                    store: frames[i],
                    parity: Some(cur),
                });
                i += 1;
            }
            if j < frames.len() {
                out.push(RefSel {
                    store: frames[j],
                    parity: Some(cur ^ 1),
                });
                j += 1;
            }
        }
        out
    }

    fn modify_list(
        &self,
        l: &mut Vec<Option<RefSel>>,
        mods: &[RefMod],
        cur_frame_num: u32,
        max_frame_num: u32,
        st: Structure,
    ) -> Result<()> {
        let n = l.len();
        let (max_pic_num, curr_pic_num) = match st {
            Structure::Frame => (max_frame_num as i32, cur_frame_num as i32),
            Structure::Field(_) => (2 * max_frame_num as i32, 2 * cur_frame_num as i32 + 1),
        };
        let mut pred = curr_pic_num;
        for (ref_idx, op) in mods.iter().enumerate() {
            if ref_idx >= n {
                break;
            }
            let pic = match *op {
                RefMod::Subtract(abs_diff) | RefMod::Add(abs_diff) => {
                    let delta = abs_diff as i32 + 1;
                    let mut no_wrap = if matches!(op, RefMod::Subtract(_)) {
                        pred - delta
                    } else {
                        pred + delta
                    };
                    if no_wrap < 0 {
                        no_wrap += max_pic_num;
                    } else if no_wrap >= max_pic_num {
                        no_wrap -= max_pic_num;
                    }
                    pred = no_wrap;
                    let pic_num = if no_wrap > curr_pic_num {
                        no_wrap - max_pic_num
                    } else {
                        no_wrap
                    };
                    self.short_by_pic_num(pic_num, st)
                }
                RefMod::LongTerm(lt) => self.long_by_pic_num(lt, st),
            };
            let Some((store, parity)) = pic else {
                return Err(RagdollError::InvalidData(
                    "H.264 reference list modification names a missing picture".into(),
                ));
            };
            let sel = Some(RefSel { store, parity });
            // Insert at ref_idx, then remove the later duplicate of the same picture.
            l.insert(ref_idx, sel);
            let mut w = ref_idx + 1;
            for c in ref_idx + 1..l.len() {
                if l[c] != sel {
                    l[w] = l[c];
                    w += 1;
                }
            }
            l.truncate(n);
        }
        Ok(())
    }
}
