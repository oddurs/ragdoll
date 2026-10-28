//! Picture order count, reference picture marking and reference list construction (ITU-T H.264 8.2).

use ragdoll_core::{RagdollError, Result};

use super::picture::{Picture, RefKind};
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

/// Computes `(TopFieldOrderCnt, BottomFieldOrderCnt, FrameNumOffset)` for a frame.
pub fn compute_poc(
    st: &PocState,
    sps: &Sps,
    sh: &SliceHeader,
    prev_had_mmco5: bool,
) -> (i32, i32, i32) {
    let max_frame_num = 1i32 << sps.log2_max_frame_num;
    let frame_num = sh.frame_num as i32;
    let prev_offset = if prev_had_mmco5 {
        0
    } else {
        st.prev_frame_num_offset
    };
    let frame_num_offset = if sh.idr {
        0
    } else if st.prev_frame_num as i32 > frame_num {
        prev_offset + max_frame_num
    } else {
        prev_offset
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
            (top, top + sh.delta_poc_bottom, frame_num_offset)
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
            let top = expected + sh.delta_poc[0];
            let bottom = top + sps.offset_for_top_to_bottom_field + sh.delta_poc[1];
            (top, bottom, frame_num_offset)
        }
        _ => {
            let temp = if sh.idr {
                0
            } else if sh.nal_ref_idc == 0 {
                2 * (frame_num_offset + frame_num) - 1
            } else {
                2 * (frame_num_offset + frame_num)
            };
            (temp, temp, frame_num_offset)
        }
    }
}

/// The decoded picture buffer: reference pictures only (output happens when a picture completes).
pub struct Dpb {
    pub pics: Vec<Picture>,
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

    fn drop_unused(&mut self) {
        self.pics.retain(|p| p.kind != RefKind::Unused);
    }

    /// Updates `FrameNumWrap` of short-term references relative to the current `frame_num` (8.2.4.1).
    pub fn update_frame_num_wrap(&mut self, cur_frame_num: u32, max_frame_num: u32) {
        for p in &mut self.pics {
            if p.is_short() {
                p.frame_num_wrap = if p.frame_num > cur_frame_num {
                    p.frame_num as i32 - max_frame_num as i32
                } else {
                    p.frame_num as i32
                };
            }
        }
    }

    /// Sliding window marking (8.2.5.3), applied before storing a new short-term reference.
    pub fn sliding_window(&mut self, max_num_ref_frames: u32) {
        let num_short = self.pics.iter().filter(|p| p.is_short()).count();
        let num_long = self.pics.iter().filter(|p| p.is_long()).count();
        if num_short > 0
            && num_short + num_long >= max_num_ref_frames.max(1) as usize
            && let Some(oldest) = self
                .pics
                .iter_mut()
                .filter(|p| p.is_short())
                .min_by_key(|p| p.frame_num_wrap)
        {
            oldest.kind = RefKind::Unused;
        }
        self.drop_unused();
    }

    /// Applies memory management control operations (8.2.5.4). Returns `true` if the current
    /// picture was marked long-term (MMCO 6).
    pub fn apply_mmcos(&mut self, ops: &[Mmco], cur_frame_num: u32, cur: &mut Picture) -> bool {
        let curr_pic_num = cur_frame_num as i32;
        let mut cur_long = false;
        for op in ops {
            match *op {
                Mmco::UnmarkShort {
                    diff_pic_nums_minus1,
                } => {
                    let pic_num = curr_pic_num - (diff_pic_nums_minus1 as i32 + 1);
                    for p in &mut self.pics {
                        if p.is_short() && p.frame_num_wrap == pic_num {
                            p.kind = RefKind::Unused;
                        }
                    }
                }
                Mmco::UnmarkLong { long_term_pic_num } => {
                    for p in &mut self.pics {
                        if p.is_long() && p.long_term_frame_idx == long_term_pic_num {
                            p.kind = RefKind::Unused;
                        }
                    }
                }
                Mmco::ShortToLong {
                    diff_pic_nums_minus1,
                    long_term_frame_idx,
                } => {
                    let pic_num = curr_pic_num - (diff_pic_nums_minus1 as i32 + 1);
                    let target = self
                        .pics
                        .iter()
                        .position(|p| p.is_short() && p.frame_num_wrap == pic_num);
                    for (i, p) in self.pics.iter_mut().enumerate() {
                        if p.is_long()
                            && p.long_term_frame_idx == long_term_frame_idx
                            && Some(i) != target
                        {
                            p.kind = RefKind::Unused;
                        }
                    }
                    if let Some(i) = target {
                        self.pics[i].kind = RefKind::Long;
                        self.pics[i].long_term_frame_idx = long_term_frame_idx;
                    }
                }
                Mmco::MaxLongTermIdx { max_plus1 } => {
                    self.max_long_term_frame_idx = max_plus1.checked_sub(1);
                    for p in &mut self.pics {
                        if p.is_long()
                            && self
                                .max_long_term_frame_idx
                                .is_none_or(|m| p.long_term_frame_idx > m)
                        {
                            p.kind = RefKind::Unused;
                        }
                    }
                }
                Mmco::Reset => {
                    for p in &mut self.pics {
                        p.kind = RefKind::Unused;
                    }
                    self.max_long_term_frame_idx = None;
                }
                Mmco::CurrentToLong {
                    long_term_frame_idx,
                } => {
                    for p in &mut self.pics {
                        if p.is_long() && p.long_term_frame_idx == long_term_frame_idx {
                            p.kind = RefKind::Unused;
                        }
                    }
                    cur.kind = RefKind::Long;
                    cur.long_term_frame_idx = long_term_frame_idx;
                    cur_long = true;
                }
            }
        }
        self.drop_unused();
        cur_long
    }

    /// Builds `RefPicList0` / `RefPicList1` for a slice as indices into [`Self::pics`] (8.2.4.2, 8.2.4.3).
    pub fn build_lists(
        &self,
        sh: &SliceHeader,
        cur_poc: i32,
        max_frame_num: u32,
    ) -> Result<[Vec<usize>; 2]> {
        let mut lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        if sh.slice_type.is_intra() {
            return Ok(lists);
        }
        let short: Vec<usize> = (0..self.pics.len())
            .filter(|&i| self.pics[i].is_short())
            .collect();
        let mut long: Vec<usize> = (0..self.pics.len())
            .filter(|&i| self.pics[i].is_long())
            .collect();
        long.sort_by_key(|&i| self.pics[i].long_term_frame_idx);

        let mut init: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        if matches!(sh.slice_type, SliceType::P | SliceType::Sp) {
            let mut s = short.clone();
            s.sort_by_key(|&i| std::cmp::Reverse(self.pics[i].frame_num_wrap));
            init[0] = s;
            init[0].extend(&long);
        } else {
            let mut before: Vec<usize> = short
                .iter()
                .copied()
                .filter(|&i| self.pics[i].poc < cur_poc)
                .collect();
            let mut after: Vec<usize> = short
                .iter()
                .copied()
                .filter(|&i| self.pics[i].poc > cur_poc)
                .collect();
            before.sort_by_key(|&i| std::cmp::Reverse(self.pics[i].poc));
            after.sort_by_key(|&i| self.pics[i].poc);
            init[0] = before
                .iter()
                .chain(after.iter())
                .chain(long.iter())
                .copied()
                .collect();
            init[1] = after
                .iter()
                .chain(before.iter())
                .chain(long.iter())
                .copied()
                .collect();
            if init[1].len() > 1 && init[1] == init[0] {
                init[1].swap(0, 1);
            }
        }

        for list in 0..2 {
            let n = sh.num_ref_idx_active[list] as usize;
            if n == 0 {
                continue;
            }
            let mut l: Vec<Option<usize>> = init[list].iter().copied().map(Some).collect();
            l.resize(n.max(l.len()), None);
            l.truncate(n);
            self.modify_list(&mut l, &sh.ref_mods[list], sh.frame_num, max_frame_num)?;
            // Missing entries point at the first available reference so damaged streams still decode.
            let fallback = l
                .iter()
                .flatten()
                .copied()
                .next()
                .or_else(|| init[list].first().copied());
            let mut out = Vec::with_capacity(n);
            for e in l {
                match e.or(fallback) {
                    Some(i) => out.push(i),
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

    fn modify_list(
        &self,
        l: &mut Vec<Option<usize>>,
        mods: &[RefMod],
        cur_frame_num: u32,
        max_frame_num: u32,
    ) -> Result<()> {
        let n = l.len();
        let max_pic_num = max_frame_num as i32;
        let curr_pic_num = cur_frame_num as i32;
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
                    let found = self
                        .pics
                        .iter()
                        .position(|p| p.is_short() && p.frame_num_wrap == pic_num);
                    found.map(|i| (i, true))
                }
                RefMod::LongTerm(lt) => self
                    .pics
                    .iter()
                    .position(|p| p.is_long() && p.long_term_frame_idx == lt)
                    .map(|i| (i, false)),
            };
            let Some((pic, _)) = pic else {
                return Err(RagdollError::InvalidData(
                    "H.264 reference list modification names a missing picture".into(),
                ));
            };
            // Insert at ref_idx, then remove the later duplicate of the same picture.
            l.insert(ref_idx, Some(pic));
            let mut w = ref_idx + 1;
            for c in ref_idx + 1..l.len() {
                if l[c] != Some(pic) {
                    l[w] = l[c];
                    w += 1;
                }
            }
            l.truncate(w.max(n));
            l.truncate(n);
        }
        Ok(())
    }
}
