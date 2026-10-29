//! Deblocking filter (ITU-T H.264 8.7).

use super::picture::{BlockInfo, MbType, NO_PIC, PicState, RefIds, Structure};
use super::tables::{ALPHA, BETA, TC0};
use ragdoll_simd::{h264_chroma_deblock_edge, h264_luma_deblock_edge};

/// Per-slice deblocking controls.
#[derive(Clone, Copy, Debug)]
pub struct SliceDeblock {
    pub idc: u32,
    pub alpha_offset: i32,
    pub beta_offset: i32,
}

/// Filters every decoded macroblock of `st` in raster order.
pub fn deblock_picture(st: &mut PicState, slices: &[SliceDeblock], has_chroma: bool) {
    if st.mbaff {
        return deblock_mbaff(st, slices, has_chroma);
    }
    let (mb_w, mb_h) = (st.mb_w, st.mb_h);
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let addr = mb_y * mb_w + mb_x;
            if st.mb_type[addr] == MbType::None {
                continue;
            }
            let Some(sl) = slices.get(st.slice_num[addr] as usize).copied() else {
                continue;
            };
            if sl.idc == 1 {
                continue;
            }
            let neighbour_ok = |n: usize| {
                st.mb_type[n] != MbType::None
                    && !(sl.idc == 2 && st.slice_num[n] != st.slice_num[addr])
            };
            let left = mb_x > 0 && neighbour_ok(addr - 1);
            let top = mb_y > 0 && neighbour_ok(addr - mb_w);
            let t8 = st.t8x8[addr];
            let local = EdgeCtx::new(st, mb_x, mb_y);
            // An inter MB with one motion and no luma coefficients has bS 0 on every inner edge.
            let no_inner = !local.intra[0] && st.motion_edges[addr] == 0 && st.cbp[addr] & 0xF == 0;
            // Edges to visit, bit `4 * dir + edge`; odd edges only exist in luma, which has none
            // inside 8x8 transform blocks.
            let mut todo: u8 = if no_inner { 0x11 } else { 0xFF };
            todo &= !(u8::from(!left) | u8::from(!top) << 4);
            if t8 {
                todo &= !0xAA;
            }
            // Vertical edges, then horizontal ones, each with its direction known at compile time.
            let at = (mb_x, mb_y, addr);
            filter_mb_dir::<0>(st, &local, at, todo & 0xF, sl, has_chroma);
            filter_mb_dir::<1>(st, &local, at, todo >> 4, sl, has_chroma);
        }
    }
}

/// Filters the edges of one direction (`DIR` 0 vertical, 1 horizontal) of the macroblock at
/// `(mb_x, mb_y, addr)`: edge `e` when bit `e` of `todo` is set.
fn filter_mb_dir<const DIR: usize>(
    st: &mut PicState,
    local: &EdgeCtx,
    (mb_x, mb_y, addr): (usize, usize, usize),
    todo: u8,
    sl: SliceDeblock,
    has_chroma: bool,
) {
    for edge in 0..4 {
        if todo & (1 << edge) == 0 {
            continue;
        }
        let bs = local.strengths(st, DIR, edge);
        if bs == [0; 4] {
            continue;
        }
        let p_addr = match (edge, DIR) {
            (0, 0) => addr - 1,
            (0, _) => addr - st.mb_w,
            _ => addr,
        };
        let qp = (st.qp[p_addr] as i32 + st.qp[addr] as i32 + 1) >> 1;
        filter_luma_edge::<DIR>(st, mb_x, mb_y, edge, bs, qp, sl);
        if has_chroma && edge % 2 == 0 {
            for c in 0..2 {
                let qp = (st.qpc[p_addr][c] as i32 + st.qpc[addr][c] as i32 + 1) >> 1;
                filter_chroma_edge::<DIR>(st, c, mb_x, mb_y, edge / 2, bs, qp, sl);
            }
        }
    }
}

/// What bS derivation (8.7.2.1) needs to know about one macroblock: its position and whether
/// it and its left and top neighbours are intra.
struct EdgeCtx {
    /// 4x4 index of the MB's top-left block, and the 4x4 stride.
    b0: usize,
    s: usize,
    /// This MB, and the left and top MBs (the p side of edges 0).
    mb: [usize; 3],
    /// Intra: this MB, the left MB, the top MB.
    intra: [bool; 3],
    field: bool,
    /// [`PicState::motion_edges`] of this MB.
    motion_edges: u8,
}

impl EdgeCtx {
    fn new(st: &PicState, mb_x: usize, mb_y: usize) -> Self {
        let addr = mb_y * st.mb_w + mb_x;
        let s = st.mb_w * 4;
        Self {
            b0: mb_y * 4 * s + mb_x * 4,
            s,
            mb: [addr, addr.wrapping_sub(1), addr.wrapping_sub(st.mb_w)],
            intra: [
                st.mb_type[addr].is_intra(),
                mb_x > 0 && st.mb_type[addr - 1].is_intra(),
                mb_y > 0 && st.mb_type[addr - st.mb_w].is_intra(),
            ],
            field: matches!(st.structure, Structure::Field(_)),
            motion_edges: st.motion_edges[addr],
        }
    }

    /// Boundary strengths of the four 4-sample segments of `edge` (0..4) in direction `dir`
    /// (0 vertical, 1 horizontal).
    #[inline(always)]
    fn strengths(&self, st: &PicState, dir: usize, edge: usize) -> [u8; 4] {
        // Field macroblocks: intra MB edges between vertically adjacent MBs get bS 3, and
        // vertical vectors are compared in field units (4 quarter frame samples = 2 quarter
        // field samples).
        // Without short-circuits: one branch on the outcome rather than one per operand.
        let p_intra = (edge == 0) & self.intra[1 + dir];
        if self.intra[0] | p_intra {
            return [if edge == 0 && !(self.field && dir == 1) {
                4
            } else {
                3
            }; 4];
        }
        let ylim = if self.field { 2 } else { 4 };
        let (step, along) = if dir == 0 { (1, self.s) } else { (self.s, 1) };
        let q0 = self.b0 + edge * step;
        // Inside a partition of one motion only coefficients count.
        let check_motion = (edge == 0) | (self.motion_edges & (1 << (4 * dir + edge)) != 0);
        let (qm, pm) = (
            self.mb[0],
            if edge == 0 {
                self.mb[1 + dir]
            } else {
                self.mb[0]
            },
        );
        // Blocks under one reference table with equal indices refer to the same pictures.
        let tq = st.mb_table[qm] as usize;
        let one_table = pm == qm || st.mb_table[pm] as usize == tq;
        // Within a table without repeated pictures, (list, index) pairs stand in for pictures.
        let by_index = one_table && st.ref_unique.get(tq).copied().unwrap_or(false);
        let index_ids = |b: &BlockInfo| -> [u32; 2] {
            [0, 1].map(|l| match b.ref_idx[l] {
                r if r >= 0 => (l as u32) << 8 | r as u32,
                _ => NO_PIC,
            })
        };
        let table = |m: usize| st.ref_tables.get(st.mb_table[m] as usize);
        let pics = |t: Option<&RefIds>, b: &BlockInfo| -> [u32; 2] {
            [0, 1].map(|l| match (t, b.ref_idx[l]) {
                (Some(t), r) if r >= 0 => t[l][r as usize & 31],
                _ => NO_PIC,
            })
        };
        let mut bs = [0u8; 4];
        for (k, slot) in bs.iter_mut().enumerate() {
            let qi = q0 + k * along;
            let pi = qi - step;
            let (bp, bq) = (&st.blocks[pi], &st.blocks[qi]);
            let nz = bp.nz | bq.nz;
            *slot = 2 * nz as u8;
            if check_motion && !nz {
                // Evaluated without short-circuits: one branch on the outcome.
                let (mp, mq) = (bp.mv, bq.mv);
                let same =
                    one_table & (bp.ref_idx == bq.ref_idx) & (mp[0] == mq[0]) & (mp[1] == mq[1]);
                if !same {
                    let (rp, rq) = if by_index {
                        (index_ids(bp), index_ids(bq))
                    } else {
                        (pics(table(pm), bp), pics(table(qm), bq))
                    };
                    *slot = motion_differs(rp, mp, rq, mq, ylim) as u8;
                }
            }
        }
        bs
    }
}

fn mv_far(a: [i16; 2], b: [i16; 2], ylim: i32) -> bool {
    (a[0] as i32 - b[0] as i32).abs() >= 4 || (a[1] as i32 - b[1] as i32).abs() >= ylim
}

/// Whether two inter blocks predict from different pictures or with vectors 4 or more quarter samples apart.
#[inline]
fn motion_differs(
    rp: [u32; 2],
    mp: [[i16; 2]; 2],
    rq: [u32; 2],
    mq: [[i16; 2]; 2],
    ylim: i32,
) -> bool {
    // Identical motion on both sides (the common case inside a partition) never differs.
    if rp[0] == rq[0] && rp[1] == rq[1] && mp[0] == mq[0] && mp[1] == mq[1] {
        return false;
    }
    let np = rp.iter().filter(|&&r| r != NO_PIC).count();
    let nq = rq.iter().filter(|&&r| r != NO_PIC).count();
    if np != nq {
        return true;
    }
    if np == 1 {
        let (lp, lq) = (
            if rp[0] != NO_PIC { 0 } else { 1 },
            if rq[0] != NO_PIC { 0 } else { 1 },
        );
        return rp[lp] != rq[lq] || mv_far(mp[lp], mq[lq], ylim);
    }
    let same_set = (rp[0] == rq[0] && rp[1] == rq[1]) || (rp[0] == rq[1] && rp[1] == rq[0]);
    if !same_set {
        return true;
    }
    if rp[0] != rp[1] {
        if rp[0] == rq[0] {
            mv_far(mp[0], mq[0], ylim) || mv_far(mp[1], mq[1], ylim)
        } else {
            mv_far(mp[0], mq[1], ylim) || mv_far(mp[1], mq[0], ylim)
        }
    } else {
        (mv_far(mp[0], mq[0], ylim) || mv_far(mp[1], mq[1], ylim))
            && (mv_far(mp[0], mq[1], ylim) || mv_far(mp[1], mq[0], ylim))
    }
}

/// [`motion_differs`] between 4x4 blocks `p` (of macroblock `pm`) and `q` (of `qm`) of `st`.
fn blocks_differ(st: &PicState, pm: usize, p: usize, qm: usize, q: usize, ylim: i32) -> bool {
    motion_differs(
        st.ref_pics(pm, p),
        st.blocks[p].mv,
        st.ref_pics(qm, q),
        st.blocks[q].mv,
        ylim,
    )
}

#[inline]
fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Filters one line of samples across an edge. `s[i]` for `i` in `-4..4` maps to `p3..p0, q0..q3`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn filter_line(
    buf: &mut [u8],
    q0: usize,
    step: usize,
    bs: u8,
    alpha: i32,
    beta: i32,
    tc0: i32,
    chroma: bool,
) {
    let at = |i: isize| (q0 as isize + i * step as isize) as usize;
    let (p0, q0v) = (buf[at(-1)] as i32, buf[at(0)] as i32);
    let (p1, q1) = (buf[at(-2)] as i32, buf[at(1)] as i32);
    if (p0 - q0v).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
        return;
    }
    if chroma {
        if bs < 4 {
            let tc = tc0 + 1;
            let d = ((((q0v - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            buf[at(-1)] = clip1(p0 + d);
            buf[at(0)] = clip1(q0v - d);
        } else {
            buf[at(-1)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            buf[at(0)] = ((2 * q1 + q0v + p1 + 2) >> 2) as u8;
        }
        return;
    }
    let (p2, q2) = (buf[at(-3)] as i32, buf[at(2)] as i32);
    let ap = (p2 - p0).abs();
    let aq = (q2 - q0v).abs();
    if bs < 4 {
        let tc = tc0 + (ap < beta) as i32 + (aq < beta) as i32;
        let d = ((((q0v - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        buf[at(-1)] = clip1(p0 + d);
        buf[at(0)] = clip1(q0v - d);
        if ap < beta {
            buf[at(-2)] =
                (p1 + ((p2 + ((p0 + q0v + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0)) as u8;
        }
        if aq < beta {
            buf[at(1)] =
                (q1 + ((q2 + ((p0 + q0v + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0)) as u8;
        }
    } else {
        let (p3, q3) = (buf[at(-4)] as i32, buf[at(3)] as i32);
        let strong = (p0 - q0v).abs() < ((alpha >> 2) + 2);
        if ap < beta && strong {
            buf[at(-1)] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0v + q1 + 4) >> 3) as u8;
            buf[at(-2)] = ((p2 + p1 + p0 + q0v + 2) >> 2) as u8;
            buf[at(-3)] = ((2 * p3 + 3 * p2 + p1 + p0 + q0v + 4) >> 3) as u8;
        } else {
            buf[at(-1)] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
        }
        if aq < beta && strong {
            buf[at(0)] = ((p1 + 2 * p0 + 2 * q0v + 2 * q1 + q2 + 4) >> 3) as u8;
            buf[at(1)] = ((p0 + q0v + q1 + q2 + 2) >> 2) as u8;
            buf[at(2)] = ((2 * q3 + 3 * q2 + q1 + q0v + p0 + 4) >> 3) as u8;
        } else {
            buf[at(0)] = ((2 * q1 + q0v + p1 + 2) >> 2) as u8;
        }
    }
}

fn thresholds(qp: i32, sl: SliceDeblock) -> (usize, i32, i32) {
    let ia = (qp + sl.alpha_offset).clamp(0, 51) as usize;
    let ib = (qp + sl.beta_offset).clamp(0, 51) as usize;
    (ia, ALPHA[ia] as i32, BETA[ib] as i32)
}

#[allow(clippy::too_many_arguments)]
fn filter_luma_edge<const DIR: usize>(
    st: &mut PicState,
    mb_x: usize,
    mb_y: usize,
    edge: usize,
    bs: [u8; 4],
    qp: i32,
    sl: SliceDeblock,
) {
    let (ia, alpha, beta) = thresholds(qp, sl);
    if alpha == 0 || beta == 0 {
        return;
    }
    let w = st.width;
    let (q0, xstride, ystride) = if DIR == 0 {
        (mb_y * 16 * w + mb_x * 16 + edge * 4, 1, w)
    } else {
        ((mb_y * 16 + edge * 4) * w + mb_x * 16, w, 1)
    };
    // bS 4 belongs to whole macroblock edges, so an edge is either all bS 4 or all below.
    if bs == [4; 4] {
        let plane = &mut st.planes[0];
        if DIR == 0 {
            h264_luma_deblock_edge::<false, true>(plane, q0, w, alpha, beta, [0; 4]);
        } else {
            h264_luma_deblock_edge::<true, true>(plane, q0, w, alpha, beta, [0; 4]);
        }
        return;
    }
    if !bs.contains(&4) {
        let tc0 = bs.map(|b| match b {
            0 => -1,
            _ => TC0[ia][b as usize - 1] as i8,
        });
        let plane = &mut st.planes[0];
        if DIR == 0 {
            h264_luma_deblock_edge::<false, false>(plane, q0, w, alpha, beta, tc0);
        } else {
            h264_luma_deblock_edge::<true, false>(plane, q0, w, alpha, beta, tc0);
        }
        return;
    }
    for (k, &b) in bs.iter().enumerate() {
        for line in 4 * k..4 * k + 4 {
            if b == 0 {
                continue;
            }
            let tc0 = if b < 4 {
                TC0[ia][b as usize - 1] as i32
            } else {
                0
            };
            let q = q0 + line * ystride;
            filter_line(&mut st.planes[0], q, xstride, b, alpha, beta, tc0, false);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_chroma_edge<const DIR: usize>(
    st: &mut PicState,
    c: usize,
    mb_x: usize,
    mb_y: usize,
    cedge: usize,
    bs: [u8; 4],
    qp: i32,
    sl: SliceDeblock,
) {
    let (ia, alpha, beta) = thresholds(qp, sl);
    if alpha == 0 || beta == 0 {
        return;
    }
    let w = st.width / 2;
    let (q0, xstride, ystride) = if DIR == 0 {
        (mb_y * 8 * w + mb_x * 8 + cedge * 4, 1, w)
    } else {
        ((mb_y * 8 + cedge * 4) * w + mb_x * 8, w, 1)
    };
    let plane = &mut st.planes[1 + c];
    if bs == [4; 4] {
        if DIR == 0 {
            h264_chroma_deblock_edge::<false, true>(plane, q0, w, alpha, beta, [0; 4]);
        } else {
            h264_chroma_deblock_edge::<true, true>(plane, q0, w, alpha, beta, [0; 4]);
        }
        return;
    }
    if !bs.contains(&4) {
        let tc0 = bs.map(|b| match b {
            0 => -1,
            _ => TC0[ia][b as usize - 1] as i8,
        });
        if DIR == 0 {
            h264_chroma_deblock_edge::<false, false>(plane, q0, w, alpha, beta, tc0);
        } else {
            h264_chroma_deblock_edge::<true, false>(plane, q0, w, alpha, beta, tc0);
        }
        return;
    }
    for k in 0..8 {
        let b = bs[k / 2];
        if b == 0 {
            continue;
        }
        let tc0 = if b < 4 {
            TC0[ia][b as usize - 1] as i32
        } else {
            0
        };
        let q = q0 + k * ystride;
        filter_line(&mut st.planes[1 + c], q, xstride, b, alpha, beta, tc0, true);
    }
}

/// Filters one line across an edge with boundary strength `bs` and average QP `qp`.
#[allow(clippy::too_many_arguments)]
fn filter_at(
    buf: &mut [u8],
    q0: usize,
    step: usize,
    bs: u8,
    qp: i32,
    sl: SliceDeblock,
    chroma: bool,
) {
    if bs == 0 {
        return;
    }
    let (ia, alpha, beta) = thresholds(qp, sl);
    if alpha == 0 || beta == 0 {
        return;
    }
    let tc0 = if bs < 4 {
        TC0[ia][bs as usize - 1] as i32
    } else {
        0
    };
    filter_line(buf, q0, step, bs, alpha, beta, tc0, chroma);
}

/// The macroblock covering frame luma line `line` in MB column `mbx` of an MBAFF frame, and the
/// row inside it.
fn mbaff_line_mb(st: &PicState, mbx: usize, line: usize) -> (usize, usize) {
    let top = 2 * (line / 32) * st.mb_w + mbx;
    let l = line % 32;
    if st.mb_field[top] {
        (top + (l % 2) * st.mb_w, l / 2)
    } else {
        (top + (l / 16) * st.mb_w, l % 16)
    }
}

/// Boundary strength between the 4x4 blocks `pi` (in MB `pm`) and `qi` (in MB `qm`) of an MBAFF
/// frame (8.7.2.1).
fn mbaff_bs(
    st: &PicState,
    pm: usize,
    pi: usize,
    qm: usize,
    qi: usize,
    mb_edge: bool,
    vertical: bool,
) -> u8 {
    let (pf, qf) = (st.mb_field[pm], st.mb_field[qm]);
    if st.mb_type[pm].is_intra() || st.mb_type[qm].is_intra() {
        return if mb_edge && (vertical || (!pf && !qf)) {
            4
        } else {
            3
        };
    }
    if st.blocks[pi].nz || st.blocks[qi].nz {
        return 2;
    }
    // Mixed edge: a field pair next to a frame pair.
    if pf != qf {
        return 1;
    }
    blocks_differ(st, pm, pi, qm, qi, if qf { 2 } else { 4 }) as u8
}

/// Deblocking of an MBAFF frame: macroblocks in decoding order (pairs, top MB first), with sample
/// positions as frame lines. A field MB's rows are the lines of its field within the pair.
fn deblock_mbaff(st: &mut PicState, slices: &[SliceDeblock], has_chroma: bool) {
    let (mb_w, mb_h) = (st.mb_w, st.mb_h);
    let (w, cw) = (st.width, st.width / 2);
    for pair in 0..mb_w * mb_h / 2 {
        for bot in 0..2 {
            let (mx, my) = (pair % mb_w, 2 * (pair / mb_w) + bot);
            let m = my * mb_w + mx;
            if st.mb_type[m] == MbType::None {
                continue;
            }
            let Some(sl) = slices.get(st.slice_num[m] as usize).copied() else {
                continue;
            };
            if sl.idc == 1 {
                continue;
            }
            let ok = |st: &PicState, n: usize| {
                st.mb_type[n] != MbType::None
                    && !(sl.idc == 2 && st.slice_num[n] != st.slice_num[m])
            };
            let f = st.mb_field[m];
            let (y0, cy0, ls) = if f {
                ((my & !1) * 16 + (my & 1), (my & !1) * 8 + (my & 1), 2)
            } else {
                (my * 16, my * 8, 1)
            };
            let qblk = |st: &PicState, x4: usize, r: usize| st.b4(mx * 4 + x4, my * 4 + r / 4);
            let pblk = |st: &PicState, pm: usize, x4: usize, row: usize| {
                st.b4((pm % mb_w) * 4 + x4, (pm / mb_w) * 4 + row / 4)
            };
            let qpc = |st: &PicState, pm: usize, c: usize| {
                (st.qpc[pm][c] as i32 + st.qpc[m][c] as i32 + 1) >> 1
            };
            let qpy = |st: &PicState, pm: usize| (st.qp[pm] as i32 + st.qp[m] as i32 + 1) >> 1;

            // Left MB edge: per line, the p samples lie in the left pair at the same frame line.
            if mx > 0 {
                let mut bs = [0u8; 16];
                let mut pms = [usize::MAX; 16];
                for r in 0..16 {
                    let line = y0 + r * ls;
                    let (pm, prow) = mbaff_line_mb(st, mx - 1, line);
                    if !ok(st, pm) {
                        continue;
                    }
                    pms[r] = pm;
                    bs[r] = mbaff_bs(st, pm, pblk(st, pm, 3, prow), m, qblk(st, 0, r), true, true);
                    let qp = qpy(st, pm);
                    filter_at(
                        &mut st.planes[0],
                        line * w + mx * 16,
                        1,
                        bs[r],
                        qp,
                        sl,
                        false,
                    );
                }
                if has_chroma {
                    let mixed = pms[0] != usize::MAX && st.mb_field[pms[0]] != f;
                    for k in 0..8 {
                        let lr = if mixed && f {
                            2 * k
                        } else {
                            ((k >> 1) << 2) + (k & 1)
                        };
                        if pms[lr] == usize::MAX {
                            continue;
                        }
                        let line = cy0 + k * ls;
                        for c in 0..2 {
                            let qp = qpc(st, pms[lr], c);
                            filter_at(
                                &mut st.planes[1 + c],
                                line * cw + mx * 8,
                                1,
                                bs[lr],
                                qp,
                                sl,
                                true,
                            );
                        }
                    }
                }
            }
            // Internal vertical edges.
            for e in 1..4 {
                if st.t8x8[m] && e % 2 == 1 {
                    continue;
                }
                let mut bs = [0u8; 4];
                for (seg, b) in bs.iter_mut().enumerate() {
                    let qi = qblk(st, e, seg * 4);
                    *b = mbaff_bs(st, m, qi - 1, m, qi, false, true);
                }
                let qp = st.qp[m] as i32;
                for r in 0..16 {
                    let q0 = (y0 + r * ls) * w + mx * 16 + 4 * e;
                    filter_at(&mut st.planes[0], q0, 1, bs[r / 4], qp, sl, false);
                }
                if has_chroma && e == 2 {
                    for k in 0..8 {
                        let q0 = (cy0 + k * ls) * cw + mx * 8 + 4;
                        for c in 0..2 {
                            let qp = st.qpc[m][c] as i32;
                            filter_at(&mut st.planes[1 + c], q0, 1, bs[k / 2], qp, sl, true);
                        }
                    }
                }
            }

            // Top MB edge: `(p MB, p row, q first line offset, line step)` per filtering pass.
            let mut passes: Vec<(usize, usize, usize)> = Vec::new();
            if f {
                if my >= 2 {
                    let (pm, prow) = mbaff_line_mb(st, mx, y0 - 2);
                    passes.push((pm, prow, 0));
                }
            } else if my % 2 == 1 {
                passes.push((m - mb_w, 15, 0));
            } else if my >= 2 {
                if st.mb_field[(my - 2) * mb_w + mx] {
                    // A frame MB below a field pair is filtered once per field (8.7.1).
                    passes.push(((my - 2) * mb_w + mx, 15, 0));
                    passes.push(((my - 1) * mb_w + mx, 15, 1));
                } else {
                    passes.push(((my - 1) * mb_w + mx, 15, 0));
                }
            }
            let double = passes.len() == 2;
            for &(pm, prow, j) in &passes {
                if !ok(st, pm) {
                    continue;
                }
                let mut bs = [0u8; 4];
                for (k, b) in bs.iter_mut().enumerate() {
                    *b = mbaff_bs(
                        st,
                        pm,
                        pblk(st, pm, k, prow),
                        m,
                        qblk(st, k, 0),
                        true,
                        false,
                    );
                    if double {
                        *b = (*b).min(3);
                    }
                }
                // Field MBs and the per-field passes step over the other field's lines.
                let step = if f || double { 2 } else { 1 };
                let qp = qpy(st, pm);
                for x in 0..16 {
                    let q0 = (y0 + j) * w + mx * 16 + x;
                    filter_at(&mut st.planes[0], q0, step * w, bs[x / 4], qp, sl, false);
                }
                if has_chroma {
                    for c in 0..2 {
                        let qp = qpc(st, pm, c);
                        for x in 0..8 {
                            let q0 = (cy0 + j) * cw + mx * 8 + x;
                            filter_at(
                                &mut st.planes[1 + c],
                                q0,
                                step * cw,
                                bs[x / 2],
                                qp,
                                sl,
                                true,
                            );
                        }
                    }
                }
            }
            // Internal horizontal edges.
            for e in 1..4 {
                if st.t8x8[m] && e % 2 == 1 {
                    continue;
                }
                let mut bs = [0u8; 4];
                for (k, b) in bs.iter_mut().enumerate() {
                    let qi = qblk(st, k, e * 4);
                    *b = mbaff_bs(st, m, qi - mb_w * 4, m, qi, false, false);
                }
                let qp = st.qp[m] as i32;
                for x in 0..16 {
                    let q0 = (y0 + 4 * e * ls) * w + mx * 16 + x;
                    filter_at(&mut st.planes[0], q0, ls * w, bs[x / 4], qp, sl, false);
                }
                if has_chroma && e == 2 {
                    for c in 0..2 {
                        let qp = st.qpc[m][c] as i32;
                        for x in 0..8 {
                            let q0 = (cy0 + 4 * ls) * cw + mx * 8 + x;
                            filter_at(&mut st.planes[1 + c], q0, ls * cw, bs[x / 2], qp, sl, true);
                        }
                    }
                }
            }
        }
    }
}
