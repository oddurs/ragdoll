//! Inter macroblocks of the [`H264Speed::Compact`] preset (`SIMD-0089`): partitions down to
//! $8\times8$, their motion vector prediction, motion compensation and syntax.

use super::compact::{ME_RANGE, PartSearch, QPEL_REFINE, RefLuma, sa8d_16x16};
use super::*;
use ragdoll_simd::{h264_avg, h264_chroma_mc, h264_chroma_mc_avg};

/// P macroblocks whose 16x16 cost is at most this many lambdas skip the partitions, which
/// rarely pay below it on the clips measured for `SIMD-0089`.
const P8X8_GATE: u32 = 40;

/// Motion of one $8\times8$ quadrant of a macroblock in both lists: reference index (`-1` when
/// the list is unused) and vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Quad {
    pub(super) r: [i8; 2],
    pub(super) mv: [[i16; 2]; 2],
}

impl Quad {
    /// List 0 only.
    pub(super) const fn l0(mv: [i16; 2]) -> Self {
        Self {
            r: [0, -1],
            mv: [mv, [0, 0]],
        }
    }
}

/// `n`, opaque to the optimizer so that a loop up to it stays rolled.
#[inline(always)]
pub(super) fn rolled(n: usize) -> usize {
    std::hint::black_box(n)
}

/// Partitioning of an inter macroblock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    S16x16,
    S16x8,
    S8x16,
    S8x8,
}

impl Shape {
    /// Partitions as `(bx, by, w, h)` in $4\times4$ blocks, in syntax order.
    #[inline(always)]
    pub(super) fn parts(self) -> &'static [(usize, usize, usize, usize)] {
        match self {
            Shape::S16x16 => &[(0, 0, 4, 4)],
            Shape::S16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
            Shape::S8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
            Shape::S8x8 => &[(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
        }
    }

    /// The directional predictor rule (8.4.1.3) of partition `p`: `1` top, `2` left (of
    /// $16\times8$ partitions), `3` left, `4` top-right (of $8\times16$ ones), else `0`.
    #[inline(always)]
    pub(super) fn rule(self, p: usize) -> u8 {
        match self {
            Shape::S16x8 => 1 + p as u8,
            Shape::S8x16 => 3 + p as u8,
            _ => 0,
        }
    }

    /// The shape whose partitions cover the quadrants with equal motion.
    #[inline(always)]
    pub(super) fn of(q: &[Quad; 4]) -> Self {
        if q[0] == q[1] && q[2] == q[3] {
            if q[0] == q[2] {
                Shape::S16x16
            } else {
                Shape::S16x8
            }
        } else if q[0] == q[2] && q[1] == q[3] {
            Shape::S8x16
        } else {
            Shape::S8x8
        }
    }
}

/// Quadrant of a partition's top-left $4\times4$ block.
#[inline(always)]
pub(super) fn quad_of(bx: usize, by: usize) -> usize {
    (bx >> 1) + (by & 2)
}

/// An analysed inter macroblock: partitioning and the motion of its quadrants.
#[derive(Clone, Copy, Debug)]
pub(super) struct InterMb {
    pub(super) shape: Shape,
    pub(super) q: [Quad; 4],
}

/// Neighbour of motion vector prediction: reference index (`-2` unavailable) and vector.
pub(super) type Nb = (i8, [i16; 2]);

/// The neighbours A, B and C (or D) of the partition whose top-left $4\times4$ block is
/// `(gx, gy)` and which is `w4` blocks wide (8.4.1.3.2), as [`pred_motion_partition`] reads them.
#[inline(always)]
pub(super) fn neighbours(
    rg: &[i8],
    mg: &[[i16; 2]],
    g_stride: usize,
    gx: usize,
    gy: usize,
    w4: usize,
) -> [Nb; 3] {
    let at = |i: usize| (rg[i], mg[i]);
    let a = if gx > 0 {
        at(gy * g_stride + gx - 1)
    } else {
        (-2, [0, 0])
    };
    let b = if gy > 0 {
        at((gy - 1) * g_stride + gx)
    } else {
        (-2, [0, 0])
    };
    let (bx, by) = (gx & 3, gy & 3);
    let tr_gx = gx + w4;
    let tr_avail = gy > 0
        && tr_gx < g_stride
        && !((by == 1 && (bx + w4 == 2 || bx + w4 == 4))
            || (by == 2 && bx + w4 == 4)
            || (by == 3 && (bx + w4 == 2 || bx + w4 == 4)));
    let c = if tr_avail && rg[(gy - 1) * g_stride + tr_gx] != -2 {
        at((gy - 1) * g_stride + tr_gx)
    } else if gy > 0 && gx > 0 {
        at((gy - 1) * g_stride + gx - 1)
    } else {
        (-2, [0, 0])
    };
    [a, b, c]
}

/// Motion vector predictor from neighbours `nb` for reference `refi` under directional `rule`
/// ([`Shape::rule`]).
#[inline(always)]
pub(super) fn mvp_from(nb: [Nb; 3], refi: i8, rule: u8) -> [i16; 2] {
    let [a, b, c] = nb;
    match rule {
        1 if b.0 == refi => return b.1,
        2 | 3 if a.0 == refi => return a.1,
        4 if c.0 == refi => return c.1,
        _ => {}
    }
    let matches = (a.0 == refi) as u8 + (b.0 == refi) as u8 + (c.0 == refi) as u8;
    if matches == 1 {
        if a.0 == refi {
            a.1
        } else if b.0 == refi {
            b.1
        } else {
            c.1
        }
    } else if matches == 0 && b.0 == -2 && c.0 == -2 && a.0 != -2 {
        a.1
    } else {
        [
            median3(a.1[0], b.1[0], c.1[0]),
            median3(a.1[1], b.1[1], c.1[1]),
        ]
    }
}

impl EncState {
    /// The reference index and vector grids of list `l`.
    #[inline(always)]
    pub(super) fn grids(&self, l: usize) -> (&[i8], &[[i16; 2]]) {
        if l == 0 {
            (&self.ref_grid, &self.mv_grid)
        } else {
            (&self.ref1_grid, &self.mv1_grid)
        }
    }

    /// Motion vector predictor of list `l` for partition `p` of `shape` in macroblock
    /// `(mb_x, mb_y)`, from the grids as they stand.
    #[inline(never)]
    pub(super) fn mvp_part(
        &self,
        l: usize,
        mb_x: usize,
        mb_y: usize,
        shape: Shape,
        p: usize,
        refi: i8,
    ) -> [i16; 2] {
        let (bx, by, w4, _) = shape.parts()[p];
        let (rg, mg) = self.grids(l);
        let nb = neighbours(rg, mg, self.mb_w * 4, mb_x * 4 + bx, mb_y * 4 + by, w4);
        mvp_from(nb, refi, shape.rule(p))
    }

    /// Stores the motion of list `l` of a `w4`x`h4` block at `(bx, by)` of macroblock
    /// `(mb_x, mb_y)` in the grids.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn set_motion(
        &mut self,
        l: usize,
        mb_x: usize,
        mb_y: usize,
        bx: usize,
        by: usize,
        w4: usize,
        h4: usize,
        r: i8,
        mv: [i16; 2],
    ) {
        let g_stride = self.mb_w * 4;
        let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
        let (rg, mg) = if l == 0 {
            (&mut self.ref_grid, &mut self.mv_grid)
        } else {
            (&mut self.ref1_grid, &mut self.mv1_grid)
        };
        fill_grid_4x4(rg, g_stride, gx, gy, w4, h4, r);
        fill_grid_4x4(mg, g_stride, gx, gy, w4, h4, mv);
    }

    /// Motion compensation of `mb` into `py`, `pu` and `pv` from the references of both lists.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(super) fn predict_inter(
        refs: [(RefLuma, &Pic); 2],
        mb_x: usize,
        mb_y: usize,
        q: &[Quad; 4],
        py: &mut [u8; 256],
        pu: &mut [u8; 64],
        pv: &mut [u8; 64],
    ) {
        let shape = Shape::of(q);
        for &(bx, by, w4, h4) in shape.parts() {
            let m = q[quad_of(bx, by)];
            let (w, h) = (w4 * 4, h4 * 4);
            let (x, y) = (mb_x * 16 + bx * 4, mb_y * 16 + by * 4);
            let (cx, cy) = (mb_x * 8 + bx * 2, mb_y * 8 + by * 2);
            let lo = by * 64 + bx * 4;
            let co = by * 16 + bx * 2;
            let mut first = true;
            for (l, &(rl, pic)) in refs.iter().enumerate() {
                if m.r[l] < 0 {
                    continue;
                }
                let mv = m.mv[l];
                let mvi = [mv[0] as i32, mv[1] as i32];
                if first {
                    rl.pred(x, y, w, h, mvi, &mut py[lo..], 16);
                } else {
                    let mut t = [0u8; 256];
                    rl.pred(x, y, w, h, mvi, &mut t, 16);
                    h264_avg(&mut py[lo..], 16, &t, 16, w, h);
                }
                let off = pic.u.at(
                    cx as isize + (mv[0] >> 3) as isize,
                    cy as isize + (mv[1] >> 3) as isize,
                );
                let (fx, fy) = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
                let ss = pic.u.stride;
                for (dst, p) in [(&mut pu[..], &pic.u), (&mut pv[..], &pic.v)] {
                    if first {
                        h264_chroma_mc(&mut dst[co..], 8, &p.buf, off, ss, w / 2, h / 2, fx, fy);
                    } else {
                        h264_chroma_mc_avg(
                            &mut dst[co..],
                            8,
                            &p.buf,
                            off,
                            ss,
                            w / 2,
                            h / 2,
                            fx,
                            fy,
                        );
                    }
                }
                first = false;
            }
        }
    }

    /// Writes and reconstructs inter macroblock `mb` of a P slice with residual `res` over the
    /// prediction in `py`, `pu` and `pv`, and stores its motion.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn code_inter_p_body(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        mb: &InterMb,
        res: &InterResidual,
        py: &mut [u8; 256],
        pu: &mut [u8; 64],
        pv: &mut [u8; 64],
    ) {
        let mb_w = self.mb_w;
        let mb_idx = mb_y * mb_w + mb_x;
        let (g_stride, cg_stride) = (mb_w * 4, mb_w * 2);
        let cbp_luma = res.cbp_luma;
        let cbp_chroma = res.cbp_chroma;
        let cbp = cbp_luma | (cbp_chroma << 4);
        let qp_delta = if cbp != 0 { qp as i32 - sl.cur_qp } else { 0 };
        let eff_qp = (sl.cur_qp + qp_delta).clamp(0, 51) as u8;
        let eff_qpc = chroma_qp(eff_qp as i32, sl.chroma_qp_offset);
        if res.luma_q8.is_some() && cbp_luma != 0 {
            self.mb_8x8_flag[mb_idx] = true;
        }
        let mb_type = match mb.shape {
            Shape::S16x16 => 0,
            Shape::S16x8 => 1,
            Shape::S8x16 => 2,
            Shape::S8x8 => 3,
        };
        match cabac {
            Some((ce, ctxs)) => {
                ce.encode_p_mb_type(ctxs, mb_type);
                if mb.shape == Shape::S8x8 {
                    for _ in 0..4 {
                        ce.encode_decision(&mut ctxs[21], 1); // P_L0_8x8
                    }
                }
            }
            None => {
                w.write_ue(mb_type as u64);
                if mb.shape == Shape::S8x8 {
                    for _ in 0..4 {
                        w.write_ue(0);
                    }
                }
            }
        }
        for (p, &(bx, by, w4, h4)) in mb.shape.parts().iter().enumerate() {
            let mv = mb.q[quad_of(bx, by)].mv[0];
            let mvp = self.mvp_part(0, mb_x, mb_y, mb.shape, p, 0);
            let mvd = [mv[0] - mvp[0], mv[1] - mvp[1]];
            let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
            match cabac {
                Some((ce, ctxs)) => {
                    let (ax, ay) = neighbor_mvd_sum_16x16(&self.mvd_grid, g_stride, gx, gy);
                    ce.encode_mvd(ctxs, 0, ax, mvd[0]);
                    ce.encode_mvd(ctxs, 1, ay, mvd[1]);
                    fill_grid_4x4(&mut self.mvd_grid, g_stride, gx, gy, w4, h4, mvd);
                }
                None => {
                    w.write_se(mvd[0] as i64);
                    w.write_se(mvd[1] as i64);
                }
            }
            self.set_motion(0, mb_x, mb_y, bx, by, w4, h4, 0, mv);
        }
        match cabac {
            Some((ce, ctxs)) => {
                encode_cabac_cbp_and_residuals(
                    ce,
                    ctxs,
                    mb_x,
                    mb_y,
                    mb_w,
                    g_stride,
                    cg_stride,
                    false,
                    sl.t8x8,
                    cbp_luma,
                    cbp_chroma,
                    qp_delta,
                    &mut sl.prev_qp_delta_nz,
                    &res.luma_q4,
                    res.luma_q8,
                    &res.chroma.dc,
                    &res.chroma.ac,
                    &mut self.luma_nnz,
                    &mut self.chroma_nnz,
                    &self.mb_is_intra,
                    &self.mb_is_i16x16,
                    &self.mb_is_skip,
                    &self.mb_8x8_flag,
                    &mut self.mb_cbp_luma,
                    &mut self.mb_cbp_chroma,
                    &mut self.mb_chroma_dc_cbf,
                );
            }
            None => {
                w.write_ue(inter_cbp_to_golomb(cbp));
                if cbp != 0 {
                    if sl.t8x8 {
                        w.write_bit(false);
                    }
                    w.write_se(qp_delta as i64);
                    write_cavlc_residuals(
                        w,
                        mb_x,
                        mb_y,
                        mb_w,
                        cbp_luma,
                        cbp_chroma,
                        &res.luma_q4,
                        &res.chroma,
                        &mut self.luma_nnz,
                        &mut self.chroma_nnz,
                    );
                }
                self.mb_cbp_luma[mb_idx] = cbp_luma;
                self.mb_cbp_chroma[mb_idx] = cbp_chroma;
            }
        }
        sl.cur_qp = eff_qp as i32;
        self.mb_qps[mb_idx] = eff_qp;
        if let Some(luma_q8) = res.luma_q8 {
            for (i8x8, q8) in luma_q8.iter().enumerate() {
                if cbp_luma & (1 << i8x8) != 0 {
                    let o = (i8x8 >> 1) * 128 + (i8x8 & 1) * 8;
                    recon_8x8(&mut py[o..], q8, eff_qp);
                }
            }
        } else {
            recon_luma_4x4(py, &res.luma_q4, cbp_luma, eff_qp);
        }
        recon_chroma(pu, pv, &res.chroma, cbp_chroma, eff_qpc);
        self.cur.store_mb(mb_x, mb_y, py, pu, pv);
    }

    out_of_line_v3!(pub(super) fn code_p_mb_compact / code_p_mb_v3 => code_p_mb_body(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
    ));

    out_of_line_v3!(fn code_i16_p / code_i16_p_v3 => code_i16(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
        mode: u8,
    ));

    out_of_line_v3!(fn code_i4_p / code_i4_p_v3 => code_i4(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
        i4: I4Choice,
    ));

    out_of_line_v3!(fn analyse_i4_p / analyse_i4_p_v3 => analyse_i4(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
        budget: u32,
    ) -> Option<I4Choice>);

    out_of_line_v3!(pub(super) fn code_inter_p / code_inter_p_v3 => code_inter_p_body(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        mb: &InterMb,
        res: &InterResidual,
        py: &mut [u8; 256],
        pu: &mut [u8; 64],
        pv: &mut [u8; 64],
    ));

    /// Analyses and codes one macroblock of a P slice with the Compact preset: `P_Skip` probe,
    /// $16\times16$ motion search, then $8\times8$ partitions when they promise to pay
    /// (`x264_mb_analyse_inter_p8x8`), intra $16\times16$ and $4\times4$, and the transform
    /// size by SA8D against SATD (`x264_mb_analyse_transform`).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn code_p_mb_body(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
    ) {
        let mb_w = self.mb_w;
        let g_stride = mb_w * 4;
        let (gx0, gy0) = (mb_x * 4, mb_y * 4);
        let (x0, y0) = (mb_x * 16, mb_y * 16);
        let mb_idx = mb_y * mb_w + mb_x;
        let qpc = chroma_qp(qp as i32, sl.chroma_qp_offset);
        let bounds = MvBounds::new(x0, y0, self.refp.y.w, self.refp.y.h);
        let lambda = LAMBDA_COMPACT[qp as usize] as u32;

        let mvp = self.mvp_part(0, mb_x, mb_y, Shape::S16x16, 0, 0);
        let pskip_mv = {
            let (r, m) = (&self.ref_grid, &self.mv_grid);
            let zero = |i: usize| r[i] == 0 && m[i] == [0, 0];
            if gx0 == 0
                || gy0 == 0
                || zero(gy0 * g_stride + gx0 - 1)
                || zero((gy0 - 1) * g_stride + gx0)
            {
                [0, 0]
            } else {
                mvp
            }
        };
        let mut py = [0u8; 256];
        let mut pu = [0u8; 64];
        let mut pv = [0u8; 64];
        if sl.skip_bounds.qp != qp {
            sl.skip_bounds = SkipBounds::new(qp, qpc, true);
        }
        // The skip probe only next to a skipped macroblock (`x264` fast P_Skip detection).
        let probe = bounds.contains(pskip_mv) && {
            let (skip, i) = (&self.mb_is_skip, mb_y * mb_w + mb_x);
            (mb_x > 0 && skip[i - 1])
                || (mb_y > 0
                    && (skip[i - mb_w]
                        || (mb_x > 0 && skip[i - mb_w - 1])
                        || (mb_x + 1 < mb_w && skip[i - mb_w + 1])))
        };
        if self.try_p_skip(
            sl, cabac, mb_x, mb_y, qp, qpc, src, pskip_mv, probe, &mut py, &mut pu, &mut pv,
        ) {
            self.mvr[0][mb_idx] = pskip_mv;
            return;
        }

        // 1. 16x16 from the predictor, zero, the neighbours' and the co-located vectors.
        let mut cands = [[0i16; 2]; 11];
        let mut n = 0;
        let mut push = |c: [i16; 2]| {
            cands[n] = c;
            n += 1;
        };
        push(mvp);
        if let Some(la) = &self.la_mv[0] {
            push(la[mb_idx].map(|v| v * 8));
        }
        // The neighbours' and the latest anchor's 16x16 search results (`x264` `mvr`).
        let (sp, tp) = (&self.mvr[0], &self.prev_mvr);
        if mb_x > 0 {
            push(sp[mb_idx - 1]);
        }
        if mb_y > 0 {
            push(sp[mb_idx - mb_w]);
            if mb_x > 0 {
                push(sp[mb_idx - mb_w - 1]);
            }
            if mb_x + 1 < mb_w {
                push(sp[mb_idx - mb_w + 1]);
            }
        }
        push(tp[mb_idx]);
        if mb_x + 1 < mb_w {
            push(tp[mb_idx + 1]);
        }
        if mb_y + 1 < self.mb_h {
            push(tp[mb_idx + mb_w]);
        }
        self.mv_costs(qp, true);
        let mv_costs = &self.mv_cost_tabs[compact_costs(qp)];
        let rl = RefLuma {
            f: &self.refp.y,
            hp: &self.hp,
        };
        let range = ME_RANGE;
        let part = |px: usize, py_: usize, w_: usize, h_: usize, mvp: [i16; 2]| PartSearch {
            rl,
            src: &src.y[py_ * 16 + px..],
            x0: x0 + px,
            y0: y0 + py_,
            w: w_,
            h: h_,
            mvp,
            lambda,
            mv_costs,
            bounds,
            thorough: true,
        };
        let me =
            |px: usize, py_: usize, w_: usize, h_: usize, mvp: [i16; 2], cands: &[[i16; 2]]| {
                part(px, py_, w_, h_, mvp).search(cands, range)
            };
        let m16 = me(0, 0, 16, 16, mvp, &cands[..n]);
        self.mvr[0][mb_idx] = m16.mv;
        // A 16x16 vector at the skip vector whose residual decimates away is P_Skip without
        // further analysis (`x264` early skip after the 16x16 search).
        if !probe
            && m16.mv == pskip_mv
            && bounds.contains(pskip_mv)
            && self.try_p_skip(
                sl, cabac, mb_x, mb_y, qp, qpc, src, pskip_mv, true, &mut py, &mut pu, &mut pv,
            )
        {
            return;
        }
        let mv_costs = &self.mv_cost_tabs[compact_costs(qp)];
        let rl = RefLuma {
            f: &self.refp.y,
            hp: &self.hp,
        };
        let part = |px: usize, py_: usize, w_: usize, h_: usize, mvp: [i16; 2]| PartSearch {
            rl,
            src: &src.y[py_ * 16 + px..],
            x0: x0 + px,
            y0: y0 + py_,
            w: w_,
            h: h_,
            mvp,
            lambda,
            mv_costs,
            bounds,
            thorough: true,
        };
        let me =
            |px: usize, py_: usize, w_: usize, h_: usize, mvp: [i16; 2], cands: &[[i16; 2]]| {
                part(px, py_, w_, h_, mvp).search(cands, range)
            };
        let mut best = InterMb {
            shape: Shape::S16x16,
            q: [Quad::l0(m16.mv); 4],
        };
        let mut best_cost = m16.cost();
        let mut best_me = [m16; 4];

        // 2. Partitions: four 8x8 searches seeded by the 16x16 vector. 16x8 and 8x16 cost more
        // time than their bits are worth once the residual rounds finely (`SIMD-0089`).
        if best_cost > lambda * P8X8_GATE {
            let mut m8 = [m16; 4];
            let mut cands8 = [m16.mv; 5];
            for i in 0..rolled(4) {
                let (bx, by) = ((i & 1) * 2, (i >> 1) * 2);
                let mvp8 = self.mvp_part(0, mb_x, mb_y, Shape::S8x8, i, 0);
                m8[i] = me(bx * 4, by * 4, 8, 8, mvp8, &cands8[..i + 1]);
                cands8[i + 1] = m8[i].mv;
                let (gx, gy) = (gx0 + bx, gy0 + by);
                fill_grid_4x4(&mut self.ref_grid, g_stride, gx, gy, 2, 2, 0);
                fill_grid_4x4(&mut self.mv_grid, g_stride, gx, gy, 2, 2, m8[i].mv);
            }
            let cost8: u32 = m8.iter().map(|m| m.cost()).sum();
            if cost8 < best_cost {
                best = InterMb {
                    shape: Shape::S8x8,
                    q: m8.map(|m| Quad::l0(m.mv)),
                };
                best_me = m8;
            }
        }

        // Quarter-pel refinement of the partitions chosen, each from the final vectors of
        // those before it.
        {
            let qref = QPEL_REFINE;
            best_cost = 0;
            for (p, &(bx, by, w4, h4)) in best.shape.parts().iter().enumerate() {
                let mvpp = self.mvp_part(0, mb_x, mb_y, best.shape, p, 0);
                let r = part(bx * 4, by * 4, w4 * 4, h4 * 4, mvpp).refine_qpel(best_me[p], qref);
                best_cost += r.cost();
                let (gx, gy) = (gx0 + bx, gy0 + by);
                fill_grid_4x4(&mut self.ref_grid, g_stride, gx, gy, w4, h4, 0);
                fill_grid_4x4(&mut self.mv_grid, g_stride, gx, gy, w4, h4, r.mv);
                for y in (by..by + h4).step_by(2) {
                    for x in (bx..bx + w4).step_by(2) {
                        best.q[quad_of(x, y)] = Quad::l0(r.mv);
                    }
                }
            }
        }

        // 3. Intra 16x16 and 4x4 against the best inter cost.
        let edges = i16_edges(&self.cur.y, mb_x, mb_y);
        let (i16_satd, i16_mode) = analyse_i16_edges(edges, src, true);
        let i16_cost = i16_satd + lambda * 4;
        if i16_cost < best_cost {
            self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
            self.code_i16_p(sl, w, cabac, mb_x, mb_y, qp, src, i16_mode);
            return;
        }
        // Intra 4x4 when intra 16x16 comes close (`x264` `mb_analyse_intra` thresholds).
        if i16_cost.min(best_cost) > lambda * 40
            && i16_cost * 2 < best_cost * 3
            && let Some(i4) = self.analyse_i4_p(mb_x, mb_y, qp, src, i16_cost.min(best_cost))
        {
            self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
            self.code_i4_p(sl, w, cabac, mb_x, mb_y, qp, src, i4);
            return;
        }

        // 4. Prediction, residual and syntax.
        let rl = RefLuma {
            f: &self.refp.y,
            hp: &self.hp,
        };
        let refs = [(rl, &self.refp), (rl, &self.refp)];
        Self::predict_inter(refs, mb_x, mb_y, &best.q, &mut py, &mut pu, &mut pv);
        let mut res = InterResidual {
            luma_q4: [[0; 16]; 16],
            luma_q8: None,
            cbp_luma: 0,
            chroma: ChromaLevels::ZERO,
            cbp_chroma: 0,
        };
        let cbp4 = quant_luma_4x4_inter(&src.y, &py, qp, &mut res.luma_q4, true);
        res.cbp_luma = cbp4;
        let q8_store;
        if sl.t8x8
            && cbp4 != 0
            && sa8d_16x16(&src.y, 16, &py, 16) < pixel_satd_16x16(&src.y, 16, &py, 16)
        {
            let (q8, cbp8, _) = quant_luma_8x8_inter(&src.y, &py, qp, true);
            q8_store = q8;
            res.luma_q8 = Some(&q8_store);
            res.cbp_luma = cbp8;
        }
        // Trailing chroma levels that do not pay are dropped by their CABAC rate. Luma levels
        // are left alone: dropping them as well gains about 1% BD-rate but codes ~0.4 dB less
        // luma PSNR at a QP (`SIMD-0089`).
        let rdo = cabac.as_ref().map(|(_, ctxs)| &ctxs[..]);
        quant_chroma_inter(src, &pu, &pv, qpc, &mut res.chroma, true, rdo);
        res.cbp_chroma = res.chroma.cbp();
        if res.cbp_luma == 0
            && res.cbp_chroma == 0
            && best.shape == Shape::S16x16
            && best.q[0].mv[0] == pskip_mv
        {
            self.code_skip(sl, cabac, mb_x, mb_y, pskip_mv);
            self.cur.store_mb(mb_x, mb_y, &py, &pu, &pv);
            return;
        }
        self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
        self.code_inter_p(
            sl, w, cabac, mb_x, mb_y, qp, &best, &res, &mut py, &mut pu, &mut pv,
        );
    }
}

impl EncState {
    /// [`EncState::deblock`] for pictures whose inter macroblocks may hold several partitions:
    /// the boundary strength of every edge segment comes from the coefficients and the list 0
    /// motion of the two $4\times4$ blocks beside it (8.7.2.1).
    /// Whether the $4\times4$ blocks of the macroblock at grid index `g0` have no coefficients
    /// and all the same motion.
    fn uniform_inter(&self, g0: usize, b_slice: bool) -> bool {
        let g_stride = self.mb_w * 4;
        fn flat<T: PartialEq + Copy>(grid: &[T], g0: usize, s: usize) -> bool {
            let v = grid[g0];
            (0..4).all(|r| grid[g0 + r * s..g0 + r * s + 4].iter().all(|&x| x == v))
        }
        (0..4).all(|r| self.luma_nnz[g0 + r * g_stride..g0 + r * g_stride + 4] == [0; 4])
            && flat(&self.ref_grid, g0, g_stride)
            && flat(&self.mv_grid, g0, g_stride)
            && (!b_slice
                || (flat(&self.ref1_grid, g0, g_stride) && flat(&self.mv1_grid, g0, g_stride)))
    }

    pub(super) fn deblock_parts(&mut self, chroma_qp_offset: i32, b_slice: bool) {
        let (mb_w, mb_h) = (self.mb_w, self.mb_h);
        let g_stride = mb_w * 4;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let m = mb_y * mb_w + mb_x;
                let g0 = mb_y * 4 * g_stride + mb_x * 4;
                let intra = self.mb_is_intra[m];
                let t8 = self.mb_8x8_flag[m];
                let qp_q = self.mb_qps[m] as i32;
                let qpc_q = chroma_qp(qp_q, chroma_qp_offset) as i32;
                // An inter macroblock without coefficients and with one motion throughout has
                // no internal edge to filter.
                let edges = if !intra && self.uniform_inter(g0, b_slice) {
                    1
                } else {
                    4
                };
                for dir in 0..2 {
                    for e in 0..edges {
                        let pm = match (e, dir) {
                            (0, 0) if mb_x > 0 => m - 1,
                            (0, 1) if mb_y > 0 => m - mb_w,
                            (0, _) => continue,
                            _ if e & 1 == 1 && t8 => continue,
                            _ => m,
                        };
                        let bs: [u8; 4] = if intra || self.mb_is_intra[pm] {
                            [if e == 0 { 4 } else { 3 }; 4]
                        } else {
                            std::array::from_fn(|k| {
                                let (q, p) = if dir == 0 {
                                    let q = g0 + k * g_stride + e;
                                    (q, q - 1)
                                } else {
                                    let q = g0 + e * g_stride + k;
                                    (q, q - g_stride)
                                };
                                if self.luma_nnz[q] != 0 || self.luma_nnz[p] != 0 {
                                    2
                                } else if !b_slice {
                                    let (mq, mp) = (self.mv_grid[q], self.mv_grid[p]);
                                    u8::from(
                                        self.ref_grid[q] != self.ref_grid[p]
                                            || (mq[0] - mp[0]).abs() >= 4
                                            || (mq[1] - mp[1]).abs() >= 4,
                                    )
                                } else {
                                    // One reference per list: the list stands for the picture.
                                    let pics = |i: usize| {
                                        [
                                            if self.ref_grid[i] >= 0 { 1 } else { 0 },
                                            if self.ref1_grid[i] >= 0 { 2 } else { 0 },
                                        ]
                                    };
                                    let mvs = |i: usize| [self.mv_grid[i], self.mv1_grid[i]];
                                    u8::from(motion_differs(pics(p), mvs(p), pics(q), mvs(q)))
                                }
                            })
                        };
                        if bs == [0; 4] {
                            continue;
                        }
                        let (qp, qpc) = if pm == m {
                            (qp_q, qpc_q)
                        } else {
                            let qp_p = self.mb_qps[pm] as i32;
                            (
                                (qp_q + qp_p + 1) >> 1,
                                (qpc_q + chroma_qp(qp_p, chroma_qp_offset) as i32 + 1) >> 1,
                            )
                        };
                        let (x, y) = if dir == 0 {
                            (mb_x * 16 + e * 4, mb_y * 16)
                        } else {
                            (mb_x * 16, mb_y * 16 + e * 4)
                        };
                        if let Some(f) = EdgeFilter::new(qp, bs) {
                            f.apply(&mut self.cur.y, x, y, dir, false);
                        }
                        if e & 1 == 0
                            && let Some(f) = EdgeFilter::new(qpc, bs)
                        {
                            f.apply(&mut self.cur.u, x / 2, y / 2, dir, true);
                            f.apply(&mut self.cur.v, x / 2, y / 2, dir, true);
                        }
                    }
                }
            }
        }
    }
}

/// Whether two inter blocks predict from different pictures (`0`: none) or with vectors four
/// or more quarter samples apart (8.7.2.1), as the decoder decides it.
fn motion_differs(rp: [u8; 2], mp: [[i16; 2]; 2], rq: [u8; 2], mq: [[i16; 2]; 2]) -> bool {
    let far = |a: [i16; 2], b: [i16; 2]| (a[0] - b[0]).abs() >= 4 || (a[1] - b[1]).abs() >= 4;
    let (np, nq) = (
        rp.iter().filter(|&&r| r != 0).count(),
        rq.iter().filter(|&&r| r != 0).count(),
    );
    if np != nq {
        return true;
    }
    if np == 1 {
        let lp = if rp[0] != 0 { 0 } else { 1 };
        let lq = if rq[0] != 0 { 0 } else { 1 };
        return rp[lp] != rq[lq] || far(mp[lp], mq[lq]);
    }
    let same_set = (rp[0] == rq[0] && rp[1] == rq[1]) || (rp[0] == rq[1] && rp[1] == rq[0]);
    if !same_set {
        return true;
    }
    if rp[0] != rp[1] {
        if rp[0] == rq[0] {
            far(mp[0], mq[0]) || far(mp[1], mq[1])
        } else {
            far(mp[0], mq[1]) || far(mp[1], mq[0])
        }
    } else {
        (far(mp[0], mq[0]) || far(mp[1], mq[1])) && (far(mp[0], mq[1]) || far(mp[1], mq[0]))
    }
}
