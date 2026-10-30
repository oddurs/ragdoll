//! B pictures of the [`H264Speed::Compact`] preset (`SIMD-0089`): the pictures between the two
//! latest anchors (the middle one of a group of two or more a reference for the others), with
//! spatial direct prediction, `B_Skip`, and list 0, list 1 and bi-predicted partitions down to
//! $8\times8$ (`x264` `mb_analyse_inter_b*`).

use super::compact::{
    ME_RANGE, MeResult, PartSearch, QPEL_REFINE, RefLuma, block_cost, block_satd_avg, sa8d_16x16,
};
use super::inter::{Quad, Shape, mvp_from, neighbours, quad_of, rolled};
use super::*;

/// `MinPositive` of 8.4.1.2.2.
#[inline(always)]
fn min_positive(a: i8, b: i8) -> i8 {
    if a >= 0 && b >= 0 { a.min(b) } else { a.max(b) }
}

/// Lists a B partition predicts from: bit 0 list 0, bit 1 list 1; `0` is direct prediction.
type Dir = u8;

/// B macroblocks whose best 16x16 mode costs at most this many lambdas skip the 8x8
/// partitions, which rarely pay below it on the clips measured for `SIMD-0089`.
const B8X8_GATE: u32 = 80;

/// `x264` `i_mb_b_cost_table` and friends: approximate bits of B macroblock and sub-macroblock
/// types, by [`Dir`].
const TYPE_BITS_16X16: [u32; 4] = [1, 3, 3, 5];
const TYPE_BITS_SUB: [u32; 4] = [1, 3, 3, 5];

/// `x264` `i_mb_b16x8_cost_table` by the [`Dir`] of the two partitions.
#[inline(always)]
fn type_bits_two(d0: Dir, d1: Dir) -> u32 {
    match (d0, d1) {
        (1, 1) | (2, 2) => 5,
        (3, _) => 9,
        _ => 7,
    }
}

/// An analysed B macroblock: partitioning, prediction of each partition (by partition for
/// $16\times8$ and $8\times16$, by quadrant for $8\times8$) and the motion of its quadrants.
#[derive(Clone, Copy, Debug)]
struct BMb {
    shape: Shape,
    dir: [Dir; 4],
    q: [Quad; 4],
}

impl BMb {
    /// `mb_type` in a B slice (Table 7-14).
    fn mb_type(&self) -> u32 {
        const TWO: [[u32; 4]; 4] = [[0; 4], [0, 4, 8, 12], [0, 10, 6, 14], [0, 16, 18, 20]];
        match self.shape {
            Shape::S16x16 => self.dir[0] as u32,
            Shape::S16x8 => TWO[self.dir[0] as usize][self.dir[1] as usize],
            Shape::S8x16 => TWO[self.dir[0] as usize][self.dir[1] as usize] + 1,
            Shape::S8x8 => 22,
        }
    }

    /// The prediction of partition `p` of the shape.
    fn part_dir(&self, p: usize) -> Dir {
        self.dir[p]
    }
}

impl EncState {
    /// `ctxIdxInc` conditions of a B slice's `mb_type`: the left and top neighbours are
    /// available and neither `B_Skip` nor `B_Direct_16x16`.
    #[inline(always)]
    pub(super) fn b_type_conds(&self, mb_x: usize, mb_y: usize) -> (u8, u8) {
        let i = mb_y * self.mb_w + mb_x;
        let ok = |j: usize| !self.mb_is_skip[j] && !self.mb_is_direct[j];
        (
            u8::from(mb_x > 0 && ok(i - 1)),
            u8::from(mb_y > 0 && ok(i - self.mb_w)),
        )
    }

    /// Spatial direct motion (8.4.1.2.2, `direct_8x8_inference_flag` set) of macroblock
    /// `(mb_x, mb_y)` per quadrant: reference indices and predictors from the neighbours, with
    /// the vectors of a list zeroed in quadrants whose co-located corner block barely moves.
    #[inline(always)]
    fn direct_motion(&self, mb_x: usize, mb_y: usize, mid: bool) -> [Quad; 4] {
        let (col_ref, col_mv) = if mid {
            (&self.colb_ref, &self.colb_mv)
        } else {
            (&self.col_ref, &self.col_mv)
        };
        let g_stride = self.mb_w * 4;
        let (gx0, gy0) = (mb_x * 4, mb_y * 4);
        let nbs = [0, 1].map(|l| {
            let (rg, mg) = self.grids(l);
            neighbours(rg, mg, g_stride, gx0, gy0, 4)
        });
        let mut r = [-1i8; 2];
        for (rl, &[a, b, c]) in r.iter_mut().zip(&nbs) {
            *rl = min_positive(a.0.max(-1), min_positive(b.0.max(-1), c.0.max(-1)));
        }
        if r[0] < 0 && r[1] < 0 {
            return [Quad {
                r: [0, 0],
                mv: [[0, 0]; 2],
            }; 4];
        }
        let mvp = [0, 1].map(|l| {
            if r[l] >= 0 {
                mvp_from(nbs[l], r[l], 0)
            } else {
                [0, 0]
            }
        });
        std::array::from_fn(|i| {
            let ci = (gy0 + (i >> 1) * 3) * g_stride + gx0 + (i & 1) * 3;
            let (cr, cm) = (col_ref[ci], col_mv[ci]);
            let col_zero = cr == 0 && (-1..=1).contains(&cm[0]) && (-1..=1).contains(&cm[1]);
            let mut q = Quad {
                r: [-1, -1],
                mv: [[0, 0]; 2],
            };
            for l in 0..2 {
                if r[l] >= 0 {
                    q.r[l] = r[l];
                    if !(r[l] == 0 && col_zero) {
                        q.mv[l] = mvp[l];
                    }
                }
            }
            q
        })
    }

    /// Stores the motion of both lists of the quadrants `q` of macroblock `(mb_x, mb_y)`.
    #[inline(always)]
    fn store_quads(&mut self, mb_x: usize, mb_y: usize, q: &[Quad; 4]) {
        for (i, m) in q.iter().enumerate() {
            let (bx, by) = ((i & 1) * 2, (i >> 1) * 2);
            for l in 0..2 {
                let mv = if m.r[l] >= 0 { m.mv[l] } else { [0, 0] };
                self.set_motion(l, mb_x, mb_y, bx, by, 2, 2, m.r[l], mv);
            }
        }
    }

    /// Writes `B_Skip` for macroblock `(mb_x, mb_y)` with direct motion `dq`.
    #[inline(always)]
    fn code_b_skip(
        &mut self,
        sl: &mut SliceState,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        dq: &[Quad; 4],
    ) {
        let mb_idx = mb_y * self.mb_w + mb_x;
        if let Some((ce, ctxs)) = cabac {
            let l = u8::from(mb_x > 0 && !self.mb_is_skip[mb_idx - 1]);
            let t = u8::from(mb_y > 0 && !self.mb_is_skip[mb_idx - self.mb_w]);
            ce.encode_b_skip(ctxs, l, t, true);
        }
        self.store_quads(mb_x, mb_y, dq);
        self.mb_is_skip[mb_idx] = true;
        self.mb_qps[mb_idx] = sl.cur_qp as u8;
        sl.prev_qp_delta_nz = false;
    }

    out_of_line_v3!(pub(super) fn code_b_mb_compact / code_b_mb_v3 => code_b_mb_body(
        &mut self,
        sl: &mut SliceState,
        w: &mut Bits,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        src: &MbSrc,
    ));

    out_of_line_v3!(fn code_i16_b / code_i16_b_v3 => code_i16(
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

    out_of_line_v3!(fn code_inter_b / code_inter_b_v3 => code_inter_b_body(
        &mut self,
        sl: &mut SliceState,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        mb: &BMb,
        res: &InterResidual,
        py: &mut [u8; 256],
        pu: &mut [u8; 64],
        pv: &mut [u8; 64],
    ));

    /// Analyses and codes one macroblock of a B picture: `B_Skip` when the direct prediction's
    /// residual decimates to nothing, else the cheapest by SATD and motion cost of direct, list
    /// 0, list 1 and bi-predicted $16\times16$, the $8\times8$, $16\times8$ and $8\times16$
    /// partitions when they promise to pay, and intra $16\times16$ (`x264` with `subme` 2).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn code_b_mb_body(
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
        let lambda = LAMBDA[qp as usize] as u32;
        let mut py = [0u8; 256];
        let mut pu = [0u8; 64];
        let mut pv = [0u8; 64];

        self.mv_costs(qp);
        // 1. Direct prediction and the B_Skip probe.
        let dq = self.direct_motion(mb_x, mb_y, sl.b_refs[1] == RefSel::Mid);
        let direct_ok = dq
            .iter()
            .all(|q| (0..2).all(|l| q.r[l] < 0 || bounds.contains(q.mv[l])));
        // Disjoint borrows of the reference pictures, so the motion grids stay writable.
        let (rp2, hp2, rp, hp, rb, hpb) = (
            &self.refp2,
            &self.hp2,
            &self.refp,
            &self.hp,
            &self.refb,
            &self.hpb,
        );
        let refs = sl.b_refs.map(|r| {
            let (pic, hp) = match r {
                RefSel::Prev => (rp2, hp2),
                RefSel::Next => (rp, hp),
                RefSel::Mid => (rb, hpb),
            };
            (RefLuma { f: &pic.y, hp }, pic)
        });
        let mut direct8 = [u32::MAX; 4];
        let mut direct_cost = u32::MAX;
        if direct_ok {
            Self::predict_inter(refs, mb_x, mb_y, &dq, &mut py, &mut pu, &mut pv);
            if probe_skip_luma(src, &py, qp) && probe_skip_chroma(src, &pu, &pv, qpc) {
                for l in 0..2 {
                    self.mvr[l][mb_idx] = if dq[0].r[l] >= 0 { dq[0].mv[l] } else { [0, 0] };
                }
                self.code_b_skip(sl, cabac, mb_x, mb_y, &dq);
                self.cur.store_mb(mb_x, mb_y, &py, &pu, &pv);
                return;
            }
            for (i, d) in direct8.iter_mut().enumerate() {
                let o = (i >> 1) * 128 + (i & 1) * 8;
                *d = block_cost(&src.y[o..], 16, &py[o..], 16, 8, 8, true);
            }
            direct_cost = direct8.iter().sum::<u32>() + lambda * TYPE_BITS_16X16[0];
        }
        let dpred = py;

        // 2. 16x16 motion search in each list, then bi-prediction from the two vectors.
        let mv_costs = &self.mv_cost_tabs[qp as usize];
        let range = ME_RANGE;
        let part =
            |l: usize, px: usize, py_: usize, w_: usize, h_: usize, mvp: [i16; 2]| PartSearch {
                rl: refs[l].0,
                src: &src.y[py_ * 16 + px..],
                x0: x0 + px,
                y0: y0 + py_,
                w: w_,
                h: h_,
                mvp,
                lambda,
                mv_costs,
                bounds,
            };
        // Loops over lists and quadrants are kept rolled (`rolled`): unrolled, a macroblock's
        // analysis runs through more code than the instruction cache holds.
        let mut m16 = [MeResult {
            mv: [0; 2],
            satd: 0,
            mv_cost: 0,
        }; 2];
        for (l, m) in m16.iter_mut().enumerate().take(rolled(2)) {
            let mvp = self.mvp_part(l, mb_x, mb_y, Shape::S16x16, 0, 0);
            let mut cands = [[0i16; 2]; 9];
            let mut n = 0;
            let mut push = |c: [i16; 2]| {
                cands[n] = c;
                n += 1;
            };
            push(mvp);
            if let Some(la) = &self.la_mv[l] {
                push(la[mb_idx].map(|v| v * 8));
            }
            if dq[0].r[l] >= 0 {
                push(dq[0].mv[l]);
            }
            let r = &self.mvr[l];
            if mb_x > 0 {
                push(r[mb_idx - 1]);
            }
            if mb_y > 0 {
                push(r[mb_idx - mb_w]);
                if mb_x + 1 < mb_w {
                    push(r[mb_idx - mb_w + 1]);
                }
            }
            *m = part(l, 0, 0, 16, 16, mvp).search(&cands[..n], range);
        }
        self.mvr[0][mb_idx] = m16[0].mv;
        self.mvr[1][mb_idx] = m16[1].mv;
        // SATD of the bi-prediction from vectors `v` over the `w`x`h` block at `(px, py)`.
        let bi_satd = |v: [[i16; 2]; 2], px: usize, py_: usize, w_: usize, h_: usize| {
            // A list's prediction is read in place when it is one plane's block, else built.
            let mut t = [[0u8; 256]; 2];
            let mut plane = [None; 2];
            for l in 0..2 {
                let mv = [v[l][0] as i32, v[l][1] as i32];
                let rl = refs[l].0;
                match rl.src(x0 + px, y0 + py_, mv) {
                    (a, None, [pa, _]) => plane[l] = Some((pa, a)),
                    _ => rl.pred(x0 + px, y0 + py_, w_, h_, mv, &mut t[l], 16),
                }
            }
            let side = |l: usize| match plane[l] {
                Some((pa, a)) => (&pa[a..], refs[l].0.f.stride),
                None => (&t[l][..], 16),
            };
            let ((p0, s0), q) = (side(0), side(1));
            block_satd_avg(&src.y[py_ * 16 + px..], 16, p0, s0, Some(q), w_, h_)
        };
        let quad = |d: Dir, v: [[i16; 2]; 2]| Quad {
            r: [
                if d & 1 != 0 { 0 } else { -1 },
                if d & 2 != 0 { 0 } else { -1 },
            ],
            mv: [
                if d & 1 != 0 { v[0] } else { [0, 0] },
                if d & 2 != 0 { v[1] } else { [0, 0] },
            ],
        };
        let mut best = BMb {
            shape: Shape::S16x16,
            dir: [0; 4],
            q: dq,
        };
        let mut best_cost = direct_cost;
        // The search results behind the best mode's vectors, by partition and list.
        let mut best_me = [m16[0]; 4];
        let v16 = [m16[0].mv, m16[1].mv];
        let cost_bi16 = bi_satd(v16, 0, 0, 16, 16)
            + m16[0].mv_cost
            + m16[1].mv_cost
            + lambda * TYPE_BITS_16X16[3];
        for (d, c) in [
            (1, m16[0].cost() + lambda * TYPE_BITS_16X16[1]),
            (2, m16[1].cost() + lambda * TYPE_BITS_16X16[2]),
            (3, cost_bi16),
        ] {
            if c < best_cost {
                best_cost = c;
                best = BMb {
                    shape: Shape::S16x16,
                    dir: [d; 4],
                    q: [quad(d, v16); 4],
                };
                best_me = [m16[0], m16[1], m16[0], m16[1]];
            }
        }

        let mut best_me8 = None;
        // 3. 8x8 partitions, unless the 16x16 modes already predict well: each quadrant searched
        // from the 16x16 vector in the list cheaper at 16x16, the other list keeping its 16x16
        // vector, then the cheapest of list 0, list 1, bi-prediction and direct.
        if best_cost > lambda * B8X8_GATE {
            let mut m8: [[MeResult; 4]; 2] = [[m16[0]; 4], [m16[1]; 4]];
            let mut satd8 = [[0u32; 4]; 3];
            let mut dir8 = [0 as Dir; 4];
            let mut q8 = [Quad::default(); 4];
            let mut cost8 = lambda * 9;
            let weak = match m16[0].cost().cmp(&m16[1].cost()) {
                std::cmp::Ordering::Less => 1,
                std::cmp::Ordering::Greater => 0,
                std::cmp::Ordering::Equal => 2,
            };
            for i in 0..rolled(4) {
                let (bx, by) = ((i & 1) * 2, (i >> 1) * 2);
                for l in 0..rolled(2) {
                    let mvp = self.mvp_part(l, mb_x, mb_y, Shape::S8x8, i, 0);
                    let ps = part(l, bx * 4, by * 4, 8, 8, mvp);
                    m8[l][i] = if l == weak {
                        let mv = m16[l].mv;
                        let satd = ps.rl.cost(
                            ps.src,
                            ps.x0,
                            ps.y0,
                            8,
                            8,
                            [mv[0] as i32, mv[1] as i32],
                            true,
                        );
                        ps.refine_qpel(
                            MeResult {
                                mv,
                                satd,
                                mv_cost: 0,
                            },
                            0,
                        )
                    } else {
                        // Quarter-pel vectors only for the partitions chosen.
                        ps.search_qpel(&[m16[l].mv], range, 0)
                    };
                    satd8[l][i] = m8[l][i].satd;
                }
                let v = [m8[0][i].mv, m8[1][i].mv];
                satd8[2][i] = bi_satd(v, bx * 4, by * 4, 8, 8);
                let mut c = (m8[0][i].cost() + lambda * TYPE_BITS_SUB[1], 1 as Dir);
                for (cc, d) in [
                    (m8[1][i].cost() + lambda * TYPE_BITS_SUB[2], 2),
                    (
                        satd8[2][i]
                            + m8[0][i].mv_cost
                            + m8[1][i].mv_cost
                            + lambda * TYPE_BITS_SUB[3],
                        3,
                    ),
                    (direct8[i].saturating_add(lambda * TYPE_BITS_SUB[0]), 0),
                ] {
                    if cc < c.0 {
                        c = (cc, d);
                    }
                }
                cost8 += c.0;
                dir8[i] = c.1;
                q8[i] = if c.1 == 0 { dq[i] } else { quad(c.1, v) };
                // Later quadrants predict their vectors from this one's.
                for l in 0..2 {
                    let mv = if q8[i].r[l] >= 0 { q8[i].mv[l] } else { [0, 0] };
                    let (gx, gy) = (gx0 + bx, gy0 + by);
                    let (rg, mg) = if l == 0 {
                        (&mut self.ref_grid, &mut self.mv_grid)
                    } else {
                        (&mut self.ref1_grid, &mut self.mv1_grid)
                    };
                    fill_grid_4x4(rg, g_stride, gx, gy, 2, 2, q8[i].r[l]);
                    fill_grid_4x4(mg, g_stride, gx, gy, 2, 2, mv);
                }
            }
            if cost8 < best_cost {
                best_cost = cost8;
                best = BMb {
                    shape: Shape::S8x8,
                    dir: dir8,
                    q: q8,
                };
                best_me8 = Some(m8);
            }

            // 16x8 and 8x16, when their cost estimated from the 8x8 SATDs promises to pay, the
            // better estimate first (`x264` `mb_analyse_inter_b16x8`).
            let estimate = |shape: Shape| {
                let pair = |p: usize| {
                    if shape == Shape::S16x8 {
                        (2 * p, 2 * p + 1)
                    } else {
                        (p, p + 2)
                    }
                };
                let mut pest = [0u32; 2];
                let mut dest = [0 as Dir; 2];
                for p in 0..2 {
                    let (a, b) = pair(p);
                    let mvc = |l: usize| (m8[l][a].mv_cost + m8[l][b].mv_cost).div_ceil(2);
                    let mut e = (satd8[0][a] + satd8[0][b] + mvc(0), 1);
                    for (c, d) in [
                        (satd8[1][a] + satd8[1][b] + mvc(1), 2),
                        (satd8[2][a] + satd8[2][b] + mvc(0) + mvc(1), 3),
                    ] {
                        if c < e.0 {
                            e = (c, d);
                        }
                    }
                    pest[p] = e.0;
                    dest[p] = e.1;
                }
                pest[1] += lambda * type_bits_two(dest[0], dest[1]);
                pest
            };
            let ests = [Shape::S16x8, Shape::S8x16].map(|sh| (sh, estimate(sh)));
            let order = if ests[1].1[0] + ests[1].1[1] < ests[0].1[0] + ests[0].1[1] {
                [ests[1], ests[0]]
            } else {
                ests
            };
            for (shape, pest) in order {
                let pair = |p: usize| {
                    if shape == Shape::S16x8 {
                        (2 * p, 2 * p + 1)
                    } else {
                        (p, p + 2)
                    }
                };
                if pest[0] + pest[1] >= best_cost {
                    continue;
                }
                let mut total = 0;
                let mut dirs = [0 as Dir; 2];
                let mut vs = [[[0i16; 2]; 2]; 2];
                let mut mes = [m16[0]; 4];
                for (p, &(bx, by, w4, h4)) in shape.parts().iter().enumerate() {
                    let (a, b) = pair(p);
                    let mut mp = [m16[0]; 2];
                    for l in 0..2 {
                        let mvp = self.mvp_part(l, mb_x, mb_y, shape, p, 0);
                        let c = [m16[l].mv, m8[l][a].mv, m8[l][b].mv];
                        // Quarter-pel vectors only for the partitions chosen.
                        mp[l] =
                            part(l, bx * 4, by * 4, w4 * 4, h4 * 4, mvp).search_qpel(&c, range, 0);
                    }
                    let v = [mp[0].mv, mp[1].mv];
                    mes[2 * p] = mp[0];
                    mes[2 * p + 1] = mp[1];
                    let bi = bi_satd(v, bx * 4, by * 4, w4 * 4, h4 * 4)
                        + mp[0].mv_cost
                        + mp[1].mv_cost
                        + lambda;
                    let mut c = (mp[0].cost(), 1 as Dir);
                    for (cc, d) in [(mp[1].cost(), 2), (bi, 3)] {
                        if cc < c.0 {
                            c = (cc, d);
                        }
                    }
                    total += c.0;
                    dirs[p] = c.1;
                    vs[p] = v;
                    // The first partition and the second's estimate already lose.
                    if p == 0 && total + pest[1] > best_cost {
                        total = u32::MAX;
                        break;
                    }
                    let qq = quad(c.1, v);
                    for l in 0..2 {
                        let mv = if qq.r[l] >= 0 { qq.mv[l] } else { [0, 0] };
                        let (gx, gy) = (gx0 + bx, gy0 + by);
                        let (rg, mg) = if l == 0 {
                            (&mut self.ref_grid, &mut self.mv_grid)
                        } else {
                            (&mut self.ref1_grid, &mut self.mv1_grid)
                        };
                        fill_grid_4x4(rg, g_stride, gx, gy, w4, h4, qq.r[l]);
                        fill_grid_4x4(mg, g_stride, gx, gy, w4, h4, mv);
                    }
                }
                total = total.saturating_add(lambda * type_bits_two(dirs[0], dirs[1]));
                if total < best_cost {
                    best_cost = total;
                    let q = std::array::from_fn(|i| {
                        let p = if shape == Shape::S16x8 { i >> 1 } else { i & 1 };
                        quad(dirs[p], vs[p])
                    });
                    best = BMb {
                        shape,
                        dir: [dirs[0], dirs[1], 0, 0],
                        q,
                    };
                    best_me = mes;
                    best_me8 = None;
                }
            }
        }

        // Quarter-pel refinement of the vectors of the mode chosen, list by list and partition
        // by partition as they are written, each list's vector by its own SATD.
        let qref = QPEL_REFINE;
        if !(best.shape == Shape::S16x16 && best.dir[0] == 0) {
            for l in 0..rolled(2) {
                for (p, &(bx, by, w4, h4)) in best.shape.parts().iter().enumerate() {
                    let qi = quad_of(bx, by);
                    let (gx, gy) = (gx0 + bx, gy0 + by);
                    let used = best.dir[p] & (1 << l) != 0;
                    let (r, mv) = if used {
                        let me = match best_me8 {
                            Some(m8) => m8[l][p],
                            None if best.shape == Shape::S16x16 => best_me[l],
                            None => best_me[2 * p + l],
                        };
                        let mvp = self.mvp_part(l, mb_x, mb_y, best.shape, p, 0);
                        let mv = part(l, bx * 4, by * 4, w4 * 4, h4 * 4, mvp)
                            .refine_qpel(me, qref)
                            .mv;
                        (0, mv)
                    } else {
                        let q = best.q[qi];
                        (q.r[l], if q.r[l] >= 0 { q.mv[l] } else { [0, 0] })
                    };
                    for y in (by..by + h4).step_by(2) {
                        for x in (bx..bx + w4).step_by(2) {
                            if used {
                                best.q[quad_of(x, y)].mv[l] = mv;
                            }
                        }
                    }
                    let (rg, mg) = if l == 0 {
                        (&mut self.ref_grid, &mut self.mv_grid)
                    } else {
                        (&mut self.ref1_grid, &mut self.mv1_grid)
                    };
                    fill_grid_4x4(rg, g_stride, gx, gy, w4, h4, r);
                    fill_grid_4x4(mg, g_stride, gx, gy, w4, h4, mv);
                }
            }
        }

        // 4. Intra 16x16.
        let edges = i16_edges(&self.cur.y, mb_x, mb_y);
        let (i16_satd, i16_mode) = analyse_i16_edges(edges, src, true);
        if i16_satd + lambda * 13 < best_cost {
            self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
            self.code_i16_b(sl, w, cabac, mb_x, mb_y, qp, src, i16_mode);
            self.set_motion(1, mb_x, mb_y, 0, 0, 4, 4, -1, [0, 0]);
            return;
        }

        // 5. Prediction, residual and syntax.
        if best.shape == Shape::S16x16 && best.dir[0] == 0 {
            py = dpred;
        } else {
            Self::predict_inter(refs, mb_x, mb_y, &best.q, &mut py, &mut pu, &mut pv);
        }
        let mut res = InterResidual {
            luma_q4: [[0; 16]; 16],
            luma_q8: None,
            cbp_luma: 0,
            chroma: ChromaLevels::ZERO,
            cbp_chroma: 0,
        };
        let cbp4 = quant_luma_4x4_inter(&src.y, &py, qp, &mut res.luma_q4);
        res.cbp_luma = cbp4;
        let q8_store;
        if sl.t8x8
            && cbp4 != 0
            && sa8d_16x16(&src.y, 16, &py, 16) < pixel_satd_16x16(&src.y, 16, &py, 16)
        {
            let (q8, cbp8, _) = quant_luma_8x8_inter(&src.y, &py, qp);
            q8_store = q8;
            res.luma_q8 = Some(&q8_store);
            res.cbp_luma = cbp8;
        }
        quant_chroma_inter(src, &pu, &pv, qpc, &mut res.chroma);
        res.cbp_chroma = res.chroma.cbp();
        if best.shape == Shape::S16x16
            && best.dir[0] == 0
            && res.cbp_luma == 0
            && res.cbp_chroma == 0
        {
            self.code_b_skip(sl, cabac, mb_x, mb_y, &dq);
            self.cur.store_mb(mb_x, mb_y, &py, &pu, &pv);
            return;
        }
        self.begin_coded_mb(sl, w, cabac, mb_x, mb_y);
        self.code_inter_b(
            sl, cabac, mb_x, mb_y, qp, &best, &res, &mut py, &mut pu, &mut pv,
        );
    }

    /// Writes and reconstructs B macroblock `mb` with residual `res` over the prediction in
    /// `py`, `pu` and `pv`, and stores its motion. The vectors of each list are written
    /// partition by partition, each predicted from the motion stored before it (7.3.5.1).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn code_inter_b_body(
        &mut self,
        sl: &mut SliceState,
        cabac: &mut Cabac,
        mb_x: usize,
        mb_y: usize,
        qp: u8,
        mb: &BMb,
        res: &InterResidual,
        py: &mut [u8; 256],
        pu: &mut [u8; 64],
        pv: &mut [u8; 64],
    ) {
        let Some((ce, ctxs)) = cabac else {
            unreachable!("B pictures are coded with CABAC");
        };
        let mb_w = self.mb_w;
        let g_stride = mb_w * 4;
        let mb_idx = mb_y * mb_w + mb_x;
        let (ca, cb) = self.b_type_conds(mb_x, mb_y);
        let mb_type = mb.mb_type();
        ce.encode_b_mb_type(ctxs, ca, cb, mb_type);
        if mb_type == 0 {
            self.mb_is_direct[mb_idx] = true;
            self.store_quads(mb_x, mb_y, &mb.q);
        } else {
            if mb.shape == Shape::S8x8 {
                for &d in &mb.dir {
                    ce.encode_b_sub_mb_type(ctxs, d);
                }
            }
            // No reference until a partition's vector is written; direct quadrants first.
            for l in 0..2 {
                self.set_motion(l, mb_x, mb_y, 0, 0, 4, 4, -1, [0, 0]);
            }
            if mb.shape == Shape::S8x8 {
                for (i, &d) in mb.dir.iter().enumerate() {
                    if d == 0 {
                        let (bx, by) = ((i & 1) * 2, (i >> 1) * 2);
                        for l in 0..2 {
                            let m = mb.q[i];
                            let mv = if m.r[l] >= 0 { m.mv[l] } else { [0, 0] };
                            self.set_motion(l, mb_x, mb_y, bx, by, 2, 2, m.r[l], mv);
                        }
                    }
                }
            }
            for l in 0..2 {
                for (p, &(bx, by, w4, h4)) in mb.shape.parts().iter().enumerate() {
                    if mb.part_dir(p) & (1 << l) == 0 {
                        continue;
                    }
                    let mv = mb.q[quad_of(bx, by)].mv[l];
                    let mvp = self.mvp_part(l, mb_x, mb_y, mb.shape, p, 0);
                    let mvd = [mv[0] - mvp[0], mv[1] - mvp[1]];
                    let (gx, gy) = (mb_x * 4 + bx, mb_y * 4 + by);
                    let grid = if l == 0 {
                        &mut self.mvd_grid
                    } else {
                        &mut self.mvd1_grid
                    };
                    let (ax, ay) = neighbor_mvd_sum_16x16(grid, g_stride, gx, gy);
                    ce.encode_mvd(ctxs, 0, ax, mvd[0]);
                    ce.encode_mvd(ctxs, 1, ay, mvd[1]);
                    fill_grid_4x4(grid, g_stride, gx, gy, w4, h4, mvd);
                    self.set_motion(l, mb_x, mb_y, bx, by, w4, h4, 0, mv);
                }
            }
        }
        let cbp_luma = res.cbp_luma;
        let cbp_chroma = res.cbp_chroma;
        let cbp = cbp_luma | (cbp_chroma << 4);
        let qp_delta = if cbp != 0 { qp as i32 - sl.cur_qp } else { 0 };
        let eff_qp = (sl.cur_qp + qp_delta).clamp(0, 51) as u8;
        let eff_qpc = chroma_qp(eff_qp as i32, sl.chroma_qp_offset);
        if res.luma_q8.is_some() && cbp_luma != 0 {
            self.mb_8x8_flag[mb_idx] = true;
        }
        encode_cabac_cbp_and_residuals(
            ce,
            ctxs,
            mb_x,
            mb_y,
            mb_w,
            g_stride,
            mb_w * 2,
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
        sl.cur_qp = eff_qp as i32;
        self.mb_qps[mb_idx] = eff_qp;
        if let Some(luma_q8) = res.luma_q8 {
            for (i8x8, q8) in luma_q8.iter().enumerate() {
                if cbp_luma & (1 << i8x8) != 0 {
                    let o = (i8x8 >> 1) * 128 + (i8x8 & 1) * 8;
                    add8x8_idct_u8(&mut py[o..], 16, &dequantize_block8x8(q8, eff_qp));
                }
            }
        } else {
            recon_luma_4x4(py, &res.luma_q4, cbp_luma, eff_qp);
        }
        recon_chroma(pu, pv, &res.chroma, cbp_chroma, eff_qpc);
        self.cur.store_mb(mb_x, mb_y, py, pu, pv);
    }
}
