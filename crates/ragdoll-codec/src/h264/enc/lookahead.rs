//! Lookahead of the [`H264Speed::Compact`] preset (`SIMD-0089`): half-resolution copies of the
//! frames waiting to be coded, their motion against the anchors around them, and the frame type
//! decision that places B pictures by the estimated costs of the coding orders (`x264`
//! `slicetype.c`).

use super::*;
use ragdoll_simd::{downscale_2x_row, pixel_sad_x4_wxh, pixel_satd_wxh};

/// Padding of a half-resolution plane: the search moves an $8\times8$ block at most
/// [`LA_RANGE`] samples past the picture edge.
const LA_PAD: usize = 32;
/// Search range of the half-resolution motion search, in half-resolution samples.
const LA_RANGE: i32 = 16;
/// Lagrangian multiplier of the lookahead (`x264` `X264_LOOKAHEAD_QP` 12).
const LA_LAMBDA: u32 = 1;
/// SATD below which a block predicted at zero is not searched: lower than `x264`'s 64, which
/// changes the frame types placed on the clips measured for `SIMD-0089`.
const LA_SKIP_SATD: u32 = 16;
/// B frame costs are divided by this over 100 (`x264` 120, `--b-bias 0`): more B frames than
/// `x264` places pay with the Compact preset's quantizer offsets.
const B_COST_SCALE: u64 = 150;

/// A frame of the lookahead: its half-resolution luma and the motion found against other frames.
#[derive(Clone, Debug, Default)]
pub(in crate::h264) struct Lowres {
    /// Display index.
    pub(super) disp: u32,
    plane: Plane,
    /// Intra cost of each $8\times8$ block, once computed.
    intra: Vec<u32>,
    /// Per reference frame (display index): the vector of each block, in half-resolution
    /// samples, and its cost.
    me: Vec<(u32, Vec<[i16; 2]>, Vec<u32>)>,
}

impl Lowres {
    /// The half-resolution copy of `frame`'s luma (`x264_frame_init_lowres`).
    pub(super) fn new(frame: &VideoFrame, disp: u32, mb_w: usize, mb_h: usize) -> Self {
        let (w, h) = (mb_w * 8, mb_h * 8);
        let mut plane = Plane::alloc(w, h, LA_PAD);
        let src = frame.plane(0);
        let (fw, fh) = (frame.width, frame.height);
        let row = |y: usize| src.row(y.min(fh - 1));
        // Columns whose source pair lies inside the picture, then the edge ones.
        let inner = (fw / 2).min(w);
        for y in 0..h {
            let (r0, r1) = (row(2 * y), row(2 * y + 1));
            let o = plane.idx(0, y);
            let dst = &mut plane.buf[o..o + w];
            downscale_2x_row(r0, r1, &mut dst[..inner]);
            for (x, d) in dst.iter_mut().enumerate().skip(inner) {
                let (x0, x1) = ((2 * x).min(fw - 1), (2 * x + 1).min(fw - 1));
                let a = (r0[x0] as u16 + r1[x0] as u16 + 1) >> 1;
                let b = (r0[x1] as u16 + r1[x1] as u16 + 1) >> 1;
                *d = ((a + b + 1) >> 1) as u8;
            }
        }
        plane.extend_borders();
        Self {
            disp,
            plane,
            intra: Vec::new(),
            me: Vec::new(),
        }
    }

    /// The vectors of this frame against reference `r` (display index), if searched; in
    /// half-resolution samples.
    pub(super) fn motion(&self, r: u32) -> Option<&[[i16; 2]]> {
        self.me.iter().find(|e| e.0 == r).map(|e| &e.1[..])
    }
}

/// SATD of two $8\times8$ blocks.
#[inline(always)]
fn satd8(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    super::compact::satd_8x8(a, sa, b, sb)
}

/// Intra cost of every block of `f` (`x264` lowres intra: the best of DC, horizontal and
/// vertical by SATD, plus a penalty).
fn intra_costs(f: &Lowres) -> Vec<u32> {
    let p = &f.plane;
    let (bw, bh) = (p.w / 8, p.h / 8);
    let mut out = Vec::with_capacity(bw * bh);
    for by in 0..bh {
        for bx in 0..bw {
            let o = p.idx(bx * 8, by * 8);
            let top: [u8; 8] = std::array::from_fn(|i| p.buf[o - p.stride + i]);
            let left: [u8; 8] = std::array::from_fn(|i| p.buf[o + i * p.stride - 1]);
            let dc = ((top.iter().chain(&left).map(|&v| v as u32).sum::<u32>() + 8) >> 4) as u8;
            let mut pred = [[0u8; 64]; 3];
            for y in 0..8 {
                for x in 0..8 {
                    pred[0][y * 8 + x] = dc;
                    pred[1][y * 8 + x] = left[y];
                    pred[2][y * 8 + x] = top[x];
                }
            }
            let best = pred
                .iter()
                .map(|q| satd8(&p.buf[o..], p.stride, q, 8))
                .min()
                .unwrap_or(0);
            out.push(best + 5 * LA_LAMBDA + 4);
        }
    }
    out
}

/// `se(v)` bits of a vector difference of the half-resolution search, by `d` modulo 256:
/// vectors and predictors stay within [`LA_RANGE`] twice over.
static LA_MV_BITS: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let d = i as i8 as i32;
        let v = d.unsigned_abs() * 2 - (d > 0) as u32;
        t[i] = (2 * (32 - (v + 1).leading_zeros()) - 1) as u8;
        i += 1;
    }
    t
};

/// The search of one block: its source, the reference plane, the predicted vector and the best
/// vector so far with its cost.
struct BlockSearch<'a> {
    src: &'a [u8],
    ps: usize,
    q: &'a [u8],
    qs: isize,
    /// Index of the block's position in `q`.
    base: isize,
    mvp: [i32; 2],
    best: [i32; 2],
    best_cost: u32,
}

impl BlockSearch<'_> {
    #[inline(always)]
    fn mvc(&self, m: [i32; 2]) -> u32 {
        let bits = |d: i32| LA_MV_BITS[(d as u32 & 255) as usize] as u32;
        LA_LAMBDA * (bits(m[0] - self.mvp[0]) + bits(m[1] - self.mvp[1]))
    }

    #[inline(always)]
    fn blk(&self, m: [i32; 2]) -> &[u8] {
        &self.q[(self.base + m[1] as isize * self.qs + m[0] as isize) as usize..]
    }

    /// Measures four points, in order, the first of equal costs winning as a sequential search
    /// would pick it; returns the last one that improved. Points outside the range are replaced
    /// by one already measured, which then cannot win.
    #[inline(always)]
    fn take(&mut self, pts: [[i32; 2]; 4]) -> Option<usize> {
        let sads = pixel_sad_x4_wxh::<8, 8>(
            self.src,
            self.ps,
            pts.map(|m| self.blk(m)),
            self.qs as usize,
        );
        let mut moved = None;
        for k in 0..4 {
            let cost = sads[k] + self.mvc(pts[k]);
            if cost < self.best_cost {
                self.best_cost = cost;
                self.best = pts[k];
                moved = Some(k);
            }
        }
        moved
    }
}

/// Searches every block of `f` in `r`, bottom-right first so that the vectors of the blocks
/// right of and below a block seed its search (`x264_slicetype_mb_cost`); returns the vectors
/// and their costs.
fn search(f: &Lowres, r: &Lowres) -> (Vec<[i16; 2]>, Vec<u32>) {
    let (p, q) = (&f.plane, &r.plane);
    let (bw, bh) = (p.w / 8, p.h / 8);
    let (ps, qs) = (p.stride, q.stride as isize);
    let mut mvs = vec![[0i16; 2]; bw * bh];
    let mut costs = vec![0u32; bw * bh];
    for by in (0..bh).rev() {
        for bx in (0..bw).rev() {
            let i = by * bw + bx;
            let src = &p.buf[p.idx(bx * 8, by * 8)..];
            let base = q.idx(bx * 8, by * 8) as isize;
            // Vectors keep the block inside the padding and the search range.
            let (x, y) = ((bx * 8) as i32, (by * 8) as i32);
            let pad = LA_PAD as i32;
            let lo = [
                (1 - pad - x).max(-2 * LA_RANGE),
                (1 - pad - y).max(-2 * LA_RANGE),
            ];
            let hi = [
                ((q.w as i32 + pad - 9) - x).min(2 * LA_RANGE),
                ((q.h as i32 + pad - 9) - y).min(2 * LA_RANGE),
            ];
            let span = [(hi[0] - lo[0]) as u32, (hi[1] - lo[1]) as u32];
            let lim = |m: [i32; 2]| {
                ((m[0] - lo[0]) as u32 <= span[0]) & ((m[1] - lo[1]) as u32 <= span[1])
            };

            // Candidates: the blocks right, below, below-left and below-right; their median
            // predicts the vector.
            let mut cands = [[0i16; 2]; 4];
            let mut n = 0;
            if bx + 1 < bw {
                cands[n] = mvs[i + 1];
                n += 1;
            }
            if by + 1 < bh {
                cands[n] = mvs[i + bw];
                n += 1;
                if bx > 0 {
                    cands[n] = mvs[i + bw - 1];
                    n += 1;
                }
                if bx + 1 < bw {
                    cands[n] = mvs[i + bw + 1];
                    n += 1;
                }
            }
            let mvp = if n <= 1 {
                [cands[0][0] as i32, cands[0][1] as i32]
            } else {
                [
                    median3(cands[0][0], cands[1][0], cands[2.min(n - 1)][0]) as i32,
                    median3(cands[0][1], cands[1][1], cands[2.min(n - 1)][1]) as i32,
                ]
            };
            // A block predicted at zero and nearly matching there keeps the zero vector
            // (`x264_slicetype_mb_cost`, which skips below 64).
            if mvp == [0, 0] {
                let c = satd8(src, ps, &q.buf[base as usize..], qs as usize);
                if c < LA_SKIP_SATD {
                    mvs[i] = [0, 0];
                    costs[i] = c;
                    continue;
                }
            }
            let mut bs = BlockSearch {
                src,
                ps,
                q: &q.buf,
                qs,
                base,
                mvp,
                best: [0; 2],
                best_cost: u32::MAX,
            };
            let mut pts = [[0i32; 2]; 8];
            let mut np = 1;
            for c in &cands[..n] {
                let m = [c[0] as i32, c[1] as i32];
                if lim(m) && !pts[..np].contains(&m) {
                    pts[np] = m;
                    np += 1;
                }
            }
            bs.take([pts[0], pts[1], pts[2], pts[3]]);
            if np > 4 {
                bs.take([pts[4]; 4]);
            }
            // Hexagon of radius 2 in circular order: after a step toward point `d` only the
            // three points around it are new, taken in this order.
            const HEX: [[i32; 2]; 6] = [[-2, 0], [-1, 2], [1, 2], [2, 0], [1, -2], [-1, -2]];
            const NEXT: [[usize; 3]; 6] = [
                [0, 1, 5],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [0, 4, 5],
            ];
            let at = |from: [i32; 2], d: [i32; 2]| {
                let m = [from[0] + d[0], from[1] + d[1]];
                if lim(m) { m } else { from }
            };
            let mut dir = {
                let from = bs.best;
                let pts = HEX.map(|d| at(from, d));
                let a = bs.take([pts[0], pts[1], pts[2], pts[3]]);
                let b = bs.take([pts[4], pts[5], from, from]);
                b.map(|k| k + 4).or(a)
            };
            for _ in 1..LA_RANGE / 2 {
                let Some(d) = dir else { break };
                let from = bs.best;
                let ks = NEXT[d];
                dir = bs
                    .take([
                        at(from, HEX[ks[0]]),
                        at(from, HEX[ks[1]]),
                        at(from, HEX[ks[2]]),
                        from,
                    ])
                    .map(|i| ks[i]);
            }
            let from = bs.best;
            bs.take([[0, -1], [-1, 0], [1, 0], [0, 1]].map(|d| at(from, d)));
            let best = bs.best;
            let mut cost = satd8(src, ps, bs.blk(best), qs as usize);
            if best != [0, 0] {
                cost += bs.mvc(best) + 5 * LA_LAMBDA;
            }
            mvs[i] = [best[0] as i16, best[1] as i16];
            costs[i] = cost;
        }
    }
    (mvs, costs)
}

/// Frames of the lookahead and the costs computed between them.
#[derive(Clone, Debug, Default)]
pub(in crate::h264) struct Lookahead {
    /// The latest anchor, then the frames waiting, in display order.
    pub(super) frames: Vec<Lowres>,
    /// Costs of coding frame `b` from `p0` and `p1` (display indices), as computed.
    cache: Vec<((u32, u32, u32), u64)>,
}

impl Lookahead {
    fn pos(&self, disp: u32) -> usize {
        self.frames
            .iter()
            .position(|f| f.disp == disp)
            .expect("frame in the lookahead")
    }

    /// Makes sure frame `b` has been searched in frame `r`.
    fn ensure_motion(&mut self, b: u32, r: u32) {
        let (ib, ir) = (self.pos(b), self.pos(r));
        if self.frames[ib].motion(r).is_some() {
            return;
        }
        let (mvs, costs) = search(&self.frames[ib], &self.frames[ir]);
        self.frames[ib].me.push((r, mvs, costs));
    }

    /// Estimated cost of coding frame `b` predicted from `p0` and, for a B frame (`b < p1`),
    /// `p1`; P frames have `b == p1` (`x264` `slicetype_frame_cost`). Border blocks are left
    /// out, and B frames are weighted by 100 / 120.
    pub(super) fn frame_cost(&mut self, p0: u32, p1: u32, b: u32) -> u64 {
        if let Some(&(_, c)) = self.cache.iter().find(|e| e.0 == (p0, p1, b)) {
            return c;
        }
        let ib = self.pos(b);
        if self.frames[ib].intra.is_empty() {
            self.frames[ib].intra = intra_costs(&self.frames[ib]);
        }
        self.ensure_motion(b, p0);
        let bidir = b != p1;
        if bidir {
            self.ensure_motion(b, p1);
        }
        let (ib, i0, i1) = (self.pos(b), self.pos(p0), self.pos(p1));
        let f = &self.frames[ib];
        let (bw, bh) = (f.plane.w / 8, f.plane.h / 8);
        let border = bw > 2 && bh > 2;
        let m0 = f.me.iter().find(|e| e.0 == p0).unwrap();
        let m1 = f.me.iter().find(|e| e.0 == p1).unwrap_or(m0);
        let mut total = 0u64;
        for by in 0..bh {
            for bx in 0..bw {
                if border && (bx == 0 || by == 0 || bx + 1 == bw || by + 1 == bh) {
                    continue;
                }
                let i = by * bw + bx;
                let mut cost = m0.2[i];
                if bidir {
                    cost = cost.min(m1.2[i]);
                    // Bi-prediction from the two vectors.
                    let (r0, r1) = (&self.frames[i0].plane, &self.frames[i1].plane);
                    let (v0, v1) = (m0.1[i], m1.1[i]);
                    let o0 = r0.at(
                        (bx * 8) as isize + v0[0] as isize,
                        (by * 8) as isize + v0[1] as isize,
                    );
                    let o1 = r1.at(
                        (bx * 8) as isize + v1[0] as isize,
                        (by * 8) as isize + v1[1] as isize,
                    );
                    let so = f.plane.idx(bx * 8, by * 8);
                    let bi = pixel_satd_wxh::<8, 8>(
                        &f.plane.buf[so..],
                        f.plane.stride,
                        &r0.buf[o0..],
                        r0.stride,
                        Some((&r1.buf[o1..], r1.stride)),
                    ) + 5 * LA_LAMBDA;
                    cost = cost.min(bi);
                } else {
                    cost = cost.min(f.intra[i]);
                }
                total += (cost + 4) as u64;
            }
        }
        if bidir {
            total = total * 100 / B_COST_SCALE;
        }
        self.cache.push(((p0, p1, b), total));
        total
    }

    /// Cost of the frames after anchor `a` coded as `path` (`'B'` or `'P'` each), up to its last
    /// anchor (`x264` `slicetype_path_cost`).
    fn path_cost(&mut self, a: u32, path: &[u8]) -> u64 {
        let mut cost = 0;
        let mut cur = a;
        let mut loc = 0;
        while loc < path.len() {
            let mut next = loc;
            while next < path.len() && path[next] == b'B' {
                next += 1;
            }
            if next == path.len() {
                break;
            }
            let p1 = a + 1 + next as u32;
            cost += self.frame_cost(cur, p1, p1);
            for k in loc..next {
                cost += self.frame_cost(cur, p1, a + 1 + k as u32);
            }
            loc = next + 1;
            cur = p1;
        }
        cost
    }

    /// Number of frames of the next group after the anchor `self.frames[0]`: B frames then
    /// their anchor, at most `max_group`, using the `n` frames that follow the anchor. Each
    /// frame becomes a B frame while coding it as one, with the next frame as anchor, costs less
    /// than coding both as P frames, each choice costed over the whole group so far (`x264`
    /// `slicetype_path_cost`). `None` when more frames are needed.
    pub(super) fn decide(&mut self, max_group: usize, n: usize, flush: bool) -> Option<usize> {
        let a = self.frames[0].disp;
        let mut path = Vec::with_capacity(max_group + 1);
        for j in 0..max_group.saturating_sub(1) {
            if j + 1 >= n {
                return flush.then_some(n.min(max_group));
            }
            path.truncate(j);
            path.extend_from_slice(b"PP");
            let cost_p = self.path_cost(a, &path);
            path.truncate(j);
            path.extend_from_slice(b"BP");
            let cost_b = self.path_cost(a, &path);
            if cost_b >= cost_p {
                return Some(j + 1);
            }
            path.truncate(j);
            path.push(b'B');
        }
        Some(max_group.min(n))
    }

    /// Drops the frames before `disp` and the costs involving them.
    pub(super) fn retire_before(&mut self, disp: u32) {
        self.frames.retain(|f| f.disp >= disp);
        self.cache.retain(|e| e.0.0 >= disp);
        for f in &mut self.frames {
            f.me.retain(|e| e.0 >= disp);
        }
    }
}
