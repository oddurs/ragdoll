//! Motion search of the [`H264Speed::Compact`] preset (`SIMD-0089`): half-sample planes of the
//! reference pictures and a partition search over them.

use super::*;
use ragdoll_simd::{
    hpel_planes, pixel_avg_wxh, pixel_sad_avg_wxh, pixel_sad_wxh, pixel_sad_x4_wxh, pixel_satd_wxh,
};

/// The three half-sample planes of a reference picture's luma, laid out like the picture
/// itself: `h` holds `b` at `(x + 1/2, y)`, `v` holds `h` at `(x, y + 1/2)` and `c` holds `j` at
/// `(x + 1/2, y + 1/2)` (ITU-T H.264 8.4.2.2.1; `x264` `filtered`).
#[derive(Clone, Debug, Default)]
pub(super) struct HpelPlanes {
    pub(super) h: Plane,
    pub(super) v: Plane,
    pub(super) c: Plane,
}

impl HpelPlanes {
    /// Interpolates the planes of `f`, whose padding is already filled: over the region whose
    /// filter taps stay inside the buffer, less a margin inside the replicated padding, then
    /// replicated outward like a picture's padding.
    pub(super) fn build(&mut self, f: &Plane) {
        if self.h.w != f.w || self.h.h != f.h || self.h.pad != f.pad {
            *self = Self {
                h: Plane::alloc(f.w, f.h, f.pad),
                v: Plane::alloc(f.w, f.h, f.pad),
                c: Plane::alloc(f.w, f.h, f.pad),
            };
        }
        let pad = f.pad as isize;
        let (w, h) = (f.w as isize, f.h as isize);
        let (x_lo, y_lo) = (-pad + 2, -pad + 2);
        let (x_hi, y_hi) = (w + pad - 11, h + pad - 3);
        hpel_planes(
            &f.buf,
            &mut self.h.buf,
            &mut self.v.buf,
            &mut self.c.buf,
            f.stride,
            f.at(x_lo, y_lo),
            (x_hi - x_lo) as usize,
            (y_hi - y_lo) as usize,
        );
        for p in [&mut self.h, &mut self.v, &mut self.c] {
            p.extend_from(x_lo, x_hi, y_lo, y_hi);
        }
    }
}

impl Plane {
    /// Replicates the samples of the region `[x0, x1) x [y0, y1)` (picture coordinates, possibly
    /// in the padding) outward to the buffer's edges.
    fn extend_from(&mut self, x0: isize, x1: isize, y0: isize, y1: isize) {
        let s = self.stride;
        let (bx0, bx1) = (self.at(x0, 0) % s, self.at(x1 - 1, 0) % s);
        let rows = self.buf.len() / s;
        let (by0, by1) = (self.at(0, y0) / s, self.at(0, y1 - 1) / s);
        for y in by0..=by1 {
            let row = &mut self.buf[y * s..(y + 1) * s];
            let (l, r) = (row[bx0], row[bx1]);
            row[..bx0].fill(l);
            row[bx1 + 1..].fill(r);
        }
        for y in 0..by0 {
            self.buf.copy_within(by0 * s..(by0 + 1) * s, y * s);
        }
        for y in by1 + 1..rows {
            self.buf.copy_within(by1 * s..(by1 + 1) * s, y * s);
        }
    }
}

/// A reference picture's luma with its half-sample planes.
#[derive(Clone, Copy)]
pub(super) struct RefLuma<'a> {
    pub(super) f: &'a Plane,
    pub(super) hp: &'a HpelPlanes,
}

impl<'a> RefLuma<'a> {
    /// The block, or the pair of blocks whose rounded average is the prediction, of the block at
    /// `(x0, y0)` displaced by quarter-pel `mv`, as buffer offsets into the planes returned; all
    /// planes share one stride.
    #[inline(always)]
    pub(super) fn src(
        &self,
        x0: usize,
        y0: usize,
        mv: [i32; 2],
    ) -> (usize, Option<usize>, [&'a [u8]; 2]) {
        let (ix, iy) = (
            x0 as isize + (mv[0] >> 2) as isize,
            y0 as isize + (mv[1] >> 2) as isize,
        );
        let f = self.f;
        let o = f.at(ix, iy);
        let s = f.stride;
        let (fb, hb, vb, cb) = (
            &f.buf[..],
            &self.hp.h.buf[..],
            &self.hp.v.buf[..],
            &self.hp.c.buf[..],
        );
        match (mv[0] & 3, mv[1] & 3) {
            (0, 0) => (o, None, [fb, fb]),
            (2, 0) => (o, None, [hb, hb]),
            (0, 2) => (o, None, [vb, vb]),
            (2, 2) => (o, None, [cb, cb]),
            (1, 0) => (o, Some(o), [fb, hb]),
            (3, 0) => (o, Some(o + 1), [hb, fb]),
            (0, 1) => (o, Some(o), [fb, vb]),
            (0, 3) => (o, Some(o + s), [vb, fb]),
            (1, 1) => (o, Some(o), [hb, vb]),
            (3, 1) => (o, Some(o + 1), [hb, vb]),
            (1, 3) => (o, Some(o + s), [vb, hb]),
            (3, 3) => (o + 1, Some(o + s), [vb, hb]),
            (2, 1) => (o, Some(o), [hb, cb]),
            (2, 3) => (o, Some(o + s), [cb, hb]),
            (1, 2) => (o, Some(o), [vb, cb]),
            _ => (o, Some(o + 1), [cb, vb]),
        }
    }

    /// The `w`x`h` prediction of the block at `(x0, y0)` displaced by `mv` into `dst` (stride
    /// `ds`).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pred(
        &self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        mv: [i32; 2],
        dst: &mut [u8],
        ds: usize,
    ) {
        let s = self.f.stride;
        match self.src(x0, y0, mv) {
            (a, None, [pa, _]) => {
                for r in 0..h {
                    dst[r * ds..r * ds + w].copy_from_slice(&pa[a + r * s..a + r * s + w]);
                }
            }
            (a, Some(b), [pa, pb]) => block_avg(dst, ds, &pa[a..], s, &pb[b..], s, w, h),
        }
    }

    /// SAD (or SATD with `satd`) of the `w`x`h` source block `src` (stride 16) against the
    /// prediction of the block at `(x0, y0)` displaced by `mv`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cost(
        &self,
        src: &[u8],
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        mv: [i32; 2],
        satd: bool,
    ) -> u32 {
        let s = self.f.stride;
        match self.src(x0, y0, mv) {
            (a, None, [pa, _]) => block_cost(src, 16, &pa[a..], s, w, h, satd),
            (a, Some(b), [pa, pb]) => {
                let (pa, pb) = (&pa[a..], &pb[b..]);
                if satd {
                    block_satd_avg(src, 16, pa, s, Some((pb, s)), w, h)
                } else {
                    block_sad_avg(src, 16, pa, s, pb, s, w, h)
                }
            }
        }
    }
}

/// Calls `$f::<W, H>($args)` for the partition size `($w, $h)` (8 or 16 each).
macro_rules! by_size {
    ($w:expr, $h:expr, $f:ident($($a:expr),*)) => {
        match ($w, $h) {
            (16, 16) => $f::<16, 16>($($a),*),
            (16, 8) => $f::<16, 8>($($a),*),
            (8, 16) => $f::<8, 16>($($a),*),
            _ => $f::<8, 8>($($a),*),
        }
    };
}

/// SAD or SATD of two `w`x`h` blocks, `w` and `h` 8 or 16.
#[inline(always)]
pub(super) fn block_cost(
    a: &[u8],
    sa: usize,
    b: &[u8],
    sb: usize,
    w: usize,
    h: usize,
    satd: bool,
) -> u32 {
    if satd {
        block_satd_avg(a, sa, b, sb, None, w, h)
    } else {
        by_size!(w, h, pixel_sad_wxh(a, sa, b, sb))
    }
}

/// SATD of the `w`x`h` block `a` against `p`, or the average of `p` and `q`.
#[inline(always)]
pub(super) fn block_satd_avg(
    a: &[u8],
    sa: usize,
    p: &[u8],
    sp: usize,
    q: Option<(&[u8], usize)>,
    w: usize,
    h: usize,
) -> u32 {
    by_size!(w, h, pixel_satd_wxh(a, sa, p, sp, q))
}

/// SAD of the `w`x`h` block `a` against the average of `p` and `q`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn block_sad_avg(
    a: &[u8],
    sa: usize,
    p: &[u8],
    sp: usize,
    q: &[u8],
    sq: usize,
    w: usize,
    h: usize,
) -> u32 {
    by_size!(w, h, pixel_sad_avg_wxh(a, sa, p, sp, q, sq))
}

/// The average of the `w`x`h` blocks `p` and `q` into `dst`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn block_avg(
    dst: &mut [u8],
    ds: usize,
    p: &[u8],
    sp: usize,
    q: &[u8],
    sq: usize,
    w: usize,
    h: usize,
) {
    by_size!(w, h, pixel_avg_wxh(dst, ds, p, sp, q, sq))
}

/// SATD of two $8\times8$ blocks.
#[inline(always)]
pub(super) fn satd_8x8(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    pixel_satd_wxh::<8, 8>(a, sa, b, sb, None)
}

/// SA8D of a $16\times16$ block: the four $8\times8$ blocks' `x264_pixel_sa8d_8x8`, summed.
#[inline(always)]
pub(super) fn sa8d_16x16(a: &[u8], sa: usize, b: &[u8], sb: usize) -> u32 {
    ragdoll_simd::pixel_sa8d_16x16(a, sa, b, sb)
}

/// Motion search of one partition of a macroblock against one reference picture.
pub(super) struct PartSearch<'a> {
    pub(super) rl: RefLuma<'a>,
    /// The partition's source samples (stride 16).
    pub(super) src: &'a [u8],
    /// The partition's top-left luma sample in the picture.
    pub(super) x0: usize,
    pub(super) y0: usize,
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) mvp: [i16; 2],
    pub(super) lambda: u32,
    /// [`EncState::mv_costs`] of `lambda`.
    pub(super) mv_costs: &'a [u16],
    /// Bounds of the macroblock's vectors.
    pub(super) bounds: MvBounds,
}

/// Result of a [`PartSearch`]: quarter-pel vector, SATD and motion vector cost.
#[derive(Clone, Copy, Debug)]
pub(super) struct MeResult {
    pub(super) mv: [i16; 2],
    pub(super) satd: u32,
    pub(super) mv_cost: u32,
}

impl MeResult {
    #[inline(always)]
    pub(super) fn cost(&self) -> u32 {
        self.satd + self.mv_cost
    }
}

impl PartSearch<'_> {
    #[inline(always)]
    fn mvd_cost(&self, d: i32) -> u32 {
        match self.mv_costs.get((d + MV_COST_RANGE) as usize) {
            Some(&c) => c as u32,
            None => self.lambda * se_bits(d),
        }
    }

    #[inline(always)]
    pub(super) fn mv_cost(&self, mv: [i32; 2]) -> u32 {
        self.mvd_cost(mv[0] - self.mvp[0] as i32) + self.mvd_cost(mv[1] - self.mvp[1] as i32)
    }

    /// The best of `cands` (quarter-pel, rounded to full-pel) and zero, a radius-2 hexagon walked
    /// for up to `range / 2` steps and a square refinement (`x264 --me hex`), then sub-pel
    /// refinement from the half-sample planes: a half-pel diamond by SAD and quarter-pel diamonds
    /// by SATD (`x264` subme 2).
    pub(super) fn search(&self, cands: &[[i16; 2]], range: i32) -> MeResult {
        self.search_qpel(cands, range, 1)
    }

    /// [`Self::search`] with `qpel` quarter-pel diamonds.
    pub(super) fn search_qpel(&self, cands: &[[i16; 2]], range: i32, qpel: i32) -> MeResult {
        match (self.w, self.h) {
            (16, 16) => self.search_wh::<16, 16>(cands, range, qpel),
            (16, 8) => self.search_wh::<16, 8>(cands, range, qpel),
            (8, 16) => self.search_wh::<8, 16>(cands, range, qpel),
            _ => self.search_wh::<8, 8>(cands, range, qpel),
        }
    }

    #[inline(always)]
    fn search_wh<const W: usize, const H: usize>(
        &self,
        cands: &[[i16; 2]],
        range: i32,
        qpel: i32,
    ) -> MeResult {
        let int_min = [(self.bounds.min[0] + 3) >> 2, (self.bounds.min[1] + 3) >> 2];
        let int_max = [self.bounds.max[0] >> 2, self.bounds.max[1] >> 2];
        let inside = |m: [i32; 2]| {
            (int_min[0]..=int_max[0]).contains(&m[0]) && (int_min[1]..=int_max[1]).contains(&m[1])
        };
        let f = self.rl.f;
        let (buf, fs) = (&f.buf[..], f.stride as isize);
        // Buffer offset of the block displaced by full-pel `m`.
        let base = f.at(self.x0 as isize, self.y0 as isize) as isize;
        let blk = |m: [i32; 2]| &buf[(base + m[1] as isize * fs + m[0] as isize) as usize..];
        let mut best = [0i32; 2];
        let mut best_cost = u32::MAX;
        // Points are measured four at a time and taken in order, the first of equal costs
        // winning as a sequential search would pick it.
        let take = |pts: &[[i32; 2]], best: &mut [i32; 2], best_cost: &mut u32| {
            let mut moved = None;
            for (chunk_at, chunk) in pts.chunks(4).enumerate() {
                let last = chunk[chunk.len() - 1];
                let at = |k: usize| blk(*chunk.get(k).unwrap_or(&last));
                let sads = pixel_sad_x4_wxh::<W, H>(
                    self.src,
                    16,
                    [at(0), at(1), at(2), at(3)],
                    fs as usize,
                );
                for (k, m) in chunk.iter().enumerate() {
                    let c = sads[k] + self.mv_cost([m[0] * 4, m[1] * 4]);
                    if c < *best_cost {
                        *best_cost = c;
                        *best = *m;
                        moved = Some(chunk_at * 4 + k);
                    }
                }
            }
            moved
        };
        let mut pts = [[0i32; 2]; 12];
        let mut n = 1;
        for c in cands {
            let m = [
                ((c[0] as i32 + 2) >> 2).clamp(int_min[0], int_max[0]),
                ((c[1] as i32 + 2) >> 2).clamp(int_min[1], int_max[1]),
            ];
            if n < pts.len() && !pts[..n].contains(&m) {
                pts[n] = m;
                n += 1;
            }
        }
        take(&pts[..n], &mut best, &mut best_cost);
        // Hexagon of radius 2 in circular order: after a step toward point `d` only the three
        // points around it are new (`x264` `hex2`).
        const HEX: [[i32; 2]; 6] = [[-2, 0], [-1, 2], [1, 2], [2, 0], [1, -2], [-1, -2]];
        let mut dir: Option<usize> = None;
        for _ in 0..range / 2 {
            let from = best;
            let mut idx = [0usize; 6];
            let mut n = 0;
            for (k, h) in HEX.iter().enumerate() {
                if let Some(d) = dir
                    && (k + 6 - d) % 6 > 1
                    && (d + 6 - k) % 6 > 1
                {
                    continue;
                }
                let m = [from[0] + h[0], from[1] + h[1]];
                if inside(m) {
                    pts[n] = m;
                    idx[n] = k;
                    n += 1;
                }
            }
            match take(&pts[..n], &mut best, &mut best_cost) {
                Some(i) => dir = Some(idx[i]),
                None => break,
            }
        }
        let from = best;
        let mut n = 0;
        for d in [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ] {
            let m = [from[0] + d[0], from[1] + d[1]];
            if inside(m) {
                pts[n] = m;
                n += 1;
            }
        }
        take(&pts[..n], &mut best, &mut best_cost);
        self.refine_wh::<W, H>([best[0] * 4, best[1] * 4], best_cost, qpel)
    }

    /// Sub-pel refinement around full-pel `start` (quarter-pel units) whose SAD cost is
    /// `start_cost`: the predictor at its sub-pel position and a half-pel diamond by SAD (`x264`
    /// `refine_subpel` at subme 2), then a quarter-pel diamond by SATD, which `x264` leaves to
    /// the partitions it picks but which pays here on every candidate partition.
    #[inline(always)]
    fn refine_wh<const W: usize, const H: usize>(
        &self,
        start: [i32; 2],
        start_cost: u32,
        qpel: i32,
    ) -> MeResult {
        let (x0, y0) = (self.x0, self.y0);
        let mut best = start;
        let mut best_cost = start_cost;
        // The predictor at its own sub-pel position may beat the full-pel winner (`x264`
        // subme below 3).
        let mvp = [self.mvp[0] as i32, self.mvp[1] as i32];
        if mvp != best && self.bounds.clamp(mvp) == mvp {
            let cost = self.rl.cost(self.src, x0, y0, W, H, mvp, false) + self.mv_cost(mvp);
            if cost < best_cost {
                best_cost = cost;
                best = mvp;
            }
        }
        const DIRS: [[i32; 2]; 4] = [[0, -2], [-2, 0], [2, 0], [0, 2]];
        {
            let from = best;
            let inb = DIRS.map(|d| {
                let mv = [from[0] + d[0], from[1] + d[1]];
                self.bounds.clamp(mv) == mv
            });
            if from[0] & 3 == 0 && from[1] & 3 == 0 && inb.iter().all(|&b| b) {
                // From a full-pel vector the four half-pel neighbours are blocks of the `v`
                // (above, below) and `h` (left, right) planes: measured in one pass.
                let (hp, s) = (self.rl.hp, self.rl.f.stride);
                let o = self.rl.f.at(
                    x0 as isize + (from[0] >> 2) as isize,
                    y0 as isize + (from[1] >> 2) as isize,
                );
                let (h, v) = (&hp.h.buf[..], &hp.v.buf[..]);
                let refs = [&v[o - s..], &h[o - 1..], &h[o..], &v[o..]];
                let sads = pixel_sad_x4_wxh::<W, H>(self.src, 16, refs, s);
                for (k, d) in DIRS.iter().enumerate() {
                    let mv = [from[0] + d[0], from[1] + d[1]];
                    let cost = sads[k] + self.mv_cost(mv);
                    if cost < best_cost {
                        best_cost = cost;
                        best = mv;
                    }
                }
            } else {
                for (k, d) in DIRS.iter().enumerate() {
                    if !inb[k] {
                        continue;
                    }
                    let mv = [from[0] + d[0], from[1] + d[1]];
                    let cost = self.rl.cost(self.src, x0, y0, W, H, mv, false) + self.mv_cost(mv);
                    if cost < best_cost {
                        best_cost = cost;
                        best = mv;
                    }
                }
            }
        }
        let best_satd = self.rl.cost(self.src, x0, y0, W, H, best, true);
        self.refine_qpel_wh::<W, H>(
            MeResult {
                mv: [best[0] as i16, best[1] as i16],
                satd: best_satd,
                mv_cost: self.mv_cost(best),
            },
            qpel,
        )
    }

    /// Quarter-pel diamonds by SATD around `m`, at most `iters` (`x264_me_refine_qpel`).
    pub(super) fn refine_qpel(&self, m: MeResult, iters: i32) -> MeResult {
        match (self.w, self.h) {
            (16, 16) => self.refine_qpel_wh::<16, 16>(m, iters),
            (16, 8) => self.refine_qpel_wh::<16, 8>(m, iters),
            (8, 16) => self.refine_qpel_wh::<8, 16>(m, iters),
            _ => self.refine_qpel_wh::<8, 8>(m, iters),
        }
    }

    #[inline(always)]
    fn refine_qpel_wh<const W: usize, const H: usize>(&self, m: MeResult, iters: i32) -> MeResult {
        let (x0, y0, s) = (self.x0, self.y0, self.rl.f.stride);
        let mut best = [m.mv[0] as i32, m.mv[1] as i32];
        let mut best_satd = m.satd;
        let mut best_cost = m.satd + self.mv_cost(best);
        // The centre of the previous diamond, whose cost is known not to win.
        let mut prev = None;
        for _ in 0..iters {
            let from = best;
            for d in [[0, -1], [0, 1], [-1, 0], [1, 0]] {
                let mv = [from[0] + d[0], from[1] + d[1]];
                if Some(mv) == prev || self.bounds.clamp(mv) != mv {
                    continue;
                }
                let satd = match self.rl.src(x0, y0, mv) {
                    (a, None, [pa, _]) => pixel_satd_wxh::<W, H>(self.src, 16, &pa[a..], s, None),
                    (a, Some(b), [pa, pb]) => {
                        pixel_satd_wxh::<W, H>(self.src, 16, &pa[a..], s, Some((&pb[b..], s)))
                    }
                };
                let cost = satd + self.mv_cost(mv);
                if cost < best_cost {
                    best_cost = cost;
                    best_satd = satd;
                    best = mv;
                }
            }
            if best == from {
                break;
            }
            prev = Some(from);
        }
        MeResult {
            mv: [best[0] as i16, best[1] as i16],
            satd: best_satd,
            mv_cost: self.mv_cost(best),
        }
    }
}

/// Full-pel search range of the Compact preset (`x264 --merange 16`).
pub(super) const ME_RANGE: i32 = 16;
/// Quarter-pel diamonds on the vectors of the partitions chosen.
pub(super) const QPEL_REFINE: i32 = 2;
