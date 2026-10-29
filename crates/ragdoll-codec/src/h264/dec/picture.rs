//! Decoded frame stores and the per-picture macroblock state used while decoding them.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use ragdoll_core::{AlignedBuffer, AlignedVec};

use super::deblock::{self, SliceDeblock};
use super::tables::Part;

/// Macroblock prediction type after parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbType {
    /// Not yet decoded in this picture.
    None,
    I4x4,
    I8x8,
    /// `I_16x16` with `Intra16x16PredMode`.
    I16x16(u8),
    IPcm,
    PSkip,
    BSkip,
    BDirect16x16,
    /// P or B inter macroblock with a partition shape (B_8x8 and P_8x8 use `Part::P8x8`).
    Inter(Part),
}

impl MbType {
    pub fn is_intra(self) -> bool {
        matches!(self, Self::I4x4 | Self::I8x8 | Self::I16x16(_) | Self::IPcm)
    }
    pub fn is_skip(self) -> bool {
        matches!(self, Self::PSkip | Self::BSkip)
    }
}

/// Coding structure of the picture being decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Frame,
    /// A field picture of the given parity (0 top, 1 bottom).
    Field(u8),
}

/// Motion a frame store keeps for direct prediction of later B pictures, in frame layout:
/// macroblock rows of field pictures are interleaved (field row `r` of parity `p` is frame row `2r + p`).
#[derive(Clone, Debug)]
pub struct Motion {
    /// Per 4x4 block.
    pub blocks: AlignedVec<BlockInfo>,
    /// Per macroblock.
    pub mb_intra: Vec<bool>,
    /// Per macroblock: one motion for all 16 blocks (see [`PicState::motion_edges`]).
    pub mb_uniform: Vec<bool>,
    /// Per macroblock: decoded as a field macroblock (field picture or field MB pair).
    pub mb_field: Vec<bool>,
    /// Decoded as an MBAFF frame (rather than as a frame or two field pictures).
    pub mbaff: bool,
    /// The referenced pictures' ids by list and reference index, per [`Motion::mb_table`] entry.
    pub ref_tables: Vec<RefIds>,
    /// Per macroblock: its entry of [`Motion::ref_tables`].
    pub mb_table: Vec<u16>,
}

impl Motion {
    pub fn new(mb_w: usize, mb_h: usize) -> Self {
        let b4 = mb_w * mb_h * 16;
        Self {
            blocks: AlignedVec::from_elem(NO_BLOCK, b4),
            mb_intra: vec![true; mb_w * mb_h],
            mb_uniform: vec![false; mb_w * mb_h],
            mb_field: vec![false; mb_w * mb_h],
            mbaff: false,
            ref_tables: Vec::new(),
            mb_table: vec![0; mb_w * mb_h],
        }
    }
}

/// Ids of the pictures a slice's reference indices refer to, by list and index (up to 32 each,
/// as field reference lists allow); [`NO_PIC`] beyond the lists.
pub type RefIds = [[u32; 32]; 2];

/// [`Motion::ref_pic`] over the parts of a picture's state.
#[inline]
fn ref_pic_in(
    tables: &[RefIds],
    mb_table: &[u16],
    blocks: &[BlockInfo],
    mb: usize,
    blk: usize,
    list: usize,
) -> u32 {
    let r = blocks[blk].ref_idx[list];
    if r < 0 {
        return NO_PIC;
    }
    tables
        .get(mb_table[mb] as usize)
        .map_or(NO_PIC, |t| t[list][r as usize & 31])
}

/// What later blocks read of one 4x4 block, in 16 bytes. Per list: the vector, the reference
/// index (`-1` unused; the referenced picture follows from the macroblock's reference table, see
/// [`PicState::ref_pic`]) and, for CABAC contexts, the absolute mvd components. Then whether its
/// transform block has nonzero coefficients (deblocking, CABAC cbf) and whether it was
/// direct-predicted (CABAC ref_idx).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockInfo {
    pub mv: [[i16; 2]; 2],
    pub ref_idx: [i8; 2],
    pub mvd: [[u8; 2]; 2],
    pub nz: bool,
    pub direct: bool,
}

/// [`BlockInfo`] of a block with no prediction (intra, or not yet decoded).
pub const NO_BLOCK: BlockInfo = BlockInfo {
    mv: [[0; 2]; 2],
    ref_idx: [-1; 2],
    mvd: [[0; 2]; 2],
    nz: false,
    direct: false,
};

/// A macroblock row of [`NO_BLOCK`], copied from memory rather than rebuilt for every row.
static NO_BLOCK_ROW: [BlockInfo; 4] = [NO_BLOCK; 4];

/// Sets the `W`x`h` rectangle of `v` whose rows start at `i`, `i + stride`, ... to `val`, with
/// one bounds check (`h >= 1`).
#[inline(always)]
pub fn fill_rect<T: Copy, const W: usize>(v: &mut [T], i: usize, stride: usize, h: usize, val: T) {
    let s = &mut v[i..i + (h - 1) * stride + W];
    if size_of::<T>() <= 8 {
        for r in 0..h {
            s[r * stride..r * stride + W].copy_from_slice(&[val; W]);
        }
        return;
    }
    // Later rows copy the first: a multi-field `T` is then built once, not once per row.
    let (first, rest) = s.split_at_mut(W);
    first.copy_from_slice(&[val; W]);
    for r in 1..h {
        rest[r * stride - W..r * stride].copy_from_slice(first);
    }
}

/// A 64-byte-aligned picture plane of `n` samples set to `v`.
pub fn plane_buf(n: usize, v: u8) -> AlignedBuffer {
    let mut b = AlignedBuffer::new_zeroed(n);
    if v != 0 {
        b.fill(v);
    }
    b
}

/// The three sample planes of the picture being decoded.
///
/// A frame picture decoded on a worker thread shares them, through the frame store, with the
/// pictures that predict from it while it is still being decoded. The state holding them is
/// their only writer, and readers on other threads read only rows published as final through
/// the picture's [`PicSync`], which the writer no longer touches. Otherwise (between pictures,
/// and when frames handed out still read them) they are made unique before being written; see
/// [`PicPlanes::make_unique`].
pub struct PicPlanes(Arc<[AlignedBuffer; 3]>);

impl PicPlanes {
    fn new(luma: usize) -> Self {
        Self(Arc::new([
            plane_buf(luma, 0),
            plane_buf(luma / 4, 128),
            plane_buf(luma / 4, 128),
        ]))
    }

    /// No samples: a placeholder for planes moved out.
    pub fn empty() -> Self {
        Self::new(0)
    }

    /// The buffers, for the frame store and the pictures that refer to it.
    pub fn shared(&self) -> Arc<[AlignedBuffer; 3]> {
        self.0.clone()
    }

    /// Replaces buffers still shared (with frames handed out or reference pictures) by a set of
    /// the same size from `pool` that no longer is, or by new buffers, so that this state can
    /// write them. The shared set joins the pool, to be taken once it is free.
    pub fn make_unique(&mut self, pool: &mut Vec<Arc<[AlignedBuffer; 3]>>) {
        // The pool's own handle does not count: it never hands out a set held elsewhere.
        let pooled = pool.iter().any(|p| Arc::ptr_eq(p, &self.0));
        if Arc::strong_count(&self.0) == 1 + pooled as usize {
            return;
        }
        let (new, _) = Self::take(pool, self.0[0].len());
        pool_add(pool, std::mem::replace(self, new).0);
    }

    /// A set of planes of `len` luma samples nothing else uses: a free one from `pool`, or a new
    /// one (then `true`). It joins the pool, to be reused once its picture is no longer read.
    pub fn take(pool: &mut Vec<Arc<[AlignedBuffer; 3]>>, len: usize) -> (Self, bool) {
        let free = pool
            .iter()
            .position(|p| Arc::strong_count(p) == 1 && p[0].len() == len);
        let (set, fresh) = match free {
            Some(i) => (Self(pool.swap_remove(i)), false),
            None => (Self::new(len), true),
        };
        pool_add(pool, set.shared());
        (set, fresh)
    }

    /// The writable Cb and Cr planes.
    pub fn chroma_mut(&mut self) -> [&mut [u8]; 2] {
        // Safety: as for `index_mut`; the two planes are distinct buffers.
        [1, 2].map(|c| {
            let (p, n) = self.raw(c);
            unsafe { std::slice::from_raw_parts_mut(p, n) }
        })
    }

    /// Plane `c`'s samples, through the pointer of the buffer's allocation (not one derived
    /// from a shared reference to the samples).
    fn raw(&self, c: usize) -> (*mut u8, usize) {
        let b = &self.0[c];
        (b.as_ptr().cast_mut(), b.len())
    }
}

impl std::ops::Index<usize> for PicPlanes {
    type Output = [u8];
    fn index(&self, c: usize) -> &[u8] {
        &self.0[c]
    }
}

impl std::ops::IndexMut<usize> for PicPlanes {
    fn index_mut(&mut self, c: usize) -> &mut [u8] {
        let (p, n) = self.raw(c);
        // Safety: the state holding these planes is their only writer (they are unique, or shared
        // only with readers of published rows, which it no longer writes), and `&mut self`
        // excludes its other borrows.
        unsafe { std::slice::from_raw_parts_mut(p, n) }
    }
}

/// Plane sets [`PicPlanes::make_unique`] keeps for reuse.
const PLANE_POOL: usize = 16;

fn pool_add(pool: &mut Vec<Arc<[AlignedBuffer; 3]>>, set: Arc<[AlignedBuffer; 3]>) {
    pool.push(set);
    // Keep the sets least likely to be freed soon out of the pool.
    if pool.len() > PLANE_POOL {
        pool.remove(0);
    }
}

/// Decoding progress of a picture decoded on a worker thread, which later pictures wait on.
#[derive(Debug, Default)]
pub struct PicSync {
    /// Luma lines (from the top) whose samples are final; [`PicSync::DONE`] once the picture
    /// is complete. Chroma line `c` is final with luma line `2 * c + 1`.
    lines: AtomicUsize,
    /// Macroblock rows whose motion is final in [`PicSync::live`].
    mv_rows: AtomicUsize,
    /// For a reference picture: its motion, filled row by row as it is decoded.
    live: OnceLock<LiveMotion>,
    /// The picture's motion, set when it is complete.
    motion: OnceLock<Arc<Motion>>,
    /// Threads sleeping on `cv`: progress wakes them only when there are any.
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

impl PicSync {
    pub const DONE: usize = usize::MAX;

    fn bump(&self, counter: &AtomicUsize, to: usize) {
        if to > counter.load(Ordering::Relaxed) {
            // SeqCst with the sleeper count: a sleeper either sees the new value or is seen.
            counter.store(to, Ordering::SeqCst);
            if self.sleepers.load(Ordering::SeqCst) > 0 {
                // Taking the lock orders the wake after a sleeper's last check.
                drop(self.lock.lock().unwrap_or_else(|e| e.into_inner()));
                self.cv.notify_all();
            }
        }
    }

    fn wait_for(&self, counter: &AtomicUsize, above: usize) {
        if counter.load(Ordering::Acquire) > above {
            return;
        }
        let mut g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        while counter.load(Ordering::SeqCst) <= above {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        self.sleepers.fetch_sub(1, Ordering::SeqCst);
    }

    /// Publishes luma lines `..lines` as final.
    pub fn publish(&self, lines: usize) {
        self.bump(&self.lines, lines);
    }

    /// Gives a reference picture the motion its worker fills (before the worker starts).
    pub fn set_live(&self, live: LiveMotion) {
        let _ = self.live.set(live);
    }

    /// The motion of a reference picture being decoded, with rows published as final by
    /// [`PicSync::publish_motion`].
    pub fn live(&self) -> Option<&LiveMotion> {
        self.live.get()
    }

    /// Publishes the motion of macroblock rows `..rows` (written to [`PicSync::live`]) as final.
    pub fn publish_motion(&self, rows: usize) {
        self.bump(&self.mv_rows, rows);
    }

    /// Waits until the motion of macroblock row `row` is final, and if it was not, for `slack`
    /// more rows (of `self`'s picture height at most).
    pub fn wait_motion(&self, row: usize, slack: usize) {
        if self.mv_rows.load(Ordering::Acquire) <= row {
            let rows = self
                .live()
                .map_or(0, |l| l.motion.mb_intra.len() / l.mb_w.max(1));
            self.wait_for(
                &self.mv_rows,
                (row + slack).min(rows.saturating_sub(1)).max(row),
            );
        }
    }

    /// Marks the picture complete: every sample and its motion.
    pub fn complete(&self) {
        // A picture completes once.
        if let Some(live) = self.live.get() {
            let _ = self.motion.set(live.motion.clone());
        }
        self.bump(&self.mv_rows, Self::DONE);
        self.publish(Self::DONE);
    }

    /// Whether luma line `line` is final, without waiting.
    #[inline]
    pub fn has(&self, line: usize) -> bool {
        self.lines.load(Ordering::Acquire) > line
    }

    /// Waits until luma line `line` is final.
    pub fn wait(&self, line: usize) {
        self.wait_for(&self.lines, line);
    }

    /// The picture's motion if it is complete.
    pub fn done_motion(&self) -> Option<Arc<Motion>> {
        self.motion.get().cloned()
    }
}

/// The motion of a reference picture decoding on a worker thread, which the worker writes row
/// by row (and nothing else writes) while direct prediction in other pictures reads the rows
/// published as final. Created from a [`Motion`] holding the picture's slices' reference tables.
#[derive(Debug)]
pub struct LiveMotion {
    motion: Arc<Motion>,
    mb_w: usize,
    blocks: *mut BlockInfo,
    mb_intra: *mut bool,
    mb_uniform: *mut bool,
    mb_table: *mut u16,
}

// Safety: the pointers address `motion`'s buffers, which live as long as it does; writes go only
// to rows not yet published, reads only to published ones (see the type's documentation).
unsafe impl Send for LiveMotion {}
unsafe impl Sync for LiveMotion {}

impl LiveMotion {
    /// `m` must hold the picture's reference tables; its frame-layout geometry is the picture's.
    pub fn new(mut m: Motion, mb_w: usize) -> Self {
        m.mbaff = false;
        m.mb_field.fill(false);
        let (blocks, mb_intra, mb_uniform, mb_table) = (
            m.blocks.as_mut_ptr(),
            m.mb_intra.as_mut_ptr(),
            m.mb_uniform.as_mut_ptr(),
            m.mb_table.as_mut_ptr(),
        );
        Self {
            motion: Arc::new(m),
            mb_w,
            blocks,
            mb_intra,
            mb_uniform,
            mb_table,
        }
    }

    /// The motion as published so far.
    pub fn view(&self) -> MotionView<'_> {
        let m = &*self.motion;
        let (n4, n) = (m.blocks.len(), m.mb_intra.len());
        // Safety: the pointers cover the buffers' lengths. Rows not yet published may change
        // while the view lives; callers read only published ones.
        unsafe {
            MotionView {
                blocks: std::slice::from_raw_parts(self.blocks, n4),
                mb_intra: std::slice::from_raw_parts(self.mb_intra, n),
                mb_uniform: std::slice::from_raw_parts(self.mb_uniform, n),
                mb_field: &m.mb_field,
                mbaff: false,
                ref_tables: &m.ref_tables,
                mb_table: std::slice::from_raw_parts(self.mb_table, n),
            }
        }
    }
}

/// Read access to a picture's stored motion (complete, or published rows of a [`LiveMotion`]).
#[derive(Clone, Copy)]
pub struct MotionView<'a> {
    pub blocks: &'a [BlockInfo],
    pub mb_intra: &'a [bool],
    pub mb_uniform: &'a [bool],
    pub mb_field: &'a [bool],
    pub mbaff: bool,
    pub ref_tables: &'a [RefIds],
    pub mb_table: &'a [u16],
}

impl<'a> MotionView<'a> {
    pub fn of(m: &'a Motion) -> Self {
        Self {
            blocks: &m.blocks,
            mb_intra: &m.mb_intra,
            mb_uniform: &m.mb_uniform,
            mb_field: &m.mb_field,
            mbaff: m.mbaff,
            ref_tables: &m.ref_tables,
            mb_table: &m.mb_table,
        }
    }

    /// Id of the picture block `blk` of macroblock `mb` refers to in `list`, or [`NO_PIC`].
    #[inline]
    pub fn ref_pic(&self, mb: usize, blk: usize, list: usize) -> u32 {
        ref_pic_in(self.ref_tables, self.mb_table, self.blocks, mb, blk, list)
    }
}

/// [`PicState::motion_edges`] of a macroblock with unknown motion: every inner edge.
pub const ANY_INNER_EDGE: u8 = 0xEE;

pub const NO_PIC: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefKind {
    Unused,
    Short,
    Long,
}

/// A decoded frame, or a field pair being assembled, held in the DPB.
#[derive(Clone, Debug)]
pub struct FrameStore {
    pub id: u32,
    pub planes: Arc<[AlignedBuffer; 3]>,
    field_planes: [OnceLock<Arc<[AlignedBuffer; 3]>>; 2],
    pub width: usize,
    pub height: usize,
    pub field_poc: [i32; 2],
    pub decoded: [bool; 2],
    pub kind: [RefKind; 2],
    pub frame_num: u32,
    pub frame_num_wrap: i32,
    pub long_term_frame_idx: u32,
    pub motion: Arc<Motion>,
    /// For a frame picture decoded on a worker thread: its progress, and its motion once
    /// complete (`motion` is then a placeholder).
    pub sync: Option<Arc<PicSync>>,
}

impl FrameStore {
    pub fn new(id: u32, width: usize, height: usize, mb_w: usize, mb_h: usize) -> Self {
        Self {
            id,
            planes: Arc::new([
                plane_buf(width * height, 0),
                plane_buf(width * height / 4, 128),
                plane_buf(width * height / 4, 128),
            ]),
            field_planes: Default::default(),
            width,
            height,
            field_poc: [0; 2],
            decoded: [false; 2],
            kind: [RefKind::Unused; 2],
            frame_num: 0,
            frame_num_wrap: 0,
            long_term_frame_idx: 0,
            motion: Arc::new(Motion::new(mb_w, mb_h)),
            sync: None,
        }
    }

    /// The store of a frame picture decoding on a worker thread into `planes`.
    pub fn decoding(
        id: u32,
        width: usize,
        height: usize,
        planes: Arc<[AlignedBuffer; 3]>,
        sync: Arc<PicSync>,
    ) -> Self {
        Self {
            id,
            planes,
            field_planes: Default::default(),
            width,
            height,
            field_poc: [0; 2],
            decoded: [false; 2],
            kind: [RefKind::Unused; 2],
            frame_num: 0,
            frame_num_wrap: 0,
            long_term_frame_idx: 0,
            motion: Arc::new(Motion::new(0, 0)),
            sync: Some(sync),
        }
    }

    /// The store's motion, `None` while its picture is still being decoded.
    pub fn current_motion(&self) -> Option<Arc<Motion>> {
        match &self.sync {
            Some(s) => s.done_motion(),
            None => Some(self.motion.clone()),
        }
    }

    /// The store's progress while its picture is still being decoded.
    pub fn pending_sync(&self) -> Option<&Arc<PicSync>> {
        self.sync.as_ref().filter(|s| !s.has(PicSync::DONE - 1))
    }

    /// POC of the frame: the smaller POC of its decoded fields.
    /// Makes a recycled store a new, empty one with id `id`. With `init`, samples and motion are
    /// reset too (a field picture fills only half of them); a frame picture replaces both.
    pub fn reuse(&mut self, id: u32, init: bool) {
        self.id = id;
        if self.sync.take().is_some() {
            // A worker-decoded store kept a placeholder: start from fresh motion.
            let n = self.width / 16 * (self.height / 16);
            if self.motion.mb_intra.len() != n {
                self.motion = Arc::new(Motion::new(self.width / 16, self.height / 16));
            }
        }
        self.field_planes = Default::default();
        self.field_poc = [0; 2];
        self.decoded = [false; 2];
        self.kind = [RefKind::Unused; 2];
        self.frame_num = 0;
        self.frame_num_wrap = 0;
        self.long_term_frame_idx = 0;
        if init {
            if let Some(planes) = Arc::get_mut(&mut self.planes) {
                planes[0].fill(0);
                planes[1].fill(128);
                planes[2].fill(128);
            } else {
                // Frames handed out still read the old samples: start from new buffers rather
                // than copying them.
                let lens = self.planes.each_ref().map(|p| p.len());
                self.planes = Arc::new(lens.map(|n| plane_buf(n, 0)));
                let planes = Arc::get_mut(&mut self.planes).expect("new planes");
                planes[1].fill(128);
                planes[2].fill(128);
            }
            let n = self.motion.mb_intra.len();
            let m = Arc::make_mut(&mut self.motion);
            m.blocks.fill(NO_BLOCK);
            m.mb_intra.fill(true);
            m.mb_uniform.fill(false);
            m.mb_field.fill(false);
            m.ref_tables.clear();
            m.mb_table.fill(0);
            debug_assert_eq!(m.mb_intra.len(), n);
        }
        Arc::make_mut(&mut self.motion).mbaff = false;
    }

    pub fn poc(&self) -> i32 {
        match self.decoded {
            [true, true] => self.field_poc[0].min(self.field_poc[1]),
            [false, true] => self.field_poc[1],
            _ => self.field_poc[0],
        }
    }

    pub fn frame_ref_id(&self) -> u32 {
        self.id * 4
    }

    pub fn field_ref_id(&self, parity: u8) -> u32 {
        self.id * 4 + 1 + parity as u32
    }

    pub fn any(&self, k: RefKind) -> bool {
        self.kind[0] == k || self.kind[1] == k
    }

    pub fn is_reference(&self) -> bool {
        self.any(RefKind::Short) || self.any(RefKind::Long)
    }

    /// The samples of one field as a half-height picture.
    pub fn field_view(&self, parity: u8) -> Arc<[AlignedBuffer; 3]> {
        self.field_planes[parity as usize]
            .get_or_init(|| {
                let p = parity as usize;
                let extract = |plane: &[u8], w: usize, h: usize| -> AlignedBuffer {
                    AlignedBuffer::from_rows(w, h / 2, |y| &plane[(2 * y + p) * w..][..w])
                };
                let (w, h) = (self.width, self.height);
                Arc::new([
                    extract(&self.planes[0], w, h),
                    extract(&self.planes[1], w / 2, h / 2),
                    extract(&self.planes[2], w / 2, h / 2),
                ])
            })
            .clone()
    }

    /// Mutable access to the frame samples; drops cached field views of `parity`.
    pub fn planes_mut(&mut self, parity: Option<u8>) -> &mut [AlignedBuffer; 3] {
        match parity {
            Some(p) => self.field_planes[p as usize] = OnceLock::new(),
            None => self.field_planes = Default::default(),
        }
        // A frame picture overwrites every sample: while handed-out frames still read the old
        // ones, it takes new buffers instead of copying them.
        if parity.is_none() && Arc::get_mut(&mut self.planes).is_none() {
            let lens = self.planes.each_ref().map(|p| p.len());
            self.planes = Arc::new(lens.map(|n| plane_buf(n, 0)));
        }
        Arc::make_mut(&mut self.planes)
    }
}

/// Per-picture decoding state, indexed by macroblock address or by 4x4 block in raster order.
/// For a field picture the geometry is that of the field.
pub struct PicState {
    pub structure: Structure,
    /// Macroblock-adaptive frame/field frame: macroblocks are decoded in vertical pairs and
    /// `mb_y = 2 * pair_row + (bottom MB)`; a field pair's MBs hold the pair's top and bottom field.
    pub mbaff: bool,
    pub mb_w: usize,
    pub mb_h: usize,
    pub width: usize,
    pub height: usize,
    pub planes: PicPlanes,

    pub slice_num: Vec<u32>,
    pub mb_type: Vec<MbType>,
    /// Field macroblock (field picture, or field pair of an MBAFF frame).
    pub mb_field: Vec<bool>,
    /// Luma CBP in bits 0..3, chroma CBP in bits 4..5.
    pub cbp: Vec<u8>,
    pub qp: Vec<u8>,
    pub qpc: Vec<[u8; 2]>,
    pub t8x8: Vec<bool>,
    pub chroma_pred_mode: Vec<u8>,
    /// CABAC DC coded_block_flags: bit 0 luma DC, bit 1 Cb DC, bit 2 Cr DC.
    pub cbf_dc: Vec<u8>,

    /// Per 4x4 luma block: total_coeff (CAVLC nC) / coefficient count.
    pub nnz: Vec<u8>,
    pub intra_mode: Vec<i8>,
    /// Per 4x4 block: motion of both lists and coefficient and direct flags.
    pub blocks: AlignedVec<BlockInfo>,
    /// Reference tables of the picture's slices (see [`Motion::ref_tables`]).
    pub ref_tables: Vec<RefIds>,
    /// Per table: no picture appears in it twice, so that within it equal pictures are equal
    /// (list, index) pairs.
    pub ref_unique: Vec<bool>,
    /// Per macroblock: its entry of [`PicState::ref_tables`].
    pub mb_table: Vec<u16>,
    /// Per macroblock: the inner luma edges (bit `4 * dir + edge`, `edge` 1..=3) across which
    /// motion may differ; `0` when inter prediction used one motion for all 16 blocks.
    pub motion_edges: Vec<u8>,
    /// Per 4x4 chroma block of each plane: total_coeff of the AC block.
    pub nnz_c: [Vec<u8>; 2],
    /// The buffers were reused from an earlier picture: each macroblock's state is restored by
    /// [`PicState::clear_mb`] when it is decoded, and [`PicState::restore_undecoded`] restores
    /// what no slice covered.
    pub recycled: bool,
    /// In-loop deblocking state.
    pub rows: RowFilter,
}

/// Deblocking that follows decoding row by row (see [`deblock::deblock_rows`]): while the
/// picture's macroblocks are decoded in raster order from the first, filtering each row once
/// the row below it is decoded, and publishing the rows that become final.
#[derive(Default)]
pub struct RowFilter {
    /// Macroblocks have been decoded in raster order from the first, without errors.
    pub on: bool,
    /// The next macroblock in raster order.
    pub next: usize,
    /// Rows filtered so far.
    pub done: usize,
    /// Rows whose motion is published (see [`PicState::publish_motion`]).
    pub mv_done: usize,
    pub slices: Vec<SliceDeblock>,
    pub has_chroma: bool,
    /// Where final rows are published, for a picture others predict from while it is decoded.
    pub sync: Option<Arc<PicSync>>,
}

impl PicState {
    /// Starts a picture's in-loop deblocking; `rows_on` when it may follow decoding (not MBAFF).
    pub fn start_rows(&mut self, has_chroma: bool, sync: Option<Arc<PicSync>>) {
        let mut slices = std::mem::take(&mut self.rows.slices);
        slices.clear();
        self.rows = RowFilter {
            on: !self.mbaff,
            next: 0,
            done: 0,
            mv_done: 0,
            slices,
            has_chroma,
            sync,
        };
    }

    /// Macroblock `addr` (at column `x`, row `y`) starts decoding. Without MBAFF, the first
    /// macroblock of a row in raster order completes the row above it, so the row above that
    /// can be filtered.
    #[inline]
    pub fn row_mb(&mut self, addr: usize, x: usize, y: usize) {
        if addr != self.rows.next {
            self.rows.on = false;
        }
        self.rows.next = addr + 1;
        if x == 0 && y >= 1 && self.rows.on {
            self.rows_done(y);
        }
    }

    /// Rows `..rows` are decoded: publishes their motion, and filters the rows above the last.
    #[cold]
    fn rows_done(&mut self, rows: usize) {
        self.publish_motion(rows);
        self.filter_rows(rows - 1);
    }

    /// Copies the motion of decoded rows `..end` to the picture's [`LiveMotion`] and publishes it.
    fn publish_motion(&mut self, end: usize) {
        let Some(sync) = self.rows.sync.clone() else {
            return;
        };
        let Some(live) = sync.live() else {
            return;
        };
        let (w, start) = (self.mb_w, self.rows.mv_done);
        if end <= start {
            return;
        }
        let (mbs, b4) = (start * w..end * w, start * w * 16..end * w * 16);
        // Safety: rows `start..end` are not yet published, so no reader reads them, and only
        // this picture's state writes its live motion; the ranges lie within the buffers, of this
        // picture's geometry.
        unsafe {
            let blocks = std::slice::from_raw_parts_mut(live.blocks.add(b4.start), b4.len());
            blocks.copy_from_slice(&self.blocks[b4]);
            for a in mbs {
                *live.mb_intra.add(a) = self.mb_type[a].is_intra();
                *live.mb_uniform.add(a) = self.motion_edges[a] == 0;
                *live.mb_table.add(a) = self.mb_table[a];
            }
        }
        self.rows.mv_done = end;
        sync.publish_motion(end);
    }

    /// Filters rows up to `end` (exclusive) and publishes the samples that became final: all
    /// but the bottom three luma lines (one chroma line) of the last row, which the next row's
    /// top edge filters.
    #[cold]
    pub fn filter_rows(&mut self, end: usize) {
        if end <= self.rows.done {
            return;
        }
        let slices = std::mem::take(&mut self.rows.slices);
        deblock::deblock_rows(self, &slices, self.rows.has_chroma, self.rows.done..end);
        self.rows.slices = slices;
        self.rows.done = end;
        if let Some(sync) = &self.rows.sync {
            sync.publish(if end == self.mb_h {
                self.height
            } else {
                16 * end - 3
            });
        }
    }

    /// Completes the picture's deblocking: rows filtered in the loop stay as they are, then
    /// undecoded macroblocks are restored and the remaining rows filtered.
    pub fn finish_rows(&mut self) {
        self.restore_undecoded();
        self.publish_motion(self.mb_h);
        if self.mbaff {
            let slices = std::mem::take(&mut self.rows.slices);
            deblock::deblock_picture(self, &slices, self.rows.has_chroma);
            self.rows.slices = slices;
        } else {
            self.filter_rows(self.mb_h);
        }
    }
}

impl PicState {
    pub fn new(structure: Structure, mb_w: usize, mb_h: usize, mbaff: bool) -> Self {
        Self::with_planes(
            structure,
            mb_w,
            mb_h,
            mbaff,
            PicPlanes::new(mb_w * mb_h * 256),
        )
    }

    fn with_planes(
        structure: Structure,
        mb_w: usize,
        mb_h: usize,
        mbaff: bool,
        planes: PicPlanes,
    ) -> Self {
        let mbs = mb_w * mb_h;
        let b4 = mbs * 16;
        let c4 = mbs * 4;
        let (w, h) = (mb_w * 16, mb_h * 16);
        Self {
            structure,
            mbaff,
            mb_w,
            mb_h,
            width: w,
            height: h,
            planes,
            slice_num: vec![u32::MAX; mbs],
            mb_type: vec![MbType::None; mbs],
            mb_field: vec![matches!(structure, Structure::Field(_)); mbs],
            cbp: vec![0; mbs],
            qp: vec![0; mbs],
            qpc: vec![[0; 2]; mbs],
            t8x8: vec![false; mbs],
            chroma_pred_mode: vec![0; mbs],
            cbf_dc: vec![0; mbs],
            nnz: vec![0; b4],
            intra_mode: vec![-1; b4],
            blocks: AlignedVec::from_elem(NO_BLOCK, b4),
            ref_tables: Vec::new(),
            ref_unique: Vec::new(),
            mb_table: vec![0; mbs],
            motion_edges: vec![ANY_INNER_EDGE; mbs],
            nnz_c: [vec![0; c4], vec![0; c4]],
            recycled: false,
            rows: RowFilter::default(),
        }
    }

    /// A state for a new picture, reusing `old`'s buffers when the geometry matches. Only the
    /// availability state (`slice_num`, `mb_type`, `mb_field`) is reset here.
    pub fn recycle(
        old: Option<Self>,
        structure: Structure,
        mb_w: usize,
        mb_h: usize,
        mbaff: bool,
        pool: &mut Vec<Arc<[AlignedBuffer; 3]>>,
    ) -> Self {
        match old {
            Some(mut st) if st.mb_w == mb_w && st.mb_h == mb_h => {
                st.reset(structure, mbaff);
                st.planes.make_unique(pool);
                st
            }
            _ => Self::new(structure, mb_w, mb_h, mbaff),
        }
    }

    /// [`Self::recycle`] for a frame picture decoding into `planes` (`fresh` when newly
    /// allocated rather than reused).
    pub fn recycle_into(
        old: Option<Self>,
        mb_w: usize,
        mb_h: usize,
        planes: PicPlanes,
        fresh: bool,
    ) -> Self {
        match old {
            Some(mut st) if st.mb_w == mb_w && st.mb_h == mb_h => {
                st.reset(Structure::Frame, false);
                st.planes = planes;
                st
            }
            _ => {
                let mut st = Self::with_planes(Structure::Frame, mb_w, mb_h, false, planes);
                st.recycled = !fresh;
                st
            }
        }
    }

    /// Resets the availability state (`slice_num`, `mb_type`, `mb_field`) for a new picture.
    fn reset(&mut self, structure: Structure, mbaff: bool) {
        self.structure = structure;
        self.mbaff = mbaff;
        self.slice_num.fill(u32::MAX);
        self.mb_type.fill(MbType::None);
        self.mb_field.fill(matches!(structure, Structure::Field(_)));
        self.ref_tables.clear();
        self.ref_unique.clear();
        self.recycled = true;
    }

    /// Id of the picture block `blk` of macroblock `mb` refers to in `list`, or [`NO_PIC`].
    #[inline]
    pub fn ref_pic(&self, mb: usize, blk: usize, list: usize) -> u32 {
        ref_pic_in(
            &self.ref_tables,
            &self.mb_table,
            &self.blocks,
            mb,
            blk,
            list,
        )
    }

    /// Adds a slice's reference table.
    pub fn push_ref_table(&mut self, t: RefIds) {
        let ids = || t.iter().flatten().filter(|&&id| id != NO_PIC);
        let unique = ids()
            .enumerate()
            .all(|(i, a)| ids().skip(i + 1).all(|b| a != b));
        self.ref_tables.push(t);
        self.ref_unique.push(unique);
    }

    /// [`Self::ref_pic`] for both lists.
    #[inline]
    pub fn ref_pics(&self, mb: usize, blk: usize) -> [u32; 2] {
        [self.ref_pic(mb, blk, 0), self.ref_pic(mb, blk, 1)]
    }

    /// Restores the coding state (all but availability) of macroblock `addr`, at column `x` and
    /// row `y`, to that of a new picture.
    pub fn clear_mb(&mut self, addr: usize, x: usize, y: usize) {
        self.cbp[addr] = 0;
        self.qp[addr] = 0;
        self.qpc[addr] = [0; 2];
        self.t8x8[addr] = false;
        self.chroma_pred_mode[addr] = 0;
        self.cbf_dc[addr] = 0;
        self.motion_edges[addr] = ANY_INNER_EDGE;
        self.clear_mb_state(x, y);
        self.reset_mb_blocks(x, y);
    }

    /// Resets the [`BlockInfo`] of the 16 4x4 blocks of the macroblock at column `x`, row `y`.
    pub fn reset_mb_blocks(&mut self, x: usize, y: usize) {
        let (i, w) = (self.b4(x * 4, y * 4), self.mb_w * 4);
        let rows = &mut self.blocks[i..i + 3 * w + 4];
        for r in 0..4 {
            rows[r * w..r * w + 4].copy_from_slice(&NO_BLOCK_ROW);
        }
    }

    /// The part of [`Self::clear_mb`] a macroblock being decoded needs: its coefficient counts,
    /// which only coded blocks write. Every macroblock writes its QP, CBP, transform size and
    /// motion (see `reset_blocks` in `mb`); the chroma prediction mode, DC flags and motion edges
    /// are read only from macroblocks whose decoding wrote them.
    pub fn clear_mb_state(&mut self, x: usize, y: usize) {
        let (i, w) = (self.b4(x * 4, y * 4), self.mb_w * 4);
        fill_rect::<_, 4>(&mut self.nnz, i, w, 4, 0);
        // `intra_mode` needs no reset: it is read only for Intra_4x4 and Intra_8x8 macroblocks,
        // which write all of it before reading their own.
        let i = self.c4(x * 2, y * 2);
        for c in &mut self.nnz_c {
            fill_rect::<_, 2>(c, i, self.mb_w * 2, 2, 0);
        }
    }

    /// For a recycled state: gives every macroblock no slice decoded the state and samples of a
    /// new picture (luma 0, chroma 128).
    pub fn restore_undecoded(&mut self) {
        if !self.recycled {
            return;
        }
        let (w, cw) = (self.width, self.width / 2);
        for addr in 0..self.mb_w * self.mb_h {
            if self.mb_type[addr] != MbType::None {
                continue;
            }
            let (x, y) = (addr % self.mb_w, addr / self.mb_w);
            self.clear_mb(addr, x, y);
            // Frame lines the macroblock covers: in an MBAFF frame a field pair interleaves.
            // The pair's other macroblock decides whether the pair is a field pair.
            let pair_field = self.mbaff && {
                let p = if y % 2 == 0 {
                    addr + self.mb_w
                } else {
                    addr - self.mb_w
                };
                self.mb_type[p] != MbType::None && self.mb_field[p]
            };
            let lines = |n: usize, top: usize| -> Vec<usize> {
                if pair_field {
                    (0..n).map(|k| (y & !1) * n + (y & 1) + 2 * k).collect()
                } else {
                    (0..n).map(|k| top + k).collect()
                }
            };
            for line in lines(16, y * 16) {
                self.planes[0][line * w + x * 16..line * w + x * 16 + 16].fill(0);
            }
            for line in lines(8, y * 8) {
                for c in 1..3 {
                    self.planes[c][line * cw + x * 8..line * cw + x * 8 + 8].fill(128);
                }
            }
        }
    }

    /// Index of the 4x4 block at absolute 4x4 coordinates.
    #[inline]
    pub fn b4(&self, x4: usize, y4: usize) -> usize {
        y4 * self.mb_w * 4 + x4
    }

    /// Index of the chroma 4x4 block at absolute chroma-4x4 coordinates.
    #[inline]
    pub fn c4(&self, x4: usize, y4: usize) -> usize {
        y4 * self.mb_w * 2 + x4
    }

    /// Moves a frame picture's motion into `m` (of the same geometry), leaving this state with
    /// `m`'s old buffers.
    pub fn swap_motion(&mut self, m: &mut Motion) {
        m.mbaff = self.mbaff;
        std::mem::swap(&mut m.blocks, &mut self.blocks);
        std::mem::swap(&mut m.ref_tables, &mut self.ref_tables);
        std::mem::swap(&mut m.mb_table, &mut self.mb_table);
        for (i, t) in self.mb_type.iter().enumerate() {
            m.mb_intra[i] = t.is_intra();
            m.mb_uniform[i] = self.motion_edges[i] == 0;
        }
        m.mb_field.copy_from_slice(&self.mb_field);
    }

    /// Moves samples and motion into `store`: a frame picture swaps buffers with it (leaving this
    /// state with the store's old buffers, for [`PicState::recycle`]); a field picture's rows are
    /// copied into its lines.
    pub fn write_into(&mut self, store: &mut FrameStore) {
        let parity = match self.structure {
            Structure::Frame => None,
            Structure::Field(p) => Some(p),
        };
        if parity.is_none() && store.planes[0].len() == self.planes[0].len() {
            // The state keeps the store's old buffers, made unique by `recycle` before use.
            store.field_planes = Default::default();
            std::mem::swap(&mut store.planes, &mut self.planes.0);
        } else {
            let planes = store.planes_mut(parity);
            for (c, dst) in planes.iter_mut().enumerate() {
                let src = &self.planes[c];
                let (w, h) = if c == 0 {
                    (self.width, self.height)
                } else {
                    (self.width / 2, self.height / 2)
                };
                match parity {
                    None => dst[..w * h].copy_from_slice(&src[..w * h]),
                    Some(p) => {
                        for y in 0..h {
                            let dy = 2 * y + p as usize;
                            dst[dy * w..dy * w + w].copy_from_slice(&src[y * w..y * w + w]);
                        }
                    }
                }
            }
        }
        let m = Arc::make_mut(&mut store.motion);
        m.mbaff = self.mbaff;
        let stride = self.mb_w * 4;
        if parity.is_none() && m.blocks.len() == self.blocks.len() {
            self.swap_motion(m);
            return;
        }
        // A field appends its slices' tables to the other field's.
        if parity.is_none() {
            m.ref_tables.clear();
        }
        let base = m.ref_tables.len() as u16;
        m.ref_tables.extend_from_slice(&self.ref_tables);
        for mby in 0..self.mb_h {
            let fy = match parity {
                None => mby,
                Some(p) => 2 * mby + p as usize,
            };
            for mbx in 0..self.mb_w {
                let src_mb = mby * self.mb_w + mbx;
                let dst_mb = fy * self.mb_w + mbx;
                m.mb_intra[dst_mb] = self.mb_type[src_mb].is_intra();
                m.mb_uniform[dst_mb] = self.motion_edges[src_mb] == 0;
                m.mb_field[dst_mb] = self.mb_field[src_mb];
                m.mb_table[dst_mb] = base + self.mb_table[src_mb];
                for r in 0..4 {
                    let s = (mby * 4 + r) * stride + mbx * 4;
                    let d = (fy * 4 + r) * stride + mbx * 4;
                    m.blocks[d..d + 4].copy_from_slice(&self.blocks[s..s + 4]);
                }
            }
        }
    }
}
