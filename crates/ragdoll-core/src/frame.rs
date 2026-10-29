use crate::color::Colorimetry;
use crate::error::{RagdollError, Result};
use crate::format::{ChannelLayout, PixelFormat, SampleFormat};
use crate::time::Timestamp;
use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Hardware cache-line and wide-SIMD alignment (64 bytes = 512 bits = 4x NEON `uint8x16_t` registers).
pub const CACHE_LINE_ALIGN: usize = 64;

/// Mandatory zeroed tail padding appended after the logical end of every `AlignedBuffer`
/// (matching FFmpeg's `AV_INPUT_BUFFER_PADDING_SIZE = 64`).
///
/// Guarantees that unaligned 64-bit bitstream refills (`u64::from_be_bytes`) and
/// 64-byte unrolled NEON vector loads at the end of a packet or plane never read out-of-bounds.
pub const INPUT_BUFFER_PADDING_SIZE: usize = 64;

/// Rounds `val` up to the nearest multiple of 64 bytes (returns 64 when `val == 0`).
#[inline]
pub const fn align_up_64(val: usize) -> usize {
    if val == 0 {
        CACHE_LINE_ALIGN
    } else {
        (val + (CACHE_LINE_ALIGN - 1)) & !(CACHE_LINE_ALIGN - 1)
    }
}

/// Computes the optimal 64-byte-aligned row stride for a video plane of `row_bytes` active width.
///
/// Incorporates `dav1d`'s L1/L2 cache set-associativity anti-aliasing optimization (`dav1d/src/picture.c`):
/// when a stride is an exact multiple of 1024 bytes (`(stride & 1023) == 0`), vertically adjacent pixels
/// `(x, y), (x, y+1), ..., (x, y+7)` in an 8-tap vertical filter or macroblock map to the exact same
/// CPU L1/L2 cache set, causing conflict evictions. Adding one 64-byte cache line staggers consecutive
/// rows across 16 distinct cache sets while preserving 64-byte SIMD alignment.
#[inline]
pub const fn optimal_plane_stride(row_bytes: usize) -> usize {
    let mut stride = align_up_64(row_bytes);
    if stride >= 1024 && (stride & 1023) == 0 {
        stride += CACHE_LINE_ALIGN;
    }
    stride
}

/// A heap buffer guaranteed to be aligned to a 64-byte cache-line boundary (`ptr % 64 == 0`)
/// and backed by at least `INPUT_BUFFER_PADDING_SIZE` (64) zeroed tail bytes past `len`.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
}

// Safety: AlignedBuffer owns its heap allocation exclusively.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocates a zero-initialized 64-byte-aligned buffer of `len` active bytes
    /// plus `INPUT_BUFFER_PADDING_SIZE` (64) zeroed tail padding bytes.
    pub fn new_zeroed(len: usize) -> Self {
        let cap = align_up_64(len) + INPUT_BUFFER_PADDING_SIZE;
        let layout = Layout::from_size_align(cap, CACHE_LINE_ALIGN)
            .expect("valid 64-byte aligned allocation layout");
        // Safety: cap >= 128 > 0 and align is 64 (power of two).
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            alloc::handle_alloc_error(layout);
        };
        Self { ptr, len, cap }
    }

    /// A buffer of `rows` rows of `stride` bytes, row `y` starting with `row(y)` (at most
    /// `stride` bytes) and zero-filled after it, plus zeroed tail padding. Every byte is written
    /// once, without zero-filling first.
    pub fn from_rows<'a>(
        stride: usize,
        rows: usize,
        mut row: impl FnMut(usize) -> &'a [u8],
    ) -> Self {
        let len = stride.saturating_mul(rows);
        let cap = align_up_64(len) + INPUT_BUFFER_PADDING_SIZE;
        let layout = Layout::from_size_align(cap, CACHE_LINE_ALIGN)
            .expect("valid 64-byte aligned allocation layout");
        // Safety: cap >= 64 > 0 and align is 64 (power of two).
        let raw = unsafe { alloc::alloc(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            alloc::handle_alloc_error(layout);
        };
        // Safety: each row's `stride` bytes and then the tail `cap - len` bytes are written in
        // full, within the allocation, before the buffer is read.
        unsafe {
            for y in 0..rows {
                let src = row(y);
                let n = src.len().min(stride);
                let dst = raw.add(y * stride);
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
                std::ptr::write_bytes(dst.add(n), 0, stride - n);
            }
            std::ptr::write_bytes(raw.add(len), 0, cap - len);
        }
        Self { ptr, len, cap }
    }

    /// Copies a byte slice into a new 64-byte-aligned buffer with 64 bytes of zeroed tail padding.
    pub fn from_slice(data: &[u8]) -> Self {
        let mut buf = Self::new_zeroed(data.len());
        buf.as_mut_slice().copy_from_slice(data);
        buf
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Physical padded capacity (always a multiple of 64 bytes and $\ge \text{len} + 64$).
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub const fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Full padded slice of length `self.capacity()` including the 64-byte zeroed tail guard.
    #[inline]
    pub const fn as_padded_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    /// Returns the guaranteed 64-byte (`INPUT_BUFFER_PADDING_SIZE`) zeroed tail safety region
    /// immediately following the logical payload `0..self.len`.
    #[inline]
    pub fn tail_padding(&self) -> &[u8] {
        &self.as_padded_slice()[self.len..self.len + INPUT_BUFFER_PADDING_SIZE]
    }

    #[inline]
    pub const fn as_padded_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }

    /// Clears the entire buffer (including tail padding) to zero without reallocating.
    #[inline]
    pub fn fill_zero(&mut self) {
        self.as_padded_mut_slice().fill(0);
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut out = Self::new_zeroed(self.len);
        out.as_padded_mut_slice()
            .copy_from_slice(self.as_padded_slice());
        out
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("cap", &self.cap)
            .field(
                "aligned_64",
                &(self.ptr.as_ptr() as usize).is_multiple_of(64),
            )
            .finish()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.cap, CACHE_LINE_ALIGN)
            .expect("valid layout on deallocation");
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Three planes of one picture shared between a producer and the frames it hands out (a
/// decoder's reference picture, for example); see [`Plane::shared`].
pub type SharedPlanes = Arc<[AlignedBuffer; 3]>;

/// Storage of a [`Plane`]: its own buffer, or one plane of [`SharedPlanes`].
#[derive(Clone, Debug)]
enum PlaneBuf {
    Owned(AlignedBuffer),
    Shared(SharedPlanes, usize),
}

impl PlaneBuf {
    #[inline]
    fn get(&self) -> &AlignedBuffer {
        match self {
            Self::Owned(b) => b,
            Self::Shared(p, i) => &p[*i],
        }
    }
}

/// A single 2D image plane with 64-byte cache-line-aligned base pointer,
/// cache-set-staggered row stride, and 64-byte zeroed tail padding.
#[derive(Clone, Debug)]
pub struct Plane {
    buf: PlaneBuf,
    width: usize,
    height: usize,
    row_bytes: usize,
    stride: usize,
    bytes_per_sample: usize,
}

impl Plane {
    /// A plane whose row `y` holds `row(y)` (truncated to the row width), zero-padded.
    pub fn from_rows<'a>(
        width: usize,
        height: usize,
        bytes_per_sample: usize,
        mut row: impl FnMut(usize) -> &'a [u8],
    ) -> Self {
        let row_bytes = width.saturating_mul(bytes_per_sample);
        let stride = optimal_plane_stride(row_bytes);
        Self {
            buf: PlaneBuf::Owned(AlignedBuffer::from_rows(stride, height, |y| {
                let r = row(y);
                &r[..r.len().min(row_bytes)]
            })),
            width,
            height,
            row_bytes,
            stride,
            bytes_per_sample,
        }
    }

    pub fn new(width: usize, height: usize, bytes_per_sample: usize) -> Self {
        let row_bytes = width.saturating_mul(bytes_per_sample);
        let stride = optimal_plane_stride(row_bytes);
        let total_bytes = stride.saturating_mul(height);
        Self {
            buf: PlaneBuf::Owned(AlignedBuffer::new_zeroed(total_bytes)),
            width,
            height,
            row_bytes,
            stride,
            bytes_per_sample,
        }
    }

    /// A plane over plane `idx` of `planes` (`height` rows of `stride` bytes from its start,
    /// `width` samples used per row), read without copying. Mutable access copies the plane
    /// into a buffer of its own first, so the producer's samples are never changed through it.
    ///
    /// # Panics
    /// If the rows do not fit in the shared plane, `stride` is not a multiple of 64 or the
    /// buffer is not 64-byte aligned.
    pub fn shared(
        planes: SharedPlanes,
        idx: usize,
        width: usize,
        height: usize,
        bytes_per_sample: usize,
        stride: usize,
    ) -> Self {
        let row_bytes = width.saturating_mul(bytes_per_sample);
        assert!(row_bytes <= stride && stride.is_multiple_of(CACHE_LINE_ALIGN));
        assert!(stride * height <= planes[idx].len());
        assert!((planes[idx].as_ptr() as usize).is_multiple_of(CACHE_LINE_ALIGN));
        Self {
            buf: PlaneBuf::Shared(planes, idx),
            width,
            height,
            row_bytes,
            stride,
            bytes_per_sample,
        }
    }

    /// The plane's own buffer, copying a shared plane's rows into one first.
    fn owned_mut(&mut self) -> &mut AlignedBuffer {
        if let PlaneBuf::Shared(p, i) = &self.buf {
            let (src, stride) = (&p[*i], self.stride);
            let own = AlignedBuffer::from_rows(stride, self.height, |y| {
                &src.as_slice()[y * stride..y * stride + stride]
            });
            self.buf = PlaneBuf::Owned(own);
        }
        match &mut self.buf {
            PlaneBuf::Owned(b) => b,
            PlaneBuf::Shared(..) => unreachable!("plane made its own above"),
        }
    }

    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Active unpadded byte width of each row (`width * bytes_per_sample`).
    #[inline]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// 64-byte aligned (and 1024-byte cache-set-dealiased) byte distance between consecutive scanlines.
    #[inline]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub const fn bytes_per_sample(&self) -> usize {
        self.bytes_per_sample
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.buf.get().as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.owned_mut().as_mut_ptr()
    }

    /// The plane's `stride * height` bytes.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.buf.get().as_slice()[..self.stride * self.height]
    }

    /// The buffer holding the plane; a shared plane's may hold more rows after its own.
    #[inline]
    pub fn buffer(&self) -> &AlignedBuffer {
        self.buf.get()
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        let n = self.stride * self.height;
        &mut self.owned_mut().as_mut_slice()[..n]
    }

    /// Returns the active (unpadded) pixel bytes of scanline `y`.
    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        debug_assert!(y < self.height);
        let start = y * self.stride;
        &self.buf.get().as_slice()[start..start + self.row_bytes]
    }

    /// Returns a mutable slice of the active (unpadded) pixel bytes of scanline `y`.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        debug_assert!(y < self.height);
        let start = y * self.stride;
        let row_bytes = self.row_bytes;
        &mut self.owned_mut().as_mut_slice()[start..start + row_bytes]
    }

    /// Returns the full 64-byte-aligned padded row of length `self.stride()` for SIMD vector loops.
    #[inline]
    pub fn row_padded(&self, y: usize) -> &[u8] {
        debug_assert!(y < self.height);
        let start = y * self.stride;
        &self.buf.get().as_slice()[start..start + self.stride]
    }

    /// Returns a mutable full 64-byte-aligned padded row of length `self.stride()` for SIMD stores.
    #[inline]
    pub fn row_padded_mut(&mut self, y: usize) -> &mut [u8] {
        debug_assert!(y < self.height);
        let start = y * self.stride;
        let stride = self.stride;
        &mut self.owned_mut().as_mut_slice()[start..start + stride]
    }

    /// Replicates the rightmost active sample of every scanline into the right stride padding
    /// so horizontal SIMD convolution kernels and motion compensation never sample zero-padding at edges.
    pub fn extend_right_edge(&mut self) {
        if self.width == 0 || self.row_bytes >= self.stride {
            return;
        }
        let bps = self.bytes_per_sample;
        let row_bytes = self.row_bytes;
        let stride = self.stride;
        for y in 0..self.height {
            let row = self.row_padded_mut(y);
            if bps == 1 {
                let edge_val = row[row_bytes - 1];
                row[row_bytes..stride].fill(edge_val);
            } else {
                let mut dst = row_bytes;
                while dst + bps <= stride {
                    for k in 0..bps {
                        row[dst + k] = row[row_bytes - bps + k];
                    }
                    dst += bps;
                }
            }
        }
    }

    /// Copies tightly packed (unpadded) plane data into this aligned `Plane`.
    pub fn copy_from_packed(&mut self, packed: &[u8]) -> Result<()> {
        let expected = self.row_bytes * self.height;
        if packed.len() < expected {
            return Err(RagdollError::InvalidData(format!(
                "packed plane buffer too short: expected {expected}, got {}",
                packed.len()
            )));
        }
        for y in 0..self.height {
            let src = &packed[y * self.row_bytes..(y + 1) * self.row_bytes];
            self.row_mut(y).copy_from_slice(src);
        }
        Ok(())
    }

    /// Writes active (unpadded) scanlines into `dst`.
    pub fn write_packed(&self, dst: &mut Vec<u8>) {
        dst.reserve(self.row_bytes * self.height);
        for y in 0..self.height {
            dst.extend_from_slice(self.row(y));
        }
    }
}

/// Frame coding type for GOP structure analysis and scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PictureType {
    /// Intra-coded keyframe / IDR.
    I,
    /// Forward predicted frame.
    P,
    /// Bi-directionally predicted frame.
    B,
    #[default]
    Unknown,
}

impl std::fmt::Display for PictureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = match self {
            Self::I => "I",
            Self::P => "P",
            Self::B => "B",
            Self::Unknown => "?",
        };
        write!(f, "{c}")
    }
}

/// A complete uncompressed video frame with 64-byte cache-line-aligned planes and full colorimetry.
#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub planes: Vec<Plane>,
    pub width: usize,
    pub height: usize,
    pub format: PixelFormat,
    pub colorimetry: Colorimetry,
    pub pts: Timestamp,
    pub duration: i64,
    pub keyframe: bool,
    pub pict_type: PictureType,
}

impl VideoFrame {
    pub fn new(width: usize, height: usize, format: PixelFormat) -> Self {
        let num_planes = format.num_planes();
        let mut planes = Vec::with_capacity(num_planes);
        for p in 0..num_planes {
            let (pw, ph) = format.plane_dimensions(width, height, p);
            let bps = format.plane_bytes_per_sample(p);
            planes.push(Plane::new(pw, ph, bps));
        }
        Self {
            planes,
            width,
            height,
            format,
            colorimetry: Colorimetry::default(),
            pts: Timestamp::default(),
            duration: 0,
            keyframe: true,
            pict_type: PictureType::I,
        }
    }

    /// A frame whose plane `p` has row `y` copied from `row(p, y)`, without zero-filling first.
    pub fn from_rows<'a>(
        width: usize,
        height: usize,
        format: PixelFormat,
        mut row: impl FnMut(usize, usize) -> &'a [u8],
    ) -> Self {
        let planes = (0..format.num_planes())
            .map(|p| {
                let (pw, ph) = format.plane_dimensions(width, height, p);
                Plane::from_rows(pw, ph, format.plane_bytes_per_sample(p), |y| row(p, y))
            })
            .collect();
        Self {
            planes,
            width,
            height,
            format,
            colorimetry: Colorimetry::default(),
            pts: Timestamp::default(),
            duration: 0,
            keyframe: true,
            pict_type: PictureType::I,
        }
    }

    /// A frame over existing planes, for example [`Plane::shared`] ones.
    ///
    /// # Panics
    /// If `planes` does not hold `format`'s planes at their sizes for `width`x`height`.
    pub fn from_planes(
        width: usize,
        height: usize,
        format: PixelFormat,
        planes: Vec<Plane>,
    ) -> Self {
        assert_eq!(planes.len(), format.num_planes());
        for (p, plane) in planes.iter().enumerate() {
            assert_eq!(
                (plane.width(), plane.height()),
                format.plane_dimensions(width, height, p)
            );
        }
        Self {
            planes,
            width,
            height,
            format,
            colorimetry: Colorimetry::default(),
            pts: Timestamp::default(),
            duration: 0,
            keyframe: true,
            pict_type: PictureType::I,
        }
    }

    #[inline]
    pub fn plane(&self, idx: usize) -> &Plane {
        &self.planes[idx]
    }

    #[inline]
    pub fn plane_mut(&mut self, idx: usize) -> &mut Plane {
        &mut self.planes[idx]
    }

    /// Replicates right-edge pixels into the SIMD stride padding across all planes.
    pub fn extend_edges(&mut self) {
        for plane in &mut self.planes {
            plane.extend_right_edge();
        }
    }

    /// Resets frame metadata for reuse from a `FramePool`.
    #[inline]
    pub fn reset_metadata(&mut self) {
        self.colorimetry = Colorimetry::default();
        self.pts = Timestamp::default();
        self.duration = 0;
        self.keyframe = true;
        self.pict_type = PictureType::I;
    }

    /// Total physical bytes allocated across all aligned planes.
    pub fn allocated_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.buf.get().capacity()).sum()
    }
}

/// An uncompressed audio frame backed by a 64-byte aligned sample buffer.
#[derive(Clone, Debug)]
pub struct AudioFrame {
    pub data: AlignedBuffer,
    pub sample_rate: u32,
    pub channels: u8,
    pub layout: ChannelLayout,
    pub format: SampleFormat,
    pub num_samples: usize,
    pub pts: Timestamp,
}

impl AudioFrame {
    pub fn new(sample_rate: u32, channels: u8, format: SampleFormat, num_samples: usize) -> Self {
        let byte_len = num_samples
            .saturating_mul(channels as usize)
            .saturating_mul(format.bytes_per_sample());
        Self {
            data: AlignedBuffer::new_zeroed(byte_len),
            sample_rate,
            channels,
            layout: ChannelLayout::from_channels(channels),
            format,
            num_samples,
            pts: Timestamp::default(),
        }
    }

    /// Interprets the buffer as a slice of `i16` samples (panics if `format != SampleFormat::S16le`).
    #[inline]
    pub fn as_i16_slice(&self) -> &[i16] {
        assert_eq!(self.format, SampleFormat::S16le);
        let total = self.num_samples * (self.channels as usize);
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i16, total) }
    }

    /// Interprets the buffer as a mutable slice of `i16` samples.
    #[inline]
    pub fn as_i16_mut_slice(&mut self) -> &mut [i16] {
        assert_eq!(self.format, SampleFormat::S16le);
        let total = self.num_samples * (self.channels as usize);
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut i16, total) }
    }

    /// Interprets the buffer as a slice of `f32` samples (panics if `format != SampleFormat::F32le`).
    #[inline]
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.format, SampleFormat::F32le);
        let total = self.num_samples * (self.channels as usize);
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, total) }
    }

    /// Interprets the buffer as a mutable slice of `f32` samples.
    #[inline]
    pub fn as_f32_mut_slice(&mut self) -> &mut [f32] {
        assert_eq!(self.format, SampleFormat::F32le);
        let total = self.num_samples * (self.channels as usize);
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, total) }
    }
}

struct FramePoolInner {
    free_list: Mutex<Vec<VideoFrame>>,
    max_cached: usize,
    max_bytes: usize,
    cached_bytes: AtomicUsize,
    allocations: AtomicUsize,
    recycled_hits: AtomicUsize,
}

/// Thread-safe, memory-budgeted recycling arena for 64-byte-aligned `VideoFrame` buffers.
///
/// Eliminates page-fault and heap-allocation overhead during steady-state
/// video decoding, scaling, and encoding.
#[derive(Clone)]
pub struct FramePool {
    inner: Arc<FramePoolInner>,
}

impl FramePool {
    /// Creates a `FramePool` caching up to `max_cached` frames (default memory cap: 512 MiB).
    pub fn new(max_cached: usize) -> Self {
        Self::with_memory_budget(max_cached, 512 * 1024 * 1024)
    }

    /// Creates a `FramePool` with explicit frame count and byte budget limits.
    pub fn with_memory_budget(max_cached: usize, max_bytes: usize) -> Self {
        Self {
            inner: Arc::new(FramePoolInner {
                free_list: Mutex::new(Vec::with_capacity(max_cached)),
                max_cached,
                max_bytes,
                cached_bytes: AtomicUsize::new(0),
                allocations: AtomicUsize::new(0),
                recycled_hits: AtomicUsize::new(0),
            }),
        }
    }

    /// Acquires a 64-byte-aligned `VideoFrame` matching `(width, height, format)`,
    /// reusing a pooled frame if one is available or allocating a new one otherwise.
    pub fn acquire(&self, width: usize, height: usize, format: PixelFormat) -> VideoFrame {
        if let Ok(mut guard) = self.inner.free_list.lock()
            && let Some(pos) = guard
                .iter()
                .rposition(|f| f.width == width && f.height == height && f.format == format)
        {
            let mut frame = guard.swap_remove(pos);
            let bytes = frame.allocated_bytes();
            self.inner.cached_bytes.fetch_sub(bytes, Ordering::Relaxed);
            frame.reset_metadata();
            self.inner.recycled_hits.fetch_add(1, Ordering::Relaxed);
            return frame;
        }
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        VideoFrame::new(width, height, format)
    }

    /// Returns a `VideoFrame` back to the arena for future reuse if within pool capacity and byte budget.
    pub fn release(&self, frame: VideoFrame) {
        let bytes = frame.allocated_bytes();
        if let Ok(mut guard) = self.inner.free_list.lock() {
            let cur_bytes = self.inner.cached_bytes.load(Ordering::Relaxed);
            if guard.len() < self.inner.max_cached && cur_bytes + bytes <= self.inner.max_bytes {
                self.inner.cached_bytes.fetch_add(bytes, Ordering::Relaxed);
                guard.push(frame);
            }
        }
    }

    /// Acquires an RAII-managed `PooledVideoFrame` that automatically returns itself
    /// to this `FramePool` when dropped.
    pub fn acquire_scoped(
        &self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> PooledVideoFrame {
        let frame = self.acquire(width, height, format);
        PooledVideoFrame {
            frame: Some(frame),
            pool: self.clone(),
        }
    }

    /// Total number of fresh heap allocations performed by this pool.
    #[inline]
    pub fn allocations(&self) -> usize {
        self.inner.allocations.load(Ordering::Relaxed)
    }

    /// Total number of times an existing `VideoFrame` was recycled without allocation.
    #[inline]
    pub fn recycled_hits(&self) -> usize {
        self.inner.recycled_hits.load(Ordering::Relaxed)
    }

    /// Number of frames currently sitting idle in the free list.
    #[inline]
    pub fn available_in_pool(&self) -> usize {
        self.inner.free_list.lock().map_or(0, |g| g.len())
    }

    /// Total bytes currently held idle in the free list.
    #[inline]
    pub fn cached_bytes(&self) -> usize {
        self.inner.cached_bytes.load(Ordering::Relaxed)
    }
}

impl Default for FramePool {
    fn default() -> Self {
        Self::new(64)
    }
}

/// RAII wrapper around `VideoFrame` that automatically recycles the frame into its `FramePool` on drop.
pub struct PooledVideoFrame {
    frame: Option<VideoFrame>,
    pool: FramePool,
}

impl PooledVideoFrame {
    /// Detaches the underlying `VideoFrame` from the pool so it is not recycled on drop.
    pub fn into_inner(mut self) -> VideoFrame {
        self.frame.take().expect("PooledVideoFrame is active")
    }
}

impl Deref for PooledVideoFrame {
    type Target = VideoFrame;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.frame.as_ref().expect("PooledVideoFrame is active")
    }
}

impl DerefMut for PooledVideoFrame {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.frame.as_mut().expect("PooledVideoFrame is active")
    }
}

impl Drop for PooledVideoFrame {
    fn drop(&mut self) {
        if let Some(frame) = self.frame.take() {
            self.pool.release(frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rows_copies_rows_and_zeroes_padding() {
        let rows: Vec<Vec<u8>> = (0..5)
            .map(|y| (0..7).map(|x| (y * 10 + x + 1) as u8).collect())
            .collect();
        let f = VideoFrame::from_rows(7, 5, PixelFormat::Yuv420p, |p, y| {
            if p == 0 { &rows[y] } else { &rows[y][..4] }
        });
        let mut reference = VideoFrame::new(7, 5, PixelFormat::Yuv420p);
        for p in 0..3 {
            let w = reference.plane(p).width();
            let h = reference.plane(p).height();
            for (y, row) in rows.iter().enumerate().take(h) {
                reference.plane_mut(p).row_mut(y)[..w].copy_from_slice(&row[..w]);
            }
        }
        for p in 0..3 {
            assert_eq!(
                f.plane(p).buf.get().as_padded_slice(),
                reference.plane(p).buf.get().as_padded_slice(),
                "plane {p}"
            );
        }
    }
}
