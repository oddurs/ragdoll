use ragdoll_core::{AlignedBuffer, RagdollError, Result};
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use std::path::Path;

/// Zero-copy read-only byte source backed by OS virtual memory mapping (`mmap`)
/// or a 64-byte cache-line-aligned in-memory buffer.
///
/// Container demuxers (`Mp4Demuxer`, `MkvDemuxer`, `MpegTsDemuxer`, `Y4mDemuxer`, `WavDemuxer`)
/// borrow `&'a [u8]` directly from `MmapSource` so `PacketSlice<'a>` payloads never copy or
/// heap-allocate compressed media bytes.
pub struct MmapSource {
    inner: MmapInner,
}

enum MmapInner {
    #[cfg(unix)]
    Mapped {
        ptr: *const u8,
        len: usize,
    },
    Buffer(AlignedBuffer),
    Empty,
}

// SAFETY: The memory map is read-only (`PROT_READ`, `MAP_PRIVATE`) and immutable for its lifetime.
unsafe impl Send for MmapSource {}
unsafe impl Sync for MmapSource {}

#[cfg(unix)]
mod sys {
    use std::os::raw::{c_int, c_void};

    pub const PROT_READ: c_int = 0x01;
    pub const MAP_PRIVATE: c_int = 0x02;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    pub const MADV_WILLNEED: c_int = 3;

    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }
}

impl MmapSource {
    /// Memory-maps the file at `path` read-only (`PROT_READ | MAP_PRIVATE`).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        Self::from_file(&file)
    }

    /// Memory-maps an open [`File`] handle read-only.
    pub fn from_file(file: &File) -> Result<Self> {
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self {
                inner: MmapInner::Empty,
            });
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            // SAFETY: Mapping a valid open file descriptor with PROT_READ and MAP_PRIVATE.
            let raw = unsafe {
                sys::mmap(
                    std::ptr::null_mut(),
                    len,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    fd,
                    0,
                )
            };
            if raw == sys::MAP_FAILED || raw.is_null() {
                return Err(RagdollError::Io(std::io::Error::last_os_error()));
            }
            Ok(Self {
                inner: MmapInner::Mapped {
                    ptr: raw as *const u8,
                    len,
                },
            })
        }

        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(0))?;
            let mut buf = AlignedBuffer::new_zeroed(len);
            f.read_exact(buf.as_mut_slice())?;
            Ok(Self {
                inner: MmapInner::Buffer(buf),
            })
        }
    }

    /// Wraps an in-memory 64-byte-aligned buffer as a `MmapSource`.
    #[inline]
    pub fn from_aligned_buffer(buf: AlignedBuffer) -> Self {
        Self {
            inner: MmapInner::Buffer(buf),
        }
    }

    /// Copies a byte slice into a 64-byte-aligned `MmapSource` (useful for tests).
    #[inline]
    pub fn from_slice(bytes: &[u8]) -> Self {
        Self {
            inner: MmapInner::Buffer(AlignedBuffer::from_slice(bytes)),
        }
    }

    /// Advises the kernel that the entire mapped file will be streamed (`MADV_WILLNEED`),
    /// pre-populating page-table entries in a single kernel call.
    #[inline]
    pub fn prefetch_all(&self) {
        self.prefetch_range(0, self.len());
    }

    /// Advises the kernel that the byte range `[offset..offset + len]` will be needed soon.
    #[inline]
    pub fn prefetch_range(&self, offset: usize, len: usize) {
        #[cfg(unix)]
        if let MmapInner::Mapped {
            ptr,
            len: total_len,
        } = self.inner
            && offset < total_len
        {
            let clamped = len.min(total_len - offset);
            if clamped > 0 {
                let page_mask = !4095usize;
                let aligned_off = offset & page_mask;
                let aligned_len = (offset - aligned_off) + clamped;
                unsafe {
                    let _ = sys::madvise(
                        ptr.add(aligned_off) as *mut std::os::raw::c_void,
                        aligned_len,
                        sys::MADV_WILLNEED,
                    );
                }
            }
        }
        #[cfg(not(unix))]
        let _ = (offset, len);
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            #[cfg(unix)]
            MmapInner::Mapped { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            MmapInner::Buffer(buf) => buf.as_slice(),
            MmapInner::Empty => &[],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}

impl Deref for MmapSource {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for MmapSource {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Drop for MmapSource {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let MmapInner::Mapped { ptr, len } = self.inner
            && !ptr.is_null()
            && len > 0
        {
            unsafe {
                let _ = sys::munmap(ptr as *mut std::os::raw::c_void, len);
            }
        }
    }
}

/// Vectored I/O writer (`writev`) that tracks exact byte offsets and avoids intermediate
/// buffer concatenation when emitting container headers + zero-copy packet payloads.
pub struct VectoredWriter<W: Write> {
    inner: W,
    bytes_written: u64,
}

impl<W: Write> VectoredWriter<W> {
    #[inline]
    pub const fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: 0,
        }
    }

    #[inline]
    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    #[inline]
    pub fn into_inner(self) -> W {
        self.inner
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Writes a single contiguous slice and advances `bytes_written`.
    #[inline]
    pub fn write_all_counted(&mut self, buf: &[u8]) -> Result<()> {
        self.inner.write_all(buf)?;
        self.bytes_written += buf.len() as u64;
        Ok(())
    }

    /// Writes multiple disjoint slices (e.g., container framing headers + zero-copy packet payload)
    /// while preserving page-aligned buffered I/O when backed by `BufWriter`.
    #[inline]
    pub fn write_vectored_all(&mut self, slices: &[&[u8]]) -> Result<()> {
        for s in slices {
            if !s.is_empty() {
                self.inner.write_all(s)?;
                self.bytes_written += s.len() as u64;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()?;
        Ok(())
    }
}
