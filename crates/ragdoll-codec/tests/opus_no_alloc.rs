//! The Opus decoder allocates nothing per packet once it has seen a stream's modes.
//!
//! Every stream in `tests/data/opus` (SILK, CELT and hybrid, all bandwidths and packet
//! durations, with and without lost packets) is decoded twice by one decoder; the second pass
//! must not touch the heap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ragdoll_codec::opus::decoder::OpusPacketDecoder;

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// Safety: forwards to the system allocator, only counting calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Packets of an `opus_demo` bitstream (4-byte length, 4-byte final range, payload).
fn read_bit(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 8 <= data.len() {
        let len = u32::from_be_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        out.push(&data[off + 8..off + 8 + len]);
        off += 8 + len;
    }
    out
}

#[test]
fn decoding_allocates_nothing_per_packet() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/opus");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "bit"))
        .collect();
    names.sort();
    assert!(!names.is_empty());
    for path in names {
        let data = std::fs::read(&path).unwrap();
        let packets = read_bit(&data);
        for (rate, ch) in [(48000, 2), (16000, 1)] {
            let mut dec = OpusPacketDecoder::new(rate, ch).unwrap();
            let mut pcm = vec![0i16; rate as usize * 120 / 1000 * ch];
            let max = pcm.len() / ch;
            for pass in 0..2 {
                ALLOCS.store(0, Ordering::Relaxed);
                COUNTING.store(pass == 1, Ordering::Relaxed);
                for (i, p) in packets.iter().enumerate() {
                    // Every seventh packet is lost, exercising concealment too.
                    let pkt = (i % 7 != 3).then_some(*p);
                    let _ = dec.decode(pkt, &mut pcm, max, false);
                }
                COUNTING.store(false, Ordering::Relaxed);
            }
            let n = ALLOCS.load(Ordering::Relaxed);
            assert_eq!(
                n,
                0,
                "{} at {rate} Hz, {ch} ch: {n} allocations",
                path.display()
            );
        }
    }
}
