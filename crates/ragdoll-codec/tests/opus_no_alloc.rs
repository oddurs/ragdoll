//! The Opus decoder allocates nothing per packet once it has seen a stream's modes, and the
//! encoder nothing per frame.
//!
//! Every stream in `tests/data/opus` (SILK, CELT and hybrid, all bandwidths and packet
//! durations, with and without lost packets) is decoded twice by one decoder; the second pass
//! must not touch the heap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ragdoll_codec::opus::decoder::OpusPacketDecoder;

struct Counting;

thread_local! {
    /// Set while a test counts its own thread's allocations (the test harness and other tests
    /// allocate on other threads meanwhile).
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        ALLOCS.with(|a| a.set(a.get() + 1));
    }
}

// Safety: forwards to the system allocator, only counting calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
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
                ALLOCS.with(|a| a.set(0));
                COUNTING.with(|c| c.set(pass == 1));
                for (i, p) in packets.iter().enumerate() {
                    // Every seventh packet is lost, exercising concealment too.
                    let pkt = (i % 7 != 3).then_some(*p);
                    let _ = dec.decode(pkt, &mut pcm, max, false);
                }
                COUNTING.with(|c| c.set(false));
            }
            let n = ALLOCS.with(Cell::get);
            assert_eq!(
                n,
                0,
                "{} at {rate} Hz, {ch} ch: {n} allocations",
                path.display()
            );
        }
    }
}

/// The encoder allocates nothing per frame either, once it has coded a frame of the size:
/// mono and stereo, tonal input with clicks (so transients), several rates and complexities.
#[test]
fn encoding_allocates_nothing_per_frame() {
    use ragdoll_codec::opus::encoder::OpusPacketEncoder;
    let mut seed = 1u32;
    let pcm: Vec<i16> = (0..48000 * 2)
        .map(|n: i32| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let tone = ((n % 97) - 48) * 400;
            let click = if n % 9600 < 4 { 24000 } else { 0 };
            (tone + ((seed >> 20) as i32 - 2048) + click).clamp(-32768, 32767) as i16
        })
        .collect();
    let mut out = vec![0u8; 1500];
    for ch in [1usize, 2] {
        for (bps, vbr, complexity) in [(96000, true, 9), (32000, false, 5), (256000, true, 10)] {
            for frame in [120usize, 240, 480, 960] {
                let mut enc = OpusPacketEncoder::new(ch).unwrap();
                enc.set_bitrate(bps);
                enc.set_vbr(vbr);
                enc.set_complexity(complexity);
                let chunks: Vec<&[i16]> = pcm.chunks_exact(frame * ch).collect();
                enc.encode(chunks[0], frame, &mut out).unwrap();
                ALLOCS.with(|a| a.set(0));
                COUNTING.with(|c| c.set(true));
                for c in &chunks[1..] {
                    enc.encode(c, frame, &mut out).unwrap();
                }
                COUNTING.with(|c| c.set(false));
                let n = ALLOCS.with(Cell::get);
                assert_eq!(n, 0, "{ch} ch {bps} bit/s {frame} samples: {n} allocations");
            }
        }
    }
}
