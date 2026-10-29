//! Decoding H.264 from a container read in decoding order (Matroska) with frame threads yields
//! every picture, identical to one thread's (`0088`): the decoder still holds the last pictures
//! when the input ends.

use ragdoll_pipeline::{
    TranscodeOptions, VideoFrameStream, synth_media, transcode_file_with_options,
};

#[test]
fn matroska_decode_is_complete_and_independent_of_threads() {
    let dir = std::env::temp_dir().join(format!("ragdoll_frame_threads_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (input, mkv) = (dir.join("in.mp4"), dir.join("in.mkv"));
    synth_media(&input, 320, 180, 30, 1.0).unwrap();
    let opts = TranscodeOptions {
        qp: 28,
        threads: 1,
        ..Default::default()
    };
    transcode_file_with_options(&input, &mkv, &opts).unwrap();
    let data = std::fs::read(&mkv).unwrap();

    let decode = |threads| {
        let mut s = VideoFrameStream::open(&data, usize::MAX, threads).unwrap();
        let n = s.frame_count();
        let mut frames = Vec::new();
        while let Some(batch) = s.next_batch().unwrap() {
            frames.extend(batch);
        }
        assert_eq!(frames.len(), n, "{threads} threads: frames decoded");
        frames
    };
    let one = decode(1);
    assert!(!one.is_empty());
    let four = decode(4);
    for (i, (a, b)) in one.iter().zip(&four).enumerate() {
        assert_eq!(a.pts, b.pts, "frame {i}: pts");
        for (pa, pb) in a.planes.iter().zip(&b.planes) {
            assert!(
                pa.data() == pb.data(),
                "frame {i}: samples differ with 4 threads"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
