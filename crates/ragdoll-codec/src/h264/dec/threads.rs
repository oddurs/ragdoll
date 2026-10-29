//! Frame threading: frame pictures decoded on worker threads, several at a time.
//!
//! The decoder's own thread parses parameter sets and slice headers, manages the DPB and builds
//! each slice's reference lists, in decode order, as it does when decoding alone. A frame
//! picture's slices are then decoded and deblocked on a worker. Pictures that predict from one
//! still being decoded read only the rows it has published as final ([`PicSync`]), waiting for
//! those their motion vectors reach, and direct prediction waits for the co-located
//! macroblock row's motion. Every picture is decoded exactly as it is on one thread, so the output is the same.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use ragdoll_core::{RagdollError, Result};

use super::deblock::SliceDeblock;
use super::mb::{self, RefPic, SliceCtx};
use super::picture::{PicPlanes, PicState, PicSync};
use super::ps::{Pps, Sps};
use super::recon::Dequant;
use super::slice::SliceHeader;

/// One slice of a picture decoded on a worker: its header, reference lists and data.
pub struct SliceJob {
    pub sh: SliceHeader,
    pub refs: [Vec<RefPic>; 2],
    pub slice_num: u32,
    pub rbsp: Vec<u8>,
}

/// A frame picture to decode on a worker, into `planes` (`fresh` when newly allocated).
pub struct PicJob {
    pub mb_w: usize,
    pub mb_h: usize,
    pub planes: PicPlanes,
    pub fresh: bool,
    pub deblock: Vec<SliceDeblock>,
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    pub dq: Arc<Dequant>,
    pub cur_poc: i32,
    pub field_poc: [i32; 2],
    pub slices: Vec<SliceJob>,
    pub sync: Arc<PicSync>,
    pub done: Arc<JobDone>,
}

/// Where a worker leaves the first error of a decoded picture's slices.
#[derive(Default)]
pub struct JobDone(Mutex<Option<Result<()>>>);

impl JobDone {
    /// The result of a job whose picture is complete (see [`PicSync::wait`]).
    pub fn take(&self) -> Option<Result<()>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).take()
    }
}

impl PicJob {
    /// Decodes the picture with the worker's own state `local`, which stays in its caches.
    fn run(mut self, local: &mut Option<PicState>) {
        let planes = std::mem::replace(&mut self.planes, PicPlanes::empty());
        let mut st = PicState::recycle_into(local.take(), self.mb_w, self.mb_h, planes, self.fresh);
        st.start_rows(true, Some(self.sync.clone()));
        st.rows.slices = std::mem::take(&mut self.deblock);
        let res = catch_unwind(AssertUnwindSafe(|| self.decode(&mut st)))
            .unwrap_or(Err(RagdollError::InvalidSyntax("H.264 worker panicked")));
        *local = Some(st);
        // The result is left before the picture is marked complete, which its owner waits on.
        *self.done.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(res);
        self.sync.complete();
    }

    fn is_ref(&self) -> bool {
        self.slices.first().is_some_and(|s| s.sh.nal_ref_idc > 0)
    }

    fn decode(&mut self, st: &mut PicState) -> Result<()> {
        let mut res = Ok(());
        for s in &mut self.slices {
            let cx = SliceCtx {
                sps: &self.sps,
                pps: &self.pps,
                sh: &s.sh,
                slice_num: s.slice_num,
                refs: std::mem::take(&mut s.refs),
                field_refs: None,
                cur_poc: self.cur_poc,
                field_poc: self.field_poc,
                dq: &self.dq,
            };
            let r = mb::decode_slice(&cx, st, &s.rbsp);
            if res.is_ok() {
                res = r;
            }
        }
        st.finish_rows();
        res
    }
}

struct Queue {
    jobs: VecDeque<PicJob>,
    stop: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    cv: Condvar,
}

/// Worker threads taking picture jobs in submission (decode) order. A job waits only on
/// pictures submitted before it, which earlier-started workers are decoding, so they always
/// make progress.
pub struct Workers {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl Workers {
    /// Starts up to `n` workers; `None` when none could be started.
    pub fn new(n: usize) -> Option<Self> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                stop: false,
            }),
            cv: Condvar::new(),
        });
        let handles: Vec<_> = (0..n)
            .map_while(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("h264-dec-{i}"))
                    .spawn(move || work(&shared))
                    .ok()
            })
            .collect();
        (!handles.is_empty()).then_some(Self { shared, handles })
    }

    pub fn threads(&self) -> usize {
        self.handles.len()
    }

    pub fn submit(&self, job: PicJob) {
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        q.jobs.push_back(job);
        self.shared.cv.notify_one();
    }
}

fn work(shared: &Shared) {
    let mut local = None;
    loop {
        let job = {
            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                // Reference pictures first: later pictures wait on them, and on nothing else.
                let i = q.jobs.iter().position(PicJob::is_ref).unwrap_or(0);
                if let Some(job) = q.jobs.remove(i) {
                    break job;
                }
                if q.stop {
                    return;
                }
                q = shared.cv.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        job.run(&mut local);
    }
}

impl Drop for Workers {
    /// Finishes the queued jobs and joins the workers.
    fn drop(&mut self) {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stop = true;
        self.shared.cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
