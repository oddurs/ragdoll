//! Runtime selection of SIMD kernels (`0061`).
//!
//! Every kernel with a SIMD path asks [`neon`] or [`avx2`] before taking it. By default that is
//! the best level the CPU supports; the `RAGDOLL_SIMD` environment variable (`scalar`, `neon` or
//! `avx2`) forces a level instead, so the whole test suite can also run on the scalar reference
//! paths of SIMD hardware. The variable is read once, on first use.

use std::sync::atomic::{AtomicU8, Ordering};

/// Environment variable that forces a [`SimdDispatch`] level.
pub const SIMD_ENV: &str = "RAGDOLL_SIMD";

/// The SIMD instruction set that kernels dispatch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdDispatch {
    /// Portable scalar reference code.
    Scalar,
    /// `aarch64` Advanced SIMD (NEON), baseline on every `aarch64` CPU.
    Neon,
    /// `x86_64` AVX2.
    Avx2,
}

const UNSET: u8 = 0;
static ACTIVE: AtomicU8 = AtomicU8::new(UNSET);

impl SimdDispatch {
    /// The best level this CPU supports.
    pub fn detected() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                Self::Avx2
            } else {
                Self::Scalar
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }

    /// The level in use: [`Self::detected`], or the one `RAGDOLL_SIMD` names.
    ///
    /// # Panics
    ///
    /// When `RAGDOLL_SIMD` names an unknown level or one this CPU does not support.
    #[inline]
    pub fn active() -> Self {
        match ACTIVE.load(Ordering::Relaxed) {
            UNSET => Self::resolve(),
            v => Self::from_code(v),
        }
    }

    #[cold]
    fn resolve() -> Self {
        let detected = Self::detected();
        let level = match std::env::var(SIMD_ENV) {
            Err(_) => detected,
            Ok(v) if v.is_empty() => detected,
            Ok(v) => {
                let wanted = match v.to_ascii_lowercase().as_str() {
                    "scalar" => Self::Scalar,
                    "neon" => Self::Neon,
                    "avx2" => Self::Avx2,
                    _ => panic!("{SIMD_ENV}={v}: expected scalar, neon or avx2"),
                };
                assert!(
                    wanted == Self::Scalar || wanted == detected,
                    "{SIMD_ENV}={v}: this CPU supports {}",
                    detected.name()
                );
                wanted
            }
        };
        ACTIVE.store(level.code(), Ordering::Relaxed);
        level
    }

    const fn code(self) -> u8 {
        match self {
            Self::Scalar => 1,
            Self::Neon => 2,
            Self::Avx2 => 3,
        }
    }

    const fn from_code(v: u8) -> Self {
        match v {
            2 => Self::Neon,
            3 => Self::Avx2,
            _ => Self::Scalar,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2 => "avx2",
        }
    }
}

/// Whether kernels should take their NEON path.
#[inline]
pub fn neon() -> bool {
    cfg!(target_arch = "aarch64") && SimdDispatch::active() == SimdDispatch::Neon
}

/// Whether kernels should take their AVX2 path.
#[inline]
pub fn avx2() -> bool {
    cfg!(target_arch = "x86_64") && SimdDispatch::active() == SimdDispatch::Avx2
}
