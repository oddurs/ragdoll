//! Support for tests that run external reference tools such as `ffmpeg` and `ffprobe` (`0061`).
//!
//! A missing tool, or a reference command that fails, fails the test: interop tests must never
//! pass without testing anything. Set `RAGDOLL_ALLOW_MISSING_TOOLS=1` to skip them instead on a
//! machine without the tools. CI never sets it.

use std::process::{Command, ExitStatus};

/// Environment variable that turns a missing tool into a skipped test instead of a failure.
pub const ALLOW_MISSING_TOOLS_ENV: &str = "RAGDOLL_ALLOW_MISSING_TOOLS";

fn allow_missing() -> bool {
    std::env::var(ALLOW_MISSING_TOOLS_ENV).is_ok_and(|v| v == "1")
}

/// Whether `tool` runs (`tool -version`). When it does not, panics, or returns `false` so the
/// caller skips its interop checks if [`ALLOW_MISSING_TOOLS_ENV`] is `1`.
pub fn require_tool(tool: &str) -> bool {
    let runs = Command::new(tool)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success());
    if runs {
        return true;
    }
    assert!(
        allow_missing(),
        "`{tool}` is required by this test and is not installed; install it, or set \
         {ALLOW_MISSING_TOOLS_ENV}=1 to skip"
    );
    eprintln!("skipping: `{tool}` is not installed ({ALLOW_MISSING_TOOLS_ENV}=1)");
    false
}

/// Whether a reference command (`what`, for the message) ran and succeeded. When it did not,
/// panics, or returns `false` if [`ALLOW_MISSING_TOOLS_ENV`] is `1`.
pub fn tool_succeeded(status: std::io::Result<ExitStatus>, what: &str) -> bool {
    match status {
        Ok(s) if s.success() => true,
        other => {
            assert!(
                allow_missing(),
                "{what} failed ({other:?}); set {ALLOW_MISSING_TOOLS_ENV}=1 to skip interop \
                 checks on a machine without the reference tools"
            );
            eprintln!("skipping: {what} failed ({other:?}; {ALLOW_MISSING_TOOLS_ENV}=1)");
            false
        }
    }
}
