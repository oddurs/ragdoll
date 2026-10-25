## Summary

- **Cairn Item**: `cairn/items/XXXX-*.md`
- **Milestone**: `m1-foundation` | `m2-pix` | `m3-format` | `m4-codec` | `m5-cli-bench`
- **Area**: `core` | `simd` | `pix` | `format` | `codec-video` | `codec-audio` | `pipeline` | `cli` | `bench` | `arch`

### Changes
- 

### SIMD & Algorithmic Notes (if applicable)
- **Scalar Fallback**: 
- **NEON / AVX2 Intrinsics**: 
- **Throughput / Benchmark Impact**: 

## Verification Checklist

- [ ] `cargo fmt --all -- --check` passes cleanly
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes with zero warnings
- [ ] `cargo test --workspace` passes (including SIMD vs. scalar bit-exact differential tests)
- [ ] `cairn check --render --strict` passes (`ROADMAP.md` is in sync with `cairn/items/`)
