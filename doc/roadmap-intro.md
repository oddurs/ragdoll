> *"What if FFmpeg were written in 2026 from first principles—with zero C legacy, cache-aligned Arena pools, fused SIMD filter kernels, zero-copy memory-mapped containers, and a CLI designed for humans?"*

**Ragdoll** is a ground-up, **100% pure Rust (zero C FFI)** media processing engine and CLI tuned for **Apple Silicon (`aarch64` NEON)** and **`x86_64` AVX2**. Every bitstream parser, Exp-Golomb/CABAC/Rice coder, integer DCT/Hadamard transform, polyphase Lanczos-3 resampler, and container muxer/demuxer is built from scratch in safe + explicit-SIMD Rust.

This roadmap is generated directly from the item backlog in `cairn/items/` under the schema in `cairn.toml`. Each item serves as both an engineering specification and a self-contained prompt (`cairn prompt <ID>`).

```sh
cairn list --view next       # Ready-to-execute work in dependency order
cairn list --view kernels    # SIMD (NEON/AVX2) mathematical & pixel kernels
cairn list --view benchmarks # Head-to-head Ragdoll vs. FFmpeg 9.0.2 benchmarks
cairn list --view decisions  # Architectural Decision Records (ADRs)
cairn board                  # Kanban view of the entire project
```
