# Contributing to Ragdoll

Ragdoll is a zero-C-FFI, explicit-SIMD multimedia engine in pure Rust. Every commit on `main` must compile with zero warnings, pass bit-exact scalar-vs-SIMD differential tests, and keep the in-tree [Cairn](https://github.com/oddurs/cairn) roadmap (`cairn/items/` and `ROADMAP.md`) in strict sync.

---

## 1. The Cairn + Git Branch Workflow

All work in Ragdoll is tracked in-tree via `cairn` (`cairn.toml` + `cairn/items/*.md`). We do not use external issue trackers or ad-hoc `TODO.md` files.

### Step 1: Claim or Create a Cairn Item

Before writing code, claim the next ready item or create a new one:

```sh
# See ready work in priority order
cairn next --view next

# Claim an item and inspect its context prompt
cairn claim <ID> --as <your-handle>
cairn prompt <ID>
```

### Step 2: Create a Topic Branch

Never commit directly to `main`. Branch from `main` using the Cairn item's type and 4-digit ID:

```sh
git checkout main && git pull --ff-only
git checkout -b <type>/<ID>-<short-slug>
```

Examples:
- `kernel/0012-bt709-bt2020-oklab-color-matrix`
- `feature/0015-isobmff-mp4-zero-copy-demuxer`
- `chore/0026-git-pr-workflow`

### Step 3: Implement & Verify Locally

Every SIMD kernel in `ragdoll-simd`, `ragdoll-pix`, or `ragdoll-codec-*` **must** follow the three-tier architecture (`ADR-0006`):
1. **Bit-exact scalar reference** (`*_scalar`) with documented fixed-point precision (`Q14`, `Q6`, etc.) and rounding bias.
2. **Explicit architecture intrinsics** (`#[target_feature(enable = "neon")]` on `aarch64`, `#[target_feature(enable = "avx2")]` on `x86_64`) encapsulated behind safe slice APIs with upfront bounds assertions.
3. **Differential unit tests** verifying that the SIMD kernel matches the scalar reference across edge dimensions, unaligned tails, and extremal inputs.

Run the local verification suite before pushing:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

### Step 4: Update Cairn & Render `ROADMAP.md`

Record architectural findings, tick verified acceptance criteria, close the item (if complete), and regenerate `ROADMAP.md`:

```sh
cairn note <ID> 'Document key decisions, register pressure, or benchmark numbers.'
cairn tick <ID> --all
cairn close <ID> --result 'Summary of delivered capability and test verification.'
cairn render
cairn check --render --strict
```

### Step 5: Open a Pull Request & Squash-Merge

Push your branch and open a Pull Request via the GitHub CLI (`gh`):

```sh
git add -A
git commit -m "<type>(<area>): <concise summary> (cairn #<ID>)"
git push -u origin HEAD

# Create a PR using .github/pull_request_template.md
gh pr create --title "<type>(<area>): <concise summary> (#<ID>)" --body-file - << 'EOF'
...
EOF

# Watch fast Light CI checks (~20s on ubuntu-24.04-arm) and squash-merge
gh pr checks --watch
gh pr merge --squash --delete-branch
```

---

## 2. Commit Message Convention

We use structured, scannable commit and PR titles:

```text
<type>(<area>): <summary> (cairn #<ID>)
```

- **`<type>`**: `feature`, `kernel`, `benchmark`, `decision`, `bug`, `chore`, `docs` (matching `cairn.toml`)
- **`<area>`**: `core`, `simd`, `pix`, `format`, `codec-video`, `codec-audio`, `pipeline`, `cli`, `bench`, `arch`
