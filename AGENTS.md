<!-- cairn:begin -->
## Roadmap and issues

This project tracks its roadmap and issues with `cairn`. Every item is a Markdown file under `cairn/items`, described by the schema in `cairn.toml`.

**Do not create ad-hoc TODO, PLAN or NOTES files.** Create a cairn item instead, so the work appears on the board and in the generated roadmap.

### The loop

1. `cairn next --view next` — what is ready to start. It excludes anything blocked by unfinished dependencies and puts work already in progress first.
2. `cairn claim <ID>` — take it before you start, so no one duplicates the work. `cairn claim --next --view next` picks and claims the top-ranked unclaimed item in one step. Then read `cairn prompt <ID>`: the item with everything it rests on — the outcome it serves, what its dependencies concluded, what done means and what earlier runs learned.
3. Do the work. Record what you learn: `cairn set <ID> <field>=<value>` for fields, `cairn note <ID> "<TEXT>"` for anything that needs a sentence — why you chose something, what you tried, what to watch for.
4. `cairn tick <ID> <N>` as each acceptance criterion becomes true — `cairn show <ID> --criteria` lists them numbered. Tick what is true, not what would let you close.
5. `cairn close <ID>` when it is done, or `cairn release <ID>` to hand it back.
6. `cairn check` before you report finished. It must pass.

### Commands

```sh
cairn next --view next --json                 # ready work, ranked
cairn claim --next --view next                # take the next ready item
cairn search <TEXT> --json        # titles, bodies and labels
cairn list --json                 # all open items
cairn list --filter 'blocked=false,priority=p0'
cairn prompt <ID>                 # the item as a prompt, with what it rests on
cairn show <ID> --json            # one item, including its body
cairn new "<TITLE>" --type <TYPE> --milestone <MILESTONE>
cairn set <ID> status=<STATUS>    # also labels+=x, or any field below
cairn note <ID> "<TEXT>"          # append reasoning; never replaces
cairn show <ID> --criteria        # acceptance criteria, numbered
cairn tick <ID> <N>               # tick one; --all for every one
cairn close <ID>
cairn check                       # validate; run before finishing
cairn render                      # regenerate ROADMAP.md
```

Selection uses saved view `next`. Additional filters only narrow it; the view's sort and columns do not change `next` ranking. Over MCP, pass `{"view":"next"}` to `next_items` and to `claim_item` without an id. A direct claim is an explicit assignment outside this selection policy. Regenerate these instructions with `cairn agent --view next --write AGENTS.md`.

Claims coordinate writers in the same item directory and are seen across the worktrees of this repository: `next` leaves out what another worktree has claimed, `claim` refuses it, and `cairn worktrees` shows what each is doing. Separate clones are not read. Agree on assignments before splitting work across clones.

Items are numbered: `0012`, and commands accept the bare number too. Write the number in `depends_on` and other id references. Each item also carries a `uid` tag; leave it alone. If a merge gives two items one number, `cairn renumber` moves the one that arrived and retargets the references that came with it.

### Schema

- **Types**: `feature`, `kernel`, `benchmark`, `decision`, `bug`, `chore`, `docs`, `milestone`
- **Statuses**: `backlog` (open), `planned` (open), `doing` (active), `blocked` (open), `done` (done), `dropped` (dropped)
- **`priority`**: one of p0, p1, p2, p3 — p0: foundational blocker; p1: core milestone deliverable; p2: optimization/polish; p3: future exploration
- **`effort`**: one of s, m, l, xl — Engineering complexity (s, m, l, xl)
- **`area`**: one of core, simd, pix, format, codec-video, codec-audio, pipeline, cli, bench, arch — Crate or subsystem boundary in the Ragdoll workspace
- **`due`**: date, YYYY-MM-DD — Optional target date (YYYY-MM-DD); releases are gated by verification evidence
- **`part_of`**: names any items, by id, several allowed — Parent epic or subsystem item this work rolls up into
- **Milestones**: `m1-foundation`, `m2-pix`, `m3-format`, `m4-codec`, `m5-cli-bench`
- **Saved views** (`cairn list --view NAME`): `now`, `next`, `dependencies`, `kernels`, `benchmarks`, `decisions`, `triage`, `history`

### Rules

1. Before starting work, find or create the item and set it to an active status.
2. Use the fields above rather than inventing new ones; add new fields to `cairn.toml` first.
3. Never hand-edit the generated roadmap file — change items and run `cairn render`.
4. `cairn check` must pass before the work is considered done.

<!-- cairn:end -->
