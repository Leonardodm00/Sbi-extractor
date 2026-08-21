# artifacts/

Local, machine-specific state for this repo. Gitignored -- checkpoints are
large binaries, and the `dsn_main` symlink target is an absolute cluster
path -- except this file, which is tracked so a fresh clone documents what
belongs here and how to reproduce it.

Populated by `relocate_artifacts.sh` (moves the three items below out of
`$HOME` into here) or, on a fresh checkout with nothing to migrate, by hand
per the notes in each section.

## frozen_dsn/

Copy (not symlink) of the frozen DSN checkpoint currently in use, plus a
`.sha256` file beside it. See `HANDOFF.md` sections 2-3 for provenance
(source path, training run) and the settled model constants read from its
config (window length, embedding dim, etc.).

Never symlinked, per the frozen-artifact pattern in `HANDOFF.md` section 7:
a training checkpoint is still being written, so only a `cp` actually
freezes it. To add one by hand:

```bash
cp "/path/to/out/refit_mea_B_rank/checkpoints/seed_0/best.pt" \
   artifacts/frozen_dsn/dsn_B_rank_<date>.pt
sha256sum artifacts/frozen_dsn/dsn_B_rank_<date>.pt \
   > artifacts/frozen_dsn/dsn_B_rank_<date>.pt.sha256
```

## specs_real.json

Copy of the real-cohort specs, resolved from a checkpoint's own
`config.data.npz_specs` pointer (`hpc/Config/npz_specs_mea.json`, a path
relative to `dsn_main`, not to `$HOME` or the CWD). 315 entries, 35
cultures, keys `path`/`name`/`condition`/`culture`. Regenerate by
re-resolving that pointer against a checkpoint's config if this file is
ever lost -- do not hand-edit it.

## dsn_main

Symlink to the `Deep-Summary-Network` repo's `Main` directory -- a
separate, actively developed repo, never copied here, only pointed at.
The symlink exists because the real path contains a space
(`"Deep Summary Network/Deep_bio/Main"`). Recreate with:

```bash
ln -s "/path/to/Deep Summary Network/Deep_bio/Main" artifacts/dsn_main
```
