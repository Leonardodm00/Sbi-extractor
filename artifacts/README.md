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

## sbi_hpc

[migration step 3, 2026-09-19] Symlink to the `hpc/` directory of the
`Simulation-Based-Inference` clone. Since step 1 the DSN lives there, at
`hpc/dsn` (a byte-identical mirror of the retired repo's `Main/`, see its
`README.md` and `ORIGIN_MANIFEST.tsv`). `env.sh` defaults `SBI_HPC_DIR` to
this symlink and `dsn_tree.py` resolves the DSN tree as `$SBI_HPC_DIR/dsn`.
Recreate with:

```bash
ln -s ~/SBI/hpc artifacts/sbi_hpc
```

## dsn_main (retired)

Nothing reads it since migration step 4 (2026-09-19); `relocate_artifacts.sh`
removes it if present. The DSN is reached through `sbi_hpc` above.
