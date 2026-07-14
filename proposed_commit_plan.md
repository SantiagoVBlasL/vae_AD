# Proposed commit plan — SIPAIM 2026 transportability snapshot

**Nothing has been committed.** Everything below is staged
(`git add`) on branch `sipaim/transportability-2026` and ready for review.
No push, no history rewrite, no deletion of any existing file.

## Proposed commit

```
git commit -m "Freeze SIPAIM 2026 transportability snapshot: manufacturer/site
decoding, transport, LOSO, frozen OASIS, Wasserstein, and the clean
fold-wise input-ComBat retrain + audit"
```

### Stat

```
147 files changed, 22214 insertions(+)
```

### Scope of this commit (what IS included)

| Area | Files | Notes |
|---|---|---|
| `.gitignore` | 1 modified | Purely additive (37 new lines appended, nothing removed) — see diff in `repo_freeze_validation.md` §6. |
| `manuscript/sipaim_2026/` | 11 new | Compact manuscript (.tex + compiled .pdf) + its 8 figure dependencies + a figure-reference audit note. |
| `figures/sipaim_2026/` | 20 new | 8 manuscript figures (flat copies for standalone reuse) + 6 supplementary SIPAIM figures + their 6 source CSVs. |
| `configs/sipaim_2026/` | 3 new | Locked reference, clean ComBat retrain, and superseded-original-run configs. |
| `scripts/sipaim_2026/` | 17 new | Every script that produced a canonical result copied into `results/sipaim_2026/`, plus the core training entry point and its calibration dependency. |
| `docs/sipaim_2026/` | 4 new | `DATA_MANIFEST.md`, `METHODS_PROVENANCE.md`, `ANALYSIS_DECISIONS.md`, `REPRODUCIBILITY.md`. |
| `results/sipaim_2026/` | ~90 new | 7 topic subdirectories + `README.md`; ≈1.4 MB total, nothing over 400 KB. |
| Repo root | 5 new | `experiment_registry.csv`, `artifact_manifest_sha256.csv`, `repo_freeze_inventory.md`, `repo_freeze_validation.md`, `topology_existing_artifact_audit.md`. |

Every file in this commit was validated per `repo_freeze_validation.md`:
no file over 1 MB, no forbidden binary extension, sha256 manifest
self-consistent, every registry path resolves, `.gitignore` exceptions
tested to work as intended.

### Deliberately EXCLUDED from this commit (left as-is, not deleted)

1. **Two pre-existing uncommitted modifications from earlier work this
   session**, unrelated to the SIPAIM freeze itself:
   - `scripts/run_vae_clf_ad_ablation.py` (ComBat wiring added for a
     separate FAST-scale pilot — out of this freeze's scope per
     `docs/sipaim_2026/ANALYSIS_DECISIONS.md` §9).
   - `scripts/run_vae_clf_ad_inference.py` (the working-tree version is
     confirmed byte-identical to the copy frozen in this commit at
     `scripts/sipaim_2026/run_vae_clf_ad_inference.py` — see
     `repo_freeze_validation.md` §9 — so nothing is lost by leaving the
     original file's modified-but-uncommitted state untouched here).
2. **A dozen pre-existing untracked files from earlier sessions**
   (`configs/runs/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706.json`,
   `docs/revision_bspc_2026/site_geometry_analysis/` (the *original*
   location of the manuscript/figures — this commit copies from there into
   `manuscript/sipaim_2026/` and `figures/sipaim_2026/` but does not touch
   or commit the original), and 11 scripts under
   `scripts/revision_bspc_2026/`). These are the **source** files this
   freeze read from; committing copies of the relevant ones into
   `scripts/sipaim_2026/` was judged sufficient for a clean, reviewable
   SIPAIM-scoped commit. **This is a scoping decision, not an oversight** —
   if you'd prefer these original loose files also committed (e.g. to keep
   `scripts/revision_bspc_2026/` itself fully version-controlled going
   forward, independent of the SIPAIM curation), that is a separate,
   easy follow-up commit:
   ```
   git add configs/runs/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706.json \
           docs/revision_bspc_2026/site_geometry_analysis/ \
           scripts/revision_bspc_2026/*.py scripts/revision_bspc_2026/*.sh
   ```
   Not included in the commit proposed here, to keep this freeze's diff
   scoped to exactly what the task asked for.
3. **Everything under `results/revision_bspc_2026/`** (the original,
   uncurated result directories this freeze read from) — untouched,
   already covered by the pre-existing blanket `results/**` gitignore
   rule, not part of this or any commit.
4. **`site_geometry_topology_20260705/`** — audited, contains no
   analysis output to preserve (see `topology_existing_artifact_audit.md`);
   correctly not copied anywhere.

### Order of operations if you approve this commit

1. Review the staged diff (`git diff --cached`) and this plan.
2. `git commit` with the message above (or your own).
3. Do **not** push without separately deciding to — this plan does not
   assume `origin` should receive this branch yet.
4. If you also want the "deliberately excluded" pre-existing loose files
   committed, do that as a **second, separate commit** (not squashed into
   this one) so the SIPAIM-freeze commit stays cleanly reviewable on its
   own.

### What happens if you do nothing

Nothing — the working tree stays exactly as staged. `git status` will
continue to show these 147 files as staged (`A`/`M`) until either committed
or unstaged (`git restore --staged <path>`).
