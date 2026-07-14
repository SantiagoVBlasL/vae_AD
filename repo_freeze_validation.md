# Repo freeze validation — SIPAIM 2026 transportability snapshot

Branch `sipaim/transportability-2026`. All checks below were actually run
against the working tree produced by this freeze (not asserted from
memory); commands are included so they can be re-run to confirm.

## 1. No file over 1 MB in any new directory — PASS

```
find manuscript/sipaim_2026 figures/sipaim_2026 configs/sipaim_2026 \
     scripts/sipaim_2026 results/sipaim_2026 -type f -size +1M
```
Zero results. Largest individual file is `cleanrepro_oasis_predictions.csv`
at 372 KB. Total across all five directories: **≈3.1 MB**.

## 2. No forbidden binary extensions — PASS

```
find manuscript/sipaim_2026 figures/sipaim_2026 configs/sipaim_2026 \
     scripts/sipaim_2026 results/sipaim_2026 -type f \
     \( -iname "*.npz" -o -iname "*.npy" -o -iname "*.pt" -o -iname "*.pth" \
        -o -iname "*.ckpt" -o -iname "*.joblib" -o -iname "*.pkl" \
        -o -iname "*.h5" -o -iname "*.hdf5" -o -iname "*.nii*" \)
```
Zero results. Every file is `.md`, `.csv`, `.json`, `.py`, `.sh`, `.tex`, or
`.pdf`.

## 3. Checksum manifest self-consistency — PASS

```
sha256sum -c <(awk -F, 'NR>1{print $1"  "$3}' artifact_manifest_sha256.csv)
```
All 136 listed files report `OK` (Spanish locale: "La suma coincide").
136 files manifested; `find ... -type f | wc -l` over the same five
directories independently counts 136 — no file was missed or duplicated in
the manifest.

## 4. Every `experiment_registry.csv` artifact path resolves — PASS (after one fix)

Initial automated check flagged a false positive caused by an embedded
semicolon inside a free-text note in the `topology_persistent_homology`
row's `artifact_paths` field (the validation script's naive
`split(";")` cut the sentence in half). Fixed by simplifying that field to
the bare path `topology_existing_artifact_audit.md`. Re-run after the fix:
all 9 rows' `artifact_paths`, and all rows' `config_path`/`script_path`
entries (excluding literal `N/A`), resolve to existing files.

## 5. Every `results/sipaim_2026/` subdirectory is covered by exactly one registry row — PASS

Cross-checked the 7 subdirectories present in `artifact_manifest_sha256.csv`
under `results/sipaim_2026/` against `experiment_registry.csv`:
`manufacturer_decoding`, `site_decoding_permutation`,
`manufacturer_transport_and_supported_sites`, `classifier_only_loso`,
`frozen_oasis_inference`, `adni_oasis_wasserstein`,
`clean_foldwise_combat_training_and_audit` — all 7 map to a registry row's
`artifact_paths`. The 2 remaining registry rows
(`paired_bootstrap_locked_vs_combat`, `topology_persistent_homology`) point
to specific files inside two of the above directories, and to the
top-level audit doc, respectively — not missing, just not a distinct
subdirectory of their own (documented in `results/sipaim_2026/README.md`).

## 6. `.gitignore` behaves as intended — PASS

- `results/sipaim_2026/` (and its contents) are **not** ignored, despite
  the pre-existing blanket `results/**` rule, due to the new
  `!results/sipaim_2026/` / `!results/sipaim_2026/**` exception added by
  this freeze. Verified: `git status --porcelain results/sipaim_2026/`
  shows it as untracked (`??`), i.e. addable — not silently dropped.
- A synthetic test file at `results/sipaim_2026/latent_cache/fold_1_test.csv`
  and `results/sipaim_2026/fold_1_oasis_fold_latent_mu.csv` **were**
  correctly re-excluded by the new `**/latent_cache/` and
  `**/*_fold_latent_mu.csv` rules even inside the allowed path (tested with
  `git check-ignore` / `git add --dry-run`, then removed — these were only
  test fixtures, not part of the freeze).
- `scripts/sipaim_2026/`, `configs/sipaim_2026/`, `manuscript/sipaim_2026/`,
  `figures/sipaim_2026/`, `docs/sipaim_2026/` were never covered by any
  blanket ignore rule in the first place (only `results/**`, `data/**`, and
  `artifacts/` have blanket rules) — confirmed all five show as plain
  untracked (`??`), addable without any exception needed.

## 7. `fig_transportability_summary.pdf` remains vector-only — PASS

`pdfimages -list manuscript/sipaim_2026/Figures/site_geometry/fig_transportability_summary.pdf`
returns zero embedded raster images (same file already validated when
originally generated; re-checked here since it was copied into three
locations by this freeze — `docs/revision_bspc_2026/.../Figures/site_geometry/`,
`manuscript/sipaim_2026/Figures/site_geometry/`, and `figures/sipaim_2026/`
— to confirm the copy operation didn't alter it. sha256 of all three copies
is identical per `artifact_manifest_sha256.csv` cross-checked against the
original.)

## 8. Branch and working-tree safety — PASS

- Current branch: `sipaim/transportability-2026` (confirmed via
  `git branch --show-current`), created from
  `exploratory/post-revision-20260630`.
- No existing tracked file was deleted or renamed in place (only new files
  added, plus additive edits to `.gitignore` — verified no existing
  `.gitignore` lines were removed, only appended; see
  `proposed_commit_plan.md` for the exact diff).
- No `git commit`, `git push`, or history-rewriting command has been run.
  Two pre-existing sets of uncommitted changes from earlier work this
  session remain exactly as they were
  (`scripts/run_vae_clf_ad_ablation.py`, `scripts/run_vae_clf_ad_inference.py`
  — the latter's *content* was copied, unmodified, into
  `scripts/sipaim_2026/run_vae_clf_ad_inference.py` for the snapshot, but
  the original file at `scripts/run_vae_clf_ad_inference.py` itself was not
  touched by this freeze).

## 9. Reproducibility source-snapshot cross-check — PASS

`scripts/sipaim_2026/run_vae_clf_ad_inference.py` (copied into this
snapshot) was diffed against
`results/revision_bspc_2026/recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706/provenance/source_snapshot/scripts/run_vae_clf_ad_inference.py`
— the actual frozen copy of the script saved by the training run itself at
training time. Result: **byte-identical** (`diff -q` reports no
difference). This confirms the copy in `scripts/sipaim_2026/` is exactly
what produced the canonical results, not a possibly-drifted working-tree
version.

## Known gaps (not fixed by this freeze, flagged for the record)

- Most `command_log.json` files across the 9 categories do not record a
  git commit hash (project-wide convention gap, see
  `docs/sipaim_2026/METHODS_PROVENANCE.md`). Only the cleanrepro training
  run itself carries one (`dc77722464cdcdb61f7603e069af2e15103d7aea`).
- `data_fingerprint` for OASIS-based experiments in `experiment_registry.csv`
  is a build-identifier description (`runwise164_pilot_parity`, N=180), not
  a computed sha256 — the OASIS tensor's location/hash was not tracked
  down as part of this freeze (out of scope; flagged as a possible future
  improvement in `docs/sipaim_2026/REPRODUCIBILITY.md`).
