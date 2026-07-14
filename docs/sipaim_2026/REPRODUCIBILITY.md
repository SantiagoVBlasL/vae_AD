# SIPAIM 2026 transportability study — reproducibility guide

Practical steps to regenerate any result in `results/sipaim_2026/` from
scratch, or to verify the frozen snapshot matches what is on disk today.
This snapshot itself contains no checkpoints or raw data (see
`DATA_MANIFEST.md`); this guide is for someone with access to the original
machine/data who wants to rerun something.

## 0. Environment

- Conda environment: `vae_ad`. Activate it (e.g. `conda activate vae_ad`)
  before invoking any command below as `python`; on a machine with multiple
  conflicting environments, invoke the environment's interpreter explicitly
  instead of a bare `python` on `$PATH`.
- Key dependency for ComBat: `neurocombat_sklearn.CombatModel` (a
  compatibility shim for its `OneHotEncoder` usage against newer
  scikit-learn is provided in `foldwise_combat_input_harmonization.py`).
- TDA libraries (`ripser`, `gudhi`, `giotto-tda`, `kmapper`) are **not**
  installed in this or any other checked environment as of this freeze —
  relevant only if someone picks up the blocked topology analysis (see
  `ANALYSIS_DECISIONS.md` §7).

## 1. Regenerating the clean fold-wise ComBat VAE retrain (the central experiment)

This is a multi-hour GPU training job (5 outer folds, beta=3.75,
latent=384, up to 10,000 epochs each with early stopping/patience=560).

```bash
cd ${VAE_AD_REPO_ROOT}   # absolute path to a local clone of this repository
git checkout sipaim/transportability-2026   # or note the frozen git hash below

# Preflight only (validates config/paths against the locked reference,
# does not train):
python scripts/sipaim_2026/run_adni_v5_1c_foldcombat_cleanrepro_20260706.py --dry-run

# Full guarded launch (trains; refuses to run if output dirs already exist
# or if any config field silently diverges from the locked reference
# beyond the intended ComBat-harmonization fields):
bash scripts/sipaim_2026/guarded_launch_foldcombat_cleanrepro_20260706.sh --confirm-training

# Poll status without touching anything:
bash scripts/sipaim_2026/monitor_foldcombat_cleanrepro_20260706.sh
```

Config: `configs/sipaim_2026/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706.json`.
Training entry point invoked internally:
`scripts/sipaim_2026/run_vae_clf_ad_inference.py`.

**Exact git commit this was trained against**: `dc77722464cdcdb61f7603e069af2e15103d7aea`
(recorded in the training run's own `provenance/git_head.txt`, which also
stores a full source-code snapshot of `scripts/`, `src/`, and `configs/` as
they existed at training time — the safest way to verify byte-for-byte
reproducibility is to diff against that snapshot, not just check out the
commit, since a few relevant files were uncommitted at training time and
only exist as this snapshot's `scripts/sipaim_2026/*.py` copies).

Downstream frozen classifier (readout-only, no VAE retraining):

```bash
python scripts/sipaim_2026/run_foldcombat_cleanrepro_downstream_classifier.py --run-dir <training_output_dir>
```

Final audit (all manuscript numbers):

```bash
python scripts/sipaim_2026/audit_foldcombat_cleanrepro_final_20260706.py
```

## 2. Regenerating the site/manufacturer decoding, transport, LOSO, and Wasserstein packages

These are read-only with respect to the trained models (they consume
existing frozen latent caches and predictions; none of them retrain a VAE
or refit ComBat):

```bash
python scripts/sipaim_2026/build_sipaim_multisite_transport_20260706.py
python scripts/sipaim_2026/reconcile_sipaim_metrics_20260706.py
python scripts/sipaim_2026/run_wasserstein_analysis.py
python scripts/sipaim_2026/run_foldlocal_geometry.py
python scripts/sipaim_2026/run_latent_site_geometry.py
```

Each script's own docstring states its precise read-only guardrails (no
VAE training, no OASIS fitting/calibration, no manuscript edits). Consult
`METHODS_PROVENANCE.md` for the fold-safety contract they all share.

## 3. Regenerating the figure

```bash
python scripts/sipaim_2026/generate_fig_transportability_summary_20260706.py
```

Reads only from `results/revision_bspc_2026/post_revision_exploratory_20260630/foldcombat_cleanrepro_final_audit_20260706/`
(the four CSVs listed in the script header) and writes
`docs/revision_bspc_2026/site_geometry_analysis/Figures/site_geometry/fig_transportability_summary.pdf`
(also preserved at `manuscript/sipaim_2026/Figures/site_geometry/fig_transportability_summary.pdf`
and `figures/sipaim_2026/fig_transportability_summary.pdf` in this
snapshot). Output is a vector PDF with embedded/subsetted fonts, validated
to contain zero raster images.

## 4. The manuscript itself

Overleaf is the sole editable source of the manuscript — see
`MANUSCRIPT_SOURCE_OF_TRUTH.md`. This repository does not track a
canonical manuscript `.tex`/PDF to recompile; any local `.tex`/PDF copies
that may exist on a given machine are, at best, historical export
snapshots and must not be treated as buildable or authoritative.

## 5. Verifying this snapshot hasn't drifted from source

```bash
sha256sum -c <(awk -F, 'NR>1{print $1"  "$3}' artifact_manifest_sha256.csv)
```

(Run from the repo root; `artifact_manifest_sha256.csv` lists every file
under the six `*/sipaim_2026/` directories with its sha256 and byte size.)

## 6. What you cannot reproduce from this repo alone

- The ADNI/OASIS raw tensors and metadata (licensed; see `DATA_MANIFEST.md`
  for their checksums so you can verify you have the right files once you
  have separately obtained them).
- The 793 MB of VAE checkpoints/training histories for the cleanrepro run
  (git-ignored; regenerate via §1 above, or request from whoever holds the
  original run directory — `results/revision_bspc_2026/recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706/`
  on the machine this was produced on).
- The topology/persistent-homology analysis (blocked; see
  `ANALYSIS_DECISIONS.md` §7 — install `ripser` or `gudhi` first).
