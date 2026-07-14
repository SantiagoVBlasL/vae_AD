# SIPAIM 2026 transportability study — data manifest

Branch: `sipaim/transportability-2026` (created from `exploratory/post-revision-20260630`).
This document describes the **inputs** to the frozen snapshot under
`manuscript/sipaim_2026/`, `figures/sipaim_2026/`, `configs/sipaim_2026/`,
`scripts/sipaim_2026/`, `results/sipaim_2026/`, `docs/sipaim_2026/`. It does
not describe the results themselves (see `results/sipaim_2026/README.md` and
`experiment_registry.csv`) or method rationale (see `METHODS_PROVENANCE.md`).

## Primary licensed data (NOT copied into this repo, referenced only)

| Dataset | Location (outside repo tracking) | Fingerprint |
|---|---|---|
| ADNI global connectivity tensor | `${VAE_AD_DATA_ROOT}/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz` (`VAE_AD_DATA_ROOT` = user-supplied absolute path to the restricted-access data root) | sha256 `f9a00b291a88d92d942ee3404fe0f5b1cfabe5178cf8ff6c139d57658cb8f609`, 216,322,727 bytes |
| ADNI metadata (patched) | `results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv` (git-ignored, lightweight CSV but excluded by the blanket `results/**` rule — see REPRODUCIBILITY.md for how to obtain it) | sha256 `bb21a389ab5ffa0e6a211a6603d6e05502edc37b71af9cb26a0059c19c4d228d`, 212,341 bytes |
| ADNI analysis cohort | N=397 CN/AD subjects (300 CN, 97 AD) from the tensor+metadata above | — |
| OASIS external validation cohort | build `runwise164_pilot_parity`, N=180 (90 CN, 90 AD), one row per subject/session ensemble | see `results/sipaim_2026/frozen_oasis_inference/oasis_provenance.md` for the full build-selection rationale |

**Neither the ADNI nor the OASIS raw connectivity tensors, nor any
subject-level neuroimaging derivative, are copied anywhere under this
branch's new directories.** Both are licensed and must be obtained
separately by anyone reproducing this work. This manifest records their
checksums so a reproduction attempt can verify it is starting from the
identical tensor/metadata pair.

## What IS copied into this snapshot (canonical, lightweight only)

All files under `manuscript/sipaim_2026/`, `figures/sipaim_2026/`,
`configs/sipaim_2026/`, `scripts/sipaim_2026/`, `results/sipaim_2026/` are:

- Reports (`.md`), tables (`.csv`), machine-readable summaries (`.json`), or
  vector figures (`.pdf`) — never raw tensors, checkpoints, or full
  per-subject latent-vector dumps.
- Individually under 400 KB; the entire snapshot totals **≈3.1 MB** across
  all six directories (see `repo_freeze_inventory.md` for the itemized
  breakdown and `artifact_manifest_sha256.csv` for a checksum of every
  individual file).
- All traceable to a source directory under `results/revision_bspc_2026/`
  and/or `scripts/revision_bspc_2026/` (documented per-file in
  `repo_freeze_inventory.md`).

## What was explicitly excluded (see `.gitignore` for the enforcing rules)

| Category | Example | Why excluded |
|---|---|---|
| VAE checkpoints / training histories | `recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706/fold_{1..5}/vae_model_fold_*.pt` (793 MB total, lives inside the repo's `results/` tree but is git-ignored) | Large binary, fully regeneratable from the frozen config + git hash |
| Full per-subject latent-vector dumps | `cleanrepro_oasis_fold_latent_mu.csv` (3.8 MB), `taskB_combat_oasis_fold_latent_mu_runwise164.csv` (3.7 MB), `oasis_mega_90_90_.../predictions.csv` (6.0 MB), `corrected_combat_latent_cache/` (8.2 MB) | Not raw data, not licensed, but bulky derived intermediates outside the "lightweight" bar; every number computed from them is already reported in the small CSVs that *are* copied |
| ADNI/OASIS raw connectivity tensors | `GLOBAL_TENSOR_*.npz`, `subject_tensors/` | Licensed data |
| Launch/monitor logs | `_launch_logs/`, `*.log` | Ephemeral, machine-specific, not needed for scientific reproducibility |

## Cross-reference

- Per-file provenance (which script produced which result file, and what
  its status is relative to earlier/superseded iterations):
  `repo_freeze_inventory.md`.
- Which analyses are canonical vs. superseded, and why:
  `ANALYSIS_DECISIONS.md`.
- Step-by-step instructions to actually rerun any of this from the tensor
  above: `REPRODUCIBILITY.md`.
