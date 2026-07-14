# SIPAIM 2026 transportability study — methods provenance

This document explains **how** each family of results was produced, the
fold-safety contract common to all of them, and the chronology that led to
the "clean" (`cleanrepro`) run being the canonical source for most
ComBat-related claims. For *what* is canonical vs. superseded, see
`ANALYSIS_DECISIONS.md`. For a one-row-per-analysis index, see
`experiment_registry.csv`.

## The two models being compared throughout

1. **Locked (non-harmonized) reference**: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`.
   beta=3.75, latent_dim=384, channels=[1,0,2]
   (`Pearson_Full_FisherZ_Signed`, `Pearson_OMST_GCE_Signed_Weighted`,
   `MI_KNN_Symmetric`), 5x5 nested CV, logistic-regression L2 readout on
   latent mu + Age + Sex. This is the frozen submitted BSPC model; it is
   never retrained or modified by any analysis in this snapshot.
2. **Fold-wise input-ComBat retrain**: same architecture/hyperparameters as
   (1), with one controlled change — `input_harmonization_mode=foldwise_combat`
   wired directly into the VAE training loop
   (`scripts/sipaim_2026/run_vae_clf_ad_inference.py`). For each outer fold:
   a channel-wise ComBat model (`neurocombat_sklearn.CombatModel`,
   batch=Manufacturer, discrete covariate=Sex, continuous covariate=Age) is
   fit on that fold's VAE train-dev pool only, then the **VAE trains on the
   harmonized tensor**, not the raw one. Classifier train/dev and outer-test
   tensors are transformed with the same frozen fitted ComBat model
   (transform-only, no refit). Diagnosis (`ResearchGroup_Mapped`) is
   explicitly excluded from the harmonizer's covariates and this exclusion
   is asserted in code, not just documented.

## Fold-safety contract (applies to every ComBat-related result in this snapshot)

Every per-fold ComBat fit/transform in this codebase is required to satisfy,
and is checked against:

- `fit_scope = outer_train_dev_only` — the harmonizer never sees that
  fold's held-out outer-test subjects, and never sees OASIS.
- `n_fit_test_subject_overlap = 0` — asserted per fold.
- `diagnosis_used_in_harmonizer = False` — asserted per fold.
- `oasis_used = False` — asserted per fold (relevant to the frozen OASIS
  inference experiment).
- `global_combat = False` — the harmonizer is never fit on the pooled
  (cross-fold) dataset.

These are written per fold as `input_harmonization_leakage_guard.csv` /
`input_harmonization_integrity.csv` inside the (git-ignored) training run
directory, and re-asserted by `audit_foldcombat_cleanrepro_final_20260706.py`
(fold-safety verdict: **FOLD_SAFE**, recorded in
`results/sipaim_2026/clean_foldwise_combat_training_and_audit/00_FINAL_CLEANREPRO_REPORT.md`).

## Chronology: why "cleanrepro" (2026-07-06) is the canonical ComBat run

1. **2026-06-02** — a *post-encoder* correction was attempted
   (`promoted_beta3p75_stageB_latent_harmonization_by_manufacturer_20260602`,
   not part of this snapshot): real ComBat was skipped because
   `neuroCombatFromTraining` lacked held-out covariate support; an OLS
   Manufacturer-dummy residualization ran instead, on the *already-trained*
   locked model's latents. This does not answer the "retrain on harmonized
   inputs" question and is out of scope for this snapshot.
2. **2026-06-07** — the first genuine fold-safe input-ComBat retrain ran to
   completion
   (`recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5`,
   config `configs/sipaim_2026/superseded_original_run_20260607_...json`).
   Its downstream classifier-readout latent cache had an **export-lineage
   bug**: it was re-inferred from raw (non-harmonized) inputs instead of the
   harmonized ones actually used to train the VAE. Analyses run against that
   defective cache (e.g. `foldcombat_mfr_age_sex_audit_20260705`) are
   superseded and are **not** included in this snapshot.
3. **2026-07-06 (correction)** — the export bug was found and corrected by
   re-running the already-trained (frozen) encoder on correctly harmonized
   inputs (`combat_external_transport_completion_20260706`,
   `combat_downstream_classifier_reconstruction_20260706`). This produced
   valid but *reconstructed* (not freshly trained end-to-end in one script
   run) numbers.
4. **2026-07-06 (cleanrepro)** — a fresh, from-scratch, single-shot
   reproduction was run using
   `scripts/sipaim_2026/run_adni_v5_1c_foldcombat_cleanrepro_20260706.py`
   (config `configs/sipaim_2026/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706.json`),
   audited by `audit_foldcombat_cleanrepro_final_20260706.py`
   (git commit `dc77722464cdcdb61f7603e069af2e15103d7aea` recorded in the
   run's own `provenance/git_head.txt`). Verdict:
   **PASS_MANUSCRIPT_GRADE_AUDIT**, fold-safety **FOLD_SAFE**. Its training
   reproducibility check (`cleanrepro_training_reproducibility.csv`) confirms
   the selected/terminal checkpoint epoch matches the June run exactly for
   all 5 folds, i.e. the fresh run is a faithful reproduction, not a
   different model. **This is the run this snapshot treats as canonical**
   for manufacturer decoding, transport, Wasserstein, and OASIS numbers.

5. **2026-07-10 (external OASIS Dataset-ComBat)** — a distinct, separate
   analysis (`oasis_external_batch_combat_20260710`, not a continuation of
   the cleanrepro chronology above) tested an *external Dataset-level*
   ComBat (batch $\in$ {ADNI, OASIS}) applied to raw connectivity features
   ahead of the same frozen locked VAE, to see whether it improves
   ADNI-to-OASIS transport. See the dedicated section below.

## External OASIS Dataset-ComBat adaptation (2026-07-10)

This is a **different harmonization axis** from the fold-wise
Manufacturer-ComBat retrain described above: batch = `Dataset`
(ADNI vs. OASIS), not `Manufacturer`, and OASIS itself is one of the two
batch levels. It answers a different question — "does harmonizing the
external dataset's feature distribution toward ADNI improve transport?" —
not "does removing manufacturer signal from the latent space cost
accuracy?".

- **Script**: `scripts/sipaim_2026/run_oasis_external_batch_combat_20260710.py`.
- **Harmonizer**: `neuroCombat.neuroCombat` (not `neurocombat_sklearn`, which
  lacks reference-batch support in this environment), covariates Age + Sex,
  Diagnosis/`ResearchGroup_Mapped` excluded from the covariate frame.
- **Two variants**: no reference batch, and ADNI as reference batch
  (`ref_batch` argument) — the ADNI-reference variant is judged the
  methodologically cleaner one since it targets the VAE's original training
  domain (see `oasis_external_batch_combat_limitations.md`).
- **VAE**: never retrained; the harmonized features are only ever passed
  through the already-frozen locked encoder.
- **Downstream readout**: reconstructed from ADNI train/dev latents only,
  with the locked inner-OOF-ECDF threshold convention applied to OASIS —
  the same convention `frozen_oasis_inference` uses, since the official
  locked downstream estimators were never serialized as frozen objects.
- **Guardrails** (asserted in `oasis_external_batch_combat_protocol_audit.md`
  and `command_log.json`): OASIS diagnosis labels never enter the ComBat
  covariate frame, calibration, threshold selection, or model selection;
  no VAE checkpoint is trained or updated.
- **Important caveat this analysis is NOT fold-safe in the same sense as
  above**: it is transductive/post-hoc — the harmonizer's covariate frame
  includes the OASIS *feature* distribution (unsupervised: Age/Sex/connectivity
  only, never Diagnosis), so it is not a held-out-transform-only design the
  way the fold-wise Manufacturer-ComBat retrain is. This is documented, not
  hidden, and is why the finding is reported as "feature-space alignment
  improves, ranking does not" rather than a transport fix.
- **Result summary**: mean all-feature Wasserstein (W1) distance to OASIS
  drops from 0.230 (raw) to 0.154 (no reference) / 0.128 (ADNI reference)
  across all 5 folds, but OASIS ROC-AUC/PR-AUC are statistically unchanged
  from the locked frozen-transfer baseline (10,000-sample paired bootstrap,
  all pairwise arm differences n.s., $p \in [0.33, 0.91]$). The sensitivity
  increase / specificity decrease seen in the Dataset-ComBat arms is a
  fixed-threshold operating-point shift, not evidence of improved ranking.

## Git-hash tracking gap (be aware of this when reading `experiment_registry.csv`)

Most of the `command_log.json` files across this project record guardrail
booleans (`did_train_vae`, `oasis_used`, etc.) and input/output paths, but
**not** a git commit hash — this is a project-wide convention gap, not
specific to this snapshot. The two places a git hash *is* recorded are:
- `.../recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706/provenance/git_head.txt` = `dc77722464cdcdb61f7603e069af2e15103d7aea` (the training run itself carries a full source-code snapshot alongside this hash).
- This snapshot's own new work (this branch).

Where `experiment_registry.csv` cannot cite a specific commit for an
analysis script, it says so explicitly rather than guessing.

## Statistical protocols used repeatedly across analyses

- **Paired bootstrap** (locked vs. ComBat, same subjects): 10,000
  resamples, diagnosis-stratified, seed `20260706`.
- **Permutation nulls** (manufacturer/site decodability): 1,000
  permutations, seed `42`, preserving `ResearchGroup_Mapped + Manufacturer`
  strata where applicable.
- **Wasserstein estimators**: all fit fold-locally on that fold's train-dev
  pool only (1D sliced W1 across 384 dims, PCA10-W1, closed-form Gaussian
  Bures-Wasserstein); no cross-fold pooling of raw latent coordinates.
- **"Supported site" gates** (used for site-level AUC and LOSO): a site
  must have supervised N>=15 with CN>=3 and AD>=3 to get a site-level AUC
  estimate; N>=20 with CN>=5 and AD>=5 to be LOSO-eligible. Only 2 of ~51
  ADNI sites (130, 035) meet the LOSO bar.
