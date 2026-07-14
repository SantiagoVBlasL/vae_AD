# Frozen-score component audit: latent / Age / Sex decomposition

Read-only. No OASIS-based fitting, no calibration, no threshold selection. Full per-subject decomposition table: `score_component_audit.csv` (ensemble = mean across the 5 fold-specific decompositions, one raw logit component set per fold in `_score_component_decomposition_by_fold.csv`).

## Missing-artifact finding (why this required a reconstruction step)

The fitted Stage-B classifier and OOF-ECDF transformer objects used to produce the canonical OASIS scores were **never persisted to disk** -- only their resulting CSV metrics were saved (confirmed: `results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration/` contains no `.joblib` files). This is the exact missing artifact.

The persisted artifacts that DO exist on disk are the **Stage-A** `classifier_logreg_final_pipeline_fold_N.joblib` pipelines. Loading these directly and scoring OASIS latents with zero fitting does **not** reproduce the canonical `y_score_raw` values used throughout this project (fold-by-fold correlation with the canonical score: fold 1: r=0.823, fold 2: r=0.992, fold 3: r=0.994, fold 4: r=0.992, fold 5: r=0.997 -- fold 1 in particular diverges substantially). Full detail: `_persisted_pipeline_mismatch_check.csv`.

## What this audit did instead

Reconstructed each fold's Stage-B classifier deterministically, using **only** the already-frozen ADNI train/dev latent cache (`classifier_only_readout/latent_cache/fold_N_trainDev_latent_mu.csv`), the exact fixed hyperparameter grid, fixed random seeds, and fixed inner-CV split recipe already established as canonical in `scripts/revision_bspc_2026/score_oasis_mega_90_90_external_inference_model_panel_20260604.py`. No OASIS data was used to fit anything; no new hyperparameter choice was made. This reconstruction reproduces the canonical `y_score_raw` and `y_score` to floating-point precision (max abs diff <5e-8 across all 5 folds; full detail in `_stageB_reconstruction_verification.csv`), which both validates the reconstruction and independently validates that the OASIS locked-arm latent cache used throughout this and the geometry audit (`promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/`) is the correct, non-defective source.

The classifier is exactly linear in its (scaled) inputs (`logit = latent_component + age_component + sex_component + intercept`, verified to 1e-15 precision against `predict_proba`), so this decomposition is exact given the reconstructed coefficients -- not an approximation.

## CDR association for the latent-only component

n = 180 subjects (Task-1 mapping B: nearest CDR within 365 days; all 180 map).

| Quantity | Spearman rho | p-value |
|---|---|---|
| Latent component only, raw | 0.2966 | 5.28e-05 |
| Latent component only, Age/Sex-adjusted (partial) | 0.2590 | 4.47e-04 |
| Full reconstructed logit (latent+age+sex+intercept) | 0.2940 | 6.18e-05 |
| Age component only | -0.1636 | 2.82e-02 |

## Reading

The CDR-severity association survives isolation to the **latent component alone** (rho=0.297 raw, rho=0.259 after removing Age and Sex) and is barely attenuated relative to the full logit (rho=0.294) -- the CDR-severity signal found throughout this project's OASIS analyses lives substantively in the latent representation itself, not merely in the classifier's explicit Age/Sex terms. The age component alone has a small, marginally significant **negative** association with CDR (rho=-0.164, p=0.028) -- a secondary, minor effect not large enough to explain the latent-component finding and not further interpreted here.
