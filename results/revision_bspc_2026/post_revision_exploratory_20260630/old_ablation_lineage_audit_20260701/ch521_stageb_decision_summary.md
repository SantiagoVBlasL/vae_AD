# ch521 [5,2,1] Stage B Decision Summary

**Audit date:** 2026-07-01
**Auditor:** claude-sonnet-4-6 (read-only)

---

## Candidate

**ch521 [5,2,1]** = DistanceCorr (ch5) + MI_KNN_Symmetric (ch2) + Pearson_Full_FisherZ_Signed (ch1)

- Run name: `recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_meta647_20260622`
- Architecture: identical to reference [1,0,2] except channels_to_use = [5, 2, 1]
- Big disk path: `/media/diego/Datos/vae_AD_results/revision_bspc_2026/recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_meta647_20260622`
- A second (earlier) run without `_meta647` suffix also exists on big disk at:
  `/media/diego/Datos/vae_AD_results/revision_bspc_2026/recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_20260622`
  (this is an earlier run with a different output_dir prefix from the config; the meta647 version used patched_metadata_candidate with N=647 strict intersection)

---

## FULL 5×5 Run Completion

All five folds completed with nonzero VAE validation splits and active metadata columns.
Stage B classifier-only readout and OOF-ECDF calibration were completed via:
`results/revision_bspc_2026/recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_meta647_20260622`
(local results symlink pointing to big disk)

Stage B readout run: `recover035_ch521_latent384_beta3p75_stageB_oof_score_calibration_20260623`

---

## Stage A Foldwise Metrics (logreg, canonical threshold at fold level)

| Fold | Logreg AUC | Logreg PR-AUC | Logreg Sens | Logreg Spec |
|:----:|:----------:|:-------------:|:-----------:|:-----------:|
| 1 | 0.7133 | 0.5603 | 0.250 | 0.967 |
| 2 | 0.8404 | 0.6220 | 0.158 | 1.000 |
| 3 | 0.8184 | 0.6309 | 0.105 | 1.000 |
| 4 | 0.6711 | 0.4295 | 0.316 | 0.933 |
| 5 | 0.8441 | 0.6855 | 0.050 | 1.000 |
| **Pooled** | **0.7615** | **0.5016** | **0.175** | **0.980** |

**Note:** Stage A pooled logreg shows a very conservative operating point (Sens≈0.18, Spec≈0.98) driven by the locked canonical logistic regression with strong L2 regularisation (C very small). This is NOT the final comparison metric; Stage B with OOF calibration corrects this.

Reference [1,0,2] Stage A pooled logreg: AUC≈0.775, PR-AUC≈0.630 (from same foldwise CSV).

---

## Stage B Primary Metrics (logreg OOF-ECDF calibrated)

| Metric | ch521 [5,2,1] | Reference [1,0,2] | Δ |
|:-------|:-------------:|:-----------------:|:---:|
| AUC (pooled) | **0.7764** | 0.7952 | **-0.0187** |
| PR-AUC (pooled) | **0.5260** | 0.5739 | **-0.0479** |
| Balanced Accuracy | 0.6852 | 0.7260 | -0.0408 |
| Sensitivity | 0.6804 | 0.7320 | -0.0516 |
| Specificity | 0.6900 | 0.7200 | -0.0300 |
| F1 | 0.5156 | 0.5635 | -0.0479 |
| Brier | 0.2353 | — | — |
| ECE | 0.2638 | — | — |
| **Promotion gate** | **False** | True | — |

**Philips CN FPR:** 0.4545 (same as reference — no improvement in Philips leakage)

---

## Paired Bootstrap (N=397 common subjects)

| Metric | Observed Δ | 95% CI | p(candidate ≤ reference) |
|:-------|:----------:|:------:|:------------------------:|
| AUC | -0.0187 | [-0.043, +0.005] | 0.9384 |
| PR-AUC | -0.0479 | [-0.103, +0.007] | 0.9542 |

**Interpretation:** The candidate is almost certainly not superior to the reference. With p≈0.94, there is very high probability that ch521 is non-superior in AUC. The CI just barely touches zero from below.

---

## Locked Promotion Gate

| Gate criterion | Required | ch521 [5,2,1] | Pass? |
|:---------------|:--------:|:-------------:|:-----:|
| AUC > reference * 0.984 | > 0.782951 | 0.7764 | **FAIL** |
| PR-AUC ≥ reference * 0.975 | ≥ 0.559873 | 0.5260 | **FAIL** |

---

## Decision

**REJECT — ch521 [5,2,1] does not pass the ADNI promotion gate.**

- AUC shortfall: 0.7764 vs gate 0.7830 (Δ = -0.0066 from gate)
- PR-AUC shortfall: 0.5260 vs gate 0.5599 (Δ = -0.0339 from gate)
- Both AUC and PR-AUC are uniformly lower than reference; no compensating gain elsewhere.
- Philips CN FPR unchanged (0.4545), so no reduction in manufacturer leakage either.

No OASIS external inference was run for this candidate.
The reference [1,0,2] `recover035` model remains the final selected model.

---

## beta=0.83 Exploratory Preflight (ch521_beta0p83_exploratory_preflight_20260623)

A further exploratory preflight was generated for [5,2,1] with beta=0.83 (lower regularisation).
This was motivated by the hypothesis that beta=3.75 may be too high for DistanceCorr to encode
useful signal. **This run was NOT launched.** The preflight PASSED all checks, and the launch
script is present at `ch521_beta0p83_exploratory_preflight_20260623/guarded_launch.sh`.

This branch remains un-executed. The argument for running it at lower beta is weak given that
the channel set itself failed the ADNI gate at full model capacity.

---

## Status After Full Evaluation

| Model | Status | Notes |
|:------|:-------|:------|
| [1,0,2] recover035 | **FINAL SELECTED** | OOF AUC=0.7952, OASIS verified |
| ch521 [5,2,1] beta=3.75 | **REJECTED** | AUC=0.7764, both gates fail |
| ch521 [5,2,1] beta=0.83 | **NOT LAUNCHED** | Exploratory preflight only |
| FAST [5,2,1] 128/300ep | **EXPLORATORY** | Hypothesis only; not comparable to FULL |
| FAST [5,2] 128/300ep | **EXPLORATORY** | 1-SE parsimonious; hypothesis only |
