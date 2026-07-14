# Wasserstein Distance Summary — Locked Model Latent Space

**Date:** 2026-07-06  
**Model:** `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`  
**Request:** Martín Belzunce (WhatsApp 2026-07-05) — "comparar las distribuciones del modelo entrenado contra los sitios y databases externas, usando por ejemplo Wasserstein distance"

---

## What was computed

Three distance metrics, all computed per fold independently (never pooling latent vectors across folds):

| Metric | What it measures | How computed |
|--------|-----------------|--------------|
| **Mean per-dim W1** | Average 1D Wasserstein-1 across all 384 latent dimensions | `scipy.stats.wasserstein_distance` per dimension |
| **PCA10-W1 mean** | Wasserstein-1 on the top-10 principal components | PCA fitted on trainDev; W1 per PC; mean across 10 PCs |
| **Gaussian-W2** | Full Wasserstein-2 between Gaussian approximations | Closed-form: mean-difference² + Bures-Wasserstein covariance term (LW covariances) |

Nuisance residualization: manufacturer comparison uses diagnosis + age + sex (OLS on trainDev, applied to test fold); ADNI vs OASIS uses diagnosis only (to preserve cohort-level domain structure while removing AD/CN case-mix).

Results are mean ± SD across 5 folds. All permutation p-values are the minimum achievable with 1000 permutations (p = 0.001).

---

## 1. ADNI manufacturer comparison (within training dataset)

These distances are between manufacturer subgroups in the held-out test fold latent space, after residualizing out diagnosis, age, and sex.

| Pair | Mean per-dim W1 | PCA10-W1 mean | Gaussian-W2 | Perm. p (all folds) |
|------|----------------|---------------|-------------|---------------------|
| GE vs Philips | 0.319 ± 0.015 | 1.314 ± 0.157 | 9.77 ± 0.40 | 0.001 |
| GE vs SIEMENS | 0.311 ± 0.013 | 1.229 ± 0.126 | 9.22 ± 0.33 | 0.001 |
| Philips vs SIEMENS | 0.295 ± 0.029 | 1.123 ± 0.164 | 9.15 ± 0.91 | 0.001 |

**Permutation context (GE vs Philips, PCA10-W1 mean):** Observed = 1.31 ± 0.16; permutation null = 0.80 ± 0.10. The observed distance is approximately **1.6× the null mean** — substantial, well above chance.

**Interpretation:** After removing diagnosis, age, and sex effects, the three manufacturers still have significantly different latent distributions. GE vs Philips is the largest gap, consistent with the known Philips CN false-positive elevation (Philips subjects occupy a shifted region of the latent space).

---

## 2. ADNI (training set) vs OASIS (external dataset)

These distances compare the ADNI training distribution (317 subjects per fold) against the OASIS external dataset (180 subjects, all SIEMENS, encoded by the same fold's frozen encoder). Residualized by diagnosis only.

| Metric | Mean ± SD across 5 folds |
|--------|--------------------------|
| Mean per-dim W1 | 0.354 ± 0.009 |
| PCA10-W1 mean | 0.983 ± 0.032 |
| Gaussian-W2 | 11.68 ± 0.15 |
| Permutation p (all folds) | 0.001 |

**Permutation context (PCA10-W1 mean):** Observed = 0.98 ± 0.03; permutation null = 0.33 ± 0.04. The observed distance is approximately **3× the null mean** — the ADNI-OASIS distributional gap is highly significant.

**Comparison with within-ADNI manufacturer gaps:**

- PCA10-W1 mean: ADNI-OASIS (0.98) < GE-Philips within-ADNI (1.31)
- Gaussian-W2: ADNI-OASIS (11.68) > all within-ADNI pairs (9.1–9.8)

The PCA-based metric (which captures the directions of greatest variance) suggests ADNI-OASIS separation is somewhat smaller than the GE-Philips gap in the top 10 dimensions. However, the full Gaussian-W2 — which accounts for the entire 384-dimensional covariance structure — shows that the ADNI vs OASIS distributional difference is **larger** than any within-ADNI manufacturer gap. This is consistent with the OASIS dataset introducing domain shift beyond what scanner manufacturer alone explains within ADNI.

**Important caveat:** OASIS is entirely SIEMENS-scanned. The ADNI-OASIS distributional gap is therefore NOT purely a manufacturer effect — it reflects a combination of site, acquisition protocol, cohort demographic, and study design differences between the two datasets.

---

## 3. Top latent dimensions contributing to ADNI-OASIS gap

The following latent dimensions show the highest per-dimension Wasserstein-1 distance between ADNI trainDev and OASIS (mean across 5 folds, after diagnosis residualization):

| Rank | Latent dim | Mean W1 (across folds) |
|------|-----------|------------------------|
| 1 | 147 | 1.129 |
| 2 | 333 | 1.115 |
| 3 | 383 | 1.052 |
| 4 | 73 | 0.987 |
| 5 | 324 | 0.958 |

Full per-fold rankings: `wasserstein_top_contributing_dims_adni_vs_oasis.csv` (top-20 per fold).

**Decoder mapping status: FOLLOW-UP REQUIRED.** Mapping these dimensions to functional connectivity patterns would require loading the VAE decoder weights and computing per-dimension reconstruction sensitivity. The existing `decoded_connectome_delta_by_channel.csv` maps traversal *directions* (ridge-regression β vectors) to FC deltas, not individual latent dimensions. This cross-reference is deferred to a future analysis step.

---

## 4. Summary statement (for SIPAIM manuscript draft)

> We quantified the distributional distance between the locked model's latent representations across manufacturers and datasets using Wasserstein distances. Within the ADNI training set, all three manufacturer pairs show significantly separated latent distributions after controlling for diagnosis, age, and sex (GE vs Philips PCA10-W1 = 1.31 ± 0.16; permutation p = 0.001 in all five folds). The ADNI vs OASIS external dataset gap is also highly significant (PCA10-W1 = 0.98 ± 0.03; Gaussian W2 = 11.68 ± 0.15; permutation p = 0.001 all folds), and in terms of full covariance geometry (Gaussian W2) exceeds the within-ADNI manufacturer separation. Since OASIS subjects were all scanned on SIEMENS equipment, the ADNI–OASIS gap is not purely a manufacturer effect — it captures the combined contribution of site, protocol, and cohort differences between the two studies.

---

## Guardrails and limitations

- All computations are per-fold; latent vectors were never pooled across folds.
- Estimators (StandardScaler, PCA, OLS for nuisance) fitted on ADNI trainDev only.
- Gaussian-W2 uses Ledoit-Wolf shrinkage covariances (shrinkage ≈ 0.54); values underestimate true population W2.
- Sinkhorn/POT not computed (library not installed). PCA10-W1 and Gaussian-W2 are the tractable approximations used here.
- OASIS latent cache used: `promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/oasis_fold_latent_mu_runwise164.csv` — encoded by the locked model's per-fold frozen encoders on 2026-06-04. No new inference was run.
