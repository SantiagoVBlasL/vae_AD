# Changelog: COVID → AD Transfer Inference Pipeline

## v3.2 — Fix Tautological Enrichment + ADNI-Derived OOD (2025-07)

### Critical Fix: Tautological Enrichment (§12b)

**Problem:** In v3.1, the Fisher enrichment test in §12b drew its
"hit set" from within the same 1 026 AD-signature edges, guaranteeing
near-perfect overlap (OR → ∞, p ≈ 0). This was scientifically
meaningless.

**Fix:** `scripts/analysis_covid_paper_fixes.py` computes Cliff's δ
for **all 8 515 upper-triangle edges** (not just signature edges), then
defines the hit set as the top 5 % by |δ| from the full universe.
Fisher's exact test measures overlap with the 1 026 AD-signature edges.
Permutation validation (5 000 shuffles) confirms analytic p-values.

**Results:**
- **ch0** (Pearson OMST): genuinely enriched (OR ≈ 3.4, p < 1e-23)
- **ch1** (Pearson Full FisherZ): NOT enriched (OR ≈ 1.0, p = 0.66)
- **ch2** (MI): NOT enriched (OR ≈ 1.0, p = 0.42)

### New: ADNI-Derived OOD Thresholds (§5)

**Problem:** §5 used within-COVID P90 reconstruction error as OOD
threshold — circular, since it always flags exactly 10 % as OOD.

**Fix:** VAE reconstruction error is computed on ADNI training data
(all 5 folds) to derive P90 and P95 thresholds from the training
distribution. COVID subjects are then flagged against this
ADNI-derived reference.

**Result:** ADNI-P90 = 0.4886 flags 167/194 subjects as OOD vs
20/194 with COVID-P90. This reflects genuine domain shift between
ADNI and COVID cohorts.

### New: Sign-Agreement Metric (§12b)

Tests whether within-COVID edge differences (AD-like − CN-like)
match the sign of ADNI-derived weights.
*Result:* ~50 % agreement across all channels (binomial p > 0.05) —
not above chance.

### Files Changed

```
notebooks/03_a_inference_covid_from_adcn.ipynb
  §5   — Load ADNI-derived OOD thresholds; dual COVID-P90 / ADNI-P90
         quadrant labelling with green reference lines on plots.
  §12b — Load pre-computed full-universe enrichment + sign agreement.
  §12 notes — Rewritten to reflect v3.2 findings.

scripts/analysis_covid_paper_fixes.py  (NEW, ~570 lines)
  Standalone script implementing fixes A–E:
    A) Full-universe MWU + Fisher + permutation enrichment
    B) Sign-agreement + Spearman edge-score associations
    C) ADNI VAE reconstruction → OOD thresholds
    D) Figure integrity verification
    E) QC summary text file
```

### New Output Files

```
Tables/
├── covid_vs_control_all_edges_tests_ch{0,1,2}.csv      (8515 rows each)
├── enrichment_signature_vs_top5pct_universe_ch{0,1,2}.csv
├── within_covid_sign_agreement_ch{0,1,2}.csv
├── within_covid_edge_score_assoc_all_edges_ch{0,1,2}.csv
├── adni_ood_recon_distribution.csv                      (5 folds + ensemble)
└── qc_summary_v3.2.txt

Tables/covid_ood_quadrants_logreg.csv                    (UPDATED)
  New columns: ood_flag_adni_p90, ood_flag_adni_p95,
               quadrant_adni_p90, quadrant_adni_p95
```

---

## v3.1 — Q1-Grade §12/§13 Refactoring (2025-03-05)

### `notebooks/03_a_inference_covid_from_adcn.ipynb` (41 cells → 43 cells)

Complete rewrite of §12 (edge-level connectome analysis) and §13
(network connectivity summary) for Q1-journal statistical rigour.

#### §12 — Edge-Level Connectome Analysis (was 3 cells → now 4 cells)

| Change | Details |
|--------|---------|
| **Two-tier design** | **Tier 1 (PRIMARY):** Within Long-COVID, AD-like vs CN-like by ADNI Youden threshold. **Tier 2 (SECONDARY):** Long-COVID vs Study Controls (robustness). |
| **Per-channel** | All edge analyses run per channel — no cross-channel averaging. |
| **Robust edge mapping** | New `normalize_roi_name()` utility with aggressive normalisation. Maps consensus edges to tensor ROI indices. Validates ≥ 95% mapping rate. Saves `consensus_edges_mapped.csv`. |
| **Enrichment test** | Fisher's exact test for overlap between COVID-differing edges and AD-signature edges. Reports odds ratio + p-value. |
| **Null-result guardrails** | Effect-size distribution (Cliff's δ), p-value histogram, top-25 exploratory edges, and enrichment test — all produced even when FDR yields no hits. |
| **Interpretation notes** | New markdown cell covering domain shift, multiple comparisons, power, effect-size benchmarks (Romano 2006). |

#### §13 — Network Connectivity Summary (was 2 cells → now 3 cells)

| Change | Details |
|--------|---------|
| **Fixed network mapping** | Now reads `network_label_in_tensor` column (was checking for non-existent `Yeo17_Network`). Section no longer skipped. |
| **Three-panel heatmap** | COVID connectivity, Controls connectivity, and difference matrix side-by-side. |
| **A-priori focus** | Highlights DefaultMode, Limbic, Salience/VentralAttention, and Control networks with dedicated table (`network_pair_apriori_focus.csv`). |
| **Interpretation notes** | New markdown cell discussing aggregation caveat, channel-averaging caveat, and a-priori network rationale with references. |

#### New/Updated Output Files

```
Tables/
├── consensus_edges_mapped.csv                               (NEW)
├── covid_within_ADlike_vs_CNlike_signature_edges_ch{idx}.csv (NEW, per channel)
├── covid_vs_control_signature_edges_ch{idx}.csv              (UPDATED, per channel)
├── covid_vs_control_signature_enrichment_ch{idx}.csv         (NEW, per channel)
├── network_pair_connectivity_summary.csv                     (UPDATED)
└── network_pair_apriori_focus.csv                            (NEW)

Figures/
├── fig13_within_covid_edges_ch{idx}.png          (NEW, per channel)
├── fig13_covid_vs_control_edges_ch{idx}.png      (NEW, per channel)
└── fig14_network_connectivity_matrix.png         (UPDATED — 3 panels)
```

#### Not Changed
- §0–§11, §14, §F: Unmodified.
- `scripts/inference_covid_from_adcn.py`: No changes.
- No new dependencies introduced.

---

## v3.0 — Doctoral-Grade Clinical Analysis Notebook (2025-06)

### `notebooks/03_a_inference_covid_from_adcn.ipynb`  (21 cells → 28 cells)

Complete rewrite of the analysis notebook from exploratory-level to paper-grade
with ADNI-derived threshold policies, formal clinical validation, and network
decomposition. The inference script (`scripts/inference_covid_from_adcn.py`) is
unchanged from v2.0.

#### Major New Capabilities

| Capability | Section | Description |
|------------|---------|-------------|
| **ADNI-derived threshold policies** | §6 | Four policies computed from ADNI out-of-fold ROC: Youden (J-statistic), Screening (sens ≥ 0.85), FixedFPR (FPR ≤ 0.10), Cost-based (FN:FP = 2:1). Replaces naïve 0.5 cutoff. |
| **Clinical validation** | §7 | Full merge with `data/SubjectsData_AAL3_COVID.csv` (214 subjects). Spearman correlations, bootstrap CIs (10 000 resamples), BH-FDR correction. |
| **Group comparisons** | §7c | Mann-Whitney / Kruskal-Wallis tests across risk categories, OOD quadrants, and ResearchGroup. Effect sizes (Cohen's d, rank-biserial, Cliff's δ). |
| **Multivariate regression** | §7d | OLS with HC3 robust SEs: score ~ MOCA + Age + Sex + FAS + severity + recovery. VIF diagnostics. Interaction model (ResearchGroup × MOCA). |
| **OOD quadrant labelling** | §5 | Formal quadrant system: AD-like_InDist / AD-like_OOD / CN-like_InDist / CN-like_OOD using P90 recon error threshold. |
| **Network decomposition** | §8b | Signature score decomposed by Yeo-17 network pairs from consensus edges. Top contributing network pairs identified. |
| **Calibration drift** | QC | ADNI-CN vs ADNI-AD vs COVID score distributions overlaid. KS tests for distributional shift. |
| **Sensitivity analysis** | QC | MOCA association re-evaluated at each threshold policy to test robustness. |
| **Risk categories** | §6b | Multi-threshold classification: Low (< all), Moderate (intermediate), High (> all) across 4 policies. |
| **Provenance manifest** | §F | `notebook_manifest.json` with timestamps, file inventory, and configuration snapshot. |

#### Cell-by-Cell Structure (28 cells)

| Cell | Type | Section | Purpose |
|------|------|---------|---------|
| 1 | code | §0 | User-editable configuration (threshold params, bootstrap N, seeds) |
| 2 | markdown | — | Title & overview |
| 3 | code | §1 | Imports, PROJECT_ROOT, utility functions (bootstrap_ci, cohens_d, rank_biserial, cliffs_delta, save_fig) |
| 4 | code | §2 | Artifact discovery — FAIL EARLY if tables missing; loads COVID metadata, ADNI OOF, consensus edges |
| 5 | markdown | §3 | Phase 0 header |
| 6 | code | §3 | Physics & data integrity: NaN/Inf/symmetry/diagonal/TR checks |
| 7 | markdown | §4 | Phase 1 header |
| 8 | code | §4a | Score distributions: histograms, bootstrap CI for mean P(AD) |
| 9 | code | §4b | Cross-fold stability: ICC(2,1), vote entropy, fold-vs-fold Pearson |
| 10 | markdown | §5 | Phase 2 header |
| 11 | code | §5 | OOD diagnostics: quadrant labelling, Spearman recon×score, Mahalanobis |
| 12 | markdown | §6 | Phase 3 header |
| 13 | code | §6a | Threshold policy derivation from ADNI OOF ROC (4 policies) |
| 14 | code | §6b | Apply thresholds to COVID; risk categories (Low/Moderate/High) |
| 15 | markdown | §7 | Phase 4 header |
| 16 | code | §7a | Merge predictions ↔ metadata (join: SubjectID ↔ ID); missingness report |
| 17 | code | §7b | Spearman correlations + bootstrap CIs + BH-FDR (MOCA, MOCA_perc, EQ-VAS × score, S_sig) |
| 18 | code | §7c | Group comparisons: Mann-Whitney/Kruskal-Wallis + effect sizes + BH-FDR |
| 19 | code | §7d | Multivariate OLS (HC3), VIF, bootstrap MOCA coef, interaction model |
| 20 | markdown | §8 | Phase 5 header |
| 21 | code | §8a | Signature score distribution (S_sig) |
| 22 | code | §8b | Network-pair decomposition by Yeo-17 |
| 23 | markdown | §9 | Subject selection header |
| 24 | code | §9 | Subject selection: AD_inlier / AD_outlier / CN_boundary with clinical annotation |
| 25 | markdown | QC | Calibration drift header |
| 26 | code | QC | Calibration drift overlay + KS tests + sensitivity analysis (MOCA × threshold) |
| 27 | markdown | §F | Outputs index header |
| 28 | code | §F | Outputs inventory + notebook_manifest.json |

#### Key Data-Flow Decisions

- **Join key**: `predictions['SubjectID']` ↔ `metadata['ID']` (both use CP0001 format).
  The metadata `SubjectID` column (333_S_XXXX format) is **not** used. 194/194 match.
- **ADNI OOF source**: Auto-discovered via glob from `results/vae_3channels_beta65_pro/all_folds_clf_predictions_MULTI_svm_*.csv`. Filtered to `classifier_type == "logreg"`. 183 rows, 5 folds.
- **Consensus edges**: Loaded from `interpretability_paper_output/tables/consensus_edges_logreg_integrated_gradients_top50.csv`.
  Contains `src_Yeo17_Network` and `dst_Yeo17_Network` columns → network decomposition is feasible.
- **statsmodels HC3 quirk**: `model.summary2().tables[1]` uses column `P>|z|` (not `P>|t|`) when `cov_type="HC3"`. Code uses dynamic detection.
- **Accent handling**: `CategoríaCOVID` (accent on í) handled with `.replace("í", "i")` fallback.

#### v2 → v3 Comparison (Notebook Only)

| Area | v2.0 (21 cells) | v3.0 (28 cells) |
|------|-----------------|-----------------|
| **Threshold** | Naïve 0.5 | 4 ADNI-derived policies (Youden, Screening, FixedFPR, CostBased) |
| **Clinical data** | None | Full merge with SubjectsData_AAL3_COVID.csv |
| **Correlations** | Spearman only (no correction) | Spearman + bootstrap CI + BH-FDR |
| **Group tests** | None | Mann-Whitney / Kruskal-Wallis + effect sizes |
| **Regression** | None | OLS with HC3, VIF, interaction model |
| **OOD system** | Scatter only | Formal quadrant labels (4 categories) |
| **Network decomp** | None | Yeo-17 network-pair contribution heatmap |
| **Calibration** | None | ADNI vs COVID distribution overlay + KS |
| **Sensitivity** | None | MOCA association at each threshold |
| **Risk strata** | Binary (AD-like / CN-like) | Three-tier (Low / Moderate / High) |
| **Multiple testing** | None | BH-FDR across all endpoints |
| **Figure count** | 5 | 10 |
| **Table count** | 10 | 18 |

### Full Output Inventory (v3.0)

```
inference_covid_paper_output/
├── Tables/
│   ├── adni_derived_thresholds.csv              (NEW — 4 policies with AUC, threshold, sens, spec)
│   ├── clinical_analysis_cohort_logreg.csv      (NEW — merged cohort for analysis)
│   ├── clinical_correlations.csv                (NEW — Spearman + CI + FDR)
│   ├── clinical_group_comparisons.csv           (NEW — group tests + effect sizes)
│   ├── covid_fold_stability_logreg.csv
│   ├── covid_inference_summary.csv
│   ├── covid_latent_distance_ensemble.csv
│   ├── covid_latent_distance_per_fold.csv
│   ├── covid_ood_quadrants_logreg.csv           (NEW — quadrant labels per subject)
│   ├── covid_predictions_ensemble.csv
│   ├── covid_predictions_per_fold.csv
│   ├── covid_recon_error_ensemble.csv
│   ├── covid_recon_error_per_fold.csv
│   ├── covid_signature_scores.csv
│   ├── covid_subject_selection_logreg.csv
│   ├── covid_threshold_analysis_logreg.csv      (NEW — per-subject predictions at each policy)
│   ├── sensitivity_moca_by_threshold.csv        (NEW — MOCA stats by threshold)
│   └── signature_network_decomposition.csv      (NEW — Yeo-17 pair contributions)
├── Figures/
│   ├── fig1_score_distribution_logreg.png
│   ├── fig2_cross_fold_stability_logreg.png
│   ├── fig3_ood_diagnostics_logreg.png
│   ├── fig4_threshold_policy_logreg.png         (NEW)
│   ├── fig5_clinical_correlations.png           (NEW)
│   ├── fig6_clinical_boxplots.png               (NEW)
│   ├── fig7_signature_score_logreg.png
│   ├── fig8_network_decomposition.png           (NEW)
│   ├── fig9_subject_map_logreg.png
│   └── fig10_calibration_drift_logreg.png       (NEW)
├── Logs/
│   └── inference.log
├── inference_config.json
├── notebook_manifest.json                       (NEW)
└── run_manifest.json
```

### Quick-Start

```bash
# 1. Run inference (if not already done)
conda run -n vae_ad python scripts/inference_covid_from_adcn.py --classifier_types logreg

# 2. Open and run all cells in the notebook
#    notebooks/03_a_inference_covid_from_adcn.ipynb
```

---

## v2.0 — Paper-Grade Refactor (2025-06)

### `scripts/inference_covid_from_adcn.py`  (870 → 1076 lines)

| Area | Legacy (v1) | Refactored (v2) |
|------|-------------|------------------|
| **Metadata** | `--covid_metadata_path` **required** — crash if absent | **Optional** — graceful degradation using per-fold `metadata_imputation.json` |
| **Classifier resolution** | Single filename pattern; crash if not found | Priority chain: `calibrated → final → raw → legacy`; logs which was loaded |
| **OOD diagnostics** | None | `compute_recon_error_offdiag()` (per-subject off-diagonal MAE) + `compute_mahalanobis_distances()` (latent distance from train-dev Gaussian) |
| **Signature scoring** | None | `compute_signature_scores()` — consensus-edge S_sig from interpretability outputs |
| **Manifest** | None | `run_manifest.json` with: git hash, timestamps, artifact checksums, python version |
| **Smoke test** | None | `--smoke_test` flag: fold 1 only, 20 subjects, logreg |
| **Skip signature** | N/A | `--skip_signature` flag |
| **Output layout** | Flat directory | `inference_covid_paper_output/{Tables,Figures,Logs}/` |
| **Path handling** | Some hard-coded paths | All paths relative to `PROJECT_ROOT` (resolved from `__file__`) |
| **AD tensor** | Not loaded | Auto-loaded from `run_config.args.global_tensor_path` for OOD baselines |
| **Ensemble** | Mean only | Mean (default), configurable |
| **Logging** | print only | `logging` module with file + console handlers |

### `notebooks/03_a_inference_covid_from_adcn.ipynb`  (24 cells → 21 cells)

| Area | Legacy | Refactored |
|------|--------|------------|
| **PROJECT_ROOT** | `os.chdir(Path("..").resolve())` | Walk-up search for `pyproject.toml` |
| **PYTHONPATH** | Hard-coded `!export PYTHONPATH=/home/diego/proyectos/betavae-xai-ad/src:...` | `sys.path.insert(0, str(SRC_DIR))` |
| **Experiment** | Points to `vae_3channels_beta25_weight` (wrong) | `vae_3channels_beta65_pro` (correct) |
| **Metadata** | Requires `covid_metadata_for_pipeline.csv` (doesn't exist!) | Graceful – skips group comparisons when no metadata |
| **Analysis phases** | Score histograms only | 6 phases: integrity, inference, OOD, signature, statistics, subject selection |
| **Figures** | Inline only | Saved to `Figures/` (png, 200 dpi) |
| **Stats** | None | Bootstrap CIs, Cohen's d, Spearman correlations, one-sample t-test |
| **Configuration** | Scattered constants | Single `§0 CONFIG` cell at top |
| **Reproducibility** | No seeds | `SEED_GLOBAL`, `SEED_SAMPLING` in config |

### Quick-Start (v2)

```bash
# Smoke test (~30s)
conda run -n vae_ad python scripts/inference_covid_from_adcn.py --smoke_test

# Full run (all 5 folds × 194 subjects)
conda run -n vae_ad python scripts/inference_covid_from_adcn.py --classifier_types logreg

# Then open notebooks/03_a_inference_covid_from_adcn.ipynb and run all cells
```
