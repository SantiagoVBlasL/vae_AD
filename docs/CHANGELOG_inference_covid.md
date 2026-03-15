# Changelog: COVID → AD Transfer Inference Pipeline

## v6.1 — Surgical Cleanup Pass (2025-07)

### Overview

Seven targeted fixes (A–G) applied to `03_a_inference_covid_from_adcn.ipynb`
bringing it from refactor-draft to paper-grade quality. No new analyses added,
no cells created or deleted — only edits, reordering, and annotation.
`NOTEBOOK_VERSION` bumped from `"v6.0"` to `"v6.1"`.

### Fixes Applied

| ID | Title | Cells touched | Summary |
|----|-------|---------------|---------|
| A | Outputs index legacy hygiene | §F (cell 39) | Added `_EXPECTED_PREFIXES` set + `_is_expected()` function. Output index now annotates each file as "current" or "legacy". Added `notebook_version` to manifest. |
| B | Explicit ADNI OOF loading | §2 (cell 4) | Replaced silent `sorted(glob(...))[0]` with explicit single-match requirement. `RuntimeError` if >1 file matches. |
| C | Canonical consensus-edges source | §2 (cell 4) | Replaced glob for consensus edges with explicit path `consensus_edges_{TARGET_CLF}_integrated_gradients_top50.csv`, consistent with notebook 06's `SUFFIX` variable. |
| D | Section reordering | §9, §10, §4-app | Moved §9, §10, and §4-app (renamed to Appendix A) after §8 interpretation. New order: §0→…→§8→§9→§10→App-A→§F. |
| E | S_sig definition consistency | §9 (cells 31–32) | Corrected code comment from "mean absolute connectivity" to "Σ\_e (w\_signed\_e · z\_e), z\_e CN-standardised". Enriched markdown with z\_e definition. |
| F | Missing interpretation statements | §5 (cell 10), §8-interp (cell 30) | Added metric-disagreement caveat (recon error vs latent distance) in §5. Added sex-confounding caveat in §8 interpretation. |
| G | Channel-specific enrichment caveat | §10 (cell 33) | Added paragraph noting enrichment is strong in ch0 only, null in ch1/ch2. |

### Section Outline (v6.1)

| § | Title | Cells |
|---|-------|-------|
| 0 | Configuration & Seeds | 1 |
| 1 | Imports & Paths | 1 |
| 2 | Artifact Discovery | 1 |
| 3 | Data Integrity & Tensor QC | 1 |
| 4 | Core Transfer Inference Results | 2 |
| 5 | OOD Detection & Score–Recon Landscape | 1 |
| 6 | Ensemble Stability & ADNI Thresholds | 2 |
| 7 | Threshold Policies, Clinical Correlations, Regression | 5 |
| 8 | Fatigue-Severity Analysis | 9 (8 code + 1 interpretation md) |
| 9 | AD-Derived Signature Projection (S\_sig) | 1 |
| 10 | Pathological Orthogonality (COVID-vs-Control) | 1 |
| App-A | Calibration Drift & Sensitivity | 2 |
| F | Final Outputs & Manifest | 1 |

**Total: 39 cells (25 code + 14 markdown), execution counts 1–27.**

### Execution

All cells execute end-to-end on a clean kernel restart (sequential counts 1–27).
No errors, no warnings.

---

## v6.0 — Strict Scope Separation & Circular-Analysis Removal (2025-07)

### Overview

Refactor of `03_a_inference_covid_from_adcn.ipynb` (50 → 39 cells) enforcing
strict scope separation from `06_further_analysis.ipynb`. Central question:
*"How does an AD-trained AD-likeness score behave when transferred to the
Long-COVID cohort, under substantial domain shift, weak expected classification
signal, and uncertain biological interpretability?"*

All anatomical interpretability (consensus heatmaps, network-pair decomposition,
glass brain, chord diagrams) is now exclusively in `06`. All circular within-COVID
edge testing (AD-like vs CN-like on AD-derived thresholds) has been removed.

### Deleted (11 cells)

| Old § | Cell ID | Reason |
|-------|---------|--------|
| §8b | `#VSC-edef10a8` | Network-pair decomposition of S_sig — anatomical = 06's territory |
| §9 | `#VSC-fc142a5f` + `#VSC-e363d263` | Subject selection markdown+code — no downstream consumer |
| §11 | `#VSC-8fe7f1c9` + `#VSC-37354aba` | Top-edges ranking markdown+code — duplicates 06 §5–§8 |
| §12a | `#VSC-480dcce9` | Within-COVID AD-like vs CN-like edge testing — circular/double-dipping (groups defined by score, tested on score-derived edges) |
| §12 notes | `#VSC-98f39b35` | Associated interpretation notes |
| §13 | `#VSC-1fb79101` + `#VSC-76daf2ee` | Network connectivity heatmaps intro+code — duplicates 06 |
| §13 notes | `#VSC-23c4256c` + `#VSC-22987425` | Dead references to removed sections |

### Rewritten / Renumbered (20 cells)

| New § | Content | Change |
|-------|---------|--------|
| §0 | Config | `NOTEBOOK_VERSION = "v6.0"`, removed `TOP_K_SUBJECTS`, `PRIMARY_CHANNEL_IDX` |
| Intro | Central question | Complete rewrite: central question, interpretability caveat box, 03_a/06 scope boundary, section table (§0–§10+§F) |
| §4 | Core Transfer Results | Reframed as transfer inference, not diagnostic |
| §9 | AD-Derived Signature Projection | Former §8a — S_sig kept as scalar only, explicit caveats about non-causal interpretation, scope boundary with 06 |
| §4-app | Calibration Drift & Sensitivity | Former QC cells — distribution shift overlay + threshold sensitivity |
| §10 | Pathological Orthogonality | Former §12b — COVID-vs-Control enrichment (only defensible edge comparison: independent groups). Removed panel (c) top-25 barplot, removed sign-agreement section. Reduced to 2-panel figure. Added inline ROI mapping code (previously in deleted §12a). |
| §8a–§8h | Fatigue Analysis | Former §10a–§10h — renumbered only |
| §F | Final Outputs | Renumbered header |

### Key Methodological Decisions

1. **Within-COVID edge testing removed (§12a):** Groups were defined as AD-like/CN-like
   by the Youden threshold on the AD-likeness score, then tested on edges derived from
   the same AD-vs-CN signature. This is circular — any "significant" result would
   necessarily recapitulate the training signal. Replaced by COVID-vs-Control (§10),
   which uses independent group labels.

2. **Anatomical interpretability moved to 06:** Network heatmaps, top-edge rankings,
   network-pair decomposition, glass brain, chord diagrams — all require the AD-vs-CN
   consensus signature and are authoritative for that domain. They don't belong in a
   transfer/OOD notebook.

3. **S_sig kept as scalar in §9, not decomposed:** The edge-level decomposition
   (network pairs, top edges) was moved to 06. Here, S_sig is presented as a single
   summary statistic with explicit caveats.

4. **Null-honest throughout:** All non-significant results reported with full effect
   sizes and confidence intervals. No "trending" language. AD-likeness score shows no
   association with MoCA (ρ = −0.078, q = 0.63), fatigue (KW p = 0.42), or
   COVID-vs-Control group (MW p = 0.69).

### Section Outline (v6.0)

| § | Title | Cells |
|---|-------|-------|
| 0 | Configuration & Seeds | 1 |
| 1 | Imports & Paths | 1 |
| 2 | Artifact Discovery | 1 |
| 3 | Data Integrity & Tensor QC | 1 |
| 4 | Core Transfer Inference Results | 2 |
| 5 | OOD Detection & Score–Recon Landscape | 1 |
| 6 | Ensemble Stability & ADNI Thresholds | 1 |
| 7 | Threshold Policies, Clinical Correlations, Regression | 5 |
| 8 | Fatigue-Severity Analysis | 8 |
| 9 | AD-Derived Signature Projection (S_sig) | 1 |
| 4-app | Calibration Drift & Sensitivity | 2 |
| 10 | Pathological Orthogonality (COVID-vs-Control) | 1 |
| F | Final Outputs & Manifest | 1 |

**Total: 39 cells (25 code + 14 markdown), execution counts 1–27.**

### Execution

All cells execute end-to-end on a clean kernel restart (sequential counts 1–27).
No errors, no warnings. Figures saved to `inference_covid_paper_output/Figures/`.

---

## v5.0 — Paper-Grade Methodological Refactoring (2025-07)

### Overview

Complete methodological refactoring of `03_a_inference_covid_from_adcn.ipynb`
(43 → 50 cells) addressing 8 mandatory fixes, adding a new fatigue-severity
analysis (§10), and removing scientifically unsound sections. Executed
end-to-end with all cells passing.

### Removed

| Section | Reason |
|---------|--------|
| **§10 UMAP** (latent-space visualisation) | Used channel-averaged latent means — masked channel-specific structure. No analytical value beyond exploratory aesthetics. |
| **§14 Clustering** (k-means on latents) | Same channel-averaging flaw. Clusters were not validated and added no insight beyond what §6b AD-Likeness categories already provide. |

### Corrected (8 Mandatory Fixes)

| # | Fix | Sections | Detail |
|---|-----|----------|--------|
| 1 | **Remove latent averaging** | §13 | Network connectivity heatmaps now computed per channel (ch1 primary, ch0, ch2). No cross-channel mean. |
| 2 | **Remove channel averaging** | §13, §12a, §12b | All edge-level and network-level analyses run per selected channel. Removed any `mean(axis=...)` across channels. |
| 3 | **Simplify OOD gate** | §5 | Primary gate changed from 4-quadrant system to single ADNI Recon-P95 threshold (0.5186). Recon-only, since P90-distance and logistic-regression gates added no discriminative value. |
| 4 | **Rename risk → AD-Likeness** | §6b, §7c, §7d | `risk_category` → `adlikeness_category`. Values: Low AD-Likeness (< 0.412), Intermediate [0.412, 0.493), High (≥ 0.493). Avoids causal connotation. |
| 5 | **Reframe §12a** | §12 markdown, §12 notes | Renamed from "COVID vs Controls" emphasis to "Post-selection characterisation". Framed as exploratory within-COVID stratification, not causal inference. Added post-selection bias caveat. |
| 6 | **Normalise network pairs** | §8b | Network-pair signature scores now divided by pair cardinality (n_ROIs_i × n_ROIs_j). Prevents large networks from dominating. Loads `roi_info_from_tensor.csv` with `network_label_in_tensor` column (16 networks). |
| 7 | **Add cohort audit** | §7a.1 | New cell with box-framed audit: COVID/CONTROL counts, missingness per column, CategoriaFAS × Sex/CategoríaCOVID cross-tabulations. |
| 8 | **Add fatigue analysis** | §10a–§10h | Full new section (8 cells) analysing CategoriaFAS vs AD-likeness score. See below. |

### New: Fatigue-Severity Analysis (§10a–§10h)

Eight cells implementing a complete fatigue analysis pipeline:

| Cell | Content | Key Result |
|------|---------|------------|
| §10a | Fatigue cohort audit | N = 150 COVID with valid CategoriaFAS (24 no fatigue, 97 fatigue, 29 extreme) |
| §10b | Descriptive table by fatigue category | Median AD-Likeness: 0.41 (none), 0.33 (fatigue), 0.38 (extreme) |
| §10c | Kruskal-Wallis + pairwise Mann-Whitney (BH-FDR) | AD-Likeness H = 1.74, p = 0.42 → **not significant**. Only EQ-VAS significant. |
| §10d | Ordinal trend (Spearman) + ordinal logistic regression | ρ = −0.03, p = 0.70. Ordinal logit: score β = 0.19, p = 0.87. Sex significant (p = 0.014). |
| §10e | Multivariable OLS: score ~ FAS + Age + Sex + severity + recovery | R² = 0.028, FAS β = 0.007, p = 0.64. No predictor significant for AD-likeness. |
| §10f | Extreme vs None focused comparison with Cliff's δ + bootstrap CIs | δ = −0.07 [−0.39, +0.24] (negligible). S_sig shows medium effect (δ = 0.32) but not FDR-significant. |
| §10g | Sensitivity: ambulatory-only, exclude-ICU, sex-stratified, COVID+CONTROL | All subgroups non-significant (all p > 0.46). |
| §10h | Publication figures: violin plots, forest plot (Cliff's δ), trend scatter | Three-panel figure + effect-size forest + ordinal trend. |

**Conclusion:** AD-likeness score is not associated with self-reported fatigue
severity. The β-VAE signature is domain-specific (neurodegenerative connectivity)
and does not capture post-COVID fatigue mechanisms.

### Narrative Changes

- **Intro/TOC**: Updated to reflect removed §10 UMAP and §14 Clustering sections.
  New §10 is fatigue analysis. Final section is §F (outputs index).
- **§0 config**: `NOTEBOOK_VERSION = "v5.0"`, `PRIMARY_CHANNEL_IDX = 0` (OMST).
- **§12 notes**: Rewritten post-selection characterisation caveat with explicit
  limitations: non-random selection, classifier-defined groups, circularity risk.
- **§13 notes**: Channel-specific framing, no cross-channel averaging claim.

### Key Scientific Results (v5.0)

| Finding | Value |
|---------|-------|
| ICC(2,1) cross-fold stability | 0.806 |
| OOD rate (ADNI Recon-P95) | 106/194 (54.6%) |
| Score–MOCA correlation | ρ = −0.078, q = 0.63 (NS) |
| Score–fatigue association | KW p = 0.42, ordinal ρ = −0.03 (NS) |
| Within-COVID edge differences (ch1) | 406/1026 FDR-significant edges |
| COVID vs Control edge differences | 0 FDR-significant edges |
| ch0 enrichment (AD-signature ∩ COVID-differing) | OR = 3.37 |
| AD-Likeness categories | Low: 122, Intermediate: 27, High: 45 |

### Files Changed

```
notebooks/03_a_inference_covid_from_adcn.ipynb   (43 cells → 50 cells)
  §0       — Version bump to v5.0, PRIMARY_CHANNEL_IDX = 0
  §Intro   — Rewritten TOC, removed §10 UMAP / §14 Clustering references
  §5       — Primary OOD gate → ADNI Recon-P95 (single threshold)
  §6b      — risk_category → adlikeness_category (Low/Intermediate/High)
  §7a.1    — NEW cohort audit cell
  §7c, §7d — risk_category → adlikeness_category throughout
  §8b      — Cardinality-normalised network decomposition
  §10a–h   — NEW fatigue analysis (8 cells)
  §12 md   — Post-selection characterisation framing
  §12 notes— Post-selection caveat
  §13 code — Channel-specific network summaries
  §13 notes— Channel-specific framing
  §10 UMAP — DELETED (markdown + code)
  §14 Clust— DELETED (markdown + code)
```

### New Output Files

```
Tables/
├── fatigue_descriptive_by_category.csv
├── fatigue_kruskal_wallis.csv
├── fatigue_pairwise_mannwhitney.csv
├── fatigue_ordinal_trend.csv
├── fatigue_ordinal_logit_summary.txt
├── fatigue_multivariable_ols.csv
├── fatigue_extreme_vs_none.csv
├── fatigue_sensitivity_analyses.csv
└── network_pair_connectivity_by_channel.csv   (UPDATED, per-channel)

Figures/
├── fig_fatigue_violin_panels.png
├── fig_fatigue_effect_sizes.png
└── fig_fatigue_trend.png
```

---

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
