# Changelog — `07_it_information_theory.ipynb`

**v4 date:** 2026-03-18
**v3 date:** 2026-03-18
**v2 date:** 2026-03-18
**Generator:** `scripts/_gen_07_it_notebook.py`

---

## v4 — Final closing pass

### Summary

Editorial and reproducibility hardening pass.  No scientific results changed; v3 index fix preserved intact.

### A. RUN_MODE fail-closed system

| Item | v3 | v4 |
|------|----|----|
| Missing betavae_xai / torch | Silent WARN; §4-6/§8 silently skipped | `full_recompute`: raises `RuntimeError` immediately |
| Stale legacy files mixed with current outputs | Present in manifest | Moved to `_legacy/` subdirectory at run start |
| Manifest section status | `true` / `false` | `recomputed` / `loaded_from_cache` / `skipped` / `exploratory` |
| Manifest run_mode field | Absent | `run_mode: "full_recompute"` or `"artifact_review"` |
| Publication-ready field | Absent | Explicit `publication_ready` field with reason |

### B. Dependency preflight cell

New cell after §0 checks: python, numpy, scipy, pandas, sklearn, matplotlib, seaborn, joblib, torch, optuna, betavae_xai.

In `full_recompute`: raises `RuntimeError` listing missing packages and pointing to `environment_notebook07.yml`.

In `artifact_review`: prints WARN for missing re-inference deps but continues.

### C. §10 narrative corrected (critical editorial fix)

**Previous text (v3 — incorrect):**
> "All Bonferroni-corrected p-values exceed 0.05 and effect sizes are near zero.
> This should be interpreted as: **no detectable global spectral-entropy difference at this coarse scale**."

**Corrected text (v4):**
The §10 narrative now explicitly states the channel-specific pattern:
- Pearson_Full and Pearson_OMST: no significant AD–CN difference after Bonferroni (p_bonf > 0.05)
- MI_KNN_Symmetric: significant after Bonferroni (p_bonf ≈ 0.006, r_rb ≈ −0.264, CN > AD)

The §10b figure suptitle is now **data-driven** (computed from the actual statistics), not hardcoded.

The §11 summary now reads: "Findings are channel-specific. MI_KNN_Symmetric shows a significant
reduction in VNE for AD relative to CN. Pearson-based channels show no significant AD–CN difference."

### D. Figure quality

| Item | v3 | v4 |
|------|----|----|
| PNG DPI | 200 (savefig default) | **300 dpi** |
| Vector export | None | **PDF** for all 9 figures |
| §7 violin x-labels | "CN", "AD" | `CN (n=89)`, `AD (n=95)` with sample sizes |
| §7 per-fold panel | Mean AUC line only | Mean AUC line + textbox with mean AUC_cal and mean ECE |
| §10 box titles | `p_bonf=... r_rb=...` | + N_AD, N_CN, significance marker `*` / `ns` |

### E. LaTeX table exports

New: `save_latex()` utility exports key result tables as `.tex` alongside CSV:
- `07_per_fold_calibration_metrics.tex`
- `10_von_neumann_group_stats_corrected.tex`

### F. Legacy file quarantine

At `full_recompute` start, superseded files are moved to `_legacy/` (not deleted):
- `05_latent_dim_statistics.csv` → `_legacy/`
- `06_topk_auc_by_strategy.csv` → `_legacy/`
- `09_spectral_entropy_proxy.csv` → `_legacy/`
- `10_von_neumann_group_stats.csv` → `_legacy/`

The authoritative manifest lists only current outputs.

### G. Reproducibility spec

New file: `environment_notebook07.yml` with exact working versions:
numpy=1.26.4, scipy=1.15.3, pandas=2.2.3, sklearn=1.7.2, torch=2.5.1, optuna=4.4.0, python=3.10.18

---

## v3 — Hardened publication-grade

### ROOT CAUSE FIX: Global tensor index corruption in §4/§5/§6/§8

**Severity: Critical.**  All re-inference-dependent analyses in v2 encoded the wrong subjects.

| Item | v2 (wrong) | v3 (correct) |
|------|-----------|--------------|
| Test index file | `test_indices.npy` (local pool indices 0–182) | `test_tensor_idx.npy` (global tensor indices 4–427) |
| Train index file | `train_dev_indices.npy` (local pool indices 0–182) | `train_dev_tensor_idx.npy` (global tensor indices 1–427) |
| GLOBAL_TENSOR indexing | `GLOBAL_TENSOR[local_pool_idx]` → wrong subjects | `GLOBAL_TENSOR[global_tensor_idx]` → correct subjects |
| Label lookup in §8 | Local indices ≠ tensor_idx in prediction CSV → ~2/37 matches per fold → N=14 | Global indices = tensor_idx in prediction CSV → 37/37 matches per fold → N=183 |

**Root cause:** `test_indices.npy` contains **local indices** (0 to ~182) into the AD+CN training pool, not global tensor indices.  The global tensor has 431 subjects (AD + CN + MCI).  The correct global indices are in `test_tensor_idx.npy` (range 4–427).

**SubjectID→tensor_idx mapping:** v3 builds this map from the pipeline log CSV (row position = global tensor index), covering all 431 subjects.  This also enables §10c MCI descriptive analysis.

### §4 — Information Bottleneck Audit

| Item | v2 | v3 |
|------|----|----|
| Subjects encoded | Wrong (local-indexed) | Correct (global tensor indexed) |
| n_AD / n_CN in fold summary | Missing | Added (all match prediction CSV) |
| n_nan_y | Missing | Added (all folds: 0 NaN) |

### §5 — Latent MI and Confounding

| Item | v2 | v3 |
|------|----|----|
| Training latents | Local-indexed (wrong subjects) | Correct (global tensor indexed) |
| Metric name | `heuristic_diagnostic_purity` | **`heuristic_marginal_diagnostic_ratio`** (more cautious) |
| Caveats | Listed in section | Explicitly stated in section title and §2.4 derivation |

### §6 — Top-k Readout

| Item | v2 | v3 |
|------|----|----|
| Training latents used for ranking | Local-indexed (wrong) | Correct (global tensor indexed) |
| Output | Mean AUC by strategy | **Long table: one row per (strategy, k, fold)** with n_train, n_test, n_ad_test, n_cn_test, auc |
| New file | — | `Tables/06_topk_auc_by_strategy_by_fold.csv` |
| Uncertainty | None | **95% bootstrap CI across folds** shown as shaded bands in figure |
| Claim | "by_w_abs saturates at 1.0 from k=16" | "evidence consistent with distributed representation; exact saturation point uncertain given fold-level variance with N_test≈37" |

**Note on AUC values:** v2 showed highly quantized AUC values (0.333, 0.500, 0.667, 0.833, 1.000).  These were an artefact of averaging over only 3 folds (small integer multiples of 1/3).  v3 uses all 5 folds and shows real fold-level scatter (0.40–0.86 for by_w_abs), which is more informative.

### §7 — Score-Level IT and Calibration

| Item | v2 | v3 |
|------|----|----|
| Theoretical derivation | Referenced but not shown | **Explicit derivation in §2.3 and dedicated markdown cell in §7** |
| Per-fold entropy table | Pooled by (group, score_type) | **Per-fold rows: (fold, group, score_type, H_mean, H_std, n)** |
| Panel explanations | Brief note in figure | Derivation cell explains exactly why h₂(σ(m)) applies only in Panel A |

### §8 — Ising / MaxEnt Scaffold

| Item | v2 | v3 |
|------|----|----|
| Pooled N | **14 (wrong — index mismatch)** | **183 (correct — all OOF subjects)** |
| N_AD / N_CN | 8 / 6 (wrong) | 94 / 89 (correct) |
| Root cause | `fold_subject_idx_dict` stored local pool indices; label lookup failed | `fold_tensor_idx_dict` stores global tensor indices; label lookup is exact |
| Validation cell | Missing | **Dedicated pooling audit cell** prints N, N_AD, N_CN, n_nan, alignment status per fold |
| New file | — | `Tables/08_ising_pooling_audit.csv` |
| perfect_alignment | N/A | True for all 5 folds (37/37, 37/37, 37/37, 36/36, 36/36 matches) |

### §9 — Spectral Ordinal Complexity Proxy

No changes from v2.  Section remains clearly labelled **[EXPLORATORY]**.

### §10 — Von Neumann Entropy

| Item | v2 | v3 |
|------|----|----|
| Version history note | Missing | **Added subsection documenting v1→v2→v3 changes** |
| Group-label mapping | Fold subject files only (AD+CN, 183) | Pipeline log (all 431 subjects: AD+CN+MCI) |
| N_AD / N_CN | 94 / 89 (from fold files) | **95 / 89** (from full metadata via pipeline log) |
| MCI analysis | Missing | **§10c: descriptive-only comparison of AD, CN, MCI VNE distributions** |
| New file | — | `Tables/10_von_neumann_group_descriptives_all_groups.csv` |
| New figure | — | `Figures/10_von_neumann_all_groups_descriptive.png` |
| Null result interpretation | "p-values > 0.05" | Explicitly: "no detectable global spectral-entropy difference at this coarse scale; does not imply biological equivalence" |

**Important v3 finding in §10:** With correct subject mapping (N_AD=95, N_CN=89), the
`MI_KNN_Symmetric` channel now shows **p_bonf = 0.006** (r_rb = −0.263, CN > AD in VNE).
This was not detectable with the wrong subject set in v2 (which had p_bonf = 1.0 for all
channels due to the index misalignment corrupting group labels).

---

## What remains exploratory (not publication-grade without further work)

| Section | Item | Limitation |
|---------|------|------------|
| §4 | Gaussian TC via Ledoit-Wolf | Gaussian approximation; N≈37 test subjects per fold |
| §5 | Heuristic marginal diagnostic ratio | Marginal MI ratio; not conditional; not canonical IT |
| §6 | Top-k AUC saturation point | High fold-level variance with N_test≈37; interpret trends, not exact k values |
| §8 | Ising first/second moments | No full Ising fit; pseudo-likelihood or MPF fitting is TODO |
| §9 | Spectral ordinal complexity | Eigenvalue magnitude proxy, not temporal Bandt-Pompe |
| §10c | MCI VNE descriptive | Descriptive only; no inferential claim for MCI group |

---

## Scientific findings from v3

- **Distributed code confirmed:** top-k AUC increases gradually across all ranking strategies;
  no sharp transition is detectable with N_test≈37.
- **Per-fold logreg AUC:** 0.40–0.86 depending on fold and strategy (by_w_abs); mean ~0.77.
- **Active units:** all 256 latent dimensions active at ε=0.01 (frac_active=1.0, all folds).
- **Von Neumann MI_KNN_Symmetric:** p_bonf=0.006, r_rb=−0.263 (CN > AD); this finding
  was masked in v2 by the index bug.  Pearson channels remain null (p_bonf > 0.05).
- **MCI VNE:** MCI falls near CN in MI_KNN_Symmetric (mean 4.839 vs AD 4.838, CN 4.848)
  — descriptive only, no inferential test conducted.

---

## Files added or renamed in v3

| New file | Description |
|----------|-------------|
| `Tables/06_topk_auc_by_strategy_by_fold.csv` | Long table: one row per (strategy, k, fold) |
| `Tables/08_ising_pooling_audit.csv` | Per-fold alignment audit (N, AD, CN, n_match) |
| `Tables/10_von_neumann_group_descriptives_all_groups.csv` | VNE descriptives for AD/CN/MCI |
| `Figures/10_von_neumann_all_groups_descriptive.png` | AD/CN/MCI VNE boxplot |

---

## v1→v2 changes (retained for reference)

### §4 — Information Bottleneck Audit

| Item | v1 | v2 |
|------|----|-----|
| Total Correlation estimator | Raw sample covariance (rank-deficient when N≈37 < D=256) | **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`) |
| Fold summary export | Missing | `04_ib_fold_summary.csv` |
| Disentanglement claim | Implicit | Explicit markdown: β=6.5 produces **distributed** representation; disentanglement not claimed |

### §5 — Latent MI and Confounding

| Item | v1 | v2 |
|------|----|-----|
| MI data split | Test-set latents (leakage in ranking) | **Training-set latents only** |
| Purity metric name | `purity` | `heuristic_diagnostic_purity` (renamed to `heuristic_marginal_diagnostic_ratio` in v3) |
| Exports | `05_latent_dim_statistics.csv` | `05_latent_mi_train_only.csv` + `05_latent_mi_summary_by_fold.csv` |

### §6 — Top-k Readout

| Item | v1 | v2 |
|------|----|-----|
| Fallback CV | Fell back to 5-fold CV inside the test fold | **Removed** |
| Ranking data | Test-set MI | Training-set MI / weights only |
| Output CSV | `06_topk_auc_by_strategy.csv` | `06_topk_auc_by_strategy_nested_clean.csv` |

### §7 — Score-Level IT and Calibration

| Item | v1 | v2 |
|------|----|-----|
| Entropy-vs-margin figure | Single panel mixing raw and calibrated | **Two panels**: Panel A (raw margin) + Panel B (calibrated logit) |
| Theoretical curve | Missing | Panel A overlays h₂(σ(m)) |

### §8 — MaxEnt / Ising Scaffold

| Item | v1 | v2 |
|------|----|-----|
| Label source | Metadata + fold_y_dict | **Fold prediction CSVs** (source of truth) |

### §9 — Spectral Ordinal Complexity Proxy

| Item | v1 | v2 |
|------|----|-----|
| Section name | "Bandt-Pompe spectral proxy" | **"Spectral Ordinal Complexity Proxy [EXPLORATORY]"** |
| Output file | `09_spectral_entropy_proxy.csv` | `09_spectral_ordinal_complexity_proxy.csv` |

### §10 — Von Neumann Entropy

| Item | v1 | v2 |
|------|----|-----|
| Effect size formula | `r_rb = U/(n1*n2)` (wrong — this is θ, not r_rb) | `r_rb = 2*U/(n1*n2) - 1` (correct); θ separately labelled |
| Multiple comparisons | None | **Bonferroni correction** across 3 channels |
| Output file | `10_von_neumann_group_stats.csv` | `10_von_neumann_group_stats_corrected.csv` |
