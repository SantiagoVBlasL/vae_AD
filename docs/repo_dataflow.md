# Repository Data-Flow & Provenance Map

> Auto-generated: 2026-03-06.  Authoritative reference for how scripts,
> notebooks, and result folders relate to each other.

---

## 1. Repository Architecture

| Folder | Purpose |
|---|---|
| `src/betavae_xai/` | Installable Python package (models, feature extraction, interpretability, QC utilities). |
| `scripts/` | CLI entry points that orchestrate training, inference, and analysis. |
| `notebooks/` | Interactive analysis / paper figures. Numbered `00–06`. |
| `data/` | Local data (large tensors `.npz` not versioned; metadata CSVs versioned). |
| `results/` | All pipeline outputs (not versioned except `README.md`). |
| `configs/` | Conda lock files, run configs. |
| `artifacts/` | Placeholder for external artifacts (currently empty). |
| `docs/` | Design documents and changelogs. |

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                                 │
│  scripts/feature_extraction_manual.py      → data/…ADNI…/GLOBAL_TENSOR  │
│  scripts/feature_extraction_manual_COVID.py → data/…COVID…/GLOBAL_TENSOR│
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│                      ADNI TRAINING (AD vs CN)                           │
│  scripts/run_vae_clf_ad_inference.py                                    │
│    → results/vae_3channels_beta65_pro/                                  │
│      fold_{1..5}/vae_model_fold_*.pt + clf + norm_params                │
│      all_folds_clf_predictions_MULTI_*.csv  (ADNI OOF)                  │
│      run_config.json, roi_info_from_tensor.csv                          │
│                                                                         │
│  (ablation variant: scripts/run_vae_clf_ad_ablation.py)                 │
│    → notebooks/ablation_full_run_fast/                                  │
└────────────┬────────────────────────────────────────────────────────────┘
             │
     ┌───────┴────────────────────────────────────┐
     │                                            │
┌────▼──────────────────────────────┐  ┌──────────▼──────────────────────────┐
│  TRANSFER INFERENCE (03_a path)   │  │  PROBING CLASSIFICATION (03_b path) │
│                                    │  │                                      │
│  scripts/inference_covid_from_     │  │  scripts/run_covid_classification_   │
│    adcn.py                         │  │    probing.py                        │
│    → …/inference_covid_paper_      │  │    → …/covid_classification_         │
│      output/Tables/                │  │      probing/Tables/                 │
│      (7 core CSV + manifest)       │  │      (7 CSV + manifest)              │
│                                    │  │                                      │
│  scripts/ood_policy_adni.py        │  └──────────┬──────────────────────────┘
│    → …/Tables/ (OOD tables)        │             │
│                                    │  ┌──────────▼──────────────────────────┐
│  scripts/analysis_covid_paper_     │  │  notebooks/03_b_covid_              │
│    fixes.py                        │  │    classification_probing_audit     │
│    → …/Tables/ (enrichment, etc.)  │  │  Pure audit/visualization + AUC    │
│                                    │  │  recomputation from OOF.            │
└────────────┬───────────────────────┘  └─────────────────────────────────────┘
             │
┌────────────▼───────────────────────┐
│  notebooks/03_a_inference_covid_   │
│    from_adcn.ipynb                 │
│  Heavy analysis notebook:          │
│  - threshold derivation            │
│  - clinical correlations           │
│  - OOD diagnostics                 │
│  - latent UMAP + clustering        │
│  - edge-level connectome analysis  │
│  (~20 tables + ~15 figures)        │
└────────────────────────────────────┘
```

---

## 3. Notebook `03_a` — COVID Transfer Inference Audit

**Purpose:** Apply the ADNI-trained β-VAE + classifier pipeline to a COVID
cohort (N=194) and evaluate AD-likeness scores with clinical, OOD, and
mechanistic analyses.  This is the main paper-grade analysis notebook.

### Execution order

```
1. scripts/inference_covid_from_adcn.py          (REQUIRED first)
2. notebooks/03_a  §1–§6                         (derives ADNI thresholds)
3. scripts/ood_policy_adni.py                     (optional, reads §6 thresholds)
4. scripts/analysis_covid_paper_fixes.py          (optional, enrichment tables)
5. notebooks/03_a  §5 onward (re-run)            (reads OOD + enrichment tables)
```

> **Circular dependency:** §6 of the notebook produces
> `adni_derived_thresholds.csv` that `ood_policy_adni.py` consumes.  Run
> the notebook through §6 first, then run the OOD script, then re-execute
> §5 of the notebook.

### Input files (essential)

| File | Producer |
|---|---|
| `results/vae_3channels_beta65_pro/run_config.json` | Training pipeline |
| `data/…/GLOBAL_TENSOR_from_COVID_….npz` | `feature_extraction_manual_COVID.py` |
| `data/SubjectsData_AAL3_COVID.csv` | External (clinical metadata) |
| `…/inference_covid_paper_output/Tables/covid_predictions_{per_fold,ensemble}.csv` | `inference_covid_from_adcn.py` |
| `…/Tables/covid_recon_error_{per_fold,ensemble}.csv` | `inference_covid_from_adcn.py` |
| `…/Tables/covid_latent_distance_{per_fold,ensemble}.csv` | `inference_covid_from_adcn.py` |
| `…/Tables/covid_signature_scores.csv` | `inference_covid_from_adcn.py` |
| `…/all_folds_clf_predictions_MULTI_*.csv` | Training pipeline (ADNI OOF) |
| `…/roi_info_from_tensor.csv` | Training pipeline |

### Input files (optional, graceful degradation)

| File | Producer | Used in |
|---|---|---|
| `…/Tables/adni_ood_distribution.csv` | `ood_policy_adni.py` | §5 |
| `…/Tables/ood_policy_sensitivity_summary.csv` | `ood_policy_adni.py` | §5 |
| `…/Tables/covid_ood_quadrants_logreg.csv` | `ood_policy_adni.py` | §5, §7c |
| `…/Tables/enrichment_signature_vs_top5pct_*.csv` | `analysis_covid_paper_fixes.py` | §12b |
| `…/interpretability_paper_output/…consensus_edges_*.csv` | `run_interpretability.py` | §8, §11, §12 |
| `fold_{k}/vae_model_fold_{k}.pt` | Training pipeline | §10 (latent UMAP) |

### Outputs written by the notebook

~20 tables + ~15 figures into `…/inference_covid_paper_output/{Tables,Figures}/`.
Full manifest in `…/notebook_manifest.json` + `…/outputs_index.csv`.

---

## 4. Notebook `03_b` — COVID Classification Probing Audit

**Purpose:** Audit and visualize a frozen-encoder probing experiment
(COVID vs CONTROL classification using ADNI β-VAE latent representations).
Purely post-hoc — all model training happens in the upstream script.

### Execution order

```
1. scripts/run_covid_classification_probing.py    (REQUIRED — generates all tables)
2. notebooks/03_b_covid_classification_probing_audit.ipynb  (visualization + AUC recomputation)
```

No circular dependencies.  No helper scripts needed.

### Input files (all essential, all from one script)

| File | Cell |
|---|---|
| `…/covid_classification_probing/Tables/metrics_by_fold_encoder_model.csv` | §1 |
| `…/Tables/oof_predictions_by_encoder_model.csv` | §1 |
| `…/Tables/oof_predictions_late_fusion.csv` | §1 |
| `…/Tables/summary_by_encoder_model.csv` | §1 |
| `…/Tables/summary_late_fusion.csv` | §1 |
| `…/Tables/feature_schema_summary.csv` | §1 |

### Inline computation

The notebook recomputes **late-fusion fold-wise AUC** from OOF predictions
(§4, using `sklearn.metrics.roc_auc_score`) as a verification step.
It also computes the PR-AUC random baseline from prevalence.
All other content is visualization.

### Outputs written by the notebook

6 audit figures (PNG + PDF) → `…/covid_classification_probing/Figures/audit/`.
No tables written.

---

## 5. Script Classification

| Script | Role | Feeds |
|---|---|---|
| `run_vae_clf_ad_inference.py` | **Core** — ADNI training pipeline | `03_run_vae_clf_ad_best`, `04_training_qc*` |
| `inference_covid_from_adcn.py` | **Core** — COVID transfer inference | `03_a` |
| `run_covid_classification_probing.py` | **Core** — COVID probing classification | `03_b` |
| `ood_policy_adni.py` | **Analysis helper** — OOD baseline calibration | `03_a` (optional) |
| `analysis_covid_paper_fixes.py` | **One-off patch** — paper fixes A–E | `03_a` (optional) |
| `run_interpretability.py` | **Core** (wrapper) — SHAP + saliency | `05_run_interpretability_ad` |
| `ablation_canales.py` | **Core** — channel selection ablation | `02_run_ablation` |
| `run_vae_clf_ad_ablation.py` | **Core** — ablation variant | `02_run_ablation` |
| `feature_extraction_manual.py` | **Core** — ADNI tensor construction | `01_feature_extraction_manual` |
| `feature_extraction_manual_COVID.py` | **Core** — COVID tensor construction | `01_feature_extraction_manual_COVID` |
| `activate_dev.sh` | **Utility** — sets PYTHONPATH | — |
