# Data-Flow Diagram: COVID → AD Inference Notebook (v3.0)

`notebooks/03_a_inference_covid_from_adcn.ipynb` — 28 cells, 10 sections.

---

## 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  External Artifacts (produced by inference script + training pipeline)  │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
    ┌────────────┼──────────────────────────────────┐
    │            │                                  │
    ▼            ▼                                  ▼
┌────────┐  ┌───────────────┐          ┌──────────────────────┐
│ COVID  │  │ ADNI OOF      │          │ Consensus Edges      │
│ Infer- │  │ Predictions   │          │ (logreg IG top50)    │
│ ence   │  │ (5-fold CV)   │          │ 1 026 edges × 17 col│
│ Tables │  │ 183 rows LR   │          │ Yeo-17 network labels│
│ (7 CSV)│  └──────┬────────┘          └──────────┬───────────┘
└───┬────┘         │                               │
    │              │                               │
    │    ┌─────────┼───────────────────────────────┼─────────────────┐
    │    │         ▼                               ▼                 │
    │    │  §6 ROC Analysis              §8b Network Decomposition  │
    │    │  ┌─────────────────┐          ┌─────────────────────┐    │
    │    │  │ 4 Threshold     │          │ S_sig by Yeo-17     │    │
    │    │  │ Policies:       │          │ network pair         │    │
    │    │  │ · Youden        │          │ (heatmap + CSV)      │    │
    │    │  │ · Screening     │          └─────────────────────┘    │
    │    │  │ · FixedFPR      │                                     │
    │    │  │ · CostBased     │                                     │
    │    │  └────────┬────────┘                                     │
    │    │           │                                              │
    │    └───────────┼──────────────────────────────────────────────┘
    │                │
    ▼                ▼
┌────────────────────────────────────────┐   ┌──────────────────────────┐
│  §4–§5  Core Inference + OOD          │   │  COVID Clinical Metadata │
│  · Score distributions + bootstrap CI │   │  SubjectsData_AAL3_COVID │
│  · ICC(2,1), vote entropy             │   │  214 rows × 19 cols      │
│  · Quadrant labelling (4 categories)  │   │  Join key: meta['ID']    │
│  · Mahalanobis + recon error          │   └────────────┬─────────────┘
└──────────────────┬─────────────────────┘               │
                   │                                     │
                   │         ┌────────────────────┐      │
                   └────────►│  §6b Apply          │◄────┘
                             │  Thresholds         │
                             │  + Risk Categories  │
                             └─────────┬───────────┘
                                       │
              ┌────────────────────────┬┴──────────────────────────┐
              │                        │                           │
              ▼                        ▼                           ▼
     §7b Correlations         §7c Group Tests          §7d Multivariate OLS
     ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
     │ Spearman     │         │ Mann-Whitney │         │ score ~ MOCA +   │
     │ + CI (10k BS)│         │ Kruskal-W    │         │   Age + Sex +    │
     │ + BH-FDR     │         │ + effect sz  │         │   FAS + severity │
     │ (8 tests)    │         │ + BH-FDR     │         │   + recovery     │
     └──────────────┘         └──────────────┘         │ HC3 robust SEs   │
                                                       │ + VIF + interact │
                                                       └──────────────────┘
              │                        │                           │
              ▼                        ▼                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                    §F  Outputs Index                         │
        │   18 Tables  ·  10 Figures  ·  3 JSON manifests              │
        └──────────────────────────────────────────────────────────────┘
```

---

## 2. Section Dependency Graph

```
§0 Config
  │
  ▼
§1 Setup (imports, utils)
  │
  ▼
§2 Artifact Discovery ──────────────────────────────────────────┐
  │ loads: 7 inference CSVs, metadata, ADNI OOF, consensus edges│
  ├───────────┬───────────┬───────────┬─────────────────────────┘
  │           │           │           │
  ▼           │           │           │
§3 Physics    │           │           │
  │           │           │           │
  ▼           ▼           │           │
§4a Scores   §4b ICC      │           │
  │           │           │           │
  └─────┬─────┘           │           │
        │                 │           │
        ▼                 │           │
§5 OOD Quadrants          │           │
  │  (uses: ensemble preds, recon error, Mahalanobis)
  │                       │           │
  ├───────────────────────┤           │
  │                       │           │
  ▼                       ▼           │
§6a Thresholds           §6b Apply    │
  │  (ADNI OOF → ROC)     │  (COVID)  │
  │                       │           │
  ├───────────────────────┤           │
  │                       │           │
  ▼                       │           │
§7a Clinical Merge ◄──────┘           │
  │  (join: SubjectID ↔ meta['ID'])   │
  │                                   │
  ├────────┬──────────┐               │
  │        │          │               │
  ▼        ▼          ▼               │
§7b       §7c        §7d             │
Corr      Groups     OLS             │
  │        │          │               │
  └───┬────┴──────────┘               │
      │                               │
      │         ┌─────────────────────┘
      │         │
      ▼         ▼
§8a Signature  §8b Network Decomp
  │  (S_sig)     │  (consensus edges × Yeo-17)
  │              │
  └──────┬───────┘
         │
         ▼
§9 Subject Selection
  │  (uses: Youden threshold, OOD quadrant, clinical annotation)
  │
  ▼
QC Calibration Drift + Sensitivity
  │  (ADNI OOF vs COVID distributions, KS tests)
  │
  ▼
§F Outputs Index + Manifest
```

---

## 3. File-Level I/O Map

### Inputs (read by notebook)

| File | Loaded in | Purpose |
|------|-----------|---------|
| `inference_covid_paper_output/Tables/covid_predictions_per_fold.csv` | §2 | Per-fold P(AD) |
| `inference_covid_paper_output/Tables/covid_predictions_ensemble.csv` | §2 | Ensemble P(AD) |
| `inference_covid_paper_output/Tables/covid_recon_error_per_fold.csv` | §2 | Per-fold recon error |
| `inference_covid_paper_output/Tables/covid_recon_error_ensemble.csv` | §2 | Ensemble recon error |
| `inference_covid_paper_output/Tables/covid_latent_distance_per_fold.csv` | §2 | Per-fold Mahalanobis |
| `inference_covid_paper_output/Tables/covid_latent_distance_ensemble.csv` | §2 | Ensemble Mahalanobis |
| `inference_covid_paper_output/Tables/covid_signature_scores.csv` | §2 | S_sig per subject |
| `data/SubjectsData_AAL3_COVID.csv` | §2 | Clinical metadata (MOCA, FAS, etc.) |
| `all_folds_clf_predictions_MULTI_svm_*.csv` | §2 | ADNI out-of-fold predictions |
| `interpretability_paper_output/tables/consensus_edges_logreg_*.csv` | §2 | Consensus edges with Yeo-17 |

### Outputs (written by notebook)

| File | Written in | Description |
|------|------------|-------------|
| `Tables/adni_derived_thresholds.csv` | §6a | 4 threshold policies |
| `Tables/covid_ood_quadrants_logreg.csv` | §5 | Quadrant labels per subject |
| `Tables/covid_threshold_analysis_logreg.csv` | §6b | Per-subject multi-threshold predictions |
| `Tables/clinical_analysis_cohort_logreg.csv` | §7a | Merged clinical+inference cohort |
| `Tables/clinical_correlations.csv` | §7b | Spearman + CI + FDR results |
| `Tables/clinical_group_comparisons.csv` | §7c | Group tests + effect sizes |
| `Tables/sensitivity_moca_by_threshold.csv` | QC | MOCA stats by threshold |
| `Tables/signature_network_decomposition.csv` | §8b | Yeo-17 pair contributions |
| `Figures/fig1–fig10_*.png` | various | 10 publication figures |
| `notebook_manifest.json` | §F | Provenance & inventory |
