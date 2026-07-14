# SIPAIM Multisite Transport Executive Summary

## Guardrail Status

- no VAE training: true
- no OASIS preprocessing/inference: true
- no manuscript/protocol rewriting: true
- no pooling of raw latent coordinates across folds for site-geometry estimators: true

## TeX/Input Audit

- Working TeX exists: True
- Existing site-geometry PDFs found: 7 (combat_before_after_summary.pdf, combat_latent_geometry_shift.pdf, covariance_geometry_by_manufacturer.pdf, decoded_manufacturer_network_blocks.pdf, decoded_manufacturer_traversal_connectomes.pdf, latent_geometry_overview.pdf, manufacturer_performance_stageB.pdf)
- pdflatex pass 1 return code: 0
- pdflatex pass 2 return code: 0

## Main Findings

1. `SiteCode` reconstructed from `SSS_S_NNNN` SubjectID is valid for the final ADNI metadata line. Malformed IDs: 0; numeric Site3 mismatches: 0.
2. Fold-local latent `mu` retains acquisition-site information after residualizing diagnosis, Age, and Sex. Mean held-out site decoding balanced accuracy: 0.153.
3. Site-specific diagnostic transport is heterogeneous; unsupported small sites are explicitly listed in `site_exclusion_table`.
4. Frozen-latent classifier/readout leave-one-site-out is a sensitivity analysis, not VAE leave-one-site-out. Mean supported-site ROC-AUC: 0.698.
5. OASIS external distribution shift was computed from the existing promoted runwise164 fold-latent file. Mean OASIS-vs-ADNI trainDev sliced Wasserstein distance: 0.349; mean Gaussian Bures-Wasserstein: 11.625.

## Main Results Table

| result                             |   n_or_folds |   primary_value |   secondary_value | metric                                         |
|:-----------------------------------|-------------:|----------------:|------------------:|:-----------------------------------------------|
| manufacturer latent decodability   |            5 |        0.727556 |         0.0903949 | balanced accuracy mean, SD                     |
| site latent decodability           |            5 |        0.152726 |         0.031954  | balanced accuracy mean, SD                     |
| locked ADNI diagnostic performance |          397 |        0.795155 |         0.573934  | ROC-AUC, PR-AUC                                |
| fold-wise input-ComBat performance |          397 |        0.778076 |         0.549936  | ROC-AUC, PR-AUC                                |
| leave-one-site-out summary         |            2 |        0.697582 |         0.577467  | mean supported-site ROC-AUC, PR-AUC            |
| OASIS external performance         |          180 |        0.647778 |         0.667838  | runwise164 one-row-per-subject ROC-AUC, PR-AUC |
| ADNI-OASIS Wasserstein distance    |            5 |        0.348886 |        11.625     | mean sliced W, mean Gaussian Bures-W           |

