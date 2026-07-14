# Reconciled Main Results Table

| result                             |   n_or_folds |   primary_value |   secondary_value | metric                                             | source_policy                                                 |
|:-----------------------------------|-------------:|----------------:|------------------:|:---------------------------------------------------|:--------------------------------------------------------------|
| manufacturer latent decodability   |            5 |        0.66586  |         0.0506202 | matched fold-local 5-NN balanced accuracy mean, SD | primary estimate selected for direct SiteCode comparison      |
| site latent decodability           |            5 |        0.152726 |         0.031954  | fold-local SiteCode balanced accuracy mean, SD     | observed BA unchanged; p-values rerun with 1000 permutations  |
| locked ADNI diagnostic performance |          397 |        0.795155 |         0.573934  | ROC-AUC, PR-AUC                                    | final selected StageB OOF-ECDF primary convention             |
| fold-wise input-ComBat performance |          397 |        0.778076 |         0.549936  | ROC-AUC, PR-AUC                                    | existing foldwise input-ComBat StageB OOF-ECDF                |
| leave-one-site-out summary         |            2 |        0.697582 |         0.577467  | mean supported-site ROC-AUC, PR-AUC                | classifier/readout transport on frozen latent representations |
| OASIS external performance         |          180 |        0.647778 |         0.667838  | runwise164 one-row-per-subject ROC-AUC, PR-AUC     | one ensemble row per subject/session                          |
| ADNI-OASIS Wasserstein distance    |            5 |        0.348886 |        11.625     | mean sliced W, mean Gaussian Bures-W               | existing fold-local ADNI trainDev to OASIS latent distance    |
