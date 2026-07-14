# Main Results Table

| result                             |   n_or_folds |   primary_value |   secondary_value | metric                                         |
|:-----------------------------------|-------------:|----------------:|------------------:|:-----------------------------------------------|
| manufacturer latent decodability   |            5 |        0.727556 |         0.0903949 | balanced accuracy mean, SD                     |
| site latent decodability           |            5 |        0.152726 |         0.031954  | balanced accuracy mean, SD                     |
| locked ADNI diagnostic performance |          397 |        0.795155 |         0.573934  | ROC-AUC, PR-AUC                                |
| fold-wise input-ComBat performance |          397 |        0.778076 |         0.549936  | ROC-AUC, PR-AUC                                |
| leave-one-site-out summary         |            2 |        0.697582 |         0.577467  | mean supported-site ROC-AUC, PR-AUC            |
| OASIS external performance         |          180 |        0.647778 |         0.667838  | runwise164 one-row-per-subject ROC-AUC, PR-AUC |
| ADNI-OASIS Wasserstein distance    |            5 |        0.348886 |        11.625     | mean sliced W, mean Gaussian Bures-W           |
