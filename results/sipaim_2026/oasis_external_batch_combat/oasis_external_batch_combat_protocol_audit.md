# OASIS external Dataset-batch ComBat protocol audit

## Library support

- `neuroCombat.neuroCombat` is installed and has signature `(dat, covars, batch_col, categorical_cols=None, continuous_cols=None, eb=True, parametric=True, mean_only=False, ref_batch=None)`.
- Reference-batch harmonisation support: **True** via `ref_batch`.
- `neuroHarmonize` is not required for this run and was not used.
- `neurocombat_sklearn.CombatModel` was not used because it does not expose a reference-batch argument in this environment.

## Data and leakage controls

- Batch variable for the new harmoniser: `Dataset`, levels `ADNI` and `OASIS`.
- Preserved covariates: `Age`, `Sex`.
- Excluded covariates: `Diagnosis`, `ResearchGroup_Mapped`.
- OASIS labels were not included in the ComBat covariate frame and were used only for final metric evaluation.
- No VAE checkpoint was trained or updated.
- No OASIS calibration, threshold selection, feature selection, or diagnostic model selection was performed.
- The downstream readout follows the existing locked OASIS scoring convention: ADNI train/dev-only readout reconstruction and locked inner-OOF ECDF threshold transfer. This is documented as a limitation because the official locked downstream readout estimators were not serialized as frozen objects.

## Variants

1. Dataset-level ComBat without reference batch.
2. Dataset-level ComBat with `ADNI` as reference batch.

## Fit audit summary

| arm                                    |   fold | channel_name                     | ref_batch   |   n_adni_train_dev |   n_oasis |   n_variable_features | all_finite   |   max_abs_diagonal_delta_adni |   max_abs_diagonal_delta_oasis |
|:---------------------------------------|-------:|:---------------------------------|:------------|-------------------:|----------:|----------------------:|:-------------|------------------------------:|-------------------------------:|
| external_dataset_combat_no_reference   |      1 | Pearson_Full_FisherZ_Signed      | none        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      1 | Pearson_OMST_GCE_Signed_Weighted | none        |                317 |       180 |                  7928 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      1 | MI_KNN_Symmetric                 | none        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      2 | Pearson_Full_FisherZ_Signed      | none        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      2 | Pearson_OMST_GCE_Signed_Weighted | none        |                317 |       180 |                  7928 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      2 | MI_KNN_Symmetric                 | none        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      3 | Pearson_Full_FisherZ_Signed      | none        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      3 | Pearson_OMST_GCE_Signed_Weighted | none        |                318 |       180 |                  7923 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      3 | MI_KNN_Symmetric                 | none        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      4 | Pearson_Full_FisherZ_Signed      | none        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      4 | Pearson_OMST_GCE_Signed_Weighted | none        |                318 |       180 |                  7927 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      4 | MI_KNN_Symmetric                 | none        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      5 | Pearson_Full_FisherZ_Signed      | none        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      5 | Pearson_OMST_GCE_Signed_Weighted | none        |                318 |       180 |                  7892 | True         |                             0 |                              0 |
| external_dataset_combat_no_reference   |      5 | MI_KNN_Symmetric                 | none        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      1 | Pearson_Full_FisherZ_Signed      | ADNI        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      1 | Pearson_OMST_GCE_Signed_Weighted | ADNI        |                317 |       180 |                  7928 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      1 | MI_KNN_Symmetric                 | ADNI        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      2 | Pearson_Full_FisherZ_Signed      | ADNI        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      2 | Pearson_OMST_GCE_Signed_Weighted | ADNI        |                317 |       180 |                  7928 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      2 | MI_KNN_Symmetric                 | ADNI        |                317 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      3 | Pearson_Full_FisherZ_Signed      | ADNI        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      3 | Pearson_OMST_GCE_Signed_Weighted | ADNI        |                318 |       180 |                  7923 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      3 | MI_KNN_Symmetric                 | ADNI        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      4 | Pearson_Full_FisherZ_Signed      | ADNI        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      4 | Pearson_OMST_GCE_Signed_Weighted | ADNI        |                318 |       180 |                  7927 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      4 | MI_KNN_Symmetric                 | ADNI        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      5 | Pearson_Full_FisherZ_Signed      | ADNI        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      5 | Pearson_OMST_GCE_Signed_Weighted | ADNI        |                318 |       180 |                  7892 | True         |                             0 |                              0 |
| external_dataset_combat_adni_reference |      5 | MI_KNN_Symmetric                 | ADNI        |                318 |       180 |                  8515 | True         |                             0 |                              0 |
