# Canonical shared-coordinate diagnostic-centroid geometry

Read-only consolidation of `results/sipaim_2026/final_blocker_resolution_20260712/diagnostic_centroid_geometry_revised.csv`. No new computation, no training, no inference, no Procrustes alignment. `results/sipaim_2026/final_evidence_package_20260712/diagnostic_axis_geometry.csv` was not read or used.

**Validity note**: Dataset-ComBat arms and locked transfer share the frozen VAE coordinates; the separately retrained Siemens-ComBat VAE does not. `previous_adni_fitted_siemens_combat` is therefore excluded from this shared-coordinate table (see `geometry_source_reconciliation.md`).

| arm                                    | role      |   n_cn |   n_ad |   adni_cn_proj_mean_ensembled |   adni_ad_proj_mean_ensembled |   oasis_cn_proj_mean |   oasis_ad_proj_mean |   marginal_shift |   marginal_shift_ci_low |   marginal_shift_ci_high |   cn_shift |   cn_shift_ci_low |   cn_shift_ci_high |   ad_shift |   ad_shift_ci_low |   ad_shift_ci_high |   ad_cn_projected_separation |   ad_cn_sep_ci_low |   ad_cn_sep_ci_high |   cohens_d |   cohens_d_ci_low |   cohens_d_ci_high |   projected_auc |   projected_auc_ci_low |   projected_auc_ci_high |   clinical_alignment_margin |   clinical_alignment_margin_ci_low |   clinical_alignment_margin_ci_high |
|:---------------------------------------|:----------|-------:|-------:|------------------------------:|------------------------------:|---------------------:|---------------------:|-----------------:|------------------------:|-------------------------:|-----------:|------------------:|-------------------:|-----------:|------------------:|-------------------:|-----------------------------:|-------------------:|--------------------:|-----------:|------------------:|-------------------:|----------------:|-----------------------:|------------------------:|----------------------------:|-----------------------------------:|------------------------------------:|
| locked_frozen_transfer                 | primary   |     90 |     90 |                       -1.9104 |                        1.9104 |              -2.4306 |              -1.4828 |          -1.9567 |                 -2.2252 |                  -1.6866 |    -0.5203 |           -0.8878 |            -0.1465 |    -3.3932 |           -3.7862 |            -3.0082 |                       0.9478 |             0.4036 |              1.4797 |     0.5163 |            0.2224 |             0.8185 |          0.6372 |                 0.5523 |                  0.7164 |                     -1.4828 |                            -1.8758 |                             -1.0979 |
| external_dataset_combat_adni_reference | primary   |     90 |     90 |                       -1.9104 |                        1.9104 |              -1.5058 |              -0.4747 |          -0.9902 |                 -1.2821 |                  -0.6981 |     0.4046 |            0.0045 |             0.8119 |    -2.385  |           -2.8098 |            -1.9738 |                       1.0311 |             0.436  |              1.6137 |     0.5182 |            0.2226 |             0.8224 |          0.6368 |                 0.5517 |                  0.7163 |                     -0.4747 |                            -0.8995 |                             -0.0635 |
| external_dataset_combat_no_reference   | secondary |     90 |     90 |                       -1.9104 |                        1.9104 |              -1.8859 |              -0.8831 |          -1.3845 |                 -1.6687 |                  -1.0984 |     0.0245 |           -0.364  |             0.422  |    -2.7935 |           -3.212  |            -2.3884 |                       1.0027 |             0.425  |              1.5698 |     0.5158 |            0.2194 |             0.8192 |          0.6369 |                 0.5516 |                  0.7162 |                     -0.8831 |                            -1.3016 |                             -0.478  |

## Arms included

- `locked_frozen_transfer` (primary)
- `external_dataset_combat_adni_reference` (primary)
- `external_dataset_combat_no_reference` (**secondary**)

## Arm excluded

- `previous_adni_fitted_siemens_combat` -- see validity note above.

## Preserved quantities (all from the revised source, unmodified)

OASIS-prevalence-weighted ADNI centroid (`adni_cn_proj_mean_ensembled` / `adni_ad_proj_mean_ensembled`), class-conditional CN and AD shifts (`cn_shift` / `ad_shift`), marginal shift, AD-CN projected separation, Cohen's d, projected AUC, clinical-alignment margin, and all subject-bootstrap 95% CIs (n_boot=10000, seed=20260710, unchanged from source).
