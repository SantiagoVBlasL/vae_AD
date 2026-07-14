# Canonical shared-coordinate paired fold deltas (arm minus locked)

Read-only consolidation of `results/sipaim_2026/final_blocker_resolution_20260712/_paired_fold_deltas_vs_locked.csv`, restricted to the two arms that share the locked model's frozen VAE coordinates. `previous_adni_fitted_siemens_combat`'s paired-delta rows are excluded for the same reason given in `canonical_shared_locked_geometry.md` and `geometry_source_reconciliation.md`.

| arm                                    | metric                     |   n_folds |   mean_delta |   sd_delta |   min_delta |   max_delta |   n_folds_arm_lower_than_locked |   n_folds_arm_higher_than_locked |
|:---------------------------------------|:---------------------------|----------:|-------------:|-----------:|------------:|------------:|--------------------------------:|---------------------------------:|
| external_dataset_combat_adni_reference | marginal_shift             |         5 |       0.9665 |     0.1974 |      0.693  |      1.2045 |                               0 |                                5 |
| external_dataset_combat_adni_reference | cn_shift                   |         5 |       0.9248 |     0.2009 |      0.6555 |      1.1808 |                               0 |                                5 |
| external_dataset_combat_adni_reference | ad_shift                   |         5 |       1.0081 |     0.1945 |      0.7305 |      1.2281 |                               0 |                                5 |
| external_dataset_combat_adni_reference | ad_cn_projected_separation |         5 |       0.0833 |     0.0226 |      0.0473 |      0.1013 |                               0 |                                5 |
| external_dataset_combat_adni_reference | cohens_d                   |         5 |       0.0019 |     0.0165 |     -0.0222 |      0.023  |                               2 |                                3 |
| external_dataset_combat_adni_reference | projected_auc              |         5 |      -0.0014 |     0.0034 |     -0.0068 |      0.0027 |                               4 |                                1 |
| external_dataset_combat_adni_reference | clinical_alignment_margin  |         5 |       1.0081 |     0.1945 |      0.7305 |      1.2281 |                               0 |                                5 |
| external_dataset_combat_no_reference   | marginal_shift             |         5 |       0.5722 |     0.1084 |      0.411  |      0.6925 |                               0 |                                5 |
| external_dataset_combat_no_reference   | cn_shift                   |         5 |       0.5448 |     0.1107 |      0.3884 |      0.6796 |                               0 |                                5 |
| external_dataset_combat_no_reference   | ad_shift                   |         5 |       0.5997 |     0.1071 |      0.4335 |      0.7054 |                               0 |                                5 |
| external_dataset_combat_no_reference   | ad_cn_projected_separation |         5 |       0.0549 |     0.0198 |      0.0258 |      0.0766 |                               0 |                                5 |
| external_dataset_combat_no_reference   | cohens_d                   |         5 |      -0.0005 |     0.0123 |     -0.0187 |      0.0153 |                               2 |                                3 |
| external_dataset_combat_no_reference   | projected_auc              |         5 |      -0.0003 |     0.002  |     -0.0026 |      0.0023 |                               3 |                                2 |
| external_dataset_combat_no_reference   | clinical_alignment_margin  |         5 |       0.5997 |     0.1071 |      0.4335 |      0.7054 |                               0 |                                5 |

Full per-fold delta arrays are preserved unchanged in `canonical_geometry_paired_deltas.csv` (`per_fold_deltas` column, n=5 folds each).
