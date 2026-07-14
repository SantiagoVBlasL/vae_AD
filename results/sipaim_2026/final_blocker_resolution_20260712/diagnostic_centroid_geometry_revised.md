# Diagnostic-centroid geometry, revised

Read-only. **Renamed and reformulated axis**: this is a **fold-local ADNI centroid-derived discriminant axis** (AD train/dev centroid minus CN train/dev centroid, per fold, in the frozen locked `latent_mu` space) -- explicitly *not* the classifier's decision-boundary weight vector (that vector also includes learned Age/Sex terms and lives in scaled-feature space; see `score_component_audit.md` for that separate object). Coordinates are never pooled across the five independently-trained fold encoders -- every centroid, axis, and projection below is computed fold-locally; only the resulting **scalar** projections are averaged across folds for the ensemble view (same convention as the project's score ensembling), never raw 384-dim vectors.

The ADNI reference point each fold's axis is centered on is a **class-prevalence-weighted centroid** (0.5*CN + 0.5*AD), matching OASIS current-180's exact 50/50 composition -- not ADNI's own imbalanced raw sample composition (ADNI train/dev pools are roughly 240 CN / 78 AD per fold; see `_adni_fold_axis_reference.json`).

## Ensemble results (mean scalar projection across 5 folds per subject, subject-bootstrap 95% CI, n_boot=10000, seed=20260710)

| arm                                    | role      |   oasis_cn_proj_mean |   oasis_ad_proj_mean |   marginal_shift |   marginal_shift_ci_low |   marginal_shift_ci_high |   cn_shift |   cn_shift_ci_low |   cn_shift_ci_high |   ad_shift |   ad_shift_ci_low |   ad_shift_ci_high |   ad_cn_projected_separation |   ad_cn_sep_ci_low |   ad_cn_sep_ci_high |   cohens_d |   cohens_d_ci_low |   cohens_d_ci_high |   projected_auc |   projected_auc_ci_low |   projected_auc_ci_high |   clinical_alignment_margin |   clinical_alignment_margin_ci_low |   clinical_alignment_margin_ci_high |
|:---------------------------------------|:----------|---------------------:|---------------------:|-----------------:|------------------------:|-------------------------:|-----------:|------------------:|-------------------:|-----------:|------------------:|-------------------:|-----------------------------:|-------------------:|--------------------:|-----------:|------------------:|-------------------:|----------------:|-----------------------:|------------------------:|----------------------------:|-----------------------------------:|------------------------------------:|
| locked_frozen_transfer                 | primary   |              -2.4306 |              -1.4828 |          -1.9567 |                 -2.2252 |                  -1.6866 |    -0.5203 |           -0.8878 |            -0.1465 |    -3.3932 |           -3.7862 |            -3.0082 |                       0.9478 |             0.4036 |              1.4797 |     0.5163 |            0.2224 |             0.8185 |          0.6372 |                 0.5523 |                  0.7164 |                     -1.4828 |                            -1.8758 |                             -1.0979 |
| previous_adni_fitted_siemens_combat    | primary   |              -0.185  |              -0.0718 |          -0.1284 |                 -0.196  |                  -0.06   |     1.7254 |            1.6215 |             1.8311 |    -1.9822 |           -2.0686 |            -1.8943 |                       0.1132 |            -0.0241 |              0.2515 |     0.2414 |           -0.0503 |             0.5446 |          0.5656 |                 0.4809 |                  0.6493 |                     -0.0718 |                            -0.1583 |                              0.0161 |
| external_dataset_combat_adni_reference | primary   |              -1.5058 |              -0.4747 |          -0.9902 |                 -1.2821 |                  -0.6981 |     0.4046 |            0.0045 |             0.8119 |    -2.385  |           -2.8098 |            -1.9738 |                       1.0311 |             0.436  |              1.6137 |     0.5182 |            0.2226 |             0.8224 |          0.6368 |                 0.5517 |                  0.7163 |                     -0.4747 |                            -0.8995 |                             -0.0635 |
| external_dataset_combat_no_reference   | secondary |              -1.8859 |              -0.8831 |          -1.3845 |                 -1.6687 |                  -1.0984 |     0.0245 |           -0.364  |             0.422  |    -2.7935 |           -3.212  |            -2.3884 |                       1.0027 |             0.425  |              1.5698 |     0.5158 |            0.2194 |             0.8192 |          0.6369 |                 0.5516 |                  0.7162 |                     -0.8831 |                            -1.3016 |                             -0.478  |

## Metric definitions

- **marginal_shift**: prevalence-weighted OASIS mean projection minus 0 (ADNI's own prevalence-weighted mean is 0 by construction, since the axis is centered on it).
- **cn_shift / ad_shift**: OASIS class-conditional mean projection minus the matched ADNI class-conditional mean projection -- the class-conditional (CN-CN, AD-AD) shifts prioritized per the task spec, as opposed to a generic marginal/pooled shift.
- **ad_cn_projected_separation**: OASIS AD mean minus OASIS CN mean, along the axis -- compare to `adni_ad_cn_separation_at_source` (fold-level file) for the ADNI-native scale.
- **cohens_d**: ad_cn_projected_separation divided by the pooled within-OASIS-class SD along the projection.
- **projected_auc**: ROC-AUC of OASIS CN/AD labels against the raw projection score (not the actual downstream classifier's score -- a purely geometric quantity).
- **clinical_alignment_margin**: min(OASIS AD mean, -OASIS CN mean) -- how far the *nearer* class sits from the ADNI-centroid-based zero point, signed so that a naive fixed-zero decision rule inherited directly from ADNI would work only if this is positive. Negative values (all four arms here) mean neither class's mean fully crosses to its expected side of the ADNI-derived zero point -- see Reading below.

## Paired fold deltas (arm minus locked; n=5 folds; also see `_paired_fold_deltas_vs_locked.csv` for the raw per-fold values)

| arm                                    | metric                     |   mean_delta |   sd_delta |   n_folds_arm_lower_than_locked |   n_folds_arm_higher_than_locked |
|:---------------------------------------|:---------------------------|-------------:|-----------:|--------------------------------:|---------------------------------:|
| previous_adni_fitted_siemens_combat    | marginal_shift             |       1.8283 |     0.6787 |                               0 |                                5 |
| previous_adni_fitted_siemens_combat    | cn_shift                   |       2.2457 |     0.8229 |                               0 |                                5 |
| previous_adni_fitted_siemens_combat    | ad_shift                   |       1.411  |     0.5756 |                               0 |                                5 |
| previous_adni_fitted_siemens_combat    | ad_cn_projected_separation |      -0.8347 |     0.4175 |                               5 |                                0 |
| previous_adni_fitted_siemens_combat    | cohens_d                   |      -0.4644 |     0.4182 |                               5 |                                0 |
| previous_adni_fitted_siemens_combat    | projected_auc              |      -0.1279 |     0.1148 |                               5 |                                0 |
| previous_adni_fitted_siemens_combat    | clinical_alignment_margin  |       1.169  |     0.2606 |                               0 |                                5 |
| external_dataset_combat_adni_reference | marginal_shift             |       0.9665 |     0.1974 |                               0 |                                5 |
| external_dataset_combat_adni_reference | cn_shift                   |       0.9248 |     0.2009 |                               0 |                                5 |
| external_dataset_combat_adni_reference | ad_shift                   |       1.0081 |     0.1945 |                               0 |                                5 |
| external_dataset_combat_adni_reference | ad_cn_projected_separation |       0.0833 |     0.0226 |                               0 |                                5 |
| external_dataset_combat_adni_reference | cohens_d                   |       0.0019 |     0.0165 |                               2 |                                3 |
| external_dataset_combat_adni_reference | projected_auc              |      -0.0014 |     0.0034 |                               4 |                                1 |
| external_dataset_combat_adni_reference | clinical_alignment_margin  |       1.0081 |     0.1945 |                               0 |                                5 |
| external_dataset_combat_no_reference   | marginal_shift             |       0.5722 |     0.1084 |                               0 |                                5 |
| external_dataset_combat_no_reference   | cn_shift                   |       0.5448 |     0.1107 |                               0 |                                5 |
| external_dataset_combat_no_reference   | ad_shift                   |       0.5997 |     0.1071 |                               0 |                                5 |
| external_dataset_combat_no_reference   | ad_cn_projected_separation |       0.0549 |     0.0198 |                               0 |                                5 |
| external_dataset_combat_no_reference   | cohens_d                   |      -0.0005 |     0.0123 |                               2 |                                3 |
| external_dataset_combat_no_reference   | projected_auc              |      -0.0003 |     0.002  |                               3 |                                2 |
| external_dataset_combat_no_reference   | clinical_alignment_margin  |       0.5997 |     0.1071 |                               0 |                                5 |

## Reading

**All four arms show OASIS's overall projection shifted toward the CN end relative to ADNI** (negative clinical-alignment margin in every arm) -- even OASIS's own AD-severity mean sits, on average, on the nominally-CN side of the ADNI-derived zero point. This does not contradict the models' positive OASIS ROC-AUCs elsewhere in this project: AUC is a ranking measure insensitive to this kind of uniform offset, but it does mean a decision rule that simply reused the ADNI zero-point as a raw threshold on this specific axis would misclassify most OASIS subjects as CN -- a concrete illustration of *why* the project's actual downstream classifiers use fitted (OOF-ECDF / inner-CV) thresholds rather than the geometric ADNI centroid directly.

**A clean bias/discriminability tradeoff separates the two ComBat families:**

- `previous_adni_fitted_siemens_combat` **nearly eliminates the marginal shift** (mean delta -0.128 vs locked's -1.957 magnitude; margin moves from -1.48 to -0.07, closest to zero of all four arms) but **at a heavy discriminability cost** -- AD-CN projected separation collapses (mean delta -0.835, worse in all 5 folds), Cohen's d roughly halves (0.516 -> 0.241), and projected AUC drops by 0.128 (worse in all 5 folds).
- `external_dataset_combat_adni_reference` and `external_dataset_combat_no_reference` (secondary) both correct the marginal shift **moderately** (mean delta +0.97 / +0.57 toward zero) while **preserving discriminability almost exactly** -- Cohen's d and projected AUC are statistically indistinguishable from locked in both arms.

This is a genuinely new geometric finding not previously reported in these terms: **Dataset-ComBat (either reference variant) buys some domain-shift correction without the discriminability cost that the ADNI-fitted Siemens-ComBat arm pays**, a tradeoff visible only once CN-CN and AD-AD shifts are separated from the pooled/marginal shift.
