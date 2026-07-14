# CDR temporal-sensitivity metrics, per mapping

Score = locked frozen promoted `[1,0,2]` ensemble score (unchanged, no recalibration). Pairwise AUC compares CDR 0 vs CDR>=1 (high severity), CDR 0.5 excluded from the pairwise comparison but included in the Spearman/partial-Spearman association (which uses the full CDRTOT ordinal: 0, 0.5, 1, 2, 3). Bootstrap: subject-level, n_boot=10000, seed=20260710. Partial Spearman controls for Age and Sex via rank-residual regression.

| mapping              |   n_mapped |   n_unmapped |   n_cdr0 |   n_cdr05 |   n_cdr_ge1 |   n_pairwise_cdr0_vs_ge1 |   pairwise_auc |   pairwise_auc_boot_mean |   pairwise_auc_ci_low |   pairwise_auc_ci_high |   n_boot_valid |   spearman_rho_raw |   spearman_pvalue_raw |   partial_spearman_rho_age_sex_adj |   partial_spearman_pvalue_age_sex_adj |   mean_abs_day_gap |   max_abs_day_gap |
|:---------------------|-----------:|-------------:|---------:|----------:|------------:|-------------------------:|---------------:|-------------------------:|----------------------:|-----------------------:|---------------:|-------------------:|----------------------:|-----------------------------------:|--------------------------------------:|-------------------:|------------------:|
| A_nearest_180d       |        169 |           11 |       81 |        51 |          37 |                      118 |         0.7297 |                   0.7309 |                0.6289 |                 0.8241 |          10000 |             0.3113 |                0      |                             0.2784 |                                0.0002 |            77.3609 |               177 |
| B_nearest_365d       |        180 |            0 |       89 |        52 |          39 |                      128 |         0.7093 |                   0.7092 |                0.6074 |                 0.7998 |          10000 |             0.2816 |                0.0001 |                             0.2439 |                                0.001  |            84.75   |               265 |
| C_latest_before_365d |        176 |            4 |       88 |        55 |          33 |                      121 |         0.7249 |                   0.7246 |                0.6214 |                 0.8199 |          10000 |             0.2828 |                0.0001 |                             0.2394 |                                0.0014 |           110.421  |               358 |

## Cross-check against the prior CDR sensitivity audit

Mapping B (nearest within 365 days) reproduces `results/sipaim_2026/oasis_label_sensitivity_20260710/` exactly: same group counts (CDR0=89, CDR0.5=52, CDR>=1=39) and the same pairwise ROC-AUC (0.709306). This confirms the original CDR assignment was, in effect, already a nearest-within-365-days mapping, even though that window was not stated explicitly at the time.

## Reading

- Tightening the window to +/-180 days (mapping A) drops 11 scans but raises the pairwise AUC to 0.730 (vs 0.709 for the looser 365-day window) -- consistent with CDR-to-scan temporal proximity mattering, not just CDR value itself.
- The backward-only mapping (C) gives a similar pairwise AUC (0.725) to the nearest-365d mapping, with larger day gaps on average (mean 110d vs 85d) since it cannot use a closer future visit.
- All three mappings retain a significant raw Spearman association (p<0.001) that survives Age/Sex adjustment (partial rho 0.24-0.28, p<0.002 in all three) -- the CDR-severity/score gradient is not an Age or Sex confound.
