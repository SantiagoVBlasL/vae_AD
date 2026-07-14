# Final Recommendation

This is a read-only external inference audit. The ADNI fold VAEs and ADNI-only Stage B OOF-ECDF readout protocol were applied to mega-OASIS tensors. No OASIS scaler, classifier, threshold, calibration, residualization, ComBat transform, or model-selection step was fit.

Primary model best mega-OASIS build: `runwise164_pilot_parity` with AUC=0.6478, PR-AUC=0.6678, BA=0.5667, F1=0.3810.
Best panel entry by OASIS AUC was `promoted_beta3p75_oof_ecdf` on `runwise164_pilot_parity` (AUC=0.6478, PR-AUC=0.6678). This is not a promotion criterion because OASIS is external evaluation, not model selection.

The ADNI-vs-OASIS PR-AUC arithmetic deltas should be interpreted with prevalence context: mega-OASIS is balanced 90/90, while the ADNI classifier cohort is AD-minority. AUROC is the cleaner ranking-transfer comparison across cohorts.

Manuscript recommendation:
- Report `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` as the primary ADNI model.
- Report mfrBalancedVAE, latent512, and manufacturer-residualized readouts only as external sensitivity analyses if their artifact validation passed.
- Interpret OASIS as an external stress test of transferability. It may support a ranking signal when AUC is above chance, but any sensitivity-model advantage is post-hoc and should not override the internally promoted ADNI model.

Primary/sensitivity labels were assigned in `interpretation_labels.csv`.
