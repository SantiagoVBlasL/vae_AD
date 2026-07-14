# Final SIPAIM Claims After Metric Reconciliation

1. The SIPAIM table should use the matched fold-local manufacturer estimate for direct comparison with SiteCode decoding: manufacturer BA = 0.6659 +/- 0.0506. The older 0.7276 +/- 0.0904 value is a scanner-leakage QC estimate from within-test CV and should not be averaged with the fold-local transport estimate.

2. SiteCode decoding observed values are unchanged: BA = 0.1527 +/- 0.0320. The site-label permutation test was rerun with 1000 permutations preserving `ResearchGroup_Mapped + Manufacturer` strata; fold p-values now have resolution about 0.001, with p-values spanning 0.0010 to 0.0629.

3. SiteCode is serialized as a three-character string in the generated SIPAIM CSV files checked here. No source metadata was modified.

4. The authoritative OASIS build remains `runwise164_pilot_parity`, N=180 with 90 CN and 90 AD, one ensemble row per subject/session. The final OASIS ROC-AUC/PR-AUC are computed once from subject/session-level ensemble scores, not as mean fold metrics and not as run-level metrics.

5. Safe wording for the four-page SIPAIM manuscript: latent representations retain measurable acquisition-domain structure. Manufacturer decodability is higher under the matched fold-local decoder than multiclass SiteCode decodability, while OASIS performance and ADNI-OASIS Wasserstein distances indicate non-trivial cross-dataset shift. These are descriptive transport and domain-shift analyses, not evidence of a causal acquisition artifact.

## Guardrails

- no VAE training: true
- no OASIS inference: true
- no TeX/manuscript edits: true
- no TDA/Mapper: true
- no tensor or metadata modification: true
