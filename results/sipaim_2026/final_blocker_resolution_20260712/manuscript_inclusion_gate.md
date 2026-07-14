# Manuscript inclusion gate

Read-only synthesis of Tasks 1-4. No manuscript was edited. This states what
is now safe to write into the compact SIPAIM manuscript, what still needs an
authorial decision, and what must not be claimed.

## Both blockers named in this task are resolved

1. **OASIS CDR-to-scan temporal alignment**: three mappings built and
   cross-validated (mapping B exactly reproduces the existing
   `oasis_label_sensitivity_20260710` numbers, confirming that result was
   already implicitly a nearest-within-365-days mapping). Tightening to
   +/-180 days raises the CDR-severity pairwise AUC from 0.709 to 0.730 on
   169/180 mappable scans; a backward-only mapping gives 0.725 on 176/180.
   All three mappings retain a significant Age/Sex-adjusted partial
   Spearman association (rho 0.24-0.28, p<0.002). **Safe to cite any of the
   three; recommend mapping B (365d nearest) as primary since it matches
   the already-published current-180/strict-128 numbers exactly, with
   mapping A (180d) as a sensitivity check.**
2. **Diagnostic-axis provenance/naming**: the axis is now explicitly named
   and computed as a **fold-local ADNI centroid-derived discriminant axis**,
   never pooled across encoders, centered on an OASIS-prevalence-weighted
   (50/50) ADNI centroid. **Safe to cite `diagnostic_centroid_geometry_revised.csv/.md`
   as the geometric-transportability evidence**, replacing any earlier
   loose "classifier decision axis" language.

## What is newly safe to write into the manuscript

- The CDR-to-scan temporal alignment table and its cross-check against the
  existing CDR label-sensitivity section (Task 1).
- The class-conditional (CN-CN, AD-AD) shift finding: Dataset-ComBat
  (either reference variant) corrects much of the marginal domain shift
  while preserving CN/AD discriminability (Cohen's d and projected AUC
  statistically indistinguishable from locked), whereas the ADNI-fitted
  Siemens-ComBat arm nearly eliminates the shift at a large discriminability
  cost (Cohen's d roughly halves, projected AUC drops 0.128, worse in all
  5 folds) -- a genuinely new bias/discriminability tradeoff finding
  (Task 2).
- The latent-component CDR association: the CDR-severity signal survives
  isolation to the latent-only contribution of the frozen logit
  (partial rho=0.259, p<0.001, Age/Sex-adjusted) -- strengthens the CDR
  label-sensitivity claim by showing it is not an Age/Sex artifact of the
  classifier (Task 3).

## What still needs an explicit missing-artifact caveat if cited

- **Official Stage-B N=45 Philips-CN-FP manufacturer-traversal count remains
  blocked** (OOF-ECDF transformer objects were never persisted). Only the
  N=12 raw-score/exploratory flip-rate result exists. Do not present the
  N=12 result as the Stage-B number.
- **Exact decomposition of the canonical OASIS score required an ADNI-only
  deterministic reconstruction** of the Stage-B classifier, because the
  fitted Stage-B pipeline/OOF-ECDF objects were never saved. The
  reconstruction is validated to floating-point precision against the
  canonical scores (max abs diff <5e-8, all 5 folds) and uses zero OASIS
  data and zero new hyperparameter choices, but if a reviewer asks whether
  a persisted, zero-fit artifact exists for this decomposition, the honest
  answer is no (`score_component_audit.md`).

## Duplication risks flagged for the manuscript writer (Task 4)

- **Channel-ablation sensitivity numbers are already BSPC supplementary
  content** (`Sup_ChannelSensitivity`). If the compact SIPAIM paper
  restates any ch1-only/[1,0,2] ADNI comparison, it must cite BSPC
  supplementary, not present the numbers as new.
- **Per-fold UMAP-by-manufacturer is already BSPC supplementary content**,
  topically adjacent to SIPAIM's new Figures 1-3 (manufacturer-stratified
  performance, residualized-PCA, covariance geometry). These are
  methodologically distinct and not literally duplicated, but the SIPAIM
  text should explicitly differentiate itself from the BSPC figure.
- The promoted model's ADNI-internal ROC-AUC/PR-AUC/BACC numbers, its
  interpretability signature, and its connectome figures are BSPC main-text
  content and must be cited as the frozen reference, never re-reported as
  new SIPAIM findings.

## Explicitly not extended

Per instruction, `results/sipaim_2026/final_evidence_package_20260712/claim_evidence_matrix.csv/md`
(the evidence package that audited `site_geometry_analysis_protocol_v4.tex`)
was **not** extended or merged with. All four outputs in this directory
(`results/sipaim_2026/final_blocker_resolution_20260712/`) are standalone
and supersede that package's diagnostic-axis geometry specifically
(`diagnostic_axis_geometry.csv` there vs. `diagnostic_centroid_geometry_revised.csv`
here) -- reconciling the two into one manuscript-facing table is an
authorial decision left unresolved here.

## No manuscript edits were made

As instructed. This file and its siblings are inputs for whoever next edits
`manuscript/sipaim_2026/site_geometry_analysis_protocol_v4.tex` or drafts
the true final compact 4-page manuscript (which, per Task 4's reconnaissance,
does not yet exist as a separate file in this repository and may currently
live only on Overleaf).
