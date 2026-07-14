# Figure caption (draft text for manuscript author's use)

**Figure. Transportability of the locked promoted `[1,0,2]` model: internal
ComBat sensitivity, shared-coordinate external geometry, and OASIS
CDR-severity association.**

**(A)** ADNI-internal sensitivity of the locked model to input-level
ComBat harmonization, on two deliberately separate y-axes. **A1:**
manufacturer identity decodability (balanced accuracy) from the frozen
latent representation, locked (0.666 ± SD across 5 folds) vs. input-ComBat
(0.440 ± SD); dashed line = 3-class chance (0.333). **A2:** ADNI
out-of-fold (OOF) ROC-AUC, locked (0.795) vs. input-ComBat (0.749); error
bars are ±1 SD across the 5 outer folds; annotated Δ and 95% CI are from a
paired, diagnosis-stratified subject bootstrap (10,000 resamples) on the
pooled OOF predictions. Points are individual fold values (jittered).

**(B)** External geometric transportability restricted to the three arms
that share the locked model's frozen VAE latent coordinate system: locked
frozen transfer (black square), Dataset-ComBat with ADNI as reference
(green circle), and Dataset-ComBat with no reference (light-blue diamond,
secondary/transductive). The independently retrained Siemens-ComBat VAE is
excluded (see note below). Axes: absolute standardized marginal shift of
the OASIS cohort relative to ADNI, projected onto each fold's unit-norm
AD-minus-CN discriminant axis (x); fold-local projected ROC-AUC of the same
projection against OASIS diagnosis (y). Arrows point from locked to each
adapted arm. Error bars are subject-level bootstrap 95% CIs (10,000
resamples, diagnosis-stratified). Dataset-ComBat (either variant) reduces
the marginal shift relative to locked without materially changing the
projected AUC.

**(C)** Frozen OASIS locked-model scores by CDR severity group, under the
primary nearest-CDR-within-180-days temporal mapping (169/180 scans
mapped; 11 scans excluded for having no CDR visit within the window).
Boxplots show the score distribution per group with individual subject
scores overlaid (jittered). Annotated: ROC-AUC of CDR 0 vs. CDR ≥1 with a
subject-level bootstrap 95% CI, and the Age/Sex-adjusted partial Spearman
association between continuous score and ordinal CDR.

**Notes.** CDR is used throughout as a clinical severity indicator, not as
etiological confirmation of Alzheimer's disease; "CDR ≥1" subjects are not
labeled AD. The Siemens-ComBat VAE in Panel B was trained independently of
the locked model and does not share its latent coordinate system, so it is
excluded from this shared-coordinate comparison (see
`geometry_source_reconciliation.md` in
`results/sipaim_2026/canonical_shared_locked_geometry_20260712/` and
`geometry_robustness_verdict.md` in
`results/sipaim_2026/standardized_locked_space_geometry_20260712/` for the
full rationale). All panels use already-computed, audited artifacts; no
model was trained, no VAE inference was run, and no threshold was
optimized to produce this figure.
