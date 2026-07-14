# SIPAIM 2026 transportability study — analysis decisions

This project accumulated multiple iterations of several analyses (an
initial pass, a bug, a correction, and in one case a full from-scratch
reproduction). This document is the single place that states **which
version is canonical** for the SIPAIM manuscript and **why**, so nobody
accidentally cites a superseded number. Cross-reference
`experiment_registry.csv` (machine-readable) and `METHODS_PROVENANCE.md`
(method detail).

## 1. Manufacturer decoding: use `cleanrepro_manufacturer_decoding.csv`, not the 2026-07-05 audit

- **Canonical**: `results/sipaim_2026/manufacturer_decoding/cleanrepro_manufacturer_decoding.csv`
  (locked BACC 0.6659±0.0506, clean ComBat BACC 0.4404±0.0533).
- **Superseded, not copied into this snapshot**:
  `foldcombat_mfr_age_sex_audit_20260705/manufacturer_leakage_from_latent_mu.csv`
  and its duplicate in `combat_full_matched_preflight_20260706/` — both
  computed from the **defective** (raw-input-inference) June latent cache,
  reaching an invalid `do_not_promote` verdict. Do not cite the 0.778/0.550
  ADNI ROC-AUC/PR-AUC figures from that audit; the valid figures are
  0.7493/0.5327 (clean ComBat) vs. 0.7952/0.5739 (locked).
- **Which manufacturer-BA number to quote when comparing against SiteCode
  decodability**: `manufacturer_decoding_reconciliation.csv`
  / `final_sipaim_claims.md` settle a second discrepancy — use the matched
  fold-local 5-NN estimate (BACC 0.6659±0.0506), **not** the older
  scanner-leakage-QC logistic-regression estimate (0.7276±0.0904), which
  used a different (within-test CV) protocol and should not be averaged
  with the transport estimate.

## 2. Site decoding: locked-model only; no ComBat-arm site decoding exists

`site_decoding_permutation/foldlocal_site_decoding.csv` and
`site_permutation_1000.csv` are **locked-model only**. No one has run
SiteCode decoding on the ComBat-harmonized latents. If the manuscript wants
to claim "ComBat also reduces site decodability," that claim is **not
supported by any artifact in this snapshot** and must not be made without
a new analysis. The safe, supported claim is the one already in
`final_sipaim_claims.md`: manufacturer decodability (matched protocol) is
higher than SiteCode decodability under the locked model.

## 3. Manufacturer transport (CN FPR): use the cleanrepro number for Philips

- **Canonical**: `results/sipaim_2026/manufacturer_transport_and_supported_sites/cleanrepro_manufacturer_transport.csv`
  (via `foldcombat_cleanrepro_final_audit_20260706`) — Philips CN FPR
  0.4545 (locked) -> 0.3232 (clean ComBat).
- Several earlier directories (`foldcombat_mfr_age_sex_audit_20260705/philips_cn_fpr_comparison.csv`,
  `combat_full_matched_preflight_20260706/philips_cn_fpr.csv`,
  `combat_downstream_classifier_reconstruction_20260706/philips_cn_fpr_comparison.csv`)
  contain the **same or very similar** locked-arm number (0.4545) but a
  ComBat-arm number computed from a defective or reconstructed (not
  cleanrepro-fresh) cache. None of these were copied into this snapshot, to
  avoid a reader finding three slightly different "the" ComBat Philips FPR
  numbers. If historical comparison is needed, they remain at their
  original paths under `results/revision_bspc_2026/post_revision_exploratory_20260630/`
  (not part of this git snapshot).
- **Not uniformly beneficial**: GE and SIEMENS CN FPR both get *worse*
  under ComBat (GE 0.1485->0.3663; SIEMENS 0.24->0.33). The manuscript
  should not present the Philips improvement as if it generalizes to all
  manufacturers — report all three.

## 4. Classifier-only LOSO: the locked-vs-ComBat comparison is not from the cleanrepro run

- **Canonical (locked only)**: `results/sipaim_2026/classifier_only_loso/leave_one_site_out_metrics.csv`
  (from `sipaim_multisite_transport_20260706`, built directly on the locked
  model's frozen latents).
- **Best-available (locked vs. ComBat)**: `results/sipaim_2026/classifier_only_loso/corrected_loso_metrics.csv`
  (from `combat_downstream_classifier_reconstruction_20260706`). This uses
  the **corrected, reconstructed** harmonized latent cache from 2026-07-06,
  **not** the cleanrepro fresh-run cache — the cleanrepro final audit
  (`audit_foldcombat_cleanrepro_final_20260706.py`) never recomputed LOSO.
  The reconstructed cache was itself validated (all lineage/leakage checks
  pass, subject hashes match the locked run), so this is a legitimate
  number, but it is one processing step removed from the single-shot
  cleanrepro pipeline that everything else in this snapshot traces to. If
  a reviewer asks for LOSO-under-cleanrepro specifically, the honest answer
  is that it was not rerun; re-running
  `reconstruct_combat_downstream_classifier_20260706.py`-equivalent logic
  against the cleanrepro run's own frozen latent cache would be the
  correct follow-up (not done here, per the no-new-experiments guardrail
  for this freeze task).

## 5. Frozen OASIS inference: the cleanrepro numbers supersede the Task-B "partial" estimate

`combat_external_transport_completion_20260706/00_FINAL_REPORT.md`
explicitly self-reports `status: PARTIAL_COMPLETE_STAGEB_LINEAGE_BLOCKER`
— its Task B OASIS scoring used a Stage-B readout fit from the (at the
time) uncorrected raw-input ADNI cache, and its own report says this is
"not a valid end-to-end ComBat estimate." That directory is **not** part
of this snapshot. The canonical, valid, end-to-end frozen OASIS numbers are
in `foldcombat_cleanrepro_final_audit_20260706` (`cleanrepro_oasis_metrics.csv`,
`cleanrepro_oasis_predictions.csv`, `cleanrepro_oasis_paired_bootstrap.csv`),
copied to `results/sipaim_2026/frozen_oasis_inference/`.

## 6. ADNI-OASIS Wasserstein: three related files, not duplicates — here's how they differ

- `adni_oasis_wasserstein/wasserstein_*` (from `site_geometry_wasserstein_20260706`)
  — the dedicated, most rigorous fold-local analysis (locked model; also
  includes manufacturer-pair Wasserstein and a 1000-permutation test).
  Cite this for the permutation p-value and the Gaussian Bures-W2 number.
- `adni_oasis_wasserstein/cleanrepro_wasserstein_*` (from
  `foldcombat_cleanrepro_final_audit_20260706`) — the ComBat-arm version,
  needed to compare locked vs. ComBat cross-dataset shift.
- `adni_oasis_wasserstein/manuscript_table_oasis_wasserstein_by_fold.*`
  (from `sipaim_multisite_transport_20260706`) — a manuscript-facing
  repackaging of the same locked-arm analysis, paired with
  `oasis_performance_vs_domain_shift.csv`, which is what actually appears
  as a figure in the manuscript's `figures/sipaim_2026/`. Use this one when
  citing the table that matches `external_dataset_wasserstein.pdf` /
  `external_performance_vs_domain_shift.pdf`.

## 7. Topology / persistent homology: no claim can be made

See `topology_existing_artifact_audit.md` in full. `site_geometry_topology_20260705`
contains only a Step-0 dependency-check report; TDA libraries
(`ripser`/`gudhi`/`giotto-tda`/`kmapper`) were and remain unavailable in
every checked Python environment. This was **not rerun** for this freeze
task, per instructions. Do not copy that directory into the git snapshot —
it has no scientific content to preserve. Any manuscript reference to
latent-space topology must say the analysis was attempted and blocked, not
that it was run and returned a negative/null result.

## 8. Manuscript figure inclusion is unchanged by this freeze

`manuscript/sipaim_2026/site_geometry_analysis_protocol_v4.tex` was **not
edited** as part of this freeze (no manuscript edits were requested, and
the .tex is "actively reviewed by Martín Belzunce via Overleaf"). Its
header comments (as of 2026-07-05) list which figures it has verified are
present in `Figures/site_geometry/` and does not yet include
`fig_transportability_summary.pdf` (generated 2026-07-06/07, after that
comment block was last updated) — that figure is preserved alongside the
manuscript per the task's instructions, but whether/where to `\sitefig{}`
it into the document body is an authorial decision left to whoever edits
the manuscript next, not made by this freeze.

## 9. Scope boundary: the FAST-scale ComBat pilot is out of scope for this snapshot

A separate FAST-scale (beta=2.50, latent=128) ComBat pilot and its
supporting `scripts/run_vae_clf_ad_ablation.py` wiring
(`results/revision_bspc_2026/post_revision_exploratory_20260630/combat_input_harmonization_20260706/`)
exist on this branch but are **not** one of the 9 categories this freeze
was scoped to, and are not copied into `results/sipaim_2026/`. They remain
available at their original path if a future manuscript revision wants to
report the FAST-vs-FULL scale-dependence finding (FULL scale shows a
significant diagnosis-AUC cost from ComBat; the FAST pilot did not).
