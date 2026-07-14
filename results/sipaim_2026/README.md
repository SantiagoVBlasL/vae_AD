# results/sipaim_2026/

Curated, lightweight (CSV/MD/JSON only, ≈1.4 MB total) results bundle for
the SIPAIM 2026 acquisition-domain and transportability study. Every file
here is a copy of a canonical artifact that already existed under
`results/revision_bspc_2026/post_revision_exploratory_20260630/` (or an
adjacent top-level `results/revision_bspc_2026/` directory) — nothing here
was computed specifically for this freeze. See `experiment_registry.csv`
(repo root) for the authoritative one-row-per-analysis index, and
`docs/sipaim_2026/ANALYSIS_DECISIONS.md` for why each file was chosen over
alternative/superseded versions.

## Subdirectories

| Directory | Contents |
|---|---|
| `manufacturer_decoding/` | Can Manufacturer (GE/Philips/SIEMENS) be decoded from VAE latents, locked vs. clean fold-wise ComBat, with permutation nulls. |
| `site_decoding_permutation/` | Finer-grained SiteCode decodability (locked model only) with a 1000-permutation null; site/manufacturer/diagnosis/fold composition tables. |
| `manufacturer_transport_and_supported_sites/` | Per-manufacturer and per-site diagnostic transport (CN false-positive rate, sensitivity, specificity, AUC), and the explicit N-based gates defining which sites are "supported" for site-level AUC vs. LOSO. |
| `classifier_only_loso/` | Frozen-readout leave-one-site-out (VAE never retrained per site) for the 2 sites with enough subjects to support it. |
| `frozen_oasis_inference/` | Fully frozen (zero OASIS-side fitting) external validation on OASIS, locked vs. clean ComBat. |
| `adni_oasis_wasserstein/` | Fold-local Wasserstein distance between ADNI and OASIS latent distributions (and between manufacturers), with a permutation test. |
| `clean_foldwise_combat_training_and_audit/` | The full audit trail of the canonical ComBat VAE retrain: lineage/leakage checks, config diff vs. locked, training reproducibility, protocol checks, and the manuscript-ready results table. Also contains the paired-bootstrap CSV for category 9 (see below). |
| `oasis_external_batch_combat/` | External *Dataset*-level ComBat (batch=ADNI/OASIS, unsupervised) applied to raw connectivity features ahead of the same frozen locked VAE — a different harmonization axis from the fold-wise Manufacturer-ComBat retrain above. See `docs/sipaim_2026/METHODS_PROVENANCE.md` for the distinction. |

## Paired locked-vs-ComBat bootstrap

Not a separate directory — it lives at
`clean_foldwise_combat_training_and_audit/cleanrepro_paired_bootstrap_vs_locked.csv`
(ADNI-internal) and `frozen_oasis_inference/cleanrepro_oasis_paired_bootstrap.csv`
(OASIS-external), since both are direct outputs of the same final-audit
script (`audit_foldcombat_cleanrepro_final_20260706.py`) as the directories
they sit in.

## What is deliberately NOT here

- No VAE checkpoints, training histories, or full per-subject latent-vector
  caches (all git-ignored and excluded from this snapshot — see
  `docs/sipaim_2026/DATA_MANIFEST.md`).
- No raw ADNI/OASIS tensors or licensed metadata.
- No topology/persistent-homology results — none exist to copy (see
  `topology_existing_artifact_audit.md` at the repo root).
- No superseded/defective intermediate analyses (the 2026-06-07 run's
  buggy export, the 2026-07-05 audit built on it, or the partially-blocked
  2026-07-06 Task-B OASIS estimate) — see
  `docs/sipaim_2026/ANALYSIS_DECISIONS.md` for exactly what was superseded
  and why.

## How to regenerate any of this

See `docs/sipaim_2026/REPRODUCIBILITY.md`.
