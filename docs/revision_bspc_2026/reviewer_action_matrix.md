# Reviewer Action Matrix — BSPC 2026 Revision
**Paper**: "Stable Network-Level Functional Connectivity Alterations in
Alzheimer's Disease Identified via Interpretable Latent Modelling"

*Last updated: 2026-03-30*
*Branch: revision_bspc_2026*

---

## How to use this table

- **Priority**: P1 = must do before resubmission · P2 = strongly recommended · P3 = optional/if time permits
- **New experiment?**: Yes = requires new model run or analysis · No = narrative / figure / table only
- **Status**: TODO · IN PROGRESS · DONE · DEFERRED

---

## Reviewer 1

| # | Comment (paraphrase) | Priority | New experiment? | Planned action | Target output | Manuscript section | Status |
|---|---------------------|----------|-----------------|---------------|---------------|-------------------|--------|
| R1.1 | *[Fill in comment]* | P1 | — | — | — | — | TODO |
| R1.2 | *[Fill in comment]* | P1 | — | — | — | — | TODO |

---

## Reviewer 2

| # | Comment (paraphrase) | Priority | New experiment? | Planned action | Target output | Manuscript section | Status |
|---|---------------------|----------|-----------------|---------------|---------------|-------------------|--------|
| R2.1 | *[Fill in comment]* | P1 | — | — | — | — | TODO |
| R2.2 | *[Fill in comment]* | P1 | — | — | — | — | TODO |

---

## Anticipated / Proactive Actions (from site audit findings)

These actions are proactively planned based on the site audit results,
regardless of specific reviewer comments.

| # | Issue identified | Priority | New experiment? | Planned action | Target script / output | Manuscript section | Status |
|---|-----------------|----------|-----------------|---------------|----------------------|-------------------|--------|
| PA.1 | **Site-leave-out (LOSO) evaluation required** — Reviewer will almost certainly ask: does the classifier generalise across acquisition sites? The current 5-fold CV does not guarantee site-independence. | P1 | Yes | Run LOSO evaluation on the 2-site main set (Sites 6, 130) and 5-site sensitivity set (Sites 6, 18, 19, 130, 305). Report per-site AUC and aggregate mean ± SD. | `scripts/revision_bspc_2026/run_loso_cv.py`; `results/revision_bspc_2026/loso_evaluation/` | Methods §2.4, Results §3.1, new Supplementary Table | TODO |
| PA.2 | **Severe class × manufacturer confound** (Cramér's V = 0.580, p = 4.2×10⁻¹⁴) — GE MEDICAL SYSTEMS and SIEMENS provide **exclusively AD** subjects (n=21 and n=27 respectively). CN subjects come entirely from Philips. This is a hard confound that limits the interpretability of the classifier and must be disclosed. | P1 | No (narrative + new table) | Add explicit confound disclosure paragraph in Methods. Report manufacturer distribution in Table 1 or Supplementary. Add a SIEMENS/GE-restricted sensitivity analysis (AD-only from those manufacturers vs Philips AD, to check feature leakage). | `results/revision_bspc_2026/site_audit/class_by_manufacturer.csv`; new table in paper | Methods §2.2, Limitations | TODO |
| PA.3 | **LOSO set is Philips-only** — All 5 LOSO-eligible sites use exclusively Philips scanners. LOSO therefore tests within-manufacturer site generalization only. Cross-manufacturer generalization cannot be evaluated with the current cohort. | P1 | No | State clearly in the LOSO discussion that cross-manufacturer generalization is not tested. Treat manufacturer as a potential source of bias that warrants future work. | Narrative addition to `site_audit_summary.md` and manuscript | Limitations | TODO |
| PA.4 | **Extremely limited LOSO power** — Only 2 sites meet the main threshold (≥5 per class). Site 130 dominates (33 subjects: 20 CN, 13 AD). Site 6 has 14 subjects (9 CN, 5 AD). This is too few LOSO folds for a robust aggregate estimate. | P1 | Yes | (a) Run 2-fold LOSO on main sites, report both folds individually. (b) Run 5-fold LOSO on sensitivity set. (c) Bootstrap confidence interval on aggregate AUC. (d) Consider pseudo-LOSO: cluster sites by ADNI phase × manufacturer and leave out clusters. | `scripts/revision_bspc_2026/run_loso_cv.py` | Methods §2.4, Results §3.1 | TODO |
| PA.5 | **3 subjects (1 AD, 2 MCI) dropped at tensor stage** — All SIEMENS. Reason investigated. 013_S_6768 and 003_S_4354 are POST_ASSEMBLY_ORPHAN (individual tensors computed after global assembly); 035_S_7021 is NEVER_PROCESSED (missing raw fMRI data). | P2 | No | Resolved. Document in Methods §2.2: all 3 SIEMENS subjects; 1 AD drop does not alter confound direction. | `scripts/revision_bspc_2026/inspect_dropped_subjects.py`; `results/revision_bspc_2026/dropped_subjects/` | Methods §2.2 (cohort), Supplementary | DONE |
| PA.6 | **CN exclusively from Philips — CN score calibration may reflect scanner, not biology** — The β-VAE learns a latent space where CN is entirely from Philips. Any "AD-likeness" score for subjects scanned on GE/SIEMENS may be driven by scanner-style differences rather than disease. | P1 | Yes | Scanner-leakage aggregated. Held-out test bacc: connectome_norm=0.537, latent_mu=0.534 (chance=0.333). Leakage present but moderate on held-out test. Flag as core limitation. | `scripts/revision_bspc_2026/aggregate_scanner_leakage.py`; `results/revision_bspc_2026/scanner_leakage/` | Methods §2.3, Limitations | DONE |
| PA.7 | **COVID transfer result (notebook 03_a)** — AD-likeness scores transferred to Long-COVID show no robust subject-level associations (all null after BH-FDR) but show group-level OMST enrichment (OR=3.37). This is a separate analysis; keep isolated in `notebooks/03_a_inference_covid_from_adcn.ipynb`. | P3 | No | Confirm this notebook is not included in BSPC submission unless specifically requested. Reference if needed as supplementary/preprint. | `notebooks/03_a_inference_covid_from_adcn.ipynb` | Not applicable to BSPC paper | DEFERRED |
| PA.8 | **Phase imbalance by class** — ADNI 3 has 279 subjects (mostly MCI), ADNI 2 has 144. Cross-phase generalization in LOSO should be verified. | P2 | No | Report phase × class breakdown in Table 1. Check LOSO fold assignments for phase balance. | `results/revision_bspc_2026/site_audit/phase_by_class.csv`; Table 1 addition | Methods §2.2 | TODO |
| PA.9 | **MCI subjects in tensor but excluded from classifier** — 248 MCI subjects are processed but not used. This is a valid design choice (binary CN vs AD), but reviewers may ask why. | P2 | No | Add one sentence in Methods justifying binary framing. Consider a sensitivity analysis including MCI or a 3-class extension as supplementary. | Methods §2.2 | Methods §2.2 | TODO |

---

## Priority Queue (ordered by urgency)

| Priority | Action ID | Blocking other work? |
|----------|-----------|----------------------|
| P1 | PA.2 — Confound disclosure | No |
| P1 | PA.6 — Scanner leakage aggregation | **DONE** |
| P1 | PA.1 + PA.4 — LOSO evaluation script | Yes (PA.3 depends on PA.1) |
| P1 | PA.3 — LOSO scope limitation disclosure | After PA.1 |
| P2 | PA.5 — Verify dropped subject | **DONE** |
| P2 | PA.8 — Phase table | No |
| P2 | PA.9 — MCI framing | No |
| P3 | PA.7 — COVID transfer note | No |

---

## Key Data Points for Manuscript

```
Analysis cohort: CN=89, AD=94 (N=183)
Metadata cohort: CN=89, AD=95, MCI=250 (N=434)
Tensor cohort:   CN=89, AD=94, MCI=248 (N=431; 3 subjects lost at feature-extraction)
Dropped:         013_S_6768 (AD), 003_S_4354 (MCI), 035_S_7021 (MCI)

Sites in analysis: 33 unique
Sites with CN+AD:  13

Manufacturer breakdown (analysis cohort):
  Philips:  89 CN + 46 AD = 135 total
  SIEMENS:   0 CN + 27 AD =  27 total  ← AD only
  GE:        0 CN + 21 AD =  21 total  ← AD only
Class×Manufacturer: Cramér's V = 0.580, p = 4.2e-14  ← SEVERE

LOSO-eligible sites (main, ≥5/class):    Site 6 (9 CN, 5 AD), Site 130 (20 CN, 13 AD)
LOSO-eligible sites (sensitivity, ≥3/class): + Site 18 (6 CN, 4 AD), Site 19 (4 CN, 8 AD), Site 305 (4 CN, 3 AD)
All LOSO-eligible sites: Philips only
```

---

## Notes on LOSO Design

Given the severe manufacturer confound (CN=Philips only, AD=all three),
the most defensible LOSO evaluation is:

1. **Philips-only LOSO** (sites 6, 18, 19, 130, 305): Clean within-manufacturer
   site test. Interpretable. Recommended as primary LOSO evaluation.

2. **Manufacturer-stratified analysis**: Compare model performance separately on
   Philips AD (n=46) vs GE/SIEMENS AD (n=48). If performance differs substantially,
   the model may be learning scanner features.

3. **Do NOT run cross-manufacturer LOSO** as a primary evaluation: leaving out
   a GE or SIEMENS site for testing is confounded by the fact that those sites
   have no CN subjects — the model cannot be trained fairly without CN from those
   manufacturers.

4. **Recommended disclosure sentence** (for Methods):
   > "Due to manufacturer imbalance in the ADNI cohort (CN subjects exclusively
   > from Philips; AD subjects from Philips, SIEMENS, and GE), cross-manufacturer
   > generalisation cannot be directly evaluated. LOSO evaluation was therefore
   > restricted to Philips sites with sufficient representation of both classes."
