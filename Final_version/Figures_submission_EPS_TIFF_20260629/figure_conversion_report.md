# Figure Conversion Report — EPS/TIFF Submission Package

**Generated:** 2026-06-29 17:59  
**Model:** recover035_latent384_beta3p75_T80_h10000_p560_full5x5  
**Output:** `/home/diego/proyectos/vae_AD/Final_version/Figures_submission_EPS_TIFF_20260629`

---

## Source discovery

**NOTE:** `Final_version/Manuscript_Rev1.tex` and `Final_version/Supplemental.tex` were not found locally (figures are Overleaf-hosted). Figure list was reconstructed from finalized outputs identified in prior audit sessions.

Total source figures identified: **15**

## Conversion summary

| Metric | Count |
|---|---|
| Source figures | 15 |
| EPS files generated | 15 |
| TIFF files generated | 15 |
| Failed conversions | 0 |

## Conversion details

| Label | Status | EPS KB | TIFF KB | Resolution | Geometry |
|-------|--------|--------|---------|------------|----------|
| Fig-2-or-3_ClassificationPerformance | OK | 183 | 533 | 600x600 | 4261x3787+0+0 |
| Fig-3_InterpretabilitySignature_q1 | OK | 683 | 1062 | 600x600 | 8620x5842+0+0 |
| Fig-4_MNI_Connectome_q1 | OK | 559 | 1103 | 600x600 | 8520x2843+0+0 |
| fig4A_systems_signature_lowertriangle | OK | 436 | 598 | 600x600 | 3859x3282+0+0 |
| fig4B_consensus_edges | OK | 383 | 425 | 600x600 | 3364x2821+0+0 |
| fig4C_right_hemi_dominance | OK | 368 | 125 | 600x600 | 2527x2309+0+0 |
| fig4D_saliency_vs_effect | OK | 645 | 366 | 600x600 | 3376x2997+0+0 |
| fig4E_glass_brain_consensus | OK | 235 | 656 | 600x600 | 5065x3504+0+0 |
| fig4F_nodestrength_signed | OK | 583 | 785 | 600x600 | 4467x2424+0+0 |
| Sup_ChannelSensitivity | OK | 490 | 1020 | 600x600 | 7593x4269+0+0 |
| Sup_S6_A_shap_TRUE_UNFROZEN_fold3_LOGREG | OK | 1243 | 3041 | 600x600 | 4527x5633+0+0 |
| Sup_S6_B_shap_latents_top3_UNFROZEN_fold3 | OK | 264 | 594 | 600x600 | 4476x1553+0+0 |
| Sup_YeoChord_AAL3manual | OK | 836 | 962 | 600x600 | 5262x4380+0+0 |
| Sup_UMAP_per_fold_manufacturer | OK | 365 | 1307 | 600x600 | 10721x6256+0+0 |
| Sup_UMAP_per_fold_detailed | OK | 358 | 1175 | 600x600 | 10721x6256+0+0 |

## EPS vector vs raster

All EPS files were generated via `pdftops -eps` from PDF sources and are **vector-derived** (Poppler PDF→EPS conversion preserves vector paths from Matplotlib output).

## TIFF specification

- DPI: 600 (set via `pdftoppm -r 600` rendering + `convert -density 600` metadata)
- Compression: LZW
- Color mode: TrueColor (RGB)

## Title verification

### Figure 3C — must NOT contain '(Brier-...)'

**Verdict:** PASS — '(Brier-...)' not found in either candidate Figure 3C panel (Note: 'Brier=...' is present in classification Panel C — see text)

Notes:
- `figure2_ad_classification_performance.pdf` Panel C title: `C. Calibration (Brier=0.225, ECE=0.258)` — contains 'Brier=' but NOT '(Brier-...)'
- `Figure3_main_signature_q1.pdf` Panel C title: `C Hemispheric distribution of top-K edges` — no Brier at all

### Figure 4A — must say 'Consensus edge map', NOT 'Consensus Biomarker Map'

**Verdict:** MISMATCH — fig4B_consensus_edges.pdf contains '(B) Consensus Biomarker Map (edge-level)'. If this panel is submitted as manuscript Figure 4A, the title must be changed to 'Consensus edge map'. Source script to update: notebooks/signature_figures_new_recover035_20260625.ipynb, Cell FIG(B), line: ax_B.set_title("(B) Consensus Biomarker Map (edge-level)", ...)

Notes:
- `fig4B_consensus_edges.pdf` (notebook Figures_Nature_recover035_20260625) title: `(B) Consensus Biomarker Map (edge-level)` ← **MISMATCH if this is Fig 4A**
- `Figure4_mni_connectome_q1.pdf` does not use 'Biomarker Map' or 'edge map' labels
- `fig4A_systems_signature_lowertriangle.pdf` title: `(A) Systems-Level Consensus Signature` — no 'Biomarker Map' conflict

### Action required

If the submitted manuscript's Figure 4A corresponds to `fig4B_consensus_edges.pdf` from the notebook, the title must be updated from `(B) Consensus Biomarker Map (edge-level)` to `Consensus edge map` (or equivalent). To fix:

```
File: notebooks/signature_figures_new_recover035_20260625.ipynb
Cell: FIG (B) — Consensus Biomarker Map
Line: ax_B.set_title("(B) Consensus Biomarker Map (edge-level)", ...)
Change to: ax_B.set_title("(A) Consensus edge map", ...)
Then re-execute the cell and copy the output PDF.
```

---

## Guardrail compliance

- ✓ No `.tex` edits
- ✓ No original figures edited
- ✓ No model retraining
- ✓ No SHAP/IG/VAE recomputation
- ✓ All outputs written inside output directory only
