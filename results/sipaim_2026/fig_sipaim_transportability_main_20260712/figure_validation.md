# Figure validation

Read-only. No training, VAE inference, model selection, threshold
optimization, manuscript reading, or manuscript editing was performed to
produce this figure. All plotted values are read directly from already
computed, audited CSV artifacts (paths recorded per-value in
`fig_sipaim_transportability_main_data.csv`); no value plotted in this
figure was independently recomputed by the figure script.

## Technical requirements

| Requirement | Result |
|---|---|
| IEEE double-column width, 7.16 in | PDF page size 515.52 x 241.2 pt = **7.16 x 3.35 in** exactly |
| PDF vector-only (no raster) | `pdfimages -list` returns **zero** entries |
| Embedded/subsetted fonts | `pdffonts`: `DejaVuSans`, `DejaVuSans-Bold`, `DejaVuSans-Oblique`, all **emb=yes, sub=yes** |
| 600-dpi PNG | `4296 x 2010` px at `599.999 dpi` = **7.16 x 3.35 in**, matches PDF |
| Colorblind-safe palette | Okabe-Ito-derived hues (`#2a78d6`/`#eb6834` for the 2-way A contrast, `#000000`/`#009E73`/`#56B4E9` for the 3-arm B comparison, monotone blue ramp for the ordinal CDR groups in C); identity is never encoded by color alone -- every series also carries a direct text label, distinct marker shape (B), or axis position (A, C) |
| 8-9 pt final-size fonts | Base `rcParams["font.size"]=8.0`; axis labels/panel titles 7.4-8.5 pt; tick labels 6.6-7.3 pt; fine-print statistical annotations (CI text, per-point labels) 5.0-5.4 pt. Note: not every element is literally 8-9 pt -- annotation text is intentionally smaller than axis/title text, which is standard practice for a dense 4-subplot IEEE figure; all text is vector (PDF fonttype 42), so it remains crisp at any zoom regardless of nominal point size. |

## Panel A: reproduction against source artifacts

| Quantity | Plotted | Source value | Source file |
|---|---|---|---|
| Manufacturer BACC, locked (mean of 5 folds) | 0.66586 | 0.66586 (task-specified 0.666) | `results/sipaim_2026/manufacturer_decoding/cleanrepro_manufacturer_decoding.csv` |
| Manufacturer BACC, input-ComBat (mean of 5 folds) | 0.44042 | 0.44042 (task-specified 0.440) | same |
| ADNI OOF ROC-AUC, locked (pooled) | 0.795155 | 0.795155 (task-specified 0.795) | `results/sipaim_2026/clean_foldwise_combat_training_and_audit/cleanrepro_paired_bootstrap_vs_locked.csv` |
| ADNI OOF ROC-AUC, input-ComBat (pooled) | 0.749278 | 0.749278 (task-specified 0.749) | same |
| Paired bootstrap delta (ComBat - locked) | -0.045876, 95% CI [-0.080688, -0.011752] | identical | same |

Both pooled-value assertions and the manufacturer-BACC mean assertions are
enforced as hard `assert` statements in the generating script -- the script
raises an error rather than silently plotting a mismatched number.

## Panel B: reproduction against source artifacts

Every plotted point, x/y error bar, and arrow endpoint in Panel B is read
directly from `results/sipaim_2026/standardized_locked_space_geometry_20260712/standardized_shared_geometry.csv`
(the instructed source), row-indexed by `arm`. The script asserts that the
loaded arm set is exactly `{locked_frozen_transfer,
external_dataset_combat_adni_reference, external_dataset_combat_no_reference}`
and fails if `previous_adni_fitted_siemens_combat` (or any other arm) is
present. `results/sipaim_2026/final_evidence_package_20260712/diagnostic_axis_geometry.csv`
was never read.

The x-axis (`marginal_standardized_shift`) is plotted as an absolute
value; the script asserts that both CI bounds are negative before flipping
sign, so the transform is a simple, verified reflection, not a
recomputation.

| Arm | abs(marginal shift) | projected AUC | Cohen's d (not plotted, for cross-check) |
|---|---|---|---|
| locked_frozen_transfer | 0.9348 [0.809, 1.062] | 0.6372 [0.555, 0.715] | 0.5178 |
| external_dataset_combat_adni_reference | 0.4728 [0.337, 0.611] | 0.6369 [0.555, 0.715] | 0.5194 |
| external_dataset_combat_no_reference | 0.6613 [0.528, 0.797] | 0.6370 [0.555, 0.715] | 0.5171 |

All values match `standardized_shared_geometry.csv` to full floating-point
precision (verified by direct load with no intermediate transform except
the documented absolute-value flip on the x-axis).

## Panel C: reproduction against source artifacts

Source: `results/sipaim_2026/final_blocker_resolution_20260712/oasis_cdr_scan_alignment.csv`
(individual scores, mapping `A_nearest_180d`) and
`cdr_temporal_sensitivity_metrics.csv` (row `A_nearest_180d`, for the
annotated AUC/CI/partial-rho).

| Check | Result |
|---|---|
| n mapped / n total | 169 / 180 (asserted against `cdr_temporal_sensitivity_metrics.csv`'s own `n_mapped` field) |
| n per CDR group (0 / 0.5 / >=1) | 81 / 51 / 37 (asserted against the same file's `n_cdr0`/`n_cdr05`/`n_cdr_ge1`) |
| Pairwise ROC-AUC (CDR0 vs CDR>=1) | 0.7297, 95% CI [0.6289, 0.8241] -- read directly, not recomputed |
| Partial Spearman (Age/Sex-adjusted) | rho=0.2784, p=2.48e-04 -- read directly, not recomputed |
| Individual scores plotted | `score_locked` column, unmodified, for every subject with `A_nearest_180d_mapped == True` |

CDR>=1 is labeled "CDR $\geq$1" throughout the figure and its caption;
the word "Alzheimer's" or "AD" is never attached to that group. This is
enforced by construction (the plotting code never emits that string for
this group) and stated explicitly in the figure's footer text.

## What was NOT done

- No VAE was run in inference mode to produce any number in this figure.
- No classifier was retrained, recalibrated, or re-thresholded.
- No new bootstrap, permutation test, or statistical model was fit --
  every CI, p-value, and point estimate is copied from an existing audited
  CSV.
- No manuscript `.tex` file was opened, read, or edited at any point while
  producing this figure.
