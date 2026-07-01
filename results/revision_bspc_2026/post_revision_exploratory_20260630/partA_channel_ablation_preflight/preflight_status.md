# Preflight Status — FAST++ Channel Ablation

**Generated:** 2026-07-01T00:11:01
**Status:** READY

---

## File checks

| Item | Path | Status |
|---|---|---|
| Global tensor NPZ | /media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz | ✓ FOUND |
| Metadata CSV | /home/diego/proyectos/vae_AD/results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv | ✓ FOUND |
| Training script | /home/diego/proyectos/vae_AD/scripts/run_vae_clf_ad_inference.py | ✓ FOUND |
| Python executable | /home/diego/anaconda3/envs/vae_ad/bin/python | ✓ FOUND |
| Output base dir | /home/diego/proyectos/vae_AD/results/revision_bspc_2026/post_revision_exploratory_20260630/partA_channel_ablation_preflight/runs | will be created at launch |

---

## Configuration check

| Parameter | Value | Rationale |
|---|---|---|
| epochs_vae | 800 | FAST screen; 10 full cosine cycles |
| lr_scheduler_T0 | 80 | Matches promoted model cycle length |
| cyclical_beta_n_cycles | 10 | 10 × 80 = 800 epochs exactly |
| early_stopping_patience | 200 | 25% of 800; ~2.5 cycles |
| beta_vae | 3.75 | Matches promoted model |
| latent_dim | 384 | Matches promoted model |
| recon_loss_mode | offdiag_channelmean_sum | MANDATORY for cross-channel fairness |
| outer_folds | 3 | FAST: 3 folds (vs 5 for FULL) |
| inner_folds | 3 | 3-fold inner CV |
| classifier | logreg L2 only | Fastest screening readout |
| n_iter_logreg | 100 | Coarse but sufficient HP search |
| classifier_calibrate | False | Skip for screening speed |

---

## Loss mode warning

The promoted model was trained with `recon_loss_mode=mse_sum_batchmean_current`.
This ablation uses `offdiag_channelmean_sum`, which normalises reconstruction
loss by the number of off-diagonal elements AND the number of channels.
**This makes AUC comparable across different channel counts but means the
ablation AUC cannot be directly compared to the promoted model's Stage A
AUC.** Compare only within this ablation series.

---

## Planned runs (7 total; 1 optional)

- `ablation_ch1_pearsonFull_fast3fold_800ep` — channels [1] — 0.1 h 
- `ablation_ch1_0_pearsonFull_OMST_fast3fold_800ep` — channels [1, 0] — 0.1 h 
- `ablation_ch1_0_2_pearsonFull_OMST_MI_fast3fold_800ep` — channels [1, 0, 2] — 0.2 h 
- `ablation_ch4_0_1_dfcStd_OMST_pearsonFull_fast3fold_800ep` — channels [4, 0, 1] — 0.2 h 
- `ablation_ch3_0_1_dfcMean_OMST_pearsonFull_fast3fold_800ep` — channels [3, 0, 1] — 0.2 h 
- `ablation_ch5_0_1_distCorr_OMST_pearsonFull_fast3fold_800ep` — channels [5, 0, 1] — 0.2 h 
- `ablation_ch3_4_0_1_dfcBoth_OMST_pearsonFull_fast3fold_800ep` — channels [3, 4, 0, 1] — 0.2 h (OPTIONAL)

---

## Gate criteria for promotion to FULL 5×5

| Criterion | Threshold |
|---|---|
| Primary: OOF ROC-AUC | ≥ 0.760 (promoted baseline 0.795 − 0.035) |
| Secondary: OOF PR-AUC | ≥ 0.500 |
| Stability: fold AUC std | ≤ 0.070 |
| Manufacturer leakage BA | ≤ 0.450 (chance ≈ 0.333) |
| Active latent units | ≥ 50 across all folds |
| Positive control (ch1_0_2) | AUC within 0.03 of promoted baseline |

**Note:** Any candidate that passes gate criteria advances to FULL 5×5 training
with `recon_loss_mode=offdiag_channelmean_sum`, `outer_folds=5`, `inner_folds=5`,
`epochs_vae=10000`, `cyclical_beta_n_cycles=125`, and full Optuna HP search
(`n_iter_logreg=300+`).

---

## Pre-launch checklist

- [ ] GPU memory confirmed available (target: ≤ 9 GB / 12 GB)
- [ ] Disk space confirmed (> 10 GB free for all 6 required runs)
- [ ] Positive control (ch1_0_2) run FIRST — abort if control fails gate
- [ ] Stage B classifier-only readout script ready after each training run
- [ ] No concurrent GPU jobs running
