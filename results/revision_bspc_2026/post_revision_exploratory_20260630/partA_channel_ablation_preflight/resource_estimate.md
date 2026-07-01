# Resource Estimate — FAST++ Channel Ablation

**Generated:** 2026-07-01T00:11:01
**GPU:** NVIDIA GeForce RTX 3060, 12 GB VRAM
**RAM:** 31 GB
**Disk:** Required output dir: /home/diego/proyectos/vae_AD/results/revision_bspc_2026/post_revision_exploratory_20260630/partA_channel_ablation_preflight/runs

## Per-run estimates

> **Note on GPU column:** `formula_gpu_MB` is a tensor+params floor estimate only.
> Real peak GPU (with Adam moments, gradients, batch-norm stats, and PyTorch overhead)
> for latent_dim=384 is empirically **3–6 GB** based on the promoted model running
> successfully on this GPU. All runs are safe on 12 GB.
>
> **Note on timing:** Wall-clock estimates are order-of-magnitude for latent_dim=384 with
> 3 folds at 800 epochs. The prior FAST 3-fold run (latent_dim=128, 960 epochs,
> offdiag_channelmean_sum) ran in ~2–3 h. Latent_dim=384 with 800 epochs is expected to
> take **2–4 h per run**.

| run_name | channels | formula_gpu_MB | est_wall_h | optional |
|---|---|---:|---:|---|
| ablation_ch1_pearsonFull_fast3fold_800ep | [1] | 99 | 1.5–2.5 | no |
| ablation_ch1_0_pearsonFull_OMST_fast3fold_800ep | [1, 0] | 179 | 2.0–3.0 | no |
| ablation_ch1_0_2_pearsonFull_OMST_MI_fast3fold_800ep | [1, 0, 2] | 260 | 2.0–3.5 | no |
| ablation_ch4_0_1_dfcStd_OMST_pearsonFull_fast3fold_800ep | [4, 0, 1] | 260 | 2.0–3.5 | no |
| ablation_ch3_0_1_dfcMean_OMST_pearsonFull_fast3fold_800ep | [3, 0, 1] | 260 | 2.0–3.5 | no |
| ablation_ch5_0_1_distCorr_OMST_pearsonFull_fast3fold_800ep | [5, 0, 1] | 260 | 2.0–3.5 | no |
| ablation_ch3_4_0_1_dfcBoth_OMST_pearsonFull_fast3fold_800ep | [3, 4, 0, 1] | 341 | 2.5–4.0 | YES |

## Summary

- **Required runs:** 6 runs × ~2–3.5 h each = **12–21 h total (sequential)**
- **Optional run (4-channel):** +2.5–4 h
- **Peak GPU (empirical):** 3–6 GB — safe on 12 GB RTX 3060 (confirmed by promoted model)
- **Disk per run:** ~500 MB–2 GB (fold artifacts + training histories at latent_dim=384)

## Notes

- Timing estimates are order-of-magnitude; early stopping (patience=200) may reduce wall time.
- Runs must be executed sequentially (one at a time) — no concurrent GPU jobs.
- 4-channel optional run: formula GPU 341 MB floor → empirical peak ~4–7 GB, still safe.
- If VAE val-split fails for any fold (small class count with 3 folds), the run aborts
  safely via `--vae_abort_if_val_split_fails`. In this case reduce to 2 folds manually.
- Run the **positive control first** (ch1_0_2). If it fails the gate, diagnose before
  running the experimental arms.
