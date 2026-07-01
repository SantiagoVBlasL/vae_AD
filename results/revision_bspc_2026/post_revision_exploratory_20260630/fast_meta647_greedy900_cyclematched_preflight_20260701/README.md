# FAST Meta647 Greedy900 Cycle-Matched Preflight

This package prepares a diagnostic rerun of the June FAST meta647 channel-ablation experiment with 900 VAE epochs and 12 beta cycles while preserving the old local temporal structure: beta-cycle length 75 epochs and cosine-warm LR cycle length T0=30 epochs.

No training, inference, OASIS scoring, tensor edits, metadata edits, checkpoint edits, prediction edits, or manuscript edits were performed.

Proposed run root: `/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_20260701`

Use `guarded_launch.sh --confirm-training full` only after explicit approval.
