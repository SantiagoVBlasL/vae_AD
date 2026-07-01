# Checkpoint Selection Policy

Historical behavior remains the default: the child ablation runner selects the checkpoint with the minimum `ValL(beta_max)` across all epochs.

For this robust Greedy900 rerun only, launch scripts pass:

`--vae_checkpoint_select_high_beta_only`

With this flag enabled:

1. Every epoch still records `ValL(beta_max)`.
2. The script records the best epoch at any beta:
   - `best_any_beta_epoch`
   - `best_any_beta_val_loss_modelsel`
   - `beta_at_best_any_beta`
3. The selected checkpoint is the minimum `ValL(beta_max)` among epochs where:
   `beta_epoch >= 0.95 * beta_vae`.
4. For beta=2.5, the high-beta threshold is 2.375.
5. If no eligible high-beta epoch exists, the fold fails explicitly.
6. The existing beta guardrail remains active; it is not removed.

Each fold writes:

- `vae_checkpoint_selection_summary_fold_<k>.json`
- `vae_checkpoint_selection_summary_fold_<k>.csv`

These record:

- `best_any_beta_epoch`
- `best_any_beta_val_loss_modelsel`
- `beta_at_best_any_beta`
- `best_high_beta_epoch`
- `best_high_beta_val_loss_modelsel`
- `beta_at_best_high_beta`
- `selected_epoch`
- `selected_epoch_beta`
- `selected_epoch_cycle_id`
- `selected_epoch_phase`
- `high_beta_threshold`
