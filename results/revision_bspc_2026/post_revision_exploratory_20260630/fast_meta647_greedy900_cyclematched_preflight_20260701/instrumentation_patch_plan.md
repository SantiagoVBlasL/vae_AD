# Instrumentation Patch Plan

Current `run_vae_clf_ad_ablation.py` can save VAE histories only when `--save_vae_training_history` is passed. The saved per-fold joblib currently includes:

- `train_loss`
- `train_recon`
- `train_kld`
- `val_loss`
- `val_recon`
- `val_kld`
- `val_loss_modelsel`
- `beta`

This satisfies total loss, reconstruction loss, KL, beta, and beta-max model-selection loss requirements. It does not store exact per-epoch learning rate or active units.

Minimal future patch, if exact LR/active-units traces are required:

1. Add `"lr": []` and `"active_units": []` to `history_data`.
2. At each epoch end, append `optimizer_vae.param_groups[0]["lr"]`.
3. Compute active units on the validation loader or a fixed train subset using the standard posterior-mu variance criterion, and append the count.
4. Include these fields in the saved joblib and any aggregate CSV export.

For this preflight, no script modification was made. The plotting script can reconstruct nominal LR schedule if exact LR is absent, but exact optimizer LR is not available without this patch.
