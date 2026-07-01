# Patch Summary

Modified scripts:

1. `scripts/run_vae_clf_ad_ablation.py`
   - Added CLI flag `--vae_checkpoint_select_high_beta_only`, default OFF.
   - When OFF, historical checkpoint selection is preserved.
   - When ON, best checkpoint selection is restricted to epochs with `beta >= 0.95 * beta_vae`.
   - Records best-any-beta and best-high-beta checkpoint diagnostics per fold.
   - Fails explicitly if no high-beta checkpoint is eligible.

2. `scripts/ablation_canales.py`
   - Added CLI flag `--vae_checkpoint_select_high_beta_only`, default OFF.
   - Parent runner forwards the flag to the child ablation script only when requested.

Backward compatibility:

- Existing runs without the new flag keep old behavior.
- The low-beta checkpoint guardrail remains active.
