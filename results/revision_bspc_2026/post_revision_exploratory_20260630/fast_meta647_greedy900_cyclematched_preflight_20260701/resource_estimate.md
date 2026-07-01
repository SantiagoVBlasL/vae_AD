# Resource Estimate

- Full dynamic greedy path candidate trainings: 28.
- Forced sentinel trainings in independent output directories: 3.
- Maximum planned trainings if full path plus sentinels are launched: 31.
- Outer folds per candidate: 3.
- Maximum VAE fold trainings: 93.
- Epoch cap per VAE fold: 900.
- FAST300 observed candidate folders: 22 before early stop after step 3.

Approximate wall-time and disk impact are uncertain because early stopping is data- and GPU-dependent. A conservative planning envelope is roughly 3x the per-candidate epoch cap of FAST300, multiplied by 31/22 if all diagnostics are run. With `--save_vae_training_history`, extra joblib/PNG history files are small relative to checkpoints and predictions.

This is diagnostic only. It should be scheduled only when no FULL/OASIS/manuscript-critical jobs are waiting for the GPU.
