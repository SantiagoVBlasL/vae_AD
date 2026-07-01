# Resource Estimate

- Full dynamic greedy path candidate trainings: 28.
- Forced sentinel trainings: 3.
- Maximum planned candidate trainings: 31.
- Outer folds per candidate: 3.
- Maximum VAE fold trainings: 93.
- Epoch cap per VAE fold: 900.
- Early-stopping patience: 150 epochs = 2 beta cycles.

The previous Greedy900 run completed only 12/28 dynamic candidates under the strict post-hoc beta guardrail. This rerun is expected to use more epochs per fold because patience increased from 30 to 150 and checkpoint selection is restricted to high-beta epochs.
