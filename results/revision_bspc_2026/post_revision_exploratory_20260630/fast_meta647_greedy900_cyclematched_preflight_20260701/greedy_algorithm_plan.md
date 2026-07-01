# Greedy Algorithm Plan

The runner `scripts/ablation_canales.py` implements dynamic greedy forward channel selection.

1. Evaluate all single channels `[0]` through `[6]`.
2. Seed the greedy path with the best single channel by mean outer-fold AUC.
3. At each step, append each remaining channel to the current best set and evaluate each candidate.
4. Select the candidate with highest mean AUC.
5. For this diagnostic rerun, use `--no_early_stop` so the path continues until all seven channels are exhausted, even if a no-improvement stop rule would otherwise fire.

Maximum dynamic greedy trainings: 28.

The 900-epoch rerun preserves the original local temporal dynamics:

- FAST300: 300 epochs / 4 beta cycles = 75 epochs per beta cycle.
- Greedy900: 900 epochs / 12 beta cycles = 75 epochs per beta cycle.
- LR T0 remains 30 epochs.

Forced sentinel candidates are run independently from the greedy path because the 900-epoch dynamic path may differ from the old FAST300 path.
