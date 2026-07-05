#!/usr/bin/env python3
"""Preflight/launcher for ch1only_latent384_beta4p25_T80_h10000_p560_full5x5.

Base reference: recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.
Allowed scientific diffs: beta_vae 3.75 -> 4.25.
"""

from __future__ import annotations

from pathlib import Path

from ch1only_targeted_followup_preflight_common_20260605 import CandidateSpec, main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "ch1only_latent384_beta4p25_T80_h10000_p560_full5x5"


if __name__ == "__main__":
    main(
        CandidateSpec(
            run_id=RUN_ID,
            config_path=PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta4p25_T80_h10000_p560_full5x5.json",
            launcher_path=PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_v5_1c_recover035_ch1only_latent384_beta4p25_T80_h10000_p560_full5x5.py",
            allowed_param_diffs={'beta_vae': (3.75, 4.25)},
        )
    )
