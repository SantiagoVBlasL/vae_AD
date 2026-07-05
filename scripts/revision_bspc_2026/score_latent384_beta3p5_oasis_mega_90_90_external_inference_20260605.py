#!/usr/bin/env python
"""Read-only OASIS mega 90/90 inference for beta3.5 vs beta3.75 promoted reference.

This reuses the frozen-ADNI OASIS panel scorer. It performs inference only with
saved ADNI fold VAEs and ADNI-derived Stage B OOF-ECDF readouts.
"""

from __future__ import annotations

from pathlib import Path

import score_oasis_mega_90_90_external_inference_model_panel_20260604 as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

base.DEFAULT_OUTPUT = RESULTS / "latent384_beta3p5_oasis_mega_90_90_external_inference_20260605"
base.CANDIDATES = [
    base.CandidateSpec(
        label="promoted_beta3p75_oof_ecdf",
        role="primary_reference",
        run_dir=RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
    ),
    base.CandidateSpec(
        label="latent384_beta3p5_oof_ecdf",
        role="local_beta_sensitivity",
        run_dir=RESULTS / "recover035_latent384_beta3p5_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent384_beta3p5_stageB_oof_score_calibration",
    ),
]


if __name__ == "__main__":
    base.main()
