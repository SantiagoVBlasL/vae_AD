#!/usr/bin/env python
"""Read-only OASIS mega 90/90 inference for latent448 beta4.0 vs promoted reference."""

from __future__ import annotations

from pathlib import Path

import score_oasis_mega_90_90_external_inference_model_panel_20260604 as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

base.DEFAULT_OUTPUT = RESULTS / "latent448_beta4p0_oasis_mega_90_90_external_inference_20260605"
base.CANDIDATES = [
    base.CandidateSpec(
        label="promoted_beta3p75_oof_ecdf",
        role="primary_reference",
        run_dir=RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
    ),
    base.CandidateSpec(
        label="latent448_beta4p0_oof_ecdf",
        role="capacity_beta_sensitivity",
        run_dir=RESULTS / "recover035_latent448_beta4p0_T80_h10000_p560_full5x5",
        oof_dir=RESULTS / "recover035_latent448_beta4p0_stageB_oof_score_calibration",
    ),
]


if __name__ == "__main__":
    base.main()
