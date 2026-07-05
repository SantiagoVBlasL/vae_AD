#!/usr/bin/env python3
"""Guarded launcher for recover035_latent384_beta3p75_T160_h10000_p1120_full5x5.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026.prepare_T160_p1120_controlled_preflight_20260609 import main


if __name__ == "__main__":
    main(default_variant="T160_p1120")
