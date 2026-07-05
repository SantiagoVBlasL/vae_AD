#!/usr/bin/env python3
"""Guarded launcher for recover035_latent384_beta9p5_T80_h10000_p560_full5x5.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026.prepare_beta9p5_T160_controlled_preflight_20260608 import main


if __name__ == "__main__":
    main(default_variant="beta9p5")
