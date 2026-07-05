#!/usr/bin/env python3
"""Launcher/preflight wrapper for recover035_ch12_latent384_beta2p75_chmeanloss_T80_h10000_p560_full5x5.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training and passes through the guarded common preflight path.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026.prepare_ch12_beta2p75_currentloss_chmeanloss_preflight_20260607 import main


if __name__ == "__main__":
    main(default_variant="chmeanloss")
