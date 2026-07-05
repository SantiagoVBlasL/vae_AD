#!/usr/bin/env python3
"""Read-only audit of ConvolutionalVAE block order."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.models import ConvolutionalVAE  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/vae_block_order_audit"


def class_names(seq: nn.Sequential) -> List[str]:
    return [module.__class__.__name__ for module in seq.children()]


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = ConvolutionalVAE(
        input_channels=3,
        latent_dim=256,
        image_size=131,
        final_activation="tanh",
        intermediate_fc_dim_config="quarter",
        dropout_rate=0.15,
        use_layernorm_fc=False,
        num_conv_layers_encoder=4,
        decoder_type="convtranspose",
        encoder_norm_mode="groupnorm",
        dropout_scope="legacy_all",
        block_order="legacy_act_norm",
    )

    rows: List[Dict[str, object]] = [
        {
            "component": "encoder_conv",
            "block_scope": "all four convolutional blocks",
            "current_order_locked": "Conv2d -> GELU -> GroupNorm -> Dropout2d",
            "module_sequence": " -> ".join(class_names(model.encoder_conv)),
            "notes": "Historical order; GroupNorm is after GELU.",
        },
        {
            "component": "encoder_fc_intermediate",
            "block_scope": "locked config use_layernorm_fc=False",
            "current_order_locked": "Linear -> GELU -> BatchNorm1d -> Dropout",
            "module_sequence": " -> ".join(class_names(model.encoder_fc_intermediate)),
            "notes": "If use_layernorm_fc=True in legacy mode, LayerNorm is inserted immediately after Linear, before GELU.",
        },
        {
            "component": "decoder_fc_intermediate",
            "block_scope": "locked config use_layernorm_fc=False",
            "current_order_locked": "Linear -> GELU -> BatchNorm1d -> Dropout",
            "module_sequence": " -> ".join(class_names(model.decoder_fc_intermediate)),
            "notes": "Same FC order as encoder intermediate block.",
        },
        {
            "component": "decoder_conv",
            "block_scope": "non-final convtranspose blocks",
            "current_order_locked": "ConvTranspose2d -> GELU -> GroupNorm -> Dropout2d",
            "module_sequence": " -> ".join(class_names(model.decoder_conv)),
            "notes": "Final decoder block is ConvTranspose2d -> Identity, followed by final Tanh output activation.",
        },
    ]
    write_csv(rows, OUTPUT_DIR / "block_order_table.csv")

    report = [
        "# VAE Block Order Audit",
        "",
        "This is a read-only audit of `src/betavae_xai/models/convolutional_vae.py` using the locked current VAE settings.",
        "",
        "## Exact Current Order",
        "",
        "- Encoder convolutional blocks: `Conv2d -> GELU -> GroupNorm -> Dropout2d`.",
        "- Encoder FC intermediate block: `Linear -> GELU -> BatchNorm1d -> Dropout` for the locked `use_layernorm_fc=False` setting.",
        "- Decoder FC intermediate block: `Linear -> GELU -> BatchNorm1d -> Dropout` for the locked `use_layernorm_fc=False` setting.",
        "- Decoder non-final convtranspose blocks: `ConvTranspose2d -> GELU -> GroupNorm -> Dropout2d`.",
        "- Decoder final block: `ConvTranspose2d -> Identity`, then final `Tanh` output activation.",
        "",
        "## Interpretation",
        "",
        "The locked model uses a historically valid `activation -> normalization` convolutional order. "
        "The proposed hygiene test keeps the locked behavior as default and only adds an explicit `norm_act` option for controlled experiments.",
        "",
        "No training was run. No tensors, metadata, ledger, or model outputs were modified.",
    ]
    (OUTPUT_DIR / "block_order_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source_file": str((PROJECT_ROOT / "src/betavae_xai/models/convolutional_vae.py").resolve()),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote block-order audit to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
