#!/usr/bin/env python3
"""Synthetic smoke test for ConvolutionalVAE block-order modes."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.models import BLOCK_ORDER_CHOICES, ConvolutionalVAE  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/vae_block_order_audit"
EXPECTED_LEGACY_ENCODER_PREFIX = ["Conv2d", "GELU", "GroupNorm", "Dropout2d"]
EXPECTED_LEGACY_FC = ["Linear", "GELU", "BatchNorm1d", "Dropout"]
EXPECTED_LEGACY_DECODER_PREFIX = ["ConvTranspose2d", "GELU", "GroupNorm", "Dropout2d"]
EXPECTED_NORM_ACT_ENCODER_PREFIX = ["Conv2d", "GroupNorm", "GELU", "Dropout2d"]
EXPECTED_NORM_ACT_FC = ["Linear", "BatchNorm1d", "GELU", "Dropout"]
EXPECTED_NORM_ACT_DECODER_PREFIX = ["ConvTranspose2d", "GroupNorm", "GELU", "Dropout2d"]


def build_model(block_order: str) -> ConvolutionalVAE:
    return ConvolutionalVAE(
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
        block_order=block_order,
    )


def class_names(seq: nn.Sequential) -> List[str]:
    return [module.__class__.__name__ for module in seq.children()]


def count_params(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def run_forward(model: ConvolutionalVAE) -> Dict[str, object]:
    x = torch.randn(2, 3, 131, 131)
    out: Dict[str, object] = {}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            recon, mu, logvar, z = model(x)
        out[f"{mode}_recon_shape"] = "x".join(map(str, recon.shape))
        out[f"{mode}_mu_shape"] = "x".join(map(str, mu.shape))
        out[f"{mode}_finite"] = bool(
            torch.isfinite(recon).all()
            and torch.isfinite(mu).all()
            and torch.isfinite(logvar).all()
            and torch.isfinite(z).all()
        )
    out["shape_ok"] = out["train_recon_shape"] == "2x3x131x131" and out["eval_recon_shape"] == "2x3x131x131"
    out["finite_ok"] = bool(out["train_finite"] and out["eval_finite"])
    return out


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    param_counts: Dict[str, int] = {}

    for block_order in BLOCK_ORDER_CHOICES:
        model = build_model(block_order)
        params = count_params(model)
        param_counts[block_order] = params
        enc = class_names(model.encoder_conv)
        enc_fc = class_names(model.encoder_fc_intermediate)
        dec_fc = class_names(model.decoder_fc_intermediate)
        dec = class_names(model.decoder_conv)
        forward = run_forward(model)

        if block_order == "legacy_act_norm":
            if enc[:4] != EXPECTED_LEGACY_ENCODER_PREFIX:
                failures.append(f"legacy encoder prefix changed: {enc[:4]}")
            if enc_fc != EXPECTED_LEGACY_FC:
                failures.append(f"legacy encoder FC changed: {enc_fc}")
            if dec_fc != EXPECTED_LEGACY_FC:
                failures.append(f"legacy decoder FC changed: {dec_fc}")
            if dec[:4] != EXPECTED_LEGACY_DECODER_PREFIX:
                failures.append(f"legacy decoder prefix changed: {dec[:4]}")
        elif block_order == "norm_act":
            if enc[:4] != EXPECTED_NORM_ACT_ENCODER_PREFIX:
                failures.append(f"norm_act encoder prefix unexpected: {enc[:4]}")
            if enc_fc != EXPECTED_NORM_ACT_FC:
                failures.append(f"norm_act encoder FC unexpected: {enc_fc}")
            if dec_fc != EXPECTED_NORM_ACT_FC:
                failures.append(f"norm_act decoder FC unexpected: {dec_fc}")
            if dec[:4] != EXPECTED_NORM_ACT_DECODER_PREFIX:
                failures.append(f"norm_act decoder prefix unexpected: {dec[:4]}")

        if not forward["shape_ok"] or not forward["finite_ok"]:
            failures.append(f"{block_order}: synthetic forward failed: {forward}")

        rows.append(
            {
                "block_order": block_order,
                "parameter_count": params,
                "encoder_conv_prefix": " -> ".join(enc[:4]),
                "encoder_fc_order": " -> ".join(enc_fc),
                "decoder_fc_order": " -> ".join(dec_fc),
                "decoder_conv_prefix": " -> ".join(dec[:4]),
                **forward,
            }
        )

    if len(set(param_counts.values())) != 1:
        failures.append(f"parameter counts differ across block orders: {param_counts}")

    write_csv(rows, OUTPUT_DIR / "block_order_smoke_test_results.csv")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "input_shape": [2, 3, 131, 131],
        "parameter_counts": param_counts,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    (OUTPUT_DIR / "block_order_smoke_command_log.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: block-order smoke test wrote {OUTPUT_DIR / 'block_order_smoke_test_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
