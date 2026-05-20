#!/usr/bin/env python3
"""
Smoke-test explicit dropout scopes for ConvolutionalVAE.

This script is intentionally synthetic/read-only with respect to project data:
it instantiates small models, runs train/eval forward passes on random tensors,
and writes lightweight audit files under results/.
"""
from __future__ import annotations

import argparse
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

from betavae_xai.models import ConvolutionalVAE, DROPOUT_SCOPE_CHOICES  # noqa: E402


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "vae_dropout_scope_preflight"
)

EXPECTED_LEGACY_LOCATIONS = [
    "encoder_conv.3",
    "encoder_conv.7",
    "encoder_conv.11",
    "encoder_conv.15",
    "encoder_fc_intermediate.3",
    "decoder_fc_intermediate.3",
    "decoder_conv.3",
    "decoder_conv.7",
    "decoder_conv.11",
]

EXPECTED_SCOPE_COUNTS: Dict[str, Dict[str, int]] = {
    "legacy_all": {"dropout": 2, "dropout2d": 7, "total": 9},
    "encoder_only": {"dropout": 1, "dropout2d": 4, "total": 5},
    "no_decoder_dropout": {"dropout": 1, "dropout2d": 4, "total": 5},
    "encoder_fc_only": {"dropout": 1, "dropout2d": 0, "total": 1},
    "encoder_conv_only": {"dropout": 0, "dropout2d": 4, "total": 4},
    "none": {"dropout": 0, "dropout2d": 0, "total": 0},
}


def build_model(scope: str) -> ConvolutionalVAE:
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
        dropout_scope=scope,
    )


def dropout_locations(model: nn.Module) -> List[str]:
    return [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, (nn.Dropout, nn.Dropout2d))
    ]


def count_dropout_modules(model: nn.Module) -> Dict[str, int]:
    n_dropout = sum(1 for module in model.modules() if isinstance(module, nn.Dropout))
    n_dropout2d = sum(1 for module in model.modules() if isinstance(module, nn.Dropout2d))
    return {
        "dropout": n_dropout,
        "dropout2d": n_dropout2d,
        "total": n_dropout + n_dropout2d,
    }


def run_forward_checks(model: ConvolutionalVAE) -> Dict[str, object]:
    x = torch.randn(2, 3, 131, 131)
    checks: Dict[str, object] = {}
    for mode in ("train", "eval"):
        if mode == "train":
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            recon, mu, logvar, z = model(x)
        checks[f"{mode}_recon_shape"] = "x".join(map(str, recon.shape))
        checks[f"{mode}_mu_shape"] = "x".join(map(str, mu.shape))
        checks[f"{mode}_logvar_shape"] = "x".join(map(str, logvar.shape))
        checks[f"{mode}_z_shape"] = "x".join(map(str, z.shape))
        checks[f"{mode}_finite"] = bool(
            torch.isfinite(recon).all()
            and torch.isfinite(mu).all()
            and torch.isfinite(logvar).all()
            and torch.isfinite(z).all()
        )
    expected_shape = tuple(x.shape)
    checks["shape_ok"] = checks["train_recon_shape"] == "x".join(map(str, expected_shape))
    checks["finite_ok"] = bool(checks["train_finite"] and checks["eval_finite"])
    return checks


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    rows: List[Dict[str, object]] = []
    failures: List[str] = []

    for scope in DROPOUT_SCOPE_CHOICES:
        model = build_model(scope)
        counts = count_dropout_modules(model)
        locations = dropout_locations(model)
        forward = run_forward_checks(model)
        expected = EXPECTED_SCOPE_COUNTS[scope]

        counts_ok = counts == expected
        legacy_locations_ok = True
        if scope == "legacy_all":
            legacy_locations_ok = locations == EXPECTED_LEGACY_LOCATIONS
        if scope in {"encoder_only", "no_decoder_dropout"}:
            decoder_locations = [
                loc for loc in locations if loc.startswith("decoder_")
            ]
            if decoder_locations:
                failures.append(
                    f"{scope}: decoder dropout locations unexpectedly present: {decoder_locations}"
                )

        if not counts_ok:
            failures.append(
                f"{scope}: expected {expected}, observed {counts}"
            )
        if not legacy_locations_ok:
            failures.append(
                f"legacy_all locations changed: expected {EXPECTED_LEGACY_LOCATIONS}, observed {locations}"
            )
        if not forward["shape_ok"] or not forward["finite_ok"]:
            failures.append(f"{scope}: synthetic forward failed: {forward}")

        rows.append(
            {
                "dropout_scope": scope,
                "dropout_modules": counts["dropout"],
                "dropout2d_modules": counts["dropout2d"],
                "total_dropout_modules": counts["total"],
                "expected_total_dropout_modules": expected["total"],
                "counts_ok": counts_ok,
                "legacy_locations_ok": legacy_locations_ok if scope == "legacy_all" else "",
                "locations": ";".join(locations),
                **forward,
            }
        )

    counts_path = output_dir / "dropout_scope_module_counts.csv"
    write_csv(rows, counts_path)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# VAE Dropout Scope Preflight",
                "",
                "This is a synthetic, read-only smoke test for explicit `ConvolutionalVAE` dropout scopes.",
                "",
                "No ADNI tensor, metadata, ledger, checkpoint, or existing result folder was modified.",
                "",
                "## Expected Behavior",
                "",
                "- `legacy_all` preserves the locked/current behavior: 9 dropout modules.",
                "- `encoder_only` and `no_decoder_dropout` keep encoder dropout and remove decoder dropout.",
                "- `encoder_fc_only` keeps only the encoder FC dropout.",
                "- `encoder_conv_only` keeps only encoder convolutional dropout.",
                "- `none` removes all explicit dropout modules.",
                "",
                "## Result",
                "",
                f"- Smoke status: {'PASS' if not failures else 'FAIL'}",
                f"- Module count table: `{counts_path.name}`",
                "",
                "## Legacy Locations",
                "",
                "Expected `legacy_all` dropout locations:",
                "",
                "\n".join(f"- `{loc}`" for loc in EXPECTED_LEGACY_LOCATIONS),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir.resolve()),
        "dropout_scope_choices": list(DROPOUT_SCOPE_CHOICES),
        "input_shape": [2, 3, 131, 131],
        "dropout_rate": 0.15,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2), encoding="utf-8"
    )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print(f"PASS: dropout-scope smoke test wrote {counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
