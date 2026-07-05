#!/usr/bin/env python3
"""Synthetic smoke test for optional Manufacturer decoder conditioning.

This is read-only with respect to tensors, metadata, ledgers, configs, and
model outputs. It verifies that the default VAE remains unchanged, that
decoder-only manufacturer conditioning has valid shapes, and that the
fold-local manufacturer transformer normalizes known variants while failing on
unseen categories.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_DIR):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from betavae_xai.models import ConvolutionalVAE
from scripts.run_vae_clf_ad_inference import (
    fit_vae_conditioning_transformer,
    transform_vae_conditioning_covariates,
)


OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_manufacturer_fast3x3_preflight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def module_keys(model: torch.nn.Module) -> List[str]:
    return sorted(model.state_dict().keys())


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def run_model_case(name: str, conditioning_mode: str, conditioning_dim: int) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = ConvolutionalVAE(
        input_channels=3,
        latent_dim=16,
        image_size=131,
        final_activation="tanh",
        intermediate_fc_dim_config="quarter",
        dropout_rate=0.15,
        num_conv_layers_encoder=4,
        decoder_type="convtranspose",
        dropout_scope="legacy_all",
        block_order="legacy_act_norm",
        conditioning_mode=conditioning_mode,
        conditioning_dim=conditioning_dim,
    )
    model.train()
    x = torch.randn(2, 3, 131, 131)
    c = torch.eye(conditioning_dim, dtype=torch.float32)[:2] if conditioning_dim else None
    recon, mu, logvar, z = model(x, condition=c)
    model.eval()
    with torch.no_grad():
        recon_eval, mu_eval, logvar_eval, z_eval = model(x, condition=c)
    return {
        "case": name,
        "conditioning_mode": conditioning_mode,
        "conditioning_dim": conditioning_dim,
        "n_parameters": count_params(model),
        "n_state_keys": len(module_keys(model)),
        "train_recon_shape": "x".join(map(str, recon.shape)),
        "train_mu_shape": "x".join(map(str, mu.shape)),
        "eval_recon_shape": "x".join(map(str, recon_eval.shape)),
        "eval_mu_shape": "x".join(map(str, mu_eval.shape)),
        "train_recon_finite": bool(torch.isfinite(recon).all().item()),
        "eval_recon_finite": bool(torch.isfinite(recon_eval).all().item()),
    }


def run_transformer_case() -> Dict[str, Any]:
    train_rows = pd.DataFrame(
        {
            "SubjectID": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "Manufacturer": [
                "GE",
                "General Electric",
                "Philips",
                "Philips Medical Systems",
                "SIEMENS",
                "Siemens Healthineers",
            ],
        }
    )
    val_rows = pd.DataFrame(
        {
            "SubjectID": ["V1", "V2", "V3"],
            "Manufacturer": ["GE Medical Systems", "Philips Healthcare", "Siemens"],
        }
    )
    transformer = fit_vae_conditioning_transformer(train_rows, "manufacturer")
    train_cond = transform_vae_conditioning_covariates(train_rows, transformer)
    val_cond = transform_vae_conditioning_covariates(val_rows, transformer)
    try:
        bad_rows = pd.DataFrame({"SubjectID": ["B1"], "Manufacturer": ["Canon"]})
        transform_vae_conditioning_covariates(bad_rows, transformer)
        unseen_failed = False
    except ValueError:
        unseen_failed = True
    return {
        "case": "manufacturer_fold_safe_transform",
        "conditioning_dim": int(train_cond.shape[1]),
        "categories": "|".join(transformer.get("manufacturer_categories", [])),
        "train_shape": "x".join(map(str, train_cond.shape)),
        "val_shape": "x".join(map(str, val_cond.shape)),
        "train_onehot_rowsum_ok": bool((train_cond.sum(axis=1) == 1.0).all()),
        "val_onehot_rowsum_ok": bool((val_cond.sum(axis=1) == 1.0).all()),
        "unseen_nonmissing_manufacturer_failed": unseen_failed,
        "transform_finite": bool(torch.isfinite(torch.tensor(train_cond)).all().item())
        and bool(torch.isfinite(torch.tensor(val_cond)).all().item()),
    }


def main() -> int:
    args = parse_args()
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    default_model = ConvolutionalVAE(input_channels=3, latent_dim=16, image_size=131)
    explicit_none = ConvolutionalVAE(
        input_channels=3,
        latent_dim=16,
        image_size=131,
        conditioning_mode="none",
        conditioning_dim=0,
    )
    default_keys_match = module_keys(default_model) == module_keys(explicit_none)
    rows = [
        run_model_case("default_implicit_none", "none", 0),
        run_model_case("explicit_none", "none", 0),
        run_model_case("decoder_only_manufacturer", "decoder_only", 3),
    ]
    for row in rows:
        row["default_keys_match_explicit_none"] = default_keys_match
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "smoke_test_results.csv", index=False)
    (outdir / "smoke_test_results.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")

    transform_case = run_transformer_case()
    transform_df = pd.DataFrame([transform_case])
    transform_df.to_csv(outdir / "smoke_test_manufacturer_transformer.csv", index=False)
    (outdir / "smoke_test_manufacturer_transformer.md").write_text(transform_df.to_markdown(index=False) + "\n", encoding="utf-8")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "output_dir": str(outdir),
        "default_keys_match_explicit_none": default_keys_match,
        "model_cases": rows,
        "transformer_case": transform_case,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    (outdir / "smoke_test_command_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(transform_df.to_string(index=False))
    if not default_keys_match:
        raise RuntimeError("Default implicit model keys differ from explicit conditioning_mode=none keys.")
    if not transform_case["unseen_nonmissing_manufacturer_failed"]:
        raise RuntimeError("Unseen Manufacturer did not fail as required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
