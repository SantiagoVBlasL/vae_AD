#!/usr/bin/env python3
"""Synthetic smoke test for optional Age/Sex decoder conditioning in ConvolutionalVAE.

Read-only test. It instantiates default and decoder-only variants, checks shape
compatibility, verifies the default parameter keys remain unchanged relative to
an explicit none/none configuration, and writes a small report.
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
    latent_covariate_corr_penalty,
    summarize_vae_conditioning_qc,
    transform_vae_conditioning_covariates,
)


OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_age_sex_fast3x3_preflight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def module_keys(model: torch.nn.Module) -> List[str]:
    return sorted(model.state_dict().keys())


def run_case(name: str, conditioning_mode: str, conditioning_dim: int) -> Dict[str, Any]:
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
    c = torch.randn(2, conditioning_dim) if conditioning_dim else None
    recon, mu, logvar, z = model(x, condition=c)
    penalty = latent_covariate_corr_penalty(mu, c) if c is not None else torch.tensor(0.0)
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
        "corr_penalty_finite": bool(torch.isfinite(penalty).item()),
        "corr_penalty_value": float(penalty.detach().cpu().item()),
    }


def run_missing_covariate_case() -> Dict[str, Any]:
    train_rows = pd.DataFrame(
        {
            "SubjectID": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "Age": [70, "NA", 74, 68, "", 72],
            "Sex": ["F", "M", "nan", None, "Female", "1"],
        }
    )
    val_rows = pd.DataFrame(
        {
            "SubjectID": ["V1", "V2", "V3"],
            "Age": [69, None, 71],
            "Sex": ["", "male", "0"],
        }
    )
    test_rows = pd.DataFrame(
        {
            "SubjectID": ["T1", "T2", "T3"],
            "Age": ["NaN", 77, 66],
            "Sex": ["None", "FEMALE", "MALE"],
        }
    )
    transformer = fit_vae_conditioning_transformer(train_rows, "age_sex")
    train_cond = transform_vae_conditioning_covariates(train_rows, transformer)
    val_cond = transform_vae_conditioning_covariates(val_rows, transformer)
    test_cond = transform_vae_conditioning_covariates(test_rows, transformer)
    qc = summarize_vae_conditioning_qc(train_rows, val_rows, test_rows, transformer)
    try:
        bad_rows = pd.DataFrame({"SubjectID": ["B1"], "Age": [70], "Sex": ["unsupported"]})
        transform_vae_conditioning_covariates(bad_rows, transformer)
        unknown_sex_failed = False
    except ValueError:
        unknown_sex_failed = True
    return {
        "case": "missing_age_sex_fold_safe_transform",
        "conditioning_mode": "decoder_only",
        "conditioning_dim": int(train_cond.shape[1]),
        "train_shape": "x".join(map(str, train_cond.shape)),
        "val_shape": "x".join(map(str, val_cond.shape)),
        "test_shape": "x".join(map(str, test_cond.shape)),
        "sex_impute_value": transformer.get("sex_impute_value"),
        "age_impute_value": transformer.get("age_impute_value"),
        "age_mean_train": transformer.get("age_mean"),
        "age_std_train": transformer.get("age_std"),
        "sex_missing_train": qc.get("sex_missing_train"),
        "sex_missing_val": qc.get("sex_missing_val"),
        "sex_missing_test": qc.get("sex_missing_test"),
        "age_missing_train": qc.get("age_missing_train"),
        "age_missing_val": qc.get("age_missing_val"),
        "age_missing_test": qc.get("age_missing_test"),
        "unknown_nonmissing_sex_failed": unknown_sex_failed,
        "transform_finite": bool(torch.isfinite(torch.tensor(train_cond)).all().item())
        and bool(torch.isfinite(torch.tensor(val_cond)).all().item())
        and bool(torch.isfinite(torch.tensor(test_cond)).all().item()),
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
        run_case("default_implicit_none", "none", 0),
        run_case("explicit_none", "none", 0),
        run_case("decoder_only_sex", "decoder_only", 1),
        run_case("decoder_only_age_sex", "decoder_only", 2),
    ]
    for row in rows:
        row["default_keys_match_explicit_none"] = default_keys_match

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "smoke_test_results.csv", index=False)
    view = df.copy()
    (outdir / "smoke_test_results.md").write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")
    missing_case = run_missing_covariate_case()
    missing_df = pd.DataFrame([missing_case])
    missing_df.to_csv(outdir / "smoke_test_conditioning_missing_values.csv", index=False)
    (outdir / "smoke_test_conditioning_missing_values.md").write_text(missing_df.to_markdown(index=False) + "\n", encoding="utf-8")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "output_dir": str(outdir),
        "default_keys_match_explicit_none": default_keys_match,
        "cases": rows,
        "missing_covariate_case": missing_case,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (outdir / "conditional_vae_smoke_command_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    if not default_keys_match:
        raise RuntimeError("Default implicit model keys differ from explicit conditioning_mode=none keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
