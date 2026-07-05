#!/usr/bin/env python3
"""Smoke-test VAE checkpoint loading for block-order FAST Stage B.

This is read-only: it reconstructs the ConvolutionalVAE from each run's
run_config.json and checks that one saved fold checkpoint loads strictly.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026.run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (  # noqa: E402
    load_config,
    make_model,
    resolve,
)


OUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_block_order_fast3x3_ch1_0_2"
)
RUN_ROOT = OUT_ROOT / "runs"
RUNS = {
    "legacy_act_norm": RUN_ROOT / "legacy_act_norm",
    "norm_act": RUN_ROOT / "norm_act",
}


def tensor_shape(tensor_path: Path) -> tuple[int, int, int, int]:
    with np.load(resolve(tensor_path), allow_pickle=False) as zf:
        return tuple(int(v) for v in zf["global_tensor_data"].shape)


def smoke_one(name: str, run_dir: Path, fold: int = 1) -> Dict[str, Any]:
    cfg = load_config(resolve(run_dir))
    full_shape = tensor_shape(cfg["global_tensor_path"])
    selected_channels = list(cfg["channels_to_use"])
    n_channels = len(selected_channels)
    image_size = int(full_shape[-1])
    model = make_model(cfg, image_size=image_size, n_channels=n_channels, device=torch.device("cpu"))
    checkpoint_path = resolve(run_dir) / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    with torch.no_grad():
        x = torch.zeros(2, n_channels, image_size, image_size)
        recon, mu, logvar, _z = model(x)
    return {
        "run_name": name,
        "run_dir": str(resolve(run_dir)),
        "fold": fold,
        "checkpoint": str(checkpoint_path),
        "status": "load_ok",
        "vae_block_order": cfg.get("vae_block_order", "legacy_act_norm"),
        "vae_dropout_scope": cfg.get("vae_dropout_scope", "legacy_all"),
        "encoder_norm_mode": cfg.get("encoder_norm_mode", "groupnorm"),
        "vae_final_activation": cfg.get("vae_final_activation", "tanh"),
        "intermediate_fc_dim_vae": cfg.get("intermediate_fc_dim_vae", "quarter"),
        "decoder_type": cfg.get("decoder_type", "convtranspose"),
        "num_conv_layers_encoder": cfg.get("num_conv_layers_encoder", 4),
        "latent_dim": cfg.get("latent_dim", 256),
        "channels_to_use": json.dumps(selected_channels),
        "input_channels": n_channels,
        "input_shape": "x".join(map(str, x.shape)),
        "recon_shape": "x".join(map(str, recon.shape)),
        "mu_shape": "x".join(map(str, mu.shape)),
        "logvar_shape": "x".join(map(str, logvar.shape)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "finite_recon": bool(torch.isfinite(recon).all().item()),
        "finite_mu": bool(torch.isfinite(mu).all().item()),
        "finite_logvar": bool(torch.isfinite(logvar).all().item()),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name, run_dir in RUNS.items():
        rows.append(smoke_one(name, run_dir, fold=1))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ROOT / "block_order_stageb_checkpoint_load_smoke.csv", index=False)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "runs": {k: str(resolve(v)) for k, v in RUNS.items()},
        "status": "ok",
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (OUT_ROOT / "command_log_stageb_smoke.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
