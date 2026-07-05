#!/usr/bin/env python3
"""Read-only reconstruction-scale probe for the post-final channel-mean loss.

The probe reuses the promoted fold-local VAE pools and VAE train indices, applies
the same fold-local normalization logic, and evaluates a zero-reconstruction
proxy under the historical and channel-normalized reconstruction losses. It does
not train a model and does not write tensors or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betavae_xai.normalization import normalize_inter_channel_fold  # noqa: E402
from run_vae_clf_ad_inference import (  # noqa: E402
    RECON_LOSS_MODE_CURRENT,
    RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM,
    vae_reconstruction_loss,
)

RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
REFERENCE_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
OUT_DEFAULT = RESULTS / "postfinal_method_branch_A_chmeanloss_B_foldcombat_preflight_20260606"
SELECTED_CHANNELS = [1, 0, 2]
SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_path_from_config(config_path: Path) -> Path:
    cfg = load_json(config_path)
    p = Path(cfg["paths"]["global_tensor_path"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def md_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return view.to_markdown(index=False) + "\n"


def best_rd_row(run_dir: Path, fold: int) -> dict[str, float]:
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
    rd = pd.read_csv(path)
    if "L_val_betaMax" in rd.columns:
        row = rd.loc[rd["L_val_betaMax"].astype(float).idxmin()]
    else:
        row = rd.iloc[-1]
    d_val = float(row.get("D_val", np.nan))
    r_nats = float(row.get("R_val_nats", np.nan))
    beta = float(row.get("beta", 3.75))
    return {
        "reference_best_epoch": float(row.get("epoch", np.nan)),
        "reference_D_val_best": d_val,
        "reference_R_val_nats_best": r_nats,
        "reference_beta_at_best": beta,
        "reference_beta_kld_over_D_best": float(beta * r_nats / d_val) if d_val else np.nan,
    }


def loss_value(x_np: np.ndarray, mode: str) -> float:
    x = torch.as_tensor(x_np, dtype=torch.float32)
    recon = torch.zeros_like(x)
    return float(vae_reconstruction_loss(recon, x, mode=mode).detach().cpu().item())


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    tensor_path = tensor_path_from_config(args.reference_config)
    with np.load(tensor_path, allow_pickle=False) as zf:
        tensor = np.asarray(zf["global_tensor_data"], dtype=np.float32)[:, SELECTED_CHANNELS, :, :]
        channel_names = [str(x) for x in np.asarray(zf["channel_names"]).astype(str).tolist()]
    selected_names = [channel_names[i] for i in SELECTED_CHANNELS]
    if selected_names != SELECTED_NAMES:
        raise RuntimeError(f"Selected channel mismatch: {selected_names}")

    rows: list[dict[str, Any]] = []
    for fold in args.folds:
        fold_dir = args.reference_run / f"fold_{fold}"
        pool_idx = np.load(fold_dir / "vae_training_pool_tensor_idx.npy").astype(int)
        train_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy").astype(int)
        val_local = np.load(fold_dir / "vae_internal_val_idx_local_to_pool.npy").astype(int)
        pool = tensor[pool_idx]
        pool_norm, _ = normalize_inter_channel_fold(
            pool,
            train_local,
            mode="zscore_offdiag",
            selected_channel_original_names=SELECTED_NAMES,
        )
        rd = best_rd_row(args.reference_run, fold)
        for split_name, local_idx in [
            ("vae_actual_train", train_local),
            ("vae_internal_val", val_local),
            ("vae_pool_all", np.arange(pool_norm.shape[0], dtype=int)),
        ]:
            if len(local_idx) == 0:
                continue
            x = pool_norm[local_idx]
            d_current = loss_value(x, RECON_LOSS_MODE_CURRENT)
            d_chmean = loss_value(x, RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM)
            ratio = d_chmean / d_current if d_current else np.nan
            est_actual_d_chmean = rd["reference_D_val_best"] * ratio
            rows.append(
                {
                    "fold": fold,
                    "split": split_name,
                    "n_subjects": int(len(local_idx)),
                    "selected_channels": ",".join(str(x) for x in SELECTED_CHANNELS),
                    "current_mode": RECON_LOSS_MODE_CURRENT,
                    "chmean_mode": RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM,
                    "probe_D_current_zero_recon": d_current,
                    "probe_D_chmean_zero_recon": d_chmean,
                    "probe_chmean_over_current_ratio": ratio,
                    "estimated_actual_D_chmean_from_reference_D": est_actual_d_chmean,
                    "estimated_beta_kld_over_D_chmean": (
                        rd["reference_beta_at_best"] * rd["reference_R_val_nats_best"] / est_actual_d_chmean
                        if est_actual_d_chmean
                        else np.nan
                    ),
                    **rd,
                }
            )

    df = pd.DataFrame(rows)
    csv = outdir / "branch_a_loss_scale_probe.csv"
    df.to_csv(csv, index=False)
    (outdir / "branch_a_loss_scale_probe.md").write_text(md_table(df), encoding="utf-8")
    summary = {
        "reference_run": str(args.reference_run),
        "reference_config": str(args.reference_config),
        "tensor_path": str(tensor_path),
        "selected_channel_names": selected_names,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "mean_chmean_over_current_ratio_vae_pool_all": float(
            df.loc[df["split"].eq("vae_pool_all"), "probe_chmean_over_current_ratio"].mean()
        ),
        "mean_estimated_beta_kld_over_D_chmean_vae_pool_all": float(
            df.loc[df["split"].eq("vae_pool_all"), "estimated_beta_kld_over_D_chmean"].mean()
        ),
    }
    (outdir / "branch_a_loss_scale_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
