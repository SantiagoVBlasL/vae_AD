#!/usr/bin/env python3
"""Audit normalized targets and VAE reconstructions for selected channel sets.

Read-only audit. It loads saved fold-specific VAE checkpoints and normalization
parameters, reconstructs normalized tensors deterministically via decode(mu),
and writes lightweight CSV/Markdown summaries. It never trains or modifies
tensors, metadata, or ledger files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from betavae_xai.models.convolutional_vae import ConvolutionalVAE  # noqa: E402
from betavae_xai.normalization import apply_normalization_params  # noqa: E402


RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
DATASET_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass"
)
TENSOR_PATH = (
    DATASET_ROOT
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
DEFAULT_OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_vae_recon_distribution_audit"

RUNS = [
    {
        "run_id": "ch1_full_5x5",
        "label": "[1]",
        "channels": [1],
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate",
    },
    {
        "run_id": "ch1_4_full_5x5",
        "label": "[1,4]",
        "channels": [1, 4],
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate",
    },
    {
        "run_id": "ch1_0_2_full_5x5",
        "label": "[1,0,2]",
        "channels": [1, 0, 2],
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tensor-path", type=Path, default=TENSOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    return path.resolve()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_torch_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object in {path}: {type(obj)}")


def run_params(run_dir: Path) -> Dict[str, Any]:
    cfg = read_json(run_dir / "run_config.json") or read_json(run_dir / "run_manifest.json")
    if not cfg:
        # Most wrappers store the authoritative config in configs/runs, but run
        # dirs also contain enough fold artifacts. Fall back to current full-run
        # defaults used by these candidates.
        return {
            "latent_dim": 256,
            "image_size": 131,
            "vae_final_activation": "tanh",
            "intermediate_fc_dim_vae": "quarter",
            "dropout_rate_vae": 0.15,
            "use_layernorm_vae_fc": False,
            "num_conv_layers_encoder": 4,
            "decoder_type": "convtranspose",
            "beta_vae": 2.5,
        }
    params = cfg.get("parameters", cfg.get("args", cfg))
    return {
        "latent_dim": int(params.get("latent_dim", 256)),
        "image_size": int(params.get("image_size", 131)),
        "vae_final_activation": str(params.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_vae": params.get("intermediate_fc_dim_vae", "quarter"),
        "dropout_rate_vae": float(params.get("dropout_rate_vae", 0.15)),
        "use_layernorm_vae_fc": bool(params.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(params.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(params.get("decoder_type", "convtranspose")),
        "beta_vae": float(params.get("beta_vae", 2.5)),
    }


def build_model(params: Dict[str, Any], n_channels: int, state_path: Path, device: torch.device) -> ConvolutionalVAE:
    model = ConvolutionalVAE(
        input_channels=n_channels,
        latent_dim=int(params["latent_dim"]),
        image_size=int(params["image_size"]),
        final_activation=str(params["vae_final_activation"]),
        intermediate_fc_dim_config=params["intermediate_fc_dim_vae"],
        dropout_rate=float(params["dropout_rate_vae"]),
        use_layernorm_fc=bool(params["use_layernorm_vae_fc"]),
        num_conv_layers_encoder=int(params["num_conv_layers_encoder"]),
        decoder_type=str(params["decoder_type"]),
    )
    model.load_state_dict(load_torch_state_dict(state_path))
    model.to(device)
    model.eval()
    return model


def split_indices(fold_dir: Path) -> Dict[str, np.ndarray]:
    pool = np.load(fold_dir / "vae_training_pool_tensor_idx.npy").astype(int)
    train_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy").astype(int)
    val_local = np.load(fold_dir / "vae_internal_val_idx_local_to_pool.npy").astype(int)
    splits = {
        "vae_pool": pool,
        "vae_train": pool[train_local],
        "vae_internal_val": pool[val_local],
    }
    test_path = fold_dir / "test_tensor_idx.npy"
    if test_path.exists():
        splits["test_adcn"] = np.load(test_path).astype(int)
    train_dev_path = fold_dir / "train_dev_tensor_idx.npy"
    if train_dev_path.exists():
        splits["train_dev_adcn"] = np.load(train_dev_path).astype(int)
    return splits


def target_stats(
    x_norm: np.ndarray,
    offdiag_mask: np.ndarray,
    diag_mask: np.ndarray,
    upper_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n, c_count, _, _ = x_norm.shape
    for c in range(c_count):
        mat = x_norm[:, c]
        off = mat[:, offdiag_mask]
        diag = mat[:, diag_mask]
        upper = mat[:, upper_mask]
        rows.append(
            {
                "target_pct_abs_gt_1_all": float((np.abs(mat) > 1.0).mean() * 100.0),
                "target_pct_abs_gt_1_offdiag": float((np.abs(off) > 1.0).mean() * 100.0),
                "target_pct_abs_gt_1_upper": float((np.abs(upper) > 1.0).mean() * 100.0),
                "target_min_all": float(np.min(mat)),
                "target_max_all": float(np.max(mat)),
                "target_mean_all": float(np.mean(mat)),
                "target_std_all": float(np.std(mat)),
                "target_min_offdiag": float(np.min(off)),
                "target_max_offdiag": float(np.max(off)),
                "target_mean_offdiag": float(np.mean(off)),
                "target_std_offdiag": float(np.std(off)),
                "target_min_upper": float(np.min(upper)),
                "target_max_upper": float(np.max(upper)),
                "target_mean_upper": float(np.mean(upper)),
                "target_std_upper": float(np.std(upper)),
                "target_diag_abs_mean": float(np.mean(np.abs(diag))),
                "n_subjects": int(n),
            }
        )
    return rows


@torch.no_grad()
def audit_split(
    model: ConvolutionalVAE,
    x_norm: np.ndarray,
    *,
    run_id: str,
    run_label: str,
    channels: List[int],
    channel_names: List[str],
    fold: int,
    split_name: str,
    beta: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n, c_count, r, _ = x_norm.shape
    offdiag_mask_np = ~np.eye(r, dtype=bool)
    diag_mask_np = np.eye(r, dtype=bool)
    upper_mask_np = np.triu(np.ones((r, r), dtype=bool), k=1)
    offdiag_mask_t = torch.from_numpy(offdiag_mask_np).to(device)
    diag_mask_t = torch.from_numpy(diag_mask_np).to(device)
    upper_mask_t = torch.from_numpy(upper_mask_np).to(device)

    tstats = target_stats(x_norm, offdiag_mask_np, diag_mask_np, upper_mask_np)
    sse_all = np.zeros(c_count, dtype=np.float64)
    sse_offdiag = np.zeros(c_count, dtype=np.float64)
    sse_diag = np.zeros(c_count, dtype=np.float64)
    sse_upper = np.zeros(c_count, dtype=np.float64)
    recon_sat_all = np.zeros(c_count, dtype=np.float64)
    recon_sat_offdiag = np.zeros(c_count, dtype=np.float64)
    recon_sat_upper = np.zeros(c_count, dtype=np.float64)
    target_count_all = float(n * r * r)
    target_count_offdiag = float(n * int(offdiag_mask_np.sum()))
    target_count_diag = float(n * r)
    target_count_upper = float(n * int(upper_mask_np.sum()))
    mu_chunks: List[np.ndarray] = []
    logvar_chunks: List[np.ndarray] = []
    kld_dim_sum: np.ndarray | None = None

    for start in range(0, n, batch_size):
        xb_np = x_norm[start : start + batch_size]
        xb = torch.from_numpy(xb_np).float().to(device)
        mu, logvar = model.encode(xb)
        recon = model.decode(mu)
        if recon.shape != xb.shape:
            recon = F.interpolate(recon, size=(xb.shape[2], xb.shape[3]), mode="bilinear", align_corners=False)
        diff2 = (recon - xb).pow(2)
        kld_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kld_np = kld_dim.detach().cpu().numpy()
        kld_dim_sum = kld_np.sum(axis=0) if kld_dim_sum is None else kld_dim_sum + kld_np.sum(axis=0)
        mu_chunks.append(mu.detach().cpu().numpy())
        logvar_chunks.append(logvar.detach().cpu().numpy())
        for c in range(c_count):
            d = diff2[:, c]
            rc = recon[:, c]
            sse_all[c] += float(d.sum().detach().cpu())
            sse_offdiag[c] += float(d[:, offdiag_mask_t].sum().detach().cpu())
            sse_diag[c] += float(d[:, diag_mask_t].sum().detach().cpu())
            sse_upper[c] += float(d[:, upper_mask_t].sum().detach().cpu())
            recon_sat_all[c] += float((rc.abs() > 0.95).sum().detach().cpu())
            recon_sat_offdiag[c] += float((rc[:, offdiag_mask_t].abs() > 0.95).sum().detach().cpu())
            recon_sat_upper[c] += float((rc[:, upper_mask_t].abs() > 0.95).sum().detach().cpu())

    mu_all = np.concatenate(mu_chunks, axis=0)
    logvar_all = np.concatenate(logvar_chunks, axis=0)
    kld_dim_mean = kld_dim_sum / float(n)
    kld_per_sample = 0.5 * (mu_all**2 + np.exp(logvar_all) - 1.0 - logvar_all).sum(axis=1)
    total_recon_sse_mean = float(sse_all.sum() / n)
    total_recon_mse_per_pixel = float(sse_all.sum() / (n * c_count * r * r))
    mean_kld = float(np.mean(kld_per_sample))
    mean_kld_per_dim = float(np.mean(kld_dim_mean))
    active_units = int((np.var(mu_all, axis=0) > 1e-4).sum())
    low_kld_dims = int((kld_dim_mean < 1e-3).sum())
    very_low_kld_dims = int((kld_dim_mean < 1e-4).sum())
    frac_active = active_units / float(mu_all.shape[1])
    if frac_active < 0.10 or mean_kld_per_dim < 1e-3:
        collapse_flag = "possible_collapse"
    elif frac_active < 0.50 or mean_kld_per_dim < 1e-2:
        collapse_flag = "watch"
    else:
        collapse_flag = "no_obvious_collapse"

    channel_rows: List[Dict[str, Any]] = []
    for c in range(c_count):
        recon_sse_channel_per_subject = float(sse_all[c] / n)
        channel_rows.append(
            {
                "run_id": run_id,
                "run_label": run_label,
                "channels": json.dumps(channels),
                "fold": fold,
                "split": split_name,
                "channel_position": c,
                "channel_index": channels[c],
                "channel_name": channel_names[c],
                **tstats[c],
                "recon_pct_abs_gt_0p95_all": float(recon_sat_all[c] / target_count_all * 100.0),
                "recon_pct_abs_gt_0p95_offdiag": float(recon_sat_offdiag[c] / target_count_offdiag * 100.0),
                "recon_pct_abs_gt_0p95_upper": float(recon_sat_upper[c] / target_count_upper * 100.0),
                "recon_mse_all": float(sse_all[c] / target_count_all),
                "recon_mse_offdiag": float(sse_offdiag[c] / target_count_offdiag),
                "recon_mse_diag": float(sse_diag[c] / target_count_diag),
                "recon_mse_upper": float(sse_upper[c] / target_count_upper),
                "recon_sse_per_subject_channel": recon_sse_channel_per_subject,
                "kld_mean": mean_kld,
                "kld_per_latent_dim": mean_kld_per_dim,
                "kld_over_channel_recon_sse": float(mean_kld / recon_sse_channel_per_subject)
                if recon_sse_channel_per_subject
                else np.nan,
                "beta_kld_over_channel_recon_sse": float(beta * mean_kld / recon_sse_channel_per_subject)
                if recon_sse_channel_per_subject
                else np.nan,
                "active_units": active_units,
                "frac_active_units": frac_active,
                "posterior_collapse_indicator": collapse_flag,
            }
        )

    latent_row = {
        "run_id": run_id,
        "run_label": run_label,
        "channels": json.dumps(channels),
        "fold": fold,
        "split": split_name,
        "n_subjects": int(n),
        "latent_dim": int(mu_all.shape[1]),
        "active_units_var_mu_gt_1e_4": active_units,
        "frac_active_units": frac_active,
        "kld_mean": mean_kld,
        "kld_per_latent_dim": mean_kld_per_dim,
        "kld_median_per_subject": float(np.median(kld_per_sample)),
        "kld_p05_per_subject": float(np.percentile(kld_per_sample, 5)),
        "kld_p95_per_subject": float(np.percentile(kld_per_sample, 95)),
        "low_kld_dims_lt_1e_3": low_kld_dims,
        "very_low_kld_dims_lt_1e_4": very_low_kld_dims,
        "mean_abs_mu": float(np.mean(np.abs(mu_all))),
        "mean_logvar": float(np.mean(logvar_all)),
        "mean_posterior_std": float(np.mean(np.exp(0.5 * logvar_all))),
        "total_recon_sse_per_subject": total_recon_sse_mean,
        "total_recon_mse_per_pixel": total_recon_mse_per_pixel,
        "kld_over_total_recon_sse": float(mean_kld / total_recon_sse_mean) if total_recon_sse_mean else np.nan,
        "beta_kld_over_total_recon_sse": float(beta * mean_kld / total_recon_sse_mean)
        if total_recon_sse_mean
        else np.nan,
        "posterior_collapse_indicator": collapse_flag,
        "reconstruction_mode": "deterministic_decode_mu",
    }
    return channel_rows, latent_row


def rate_distortion_rows(run: Dict[str, Any], run_dir: Path, params: Dict[str, Any], channels: List[int], n_rois: int, folds: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_pixels = len(channels) * n_rois * n_rois
    latent_dim = int(params["latent_dim"])
    beta_max = float(params["beta_vae"])
    for fold in range(1, folds + 1):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            out = {
                "run_id": run["run_id"],
                "run_label": run["label"],
                "channels": json.dumps(channels),
                "fold": fold,
                "epoch": int(r["epoch"]),
                "beta": float(r.get("beta", np.nan)),
                "n_pixels": int(n_pixels),
                "latent_dim": latent_dim,
                "D_train": float(r.get("D_train", np.nan)),
                "D_val": float(r.get("D_val", np.nan)),
                "R_train_nats": float(r.get("R_train_nats", np.nan)),
                "R_val_nats": float(r.get("R_val_nats", np.nan)),
            }
            out["D_train_per_pixel"] = out["D_train"] / n_pixels if n_pixels else np.nan
            out["D_val_per_pixel"] = out["D_val"] / n_pixels if n_pixels else np.nan
            out["R_train_per_latent_dim"] = out["R_train_nats"] / latent_dim if latent_dim else np.nan
            out["R_val_per_latent_dim"] = out["R_val_nats"] / latent_dim if latent_dim else np.nan
            out["R_train_over_D_train"] = out["R_train_nats"] / out["D_train"] if out["D_train"] else np.nan
            out["R_val_over_D_val"] = out["R_val_nats"] / out["D_val"] if out["D_val"] else np.nan
            out["beta_max_R_train_over_D_train"] = beta_max * out["R_train_nats"] / out["D_train"] if out["D_train"] else np.nan
            out["beta_max_R_val_over_D_val"] = beta_max * out["R_val_nats"] / out["D_val"] if out["D_val"] else np.nan
            rows.append(out)
    return rows


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_readme(
    output_dir: Path,
    summary: pd.DataFrame,
    latent_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# VAE Target/Reconstruction Distribution Audit",
        "",
        "This is a read-only audit of saved VAE checkpoints for `[1]`, `[1,4]`, and `[1,0,2]`.",
        "No training was launched and no tensor, metadata, or ledger files were modified.",
        "",
        "Reconstructions are deterministic `decode(mu)` outputs in `model.eval()` mode. This avoids stochastic sampling noise and measures the decoder/readout representation used downstream.",
        "",
        "## Key Interpretation",
        "",
        "- `target_pct_abs_gt_1_offdiag` quantifies how much of the z-score-normalized target sits outside the `tanh` decoder range.",
        "- `recon_pct_abs_gt_0p95_offdiag` quantifies decoder saturation near the `tanh` limits.",
        "- `recon_mse_all`, `recon_mse_offdiag`, and `recon_mse_diag` are per-element MSEs.",
        "- `kld_over_total_recon_sse` and `beta_kld_over_total_recon_sse` use the original summed reconstruction scale, matching the training loss scale.",
        "- Rate-distortion curves are additionally normalized by `n_pixels = C * 131 * 131`.",
        "",
        "## Fold/Channel Summary",
        "",
        md_table(summary),
        "",
        "## Latent/KLD Summary",
        "",
        md_table(latent_summary),
        "",
        "## Outputs",
        "",
        "- `target_distribution_by_channel.csv`",
        "- `reconstruction_saturation_by_fold.csv`",
        "- `loss_scale_comparison.csv`",
        "- `latent_capacity_summary.csv`",
        "- `recommended_next_experiments.md`",
        "",
        "Additional detailed outputs:",
        "",
        "- `target_reconstruction_by_fold_channel_split.csv`",
        "- `target_reconstruction_summary_by_run_channel_split.csv`",
        "- `latent_kld_collapse_by_fold_split.csv`",
        "- `latent_kld_collapse_summary_by_run_split.csv`",
        "- `rate_distortion_normalized.csv`",
        "- `rate_distortion_normalized_summary.csv`",
        "- `command_log.json`",
        "",
        "## Safety",
        "",
        f"- device requested: `{args.device}`",
        f"- batch_size: `{args.batch_size}`",
        "- training_launched: `false`",
        "- tensor_modified: `false`",
        "- metadata_modified: `false`",
        "- ledger_modified: `false`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next_experiments(output_dir: Path) -> None:
    lines = [
        "# Recommended Next Experiments",
        "",
        "These recommendations follow directly from the objective-fit audit. They are minimal, isolated tests and should not be combined initially.",
        "",
        "## 1. Off-Diagonal Mean MSE Objective",
        "",
        "Problem: the current VAE uses `MSE sum / batch_size`, so the reconstruction term grows with `C * 131 * 131` while KLD does not.",
        "",
        "Test:",
        "",
        "- Replace reconstruction loss with mean squared error over off-diagonal entries and selected channels.",
        "- Keep beta, latent_dim, architecture, splits, and classifier readout fixed.",
        "- Compare `[1]` and `[1,0,2]` first.",
        "",
        "Expected value: this directly tests whether channel-selection differences are partly caused by channel-count-dependent beta pressure.",
        "",
        "## 2. Linear Decoder Output For zscore_offdiag",
        "",
        "Problem: normalized targets are z-scored and unbounded, while the current decoder uses `tanh` and cannot reconstruct values outside `[-1,1]`.",
        "",
        "Test:",
        "",
        "- Add a no-final-activation/linear decoder option.",
        "- Keep the current loss unchanged for the first pass.",
        "- Run `[1]` and `[1,0,2]`.",
        "",
        "Expected value: this isolates whether `tanh` saturation is suppressing useful edge variation.",
        "",
        "## 3. Objective Scaling Report On Existing Runs",
        "",
        "Before retraining, report per-fold:",
        "",
        "- `D_per_pixel = reconstruction_sum / (C * 131 * 131)`",
        "- `KLD_per_dim = KLD / latent_dim`",
        "- `beta*KLD / reconstruction_sum`",
        "- `beta*KLD_per_dim / D_per_pixel`",
        "",
        "This should accompany any manuscript interpretation of channel ablations.",
        "",
        "## 4. Upper-Triangle Reconstruction Loss For Symmetric Channels",
        "",
        "Problem: symmetric connectomes are treated as generic images; both triangles duplicate the same edge information.",
        "",
        "Test only after Experiment 1:",
        "",
        "- Use upper-triangle off-diagonal MSE for symmetric channels.",
        "- Preserve full off-diagonal loss for any directional channel if used.",
        "",
        "## 5. Capacity Sensitivity After Loss Fix",
        "",
        "Only after the objective is made comparable across channel counts:",
        "",
        "- Test `intermediate_fc_dim_vae=half` for `[1,0,2]`.",
        "- Avoid broad beta/architecture sweeps until the loss scaling issue is resolved.",
        "",
        "## Current Recommendation",
        "",
        "The first real follow-up should be `[1]` vs `[1,0,2]` with off-diagonal mean MSE and the same classifier-only `logreg_l2` leakage-safe threshold readout. This is the cleanest test of whether the current objective, not just channel biology, is driving performance differences.",
    ]
    (output_dir / "recommended_next_experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    if args.dry_run:
        print("DRY RUN: would audit saved VAE reconstructions")
        print(f"Tensor: {resolve(args.tensor_path)}")
        for run in RUNS:
            print(f"- {run['run_id']}: {resolve(run['run_dir'])}, channels={run['channels']}")
        print(f"Output: {output_dir}")
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tensor_path = resolve(args.tensor_path)
    npz = np.load(tensor_path, allow_pickle=False)
    tensor = npz["global_tensor_data"].astype(np.float32, copy=False)
    all_channel_names = [str(x) for x in npz["channel_names"].tolist()]
    n_rois = int(tensor.shape[-1])

    channel_rows: List[Dict[str, Any]] = []
    latent_rows: List[Dict[str, Any]] = []
    rd_rows: List[Dict[str, Any]] = []
    availability_rows: List[Dict[str, Any]] = []

    for run in RUNS:
        run_dir = resolve(run["run_dir"])
        params = run_params(run_dir)
        channels = list(run["channels"])
        channel_names = [all_channel_names[i] for i in channels]
        selected_tensor = tensor[:, channels, :, :]
        rd_rows.extend(rate_distortion_rows(run, run_dir, params, channels, n_rois, args.folds))
        print(f"Auditing {run['run_id']} channels={channels} dir={run_dir}")
        for fold in range(1, args.folds + 1):
            fold_dir = run_dir / f"fold_{fold}"
            state_path = fold_dir / f"vae_model_fold_{fold}.pt"
            norm_path = fold_dir / "vae_norm_params.joblib"
            ok = state_path.exists() and norm_path.exists()
            availability_rows.append(
                {
                    "run_id": run["run_id"],
                    "fold": fold,
                    "fold_dir": str(fold_dir),
                    "vae_model_exists": state_path.exists(),
                    "norm_params_exists": norm_path.exists(),
                }
            )
            if not ok:
                print(f"  fold {fold}: missing model or norm params, skipping")
                continue
            model = build_model(params, len(channels), state_path, device)
            norm_params = joblib.load(norm_path)
            splits = split_indices(fold_dir)
            for split_name, idx in splits.items():
                x_raw = selected_tensor[idx]
                x_norm = apply_normalization_params(x_raw, norm_params).astype(np.float32, copy=False)
                rows, latent = audit_split(
                    model,
                    x_norm,
                    run_id=run["run_id"],
                    run_label=run["label"],
                    channels=channels,
                    channel_names=channel_names,
                    fold=fold,
                    split_name=split_name,
                    beta=float(params["beta_vae"]),
                    batch_size=args.batch_size,
                    device=device,
                )
                channel_rows.extend(rows)
                latent_rows.append(latent)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    channel_df = pd.DataFrame(channel_rows)
    latent_df = pd.DataFrame(latent_rows)
    rd_df = pd.DataFrame(rd_rows)
    availability_df = pd.DataFrame(availability_rows)

    channel_summary = (
        channel_df.groupby(["run_id", "run_label", "channels", "split", "channel_index", "channel_name"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            target_pct_abs_gt_1_offdiag_mean=("target_pct_abs_gt_1_offdiag", "mean"),
            target_pct_abs_gt_1_upper_mean=("target_pct_abs_gt_1_upper", "mean"),
            target_min_offdiag_min=("target_min_offdiag", "min"),
            target_max_offdiag_max=("target_max_offdiag", "max"),
            target_mean_offdiag_mean=("target_mean_offdiag", "mean"),
            target_std_offdiag_mean=("target_std_offdiag", "mean"),
            recon_pct_abs_gt_0p95_offdiag_mean=("recon_pct_abs_gt_0p95_offdiag", "mean"),
            recon_pct_abs_gt_0p95_upper_mean=("recon_pct_abs_gt_0p95_upper", "mean"),
            recon_mse_all_mean=("recon_mse_all", "mean"),
            recon_mse_offdiag_mean=("recon_mse_offdiag", "mean"),
            recon_mse_upper_mean=("recon_mse_upper", "mean"),
            recon_mse_diag_mean=("recon_mse_diag", "mean"),
            kld_over_channel_recon_sse_mean=("kld_over_channel_recon_sse", "mean"),
            beta_kld_over_channel_recon_sse_mean=("beta_kld_over_channel_recon_sse", "mean"),
            active_units_mean=("active_units", "mean"),
            frac_active_units_mean=("frac_active_units", "mean"),
        )
        .reset_index()
    )
    latent_summary = (
        latent_df.groupby(["run_id", "run_label", "channels", "split"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_subjects_mean=("n_subjects", "mean"),
            active_units_mean=("active_units_var_mu_gt_1e_4", "mean"),
            frac_active_units_mean=("frac_active_units", "mean"),
            kld_mean=("kld_mean", "mean"),
            kld_per_latent_dim_mean=("kld_per_latent_dim", "mean"),
            total_recon_mse_per_pixel_mean=("total_recon_mse_per_pixel", "mean"),
            kld_over_total_recon_sse_mean=("kld_over_total_recon_sse", "mean"),
            beta_kld_over_total_recon_sse_mean=("beta_kld_over_total_recon_sse", "mean"),
        )
        .reset_index()
    )
    if not rd_df.empty:
        rd_summary = (
            rd_df.groupby(["run_id", "run_label", "channels"], dropna=False)
            .agg(
                n_epochs_total=("epoch", "count"),
                D_train_per_pixel_final_mean=("D_train_per_pixel", lambda s: s.groupby(rd_df.loc[s.index, "fold"]).tail(1).mean()),
                D_val_per_pixel_final_mean=("D_val_per_pixel", lambda s: s.groupby(rd_df.loc[s.index, "fold"]).tail(1).mean()),
                R_train_per_latent_dim_final_mean=("R_train_per_latent_dim", lambda s: s.groupby(rd_df.loc[s.index, "fold"]).tail(1).mean()),
                R_val_per_latent_dim_final_mean=("R_val_per_latent_dim", lambda s: s.groupby(rd_df.loc[s.index, "fold"]).tail(1).mean()),
                beta_max_R_val_over_D_val_final_mean=("beta_max_R_val_over_D_val", lambda s: s.groupby(rd_df.loc[s.index, "fold"]).tail(1).mean()),
            )
            .reset_index()
        )
    else:
        rd_summary = pd.DataFrame()

    channel_df.to_csv(output_dir / "target_reconstruction_by_fold_channel_split.csv", index=False)
    channel_summary.to_csv(output_dir / "target_reconstruction_summary_by_run_channel_split.csv", index=False)
    latent_df.to_csv(output_dir / "latent_kld_collapse_by_fold_split.csv", index=False)
    latent_summary.to_csv(output_dir / "latent_kld_collapse_summary_by_run_split.csv", index=False)
    rd_df.to_csv(output_dir / "rate_distortion_normalized.csv", index=False)
    rd_summary.to_csv(output_dir / "rate_distortion_normalized_summary.csv", index=False)
    availability_df.to_csv(output_dir / "fold_artifact_availability.csv", index=False)

    target_distribution = channel_df[
        [
            "run_id",
            "run_label",
            "channels",
            "fold",
            "split",
            "channel_index",
            "channel_name",
            "n_subjects",
            "target_min_all",
            "target_max_all",
            "target_mean_all",
            "target_std_all",
            "target_min_offdiag",
            "target_max_offdiag",
            "target_mean_offdiag",
            "target_std_offdiag",
            "target_min_upper",
            "target_max_upper",
            "target_mean_upper",
            "target_std_upper",
            "target_pct_abs_gt_1_all",
            "target_pct_abs_gt_1_offdiag",
            "target_pct_abs_gt_1_upper",
        ]
    ]
    reconstruction_saturation = channel_df[
        [
            "run_id",
            "run_label",
            "channels",
            "fold",
            "split",
            "channel_index",
            "channel_name",
            "recon_pct_abs_gt_0p95_all",
            "recon_pct_abs_gt_0p95_offdiag",
            "recon_pct_abs_gt_0p95_upper",
            "recon_mse_all",
            "recon_mse_offdiag",
            "recon_mse_upper",
            "recon_mse_diag",
            "recon_sse_per_subject_channel",
        ]
    ]
    loss_scale = latent_df[
        [
            "run_id",
            "run_label",
            "channels",
            "fold",
            "split",
            "n_subjects",
            "latent_dim",
            "kld_mean",
            "kld_per_latent_dim",
            "total_recon_sse_per_subject",
            "total_recon_mse_per_pixel",
            "kld_over_total_recon_sse",
            "beta_kld_over_total_recon_sse",
        ]
    ]
    latent_capacity = latent_df[
        [
            "run_id",
            "run_label",
            "channels",
            "fold",
            "split",
            "n_subjects",
            "latent_dim",
            "active_units_var_mu_gt_1e_4",
            "frac_active_units",
            "kld_mean",
            "kld_per_latent_dim",
            "low_kld_dims_lt_1e_3",
            "very_low_kld_dims_lt_1e_4",
            "mean_abs_mu",
            "mean_logvar",
            "mean_posterior_std",
            "posterior_collapse_indicator",
        ]
    ]
    target_distribution.to_csv(output_dir / "target_distribution_by_channel.csv", index=False)
    reconstruction_saturation.to_csv(output_dir / "reconstruction_saturation_by_fold.csv", index=False)
    loss_scale.to_csv(output_dir / "loss_scale_comparison.csv", index=False)
    latent_capacity.to_csv(output_dir / "latent_capacity_summary.csv", index=False)

    readme_channel = channel_summary[channel_summary["split"].eq("vae_pool")].copy()
    readme_channel = readme_channel[
        [
            "run_label",
            "channel_name",
            "target_pct_abs_gt_1_offdiag_mean",
            "recon_pct_abs_gt_0p95_offdiag_mean",
            "recon_mse_offdiag_mean",
            "recon_mse_upper_mean",
            "recon_mse_diag_mean",
        ]
    ]
    readme_latent = latent_summary[latent_summary["split"].eq("vae_pool")].copy()
    readme_latent = readme_latent[
        [
            "run_label",
            "active_units_mean",
            "kld_per_latent_dim_mean",
            "total_recon_mse_per_pixel_mean",
            "beta_kld_over_total_recon_sse_mean",
        ]
    ]
    write_readme(output_dir, readme_channel, readme_latent, args)
    write_recommended_next_experiments(output_dir)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "tensor_path": str(tensor_path),
        "runs": [{**r, "run_dir": str(resolve(r["run_dir"]))} for r in RUNS],
        "reconstruction_mode": "deterministic_decode_mu",
        "device": str(device),
        "batch_size": args.batch_size,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote audit outputs to: {output_dir}")
    print(readme_channel.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
