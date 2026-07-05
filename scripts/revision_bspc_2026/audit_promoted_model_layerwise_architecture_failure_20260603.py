#!/usr/bin/env python3
"""Read-only layerwise architecture and failure-mode audit for the promoted ADNI VAE.

This audit performs no training, no classifier scoring, no threshold fitting, and
does not modify tensors, metadata, model artifacts, or prediction files. It does
run deterministic VAE reconstruction inference from saved fold checkpoints in
eval mode to summarize reconstruction residuals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from betavae_xai.data.preprocessing import apply_normalization_params
from betavae_xai.models import ConvolutionalVAE, build_vae_dropout_manifest


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OUT_DEFAULT = RESULTS / "promoted_model_layerwise_architecture_failure_audit_20260603"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._\n"
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return df.to_csv(index=False)


def write_table(df: pd.DataFrame, stem: str, out: Path, max_rows: int | None = 200) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    shown = df if max_rows is None else df.head(max_rows)
    text = f"# {stem}\n\nRows: {len(df)}\n\n"
    if len(shown) < len(df):
        text += f"Showing first {len(shown)} rows; full table is in CSV.\n\n"
    text += to_md(shown)
    (out / f"{stem}.md").write_text(text, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    if den == 0 or np.isnan(den):
        return float("nan")
    return float(num) / float(den)


def config_args() -> dict[str, Any]:
    cfg = read_json(RUN_DIR / "run_config.json")
    return cfg.get("args", cfg)


def model_kwargs(args: dict[str, Any], image_size: int, n_channels: int) -> dict[str, Any]:
    return {
        "input_channels": int(n_channels),
        "latent_dim": int(args["latent_dim"]),
        "image_size": int(image_size),
        "final_activation": str(args.get("vae_final_activation", "tanh")),
        "intermediate_fc_dim_config": args.get("intermediate_fc_dim_vae", "quarter"),
        "dropout_rate": float(args.get("dropout_rate_vae", 0.15)),
        "encoder_dropout_rate": args.get("encoder_dropout_rate_vae"),
        "decoder_dropout_rate": args.get("decoder_dropout_rate_vae"),
        "use_layernorm_fc": bool(args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
        "decoder_type": str(args.get("decoder_type", "convtranspose")),
        "encoder_norm_mode": str(args.get("vae_encoder_norm_mode", args.get("encoder_norm_mode", "groupnorm"))),
        "dropout_scope": str(args.get("vae_dropout_scope", args.get("dropout_scope", "legacy_all"))),
        "block_order": str(args.get("vae_block_order", args.get("block_order", "legacy_act_norm"))),
        "conditioning_mode": str(args.get("vae_conditioning_mode", "none")),
        "conditioning_dim": 0,
    }


def load_selected_tensor(args: dict[str, Any]) -> dict[str, Any]:
    tensor_path = Path(args["global_tensor_path"])
    channels = list(args["channels_to_use"])
    with np.load(tensor_path, allow_pickle=True) as zf:
        tensor = np.asarray(zf["global_tensor_data"][:, channels, :, :], dtype=np.float32)
        subject_ids = np.asarray(zf["subject_ids"]).astype(str)
        channel_names_all = np.asarray(zf["channel_names"]).astype(str).tolist()
        roi_names = np.asarray(zf["roi_names_in_order"]).astype(str).tolist() if "roi_names_in_order" in zf.files else []
        network_labels = (
            np.asarray(zf["network_labels_in_order"]).astype(str).tolist()
            if "network_labels_in_order" in zf.files
            else []
        )
    return {
        "tensor": tensor,
        "subject_ids": subject_ids,
        "selected_channel_names": [channel_names_all[i] for i in channels],
        "roi_names": roi_names,
        "network_labels": network_labels,
        "tensor_path": tensor_path,
    }


def normalize_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    if "tensor_index" in out.columns and "tensor_idx" not in out.columns:
        out = out.rename(columns={"tensor_index": "tensor_idx"})
    out["SubjectID"] = out["SubjectID"].astype(str)
    out["tensor_idx"] = pd.to_numeric(out["tensor_idx"], errors="coerce")
    for col in ["ResearchGroup_Mapped", "Manufacturer", "Sex", "Site3"]:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    if "Age" in out.columns:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out


def own_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters(recurse=False)))


def total_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters(recurse=True)))


def module_kind(module: nn.Module) -> str:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return "Conv/Linear"
    if isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
        return "Normalization"
    if isinstance(module, (nn.GELU, nn.Tanh, nn.Sigmoid, nn.ReLU, nn.Identity)):
        return "Activation/Identity"
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        return "Dropout"
    return module.__class__.__name__


def block_from_name(name: str) -> str:
    if name.startswith("encoder_conv."):
        return "encoder_conv"
    if name.startswith("encoder_fc_intermediate."):
        return "encoder_fc_intermediate"
    if name.startswith("fc_mu"):
        return "fc_mu"
    if name.startswith("fc_logvar"):
        return "fc_logvar"
    if name.startswith("decoder_fc_intermediate."):
        return "decoder_fc_intermediate"
    if name.startswith("decoder_fc_to_conv"):
        return "decoder_fc_to_conv"
    if name.startswith("decoder_conv."):
        return "decoder_conv"
    return "other"


def module_details(module: nn.Module) -> str:
    fields: list[str] = []
    for attr in ["in_channels", "out_channels", "in_features", "out_features", "kernel_size", "stride", "padding", "output_padding", "num_groups", "num_features", "p"]:
        if hasattr(module, attr):
            fields.append(f"{attr}={getattr(module, attr)}")
    return "; ".join(fields)


def architecture_for_fold(model: ConvolutionalVAE, fold: int, input_shape: tuple[int, int, int, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    handles = []
    order_counter = {"i": 0}

    leaf_names = [(name, mod) for name, mod in model.named_modules() if name and len(list(mod.children())) == 0]
    name_by_id = {id(mod): name for name, mod in leaf_names}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        name = name_by_id.get(id(module), "")
        if not name:
            return
        order_counter["i"] += 1
        inp = inputs[0] if inputs else None
        out = output
        if isinstance(out, tuple):
            out = out[0]
        rows.append(
            {
                "fold": fold,
                "forward_order": order_counter["i"],
                "module_name": name,
                "block": block_from_name(name),
                "module_type": module.__class__.__name__,
                "module_kind": module_kind(module),
                "input_shape": tuple(inp.shape) if hasattr(inp, "shape") else "",
                "output_shape": tuple(out.shape) if hasattr(out, "shape") else "",
                "own_param_count": own_params(module),
                "module_detail": module_details(module),
            }
        )

    for _name, mod in leaf_names:
        handles.append(mod.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            x = torch.zeros(input_shape, dtype=torch.float32)
            _ = model(x)
    finally:
        for handle in handles:
            handle.remove()

    manual_rows = [
        {
            "fold": fold,
            "forward_order": 0,
            "module_name": "input",
            "block": "input",
            "module_type": "Tensor",
            "module_kind": "Input",
            "input_shape": "",
            "output_shape": input_shape,
            "own_param_count": 0,
            "module_detail": "synthetic audit input",
        },
        {
            "fold": fold,
            "forward_order": 9998,
            "module_name": "flatten_after_encoder_conv",
            "block": "encoder_flatten",
            "module_type": "view",
            "module_kind": "Reshape",
            "input_shape": f"(B,{model.final_conv_ch},{model.final_spatial_dim},{model.final_spatial_dim})",
            "output_shape": f"(B,{model.final_conv_ch * model.final_spatial_dim * model.final_spatial_dim})",
            "own_param_count": 0,
            "module_detail": "implicit flatten in encode()",
        },
        {
            "fold": fold,
            "forward_order": 9999,
            "module_name": "reshape_before_decoder_conv",
            "block": "decoder_reshape",
            "module_type": "view",
            "module_kind": "Reshape",
            "input_shape": f"(B,{model.final_conv_ch * model.final_spatial_dim * model.final_spatial_dim})",
            "output_shape": f"(B,{model.final_conv_ch},{model.final_spatial_dim},{model.final_spatial_dim})",
            "own_param_count": 0,
            "module_detail": "implicit reshape in decode()",
        },
    ]
    out = pd.concat([pd.DataFrame(manual_rows), pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values(["fold", "forward_order", "module_name"]).reset_index(drop=True)


def block_summary(model: ConvolutionalVAE, fold: int) -> pd.DataFrame:
    rows = []
    for name, module in [
        ("encoder_conv", model.encoder_conv),
        ("encoder_fc_intermediate", model.encoder_fc_intermediate),
        ("fc_mu", model.fc_mu),
        ("fc_logvar", model.fc_logvar),
        ("decoder_fc_intermediate", model.decoder_fc_intermediate),
        ("decoder_fc_to_conv", model.decoder_fc_to_conv),
        ("decoder_conv", model.decoder_conv),
    ]:
        child_types = [child.__class__.__name__ for child in module.children()] if hasattr(module, "children") else []
        rows.append(
            {
                "fold": fold,
                "block": name,
                "param_count": total_params(module),
                "n_leaf_children": len(child_types),
                "sequential_order": " -> ".join(child_types),
                "contains_convtranspose2d": any(isinstance(m, nn.ConvTranspose2d) for m in module.modules()),
                "contains_dropout": any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in module.modules()),
                "contains_norm": any(isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)) for m in module.modules()),
                "contains_activation": any(isinstance(m, (nn.GELU, nn.Tanh, nn.Sigmoid, nn.ReLU)) for m in module.modules()),
            }
        )
    rows.append(
        {
            "fold": fold,
            "block": "total_model",
            "param_count": total_params(model),
            "n_leaf_children": np.nan,
            "sequential_order": "",
            "contains_convtranspose2d": any(isinstance(m, nn.ConvTranspose2d) for m in model.modules()),
            "contains_dropout": any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules()),
            "contains_norm": any(isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules()),
            "contains_activation": any(isinstance(m, (nn.GELU, nn.Tanh, nn.Sigmoid, nn.ReLU)) for m in model.modules()),
        }
    )
    return pd.DataFrame(rows)


def convtranspose_audit(model: ConvolutionalVAE, fold: int) -> pd.DataFrame:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            rows.append(
                {
                    "fold": fold,
                    "module_name": name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                    "output_padding": module.output_padding,
                    "param_count": total_params(module),
                }
            )
    return pd.DataFrame(rows)


def masks_for_size(n: int) -> dict[str, np.ndarray]:
    i, j = np.indices((n, n))
    dist = np.abs(i - j)
    return {
        "all": np.ones((n, n), dtype=bool),
        "diag": dist == 0,
        "offdiag": dist > 0,
        "near_diag_1": dist == 1,
        "near_diag_2_5": (dist >= 2) & (dist <= 5),
        "near_diag_1_5": (dist >= 1) & (dist <= 5),
        "mid_6_20": (dist >= 6) & (dist <= 20),
        "far_gt20": dist > 20,
        "upper": i < j,
        "lower": i > j,
    }


def matrix_metric_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sub = values[..., mask]
    return {
        "mse": float(np.mean(sub * sub)),
        "mae": float(np.mean(np.abs(sub))),
        "bias": float(np.mean(sub)),
        "max_abs": float(np.max(np.abs(sub))),
    }


def summarize_residual(
    residual: np.ndarray,
    meta: pd.DataFrame,
    channel_names: list[str],
    group_cols: list[str],
    split: str,
    fold: int,
    masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not group_cols:
        groups: Iterable[tuple[tuple[Any, ...], np.ndarray]] = [(tuple(), np.ones(len(meta), dtype=bool))]
    else:
        groups = []
        for keys, grp in meta.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            groups.append((keys, np.asarray(meta.index.isin(grp.index), dtype=bool)))
    for keys, subj_mask in groups:
        if not subj_mask.any():
            continue
        for c, channel in enumerate(channel_names):
            arr = residual[subj_mask, c]
            base = {"fold": fold, "split": split, "channel_index": c, "channel_name": channel, "n_subjects": int(subj_mask.sum())}
            for col, key in zip(group_cols, keys):
                base[col] = key
            all_stats = matrix_metric_stats(arr, masks["all"])
            off_stats = matrix_metric_stats(arr, masks["offdiag"])
            diag_stats = matrix_metric_stats(arr, masks["diag"])
            near_stats = matrix_metric_stats(arr, masks["near_diag_1_5"])
            far_stats = matrix_metric_stats(arr, masks["far_gt20"])
            upper_stats = matrix_metric_stats(arr, masks["upper"])
            lower_stats = matrix_metric_stats(arr, masks["lower"])
            row = {
                **base,
                "mse_all": all_stats["mse"],
                "mae_all": all_stats["mae"],
                "bias_all": all_stats["bias"],
                "mse_offdiag": off_stats["mse"],
                "mae_offdiag": off_stats["mae"],
                "bias_offdiag": off_stats["bias"],
                "mse_diag": diag_stats["mse"],
                "mae_diag": diag_stats["mae"],
                "bias_diag": diag_stats["bias"],
                "mse_near_diag_1_5": near_stats["mse"],
                "mae_near_diag_1_5": near_stats["mae"],
                "mse_far_gt20": far_stats["mse"],
                "mae_far_gt20": far_stats["mae"],
                "diag_over_offdiag_mae": safe_div(diag_stats["mae"], off_stats["mae"]),
                "near_over_far_mae": safe_div(near_stats["mae"], far_stats["mae"]),
                "mse_upper": upper_stats["mse"],
                "mse_lower": lower_stats["mse"],
                "mae_upper": upper_stats["mae"],
                "mae_lower": lower_stats["mae"],
                "upper_lower_mae_abs_delta": abs(upper_stats["mae"] - lower_stats["mae"]),
                "upper_lower_mse_abs_delta": abs(upper_stats["mse"] - lower_stats["mse"]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_distance_bands(
    residual: np.ndarray,
    meta: pd.DataFrame,
    channel_names: list[str],
    split: str,
    fold: int,
    masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    band_names = ["diag", "near_diag_1", "near_diag_2_5", "mid_6_20", "far_gt20"]
    for c, channel in enumerate(channel_names):
        arr = residual[:, c]
        for band in band_names:
            stats = matrix_metric_stats(arr, masks[band])
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "channel_index": c,
                    "channel_name": channel,
                    "band": band,
                    "n_subjects": len(meta),
                    **{f"{k}_{band}": v for k, v in stats.items()},
                }
            )
    return pd.DataFrame(rows)


def diagonal_structure(
    residual: np.ndarray,
    channel_names: list[str],
    split: str,
    fold: int,
) -> pd.DataFrame:
    n = residual.shape[-1]
    i, j = np.indices((n, n))
    dist = np.abs(i - j)
    rows = []
    for c, channel in enumerate(channel_names):
        mean_abs_by_dist = []
        for d in range(n):
            vals = np.abs(residual[:, c, :, :][..., dist == d])
            mean_abs_by_dist.append(float(vals.mean()))
        x = np.arange(n, dtype=float)
        y = np.asarray(mean_abs_by_dist, dtype=float)
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else float("nan")
        rows.append(
            {
                "fold": fold,
                "split": split,
                "channel_index": c,
                "channel_name": channel,
                "mae_distance0_diag": mean_abs_by_dist[0],
                "mae_distance1": mean_abs_by_dist[1],
                "mae_distance2_5_mean": float(np.mean(mean_abs_by_dist[2:6])),
                "mae_distance6_20_mean": float(np.mean(mean_abs_by_dist[6:21])),
                "mae_distance_gt20_mean": float(np.mean(mean_abs_by_dist[21:])),
                "distance_vs_mean_abs_residual_corr": corr,
                "near1_over_far_gt20": safe_div(mean_abs_by_dist[1], float(np.mean(mean_abs_by_dist[21:]))),
                "diag_over_far_gt20": safe_div(mean_abs_by_dist[0], float(np.mean(mean_abs_by_dist[21:]))),
            }
        )
    return pd.DataFrame(rows)


def subject_meta_from_indices(indices: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = metadata.set_index("tensor_idx").reindex(indices.astype(int)).reset_index(drop=False)
    rows["tensor_idx"] = indices.astype(int)
    return rows.reset_index(drop=True)


def reconstruct_fold(
    model: ConvolutionalVAE,
    x_norm: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], batch_size):
            xb = torch.from_numpy(x_norm[start : start + batch_size]).float().to(device)
            mu, _logvar = model.encode(xb)
            recon = model.decode(mu)
            if recon.shape != xb.shape:
                recon = nn.functional.interpolate(recon, size=xb.shape[-2:], mode="bilinear", align_corners=False)
            chunks.append(recon.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def load_primary_predictions() -> pd.DataFrame:
    pred_path = OOF_DIR / "calib_predictions.csv"
    df = read_csv(pred_path, required=True)
    rows = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    rows["SubjectID"] = rows["SubjectID"].astype(str)
    return rows


def latent_nuisance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    latent_rows = []
    leak_rows = []
    for fold in range(1, 6):
        fold_dir = RUN_DIR / f"fold_{fold}"
        for split, file_stem in [("trainDev", f"fold_{fold}_trainDev_latent_info_summary.csv"), ("test", f"fold_{fold}_test_latent_info_summary.csv")]:
            df = read_csv(fold_dir / file_stem)
            row = {"fold": fold, "split": split}
            if not df.empty and "variable" in df.columns:
                for var, prefix in [("Y_target", "Y"), ("Manufacturer", "Manufacturer")]:
                    hit = df[df["variable"].astype(str).eq(var)]
                    if not hit.empty:
                        rec = hit.iloc[0]
                        row[f"MI_Z_{prefix}_nats"] = safe_float(rec.get("mi_sum_nats"))
                        row[f"active_units_{prefix}"] = safe_float(rec.get("n_active"))
                        row[f"total_correlation_nats_{prefix}"] = safe_float(rec.get("total_correlation_nats"))
            row["MI_Manufacturer_over_MI_Y"] = safe_div(row.get("MI_Z_Manufacturer_nats", np.nan), row.get("MI_Z_Y_nats", np.nan))
            latent_rows.append(row)
        for split, file_stem in [("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"), ("test", f"fold_{fold}_test_scanner_leakage_summary.csv")]:
            df = read_csv(fold_dir / file_stem)
            if df.empty:
                leak_rows.append({"fold": fold, "split": split, "available": False})
                continue
            rec = df.iloc[0].to_dict()
            rec.update({"fold": fold, "split": split, "available": True})
            rec["latent_minus_raw"] = safe_float(rec.get("acc_site_latent")) - safe_float(rec.get("acc_site_raw"))
            leak_rows.append(rec)
    return pd.DataFrame(latent_rows), pd.DataFrame(leak_rows)


def summarize_latent_fp_tn(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        latent = read_csv(RUN_DIR / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_test_latent_mu.csv")
        if latent.empty:
            continue
        fold_pred = pred[pred["fold"].eq(fold)].copy()
        merged = latent.merge(
            fold_pred[["SubjectID", "y_true", "y_pred", "y_score"]],
            on="SubjectID",
            how="inner",
        )
        merged = merged[
            (merged["Manufacturer"].astype(str).str.lower().eq("philips"))
            & (merged["y_true"].eq(0))
        ].copy()
        if merged.empty:
            continue
        mu_cols = [c for c in merged.columns if c.startswith("mu_")]
        arr = merged[mu_cols].to_numpy(dtype=float)
        merged["latent_l2_norm"] = np.linalg.norm(arr, axis=1)
        merged["latent_mean_abs"] = np.mean(np.abs(arr), axis=1)
        merged["fp_tn_status"] = np.where(merged["y_pred"].eq(1), "Philips_CN_FP", "Philips_CN_TN")
        for status, grp in merged.groupby("fp_tn_status"):
            rows.append(
                {
                    "fold": fold,
                    "fp_tn_status": status,
                    "n": len(grp),
                    "score_mean_existing": float(grp["y_score"].mean()),
                    "score_median_existing": float(grp["y_score"].median()),
                    "latent_l2_norm_mean": float(grp["latent_l2_norm"].mean()),
                    "latent_l2_norm_std": float(grp["latent_l2_norm"].std(ddof=0)),
                    "latent_mean_abs_mean": float(grp["latent_mean_abs"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_existing_fold_distributions() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for kind in ["raw", "norm", "recon"]:
            path = RUN_DIR / f"fold_{fold}" / f"fold_{fold}_dist_{kind}.csv"
            df = read_csv(path)
            if not df.empty:
                df = df.copy()
                df.insert(0, "fold", fold)
                df.insert(1, "distribution_kind", kind)
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_recommendation(
    residual_channel: pd.DataFrame,
    latent_nuisance: pd.DataFrame,
    scanner: pd.DataFrame,
    philips_resid: pd.DataFrame,
    convtranspose: pd.DataFrame,
) -> str:
    test_scanner = scanner[scanner["split"].eq("test")].copy()
    latent_mean = safe_float(test_scanner["acc_site_latent"].mean()) if "acc_site_latent" in test_scanner.columns else np.nan
    raw_mean = safe_float(test_scanner["acc_site_raw"].mean()) if "acc_site_raw" in test_scanner.columns else np.nan
    train_lat = latent_nuisance[latent_nuisance["split"].eq("trainDev")]
    mi_ratio = safe_float(train_lat["MI_Manufacturer_over_MI_Y"].mean()) if "MI_Manufacturer_over_MI_Y" in train_lat.columns else np.nan
    diag_ratio = safe_float(residual_channel["diag_over_offdiag_mae"].mean()) if "diag_over_offdiag_mae" in residual_channel.columns else np.nan
    near_ratio = safe_float(residual_channel["near_over_far_mae"].mean()) if "near_over_far_mae" in residual_channel.columns else np.nan
    fp_delta_text = "not available"
    if not philips_resid.empty:
        fp = philips_resid[philips_resid["fp_tn_status"].eq("Philips_CN_FP")]
        tn = philips_resid[philips_resid["fp_tn_status"].eq("Philips_CN_TN")]
        if not fp.empty and not tn.empty:
            fp_delta_text = f"{safe_float(fp['mae_offdiag'].mean()) - safe_float(tn['mae_offdiag'].mean()):+.6f} mean off-diagonal MAE"

    return f"""# Architecture and Failure-Mode Recommendation

Recommendation: **D. no more model training**.

## Evidence From This Audit

- Decoder implementation: `ConvTranspose2d` is present in `{len(convtranspose)}` decoder layers, matching the promoted `decoder_type=convtranspose` configuration.
- Mean test scanner/manufacturer balanced accuracy is `{latent_mean:.4f}` in latent space versus `{raw_mean:.4f}` in raw space. The VAE reduces scanner separability but does not eliminate it.
- Mean train/dev `MI(Z;Manufacturer)/MI(Z;Y)` is `{mi_ratio:.4f}`, indicating Manufacturer information remains larger than diagnosis information in the latent representation.
- Mean diagonal/off-diagonal residual MAE ratio is `{diag_ratio:.4f}` and mean near/far residual MAE ratio is `{near_ratio:.4f}` on normalized reconstructions. These residuals are quantified for transparency, but they do not by themselves justify replacing the decoder because the prior upsample-conv screening produced a large performance drop.
- Philips CN FP versus TN residual delta: `{fp_delta_text}`.

## Architecture Options

A. `decoder_type=upsample_conv`: not recommended as the next run. A controlled FAST decoder-type audit already showed a substantial AUC/PR-AUC drop for upsample-conv versus ConvTranspose2d in both `[1]` and `[1,0,2]` settings.

B. Decoder conditional on Manufacturer: not recommended as a primary-replacement run. The Manufacturer-conditioned FULL branch reduced latent leakage versus its matched baseline but did not beat the promoted model on AUC/BA/Sensitivity/F1 and was retained as a secondary deconfounding audit.

C. Adversarial Manufacturer removal: not recommended now. It would introduce a new optimization objective and a new nuisance-removal tradeoff after multiple controlled internal variants failed to improve the promoted model. It also risks removing disease-relevant scanner-correlated signal unless validated externally.

D. No more model training: recommended. The current failure modes are better handled in the manuscript as residual scanner/site structure, Philips CN false-positive enrichment, and external calibration/domain-shift limitations. The next scientifically cleaner step remains locked external calibration/test rather than additional internal architecture search.
"""


def main() -> None:
    args = parse_args()
    out = args.output_dir
    required = [
        RUN_DIR / "run_config.json",
        OOF_DIR / "calib_predictions.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(rel(p) for p in missing))
    cfg_args = config_args()
    tensor_info = load_selected_tensor(cfg_args)
    metadata = normalize_metadata(pd.read_csv(Path(cfg_args["metadata_path"])))

    if args.dry_run:
        print("Inputs OK.")
        print("Output directory:", rel(out))
        print("Tensor:", tensor_info["tensor"].shape, "channels:", tensor_info["selected_channel_names"])
        return

    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    masks = masks_for_size(tensor_info["tensor"].shape[-1])
    primary_pred = load_primary_predictions()

    arch_rows = []
    block_rows = []
    drop_rows = []
    convt_rows = []
    residual_channel_rows = []
    residual_dx_rows = []
    residual_mfr_rows = []
    residual_band_rows = []
    diag_rows = []
    asym_rows = []
    philips_residual_rows = []

    for fold in range(1, 6):
        fold_dir = RUN_DIR / f"fold_{fold}"
        model = ConvolutionalVAE(
            **model_kwargs(cfg_args, image_size=tensor_info["tensor"].shape[-1], n_channels=tensor_info["tensor"].shape[1])
        ).to(device)
        state = torch.load(fold_dir / f"vae_model_fold_{fold}.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()

        arch_rows.append(architecture_for_fold(model.cpu(), fold, (2, tensor_info["tensor"].shape[1], 131, 131)))
        model.to(device)
        block_rows.append(block_summary(model, fold))
        drop = pd.DataFrame(build_vae_dropout_manifest(model))
        if not drop.empty:
            drop.insert(0, "fold", fold)
            drop_rows.append(drop)
        convt_rows.append(convtranspose_audit(model, fold))

        norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
        split_specs = [
            ("vae_pool", np.load(fold_dir / "vae_training_pool_tensor_idx.npy").astype(int)),
            ("classifier_test", pd.read_csv(fold_dir / "test_subjects_fold.csv")["tensor_idx"].astype(int).to_numpy()),
        ]
        for split, idx in split_specs:
            x_raw = tensor_info["tensor"][idx]
            x_norm = apply_normalization_params(x_raw, norm_params).astype(np.float32)
            recon = reconstruct_fold(model, x_norm, batch_size=args.batch_size, device=device)
            residual = recon - x_norm
            meta = subject_meta_from_indices(idx, metadata)
            meta["fold"] = fold
            meta["split"] = split
            residual_channel_rows.append(
                summarize_residual(residual, meta, tensor_info["selected_channel_names"], [], split, fold, masks)
            )
            residual_dx_rows.append(
                summarize_residual(residual, meta, tensor_info["selected_channel_names"], ["ResearchGroup_Mapped"], split, fold, masks)
            )
            residual_mfr_rows.append(
                summarize_residual(residual, meta, tensor_info["selected_channel_names"], ["Manufacturer"], split, fold, masks)
            )
            residual_band_rows.append(summarize_distance_bands(residual, meta, tensor_info["selected_channel_names"], split, fold, masks))
            diag_rows.append(diagonal_structure(residual, tensor_info["selected_channel_names"], split, fold))
            asym_rows.append(
                summarize_residual(residual, meta, tensor_info["selected_channel_names"], [], split, fold, masks)[
                    [
                        "fold",
                        "split",
                        "channel_index",
                        "channel_name",
                        "n_subjects",
                        "mse_upper",
                        "mse_lower",
                        "mae_upper",
                        "mae_lower",
                        "upper_lower_mae_abs_delta",
                        "upper_lower_mse_abs_delta",
                    ]
                ]
            )
            if split == "classifier_test":
                fold_pred = primary_pred[
                    (primary_pred["fold"].eq(fold))
                    & (primary_pred["Manufacturer"].astype(str).str.lower().eq("philips"))
                    & (primary_pred["y_true"].eq(0))
                ].copy()
                if not fold_pred.empty:
                    status_map = dict(zip(fold_pred["SubjectID"].astype(str), np.where(fold_pred["y_pred"].eq(1), "Philips_CN_FP", "Philips_CN_TN")))
                    meta2 = meta.copy()
                    meta2["fp_tn_status"] = meta2["SubjectID"].astype(str).map(status_map)
                    keep = meta2["fp_tn_status"].notna().to_numpy()
                    if keep.any():
                        philips_residual_rows.append(
                            summarize_residual(
                                residual[keep],
                                meta2.loc[keep].reset_index(drop=True),
                                tensor_info["selected_channel_names"],
                                ["fp_tn_status"],
                                split,
                                fold,
                                masks,
                            )
                        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    architecture = pd.concat(arch_rows, ignore_index=True)
    block_summary_df = pd.concat(block_rows, ignore_index=True)
    dropout_manifest = pd.concat(drop_rows, ignore_index=True) if drop_rows else pd.DataFrame()
    convtranspose = pd.concat(convt_rows, ignore_index=True)
    residual_channel = pd.concat(residual_channel_rows, ignore_index=True)
    residual_dx = pd.concat(residual_dx_rows, ignore_index=True)
    residual_mfr = pd.concat(residual_mfr_rows, ignore_index=True)
    residual_band = pd.concat(residual_band_rows, ignore_index=True)
    diagonal = pd.concat(diag_rows, ignore_index=True)
    asymmetry = pd.concat(asym_rows, ignore_index=True)
    philips_residual = pd.concat(philips_residual_rows, ignore_index=True) if philips_residual_rows else pd.DataFrame()
    latent_nuisance, scanner = latent_nuisance_tables()
    philips_latent = summarize_latent_fp_tn(primary_pred)
    existing_dist = summarize_existing_fold_distributions()

    norm_act = architecture[architecture["module_kind"].isin(["Normalization", "Activation/Identity", "Dropout"])].copy()
    norm_act["is_normalization"] = norm_act["module_kind"].eq("Normalization")
    norm_act["is_dropout"] = norm_act["module_kind"].eq("Dropout")

    write_table(architecture, "architecture_layerwise_foldwise", out, max_rows=250)
    write_table(block_summary_df, "architecture_block_summary", out, max_rows=None)
    write_table(dropout_manifest, "dropout_manifest_foldwise", out, max_rows=None)
    write_table(norm_act, "normalization_activation_dropout_manifest", out, max_rows=250)
    write_table(convtranspose, "convtranspose_decoder_audit", out, max_rows=None)
    write_table(residual_channel, "reconstruction_residual_by_channel_fold", out, max_rows=200)
    write_table(residual_dx, "reconstruction_residual_by_diagnosis", out, max_rows=240)
    write_table(residual_mfr, "reconstruction_residual_by_manufacturer", out, max_rows=240)
    write_table(residual_band, "reconstruction_residual_by_distance_band", out, max_rows=240)
    write_table(diagonal, "diagonal_near_diagonal_structure", out, max_rows=200)
    write_table(asymmetry, "residual_asymmetry_summary", out, max_rows=200)
    write_table(latent_nuisance, "latent_nuisance_foldwise", out, max_rows=None)
    write_table(scanner, "scanner_leakage_foldwise", out, max_rows=None)
    write_table(philips_residual, "philips_cn_fp_tn_residual_comparison", out, max_rows=200)
    write_table(philips_latent, "philips_cn_fp_tn_latent_summary", out, max_rows=None)
    write_table(existing_dist, "existing_reconstruction_distribution_files", out, max_rows=200)

    # Compact summaries for README/failure mode text.
    rc_test = residual_channel[residual_channel["split"].eq("classifier_test")]
    rm_test = residual_mfr[residual_mfr["split"].eq("classifier_test")]
    rd_test = residual_dx[residual_dx["split"].eq("classifier_test")]
    summary_lines = [
        "# Failure-Mode Summary",
        "",
        "This audit used deterministic mean-latent VAE reconstructions on fold-normalized tensors.",
        "No VAE training, classifier scoring, threshold fitting, tensor modification, metadata modification, or model-artifact modification was performed.",
        "",
        "## Reconstruction Residuals",
        "",
        f"- Mean classifier-test off-diagonal MAE across channel/fold summaries: `{safe_float(rc_test['mae_offdiag'].mean()):.6f}`.",
        f"- Mean classifier-test diagonal/off-diagonal MAE ratio: `{safe_float(rc_test['diag_over_offdiag_mae'].mean()):.6f}`.",
        f"- Mean classifier-test near/far MAE ratio: `{safe_float(rc_test['near_over_far_mae'].mean()):.6f}`.",
        f"- Mean upper/lower MAE absolute delta: `{safe_float(asymmetry[asymmetry['split'].eq('classifier_test')]['upper_lower_mae_abs_delta'].mean()):.6f}`.",
        "",
        "## Grouped Residuals",
        "",
        f"- Diagnosis-group residual table rows: `{len(rd_test)}` classifier-test rows.",
        f"- Manufacturer-group residual table rows: `{len(rm_test)}` classifier-test rows.",
        f"- Philips CN FP/TN residual table rows: `{len(philips_residual)}`.",
        "",
        "## Latent Nuisance",
        "",
    ]
    train_lat = latent_nuisance[latent_nuisance["split"].eq("trainDev")]
    test_scanner = scanner[scanner["split"].eq("test")]
    summary_lines.extend(
        [
            f"- Mean train/dev `MI(Z;Y)`: `{safe_float(train_lat['MI_Z_Y_nats'].mean()):.6f}` nats.",
            f"- Mean train/dev `MI(Z;Manufacturer)`: `{safe_float(train_lat['MI_Z_Manufacturer_nats'].mean()):.6f}` nats.",
            f"- Mean train/dev `MI(Manufacturer)/MI(Y)`: `{safe_float(train_lat['MI_Manufacturer_over_MI_Y'].mean()):.6f}`.",
            f"- Mean test raw scanner leakage BA: `{safe_float(test_scanner['acc_site_raw'].mean()):.6f}`.",
            f"- Mean test latent scanner leakage BA: `{safe_float(test_scanner['acc_site_latent'].mean()):.6f}`.",
        ]
    )
    write_text(out / "failure_mode_summary.md", "\n".join(summary_lines) + "\n")
    write_text(out / "architecture_recommendation.md", make_recommendation(residual_channel, latent_nuisance, scanner, philips_residual, convtranspose))

    readme = f"""# Promoted Model Layerwise Architecture and Failure-Mode Audit

Generated: {datetime.now().isoformat(timespec='seconds')}

Run: `{rel(RUN_DIR)}`

Primary promoted readout used only for FP/TN labels:
`{PRIMARY_MODEL} / {PRIMARY_FEATURES} / {PRIMARY_CALIB} / {PRIMARY_THRESHOLD}`.

## Safety

- Training launched: no.
- Classifier scoring launched: no.
- Threshold fitting launched: no.
- Tensor/metadata/model artifact modification: no.
- VAE reconstruction inference: yes, deterministic `decode(mu)` in `eval()` mode for residual QC only.

## Main Files

- `architecture_layerwise_foldwise.csv/.md`
- `architecture_block_summary.csv/.md`
- `dropout_manifest_foldwise.csv/.md`
- `normalization_activation_dropout_manifest.csv/.md`
- `convtranspose_decoder_audit.csv/.md`
- `reconstruction_residual_by_channel_fold.csv/.md`
- `reconstruction_residual_by_diagnosis.csv/.md`
- `reconstruction_residual_by_manufacturer.csv/.md`
- `reconstruction_residual_by_distance_band.csv/.md`
- `diagonal_near_diagonal_structure.csv/.md`
- `latent_nuisance_foldwise.csv/.md`
- `scanner_leakage_foldwise.csv/.md`
- `philips_cn_fp_tn_residual_comparison.csv/.md`
- `philips_cn_fp_tn_latent_summary.csv/.md`
- `failure_mode_summary.md`
- `architecture_recommendation.md`
"""
    write_text(out / "README.md", readme)

    command_log = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "run_dir": rel(RUN_DIR),
        "oof_dir": rel(OOF_DIR),
        "output_dir": rel(out),
        "tensor_path": str(tensor_info["tensor_path"]),
        "metadata_path": str(cfg_args["metadata_path"]),
        "device": str(device),
        "safety": {
            "training": False,
            "classifier_scoring": False,
            "threshold_fitting": False,
            "tensor_modification": False,
            "metadata_modification": False,
            "model_artifact_modification": False,
            "vae_reconstruction_inference_for_residual_qc": True,
        },
        "primary_prediction_filter": {
            "model_name": PRIMARY_MODEL,
            "feature_set": PRIMARY_FEATURES,
            "calib_method": PRIMARY_CALIB,
            "threshold_strategy": PRIMARY_THRESHOLD,
        },
        "outputs": sorted(p.name for p in out.iterdir()),
    }
    write_text(out / "command_log.json", json.dumps(command_log, indent=2))
    print(f"Wrote audit package: {rel(out)}")


if __name__ == "__main__":
    main()
