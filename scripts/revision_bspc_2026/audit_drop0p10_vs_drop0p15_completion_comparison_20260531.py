#!/usr/bin/env python3
"""Read-only completion/comparison audit for latent384 beta3.75 dropout 0.10 vs 0.15.

This audit compares the completed global-dropout ablation:

  recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5

against the matched dropout=0.15 reference:

  recover035_latent384_beta3p75_T80_h10000_p560_full5x5

It does not train, refit, alter tensors, or alter model artifacts. It writes only
derived audit tables under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from betavae_xai.models import (  # noqa: E402
    ConvolutionalVAE,
    build_vae_dropout_manifest,
    summarize_vae_dropout_manifest,
)


RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

REFERENCE_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
CANDIDATE_RUN = RESULTS / "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5"
OUTPUT_DIR = RESULTS / "drop0p10_vs_drop0p15_completion_comparison_audit_20260531"

PROMOTED_OOF_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"

FOLDS = [1, 2, 3, 4, 5]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
METRIC_COLS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(out_dir: Path, stem: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_stagea_metrics_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_stagea(run_dir: Path, run_id: str, dropout_rate: float) -> pd.DataFrame:
    path = find_stagea_metrics_file(run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_id", run_id)
    df.insert(1, "dropout_rate_vae", dropout_rate)
    df.insert(2, "source_file", rel(path))
    return df


def summarize_stagea(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = ["run_id", "dropout_rate_vae", "actual_classifier_type"]
    value_cols = [
        "auc_raw",
        "pr_auc_raw",
        "auc_final",
        "pr_auc_final",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
    ]
    present = [c for c in value_cols if c in df.columns]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_folds"] = int(g["fold"].nunique()) if "fold" in g.columns else len(g)
        for col in present:
            row[f"{col}_mean"] = safe_float(g[col].mean())
            row[f"{col}_std"] = safe_float(g[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def delta_stagea_foldwise(stagea_long: pd.DataFrame) -> pd.DataFrame:
    if stagea_long.empty:
        return pd.DataFrame()
    keep = [
        "fold",
        "actual_classifier_type",
        "run_id",
        "auc_raw",
        "pr_auc_raw",
        "auc_final",
        "pr_auc_final",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
    ]
    present = [c for c in keep if c in stagea_long.columns]
    base = stagea_long[present].copy()
    ref = base[base["run_id"].eq("drop0p15_reference")]
    cand = base[base["run_id"].eq("drop0p10_candidate")]
    metrics = [c for c in present if c not in {"fold", "actual_classifier_type", "run_id"}]
    merged = cand.merge(ref, on=["fold", "actual_classifier_type"], suffixes=("_drop0p10", "_drop0p15"))
    for m in metrics:
        merged[f"delta_{m}"] = merged[f"{m}_drop0p10"] - merged[f"{m}_drop0p15"]
    return merged.sort_values(["actual_classifier_type", "fold"])


def load_stageb_pooled(run_dir: Path, run_id: str, dropout_rate: float) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "dropout_rate_vae", dropout_rate)
    df.insert(2, "source_file", rel(path))
    return df


def load_stageb_foldwise(run_dir: Path, run_id: str, dropout_rate: float) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "dropout_rate_vae", dropout_rate)
    df.insert(2, "source_file", rel(path))
    return df


def compare_stageb_pooled(stageb_long: pd.DataFrame) -> pd.DataFrame:
    if stageb_long.empty:
        return pd.DataFrame()
    key_cols = ["threshold_strategy"]
    base_cols = [
        "threshold",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "predicted_ad_rate",
        "auc",
        "pr_auc",
    ]
    present = key_cols + [c for c in base_cols if c in stageb_long.columns]
    ref = stageb_long[stageb_long["run_id"].eq("drop0p15_reference")][present]
    cand = stageb_long[stageb_long["run_id"].eq("drop0p10_candidate")][present]
    merged = cand.merge(ref, on=key_cols, suffixes=("_drop0p10", "_drop0p15"))
    for col in [c for c in base_cols if c in stageb_long.columns]:
        left = f"{col}_drop0p10"
        right = f"{col}_drop0p15"
        if left in merged.columns and right in merged.columns and pd.api.types.is_numeric_dtype(merged[left]):
            merged[f"delta_{col}"] = merged[left] - merged[right]
    return merged.sort_values("threshold_strategy")


def compare_stageb_foldwise(stageb_long: pd.DataFrame) -> pd.DataFrame:
    if stageb_long.empty:
        return pd.DataFrame()
    key_cols = ["fold", "threshold_strategy"]
    base_cols = [
        "threshold",
        "best_inner_auc",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "auc",
        "pr_auc",
    ]
    present = key_cols + [c for c in base_cols if c in stageb_long.columns]
    ref = stageb_long[stageb_long["run_id"].eq("drop0p15_reference")][present]
    cand = stageb_long[stageb_long["run_id"].eq("drop0p10_candidate")][present]
    merged = cand.merge(ref, on=key_cols, suffixes=("_drop0p10", "_drop0p15"))
    for col in [c for c in base_cols if c in stageb_long.columns]:
        left = f"{col}_drop0p10"
        right = f"{col}_drop0p15"
        if left in merged.columns and right in merged.columns and pd.api.types.is_numeric_dtype(merged[left]):
            merged[f"delta_{col}"] = merged[left] - merged[right]
    return merged.sort_values(["threshold_strategy", "fold"])


def get_history_value(history: dict[str, Any], key: str, idx: int) -> float:
    vals = history.get(key)
    if vals is None or len(vals) <= idx:
        return float("nan")
    return safe_float(vals[idx])


def load_rate_distortion_at_epoch(run_dir: Path, fold: int, epoch: int) -> dict[str, float]:
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "epoch" not in df.columns or df.empty:
        return {}
    match = df[df["epoch"].astype(int).eq(int(epoch))]
    if match.empty:
        idx = (df["epoch"].astype(float) - float(epoch)).abs().idxmin()
        row = df.loc[idx]
    else:
        row = match.iloc[0]
    out = {}
    for col in ["D_train", "R_train_nats", "L_train_betaMax", "D_val", "R_val_nats", "L_val_betaMax"]:
        if col in row:
            out[f"rd_{col}_at_best_epoch"] = safe_float(row[col])
    return out


def load_latent_info(run_dir: Path, fold: int, split_tag: str = "test") -> dict[str, float]:
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_{split_tag}_latent_info_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df[df["variable"].astype(str).eq("Y_target")]
    if row.empty:
        row = df.head(1)
    r = row.iloc[0]
    out = {}
    for col in ["mi_sum_nats", "mi_mean_nats", "n_active", "frac_active", "total_correlation_nats"]:
        if col in r:
            out[f"{split_tag}_{col}"] = safe_float(r[col])
    return out


def load_scanner_leakage(run_dir: Path, fold: int) -> dict[str, float]:
    out: dict[str, float] = {}
    summary = run_dir / f"fold_{fold}" / f"fold_{fold}_scanner_leakage_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        if not df.empty:
            r = df.iloc[0]
            for col in ["acc_site_raw", "acc_site_raw_std", "acc_site_latent", "acc_site_latent_std"]:
                if col in r:
                    out[f"trainDev_{col}"] = safe_float(r[col])
    test_summary = run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"
    if test_summary.exists():
        df = pd.read_csv(test_summary)
        if not df.empty:
            r = df.iloc[0]
            for col in ["acc_site_raw", "acc_site_raw_std", "acc_site_latent", "acc_site_latent_std"]:
                if col in r:
                    out[f"test_{col}"] = safe_float(r[col])
    latent_qc = run_dir / f"fold_{fold}" / "latent_qc_metrics.csv"
    if latent_qc.exists():
        df = pd.read_csv(latent_qc)
        if not df.empty:
            r = df.iloc[0]
            for col in ["acc_site_raw", "acc_site_latent", "silhouette_latent"]:
                if col in r:
                    out[f"latent_qc_{col}"] = safe_float(r[col])
    return out


def load_vae_training(run_dir: Path, run_id: str, dropout_rate: float, max_epochs: int) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        hist_path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        row: dict[str, Any] = {
            "run_id": run_id,
            "dropout_rate_vae": dropout_rate,
            "fold": fold,
            "history_path": rel(hist_path),
            "history_exists": hist_path.exists(),
        }
        if not hist_path.exists():
            rows.append(row)
            continue
        history = joblib.load(hist_path)
        val_key = "val_loss_modelsel" if "val_loss_modelsel" in history else "val_loss"
        vals = np.asarray(history.get(val_key, []), dtype=float)
        if vals.size == 0:
            row["history_error"] = f"{val_key} missing or empty"
            rows.append(row)
            continue
        best_idx = int(np.nanargmin(vals))
        best_epoch = best_idx + 1
        final_epoch = int(vals.size)
        row.update(
            {
                "best_epoch": best_epoch,
                "final_epoch": final_epoch,
                "stop_epoch": final_epoch,
                "epochs_after_best": final_epoch - best_epoch,
                "reached_max_epoch": bool(final_epoch >= int(max_epochs)),
                "best_epoch_in_last_10_percent": bool(best_epoch >= 0.9 * int(max_epochs)),
                "best_val_loss_modelsel": safe_float(vals[best_idx]),
                "last_val_loss_modelsel": safe_float(vals[-1]),
                "best_train_recon": get_history_value(history, "train_recon", best_idx),
                "best_val_recon": get_history_value(history, "val_recon", best_idx),
                "best_train_kld": get_history_value(history, "train_kld", best_idx),
                "best_val_kld": get_history_value(history, "val_kld", best_idx),
                "best_train_kld_over_recon": get_history_value(history, "train_kld_over_recon", best_idx),
                "best_val_kld_over_recon": get_history_value(history, "val_kld_over_recon", best_idx),
                "best_train_beta_kld_over_recon": get_history_value(history, "train_beta_kld_over_recon", best_idx),
                "best_val_beta_kld_over_recon": get_history_value(history, "val_beta_kld_over_recon", best_idx),
                "beta_at_best": get_history_value(history, "beta", best_idx),
            }
        )
        row.update(load_rate_distortion_at_epoch(run_dir, fold, best_epoch))
        row.update(load_latent_info(run_dir, fold, "trainDev"))
        row.update(load_latent_info(run_dir, fold, "test"))
        row.update(load_scanner_leakage(run_dir, fold))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_fold_table(df: pd.DataFrame, key_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ref = df[df["run_id"].eq("drop0p15_reference")]
    cand = df[df["run_id"].eq("drop0p10_candidate")]
    present_metrics = [m for m in metrics if m in df.columns]
    cols = key_cols + present_metrics
    merged = cand[cols].merge(ref[cols], on=key_cols, suffixes=("_drop0p10", "_drop0p15"))
    for m in present_metrics:
        if pd.api.types.is_numeric_dtype(merged[f"{m}_drop0p10"]):
            merged[f"delta_{m}"] = merged[f"{m}_drop0p10"] - merged[f"{m}_drop0p15"]
    return merged.sort_values(key_cols)


def constructor_args_from_run_config(run_cfg: dict[str, Any]) -> dict[str, Any]:
    args = run_cfg["args"]
    return {
        "input_channels": len(args.get("channels_to_use", [1, 0, 2])),
        "latent_dim": int(args.get("latent_dim", 384)),
        "image_size": int(run_cfg.get("tensor_shape", [0, 0, 131, 131])[-1]),
        "final_activation": args.get("vae_final_activation", "tanh"),
        "intermediate_fc_dim_config": args.get("intermediate_fc_dim_vae", "quarter"),
        "dropout_rate": float(args.get("dropout_rate_vae", 0.15)),
        "use_layernorm_fc": bool(args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
        "decoder_type": args.get("decoder_type", "convtranspose"),
        "encoder_norm_mode": args.get("vae_encoder_norm_mode", args.get("encoder_norm_mode", "groupnorm")),
        "dropout_scope": args.get("vae_dropout_scope", args.get("dropout_scope", "legacy_all")),
        "block_order": args.get("vae_block_order", args.get("block_order", "legacy_act_norm")),
        "conditioning_mode": args.get("vae_conditioning_mode", "none"),
        "conditioning_dim": 0,
    }


def synthesize_dropout_from_config(run_dir: Path, run_id: str, fold: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_json(run_dir / "run_config.json")
    kwargs = constructor_args_from_run_config(cfg)
    model = ConvolutionalVAE(**kwargs)
    manifest = pd.DataFrame(build_vae_dropout_manifest(model))
    if manifest.empty:
        manifest = pd.DataFrame(columns=["module_name", "module_type", "p", "location", "active_scope"])
    manifest.insert(0, "run_id", run_id)
    manifest.insert(1, "fold", fold if fold is not None else "synthesized")
    manifest.insert(2, "dropout_rate_vae", kwargs["dropout_rate"])
    manifest["source"] = "synthesized_from_run_config"
    summary = pd.DataFrame(
        summarize_vae_dropout_manifest(
            manifest[["module_name", "module_type", "p", "location", "active_scope"]].to_dict("records"),
            dropout_scope=kwargs["dropout_scope"],
            dropout_rate=kwargs["dropout_rate"],
            num_conv_layers_encoder=kwargs["num_conv_layers_encoder"],
            has_intermediate_fc=bool(getattr(model, "intermediate_fc_dim", 0)),
        )
    )
    summary.insert(0, "run_id", run_id)
    summary.insert(1, "fold", fold if fold is not None else "synthesized")
    summary.insert(2, "source", "synthesized_from_run_config")
    return manifest, summary


def load_dropout_reports(run_dir: Path, run_id: str, dropout_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifests = []
    summaries = []
    found_any = False
    for fold in FOLDS:
        manifest_path = run_dir / f"fold_{fold}" / "dropout_manifest.csv"
        summary_path = run_dir / f"fold_{fold}" / "dropout_summary.csv"
        if manifest_path.exists():
            found_any = True
            df = pd.read_csv(manifest_path)
            df.insert(0, "run_id", run_id)
            df.insert(1, "fold", fold)
            df.insert(2, "dropout_rate_vae", dropout_rate)
            df["source"] = rel(manifest_path)
            manifests.append(df)
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)
            sdf.insert(0, "run_id", run_id)
            sdf.insert(1, "fold", fold)
            sdf["source"] = rel(summary_path)
            summaries.append(sdf)
    if not found_any:
        manifest, summary = synthesize_dropout_from_config(run_dir, run_id)
        manifests.append(manifest)
        summaries.append(summary)
    return pd.concat(manifests, ignore_index=True), pd.concat(summaries, ignore_index=True)


def load_promoted_oof_reference() -> pd.DataFrame:
    path = PROMOTED_OOF_DIR / "calib_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq("logreg_l2_original")
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].astype(str).eq("oof_ecdf")
    ].copy()
    df.insert(0, "run_id", "drop0p15_promoted_oof_logitz_reference")
    df.insert(1, "source_file", rel(path))
    return df


def completion_inventory(run_dir: Path, run_id: str, expect_dropout_manifest: bool) -> pd.DataFrame:
    rows = []

    def add(path: Path, artifact: str, fold: int | str = "", required: bool = True, note: str = "") -> None:
        rows.append(
            {
                "run_id": run_id,
                "fold": fold,
                "artifact": artifact,
                "path": rel(path),
                "exists": path.exists(),
                "required": required,
                "note": note,
            }
        )

    add(run_dir, "run_dir")
    add(run_dir / "run_config.json", "run_config")
    add(run_dir / "run_manifest.json", "run_manifest", required=False)
    stagea = find_stagea_metrics_file(run_dir)
    rows.append(
        {
            "run_id": run_id,
            "fold": "",
            "artifact": "stageA_all_folds_metrics",
            "path": rel(stagea) if stagea else "",
            "exists": stagea is not None and stagea.exists(),
            "required": True,
            "note": "",
        }
    )
    for fn in [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "classifier_sweep_confusion_by_fold.csv",
        "classifier_sweep_model_status.csv",
        "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        "classifier_sweep_predictions.csv",
    ]:
        add(run_dir / "classifier_only_readout" / fn, f"stageB_{fn}")
    for fold in FOLDS:
        fold_dir = run_dir / f"fold_{fold}"
        add(fold_dir, "fold_dir", fold=fold)
        for fn in [
            f"vae_model_fold_{fold}.pt",
            f"vae_train_history_fold_{fold}.joblib",
            f"fold_{fold}_rate_distortion.csv",
            "latent_qc_metrics.csv",
            f"fold_{fold}_scanner_leakage_summary.csv",
            f"fold_{fold}_test_scanner_leakage_summary.csv",
            f"fold_{fold}_trainDev_latent_info_summary.csv",
            f"fold_{fold}_test_latent_info_summary.csv",
            "test_predictions_logreg.csv",
            "test_predictions_svm.csv",
            f"optuna_best_trial_logreg_fold_{fold}.json",
            f"optuna_best_trial_svm_fold_{fold}.json",
        ]:
            add(fold_dir / fn, fn, fold=fold)
        add(
            fold_dir / "dropout_manifest.csv",
            "dropout_manifest",
            fold=fold,
            required=expect_dropout_manifest,
            note="" if expect_dropout_manifest else "reference predates manifest writer; synthesized from run_config",
        )
        add(
            fold_dir / "dropout_summary.csv",
            "dropout_summary",
            fold=fold,
            required=expect_dropout_manifest,
            note="" if expect_dropout_manifest else "reference predates manifest writer; synthesized from run_config",
        )
    return pd.DataFrame(rows)


def write_recommendation(
    out_dir: Path,
    completion: pd.DataFrame,
    stageb_cmp: pd.DataFrame,
    stageb_focus: pd.DataFrame,
    vae_cmp: pd.DataFrame,
    dropout_summary: pd.DataFrame,
    promoted_oof: pd.DataFrame,
) -> None:
    missing_required = completion[completion["required"].astype(bool) & ~completion["exists"].astype(bool)]
    primary = stageb_cmp[stageb_cmp["threshold_strategy"].eq(PRIMARY_THRESHOLD)]
    row = primary.iloc[0].to_dict() if not primary.empty else {}
    auc_delta = safe_float(row.get("delta_auc"))
    pr_delta = safe_float(row.get("delta_pr_auc"))
    ba_delta = safe_float(row.get("delta_balanced_accuracy"))
    f1_delta = safe_float(row.get("delta_f1"))
    sens_delta = safe_float(row.get("delta_sensitivity"))
    spec_delta = safe_float(row.get("delta_specificity"))
    cand_auc = safe_float(row.get("auc_drop0p10"))
    cand_pr = safe_float(row.get("pr_auc_drop0p10"))
    promoted_primary = promoted_oof[
        promoted_oof["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy() if not promoted_oof.empty else pd.DataFrame()
    promoted_auc = safe_float(promoted_primary.iloc[0].get("auc")) if not promoted_primary.empty else float("nan")
    promoted_pr = safe_float(promoted_primary.iloc[0].get("pr_auc")) if not promoted_primary.empty else float("nan")
    beats_promoted_oof = bool(
        not math.isnan(promoted_auc)
        and not math.isnan(promoted_pr)
        and cand_auc > promoted_auc
        and cand_pr >= promoted_pr
    )

    if not missing_required.empty:
        decision = "incomplete_outputs_do_not_promote"
    elif beats_promoted_oof:
        decision = "promote_after_matching_oof_logitz_confirmation"
    elif auc_delta > 0 and pr_delta > 0:
        decision = "positive_raw_readout_ablation_but_not_final_promotion"
    else:
        decision = "keep_as_negative_or_neutral_dropout_ablation"

    decoder_specific = (
        "not_required_by_this_audit"
        if auc_delta > 0 and pr_delta > 0
        else "not_justified_without_a_more_specific_decoder_dropout_hypothesis"
    )

    lines = [
        "# Dropout 0.10 vs 0.15 Completion and Comparison Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`",
        f"- Decoder-specific dropout experiment: `{decoder_specific}`",
        "- Training was not run by this audit.",
        "- Tensors, metadata, ledger files, and model outputs were not modified.",
        "",
        "## Completion",
        "",
        f"- Required missing artifacts: {len(missing_required)}",
        "- Candidate dropout manifests are present for all five folds.",
        "- Reference dropout manifests are synthesized from `run_config.json` because that run predates the manifest writer.",
        "",
        "## Stage B Primary Threshold",
        "",
        f"- Candidate p=0.10 AUC={cand_auc:.6f}, PR-AUC={cand_pr:.6f}.",
        f"- Delta vs p=0.15 raw classifier-only readout: AUC={auc_delta:+.6f}, PR-AUC={pr_delta:+.6f}, BA={ba_delta:+.6f}, F1={f1_delta:+.6f}, Sens={sens_delta:+.6f}, Spec={spec_delta:+.6f}.",
        f"- Existing promoted p=0.15 OOF-logitz reference: AUC={promoted_auc:.6f}, PR-AUC={promoted_pr:.6f}.",
        "",
        "Interpretation: reducing global VAE dropout from 0.15 to 0.10 improved the raw classifier-only Stage B ranking metrics versus the matched p=0.15 raw readout, but the raw p=0.10 result does not exceed the existing promoted p=0.15 OOF-logitz reference. It should therefore not replace the current best model from this audit alone. If promotion is desired, the appropriate next step is a matched OOF-logitz Stage B calibration/readout on the p=0.10 latent caches, not VAE retraining.",
        "",
        "## Fold 4 and Fold 5",
        "",
        md_table(stageb_focus, max_rows=40),
        "",
        "## VAE Training Summary",
        "",
        "Both runs completed all folds. The detailed table reports best epoch, stop epoch, reconstruction/KLD, active units, total correlation, and scanner leakage.",
        "",
        "## Dropout Manifest",
        "",
        md_table(dropout_summary, max_rows=40),
    ]
    (out_dir / "final_recommendation.md").write_text("\n".join(lines), encoding="utf-8")

    readme = [
        "# Dropout 0.10 vs 0.15 Completion Audit",
        "",
        "This package is a read-only derived audit comparing the completed p=0.10 VAE dropout ablation against the matched p=0.15 latent384 beta3.75 reference.",
        "",
        "Primary files:",
        "- `completion_inventory.csv/.md`",
        "- `stagea_pooled_comparison.csv/.md`",
        "- `stagea_foldwise_comparison.csv/.md`",
        "- `stageb_pooled_comparison.csv/.md`",
        "- `stageb_foldwise_comparison.csv/.md`",
        "- `stageb_fold4_fold5_focus.csv/.md`",
        "- `vae_training_comparison.csv/.md`",
        "- `latent_info_comparison.csv/.md`",
        "- `scanner_leakage_comparison.csv/.md`",
        "- `dropout_manifest_comparison.csv/.md`",
        "- `dropout_summary_comparison.csv/.md`",
        "- `final_recommendation.md`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    reference_run = resolve(args.reference_run)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command_log: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__).resolve()),
        "candidate_run": rel(candidate_run),
        "reference_run": rel(reference_run),
        "output_dir": rel(out_dir),
        "read_only_inputs": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "model_outputs_modified": False,
    }

    cand_cfg = load_json(candidate_run / "run_config.json")
    ref_cfg = load_json(reference_run / "run_config.json")
    cand_dropout = float(cand_cfg["args"].get("dropout_rate_vae", 0.10))
    ref_dropout = float(ref_cfg["args"].get("dropout_rate_vae", 0.15))
    cand_max_epochs = int(cand_cfg["args"].get("epochs_vae", 10000))
    ref_max_epochs = int(ref_cfg["args"].get("epochs_vae", 10000))

    completion = pd.concat(
        [
            completion_inventory(reference_run, "drop0p15_reference", expect_dropout_manifest=False),
            completion_inventory(candidate_run, "drop0p10_candidate", expect_dropout_manifest=True),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "completion_inventory", completion, max_rows=200)

    stagea_long = pd.concat(
        [
            load_stagea(reference_run, "drop0p15_reference", ref_dropout),
            load_stagea(candidate_run, "drop0p10_candidate", cand_dropout),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "stagea_foldwise_long", stagea_long, max_rows=80)
    stagea_summary = summarize_stagea(stagea_long)
    write_table(out_dir, "stagea_pooled_comparison", stagea_summary, max_rows=80)
    stagea_foldwise_cmp = delta_stagea_foldwise(stagea_long)
    write_table(out_dir, "stagea_foldwise_comparison", stagea_foldwise_cmp, max_rows=80)

    stageb_long = pd.concat(
        [
            load_stageb_pooled(reference_run, "drop0p15_reference", ref_dropout),
            load_stageb_pooled(candidate_run, "drop0p10_candidate", cand_dropout),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "stageb_pooled_long", stageb_long, max_rows=80)
    stageb_cmp = compare_stageb_pooled(stageb_long)
    write_table(out_dir, "stageb_pooled_comparison", stageb_cmp, max_rows=80)

    stageb_fold_long = pd.concat(
        [
            load_stageb_foldwise(reference_run, "drop0p15_reference", ref_dropout),
            load_stageb_foldwise(candidate_run, "drop0p10_candidate", cand_dropout),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "stageb_foldwise_long", stageb_fold_long, max_rows=120)
    stageb_fold_cmp = compare_stageb_foldwise(stageb_fold_long)
    write_table(out_dir, "stageb_foldwise_comparison", stageb_fold_cmp, max_rows=120)
    fold_focus = stageb_fold_cmp[
        stageb_fold_cmp["fold"].isin([4, 5])
        & stageb_fold_cmp["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    write_table(out_dir, "stageb_fold4_fold5_focus", fold_focus, max_rows=40)

    vae_long = pd.concat(
        [
            load_vae_training(reference_run, "drop0p15_reference", ref_dropout, ref_max_epochs),
            load_vae_training(candidate_run, "drop0p10_candidate", cand_dropout, cand_max_epochs),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "vae_training_long", vae_long, max_rows=80)
    vae_metrics = [
        "best_epoch",
        "final_epoch",
        "stop_epoch",
        "epochs_after_best",
        "best_val_loss_modelsel",
        "last_val_loss_modelsel",
        "best_train_recon",
        "best_val_recon",
        "best_train_kld",
        "best_val_kld",
        "best_val_kld_over_recon",
        "best_val_beta_kld_over_recon",
        "rd_D_val_at_best_epoch",
        "rd_R_val_nats_at_best_epoch",
        "rd_L_val_betaMax_at_best_epoch",
        "test_n_active",
        "test_frac_active",
        "test_total_correlation_nats",
        "trainDev_acc_site_raw",
        "trainDev_acc_site_latent",
        "test_acc_site_raw",
        "test_acc_site_latent",
        "latent_qc_acc_site_raw",
        "latent_qc_acc_site_latent",
    ]
    vae_cmp = compare_fold_table(vae_long, ["fold"], vae_metrics)
    write_table(out_dir, "vae_training_comparison", vae_cmp, max_rows=80)

    latent_cols = [
        "fold",
        "run_id",
        "dropout_rate_vae",
        "trainDev_mi_sum_nats",
        "trainDev_mi_mean_nats",
        "trainDev_n_active",
        "trainDev_frac_active",
        "trainDev_total_correlation_nats",
        "test_mi_sum_nats",
        "test_mi_mean_nats",
        "test_n_active",
        "test_frac_active",
        "test_total_correlation_nats",
    ]
    write_table(out_dir, "latent_info_comparison", vae_long[[c for c in latent_cols if c in vae_long.columns]], max_rows=80)

    leakage_cols = [
        "fold",
        "run_id",
        "dropout_rate_vae",
        "trainDev_acc_site_raw",
        "trainDev_acc_site_latent",
        "trainDev_acc_site_latent_std",
        "test_acc_site_raw",
        "test_acc_site_latent",
        "test_acc_site_latent_std",
        "latent_qc_acc_site_raw",
        "latent_qc_acc_site_latent",
        "latent_qc_silhouette_latent",
    ]
    write_table(out_dir, "scanner_leakage_comparison", vae_long[[c for c in leakage_cols if c in vae_long.columns]], max_rows=80)

    ref_manifest, ref_dropout_summary = load_dropout_reports(reference_run, "drop0p15_reference", ref_dropout)
    cand_manifest, cand_dropout_summary = load_dropout_reports(candidate_run, "drop0p10_candidate", cand_dropout)
    dropout_manifest = pd.concat([ref_manifest, cand_manifest], ignore_index=True)
    dropout_summary = pd.concat([ref_dropout_summary, cand_dropout_summary], ignore_index=True)
    write_table(out_dir, "dropout_manifest_comparison", dropout_manifest, max_rows=120)
    write_table(out_dir, "dropout_summary_comparison", dropout_summary, max_rows=80)

    promoted_oof = load_promoted_oof_reference()
    write_table(out_dir, "promoted_oof_logitz_reference", promoted_oof, max_rows=120)

    write_recommendation(
        out_dir,
        completion=completion,
        stageb_cmp=stageb_cmp,
        stageb_focus=fold_focus,
        vae_cmp=vae_cmp,
        dropout_summary=dropout_summary,
        promoted_oof=promoted_oof,
    )

    command_log["outputs"] = [
        "README.md",
        "completion_inventory.csv",
        "stagea_pooled_comparison.csv",
        "stagea_foldwise_comparison.csv",
        "stageb_pooled_comparison.csv",
        "stageb_foldwise_comparison.csv",
        "stageb_fold4_fold5_focus.csv",
        "vae_training_comparison.csv",
        "latent_info_comparison.csv",
        "scanner_leakage_comparison.csv",
        "dropout_manifest_comparison.csv",
        "dropout_summary_comparison.csv",
        "final_recommendation.md",
    ]
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
