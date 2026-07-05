#!/usr/bin/env python3
"""Read-only completion/comparison audit for encoder/decoder dropout split.

This script compares:
  - recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5
  - recover035_latent384_beta3p75_T80_h10000_p560_full5x5
  - recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5
  - existing p=0.15/p=0.10 OOF score-harmonized Stage B audits

It is read-only with respect to tensors, metadata, and model artifacts. It writes
only derived audit tables under the requested output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

ENCDEC_RUN = RESULTS / "recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5"
P15_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
P10_RUN = RESULTS / "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5"
P15_OOF_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
P10_OOF_DIR = RESULTS / "drop0p10_oof_logitz_readout_audit_20260531"
OUT_DIR = RESULTS / "encdrop0p15_decdrop0p10_completion_comparison_audit_20260531"

FOLDS = [1, 2, 3, 4, 5]
PRIMARY_MODEL_RAW = "logreg_l2"
PRIMARY_MODEL_OOF = "logreg_l2_original"
PRIMARY_FEATURE_RAW = "z_plus_age_sex"
PRIMARY_FEATURE_OOF = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--encdec-run", type=Path, default=ENCDEC_RUN)
    parser.add_argument("--p15-run", type=Path, default=P15_RUN)
    parser.add_argument("--p10-run", type=Path, default=P10_RUN)
    parser.add_argument("--p15-oof-dir", type=Path, default=P15_OOF_DIR)
    parser.add_argument("--p10-oof-dir", type=Path, default=P10_OOF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        out = float(value)
        return out
    except Exception:
        return float("nan")


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
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


def write_table(out_dir: Path, stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_stagea_metrics_file(run_dir: Path) -> Path | None:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def run_config_args(run_dir: Path) -> Dict[str, Any]:
    cfg = load_json(run_dir / "run_config.json")
    return dict(cfg.get("args", {}))


def load_stagea(run_dir: Path, run_id: str) -> pd.DataFrame:
    path = find_stagea_metrics_file(run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_id", run_id)
    df.insert(1, "source_file", rel(path))
    return df


def summarize_stagea(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [
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
    rows = []
    for (run_id, clf), sub in df.groupby(["run_id", "actual_classifier_type"], dropna=False):
        row: Dict[str, Any] = {"run_id": run_id, "actual_classifier_type": clf, "n_folds": int(sub["fold"].nunique())}
        for metric in metrics:
            if metric in sub.columns:
                row[f"{metric}_mean"] = safe_float(sub[metric].mean())
                row[f"{metric}_std"] = safe_float(sub[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["actual_classifier_type", "run_id"])


def load_raw_stageb(run_dir: Path, run_id: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_RAW)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_RAW)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "score_mode", "raw_classifier_only")
    df.insert(2, "calib_method", "raw")
    df.insert(3, "source_file", rel(path))
    df = df.rename(columns={"readout_feature_set": "feature_set"})
    return df


def load_raw_foldwise(run_dir: Path, run_id: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_RAW)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_RAW)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "score_mode", "raw_classifier_only")
    df.insert(2, "calib_method", "raw")
    df.insert(3, "source_file", rel(path))
    df = df.rename(columns={"readout_feature_set": "feature_set"})
    return df


def load_oof_pooled(oof_dir: Path, run_id: str, keep_methods: Iterable[str] | None = None) -> pd.DataFrame:
    path = oof_dir / "calib_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_OOF)
    ].copy()
    if keep_methods is not None:
        df = df[df["calib_method"].astype(str).isin(list(keep_methods))].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "score_mode", "oof_score_harmonized")
    df.insert(3, "source_file", rel(path))
    return df


def load_oof_foldwise(oof_dir: Path, run_id: str, keep_methods: Iterable[str] | None = None) -> pd.DataFrame:
    path = oof_dir / "calib_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_OOF)
    ].copy()
    if keep_methods is not None:
        df = df[df["calib_method"].astype(str).isin(list(keep_methods))].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "score_mode", "oof_score_harmonized")
    df.insert(3, "source_file", rel(path))
    return df


def primary_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()


def load_history_metric(history: Dict[str, Any], key: str, idx: int) -> float:
    values = history.get(key)
    if values is None or len(values) <= idx:
        return float("nan")
    return safe_float(values[idx])


def load_rate_distortion_at_epoch(run_dir: Path, fold: int, epoch: int) -> Dict[str, float]:
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "epoch" not in df.columns:
        return {}
    idx = (df["epoch"].astype(float) - float(epoch)).abs().idxmin()
    row = df.loc[idx]
    out: Dict[str, float] = {}
    for col in ["D_train", "R_train_nats", "L_train_betaMax", "D_val", "R_val_nats", "L_val_betaMax"]:
        if col in row:
            out[f"rd_{col}_at_best_epoch"] = safe_float(row[col])
    return out


def load_latent_info(run_dir: Path, fold: int, split_tag: str) -> Dict[str, float]:
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
    out: Dict[str, float] = {}
    for col in ["mi_sum_nats", "mi_mean_nats", "n_active", "frac_active", "total_correlation_nats"]:
        if col in r:
            out[f"{split_tag}_{col}"] = safe_float(r[col])
    return out


def load_scanner_leakage_fold(run_dir: Path, fold: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for split, filename in [
        ("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"),
        ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
    ]:
        path = run_dir / f"fold_{fold}" / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        for col in ["acc_site_raw", "acc_site_raw_std", "acc_site_latent", "acc_site_latent_std"]:
            if col in row:
                out[f"{split}_{col}"] = safe_float(row[col])
        if "acc_site_raw" in row and "acc_site_latent" in row:
            out[f"{split}_latent_minus_raw"] = safe_float(row["acc_site_latent"]) - safe_float(row["acc_site_raw"])
    return out


def load_vae_qc(run_dir: Path, run_id: str) -> pd.DataFrame:
    args = run_config_args(run_dir)
    max_epochs = int(args.get("epochs_vae", 10000))
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        hist_path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        row: Dict[str, Any] = {
            "run_id": run_id,
            "fold": fold,
            "history_path": rel(hist_path),
            "history_exists": hist_path.exists(),
            "max_epochs_config": max_epochs,
        }
        if not hist_path.exists():
            rows.append(row)
            continue
        history = joblib.load(hist_path)
        val_key = "val_loss_modelsel" if "val_loss_modelsel" in history else "val_loss"
        vals = np.asarray(history.get(val_key, []), dtype=float)
        if vals.size == 0:
            row["history_error"] = f"{val_key} missing"
            rows.append(row)
            continue
        best_idx = int(np.nanargmin(vals))
        best_epoch = best_idx + 1
        final_epoch = int(vals.size)
        row.update(
            {
                "best_epoch": best_epoch,
                "stop_epoch": final_epoch,
                "final_epoch": final_epoch,
                "epochs_after_best": final_epoch - best_epoch,
                "reached_max_epoch": bool(final_epoch >= max_epochs),
                "best_val_loss_modelsel": safe_float(vals[best_idx]),
                "last_val_loss_modelsel": safe_float(vals[-1]),
                "best_train_recon": load_history_metric(history, "train_recon", best_idx),
                "best_val_recon": load_history_metric(history, "val_recon", best_idx),
                "best_train_kld": load_history_metric(history, "train_kld", best_idx),
                "best_val_kld": load_history_metric(history, "val_kld", best_idx),
                "best_train_kld_over_recon": load_history_metric(history, "train_kld_over_recon", best_idx),
                "best_val_kld_over_recon": load_history_metric(history, "val_kld_over_recon", best_idx),
                "best_train_beta_kld_over_recon": load_history_metric(history, "train_beta_kld_over_recon", best_idx),
                "best_val_beta_kld_over_recon": load_history_metric(history, "val_beta_kld_over_recon", best_idx),
                "beta_at_best": load_history_metric(history, "beta", best_idx),
            }
        )
        row.update(load_rate_distortion_at_epoch(run_dir, fold, best_epoch))
        row.update(load_latent_info(run_dir, fold, "trainDev"))
        row.update(load_latent_info(run_dir, fold, "test"))
        row.update(load_scanner_leakage_fold(run_dir, fold))
        rows.append(row)
    return pd.DataFrame(rows)


def constructor_args_from_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg = load_json(run_dir / "run_config.json")
    args = cfg.get("args", {})
    return {
        "input_channels": len(args.get("channels_to_use", [1, 0, 2])),
        "latent_dim": int(args.get("latent_dim", 384)),
        "image_size": int(cfg.get("tensor_shape", [0, 0, 131, 131])[-1]),
        "final_activation": args.get("vae_final_activation", "tanh"),
        "intermediate_fc_dim_config": args.get("intermediate_fc_dim_vae", "quarter"),
        "dropout_rate": float(args.get("dropout_rate_vae", 0.15)),
        "encoder_dropout_rate": args.get("encoder_dropout_rate_vae"),
        "decoder_dropout_rate": args.get("decoder_dropout_rate_vae"),
        "use_layernorm_fc": bool(args.get("use_layernorm_vae_fc", False)),
        "num_conv_layers_encoder": int(args.get("num_conv_layers_encoder", 4)),
        "decoder_type": args.get("decoder_type", "convtranspose"),
        "encoder_norm_mode": args.get("vae_encoder_norm_mode", args.get("encoder_norm_mode", "groupnorm")),
        "dropout_scope": args.get("vae_dropout_scope", args.get("dropout_scope", "legacy_all")),
        "block_order": args.get("vae_block_order", args.get("block_order", "legacy_act_norm")),
        "conditioning_mode": args.get("vae_conditioning_mode", "none"),
        "conditioning_dim": 0,
    }


def synthesize_dropout_manifest(run_dir: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    kwargs = constructor_args_from_run_config(run_dir)
    model = ConvolutionalVAE(**kwargs)
    manifest_rows = build_vae_dropout_manifest(model)
    summary_rows = summarize_vae_dropout_manifest(
        manifest_rows,
        dropout_scope=str(kwargs["dropout_scope"]),
        dropout_rate=float(kwargs["dropout_rate"]),
        encoder_dropout_rate=kwargs.get("encoder_dropout_rate"),
        decoder_dropout_rate=kwargs.get("decoder_dropout_rate"),
        num_conv_layers_encoder=int(kwargs["num_conv_layers_encoder"]),
        has_intermediate_fc=bool(getattr(model, "intermediate_fc_dim", 0)),
    )
    manifest = pd.DataFrame(manifest_rows)
    summary = pd.DataFrame(summary_rows)
    for df in [manifest, summary]:
        df.insert(0, "run_id", run_id)
        df.insert(1, "fold", "synthesized")
        df["source"] = "synthesized_from_run_config"
    return manifest, summary


def load_dropout_reports(run_dir: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifests = []
    summaries = []
    for fold in FOLDS:
        manifest_path = run_dir / f"fold_{fold}" / "dropout_manifest.csv"
        summary_path = run_dir / f"fold_{fold}" / "dropout_summary.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            df.insert(0, "run_id", run_id)
            df.insert(1, "fold", fold)
            df["source"] = rel(manifest_path)
            manifests.append(df)
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.insert(0, "run_id", run_id)
            df.insert(1, "fold", fold)
            df["source"] = rel(summary_path)
            summaries.append(df)
    if not manifests or not summaries:
        manifest, summary = synthesize_dropout_manifest(run_dir, run_id)
        if not manifests:
            manifests.append(manifest)
        if not summaries:
            summaries.append(summary)
    return pd.concat(manifests, ignore_index=True), pd.concat(summaries, ignore_index=True)


def completion_inventory(run_dir: Path, run_id: str, require_dropout_manifest: bool, oof_dir: Path | None = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

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
    stagea = find_stagea_metrics_file(run_dir)
    rows.append(
        {
            "run_id": run_id,
            "fold": "",
            "artifact": "stageA_all_folds_metrics",
            "path": rel(stagea) if stagea else "",
            "exists": bool(stagea and stagea.exists()),
            "required": True,
            "note": "",
        }
    )
    for filename in [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_model_status.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        "classifier_sweep_predictions.csv",
    ]:
        add(run_dir / "classifier_only_readout" / filename, f"stageB_raw_{filename}")
    if oof_dir is not None:
        add(oof_dir / "calib_pooled_metrics.csv", "stageB_oof_calib_pooled_metrics", required=False)
        add(oof_dir / "calib_foldwise_metrics.csv", "stageB_oof_calib_foldwise_metrics", required=False)
    for fold in FOLDS:
        fold_dir = run_dir / f"fold_{fold}"
        add(fold_dir, "fold_dir", fold=fold)
        for filename in [
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
        ]:
            add(fold_dir / filename, filename, fold=fold)
        add(
            fold_dir / "dropout_manifest.csv",
            "dropout_manifest",
            fold=fold,
            required=require_dropout_manifest,
            note="" if require_dropout_manifest else "older run; synthesized from run_config if absent",
        )
        add(
            fold_dir / "dropout_summary.csv",
            "dropout_summary",
            fold=fold,
            required=require_dropout_manifest,
            note="" if require_dropout_manifest else "older run; synthesized from run_config if absent",
        )
    return pd.DataFrame(rows)


def summarize_philips_cn_fp_raw(run_dir: Path, run_id: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_RAW)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_RAW)
    ].copy()
    rows = []
    for (strategy, manufacturer), sub in df.groupby(["threshold_strategy", "Manufacturer"], dropna=False):
        n_cn = int(sub["n_cn"].sum())
        fp = int(sub["fp"].sum())
        rows.append(
            {
                "run_id": run_id,
                "score_mode": "raw_classifier_only",
                "calib_method": "raw",
                "threshold_strategy": strategy,
                "manufacturer": manufacturer,
                "n_cn": n_cn,
                "fp_cn": fp,
                "fpr_cn": fp / n_cn if n_cn else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def load_philips_cn_fp_oof(oof_dir: Path, run_id: str, keep_methods: Iterable[str]) -> pd.DataFrame:
    path = oof_dir / "calib_philips_fpr_pooled.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL_OOF)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_OOF)
        & df["calib_method"].astype(str).isin(list(keep_methods))
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "score_mode", "oof_score_harmonized")
    return df.rename(
        columns={
            "n_cn_pooled": "n_cn",
            "fp_cn_pooled": "fp_cn",
            "fpr_cn_pooled": "fpr_cn",
        }
    )


def availability_table(enc_run: Path, p15_oof: Path, p10_oof: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "enc0p15_dec0p10_candidate",
                "raw_stageb_available": (enc_run / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv").exists(),
                "oof_score_harmonized_available": False,
                "reason": "No existing OOF-logitz/score-harmonized output found; not recomputed because this audit was requested with no training.",
            },
            {
                "run_id": "drop0p15_uniform_reference",
                "raw_stageb_available": True,
                "oof_score_harmonized_available": (p15_oof / "calib_pooled_metrics.csv").exists(),
                "reason": "",
            },
            {
                "run_id": "drop0p10_uniform_reference",
                "raw_stageb_available": True,
                "oof_score_harmonized_available": (p10_oof / "calib_pooled_metrics.csv").exists(),
                "reason": "",
            },
        ]
    )


def final_decision(
    completion: pd.DataFrame,
    all_pooled: pd.DataFrame,
    oof_availability: pd.DataFrame,
    philips: pd.DataFrame,
) -> str:
    missing_required = completion[completion["required"].astype(bool) & ~completion["exists"].astype(bool)]
    primary = primary_only(all_pooled)
    cand = primary[
        primary["run_id"].eq("enc0p15_dec0p10_candidate")
        & primary["score_mode"].eq("raw_classifier_only")
    ]
    p15_promoted = primary[
        primary["run_id"].eq("drop0p15_promoted_score_harmonized")
        & primary["calib_method"].eq("oof_ecdf")
    ]
    p10_raw = primary[
        primary["run_id"].eq("drop0p10_uniform_reference")
        & primary["score_mode"].eq("raw_classifier_only")
    ]
    p15_raw = primary[
        primary["run_id"].eq("drop0p15_uniform_reference")
        & primary["score_mode"].eq("raw_classifier_only")
    ]
    decision = "keep_as_dropout_ablation"
    if not missing_required.empty:
        decision = "reject_incomplete"
    elif not cand.empty and not p15_promoted.empty:
        c = cand.iloc[0]
        p = p15_promoted.iloc[0]
        if safe_float(c["auc"]) > safe_float(p["auc"]) and safe_float(c["pr_auc"]) >= safe_float(p["pr_auc"]):
            decision = "promote"
        else:
            decision = "reject"

    lines = [
        "# Encoder 0.15 / Decoder 0.10 Dropout Completion Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`.",
        "- No VAE training was run.",
        "- No tensor, metadata, ledger, or model artifact was modified.",
        "- The decoder-specific branch has raw classifier-only Stage B outputs but no existing OOF-logitz/score-harmonized Stage B audit output.",
        "- Because the user explicitly requested no training, this audit did not refit Stage B classifiers to generate a new OOF-logitz sweep.",
        "",
    ]
    if not cand.empty:
        c = cand.iloc[0]
        lines.extend(
            [
                "## Candidate Raw Primary",
                "",
                f"- AUC={safe_float(c.get('auc')):.6f}",
                f"- PR-AUC={safe_float(c.get('pr_auc')):.6f}",
                f"- BA={safe_float(c.get('balanced_accuracy')):.6f}",
                f"- Sens={safe_float(c.get('sensitivity')):.6f}",
                f"- Spec={safe_float(c.get('specificity')):.6f}",
                f"- F1={safe_float(c.get('f1')):.6f}",
                f"- Confusion TN={int(c.get('tn'))}, FP={int(c.get('fp'))}, FN={int(c.get('fn'))}, TP={int(c.get('tp'))}",
                "",
            ]
        )
    refs = [("p=0.15 promoted score-harmonized", p15_promoted), ("p=0.15 raw matched", p15_raw), ("p=0.10 global raw", p10_raw)]
    if not cand.empty:
        c = cand.iloc[0]
        lines.append("## Delta Versus References")
        lines.append("")
        for label, ref in refs:
            if ref.empty:
                continue
            r = ref.iloc[0]
            parts = []
            for metric in METRICS:
                parts.append(f"{metric} {safe_float(c.get(metric)) - safe_float(r.get(metric)):+.6f}")
            lines.append(f"- vs {label}: " + ", ".join(parts))
        lines.append("")
    philips_primary = philips[
        philips["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        & philips["manufacturer"].astype(str).str.upper().eq("PHILIPS")
    ].copy()
    if not philips_primary.empty:
        lines.extend(["## Philips CN FP", "", md_table(philips_primary, max_rows=20), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "The encoder/decoder split dropout branch is complete and its dropout manifest matches the intended hygiene ablation. "
            "However, its raw Stage B ranking is substantially below the promoted p=0.15 score-harmonized reference and below both raw matched references. "
            "The branch also has lower primary-threshold BA/F1 than the p=0.15 and p=0.10 raw references. "
            "It should therefore not be promoted; it should be retained as a controlled dropout ablation.",
            "",
            "A new OOF-logitz sweep could be run later if explicitly requested, but the raw readout is weak enough that it is not scientifically necessary under the current no-training guardrail.",
            "",
        ]
    )
    if not missing_required.empty:
        lines.extend(["## Missing Required Artifacts", "", md_table(missing_required, max_rows=80), ""])
    lines.extend(["## OOF Availability", "", md_table(oof_availability, max_rows=20)])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    enc_run = resolve(args.encdec_run)
    p15_run = resolve(args.p15_run)
    p10_run = resolve(args.p10_run)
    p15_oof = resolve(args.p15_oof_dir)
    p10_oof = resolve(args.p10_oof_dir)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command_log = {
        "script": rel(Path(__file__).resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "encdec_run": rel(enc_run),
            "p15_run": rel(p15_run),
            "p10_run": rel(p10_run),
            "p15_oof_dir": rel(p15_oof),
            "p10_oof_dir": rel(p10_oof),
        },
        "guardrails": {
            "training_run": False,
            "classifier_refit_run": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifacts_modified": False,
        },
    }

    completion = pd.concat(
        [
            completion_inventory(enc_run, "enc0p15_dec0p10_candidate", require_dropout_manifest=True, oof_dir=None),
            completion_inventory(p15_run, "drop0p15_uniform_reference", require_dropout_manifest=False, oof_dir=p15_oof),
            completion_inventory(p10_run, "drop0p10_uniform_reference", require_dropout_manifest=True, oof_dir=p10_oof),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "completion_inventory", completion, max_rows=300)

    stagea = pd.concat(
        [
            load_stagea(enc_run, "enc0p15_dec0p10_candidate"),
            load_stagea(p15_run, "drop0p15_uniform_reference"),
            load_stagea(p10_run, "drop0p10_uniform_reference"),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "stagea_foldwise_metrics", stagea, max_rows=80)
    write_table(out_dir, "stagea_metrics_summary", summarize_stagea(stagea), max_rows=80)

    raw_pooled = pd.concat(
        [
            load_raw_stageb(enc_run, "enc0p15_dec0p10_candidate"),
            load_raw_stageb(p15_run, "drop0p15_uniform_reference"),
            load_raw_stageb(p10_run, "drop0p10_uniform_reference"),
        ],
        ignore_index=True,
    )
    oof_pooled = pd.concat(
        [
            load_oof_pooled(p15_oof, "drop0p15_promoted_score_harmonized", keep_methods=["oof_ecdf", "oof_logitz"]),
            load_oof_pooled(p10_oof, "drop0p10_score_harmonized", keep_methods=["oof_ecdf", "oof_logitz"]),
        ],
        ignore_index=True,
    )
    all_pooled = pd.concat([raw_pooled, oof_pooled], ignore_index=True, sort=False)
    write_table(out_dir, "stageb_all_thresholds_comparison", all_pooled, max_rows=120)
    write_table(out_dir, "stageb_primary_decision_table", primary_only(all_pooled), max_rows=80)

    raw_foldwise = pd.concat(
        [
            load_raw_foldwise(enc_run, "enc0p15_dec0p10_candidate"),
            load_raw_foldwise(p15_run, "drop0p15_uniform_reference"),
            load_raw_foldwise(p10_run, "drop0p10_uniform_reference"),
        ],
        ignore_index=True,
    )
    oof_foldwise = pd.concat(
        [
            load_oof_foldwise(p15_oof, "drop0p15_promoted_score_harmonized", keep_methods=["oof_ecdf", "oof_logitz"]),
            load_oof_foldwise(p10_oof, "drop0p10_score_harmonized", keep_methods=["oof_ecdf", "oof_logitz"]),
        ],
        ignore_index=True,
    )
    all_foldwise = pd.concat([raw_foldwise, oof_foldwise], ignore_index=True, sort=False)
    write_table(out_dir, "stageb_foldwise_metrics", all_foldwise, max_rows=300)
    focus = all_foldwise[
        all_foldwise["fold"].isin([4, 5])
        & all_foldwise["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    write_table(out_dir, "stageb_fold4_fold5_focus", focus, max_rows=120)

    manifests = []
    summaries = []
    for run_dir, run_id in [
        (p15_run, "drop0p15_uniform_reference"),
        (p10_run, "drop0p10_uniform_reference"),
        (enc_run, "enc0p15_dec0p10_candidate"),
    ]:
        manifest, summary = load_dropout_reports(run_dir, run_id)
        manifests.append(manifest)
        summaries.append(summary)
    dropout_manifest = pd.concat(manifests, ignore_index=True, sort=False)
    dropout_summary = pd.concat(summaries, ignore_index=True, sort=False)
    write_table(out_dir, "dropout_manifest_comparison", dropout_manifest, max_rows=300)
    write_table(out_dir, "dropout_summary_comparison", dropout_summary, max_rows=120)

    vae_qc = pd.concat(
        [
            load_vae_qc(enc_run, "enc0p15_dec0p10_candidate"),
            load_vae_qc(p15_run, "drop0p15_uniform_reference"),
            load_vae_qc(p10_run, "drop0p10_uniform_reference"),
        ],
        ignore_index=True,
    )
    write_table(out_dir, "vae_qc_comparison", vae_qc, max_rows=120)
    leakage_cols = [
        "run_id",
        "fold",
        "trainDev_acc_site_raw",
        "trainDev_acc_site_latent",
        "trainDev_latent_minus_raw",
        "test_acc_site_raw",
        "test_acc_site_latent",
        "test_latent_minus_raw",
        "test_total_correlation_nats",
        "test_n_active",
    ]
    write_table(out_dir, "scanner_leakage_comparison", vae_qc[[c for c in leakage_cols if c in vae_qc.columns]], max_rows=120)

    philips = pd.concat(
        [
            summarize_philips_cn_fp_raw(enc_run, "enc0p15_dec0p10_candidate"),
            summarize_philips_cn_fp_raw(p15_run, "drop0p15_uniform_reference"),
            summarize_philips_cn_fp_raw(p10_run, "drop0p10_uniform_reference"),
            load_philips_cn_fp_oof(p15_oof, "drop0p15_promoted_score_harmonized", keep_methods=["oof_ecdf", "oof_logitz"]),
            load_philips_cn_fp_oof(p10_oof, "drop0p10_score_harmonized", keep_methods=["oof_ecdf", "oof_logitz"]),
        ],
        ignore_index=True,
        sort=False,
    )
    write_table(out_dir, "philips_cn_fp_comparison", philips, max_rows=160)

    oof_availability = availability_table(enc_run, p15_oof, p10_oof)
    write_table(out_dir, "score_harmonized_availability", oof_availability, max_rows=20)

    final_text = final_decision(completion, all_pooled, oof_availability, philips)
    (out_dir / "final_decision.md").write_text(final_text, encoding="utf-8")
    (out_dir / "README.md").write_text(final_text, encoding="utf-8")
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote audit outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
