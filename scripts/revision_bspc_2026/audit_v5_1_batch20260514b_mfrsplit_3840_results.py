#!/usr/bin/env python3
"""Audit ADNI v5.1 batch20260514b manufacturer-aware 3840-epoch results.

Read-only audit. It parses completed run artifacts and writes derived tables
under results/revision_bspc_2026. It does not retrain, load/modify tensors, or
modify metadata/ledgers.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
BIG_RESULTS_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

RUN_SPECS = [
    {
        "run_id": "mfrsplit_3840",
        "label": "v5.1 batch20260514b manufacturer-aware split 3840 epochs",
        "local_dir": RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate",
        "big_dir": BIG_RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate",
        "is_main": True,
    },
    {
        "run_id": "baseline_2560",
        "label": "v5.1 batch20260514b current final-candidate baseline 2560 epochs",
        "local_dir": RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline",
        "big_dir": BIG_RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_final_candidate_baseline",
        "is_main": False,
    },
    {
        "run_id": "interim_20260514",
        "label": "v5.1 batch20260514 interim baseline 2560 epochs",
        "local_dir": RESULTS_ROOT / "adni_v5_1_batch20260514_ch1_0_2_interim_baseline",
        "big_dir": BIG_RESULTS_ROOT / "adni_v5_1_batch20260514_ch1_0_2_interim_baseline",
        "is_main": False,
    },
]

DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "adni_v5_1_batch20260514b_mfrsplit_3840_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit completed ADNI v5.1 batch20260514b mfrsplit 3840 final-candidate run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_run_dir(spec: Dict[str, Any]) -> Path:
    if Path(spec["big_dir"]).exists():
        return Path(spec["big_dir"])
    if Path(spec["local_dir"]).exists():
        return Path(spec["local_dir"])
    raise FileNotFoundError(f"Run not found: {spec['big_dir']} or {spec['local_dir']}")


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    generated = [
        "README.md",
        "foldwise_comparison_table.csv",
        "pooled_confusion_0p5.csv",
        "subgroup_metrics_by_manufacturer_and_sex.csv",
        "fold4_error_audit.csv",
        "run_comparison_summary.csv",
        "run_comparison_deltas.csv",
        "vae_best_epoch_audit.csv",
        "scanner_leakage_by_fold.csv",
        "run_manifest_summary.csv",
        "artifact_inventory.csv",
        "command_log.json",
    ]
    if path.exists() and any((path / name).exists() for name in generated):
        if not overwrite:
            raise FileExistsError(f"{path} already contains generated audit outputs; pass --overwrite")
        for name in generated:
            p = path / name
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_file(path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def normalize_classifier_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "classifier" not in out.columns:
        if "actual_classifier_type" in out.columns:
            out["classifier"] = out["actual_classifier_type"]
        elif "classifier_type" in out.columns:
            out["classifier"] = out["classifier_type"]
    return out


def normalize_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["SubjectID"] = out["SubjectID"].astype(str)
    defaults = {
        "ResearchGroup_Mapped": "UNKNOWN",
        "Diagnosis": "UNKNOWN",
        "Manufacturer": "UNKNOWN",
        "Sex": "UNKNOWN",
        "source_batch": "UNKNOWN",
        "source_label": "UNKNOWN",
        "tensor_source": "UNKNOWN",
        "included_in_dataset_version": "UNKNOWN",
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
        out[col] = out[col].fillna(val).astype(str)
    if "Age" not in out.columns:
        out["Age"] = np.nan
    else:
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    return out


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, float] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, len(y)),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "predicted_ad_rate": float(pred.mean()) if len(pred) else float("nan"),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def load_metadata_for_run(run_dir: Path) -> pd.DataFrame:
    cfg = read_json_optional(run_dir / "run_config.json") or {}
    args = cfg.get("args", {})
    metadata_path = cfg.get("metadata_path") or args.get("metadata_path")
    if metadata_path is None:
        manifest = read_json_optional(run_dir / "run_manifest.json") or {}
        metadata_path = manifest.get("metadata_path")
    if metadata_path is None:
        raise FileNotFoundError(f"Could not infer metadata path from {run_dir}")
    return normalize_metadata(pd.read_csv(metadata_path))


def load_metrics(run_dir: Path) -> pd.DataFrame:
    path = first_file(run_dir, "all_folds_metrics_MULTI*.csv")
    if path is None:
        raise FileNotFoundError(f"No all_folds_metrics_MULTI*.csv under {run_dir}")
    metrics = normalize_classifier_col(pd.read_csv(path))
    metrics["metrics_path"] = str(path)
    return metrics


def load_predictions(run_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    path = first_file(run_dir, "all_folds_clf_predictions_MULTI*.csv")
    if path is not None:
        pred = pd.read_csv(path)
    else:
        chunks: List[pd.DataFrame] = []
        for fold_dir in sorted(run_dir.glob("fold_*")):
            if not fold_dir.is_dir():
                continue
            try:
                fold = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            for pred_path in sorted(fold_dir.glob("test_predictions_*.csv")):
                chunk = pd.read_csv(pred_path)
                chunk["fold"] = fold
                chunk["classifier"] = pred_path.stem.replace("test_predictions_", "")
                chunks.append(chunk)
        if not chunks:
            raise FileNotFoundError(f"No prediction files under {run_dir}")
        pred = pd.concat(chunks, ignore_index=True)
    pred = normalize_classifier_col(pred)
    pred["SubjectID"] = pred["SubjectID"].astype(str)
    if "y_score_final" not in pred.columns:
        if "y_score_cal" in pred.columns:
            pred["y_score_final"] = pred["y_score_cal"]
        elif "y_score_raw" in pred.columns:
            pred["y_score_final"] = pred["y_score_raw"]
        else:
            raise RuntimeError(f"Prediction table lacks score columns in {run_dir}")
    pred["y_pred_0p5"] = (pred["y_score_final"].astype(float) >= 0.5).astype(int)
    if "y_pred" not in pred.columns:
        pred["y_pred"] = pred["y_pred_0p5"]
    keep_meta = [
        col
        for col in [
            "SubjectID",
            "ResearchGroup_Mapped",
            "Diagnosis",
            "Age",
            "Sex",
            "Manufacturer",
            "Site3",
            "ImageID",
            "Visit",
            "tensor_source",
            "source_label",
            "source_batch",
            "included_in_dataset_version",
        ]
        if col in metadata.columns
    ]
    pred = pred.merge(metadata[keep_meta].drop_duplicates("SubjectID"), on="SubjectID", how="left")
    pred["Manufacturer"] = pred["Manufacturer"].fillna("UNKNOWN").astype(str)
    pred["Sex"] = pred["Sex"].fillna("UNKNOWN").astype(str)
    pred["ResearchGroup_Mapped"] = pred["ResearchGroup_Mapped"].fillna(
        pred["y_true"].map({0: "CN", 1: "AD"}).fillna("UNKNOWN")
    )
    for col in ["source_batch", "source_label", "tensor_source"]:
        if col not in pred.columns:
            pred[col] = "UNKNOWN"
        pred[col] = pred[col].fillna("UNKNOWN").astype(str)
    pred["error_type_0p5"] = np.select(
        [
            pred["y_true"].eq(1) & pred["y_pred_0p5"].eq(0),
            pred["y_true"].eq(0) & pred["y_pred_0p5"].eq(1),
            pred["y_true"].eq(1) & pred["y_pred_0p5"].eq(1),
            pred["y_true"].eq(0) & pred["y_pred_0p5"].eq(0),
        ],
        ["false_negative", "false_positive", "true_positive", "true_negative"],
        default="unknown",
    )
    return pred


def load_vae_best_epochs(run_id: str, run_dir: Path) -> pd.DataFrame:
    cfg = read_json_optional(run_dir / "run_config.json") or {}
    args = cfg.get("args", {})
    max_epochs = int(args.get("epochs_vae", 0) or 0)
    patience = int(args.get("early_stopping_patience_vae", 0) or 0)
    rows: List[Dict[str, Any]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        path = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        row: Dict[str, Any] = {
            "run_id": run_id,
            "fold": fold,
            "history_path": str(path),
            "history_available": path.exists(),
            "configured_max_epochs": max_epochs,
            "early_stopping_patience": patience,
        }
        if path.exists():
            hist = joblib.load(path)
            metric = "val_loss_modelsel" if "val_loss_modelsel" in hist else ("val_loss" if "val_loss" in hist else "train_loss")
            values = np.asarray(hist.get(metric, []), dtype=float)
            valid = np.isfinite(values)
            if valid.any():
                valid_idx = np.where(valid)[0]
                best_zero = int(valid_idx[np.argmin(values[valid])])
                best_epoch = best_zero + 1
                best_value = float(values[best_zero])
            else:
                best_epoch = np.nan
                best_value = np.nan
            epochs_completed = int(len(values))
            final_value = float(values[-1]) if len(values) and np.isfinite(values[-1]) else np.nan
            row.update(
                {
                    "metric_used": metric,
                    "best_epoch": best_epoch,
                    "epochs_completed": epochs_completed,
                    "best_value": best_value,
                    "final_value": final_value,
                    "final_minus_best": final_value - best_value if np.isfinite(final_value) and np.isfinite(best_value) else np.nan,
                    "reached_configured_max_epochs": bool(max_epochs and epochs_completed >= max_epochs),
                    "early_stopped_before_configured_max": bool(max_epochs and epochs_completed < max_epochs),
                    "distance_best_to_final_epoch": epochs_completed - best_epoch if np.isfinite(best_epoch) else np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_scanner_leakage(run_id: str, run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        row: Dict[str, Any] = {"run_id": run_id, "fold": fold}
        for prefix, name in [
            ("train_dev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ("latent_qc", "latent_qc_metrics.csv"),
        ]:
            path = fold_dir / name
            row[f"{prefix}_path"] = str(path) if path.exists() else ""
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            d = df.iloc[0].to_dict()
            for key in [
                "site_col",
                "n_sites",
                "n_splits",
                "n_samples",
                "chance_level",
                "acc_site_raw",
                "acc_site_raw_std",
                "acc_site_latent",
                "acc_site_latent_std",
            ]:
                if key in d:
                    row[f"{prefix}_{key}"] = d[key]
        rows.append(row)
    return pd.DataFrame(rows)


def run_manifest_summary(run_id: str, run_dir: Path) -> Dict[str, Any]:
    cfg = read_json_optional(run_dir / "run_config.json") or {}
    manifest = read_json_optional(run_dir / "run_manifest.json") or {}
    args = cfg.get("args", {})
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_config_available": bool(cfg),
        "run_manifest_available": bool(manifest),
        "global_tensor_path": cfg.get("global_tensor_path") or args.get("global_tensor_path") or manifest.get("metadata_path"),
        "metadata_path": cfg.get("metadata_path") or args.get("metadata_path") or manifest.get("metadata_path"),
        "channels": cfg.get("channels_to_use_indices") or args.get("channels_to_use") or manifest.get("selected_channels"),
        "channel_names": cfg.get("channel_names_selected") or manifest.get("selected_channel_names"),
        "python_bandpass_applied": manifest.get("python_bandpass_applied", False),
        "epochs_vae": args.get("epochs_vae") or manifest.get("epochs_vae"),
        "cyclical_beta_n_cycles": args.get("cyclical_beta_n_cycles") or manifest.get("cyclical_beta_n_cycles"),
        "early_stopping_patience_vae": args.get("early_stopping_patience_vae"),
        "classifier_stratify_cols": args.get("classifier_stratify_cols"),
        "vae_stratify_cols": args.get("vae_stratify_cols"),
        "split_strategy": json.dumps(manifest.get("split_strategy"), sort_keys=True) if manifest.get("split_strategy") else "",
        "created_utc": cfg.get("created_utc") or manifest.get("created_utc"),
    }


def artifact_inventory(run_id: str, run_dir: Path) -> Dict[str, Any]:
    fold_dirs = [p for p in run_dir.glob("fold_*") if p.is_dir()]
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "fold_dirs": len(fold_dirs),
        "combined_metrics_files": len(list(run_dir.glob("all_folds_metrics_MULTI*.csv"))),
        "combined_prediction_files": len(list(run_dir.glob("all_folds_clf_predictions_MULTI*.csv"))),
        "per_fold_prediction_files": len(list(run_dir.glob("fold_*/test_predictions_*.csv"))),
        "vae_history_files": len(list(run_dir.glob("fold_*/vae_train_history_fold_*.joblib"))),
        "latent_qc_files": len(list(run_dir.glob("fold_*/latent_qc_metrics.csv"))),
        "scanner_leakage_summary_files": len(list(run_dir.glob("fold_*/fold_*_scanner_leakage_summary.csv"))),
        "test_scanner_leakage_summary_files": len(list(run_dir.glob("fold_*/fold_*_test_scanner_leakage_summary.csv"))),
        "run_config": (run_dir / "run_config.json").exists(),
        "run_manifest": (run_dir / "run_manifest.json").exists(),
    }


def make_foldwise_comparison(
    all_metrics: pd.DataFrame,
    vae_epochs: pd.DataFrame,
    scanner: pd.DataFrame,
) -> pd.DataFrame:
    metrics = all_metrics.copy()
    metrics["auc"] = metrics.get("auc_final", metrics.get("auc"))
    metrics["pr_auc"] = metrics.get("pr_auc_final", metrics.get("pr_auc"))
    metrics["f1"] = metrics.get("f1_score", metrics.get("f1", np.nan))
    cols = [
        "run_id",
        "run_label",
        "fold",
        "classifier",
        "auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]
    out = metrics[cols].copy()
    out = out.merge(
        vae_epochs[
            [
                "run_id",
                "fold",
                "best_epoch",
                "epochs_completed",
                "configured_max_epochs",
                "early_stopped_before_configured_max",
                "distance_best_to_final_epoch",
            ]
        ],
        on=["run_id", "fold"],
        how="left",
    )
    leak_cols = [
        "run_id",
        "fold",
        "train_dev_acc_site_raw",
        "train_dev_acc_site_latent",
        "train_dev_chance_level",
        "test_acc_site_raw",
        "test_acc_site_latent",
        "test_chance_level",
        "latent_qc_acc_site_raw",
        "latent_qc_acc_site_latent",
    ]
    leak_cols = [c for c in leak_cols if c in scanner.columns]
    out = out.merge(scanner[leak_cols], on=["run_id", "fold"], how="left")
    return out.sort_values(["run_id", "classifier", "fold"]).reset_index(drop=True)


def make_pooled_confusion(all_pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (run_id, label, clf), sub in all_pred.groupby(["run_id", "run_label", "classifier"], dropna=False):
        row = {
            "run_id": run_id,
            "run_label": label,
            "classifier": clf,
            "threshold": 0.5,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score_final"], sub["y_pred_0p5"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["run_id", "classifier"]).reset_index(drop=True)


def make_subgroup_metrics(all_pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for grouping, col in [("Manufacturer", "Manufacturer"), ("Sex", "Sex")]:
        for (run_id, label, clf, value), sub in all_pred.groupby(["run_id", "run_label", "classifier", col], dropna=False):
            row = {
                "run_id": run_id,
                "run_label": label,
                "classifier": clf,
                "grouping": grouping,
                "group_value": value,
                "threshold": 0.5,
                "score_mean": float(sub["y_score_final"].mean()),
                "score_median": float(sub["y_score_final"].median()),
            }
            row.update(binary_metrics(sub["y_true"], sub["y_score_final"], sub["y_pred_0p5"]))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["run_id", "classifier", "grouping", "group_value"]).reset_index(drop=True)


def make_fold4_error_audit(main_pred: pd.DataFrame) -> pd.DataFrame:
    fold4 = main_pred[main_pred["fold"].eq(4)].copy()
    rows: List[Dict[str, Any]] = []
    for clf, sub in fold4.groupby("classifier", dropna=False):
        errors = sub[sub["error_type_0p5"].isin(["false_negative", "false_positive"])].copy()
        errors = errors.sort_values(["error_type_0p5", "y_score_final"], ascending=[True, True])
        for _, r in errors.iterrows():
            rows.append(
                {
                    "classifier": clf,
                    "fold": 4,
                    "error_type": r["error_type_0p5"],
                    "SubjectID": r.get("SubjectID"),
                    "y_true": int(r.get("y_true")),
                    "y_pred_0p5": int(r.get("y_pred_0p5")),
                    "y_score_final": float(r.get("y_score_final")),
                    "y_score_raw": safe_float(r.get("y_score_raw")),
                    "y_score_cal": safe_float(r.get("y_score_cal")),
                    "ResearchGroup_Mapped": r.get("ResearchGroup_Mapped"),
                    "Manufacturer": r.get("Manufacturer"),
                    "Age": safe_float(r.get("Age")),
                    "Sex": r.get("Sex"),
                    "source_batch": r.get("source_batch"),
                    "source_label": r.get("source_label"),
                    "tensor_source": r.get("tensor_source"),
                    "ImageID": r.get("ImageID", ""),
                    "Visit": r.get("Visit", ""),
                }
            )
    return pd.DataFrame(rows)


def make_run_summary(pooled: pd.DataFrame, foldwise: pd.DataFrame, scanner: pd.DataFrame, vae: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_id, sub in pooled.groupby("run_id", dropna=False):
        row: Dict[str, Any] = {"run_id": run_id}
        row["run_label"] = sub["run_label"].iloc[0]
        for _, r in sub.iterrows():
            clf = r["classifier"]
            for metric in ["auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "tp", "fn", "tn", "fp"]:
                row[f"{clf}_pooled_{metric}"] = r.get(metric)
        fw = foldwise[foldwise["run_id"].eq(run_id)]
        for clf, clf_df in fw.groupby("classifier", dropna=False):
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                row[f"{clf}_{metric}_mean"] = float(clf_df[metric].mean())
                row[f"{clf}_{metric}_std"] = float(clf_df[metric].std(ddof=1))
                row[f"{clf}_{metric}_min"] = float(clf_df[metric].min())
        sc = scanner[scanner["run_id"].eq(run_id)]
        for col in ["test_acc_site_raw", "test_acc_site_latent", "train_dev_acc_site_raw", "train_dev_acc_site_latent"]:
            if col in sc.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(sc[col], errors="coerce").mean())
        ve = vae[vae["run_id"].eq(run_id)]
        if not ve.empty:
            row["best_epoch_mean"] = float(pd.to_numeric(ve["best_epoch"], errors="coerce").mean())
            row["epochs_completed_mean"] = float(pd.to_numeric(ve["epochs_completed"], errors="coerce").mean())
            row["early_stopped_folds"] = int(ve.get("early_stopped_before_configured_max", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def make_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    target = summary[summary["run_id"].eq("mfrsplit_3840")]
    if target.empty:
        return pd.DataFrame()
    t = target.iloc[0]
    rows: List[Dict[str, Any]] = []
    for ref in ["baseline_2560", "interim_20260514"]:
        ref_row = summary[summary["run_id"].eq(ref)]
        if ref_row.empty:
            continue
        r = ref_row.iloc[0]
        for clf in ["logreg", "svm"]:
            for metric in ["auc_mean", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "pooled_auc", "pooled_sensitivity", "pooled_specificity", "pooled_balanced_accuracy"]:
                col = f"{clf}_{metric}"
                if col not in summary.columns or pd.isna(t.get(col)) or pd.isna(r.get(col)):
                    continue
                rows.append(
                    {
                        "target_run": "mfrsplit_3840",
                        "reference_run": ref,
                        "classifier": clf,
                        "metric": metric,
                        "target_value": float(t[col]),
                        "reference_value": float(r[col]),
                        "delta_target_minus_reference": float(t[col] - r[col]),
                    }
                )
    return pd.DataFrame(rows)


def classify_improvement(delta: float, eps: float = 0.01) -> str:
    if not np.isfinite(delta):
        return "unknown"
    if delta > eps:
        return "improved"
    if delta < -eps:
        return "worse"
    return "similar"


def fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
        if math.isnan(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


def readme_table(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    if df.empty:
        return ["No rows."]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        vals: List[str] = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(fmt(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def make_readme(
    outdir: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    foldwise: pd.DataFrame,
    pooled: pd.DataFrame,
    fold4_errors: pd.DataFrame,
    scanner: pd.DataFrame,
    artifact_df: pd.DataFrame,
) -> None:
    main = summary[summary["run_id"].eq("mfrsplit_3840")]
    baseline = summary[summary["run_id"].eq("baseline_2560")]
    main_row = main.iloc[0] if not main.empty else pd.Series(dtype=object)
    baseline_row = baseline.iloc[0] if not baseline.empty else pd.Series(dtype=object)

    logreg_auc_delta = safe_float(main_row.get("logreg_auc_mean")) - safe_float(baseline_row.get("logreg_auc_mean"))
    logreg_ba_delta = safe_float(main_row.get("logreg_balanced_accuracy_mean")) - safe_float(baseline_row.get("logreg_balanced_accuracy_mean"))
    logreg_sens_delta = safe_float(main_row.get("logreg_sensitivity_mean")) - safe_float(baseline_row.get("logreg_sensitivity_mean"))
    svm_auc_delta = safe_float(main_row.get("svm_auc_mean")) - safe_float(baseline_row.get("svm_auc_mean"))
    svm_ba_delta = safe_float(main_row.get("svm_balanced_accuracy_mean")) - safe_float(baseline_row.get("svm_balanced_accuracy_mean"))
    svm_sens_delta = safe_float(main_row.get("svm_sensitivity_mean")) - safe_float(baseline_row.get("svm_sensitivity_mean"))
    auc_call = classify_improvement(np.nanmean([logreg_auc_delta, svm_auc_delta]))
    operating_call = classify_improvement(np.nanmean([logreg_ba_delta, svm_ba_delta, logreg_sens_delta, svm_sens_delta]))
    if auc_call == "improved" and operating_call in {"worse", "similar"}:
        robustness_call = "mixed_not_clear_improvement"
    elif auc_call == operating_call:
        robustness_call = auc_call
    else:
        robustness_call = "mixed"

    fw_main = foldwise[foldwise["run_id"].eq("mfrsplit_3840")]
    fold4_main = fw_main[fw_main["fold"].eq(4)]
    other_main = fw_main[~fw_main["fold"].eq(4)]
    fold4_is_dominant_threshold = False
    fold4_is_dominant_ranking = False
    if not fold4_main.empty and not other_main.empty:
        fold4_is_dominant_threshold = bool(
            fold4_main.groupby("classifier")["sensitivity"].mean().mean()
            < other_main.groupby("classifier")["sensitivity"].mean().mean() - 0.10
        )
        fold4_is_dominant_ranking = bool(
            fold4_main.groupby("classifier")["auc"].mean().mean()
            <= fw_main.groupby("fold")["auc"].mean().min() + 1e-12
        )
    ranking_vs_threshold = "threshold/sensitivity" if safe_float(main_row.get("logreg_pooled_auc")) >= 0.70 and safe_float(main_row.get("logreg_pooled_sensitivity")) < 0.50 else "ranking"
    sc_main = scanner[scanner["run_id"].eq("mfrsplit_3840")]
    latent_leak = float(pd.to_numeric(sc_main.get("test_acc_site_latent", pd.Series(dtype=float)), errors="coerce").mean()) if not sc_main.empty else np.nan
    raw_leak = float(pd.to_numeric(sc_main.get("test_acc_site_raw", pd.Series(dtype=float)), errors="coerce").mean()) if not sc_main.empty else np.nan
    leakage_call = "present" if np.isfinite(latent_leak) and latent_leak > 0.50 else "not clearly elevated"

    pooled_cols = [
        "run_id",
        "classifier",
        "auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "tp",
        "fn",
        "tn",
        "fp",
    ]
    pooled_short = pooled[pooled["run_id"].isin(["mfrsplit_3840", "baseline_2560", "interim_20260514"])][pooled_cols]

    lines = [
        "# ADNI v5.1 batch20260514b Manufacturer-Aware 3840 Audit",
        "",
        "## Scope",
        "",
        "- Main run: `adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate`.",
        "- Comparators: current `batch20260514b` 2560 baseline and `batch20260514` interim baseline.",
        "- Read-only audit: no training, no tensor changes, no metadata changes, no ledger changes.",
        "",
        "## Pooled Threshold 0.5 Results",
        "",
        *readme_table(pooled_short, pooled_cols),
        "",
        "## Scientific Interpretation",
        "",
        f"- Manufacturer-aware split robustness call: `{robustness_call}` versus the 2560 baseline. AUC improved modestly (LogReg delta={fmt(logreg_auc_delta)}, SVM delta={fmt(svm_auc_delta)}), but threshold-0.5 operating metrics worsened (LogReg BA delta={fmt(logreg_ba_delta)}, SVM BA delta={fmt(svm_ba_delta)}, LogReg sensitivity delta={fmt(logreg_sens_delta)}, SVM sensitivity delta={fmt(svm_sens_delta)}).",
        f"- Fold 4 dominant ranking failure: `{fold4_is_dominant_ranking}`. Fold 4 dominant threshold/sensitivity failure: `{fold4_is_dominant_threshold}`. Fold 4 error rows at threshold 0.5: `{len(fold4_errors)}`.",
        f"- Remaining problem is best characterized as `{ranking_vs_threshold}`: pooled AUC remains materially above chance, but threshold 0.5 sensitivity is low.",
        f"- Scanner/manufacturer leakage evidence: `{leakage_call}`. Mean test scanner/manufacturer balanced accuracy raw={fmt(raw_leak)}, latent={fmt(latent_leak)}; chance is about 0.333 for three manufacturers.",
        "",
        "## Artifact Coverage",
        "",
        *readme_table(
            artifact_df,
            [
                "run_id",
                "fold_dirs",
                "combined_metrics_files",
                "combined_prediction_files",
                "vae_history_files",
                "latent_qc_files",
                "scanner_leakage_summary_files",
                "test_scanner_leakage_summary_files",
            ],
        ),
        "",
        "## Outputs",
        "",
        "- `foldwise_comparison_table.csv`",
        "- `pooled_confusion_0p5.csv`",
        "- `subgroup_metrics_by_manufacturer_and_sex.csv`",
        "- `fold4_error_audit.csv`",
        "- `run_comparison_summary.csv`",
        "- `run_comparison_deltas.csv`",
        "- `vae_best_epoch_audit.csv`",
        "- `scanner_leakage_by_fold.csv`",
        "- `run_manifest_summary.csv`",
        "- `artifact_inventory.csv`",
        "- `command_log.json`",
    ]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    all_metrics: List[pd.DataFrame] = []
    all_pred: List[pd.DataFrame] = []
    all_vae: List[pd.DataFrame] = []
    all_scanner: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []
    run_dirs: Dict[str, str] = {}

    for spec in RUN_SPECS:
        run_id = spec["run_id"]
        run_dir = resolve_run_dir(spec)
        run_dirs[run_id] = str(run_dir)
        metadata = load_metadata_for_run(run_dir)
        metrics = load_metrics(run_dir)
        metrics["run_id"] = run_id
        metrics["run_label"] = spec["label"]
        all_metrics.append(metrics)

        pred = load_predictions(run_dir, metadata)
        pred["run_id"] = run_id
        pred["run_label"] = spec["label"]
        all_pred.append(pred)

        all_vae.append(load_vae_best_epochs(run_id, run_dir))
        all_scanner.append(load_scanner_leakage(run_id, run_dir))
        manifest_rows.append(run_manifest_summary(run_id, run_dir))
        artifact_rows.append(artifact_inventory(run_id, run_dir))

    metrics_df = pd.concat(all_metrics, ignore_index=True, sort=False)
    pred_df = pd.concat(all_pred, ignore_index=True, sort=False)
    vae_df = pd.concat(all_vae, ignore_index=True, sort=False)
    scanner_df = pd.concat(all_scanner, ignore_index=True, sort=False)
    manifest_df = pd.DataFrame(manifest_rows)
    artifact_df = pd.DataFrame(artifact_rows)

    foldwise = make_foldwise_comparison(metrics_df, vae_df, scanner_df)
    pooled = make_pooled_confusion(pred_df)
    subgroup = make_subgroup_metrics(pred_df)
    main_pred = pred_df[pred_df["run_id"].eq("mfrsplit_3840")].copy()
    fold4_errors = make_fold4_error_audit(main_pred)
    summary = make_run_summary(pooled, foldwise, scanner_df, vae_df)
    deltas = make_deltas(summary)

    foldwise.to_csv(outdir / "foldwise_comparison_table.csv", index=False)
    pooled.to_csv(outdir / "pooled_confusion_0p5.csv", index=False)
    subgroup.to_csv(outdir / "subgroup_metrics_by_manufacturer_and_sex.csv", index=False)
    fold4_errors.to_csv(outdir / "fold4_error_audit.csv", index=False)
    summary.to_csv(outdir / "run_comparison_summary.csv", index=False)
    deltas.to_csv(outdir / "run_comparison_deltas.csv", index=False)
    vae_df.to_csv(outdir / "vae_best_epoch_audit.csv", index=False)
    scanner_df.to_csv(outdir / "scanner_leakage_by_fold.csv", index=False)
    manifest_df.to_csv(outdir / "run_manifest_summary.csv", index=False)
    artifact_df.to_csv(outdir / "artifact_inventory.csv", index=False)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_dir": str(outdir),
        "run_dirs": run_dirs,
        "training_run": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "n_prediction_rows_parsed": int(len(pred_df)),
        "n_metric_rows_parsed": int(len(metrics_df)),
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_readme(outdir, summary, deltas, foldwise, pooled, fold4_errors, scanner_df, artifact_df)

    print(f"output_dir={outdir}")
    print("training_run=False")
    print("tensor_modified=False")
    print("metadata_modified=False")
    print("ledger_modified=False")
    print("pooled threshold 0.5:")
    print(
        pooled[
            [
                "run_id",
                "classifier",
                "auc",
                "pr_auc",
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "tp",
                "fn",
                "tn",
                "fp",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
