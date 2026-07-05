#!/usr/bin/env python3
"""Read-only comparison of V4 full-config channel runs.

This script reads only small CSV/TXT/JSON/log artifacts. It does not load
tensors, checkpoints, joblibs, or large arrays, and it does not retrain.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/v4_channel_fullrun_comparison"
V4_METADATA = (
    PROJECT_ROOT
    / "data/revision_bspc_2026/adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    channels: Tuple[int, ...]
    channel_names: Tuple[str, ...]
    run_dir: Path

    @property
    def channels_indices(self) -> str:
        return "[" + ",".join(str(c) for c in self.channels) + "]"

    @property
    def channels_pretty(self) -> str:
        return " + ".join(f"{idx}:{name}" for idx, name in zip(self.channels, self.channel_names))


RUNS = [
    RunSpec(
        run_name="v4_static3_ch1_0_2",
        channels=(1, 0, 2),
        channel_names=(
            "Pearson_Full_FisherZ_Signed",
            "Pearson_OMST_GCE_Signed_Weighted",
            "MI_KNN_Symmetric",
        ),
        run_dir=PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_static3",
    ),
    RunSpec(
        run_name="v4_ch4_1",
        channels=(4, 1),
        channel_names=("dFC_StdDev", "Pearson_Full_FisherZ_Signed"),
        run_dir=PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1",
    ),
    RunSpec(
        run_name="v4_ch4_1_0",
        channels=(4, 1, 0),
        channel_names=(
            "dFC_StdDev",
            "Pearson_Full_FisherZ_Signed",
            "Pearson_OMST_GCE_Signed_Weighted",
        ),
        run_dir=PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0",
    ),
]

OUTPUT_FILES = [
    "model_comparison_v4_channel_fullruns.csv",
    "scanner_latent_comparison_v4_channel_fullruns.csv",
    "latent_info_comparison_v4_channel_fullruns.csv",
    "threshold_comparison_v4_channel_fullruns.csv",
    "manufacturer_error_comparison_v4_channel_fullruns.csv",
    "runtime_comparison_v4_channel_fullruns.csv",
    "README.md",
]

RUNTIME_REFRESH_FILES = {
    "model_comparison_v4_channel_fullruns.csv",
    "runtime_comparison_v4_channel_fullruns.csv",
    "README.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only comparison of V4 channel full-runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.resolve() if output_dir.exists() else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing comparison outputs. Use --overwrite:\n"
            + "\n".join(str(p) for p in existing)
        )
    return output_dir


def write_csv_refresh(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite and path.name in RUNTIME_REFRESH_FILES:
        raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {path}")
    if path.exists() and overwrite and path.name not in RUNTIME_REFRESH_FILES:
        return
    df.to_csv(path, index=False)


def write_text_refresh(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite and path.name in RUNTIME_REFRESH_FILES:
        raise FileExistsError(f"Refusing to overwrite existing output without --overwrite: {path}")
    if path.exists() and overwrite and path.name not in RUNTIME_REFRESH_FILES:
        return
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def lower_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in columns}


def pick_column(df: pd.DataFrame, preferred: Sequence[str], fuzzy: Sequence[str] = ()) -> Optional[str]:
    by_lower = lower_map(df.columns)
    for name in preferred:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    if fuzzy:
        matches = []
        for col in df.columns:
            col_l = str(col).lower()
            if all(token in col_l for token in fuzzy):
                matches.append(col)
        if len(matches) == 1:
            return matches[0]
    return None


def to_binary(series: pd.Series) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
            return int(value)
        text = str(value).strip().upper()
        if text in {"CN", "CONTROL", "NORMAL", "0", "0.0"}:
            return 0
        if text in {"AD", "DEMENTIA", "ALZHEIMER", "1", "1.0"}:
            return 1
        return np.nan

    out = series.map(convert)
    return out.astype("Int64")


def score_is_probability(values: pd.Series) -> bool:
    scores = pd.to_numeric(values, errors="coerce").dropna()
    return bool(len(scores) and scores.between(0.0, 1.0).all())


def safe_div(num: float, den: float) -> float:
    return np.nan if den == 0 else num / den


def run_realpath(spec: RunSpec) -> Path:
    return spec.run_dir.resolve()


def load_run_config(spec: RunSpec) -> Dict:
    return read_json(run_realpath(spec) / "run_config.json")


def metadata_path_for_run(spec: RunSpec) -> Path:
    cfg = load_run_config(spec)
    path = cfg.get("args", {}).get("metadata_path")
    if path:
        return Path(path)
    return V4_METADATA


def load_metadata(spec: RunSpec) -> pd.DataFrame:
    path = metadata_path_for_run(spec)
    if not path.exists():
        return pd.DataFrame(columns=["SubjectID", "Manufacturer"])
    meta = pd.read_csv(path)
    if "SubjectID" not in meta.columns:
        return pd.DataFrame(columns=["SubjectID", "Manufacturer"])
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    if "Manufacturer" not in meta.columns:
        meta["Manufacturer"] = np.nan
    return meta.drop_duplicates("SubjectID", keep="first")


def load_fold_predictions(spec: RunSpec) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    frames: List[pd.DataFrame] = []
    root = run_realpath(spec)
    for fold_dir in sorted(root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold_match = re.search(r"fold_(\d+)", fold_dir.name)
        fold = int(fold_match.group(1)) if fold_match else np.nan
        for path in sorted(fold_dir.glob("test_predictions_*.csv")):
            raw = safe_read_csv(path)
            if raw.empty:
                warnings.append(f"{spec.run_name}: empty/unreadable prediction file {path}")
                continue
            classifier = path.stem.replace("test_predictions_", "")
            subject_col = pick_column(raw, ["SubjectID", "subject_id", "PTID"], ("subject",))
            y_true_col = pick_column(raw, ["y_true", "label", "true_label"], ("true",))
            y_score_col = pick_column(
                raw,
                ["y_score_final", "y_score", "y_proba", "proba_ad", "prob_ad", "y_score_cal", "y_score_raw"],
                ("score",),
            )
            y_pred_col = pick_column(raw, ["y_pred", "pred", "predicted_label"], ("pred",))
            if not all([subject_col, y_true_col, y_score_col, y_pred_col]):
                warnings.append(f"{spec.run_name}: could not detect prediction columns in {path}: {list(raw.columns)}")
                continue
            frame = pd.DataFrame(
                {
                    "run_name": spec.run_name,
                    "fold": fold,
                    "classifier": classifier,
                    "SubjectID": raw[subject_col].astype(str).str.strip(),
                    "y_true": to_binary(raw[y_true_col]),
                    "y_score": pd.to_numeric(raw[y_score_col], errors="coerce"),
                    "y_pred_saved": to_binary(raw[y_pred_col]),
                    "prediction_source": str(path),
                }
            )
            frame = frame.dropna(subset=["y_true", "y_score"])
            frame["y_true"] = frame["y_true"].astype(int)
            frames.append(frame)
    if not frames:
        warnings.append(f"{spec.run_name}: no fold test prediction files found")
        return pd.DataFrame(), warnings
    return pd.concat(frames, ignore_index=True), warnings


def load_predictions_with_metadata(spec: RunSpec) -> Tuple[pd.DataFrame, List[str]]:
    audit_path = run_realpath(spec) / "audit_v4/pooled_test_predictions_with_metadata.csv"
    if audit_path.exists():
        audit = safe_read_csv(audit_path)
        if not audit.empty and {"SubjectID", "classifier", "y_true", "y_score"}.issubset(audit.columns):
            audit = audit.copy()
            audit["run_name"] = spec.run_name
            if "Manufacturer" not in audit.columns:
                audit["Manufacturer"] = np.nan
            audit["y_true"] = to_binary(audit["y_true"]).astype(int)
            audit["y_score"] = pd.to_numeric(audit["y_score"], errors="coerce")
            return audit, []
    preds, warnings = load_fold_predictions(spec)
    if preds.empty:
        return preds, warnings
    meta = load_metadata(spec)
    merged = preds.merge(meta[["SubjectID", "Manufacturer"]], on="SubjectID", how="left", validate="many_to_one")
    missing = int(merged["Manufacturer"].isna().sum()) if "Manufacturer" in merged.columns else len(merged)
    if missing:
        warnings.append(f"{spec.run_name}: Manufacturer missing for {missing} prediction rows after metadata merge")
    return merged, warnings


def compute_core_metrics(group: pd.DataFrame, threshold: float = 0.5, require_both: bool = False) -> Dict[str, float]:
    if group.empty:
        return {}
    y_true = group["y_true"].astype(int).to_numpy()
    y_score = pd.to_numeric(group["y_score"], errors="coerce").to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    has_both = n_cn > 0 and n_ad > 0
    mean_cn = float(np.mean(y_score[y_true == 0])) if n_cn else np.nan
    mean_ad = float(np.mean(y_score[y_true == 1])) if n_ad else np.nan
    out: Dict[str, float] = {
        "n_total": int(len(group)),
        "n_CN": n_cn,
        "n_AD": n_ad,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if has_both else np.nan,
        "sensitivity_AD": recall_score(y_true, y_pred, pos_label=1, zero_division=0) if has_both else np.nan,
        "specificity_CN": recall_score(y_true, y_pred, pos_label=0, zero_division=0) if has_both else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "roc_auc": roc_auc_score(y_true, y_score) if has_both else np.nan,
        "pr_auc": average_precision_score(y_true, y_score) if has_both else np.nan,
        "brier": brier_score_loss(y_true, y_score) if score_is_probability(group["y_score"]) and len(y_true) else np.nan,
        "mean_score_CN": mean_cn,
        "mean_score_AD": mean_ad,
        "score_separation_AD_minus_CN": mean_ad - mean_cn if n_cn and n_ad else np.nan,
        "false_positive_rate_CN": safe_div(fp, fp + tn),
        "false_negative_rate_AD": safe_div(fn, fn + tp),
    }
    if require_both and not has_both:
        for key in ["roc_auc", "balanced_accuracy", "sensitivity_AD", "specificity_CN"]:
            out[key] = np.nan
    return out


def load_fold_metric_summary(spec: RunSpec) -> pd.DataFrame:
    frames = []
    for path in sorted(run_realpath(spec).glob("all_folds_metrics_MULTI_*.csv")):
        df = safe_read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    clf_col = "actual_classifier_type" if "actual_classifier_type" in metrics.columns else "classifier"
    if clf_col not in metrics.columns:
        return pd.DataFrame()
    rows = []
    for classifier, group in metrics.groupby(clf_col, dropna=False, sort=True):
        row = {"classifier": str(classifier)}
        for col, prefix in [
            ("auc", "auc"),
            ("pr_auc", "pr_auc"),
            ("accuracy", "accuracy"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("sensitivity", "sensitivity_AD"),
            ("specificity", "specificity_CN"),
            ("f1_score", "f1"),
        ]:
            values = pd.to_numeric(group[col], errors="coerce") if col in group.columns else pd.Series(dtype=float)
            row[f"{prefix}_mean_fold"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{prefix}_std_fold"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0 if values.notna().sum() == 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def effective_n_iter_by_classifier(spec: RunSpec) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for classifier in ["logreg", "svm"]:
        values = []
        for path in sorted(run_realpath(spec).glob(f"fold_*/optuna_best_trial_{classifier}_fold_*.json")):
            payload = read_json(path)
            value = payload.get("effective_n_trials", payload.get("n_trials"))
            if value is not None:
                values.append(value)
        unique = sorted(set(values))
        if len(unique) == 1:
            result[classifier] = unique[0]
        elif unique:
            result[classifier] = ";".join(str(v) for v in unique)
        else:
            cfg = load_run_config(spec)
            arg_value = cfg.get("args", {}).get(f"n_iter_{classifier}")
            result[classifier] = arg_value if arg_value is not None else np.nan
    return result


def discover_log_paths(spec: RunSpec) -> List[Path]:
    candidates: List[Path] = []
    for base in [spec.run_dir, run_realpath(spec)]:
        candidates.extend(sorted((base / "Logs").glob("*.log")))

    external_log_dir = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/run_logs")
    if spec.run_name == "v4_ch4_1_0":
        candidates.extend(sorted(external_log_dir.glob("*ch4_1_0*.log")))
        candidates.extend(sorted(external_log_dir.glob("*v4_ch4_1_0*.log")))

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        try:
            key = path.resolve()
        except Exception:
            key = path
        if key in seen or not path.exists():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def parse_logs(spec: RunSpec) -> pd.DataFrame:
    rows = []
    log_paths = discover_log_paths(spec)
    if not log_paths:
        return pd.DataFrame(
            [
                {
                    "run_name": spec.run_name,
                    "fold": np.nan,
                    "fold_runtime_sec": np.nan,
                    "vae_best_epoch": np.nan,
                    "vae_early_stop_epoch": np.nan,
                    "total_pipeline_sec": np.nan,
                    "log_path": "",
                    "note": "No log files found in run_dir/Logs, realpath(run_dir)/Logs, or configured external run_logs patterns; detailed timing not available.",
                }
            ]
        )
    for log_path in log_paths:
        current_fold: Optional[int] = None
        fold_state: Dict[int, Dict[str, object]] = {}
        total_pipeline_sec = np.nan
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"Iniciando Fold\s+(\d+)/", line)
            if m:
                current_fold = int(m.group(1))
                fold_state.setdefault(current_fold, {})
            m = re.search(r"Starting Fold\s+(\d+)/", line, flags=re.IGNORECASE)
            if m:
                current_fold = int(m.group(1))
                fold_state.setdefault(current_fold, {})
            m = re.search(r"Early stopping VAE en epoch\s+(\d+).*?\(época\s+(\d+)\)", line)
            if m and current_fold is not None:
                fold_state.setdefault(current_fold, {})["vae_early_stop_epoch"] = int(m.group(1))
                fold_state.setdefault(current_fold, {})["vae_best_epoch"] = int(m.group(2))
            m = re.search(r"early stopping.*?epoch\s+(\d+).*?(?:best.*?epoch\s+(\d+))?", line, flags=re.IGNORECASE)
            if m and current_fold is not None:
                fold_state.setdefault(current_fold, {})["vae_early_stop_epoch"] = int(m.group(1))
                if m.group(2):
                    fold_state.setdefault(current_fold, {})["vae_best_epoch"] = int(m.group(2))
            m = re.search(r"completado en\s+([0-9.]+)\s+segundos", line)
            if m and current_fold is not None:
                fold_state.setdefault(current_fold, {})["fold_runtime_sec"] = float(m.group(1))
            m = re.search(r"completed in\s+([0-9.]+)\s*s\.?", line, flags=re.IGNORECASE)
            if m and current_fold is not None:
                fold_state.setdefault(current_fold, {})["fold_runtime_sec"] = float(m.group(1))
            m = re.search(r"Pipeline completo en\s+([0-9.]+)\s+segundos", line)
            if m:
                total_pipeline_sec = float(m.group(1))
            m = re.search(r"Pipeline completed in\s+([0-9.]+)\s*s\.?", line, flags=re.IGNORECASE)
            if m:
                total_pipeline_sec = float(m.group(1))
        for fold, state in sorted(fold_state.items()):
            rows.append(
                {
                    "run_name": spec.run_name,
                    "fold": fold,
                    "fold_runtime_sec": state.get("fold_runtime_sec", np.nan),
                    "vae_best_epoch": state.get("vae_best_epoch", np.nan),
                    "vae_early_stop_epoch": state.get("vae_early_stop_epoch", np.nan),
                    "total_pipeline_sec": np.nan,
                    "log_path": str(log_path),
                    "note": "Parsed from log." if state else "Fold found in log but detailed stage timers were not available.",
                }
            )
        rows.append(
            {
                "run_name": spec.run_name,
                "fold": "total",
                "fold_runtime_sec": np.nan,
                "vae_best_epoch": np.nan,
                "vae_early_stop_epoch": np.nan,
                "total_pipeline_sec": total_pipeline_sec,
                "log_path": str(log_path),
                "note": "Parsed total pipeline runtime from log."
                if pd.notna(total_pipeline_sec)
                else "Total pipeline runtime not parseable from log.",
            }
        )
    return pd.DataFrame(rows)


def runtime_summary(runtime_df: pd.DataFrame) -> Dict[str, float]:
    if runtime_df.empty:
        return {"runtime_total_sec": np.nan, "runtime_total_min": np.nan, "mean_fold_runtime_sec": np.nan}
    total = pd.to_numeric(runtime_df["total_pipeline_sec"], errors="coerce").dropna()
    fold_rt = pd.to_numeric(runtime_df["fold_runtime_sec"], errors="coerce").dropna()
    runtime_total_sec = float(total.iloc[-1]) if len(total) else np.nan
    return {
        "runtime_total_sec": runtime_total_sec,
        "runtime_total_min": runtime_total_sec / 60.0 if pd.notna(runtime_total_sec) else np.nan,
        "mean_fold_runtime_sec": float(fold_rt.mean()) if len(fold_rt) else np.nan,
    }


def build_model_comparison(run_data: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        data = run_data[spec.run_name]
        preds = data["predictions"]
        fold_summary = data["fold_metric_summary"]
        n_iters = data["effective_n_iter"]
        rt = data["runtime_summary"]
        status_base = "ok" if not preds.empty else "missing_predictions"
        for classifier in sorted(set(preds["classifier"].astype(str))) if not preds.empty else ["logreg", "svm"]:
            group = preds[preds["classifier"].astype(str) == classifier] if not preds.empty else pd.DataFrame()
            pooled = compute_core_metrics(group)
            fold_row = (
                fold_summary[fold_summary["classifier"].astype(str) == classifier].iloc[0].to_dict()
                if not fold_summary.empty and not fold_summary[fold_summary["classifier"].astype(str) == classifier].empty
                else {}
            )
            rows.append(
                {
                    "run_name": spec.run_name,
                    "channels_indices": spec.channels_indices,
                    "channels_pretty": spec.channels_pretty,
                    "classifier": classifier,
                    "n_total": pooled.get("n_total", np.nan),
                    "n_CN": pooled.get("n_CN", np.nan),
                    "n_AD": pooled.get("n_AD", np.nan),
                    "auc_mean_fold": fold_row.get("auc_mean_fold", np.nan),
                    "auc_std_fold": fold_row.get("auc_std_fold", np.nan),
                    "auc_pooled": pooled.get("roc_auc", np.nan),
                    "pr_auc_mean_fold": fold_row.get("pr_auc_mean_fold", np.nan),
                    "pr_auc_std_fold": fold_row.get("pr_auc_std_fold", np.nan),
                    "pr_auc_pooled": pooled.get("pr_auc", np.nan),
                    "accuracy_mean_fold": fold_row.get("accuracy_mean_fold", np.nan),
                    "balanced_accuracy_mean_fold": fold_row.get("balanced_accuracy_mean_fold", np.nan),
                    "sensitivity_AD_mean_fold": fold_row.get("sensitivity_AD_mean_fold", np.nan),
                    "specificity_CN_mean_fold": fold_row.get("specificity_CN_mean_fold", np.nan),
                    "f1_mean_fold": fold_row.get("f1_mean_fold", np.nan),
                    "brier_pooled": pooled.get("brier", np.nan),
                    "mean_score_CN": pooled.get("mean_score_CN", np.nan),
                    "mean_score_AD": pooled.get("mean_score_AD", np.nan),
                    "score_separation_AD_minus_CN": pooled.get("score_separation_AD_minus_CN", np.nan),
                    "threshold_0p5_TN": pooled.get("tn", np.nan),
                    "threshold_0p5_FP": pooled.get("fp", np.nan),
                    "threshold_0p5_FN": pooled.get("fn", np.nan),
                    "threshold_0p5_TP": pooled.get("tp", np.nan),
                    "n_iter_logreg_effective": n_iters.get("logreg", np.nan),
                    "n_iter_svm_effective": n_iters.get("svm", np.nan),
                    "runtime_total_sec": rt.get("runtime_total_sec", np.nan),
                    "runtime_total_min": rt.get("runtime_total_min", np.nan),
                    "mean_fold_runtime_sec": rt.get("mean_fold_runtime_sec", np.nan),
                    "output_symlink_path": str(spec.run_dir),
                    "output_realpath": str(run_realpath(spec)),
                    "status": status_base,
                }
            )
    return pd.DataFrame(rows)


def infer_split_from_path_or_fold_tag(path: Path, frame: pd.DataFrame) -> str:
    name = path.name.lower()
    if "traindev" in name or "train" in name:
        return "trainDev"
    if "test" in name:
        return "test"
    if re.match(r"fold_\d+_scanner_leakage_summary\.csv$", name):
        return "trainDev"
    if "fold_tag" in frame.columns:
        tags = frame["fold_tag"].astype(str).str.lower()
        if tags.str.contains("traindev|train").any():
            return "trainDev"
        if tags.str.contains("test").any():
            return "test"
    return "unknown"


def build_scanner_comparison() -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        frames = []
        audit_path = run_realpath(spec) / "audit_v4/scanner_leakage_summary.csv"
        if audit_path.exists():
            audit = safe_read_csv(audit_path)
            if not audit.empty:
                for _, row in audit.iterrows():
                    raw = row.get("acc_site_raw_mean", np.nan)
                    latent = row.get("acc_site_latent_mean", np.nan)
                    chance = row.get("chance_level_mean", np.nan)
                    rows.append(
                        {
                            "run_name": spec.run_name,
                            "split": row.get("split", "unknown"),
                            "acc_site_raw_mean": raw,
                            "acc_site_raw_std": row.get("acc_site_raw_std", np.nan),
                            "acc_site_latent_mean": latent,
                            "acc_site_latent_std": row.get("acc_site_latent_std", np.nan),
                            "chance_level_mean": chance,
                            "latent_minus_raw_site_acc": latent - raw if pd.notna(raw) and pd.notna(latent) else np.nan,
                            "manufacturer_leakage_flag": bool(latent > chance + 0.10) if pd.notna(latent) and pd.notna(chance) else np.nan,
                        }
                    )
                continue
        for path in sorted(run_realpath(spec).glob("fold_*/*scanner_leakage_summary.csv")):
            df = safe_read_csv(path)
            if df.empty:
                continue
            df["split"] = infer_split_from_path_or_fold_tag(path, df)
            frames.append(df)
        if not frames:
            rows.append(
                {
                    "run_name": spec.run_name,
                    "split": "missing",
                    "acc_site_raw_mean": np.nan,
                    "acc_site_raw_std": np.nan,
                    "acc_site_latent_mean": np.nan,
                    "acc_site_latent_std": np.nan,
                    "chance_level_mean": np.nan,
                    "latent_minus_raw_site_acc": np.nan,
                    "manufacturer_leakage_flag": np.nan,
                }
            )
            continue
        scanner = pd.concat(frames, ignore_index=True)
        for split, group in scanner.groupby("split", dropna=False, sort=True):
            raw = pd.to_numeric(group.get("acc_site_raw"), errors="coerce")
            latent = pd.to_numeric(group.get("acc_site_latent"), errors="coerce")
            chance = pd.to_numeric(group.get("chance_level"), errors="coerce")
            raw_mean = float(raw.mean()) if raw.notna().any() else np.nan
            latent_mean = float(latent.mean()) if latent.notna().any() else np.nan
            chance_mean = float(chance.mean()) if chance.notna().any() else np.nan
            rows.append(
                {
                    "run_name": spec.run_name,
                    "split": split,
                    "acc_site_raw_mean": raw_mean,
                    "acc_site_raw_std": float(raw.std(ddof=1)) if raw.notna().sum() > 1 else 0.0 if raw.notna().sum() == 1 else np.nan,
                    "acc_site_latent_mean": latent_mean,
                    "acc_site_latent_std": float(latent.std(ddof=1))
                    if latent.notna().sum() > 1
                    else 0.0
                    if latent.notna().sum() == 1
                    else np.nan,
                    "chance_level_mean": chance_mean,
                    "latent_minus_raw_site_acc": latent_mean - raw_mean
                    if pd.notna(raw_mean) and pd.notna(latent_mean)
                    else np.nan,
                    "manufacturer_leakage_flag": bool(latent_mean > chance_mean + 0.10)
                    if pd.notna(latent_mean) and pd.notna(chance_mean)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_latent_info_comparison() -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        cfg_latent_dim = load_run_config(spec).get("args", {}).get("latent_dim", np.nan)
        frames = []
        audit_path = run_realpath(spec) / "audit_v4/latent_info_summary.csv"
        if audit_path.exists():
            audit = safe_read_csv(audit_path)
            if not audit.empty and "variable" in audit.columns:
                for split, group in audit.groupby("split", dropna=False, sort=True):
                    y = group[group["variable"].astype(str) == "Y_target"]
                    man = group[group["variable"].astype(str) == "Manufacturer"]
                    sex = group[group["variable"].astype(str) == "Sex"]
                    ratio = group[group["variable"].astype(str) == "MI_Manufacturer_over_Y_target"]
                    rows.append(
                        {
                            "run_name": spec.run_name,
                            "split": split,
                            "mi_y_target_mean": y["mi_sum_nats_mean"].iloc[0] if not y.empty and "mi_sum_nats_mean" in y else np.nan,
                            "mi_manufacturer_mean": man["mi_sum_nats_mean"].iloc[0] if not man.empty and "mi_sum_nats_mean" in man else np.nan,
                            "mi_sex_mean": sex["mi_sum_nats_mean"].iloc[0] if not sex.empty and "mi_sum_nats_mean" in sex else np.nan,
                            "mi_manufacturer_over_y_ratio": ratio["mi_manufacturer_over_y_ratio"].iloc[0]
                            if not ratio.empty and "mi_manufacturer_over_y_ratio" in ratio
                            else np.nan,
                            "n_active_mean": y["n_active_mean"].iloc[0] if not y.empty and "n_active_mean" in y else np.nan,
                            "frac_active_mean": y["frac_active_mean"].iloc[0] if not y.empty and "frac_active_mean" in y else np.nan,
                            "total_correlation_nats_mean": y["total_correlation_nats_mean"].iloc[0]
                            if not y.empty and "total_correlation_nats_mean" in y
                            else np.nan,
                            "latent_dim": cfg_latent_dim,
                        }
                    )
                continue
        for path in sorted(run_realpath(spec).glob("fold_*/*latent_info_summary.csv")):
            df = safe_read_csv(path)
            if df.empty:
                continue
            df["split"] = infer_split_from_path_or_fold_tag(path, df)
            frames.append(df)
        if not frames:
            rows.append(
                {
                    "run_name": spec.run_name,
                    "split": "missing",
                    "mi_y_target_mean": np.nan,
                    "mi_manufacturer_mean": np.nan,
                    "mi_sex_mean": np.nan,
                    "mi_manufacturer_over_y_ratio": np.nan,
                    "n_active_mean": np.nan,
                    "frac_active_mean": np.nan,
                    "total_correlation_nats_mean": np.nan,
                    "latent_dim": np.nan,
                }
            )
            continue
        latent = pd.concat(frames, ignore_index=True)
        for split, group in latent.groupby("split", dropna=False, sort=True):
            values = {}
            for variable, key in [("Y_target", "mi_y_target_mean"), ("Manufacturer", "mi_manufacturer_mean"), ("Sex", "mi_sex_mean")]:
                subset = group[group["variable"].astype(str) == variable]
                vals = pd.to_numeric(subset.get("mi_sum_nats"), errors="coerce") if not subset.empty else pd.Series(dtype=float)
                values[key] = float(vals.mean()) if vals.notna().any() else np.nan
            ratio = safe_div(values["mi_manufacturer_mean"], values["mi_y_target_mean"]) if pd.notna(values["mi_y_target_mean"]) else np.nan
            y_subset = group[group["variable"].astype(str) == "Y_target"]
            base = y_subset if not y_subset.empty else group
            rows.append(
                {
                    "run_name": spec.run_name,
                    "split": split,
                    **values,
                    "mi_manufacturer_over_y_ratio": ratio,
                    "n_active_mean": pd.to_numeric(base.get("n_active"), errors="coerce").mean(),
                    "frac_active_mean": pd.to_numeric(base.get("frac_active"), errors="coerce").mean(),
                    "total_correlation_nats_mean": pd.to_numeric(base.get("total_correlation_nats"), errors="coerce").mean(),
                    "latent_dim": pd.to_numeric(base.get("latent_dim"), errors="coerce").median(),
                }
            )
    return pd.DataFrame(rows)


def build_threshold_comparison(run_data: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        audit_path = run_realpath(spec) / "audit_v4/threshold_analysis_pooled_exploratory.csv"
        if audit_path.exists():
            audit = safe_read_csv(audit_path)
            if not audit.empty:
                for _, row in audit.iterrows():
                    rows.append(
                        {
                            "run_name": spec.run_name,
                            "classifier": row.get("classifier"),
                            "strategy": row.get("strategy"),
                            "is_exploratory": bool(row.get("is_exploratory", False)),
                            "valid_for_final_claim": bool(row.get("valid_for_final_claim", False)),
                            "threshold": row.get("threshold"),
                            "balanced_accuracy": row.get("balanced_accuracy"),
                            "sensitivity_AD": row.get("sensitivity_AD"),
                            "specificity_CN": row.get("specificity_CN"),
                            "precision": row.get("precision"),
                            "f1": row.get("f1"),
                            "note": row.get("selection_note", "Read from audit_v4 threshold analysis."),
                        }
                    )
                continue
        preds = run_data[spec.run_name]["predictions"]
        for classifier, group in preds.groupby("classifier", sort=True) if not preds.empty else []:
            metrics = compute_core_metrics(group, threshold=0.5)
            rows.append(
                {
                    "run_name": spec.run_name,
                    "classifier": classifier,
                    "strategy": "threshold_0.5",
                    "is_exploratory": False,
                    "valid_for_final_claim": True,
                    "threshold": 0.5,
                    "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                    "sensitivity_AD": metrics.get("sensitivity_AD", np.nan),
                    "specificity_CN": metrics.get("specificity_CN", np.nan),
                    "precision": metrics.get("precision", np.nan),
                    "f1": metrics.get("f1", np.nan),
                    "note": "Computed by this comparison script from fold test predictions only.",
                }
            )
    return pd.DataFrame(rows)


def build_manufacturer_error_comparison(run_data: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for spec in RUNS:
        preds = run_data[spec.run_name]["predictions_with_metadata"]
        if preds.empty:
            continue
        preds = preds.copy()
        if "Manufacturer" not in preds.columns:
            preds["Manufacturer"] = "NA"
        preds["Manufacturer"] = preds["Manufacturer"].fillna("NA").astype(str)
        for (classifier, manufacturer), group in preds.groupby(["classifier", "Manufacturer"], dropna=False, sort=True):
            metrics = compute_core_metrics(group, threshold=0.5, require_both=True)
            rows.append(
                {
                    "run_name": spec.run_name,
                    "classifier": classifier,
                    "Manufacturer": manufacturer,
                    "n": metrics.get("n_total", np.nan),
                    "n_CN": metrics.get("n_CN", np.nan),
                    "n_AD": metrics.get("n_AD", np.nan),
                    "roc_auc": metrics.get("roc_auc", np.nan),
                    "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
                    "sensitivity_AD": metrics.get("sensitivity_AD", np.nan),
                    "specificity_CN": metrics.get("specificity_CN", np.nan),
                    "false_positive_rate_CN": metrics.get("false_positive_rate_CN", np.nan),
                    "false_negative_rate_AD": metrics.get("false_negative_rate_AD", np.nan),
                    "mean_score_CN": metrics.get("mean_score_CN", np.nan),
                    "mean_score_AD": metrics.get("mean_score_AD", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_readme(
    output_dir: Path,
    model_df: pd.DataFrame,
    scanner_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    warnings: List[str],
) -> str:
    lines = [
        "# V4 Channel Full-Run Comparison",
        "",
        "## Methods",
        "This is a read-only audit. It reads existing CSV/TXT/JSON/log artifacts only. It does not retrain, load tensors, load checkpoints, load joblibs, or copy large files.",
        "",
        "The three compared runs are:",
    ]
    for spec in RUNS:
        lines.append(f"- `{spec.run_name}`: channels `{spec.channels_indices}` = {spec.channels_pretty}; symlink `{spec.run_dir}` -> `{run_realpath(spec)}`.")
    lines.extend(
        [
            "",
            "## Metric Interpretation",
            "Mean-fold metrics are the training pipeline's mean across outer folds from `all_folds_metrics_MULTI_*.csv`.",
            "Pooled metrics are recomputed here by concatenating fold test predictions and evaluating the pooled out-of-fold table. These can differ from mean-fold metrics because fold sizes and class counts are not identical.",
            "",
            "## Ranking by AUC Mean-Fold",
        ]
    )
    rank_auc = model_df.sort_values("auc_mean_fold", ascending=False)
    lines.append(rank_auc[["run_name", "classifier", "channels_indices", "auc_mean_fold", "pr_auc_mean_fold", "balanced_accuracy_mean_fold", "sensitivity_AD_mean_fold"]].to_markdown(index=False))
    lines.extend(["", "## Ranking by AD Sensitivity Mean-Fold"])
    rank_sens = model_df.sort_values("sensitivity_AD_mean_fold", ascending=False)
    lines.append(rank_sens[["run_name", "classifier", "channels_indices", "sensitivity_AD_mean_fold", "specificity_CN_mean_fold", "auc_mean_fold", "balanced_accuracy_mean_fold"]].to_markdown(index=False))
    lines.extend(["", "## Scanner / Manufacturer Interpretation"])
    if scanner_df.empty:
        lines.append("Scanner leakage summaries were not available.")
    else:
        scanner_view = scanner_df.copy()
        scanner_view = scanner_view[["run_name", "split", "acc_site_raw_mean", "acc_site_latent_mean", "latent_minus_raw_site_acc", "manufacturer_leakage_flag"]]
        lines.append(scanner_view.to_markdown(index=False))
    lines.extend(
        [
            "",
            "Higher AUC is not sufficient for selecting a final model if MI(Manufacturer)/MI(Y_target) or acc_site_latent increases. Channel variants that improve discrimination but amplify manufacturer information should be treated as exploratory until validated by scanner/site stress tests.",
            "",
            "## Latent Information Caveat",
        ]
    )
    if latent_df.empty:
        lines.append("Latent information summaries were not available.")
    else:
        latent_view = latent_df[["run_name", "split", "mi_y_target_mean", "mi_manufacturer_mean", "mi_manufacturer_over_y_ratio", "frac_active_mean", "total_correlation_nats_mean"]].copy()
        lines.append(latent_view.to_markdown(index=False))

    best_auc = rank_auc.iloc[0] if not rank_auc.empty else None
    best_sens = rank_sens.iloc[0] if not rank_sens.empty else None
    lines.extend(["", "## Recommended Next Action"])
    if best_auc is not None and best_sens is not None:
        lines.append(
            f"The top mean-fold AUC is `{best_auc['run_name']}` / `{best_auc['classifier']}` "
            f"(AUC={best_auc['auc_mean_fold']:.3f}). The top mean-fold AD sensitivity is "
            f"`{best_sens['run_name']}` / `{best_sens['classifier']}` "
            f"(sensitivity={best_sens['sensitivity_AD_mean_fold']:.3f})."
        )
        lines.append(
            "Before launching another full model, inspect manufacturer error rates and scanner leakage for the apparent winner, then prioritize an external/stress-test confirmation rather than selecting solely by AUC."
        )
    else:
        lines.append("No complete model metrics were available; inspect missing artifacts before choosing another run.")
    if warnings:
        lines.extend(["", "## Warnings / Missing Optional Artifacts"])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", f"Outputs written to `{output_dir}`."])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    warnings: List[str] = []
    run_data: Dict[str, Dict] = {}

    runtime_frames = []
    for spec in RUNS:
        real = run_realpath(spec)
        if not real.exists():
            warnings.append(f"{spec.run_name}: run directory missing: {spec.run_dir} -> {real}")
        preds, pred_warnings = load_fold_predictions(spec)
        warnings.extend(pred_warnings)
        preds_meta, meta_warnings = load_predictions_with_metadata(spec)
        warnings.extend(meta_warnings)
        fold_summary = load_fold_metric_summary(spec)
        if fold_summary.empty:
            warnings.append(f"{spec.run_name}: all_folds_metrics_MULTI_*.csv unavailable or unreadable")
        runtime = parse_logs(spec)
        runtime_frames.append(runtime)
        run_data[spec.run_name] = {
            "predictions": preds,
            "predictions_with_metadata": preds_meta,
            "fold_metric_summary": fold_summary,
            "effective_n_iter": effective_n_iter_by_classifier(spec),
            "runtime": runtime,
            "runtime_summary": runtime_summary(runtime),
        }

    model_df = build_model_comparison(run_data)
    scanner_df = build_scanner_comparison()
    latent_df = build_latent_info_comparison()
    threshold_df = build_threshold_comparison(run_data)
    manufacturer_df = build_manufacturer_error_comparison(run_data)
    runtime_df = pd.concat(runtime_frames, ignore_index=True) if runtime_frames else pd.DataFrame()

    write_csv_refresh(output_dir / "model_comparison_v4_channel_fullruns.csv", model_df, args.overwrite)
    write_csv_refresh(output_dir / "scanner_latent_comparison_v4_channel_fullruns.csv", scanner_df, args.overwrite)
    write_csv_refresh(output_dir / "latent_info_comparison_v4_channel_fullruns.csv", latent_df, args.overwrite)
    write_csv_refresh(output_dir / "threshold_comparison_v4_channel_fullruns.csv", threshold_df, args.overwrite)
    write_csv_refresh(output_dir / "manufacturer_error_comparison_v4_channel_fullruns.csv", manufacturer_df, args.overwrite)
    write_csv_refresh(output_dir / "runtime_comparison_v4_channel_fullruns.csv", runtime_df, args.overwrite)
    write_text_refresh(
        output_dir / "README.md",
        build_readme(output_dir, model_df, scanner_df, latent_df, warnings),
        args.overwrite,
    )

    print(f"Comparison outputs written to: {output_dir}")
    print(f"Runs compared: {len(RUNS)}")
    print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
