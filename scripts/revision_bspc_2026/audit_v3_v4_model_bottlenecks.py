#!/usr/bin/env python3
"""Read-only bottleneck audit for ADNI expanded v3/v4 planning.

The script consolidates existing stress-test inference tables, finished v3
training metrics, latent QC artifacts, and exploratory channel-ablation
summaries. It does not train models or alter source artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    silhouette_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "results/revision_bspc_2026/adni_expanded_v3_beta25_static3"
DEFAULT_OUTDIR = REPO_ROOT / "results/revision_bspc_2026/model_bottleneck_audit"
DEFAULT_V3_DIR = REPO_ROOT / "data/revision_bspc_2026/adni_expanded_v3_all_available"
DEFAULT_V3_TENSOR = DEFAULT_V3_DIR / "GLOBAL_TENSOR_ADNI_expanded_v3_all_available.npz"
DEFAULT_V3_METADATA = DEFAULT_V3_DIR / "subject_metadata_adni_expanded_v3_all_available.csv"

OUTPUT_FILES = [
    "README.md",
    "stress_test_summary.csv",
    "v3_metrics_summary.csv",
    "latent_space_audit.csv",
    "channel_ablation_comparison.csv",
    "recommended_next_runs.json",
]


@dataclass
class StressSpec:
    cohort: str
    description: str
    paths: Sequence[Path]
    filter_cn: bool = False  # if True, keep only ResearchGroup_Mapped == "CN"


STRESS_SPECS = [
    StressSpec(
        "siemens_cn_new",
        "Santiago Siemens5 CN (all subjects are CN by selection), original paper model inference",
        [
            Path("/media/diego/Datos/adni_expansion/SIEMENS_available/inference_outputs/Tables/covid_predictions_ensemble.csv"),
        ],
        filter_cn=False,
    ),
    StressSpec(
        "ge_cn_new",
        "Santiago GE10 CN (all subjects are CN by selection), original paper model inference",
        [
            Path("/media/diego/Datos/adni_expansion/GE_smoketest3/inference_outputs/Tables/covid_predictions_ensemble.csv"),
            Path("/media/diego/Datos/adni_expansion/GE_batch7/inference_outputs/Tables/covid_predictions_ensemble.csv"),
        ],
        filter_cn=False,
    ),
    StressSpec(
        "martin59_cn",
        "Martin59 CN only (mixed cohort, CN filtered via ResearchGroup_Mapped), original paper model inference",
        [
            Path("/media/diego/Datos/adni_expansion/MARTIN59/inference_outputs/Tables/martin59_predictions_with_metadata.csv"),
        ],
        filter_cn=True,
    ),
    StressSpec(
        "philips3_cn",
        "Philips3 CN stress set (all subjects are CN by selection), original paper model inference",
        [
            Path("/media/diego/Datos/adni_expansion/PHILIPS_CN_STRESS/inference_outputs_first1/Tables/covid_predictions_ensemble.csv"),
            Path("/media/diego/Datos/adni_expansion/PHILIPS_CN_STRESS/inference_outputs_plus2/Tables/philips_plus2_predictions_with_metadata.csv"),
        ],
        filter_cn=False,
    ),
    StressSpec(
        "philips7_cn",
        "Martin 2026-04-29 Philips7 CN only (mixed cohort, CN filtered via ResearchGroup_Mapped), original paper model inference",
        [
            Path("/media/diego/Datos/adni_expansion/MARTIN_20260429_PHILIPS10/inference_original_model/Tables/philips7_original_model_predictions_with_metadata.csv"),
        ],
        filter_cn=True,
    ),
]


ABLATION_SUMMARIES = [
    {
        "run_label": "historical_original_notebook_ablation",
        "dataset": "historical_adni",
        "path": REPO_ROOT / "notebooks/ablation_full_run_fast/summary_ablation.csv",
        "notes": "Old notebook output; beta=4.6, latent_dim=128, outer=3. Greedy channel selection is exploratory.",
    },
    {
        "run_label": "v3_original_channels_exploratory",
        "dataset": "adni_expanded_v3",
        "path": REPO_ROOT / "results/revision_bspc_2026/ablation_v3_exploratory/ablation_runs/summary_ablation.csv",
        "notes": "V3 exploratory ablation over original/static channels; channel selection is exploratory.",
    },
    {
        "run_label": "v3_fast128_all7_exploratory",
        "dataset": "adni_expanded_v3",
        "path": REPO_ROOT / "results/revision_bspc_2026/ablation_v3_fast128_all7/ablation_runs/summary_ablation.csv",
        "notes": "V3 fast all-7 exploratory ablation; beta=2.5, latent_dim=128, outer=3. Not final performance.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive v3/v4 model bottleneck audit."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--v3-tensor", type=Path, default=DEFAULT_V3_TENSOR)
    parser.add_argument("--v3-metadata", type=Path, default=DEFAULT_V3_METADATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing audit files inside --outdir.")
    parser.add_argument(
        "--skip-latent-encoding",
        action="store_true",
        help="Do not encode existing fold checkpoints to compute covariance/effective-rank metrics.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for optional latent encoding.")
    return parser.parse_args()


def prepare_outdir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and not outdir.is_dir():
        raise FileExistsError(f"Output path exists but is not a directory: {outdir}")
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)
        return
    existing = [outdir / name for name in OUTPUT_FILES if (outdir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing audit outputs. Use --overwrite if intentional:\n"
            + "\n".join(str(path) for path in existing)
        )


def read_csv_if_exists(path: Path, missing: List[str], nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        missing.append(str(path))
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as exc:
        missing.append(f"{path} (read failed: {exc})")
        return None


def first_existing_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_subject_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "SubjectID" in df.columns:
        df = df.copy()
        df["SubjectID"] = df["SubjectID"].astype(str).str.strip()
    return df


def build_stress_test_summary(missing: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in STRESS_SPECS:
        frames = []
        used_paths = []
        for path in spec.paths:
            df = read_csv_if_exists(path, missing)
            if df is None:
                continue
            df = normalize_subject_ids(df)
            df["_source_path"] = str(path)
            frames.append(df)
            used_paths.append(str(path))
        if not frames:
            rows.append(
                {
                    "stress_cohort": spec.cohort,
                    "description": spec.description,
                    "classifier": "NA",
                    "n": 0,
                    "n_pred_ad": np.nan,
                    "ad_like_rate": np.nan,
                    "mean_score": np.nan,
                    "std_score": np.nan,
                    "mean_fold_score_std": np.nan,
                    "majority_vote_ad_like_rate": np.nan,
                    "n_majority_vote_ad": np.nan,
                    "source_paths": "",
                    "status": "missing_predictions",
                }
            )
            continue
        combined = pd.concat(frames, ignore_index=True)

        # Filter to CN-only when requested (requires ResearchGroup_Mapped column)
        if spec.filter_cn:
            rgm_col = first_existing_col(combined, ["ResearchGroup_Mapped", "ResearchGroup", "research_group"])
            if rgm_col is not None:
                n_before = len(combined)
                combined = combined[combined[rgm_col].astype(str).str.strip() == "CN"].copy()
                if len(combined) < n_before:
                    pass  # silently filtered; captured in n column
            else:
                missing.append(
                    f"{spec.cohort}: filter_cn=True but ResearchGroup_Mapped column not found — using all subjects"
                )

        classifier_col = first_existing_col(combined, ["classifier", "classifier_type", "model"])
        score_col = first_existing_col(
            combined,
            ["y_score_ensemble", "AD_like_logreg", "AD_like_svm", "y_score", "prob_ad", "y_score_final"],
        )
        pred_col = first_existing_col(combined, ["y_pred_ensemble", "Predicted_AD_like_at_0p5", "y_pred"])
        majority_col = first_existing_col(combined, ["y_pred_majority_vote"])
        fold_std_col = first_existing_col(combined, ["y_score_std", "AD_like_logreg_fold_sd", "AD_like_svm_fold_sd"])

        if classifier_col is None and {"AD_like_logreg", "AD_like_svm"}.intersection(combined.columns):
            long_frames = []
            for clf, score_name, sd_name in [
                ("logreg", "AD_like_logreg", "AD_like_logreg_fold_sd"),
                ("svm", "AD_like_svm", "AD_like_svm_fold_sd"),
            ]:
                if score_name in combined.columns:
                    tmp = combined.copy()
                    tmp["classifier"] = clf
                    tmp["y_score_ensemble"] = pd.to_numeric(tmp[score_name], errors="coerce")
                    if sd_name in tmp.columns:
                        tmp["y_score_std"] = pd.to_numeric(tmp[sd_name], errors="coerce")
                    tmp["y_pred_ensemble"] = (tmp["y_score_ensemble"] >= 0.5).astype(int)
                    long_frames.append(tmp)
            combined = pd.concat(long_frames, ignore_index=True) if long_frames else combined
            classifier_col = "classifier"
            score_col = "y_score_ensemble"
            pred_col = "y_pred_ensemble"
            fold_std_col = "y_score_std" if "y_score_std" in combined.columns else None

        if classifier_col is None:
            classifier_groups = [("all", combined)]
        else:
            classifier_groups = combined.groupby(classifier_col, dropna=False, sort=True)

        for classifier, group in classifier_groups:
            group = group.copy()
            if "SubjectID" in group.columns:
                group = group.drop_duplicates(["SubjectID", classifier_col] if classifier_col else ["SubjectID"])
            score = pd.to_numeric(group[score_col], errors="coerce") if score_col else pd.Series(dtype=float)
            if pred_col:
                pred = group[pred_col].map(lambda x: bool(x) if not pd.isna(x) else np.nan)
                pred_num = pred.astype(float)
            elif not score.empty:
                pred_num = (score >= 0.5).astype(float)
            else:
                pred_num = pd.Series(dtype=float)
            majority_num = (
                group[majority_col].map(lambda x: bool(x) if not pd.isna(x) else np.nan).astype(float)
                if majority_col
                else pd.Series(dtype=float)
            )
            fold_std = pd.to_numeric(group[fold_std_col], errors="coerce") if fold_std_col else pd.Series(dtype=float)
            rows.append(
                {
                    "stress_cohort": spec.cohort,
                    "description": spec.description,
                    "classifier": str(classifier),
                    "n": int(len(group)),
                    "n_pred_ad": int(pred_num.sum()) if len(pred_num) else np.nan,
                    "ad_like_rate": float(pred_num.mean()) if len(pred_num) else np.nan,
                    "mean_score": float(score.mean()) if len(score) else np.nan,
                    "std_score": float(score.std(ddof=1)) if len(score) > 1 else 0.0 if len(score) == 1 else np.nan,
                    "mean_fold_score_std": float(fold_std.mean()) if len(fold_std) else np.nan,
                    "majority_vote_ad_like_rate": float(majority_num.mean()) if len(majority_num) else np.nan,
                    "n_majority_vote_ad": int(majority_num.sum()) if len(majority_num) else np.nan,
                    "source_paths": ";".join(used_paths),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def to_binary(series: pd.Series) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
            return int(value)
        text = str(value).strip().upper()
        if text in {"CN", "CONTROL", "0", "0.0"}:
            return 0
        if text in {"AD", "DEMENTIA", "1", "1.0"}:
            return 1
        return np.nan

    out = series.map(convert)
    if out.isna().any():
        bad = sorted(series[out.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Could not map labels to binary CN=0/AD=1: {bad}")
    return out.astype(int)


def metrics_from_predictions(df: pd.DataFrame, classifier: str, strategy: str, threshold: float) -> Dict[str, object]:
    y_true = to_binary(df["y_true"]).to_numpy()
    score_col = first_existing_col(df, ["y_score_final", "y_score_cal", "y_score", "y_score_raw"])
    if score_col is None:
        raise ValueError("No score column in predictions.")
    y_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "source": "computed_from_pooled_predictions",
        "classifier": classifier,
        "fold": np.nan,
        "strategy": strategy,
        "threshold": threshold,
        "is_exploratory": strategy != "threshold_0.5",
        "n": int(len(df)),
        "n_CN": int((y_true == 0).sum()),
        "n_AD": int((y_true == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_AD": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity_CN": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
    }


def build_v3_metrics_summary(run_dir: Path, missing: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    global_path = run_dir / "audit_v3/global_metrics_by_classifier.csv"
    fold_path = run_dir / "classification_metrics_by_fold_manual_summary.csv"
    threshold_path = run_dir / "audit_v3/threshold_analysis.csv"
    pooled_path = run_dir / "pooled_test_predictions_all_folds.csv"

    global_df = read_csv_if_exists(global_path, missing)
    if global_df is not None:
        for _, row in global_df.iterrows():
            out = row.to_dict()
            out.update(
                {
                    "source": "global_metrics_by_classifier",
                    "fold": np.nan,
                    "strategy": "threshold_0.5",
                    "threshold": 0.5,
                    "is_exploratory": False,
                }
            )
            rows.append(out)

    fold_df = read_csv_if_exists(fold_path, missing)
    if fold_df is not None:
        for _, row in fold_df.iterrows():
            out = row.to_dict()
            out.update(
                {
                    "source": "classification_metrics_by_fold_manual_summary",
                    "strategy": "threshold_0.5",
                    "threshold": 0.5,
                    "is_exploratory": False,
                }
            )
            rows.append(out)

    threshold_df = read_csv_if_exists(threshold_path, missing)
    if threshold_df is not None:
        for _, row in threshold_df.iterrows():
            out = row.to_dict()
            strategy = str(out.get("strategy", ""))
            out.update(
                {
                    "source": "threshold_analysis",
                    "fold": np.nan,
                    "is_exploratory": strategy != "threshold_0.5",
                    "brier": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "accuracy": np.nan,
                }
            )
            rows.append(out)

    if global_df is None:
        pooled_df = read_csv_if_exists(pooled_path, missing)
        if pooled_df is not None and "classifier" in pooled_df.columns:
            for classifier, group in pooled_df.groupby("classifier"):
                rows.append(metrics_from_predictions(group, str(classifier), "threshold_0.5", 0.5))

    return pd.DataFrame(rows)


def load_run_args(run_dir: Path, missing: List[str]) -> Dict[str, object]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        missing.append(str(config_path))
        return {}
    try:
        data = json.loads(config_path.read_text())
    except Exception as exc:
        missing.append(f"{config_path} (read failed: {exc})")
        return {}
    return data.get("args", data)


def apply_norm_params(data: np.ndarray, norm_params: Sequence[Dict[str, object]]) -> np.ndarray:
    out = data.copy()
    n_rois = out.shape[-1]
    off_diag = ~np.eye(n_rois, dtype=bool)
    for c_idx, params in enumerate(norm_params):
        if params.get("no_scale", False):
            diag = np.arange(n_rois)
            out[:, c_idx, diag, diag] = 0.0
            continue
        mode = params.get("mode", "zscore_offdiag")
        current = out[:, c_idx, :, :]
        if mode == "zscore_offdiag":
            mean = float(params.get("mean", 0.0))
            std = float(params.get("std", 1.0))
            if std > 1e-9:
                current[:, off_diag] = (current[:, off_diag] - mean) / std
        elif mode == "minmax_offdiag":
            mn = float(params.get("min", 0.0))
            mx = float(params.get("max", 1.0))
            rng = mx - mn
            current[:, off_diag] = (current[:, off_diag] - mn) / rng if rng > 1e-9 else 0.0
        diag = np.arange(n_rois)
        current[:, diag, diag] = 0.0
        out[:, c_idx, :, :] = current
    return out


def load_vae_class(repo_root: Path):
    module_path = repo_root / "src/betavae_xai/models/convolutional_vae.py"
    spec = importlib.util.spec_from_file_location("convolutional_vae_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvolutionalVAE


def encode_mu(
    model,
    data: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    import torch

    model.eval()
    mus = []
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = torch.from_numpy(data[start : start + batch_size]).float().to(device)
            mu, _ = model.encode(batch)
            mus.append(mu.detach().cpu().numpy())
    return np.vstack(mus) if mus else np.empty((0, getattr(model, "latent_dim", 0)))


def safe_silhouette(mu: np.ndarray, labels: Sequence[object]) -> float:
    labels_series = pd.Series(labels).fillna("NA").astype(str)
    counts = labels_series.value_counts()
    if len(counts) < 2 or len(labels_series) <= len(counts):
        return np.nan
    if counts.min() < 1:
        return np.nan
    try:
        return float(silhouette_score(mu, labels_series.to_numpy()))
    except Exception:
        return np.nan


def covariance_metrics(mu: np.ndarray, ridge: float = 1e-6, active_eps: float = 1e-4) -> Dict[str, object]:
    if mu.shape[0] < 2:
        return {
            "mu_variance_mean": np.nan,
            "mu_variance_median": np.nan,
            "mu_variance_max": np.nan,
            "n_active_from_mu": np.nan,
            "frac_active_from_mu": np.nan,
            "cov_eig_1": np.nan,
            "cov_eig_2": np.nan,
            "cov_eig_5": np.nan,
            "cov_eig_10": np.nan,
            "effective_rank_entropy": np.nan,
            "participation_ratio": np.nan,
            "gaussian_total_correlation_nats_from_mu": np.nan,
        }
    var = np.var(mu, axis=0)
    centered = mu - np.mean(mu, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    eigsum = float(eigvals.sum())
    if eigsum > 0:
        p = eigvals / eigsum
        p_nonzero = p[p > 0]
        effective_rank = float(np.exp(-(p_nonzero * np.log(p_nonzero)).sum()))
        participation_ratio = float((eigsum ** 2) / np.sum(eigvals ** 2)) if np.sum(eigvals ** 2) > 0 else np.nan
    else:
        effective_rank = np.nan
        participation_ratio = np.nan
    cov_r = cov + np.eye(cov.shape[0]) * ridge
    var_r = np.diag(cov) + ridge
    try:
        sign, logdet = np.linalg.slogdet(cov_r)
        tc = 0.5 * (float(np.sum(np.log(var_r))) - float(logdet)) if sign > 0 else np.nan
    except Exception:
        tc = np.nan
    return {
        "mu_variance_mean": float(np.mean(var)),
        "mu_variance_median": float(np.median(var)),
        "mu_variance_max": float(np.max(var)),
        "n_active_from_mu": int(np.sum(var > active_eps)),
        "frac_active_from_mu": float(np.mean(var > active_eps)),
        "cov_eig_1": float(eigvals[0]) if len(eigvals) >= 1 else np.nan,
        "cov_eig_2": float(eigvals[1]) if len(eigvals) >= 2 else np.nan,
        "cov_eig_5": float(eigvals[4]) if len(eigvals) >= 5 else np.nan,
        "cov_eig_10": float(eigvals[9]) if len(eigvals) >= 10 else np.nan,
        "effective_rank_entropy": effective_rank,
        "participation_ratio": participation_ratio,
        "gaussian_total_correlation_nats_from_mu": tc,
    }


def build_metadata_index(tensor_path: Path, metadata_path: Path, missing: List[str]) -> Optional[pd.DataFrame]:
    if not tensor_path.exists():
        missing.append(str(tensor_path))
        return None
    if not metadata_path.exists():
        missing.append(str(metadata_path))
        return None
    try:
        with np.load(tensor_path, allow_pickle=True) as npz:
            subject_ids = np.asarray(npz["subject_ids"]).astype(str)
        meta = pd.read_csv(metadata_path)
        meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
        tensor_df = pd.DataFrame({"SubjectID": subject_ids, "tensor_idx": np.arange(len(subject_ids), dtype=int)})
        merged = tensor_df.merge(meta.drop_duplicates("SubjectID"), on="SubjectID", how="left")
        if "Site3" not in merged.columns:
            merged["Site3"] = merged["SubjectID"].str.slice(0, 3)
        return merged.set_index("tensor_idx", drop=False)
    except Exception as exc:
        missing.append(f"metadata/tensor index build failed ({exc})")
        return None


def add_existing_latent_qc_rows(run_dir: Path, missing: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold_text = fold_dir.name.replace("fold_", "")
        try:
            fold = int(fold_text)
        except ValueError:
            fold = np.nan

        for split_name, pattern in [
            ("test", f"fold_{fold_text}_test_latent_info_summary.csv"),
            ("trainDev", f"fold_{fold_text}_trainDev_latent_info_summary.csv"),
        ]:
            p = fold_dir / pattern
            df = read_csv_if_exists(p, missing)
            if df is None:
                continue
            for _, r in df.iterrows():
                rows.append(
                    {
                        "source": "latent_info_summary_existing",
                        "fold": fold,
                        "split": split_name,
                        "variable": r.get("variable"),
                        "n_samples": r.get("n_samples"),
                        "latent_dim": r.get("latent_dim"),
                        "n_active_existing": r.get("n_active"),
                        "frac_active_existing": r.get("frac_active"),
                        "total_correlation_nats_existing": r.get("total_correlation_nats"),
                        "mi_sum_nats": r.get("mi_sum_nats"),
                        "mi_mean_nats": r.get("mi_mean_nats"),
                        "top_dims": r.get("top_dims"),
                    }
                )

        for split_name, filename in [
            ("test", f"fold_{fold_text}_test_scanner_leakage_summary.csv"),
            ("trainDev", f"fold_{fold_text}_scanner_leakage_summary.csv"),
        ]:
            p = fold_dir / filename
            df = read_csv_if_exists(p, missing)
            if df is None:
                continue
            for _, r in df.iterrows():
                rows.append(
                    {
                        "source": "scanner_leakage_existing",
                        "fold": fold,
                        "split": split_name,
                        "variable": r.get("site_col"),
                        "n_samples": r.get("n_samples"),
                        "n_sites": r.get("n_sites"),
                        "scanner_chance_level": r.get("chance_level"),
                        "scanner_acc_raw": r.get("acc_site_raw"),
                        "scanner_acc_raw_std": r.get("acc_site_raw_std"),
                        "scanner_acc_latent": r.get("acc_site_latent"),
                        "scanner_acc_latent_std": r.get("acc_site_latent_std"),
                    }
                )

        p = fold_dir / "latent_qc_metrics.csv"
        df = read_csv_if_exists(p, missing)
        if df is not None:
            for _, r in df.iterrows():
                rows.append(
                    {
                        "source": "latent_qc_metrics_existing",
                        "fold": fold,
                        "split": "test",
                        "variable": "Y_target",
                        "silhouette_diagnosis_existing": r.get("silhouette_latent"),
                        "scanner_acc_latent": r.get("acc_site_latent"),
                        "scanner_acc_raw": r.get("acc_site_raw"),
                        "scanner_chance_level": r.get("chance_level"),
                        "latent_dim": r.get("latent_dim"),
                        "beta_max": r.get("beta_max"),
                        "channels_used": r.get("channels_used"),
                    }
                )
    return rows


def build_encoded_latent_rows(
    run_dir: Path,
    tensor_path: Path,
    metadata_path: Path,
    run_args: Dict[str, object],
    device: str,
    missing: List[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    try:
        import torch
        import joblib
    except Exception as exc:
        missing.append(f"latent encoding skipped: required imports unavailable ({exc})")
        return rows

    if device == "cuda" and not torch.cuda.is_available():
        missing.append("latent encoding requested cuda but CUDA is unavailable; using CPU")
        device = "cpu"

    try:
        ConvolutionalVAE = load_vae_class(REPO_ROOT)
    except Exception as exc:
        missing.append(f"latent encoding skipped: could not load ConvolutionalVAE ({exc})")
        return rows

    meta_idx = build_metadata_index(tensor_path, metadata_path, missing)
    if meta_idx is None:
        return rows

    channels = run_args.get("channels_to_use") or run_args.get("channels_to_use_indices") or [1, 0, 2]
    channels = [int(c) for c in channels]
    latent_dim = int(run_args.get("latent_dim", 256))
    dropout = float(run_args.get("dropout_rate_vae", 0.15))
    final_activation = str(run_args.get("vae_final_activation", "tanh"))
    intermediate_fc = run_args.get("intermediate_fc_dim_vae", "quarter")
    use_layernorm = bool(run_args.get("use_layernorm_vae_fc", False))
    num_conv_layers = int(run_args.get("num_conv_layers_encoder", 4))
    decoder_type = str(run_args.get("decoder_type", "convtranspose"))
    active_eps = float(run_args.get("qc_var_eps_active", 1e-4))
    tc_ridge = float(run_args.get("qc_tc_ridge", 1e-6))
    batch_size = int(run_args.get("batch_size", 64))

    try:
        with np.load(tensor_path, allow_pickle=True) as npz:
            tensor = np.asarray(npz["global_tensor_data"][:, channels, :, :], dtype=np.float32)
    except Exception as exc:
        missing.append(f"latent encoding skipped: tensor load failed ({exc})")
        return rows

    for fold_dir in sorted(run_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold_text = fold_dir.name.replace("fold_", "")
        try:
            fold = int(fold_text)
        except ValueError:
            continue
        model_path = fold_dir / f"vae_model_fold_{fold}.pt"
        norm_path = fold_dir / "vae_norm_params.joblib"
        if not model_path.exists():
            missing.append(str(model_path))
            continue
        if not norm_path.exists():
            missing.append(str(norm_path))
            continue
        try:
            norm_params = joblib.load(norm_path)
            model = ConvolutionalVAE(
                input_channels=len(channels),
                latent_dim=latent_dim,
                image_size=int(tensor.shape[-1]),
                final_activation=final_activation,
                intermediate_fc_dim_config=intermediate_fc,
                dropout_rate=dropout,
                use_layernorm_fc=use_layernorm,
                num_conv_layers_encoder=num_conv_layers,
                decoder_type=decoder_type,
            ).to(device)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
        except Exception as exc:
            missing.append(f"fold_{fold} latent encoding setup failed ({exc})")
            continue

        for split_name, idx_file in [("trainDev", "train_dev_tensor_idx.npy"), ("test", "test_tensor_idx.npy")]:
            idx_path = fold_dir / idx_file
            if not idx_path.exists():
                missing.append(str(idx_path))
                continue
            try:
                idx = np.load(idx_path).astype(int)
                split_tensor = apply_norm_params(tensor[idx], norm_params)
                mu = encode_mu(model, split_tensor, device=device, batch_size=batch_size)
                split_meta = meta_idx.loc[idx]
                cov = covariance_metrics(mu, ridge=tc_ridge, active_eps=active_eps)
                row = {
                    "source": "encoded_existing_vae_checkpoint",
                    "fold": fold,
                    "split": split_name,
                    "variable": "mu_covariance",
                    "n_samples": int(mu.shape[0]),
                    "latent_dim": int(mu.shape[1]) if mu.ndim == 2 else latent_dim,
                    "silhouette_diagnosis_from_mu": safe_silhouette(mu, split_meta.get("ResearchGroup_Mapped", pd.Series(index=split_meta.index))),
                    "silhouette_manufacturer_from_mu": safe_silhouette(mu, split_meta.get("Manufacturer", pd.Series(index=split_meta.index))),
                    "silhouette_site3_from_mu": safe_silhouette(mu, split_meta.get("Site3", pd.Series(index=split_meta.index))),
                    "silhouette_sourcecohort_from_mu": safe_silhouette(mu, split_meta.get("SourceCohort", pd.Series(index=split_meta.index))),
                }
                row.update(cov)
                rows.append(row)
            except Exception as exc:
                missing.append(f"fold_{fold} {split_name} latent encoding failed ({exc})")
        try:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
    return rows


def build_latent_space_audit(
    run_dir: Path,
    tensor_path: Path,
    metadata_path: Path,
    run_args: Dict[str, object],
    device: str,
    skip_encoding: bool,
    missing: List[str],
) -> pd.DataFrame:
    rows = add_existing_latent_qc_rows(run_dir, missing)
    if skip_encoding:
        missing.append("latent encoding skipped by --skip-latent-encoding; covariance/effective-rank metrics unavailable")
    else:
        rows.extend(build_encoded_latent_rows(run_dir, tensor_path, metadata_path, run_args, device, missing))
    return pd.DataFrame(rows)


def parse_channel_set(value: object) -> List[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    out = []
    for token in text.replace(",", " ").split():
        try:
            out.append(int(token))
        except ValueError:
            pass
    return out


def build_channel_ablation_comparison(missing: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in ABLATION_SUMMARIES:
        path = Path(spec["path"])
        df = read_csv_if_exists(path, missing)
        if df is None:
            rows.append(
                {
                    "run_label": spec["run_label"],
                    "dataset": spec["dataset"],
                    "status": "missing_summary",
                    "source_path": str(path),
                    "is_exploratory_channel_selection": True,
                    "not_final_performance": True,
                    "notes": spec["notes"],
                }
            )
            continue
        for _, row in df.iterrows():
            channels = parse_channel_set(row.get("channels_indices"))
            out = row.to_dict()
            out.update(
                {
                    "run_label": spec["run_label"],
                    "dataset": spec["dataset"],
                    "status": "ok",
                    "source_path": str(path),
                    "channels_set_sorted": " ".join(map(str, sorted(channels))),
                    "matches_original_paper_subset_set_0_1_2": set(channels) == {0, 1, 2},
                    "is_exploratory_channel_selection": True,
                    "not_final_performance": True,
                    "notes": spec["notes"],
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def metric_lookup(v3_metrics: pd.DataFrame, classifier: str, source: str = "global_metrics_by_classifier") -> Optional[pd.Series]:
    if v3_metrics.empty:
        return None
    subset = v3_metrics[(v3_metrics.get("classifier") == classifier) & (v3_metrics.get("source") == source)]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_recommendations(
    stress: pd.DataFrame,
    metrics: pd.DataFrame,
    latent: pd.DataFrame,
    ablation: pd.DataFrame,
    missing: List[str],
) -> Dict[str, object]:
    svm = metric_lookup(metrics, "svm")
    logreg = metric_lookup(metrics, "logreg")
    def get_float(row: Optional[pd.Series], col: str) -> Optional[float]:
        if row is None or col not in row or pd.isna(row[col]):
            return None
        return float(row[col])

    stress_high = stress[(stress["status"] == "ok") & (stress["ad_like_rate"].notna())].copy()
    stress_high = stress_high.sort_values("ad_like_rate", ascending=False).head(6)

    recommendations = [
        {
            "name": "static3_current_control",
            "channels_to_use": [1, 0, 2],
            "channel_names": [
                "Pearson_Full_FisherZ_Signed",
                "Pearson_OMST_GCE_Signed_Weighted",
                "MI_KNN_Symmetric",
            ],
            "priority": 1,
            "rationale": "Current full v3 configuration and paper baseline; rerun only as the v4 control when Philips7 is added.",
        },
        {
            "name": "historical_best_two_channel",
            "channels_to_use": [1, 2],
            "channel_names": ["Pearson_Full_FisherZ_Signed", "MI_KNN_Symmetric"],
            "priority": 2,
            "rationale": "Old historical ablation selected this 2-channel subset before adding OMST; tests whether OMST contributes to low AD sensitivity.",
        },
        {
            "name": "v3_static_omst_only",
            "channels_to_use": [0],
            "channel_names": ["Pearson_OMST_GCE_Signed_Weighted"],
            "priority": 3,
            "rationale": "Best path in v3 original-channel exploratory ablation; useful bottleneck check for a simpler static representation.",
        },
        {
            "name": "fast_all7_best_dynamic_pair",
            "channels_to_use": [4, 1],
            "channel_names": ["dFC_StdDev", "Pearson_Full_FisherZ_Signed"],
            "priority": 4,
            "rationale": "Best fast all-7 exploratory pair; should be tested only under the full v3/v4 configuration before any claim.",
        },
    ]
    return {
        "generated_by": "audit_v3_v4_model_bottlenecks.py",
        "recommendation_scope": "Run these under the full model configuration; do not interpret exploratory ablation scores as final performance.",
        "current_v3_global_metrics": {
            "logreg_roc_auc": get_float(logreg, "roc_auc"),
            "logreg_sensitivity_AD": get_float(logreg, "sensitivity_AD"),
            "logreg_specificity_CN": get_float(logreg, "specificity_CN"),
            "svm_roc_auc": get_float(svm, "roc_auc"),
            "svm_sensitivity_AD": get_float(svm, "sensitivity_AD"),
            "svm_specificity_CN": get_float(svm, "specificity_CN"),
        },
        "stress_tests_highest_ad_like_rates": stress_high[
            ["stress_cohort", "classifier", "n", "ad_like_rate", "mean_score"]
        ].to_dict(orient="records"),
        "recommended_channel_subset_runs": recommendations,
        "methodological_notes": [
            "Threshold tuning in current audit outputs is pooled/exploratory unless thresholds are estimated inside train/dev folds.",
            "Channel ablation summaries are exploratory channel-selection results and are not unbiased final performance estimates.",
            "The v3 model has high CN specificity at threshold 0.5 but low AD sensitivity; next runs should prioritize sensitivity/balanced accuracy without sacrificing stress-test CN behavior.",
        ],
        "missing_or_unavailable_artifacts": sorted(set(missing)),
    }


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def summarize_stress_for_readme(stress: pd.DataFrame) -> List[str]:
    lines = []
    if stress.empty:
        return ["- No stress-test rows were available."]
    for _, row in stress.sort_values(["stress_cohort", "classifier"]).iterrows():
        if row.get("status") != "ok":
            lines.append(f"- {row.get('stress_cohort')}: missing predictions.")
            continue
        lines.append(
            f"- {row['stress_cohort']} / {row['classifier']}: n={int(row['n'])}, "
            f"AD-like={fmt_float(row['ad_like_rate'])}, mean score={fmt_float(row['mean_score'])}, "
            f"majority-vote AD-like={fmt_float(row['majority_vote_ad_like_rate'])}."
        )
    return lines


def write_readme(
    outdir: Path,
    stress: pd.DataFrame,
    metrics: pd.DataFrame,
    latent: pd.DataFrame,
    ablation: pd.DataFrame,
    recommendations: Dict[str, object],
    missing: List[str],
) -> None:
    global_rows = metrics[metrics.get("source") == "global_metrics_by_classifier"] if not metrics.empty else pd.DataFrame()
    threshold_rows = metrics[metrics.get("source") == "threshold_analysis"] if not metrics.empty else pd.DataFrame()
    encoded_rows = latent[latent.get("source") == "encoded_existing_vae_checkpoint"] if not latent.empty else pd.DataFrame()
    lines = [
        "# V3/V4 Model Bottleneck Audit",
        "",
        "This audit is read-only with respect to existing model outputs. It consolidates available stress tests, v3 metrics, latent QC artifacts, and exploratory channel-ablation summaries.",
        "",
        "## Stress-Test Summary",
    ]
    lines.extend(summarize_stress_for_readme(stress))
    lines.extend(["", "## V3 Classification Metrics"])
    if global_rows.empty:
        lines.append("- Global metrics artifact was not available.")
    else:
        for _, row in global_rows.sort_values("classifier").iterrows():
            lines.append(
                f"- {row['classifier']}: ROC-AUC={fmt_float(row.get('roc_auc'))}, PR-AUC={fmt_float(row.get('pr_auc'))}, "
                f"Brier={fmt_float(row.get('brier'))}, sensitivity_AD={fmt_float(row.get('sensitivity_AD'))}, "
                f"specificity_CN={fmt_float(row.get('specificity_CN'))}, balanced accuracy={fmt_float(row.get('balanced_accuracy'))}."
            )
    if not threshold_rows.empty:
        lines.extend(["", "## Threshold Audit"])
        for _, row in threshold_rows.sort_values(["classifier", "strategy"]).iterrows():
            label = "exploratory" if bool(row.get("is_exploratory", False)) else "fixed"
            lines.append(
                f"- {row['classifier']} / {row['strategy']} ({label}): threshold={fmt_float(row.get('threshold'))}, "
                f"sensitivity_AD={fmt_float(row.get('sensitivity_AD'))}, specificity_CN={fmt_float(row.get('specificity_CN'))}, "
                f"balanced accuracy={fmt_float(row.get('balanced_accuracy'))}."
            )
    lines.extend(["", "## Latent Bottleneck Audit"])
    if encoded_rows.empty:
        lines.append("- Latent covariance/effective-rank metrics could not be recomputed from checkpoints; see missing artifacts list.")
    else:
        test_rows = encoded_rows[encoded_rows["split"] == "test"]
        if not test_rows.empty:
            lines.append(
                "- Encoded test μ across folds: "
                f"mean effective rank={fmt_float(test_rows['effective_rank_entropy'].mean())}, "
                f"mean participation ratio={fmt_float(test_rows['participation_ratio'].mean())}, "
                f"mean active units={fmt_float(test_rows['n_active_from_mu'].mean())}, "
                f"mean Gaussian TC={fmt_float(test_rows['gaussian_total_correlation_nats_from_mu'].mean())}."
            )
    existing_tc = latent[
        (latent.get("source") == "latent_info_summary_existing") & (latent.get("variable") == "Y_target")
    ] if not latent.empty else pd.DataFrame()
    if not existing_tc.empty:
        lines.append(
            "- Existing latent-info summaries report active units and total correlation; see `latent_space_audit.csv` for trainDev/test and nuisance-variable MI."
        )
    lines.extend(["", "## Channel Ablation"])
    if ablation.empty:
        lines.append("- No ablation summaries were available.")
    else:
        best_rows = ablation[ablation["status"] == "ok"].copy()
        if not best_rows.empty:
            best_rows["metric_mean_numeric"] = pd.to_numeric(best_rows.get("metric_mean"), errors="coerce")
            for run_label, group in best_rows.groupby("run_label"):
                best = group.sort_values("metric_mean_numeric", ascending=False).iloc[0]
                lines.append(
                    f"- {run_label}: best exploratory row `{best.get('channels_indices')}` "
                    f"({best.get('channels_pretty')}) with {best.get('metric')}={fmt_float(best.get('metric_mean_numeric'))}. "
                    "This is channel-selection evidence, not final performance."
                )
    lines.extend(["", "## Recommended Next Full-Config Runs"])
    for rec in recommendations["recommended_channel_subset_runs"]:
        lines.append(
            f"- {rec['name']}: channels {rec['channels_to_use']} "
            f"({', '.join(rec['channel_names'])}). {rec['rationale']}"
        )
    lines.extend(["", "## Missing Or Limited Artifacts"])
    if missing:
        for item in sorted(set(missing)):
            lines.append(f"- {item}")
    else:
        lines.append("- No missing artifacts were detected for requested outputs.")
    lines.extend(
        [
            "",
            "## Output Tables",
            "- `stress_test_summary.csv`",
            "- `v3_metrics_summary.csv`",
            "- `latent_space_audit.csv`",
            "- `channel_ablation_comparison.csv`",
            "- `recommended_next_runs.json`",
            "",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.outdir.resolve()
    run_dir = args.run_dir.resolve()
    tensor_path = args.v3_tensor.resolve()
    metadata_path = args.v3_metadata.resolve()
    prepare_outdir(outdir, args.overwrite)

    missing: List[str] = []
    stress = build_stress_test_summary(missing)
    metrics = build_v3_metrics_summary(run_dir, missing)
    run_args = load_run_args(run_dir, missing)
    latent = build_latent_space_audit(
        run_dir=run_dir,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        run_args=run_args,
        device=args.device,
        skip_encoding=args.skip_latent_encoding,
        missing=missing,
    )
    ablation = build_channel_ablation_comparison(missing)
    recommendations = build_recommendations(stress, metrics, latent, ablation, missing)

    stress.to_csv(outdir / "stress_test_summary.csv", index=False)
    metrics.to_csv(outdir / "v3_metrics_summary.csv", index=False)
    latent.to_csv(outdir / "latent_space_audit.csv", index=False)
    ablation.to_csv(outdir / "channel_ablation_comparison.csv", index=False)
    def _sanitize_for_json(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    (outdir / "recommended_next_runs.json").write_text(
        json.dumps(_sanitize_for_json(recommendations), indent=2), encoding="utf-8"
    )
    write_readme(outdir, stress, metrics, latent, ablation, recommendations, missing)

    print(f"Audit written to: {outdir}")
    print(f"Stress rows: {len(stress)}")
    print(f"Metric rows: {len(metrics)}")
    print(f"Latent audit rows: {len(latent)}")
    print(f"Ablation rows: {len(ablation)}")
    print(f"Missing/limited artifacts: {len(set(missing))}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        raise SystemExit(main())
