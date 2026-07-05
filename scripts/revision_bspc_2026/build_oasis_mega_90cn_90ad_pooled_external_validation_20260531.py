#!/usr/bin/env python
"""Create a secondary mega-OASIS 90CN/90AD external validation package.

This package pools the previous OASIS pilot 30CN/30AD batch and the new
OASIS 60CN/60AD batch into harmonized tensors and frozen ADNI-model scoring
outputs. It is explicitly exploratory: no OASIS training, no OASIS model
selection, and no threshold fitting on the subjects used for final threshold
metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_oasis_tanda_20260525_external_adni import (  # noqa: E402
    AdniModelSpec,
    DEFAULT_PRIMARY_ADNI_RUN,
    PRIMARY_THRESHOLD_STRATEGY,
    binary_metrics as binary_metrics_from_pred,
    make_ensemble_predictions,
    score_one_model_on_one_tensor,
)
from score_oasis_next_60cn_60ad_external_20260530 import (  # noqa: E402
    MODEL_SPECS as RAW_MODEL_SPECS,
    SELECTED_CHANNEL_NAMES,
    score_subjects_all_folds,
    select_channels as select_channels_for_raw,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "oasis_mega_90cn_90ad_pooled_external_validation_20260531"

PILOT_CONNECTOME_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_connectomes"
PILOT_140_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_140TR_sensitivity"
PILOT_AUDIT_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_audit"

NEW_TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_20260530"
NEW_PILOT_PARITY_RUNWISE_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_pilot_parity_runwise_20260531"
NEW_HANDOFF_QC_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_processed_aal3_handoff_qc_20260530"
NEW_SPLIT_CSV = RESULTS_DIR / "oasis_60cn_60ad_calibration_test_protocol" / "split_calibration_test.csv"

BUILD_SOURCES = {
    "concatenated_timeseries": {
        "pilot": PILOT_CONNECTOME_DIR / "tensor_concatenated_timeseries.npz",
        "new": NEW_TENSOR_DIR / "tensor_concatenated_timeseries.npz",
        "construction_note": "Concatenate QC-usable ROI time series per subject/session, then compute connectomes.",
    },
    "runwise_140TR_pilot_parity": {
        "pilot": PILOT_140_DIR / "tensor_runwise_140TR_connectome_average.npz",
        "new": NEW_PILOT_PARITY_RUNWISE_DIR / "tensor_runwise_140TR_pilot_parity.npz",
        "construction_note": "Pilot-compatible runwise order: compute per-run 140TR connectomes, normalize each run, then average normalized run connectomes.",
    },
    "runwise164_pilot_parity": {
        "pilot": PILOT_CONNECTOME_DIR / "tensor_runwise_connectome_average.npz",
        "new": NEW_PILOT_PARITY_RUNWISE_DIR / "tensor_runwise164_pilot_parity.npz",
        "construction_note": "Pilot-compatible runwise order: compute per-run 164TR connectomes, normalize each run, then average normalized run connectomes.",
    },
}

CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
RNG_SEED = 42


@dataclass(frozen=True)
class LoadedTensor:
    label: str
    path: Path
    tensor: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    experiment_ids: np.ndarray
    diagnosis: np.ndarray
    channel_names: list[str]
    roi_names: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-adni-run-dir", type=Path, default=DEFAULT_PRIMARY_ADNI_RUN)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-recover035", action="store_true")
    return parser.parse_args()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def normalize_sex(value: Any) -> str:
    s = str(value).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def normalize_diag(value: Any) -> str:
    s = str(value).strip().upper()
    if s == "CN":
        return "CN"
    if s in {"AD", "AD_DEMENTIA", "DEMENTIA"}:
        return "AD_DEMENTIA"
    return str(value)


def y_from_diag(series: pd.Series) -> pd.Series:
    mapped = series.map({"CN": 0, "AD_DEMENTIA": 1, "AD": 1})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected diagnosis values: {bad}")
    return mapped.astype(int)


def load_npz_tensor(path: Path, label: str) -> LoadedTensor:
    require(path, f"{label} tensor")
    data = np.load(path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    channel_names = data["channel_names"].astype(str).tolist()
    roi_names = None
    for key in ("roi_names_in_order", "roi_names"):
        if key in data.files:
            roi_names = data[key].astype(str)
            break
    return LoadedTensor(
        label=label,
        path=path,
        tensor=tensor,
        subject_ids=data["subject_ids"].astype(str),
        session_ids=data["session_ids"].astype(str),
        experiment_ids=data["experiment_ids"].astype(str),
        diagnosis=np.array([normalize_diag(x) for x in data["diagnosis"].astype(str)]),
        channel_names=channel_names,
        roi_names=roi_names,
    )


def subject_key_frame(t: LoadedTensor) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SubjectID": t.subject_ids,
            "subject_id": t.subject_ids,
            "session_id": t.session_ids,
            "experiment_id": t.experiment_ids,
            "diagnosis": t.diagnosis,
        }
    )


def validate_tensor_pair(pilot: LoadedTensor, new: LoadedTensor, build_name: str) -> None:
    for label, t, expected_n in [("pilot", pilot, 60), ("new", new, 120)]:
        if t.tensor.shape != (expected_n, 3, 131, 131):
            raise ValueError(f"{build_name}/{label} has unexpected tensor shape: {t.tensor.shape}")
        if t.channel_names != CHANNEL_NAMES:
            raise ValueError(f"{build_name}/{label} channel mismatch: {t.channel_names}")
        if not np.isfinite(t.tensor).all():
            raise ValueError(f"{build_name}/{label} contains NaN or Inf values")
        diag = np.diagonal(t.tensor, axis1=2, axis2=3)
        if float(np.nanmax(np.abs(diag))) > 1e-6:
            raise ValueError(f"{build_name}/{label} has nonzero diagonal max={float(np.nanmax(np.abs(diag)))}")
        sym = np.nanmax(np.abs(t.tensor - np.swapaxes(t.tensor, -1, -2)))
        if float(sym) > 1e-5:
            raise ValueError(f"{build_name}/{label} is not symmetric, max abs diff={float(sym)}")
    if pilot.roi_names is not None and new.roi_names is not None:
        if len(pilot.roi_names) != len(new.roi_names) or not np.array_equal(pilot.roi_names, new.roi_names):
            raise ValueError(f"{build_name} ROI order mismatch between pilot and new tensors")


def save_mega_tensor(build_name: str, pilot: LoadedTensor, new: LoadedTensor, output_dir: Path) -> Path:
    validate_tensor_pair(pilot, new, build_name)
    tensor = np.concatenate([pilot.tensor, new.tensor], axis=0).astype(np.float32)
    subject_ids = np.concatenate([pilot.subject_ids, new.subject_ids]).astype(str)
    session_ids = np.concatenate([pilot.session_ids, new.session_ids]).astype(str)
    experiment_ids = np.concatenate([pilot.experiment_ids, new.experiment_ids]).astype(str)
    diagnosis = np.concatenate([pilot.diagnosis, new.diagnosis]).astype(str)
    out_path = output_dir / f"tensor_{build_name}.npz"
    kwargs: dict[str, Any] = {
        "global_tensor_data": tensor,
        "subject_ids": subject_ids,
        "session_ids": session_ids,
        "experiment_ids": experiment_ids,
        "diagnosis": diagnosis,
        "channel_names": np.array(CHANNEL_NAMES, dtype=object),
        "rois_count": np.array([131]),
        "roi_order_name": np.array(["ADNI_AAL3_131_locked_order"], dtype=object),
        "build_candidate": np.array([build_name], dtype=object),
        "pilot_tensor_path": np.array([str(pilot.path)], dtype=object),
        "new_tensor_path": np.array([str(new.path)], dtype=object),
        "external_validation_only": np.array([True]),
        "secondary_exploratory_pooled_oasis": np.array([True]),
    }
    if pilot.roi_names is not None:
        kwargs["roi_names_in_order"] = pilot.roi_names.astype(object)
    np.savez_compressed(out_path, **kwargs)
    return out_path


def load_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def aggregate_motion(motion_df: pd.DataFrame, batch: str) -> pd.DataFrame:
    if motion_df.empty:
        return pd.DataFrame(columns=["subject_id", "source_batch", "mean_fd_subject", "max_fd_subject", "n_motion_runs"])
    df = motion_df.copy()
    if "SubjectID" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"SubjectID": "subject_id"})
    if "mean_fd_jenkinson" in df.columns:
        mean_col = "mean_fd_jenkinson"
    elif "mean_fd" in df.columns:
        mean_col = "mean_fd"
    else:
        mean_col = None
    if "max_fd_jenkinson" in df.columns:
        max_col = "max_fd_jenkinson"
    elif "max_fd" in df.columns:
        max_col = "max_fd"
    else:
        max_col = None
    if "subject_id" not in df.columns or mean_col is None:
        return pd.DataFrame(columns=["subject_id", "source_batch", "mean_fd_subject", "max_fd_subject", "n_motion_runs"])
    agg = df.groupby("subject_id", dropna=False).agg(
        mean_fd_subject=(mean_col, "mean"),
        max_fd_subject=(max_col, "max") if max_col is not None else (mean_col, "max"),
        n_motion_runs=(mean_col, "count"),
    ).reset_index()
    agg["source_batch"] = batch
    return agg


def make_batch_manifest(base: pd.DataFrame, batch: str) -> pd.DataFrame:
    df = base.copy()
    if "SubjectID" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"SubjectID": "subject_id"})
    if "bids_session" in df.columns and "session_id" not in df.columns:
        df = df.rename(columns={"bids_session": "session_id"})
    df["source_batch"] = batch
    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].map(normalize_diag)
    if "age_at_MR" in df.columns:
        df["age_at_MR"] = pd.to_numeric(df["age_at_MR"], errors="coerce")
    if "sex" in df.columns:
        df["sex_normalized"] = df["sex"].map(normalize_sex)
    return df


def build_mega_manifest(output_dir: Path) -> pd.DataFrame:
    pilot_subjects = make_batch_manifest(pd.read_csv(PILOT_CONNECTOME_DIR / "subject_manifest.csv"), "pilot")
    new_subjects = make_batch_manifest(pd.read_csv(NEW_TENSOR_DIR / "subject_manifest.csv"), "new")

    pilot_run_manifest = load_optional_csv(PILOT_AUDIT_DIR / "run_manifest.csv")
    if not pilot_run_manifest.empty:
        cdr_cols = [c for c in ["subject_id", "session_id", "experiment_id", "CDRTOT", "CDRSUM"] if c in pilot_run_manifest.columns]
        if {"subject_id", "session_id"}.issubset(cdr_cols):
            cdr = pilot_run_manifest[cdr_cols].drop_duplicates(["subject_id", "session_id"])
            pilot_subjects = pilot_subjects.merge(cdr, on=["subject_id", "session_id"], how="left", suffixes=("", "_audit"))
            for c in ["CDRTOT", "CDRSUM"]:
                audit_c = f"{c}_audit"
                if audit_c in pilot_subjects.columns:
                    pilot_subjects[c] = pilot_subjects[c].combine_first(pilot_subjects[audit_c])
                    pilot_subjects = pilot_subjects.drop(columns=[audit_c])

    new_split = pd.read_csv(NEW_SPLIT_CSV)
    new_split_keep = [
        c
        for c in [
            "subject_id",
            "protocol_subset",
            "TR_seconds",
            "expected_tr2_rest_runs",
        ]
        if c in new_split.columns
    ]
    if new_split_keep:
        new_subjects = new_subjects.merge(new_split[new_split_keep].drop_duplicates("subject_id"), on="subject_id", how="left")

    pilot_motion = aggregate_motion(pilot_run_manifest, "pilot")
    new_motion = aggregate_motion(load_optional_csv(NEW_HANDOFF_QC_DIR / "motion_qc_by_file.csv"), "new")
    motion = pd.concat([pilot_motion, new_motion], ignore_index=True)

    manifest = pd.concat([pilot_subjects, new_subjects], ignore_index=True, sort=False)
    if not motion.empty:
        manifest = manifest.merge(motion, on=["subject_id", "source_batch"], how="left")
    for c in ["CDRTOT", "CDRSUM"]:
        if c not in manifest.columns:
            manifest[c] = np.nan

    keep_cols = [
        "subject_id",
        "session_id",
        "experiment_id",
        "source_batch",
        "protocol_subset",
        "diagnosis",
        "age_at_MR",
        "sex",
        "sex_normalized",
        "CDRTOT",
        "CDRSUM",
        "Manufacturer",
        "ScannerModel",
        "selected_qc_runs",
        "selected_run_ids",
        "selected_total_timepoints",
        "mean_fd_subject",
        "max_fd_subject",
        "n_motion_runs",
    ]
    for c in keep_cols:
        if c not in manifest.columns:
            manifest[c] = np.nan
    manifest = manifest[keep_cols].copy()
    manifest["diagnosis"] = manifest["diagnosis"].map(normalize_diag)
    manifest["y_true"] = y_from_diag(manifest["diagnosis"])
    manifest = manifest.sort_values(["source_batch", "diagnosis", "subject_id"]).reset_index(drop=True)
    write_csv_md(manifest, output_dir / "mega_manifest.csv", output_dir / "mega_manifest.md", "Mega-OASIS Manifest")
    return manifest


def overlap_audit(manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    pilot = manifest[manifest["source_batch"].eq("pilot")]
    new = manifest[manifest["source_batch"].eq("new")]
    rows = []
    for col in ["subject_id", "session_id", "experiment_id"]:
        p = set(pilot[col].dropna().astype(str))
        n = set(new[col].dropna().astype(str))
        rows.append(
            {
                "identifier": col,
                "pilot_unique": len(p),
                "new_unique": len(n),
                "overlap_n": len(p & n),
                "overlap_values": ";".join(sorted(p & n)[:20]),
            }
        )
    counts = manifest.groupby(["source_batch", "diagnosis"], dropna=False).size().reset_index(name="n")
    rows.append(
        {
            "identifier": "diagnosis_balance",
            "pilot_unique": int(counts[counts["source_batch"].eq("pilot")]["n"].sum()),
            "new_unique": int(counts[counts["source_batch"].eq("new")]["n"].sum()),
            "overlap_n": 0,
            "overlap_values": counts.to_json(orient="records"),
        }
    )
    df = pd.DataFrame(rows)
    write_csv_md(df, output_dir / "overlap_audit.csv", output_dir / "overlap_audit.md", "Pilot-New Overlap Audit")
    source_counts = manifest.groupby(["source_batch", "diagnosis"], dropna=False).size().reset_index(name="n")
    write_csv_md(source_counts, output_dir / "source_batch_diagnosis_counts.csv", output_dir / "source_batch_diagnosis_counts.md", "Source Batch Diagnosis Counts")
    return df


def tensor_qc(build_name: str, tensor_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = np.load(tensor_path, allow_pickle=True)
    x = np.asarray(data["global_tensor_data"], dtype=np.float32)
    diag = np.diagonal(x, axis1=2, axis2=3)
    sym = np.abs(x - np.swapaxes(x, -1, -2))
    diag_mask = np.eye(x.shape[-1], dtype=bool)
    off = x[:, :, ~diag_mask]
    qc = pd.DataFrame(
        [
            {
                "build_candidate": build_name,
                "tensor_path": str(tensor_path),
                "shape": "x".join(map(str, x.shape)),
                "subject_count": int(x.shape[0]),
                "channel_count": int(x.shape[1]),
                "roi_count": int(x.shape[-1]),
                "n_cn": int((data["diagnosis"].astype(str) == "CN").sum()),
                "n_ad_dementia": int((data["diagnosis"].astype(str) == "AD_DEMENTIA").sum()),
                "nan_count": int(np.isnan(x).sum()),
                "inf_count": int(np.isinf(x).sum()),
                "diag_abs_max": float(np.nanmax(np.abs(diag))),
                "symmetry_abs_max": float(np.nanmax(sym)),
                "channels": "|".join(data["channel_names"].astype(str).tolist()),
            }
        ]
    )
    rows = []
    for i, name in enumerate(data["channel_names"].astype(str).tolist()):
        vals = off[:, i, :].reshape(-1)
        rows.append(
            {
                "build_candidate": build_name,
                "channel_index": i,
                "channel_name": name,
                "offdiag_mean": float(np.nanmean(vals)),
                "offdiag_std": float(np.nanstd(vals)),
                "offdiag_min": float(np.nanmin(vals)),
                "offdiag_p01": float(np.nanpercentile(vals, 1)),
                "offdiag_median": float(np.nanmedian(vals)),
                "offdiag_p99": float(np.nanpercentile(vals, 99)),
                "offdiag_max": float(np.nanmax(vals)),
            }
        )
    return qc, pd.DataFrame(rows)


def metric_common(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> dict[str, Any]:
    return binary_metrics_from_pred(y_true, y_score, y_pred)


def threshold_sens_ge_070_max_spec(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    thresholds = np.sort(np.unique(score))[::-1]
    if thresholds.size == 0:
        return np.nan
    best_t = float(thresholds[-1]) - 1e-6
    best_spec = -1.0
    best_sens = -1.0
    for t in thresholds:
        pred = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if sens >= 0.70 and (spec > best_spec or (np.isclose(spec, best_spec) and sens > best_sens)):
            best_t = float(t)
            best_spec = float(spec)
            best_sens = float(sens)
    return best_t


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(float(fn(y_true[idx], y_score[idx])))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_auc_pvalue(y_true: np.ndarray, y_score: np.ndarray, n_perm: int, seed: int) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    obs = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y_true)
        val = float(roc_auc_score(yp, y_score))
        if val >= obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def pooled_ranking_metrics(predictions: pd.DataFrame, output_dir: Path, n_boot: int, n_perm: int) -> pd.DataFrame:
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    rows = []
    for (build, model), sub in ens.groupby(["build_candidate", "adni_model"], dropna=False):
        y = sub["y_true"].astype(int).to_numpy()
        s = sub["y_score"].astype(float).to_numpy()
        auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan
        pr = float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else np.nan
        auc_lo, auc_hi = bootstrap_metric(y, s, roc_auc_score, n_boot=n_boot, seed=RNG_SEED)
        pr_lo, pr_hi = bootstrap_metric(y, s, average_precision_score, n_boot=n_boot, seed=RNG_SEED + 1)
        rows.append(
            {
                "build_candidate": build,
                "adni_model": model,
                "prediction_level": "ensemble_mean_score_majority_vote",
                "analysis_role": "secondary_exploratory_pooled_90cn_90ad",
                "n": int(len(y)),
                "n_cn": int((y == 0).sum()),
                "n_ad": int((y == 1).sum()),
                "auc": auc,
                "auc_bootstrap95_lo": auc_lo,
                "auc_bootstrap95_hi": auc_hi,
                "auc_permutation_p_ge_observed": permutation_auc_pvalue(y, s, n_perm=n_perm, seed=RNG_SEED + 2),
                "pr_auc": pr,
                "pr_auc_bootstrap95_lo": pr_lo,
                "pr_auc_bootstrap95_hi": pr_hi,
                "n_bootstrap": n_boot,
                "n_permutations": n_perm,
            }
        )
    df = pd.DataFrame(rows)
    write_csv_md(df, output_dir / "pooled_ranking_metrics.csv", output_dir / "pooled_ranking_metrics.md", "Pooled 90CN/90AD Ranking Metrics")
    return df


def threshold_metrics(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    rows = []
    for (build, model), sub in ens.groupby(["build_candidate", "adni_model"], dropna=False):
        sub = sub.copy()
        # ADNI fixed threshold/majority vote over pooled 90/90.
        m = {
            "build_candidate": build,
            "adni_model": model,
            "threshold_strategy": "adni_fixed",
            "threshold_fit_subset": "ADNI_inner_OOF",
            "evaluation_subset": "pooled_90cn_90ad",
            "same_subject_threshold_fit_and_eval": False,
        }
        m.update(metric_common(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(m)

        pilot = sub[sub["source_batch"].eq("pilot")].copy()
        new = sub[sub["source_batch"].eq("new")].copy()
        if not pilot.empty and not new.empty and len(np.unique(pilot["y_true"])) == 2:
            t = threshold_sens_ge_070_max_spec(pilot["y_true"], pilot["y_score"])
            pred = (new["y_score"].to_numpy(float) >= t).astype(int)
            m = {
                "build_candidate": build,
                "adni_model": model,
                "threshold_strategy": "pilot_calibrated_sens_ge_0p70_max_spec",
                "threshold": t,
                "threshold_fit_subset": "pilot_30cn_30ad",
                "evaluation_subset": "new_60cn_60ad",
                "same_subject_threshold_fit_and_eval": False,
            }
            m.update(metric_common(new["y_true"], new["y_score"], pred))
            rows.append(m)

        cal = sub[sub["protocol_subset"].eq("calibration")].copy()
        test = sub[sub["protocol_subset"].eq("locked_test")].copy()
        if not cal.empty and not test.empty and len(np.unique(cal["y_true"])) == 2:
            t = threshold_sens_ge_070_max_spec(cal["y_true"], cal["y_score"])
            pred = (test["y_score"].to_numpy(float) >= t).astype(int)
            m = {
                "build_candidate": build,
                "adni_model": model,
                "threshold_strategy": "new_calibration_sens_ge_0p70_max_spec",
                "threshold": t,
                "threshold_fit_subset": "new_calibration_30cn_30ad",
                "evaluation_subset": "new_locked_test_30cn_30ad",
                "same_subject_threshold_fit_and_eval": False,
            }
            m.update(metric_common(test["y_true"], test["y_score"], pred))
            rows.append(m)
    df = pd.DataFrame(rows)
    write_csv_md(df, output_dir / "threshold_metrics.csv", output_dir / "threshold_metrics.md", "Threshold Metrics")
    return df


def stratified_metrics(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    rows = []
    ens["age_bin"] = pd.cut(
        pd.to_numeric(ens["Age"], errors="coerce"),
        bins=[-np.inf, 70, 80, np.inf],
        labels=["age_lt70", "age_70_79", "age_ge80"],
    ).astype(str)
    if "mean_fd_subject" in ens.columns and ens["mean_fd_subject"].notna().sum() >= 10:
        med = float(ens["mean_fd_subject"].median())
        ens["motion_bin"] = np.where(ens["mean_fd_subject"].astype(float) <= med, "motion_low", "motion_high")
    else:
        ens["motion_bin"] = "motion_unavailable"
    cdr = pd.to_numeric(ens.get("CDRTOT", np.nan), errors="coerce")
    ens["cdr_group"] = np.where(cdr.eq(0.5), "CDR_0p5", np.where(cdr.ge(1.0), "CDR_ge1", "CDR_other_or_missing"))

    strata = [
        ("source_batch", "source_batch"),
        ("sex", "Sex"),
        ("age_bin", "age_bin"),
        ("motion_bin", "motion_bin"),
        ("cdr_group", "cdr_group"),
    ]
    for (build, model), sub_model in ens.groupby(["build_candidate", "adni_model"], dropna=False):
        for strata_name, col in strata:
            for value, sub in sub_model.groupby(col, dropna=False):
                if len(sub) < 3:
                    continue
                m = {
                    "build_candidate": build,
                    "adni_model": model,
                    "stratum": strata_name,
                    "stratum_value": value,
                    "threshold_strategy": "adni_fixed",
                }
                m.update(metric_common(sub["y_true"], sub["y_score"], sub["y_pred"]))
                rows.append(m)
    df = pd.DataFrame(rows)
    write_csv_md(df, output_dir / "stratified_metrics.csv", output_dir / "stratified_metrics.md", "Stratified Metrics")
    return df


def score_primary_model(tensor_paths: dict[str, Path], manifest_by_build: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    device = resolve_device(args.device)
    spec = AdniModelSpec("primary_v5_1b_horizon4480_classifier_only", args.primary_adni_run_dir, True)
    frames = []
    for build, path in tensor_paths.items():
        lt = load_npz_tensor(path, build)
        subjects = manifest_by_build[build].copy()
        tensor_subjects = subject_key_frame(lt)
        key_cols = ["SubjectID", "session_id", "experiment_id"]
        subjects = tensor_subjects[key_cols + ["diagnosis"]].merge(
            subjects.drop(columns=["diagnosis"], errors="ignore"),
            left_on=["SubjectID", "session_id", "experiment_id"],
            right_on=["subject_id", "session_id", "experiment_id"],
            how="left",
            validate="one_to_one",
        )
        subjects["diagnosis"] = tensor_subjects["diagnosis"].values
        subjects["ResearchGroup_Mapped"] = subjects["diagnosis"].map({"CN": "CN", "AD_DEMENTIA": "AD"})
        subjects["y_true"] = y_from_diag(subjects["diagnosis"])
        subjects["Age"] = pd.to_numeric(subjects["age_at_MR"], errors="coerce")
        subjects["Sex"] = subjects["sex"].map(normalize_sex)
        if subjects["Age"].isna().any() or subjects["Sex"].eq("UNKNOWN").any():
            bad = subjects.loc[subjects["Age"].isna() | subjects["Sex"].eq("UNKNOWN"), ["SubjectID", "Age", "Sex"]]
            raise RuntimeError(f"Bad Age/Sex for primary scoring:\n{bad.to_string(index=False)}")
        pred = score_one_model_on_one_tensor(
            spec=spec,
            tensor=lt.tensor,
            oasis_subjects=subjects,
            oasis_channel_names=lt.channel_names,
            batch_size=int(args.batch_size),
            device=device,
        )
        pred["build_candidate"] = build
        frames.append(pred)
    fold_pred = pd.concat(frames, ignore_index=True)
    ens = make_ensemble_predictions(fold_pred)
    return pd.concat([fold_pred, ens], ignore_index=True, sort=False)


def score_recover035_models(
    tensor_paths: dict[str, Path],
    manifest_by_build: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> pd.DataFrame:
    device = resolve_device(args.device)
    specs = [s for s in RAW_MODEL_SPECS if s.label.startswith("recover035")]
    frames = []
    for spec in specs:
        if not spec.run_dir.exists():
            continue
        try:
            for fold in range(1, 6):
                require(spec.run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt", f"{spec.label} fold {fold} checkpoint")
                require(spec.run_dir / f"fold_{fold}" / f"classifier_logreg_raw_pipeline_fold_{fold}.joblib", f"{spec.label} fold {fold} raw classifier")
        except FileNotFoundError:
            continue
        for build, path in tensor_paths.items():
            lt = load_npz_tensor(path, build)
            subjects = manifest_by_build[build].copy()
            tensor_subjects = subject_key_frame(lt)
            key_cols = ["SubjectID", "session_id", "experiment_id"]
            subjects = tensor_subjects[key_cols + ["diagnosis"]].merge(
                subjects.drop(columns=["diagnosis"], errors="ignore"),
                left_on=["SubjectID", "session_id", "experiment_id"],
                right_on=["subject_id", "session_id", "experiment_id"],
                how="left",
                validate="one_to_one",
            )
            subjects["diagnosis"] = tensor_subjects["diagnosis"].values
            subjects["y_true"] = y_from_diag(subjects["diagnosis"])
            subjects["Age"] = pd.to_numeric(subjects["age_at_MR"], errors="coerce")
            subjects["Sex"] = subjects["sex"].map(normalize_sex)
            x = select_channels_for_raw(lt.tensor, lt.channel_names, SELECTED_CHANNEL_NAMES)
            pred = score_subjects_all_folds(
                spec=spec,
                x_subset=x,
                subjects_meta=subjects,
                batch_size=int(args.batch_size),
                device=device,
            )
            pred["adni_model"] = pred["model_label"]
            pred["build_candidate"] = build
            pred["threshold_strategy"] = PRIMARY_THRESHOLD_STRATEGY
            pred["prediction_level"] = np.where(pred["fold"].astype(str).eq("ensemble"), "ensemble_mean_score_majority_vote", "fold_model")
            pred["adni_threshold"] = pred["fold"].map(lambda f: np.mean(list(spec.adni_thresholds.values())) if str(f) == "ensemble" else spec.adni_thresholds[int(f)])
            pred["y_pred"] = (pred["y_score"].astype(float) >= pred["adni_threshold"].astype(float)).astype(int)
            # recover metadata lost by the raw scorer.
            extra_cols = [
                "subject_id",
                "source_batch",
                "protocol_subset",
                "CDRTOT",
                "CDRSUM",
                "selected_qc_runs",
                "selected_run_ids",
                "selected_total_timepoints",
                "mean_fd_subject",
                "max_fd_subject",
                "n_motion_runs",
            ]
            extra = subjects[[c for c in extra_cols if c in subjects.columns]].copy()
            extra = extra.rename(columns={"subject_id": "SubjectID"})
            pred = pred.merge(extra.drop_duplicates("SubjectID"), on="SubjectID", how="left")
            frames.append(pred)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_manifest_by_build(tensor_paths: dict[str, Path], mega_manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for build, path in tensor_paths.items():
        lt = load_npz_tensor(path, build)
        keys = subject_key_frame(lt)[["subject_id", "session_id", "experiment_id"]]
        out[build] = keys.merge(mega_manifest, on=["subject_id", "session_id", "experiment_id"], how="left", validate="one_to_one")
    return out


def write_readme(output_dir: Path, scored: bool, ranking: pd.DataFrame | None, threshold: pd.DataFrame | None) -> None:
    lines = [
        "# Mega-OASIS 90CN/90AD Pooled External Validation",
        "",
        "Status: `" + ("complete" if scored else "tensors_and_manifest_only") + "`",
        "",
        "This is a secondary/exploratory pooled analysis combining the OASIS pilot",
        "30CN/30AD batch and the new non-overlapping 60CN/60AD batch. The new",
        "60CN/60AD calibration/test protocol remains the primary pre-specified",
        "external analysis.",
        "",
        "## Guardrails",
        "",
        "- No OASIS data were used to train VAE or classifier weights.",
        "- No OASIS data were used to select the ADNI model.",
        "- No threshold was fit on the same subjects used for final threshold metrics.",
        "- The pooled 90CN/90AD ranking analysis is secondary/exploratory.",
        "- Existing pilot/new tensors were not overwritten.",
        "",
        "## Tensor Builds",
        "",
        "- `concatenated_timeseries`: harmonized by stacking pilot and new concatenated tensors.",
        "- `runwise_140TR_pilot_parity`: pilot 140TR plus new pilot-parity 140TR runwise tensor.",
        "- `runwise164_pilot_parity`: pilot runwise164 plus new pilot-parity runwise164 tensor.",
    ]
    if scored and ranking is not None:
        lines += ["", "## Pooled Ranking Metrics", "", ranking.to_markdown(index=False)]
    if scored and threshold is not None:
        lines += ["", "## Threshold Metrics", "", threshold.to_markdown(index=False)]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_recommendation(output_dir: Path, ranking: pd.DataFrame | None, threshold: pd.DataFrame | None) -> None:
    lines = [
        "# Final Recommendation",
        "",
        "Decision: `secondary_exploratory_external_analysis_only`",
        "",
        "The mega-OASIS 90CN/90AD package is suitable for pooled descriptive",
        "external-validation reporting, but it must not be used to select the ADNI",
        "model. The pre-specified new 60CN/60AD calibration/test split remains the",
        "primary external analysis.",
    ]
    if ranking is not None and not ranking.empty:
        top = ranking.sort_values(["auc", "pr_auc"], ascending=False).iloc[0]
        lines += [
            "",
            "## Highest Pooled Ranking Row",
            "",
            f"- build_candidate: `{top['build_candidate']}`",
            f"- adni_model: `{top['adni_model']}`",
            f"- AUC: `{float(top['auc']):.6f}`",
            f"- PR-AUC: `{float(top['pr_auc']):.6f}`",
        ]
    if threshold is not None and not threshold.empty:
        lines += [
            "",
            "Threshold-calibrated rows are provided only where the threshold was fit on",
            "a disjoint calibration set and evaluated on held-out subjects.",
        ]
    lines += [
        "",
        "No OASIS-based model promotion is recommended from this pooled analysis.",
    ]
    output_dir.joinpath("final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_rows = []
    for build, sources in BUILD_SOURCES.items():
        for batch in ["pilot", "new"]:
            p = sources[batch]
            artifact_rows.append({"artifact_type": "source_tensor", "build_candidate": build, "source_batch": batch, "path": str(p), "exists": p.exists()})
            require(p, f"{build} {batch} source tensor")
    for p, label in [
        (PILOT_CONNECTOME_DIR / "subject_manifest.csv", "pilot subject manifest"),
        (NEW_TENSOR_DIR / "subject_manifest.csv", "new subject manifest"),
        (NEW_SPLIT_CSV, "new fixed calibration/test split"),
        (args.primary_adni_run_dir / "classifier_only_readout", "primary ADNI Stage B readout"),
    ]:
        artifact_rows.append({"artifact_type": label, "build_candidate": "", "source_batch": "", "path": str(p), "exists": p.exists()})
        require(p, label)
    for spec in [s for s in RAW_MODEL_SPECS if s.label.startswith("recover035")]:
        artifact_rows.append(
            {
                "artifact_type": "optional_recover035_model",
                "build_candidate": "",
                "source_batch": "",
                "path": str(spec.run_dir),
                "exists": spec.run_dir.exists(),
                "label": spec.label,
            }
        )
        for fold in range(1, 6):
            for rel in [
                f"fold_{fold}/vae_model_fold_{fold}.pt",
                f"fold_{fold}/classifier_logreg_raw_pipeline_fold_{fold}.joblib",
                f"fold_{fold}/feature_columns.json",
            ]:
                p = spec.run_dir / rel
                artifact_rows.append(
                    {
                        "artifact_type": "optional_recover035_fold_artifact",
                        "build_candidate": "",
                        "source_batch": "",
                        "path": str(p),
                        "exists": p.exists(),
                        "label": f"{spec.label}_fold{fold}",
                    }
                )
    artifact_df = pd.DataFrame(artifact_rows)
    write_csv_md(artifact_df, output_dir / "artifact_validation.csv", output_dir / "artifact_validation.md", "Artifact Validation")

    mega_manifest = build_mega_manifest(output_dir)
    if len(mega_manifest) != 180:
        raise RuntimeError(f"Mega manifest expected 180 rows, found {len(mega_manifest)}")
    counts = mega_manifest["diagnosis"].value_counts().to_dict()
    if counts.get("CN", 0) != 90 or counts.get("AD_DEMENTIA", 0) != 90:
        raise RuntimeError(f"Mega manifest is not 90 CN / 90 AD_DEMENTIA: {counts}")
    overlap = overlap_audit(mega_manifest, output_dir)
    if int(overlap.loc[overlap["identifier"].eq("subject_id"), "overlap_n"].iloc[0]) != 0:
        raise RuntimeError("Pilot/new subject overlap detected")

    tensor_paths: dict[str, Path] = {}
    tensor_manifest_rows = []
    qc_frames = []
    channel_frames = []
    reference_ids: list[tuple[str, str, str]] | None = None
    for build, sources in BUILD_SOURCES.items():
        pilot = load_npz_tensor(sources["pilot"], f"{build}_pilot")
        new = load_npz_tensor(sources["new"], f"{build}_new")
        out_path = save_mega_tensor(build, pilot, new, output_dir)
        tensor_paths[build] = out_path
        ids = list(zip(np.concatenate([pilot.subject_ids, new.subject_ids]), np.concatenate([pilot.session_ids, new.session_ids]), np.concatenate([pilot.experiment_ids, new.experiment_ids])))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Subject/session/experiment order differs for build {build}")
        tensor_manifest_rows.append(
            {
                "build_candidate": build,
                "tensor_path": str(out_path),
                "pilot_source_tensor": str(sources["pilot"]),
                "new_source_tensor": str(sources["new"]),
                "construction_note": sources["construction_note"],
            }
        )
        qc, ch = tensor_qc(build, out_path, output_dir)
        qc_frames.append(qc)
        channel_frames.append(ch)
    tensor_manifest = pd.DataFrame(tensor_manifest_rows)
    write_csv_md(tensor_manifest, output_dir / "tensor_manifest.csv", output_dir / "tensor_manifest.md", "Tensor Manifest")
    tensor_qc_df = pd.concat(qc_frames, ignore_index=True)
    channel_df = pd.concat(channel_frames, ignore_index=True)
    write_csv_md(tensor_qc_df, output_dir / "tensor_qc_summary.csv", output_dir / "tensor_qc_summary.md", "Tensor QC Summary")
    write_csv_md(channel_df, output_dir / "channel_statistics.csv", output_dir / "channel_statistics.md", "Channel Statistics")

    ranking = None
    thresholds = None
    predictions = pd.DataFrame()
    scored = not args.skip_scoring
    if scored:
        manifest_by_build = build_manifest_by_build(tensor_paths, mega_manifest)
        primary_pred = score_primary_model(tensor_paths, manifest_by_build, args)
        frames = [primary_pred]
        if not args.skip_recover035:
            recover_pred = score_recover035_models(tensor_paths, manifest_by_build, args)
            if not recover_pred.empty:
                frames.append(recover_pred)
        predictions = pd.concat(frames, ignore_index=True, sort=False)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        ranking = pooled_ranking_metrics(predictions, output_dir, n_boot=int(args.n_bootstrap), n_perm=int(args.n_permutations))
        thresholds = threshold_metrics(predictions, output_dir)
        stratified = stratified_metrics(predictions, output_dir)
        score_dist = (
            predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")]
            .groupby(["build_candidate", "adni_model", "source_batch", "diagnosis"], dropna=False)["y_score"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "score_mean", "std": "score_std", "min": "score_min", "median": "score_median", "max": "score_max"})
        )
        write_csv_md(score_dist, output_dir / "score_distribution_by_source_diagnosis.csv", output_dir / "score_distribution_by_source_diagnosis.md", "Score Distribution By Source And Diagnosis")

    write_readme(output_dir, scored=scored, ranking=ranking, threshold=thresholds)
    write_final_recommendation(output_dir, ranking=ranking, threshold=thresholds)
    write_json(
        output_dir / "command_log.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "output_dir": str(output_dir),
            "mode": "score_and_package" if scored else "package_only",
            "secondary_exploratory_pooled_oasis": True,
            "oasis_training": False,
            "oasis_model_selection": False,
            "same_subject_threshold_fit_and_eval": False,
            "primary_external_analysis_remains_new_60cn_60ad_protocol": True,
            "n_bootstrap": int(args.n_bootstrap),
            "n_permutations": int(args.n_permutations),
            "tensor_paths": {k: str(v) for k, v in tensor_paths.items()},
            "prediction_rows": int(len(predictions)) if not predictions.empty else 0,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "scored": scored, "n_predictions": int(len(predictions))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
