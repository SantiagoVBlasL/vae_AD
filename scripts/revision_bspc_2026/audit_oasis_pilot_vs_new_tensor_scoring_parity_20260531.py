#!/usr/bin/env python3
"""Read-only parity audit for OASIS pilot vs new 60CN/60AD scoring.

The audit compares tensor construction, scoring implementation, cohort overlap,
score distributions, and cross-scoring behavior. It writes only to a new audit
folder and does not modify tensors, ADNI model outputs, or OASIS inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision_bspc_2026"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_oasis_next_60cn_60ad_external_20260530 as next_scorer  # noqa: E402
import score_oasis_tanda_20260525_external_adni as pilot_scorer  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"

PILOT_CONNECTOME_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_connectomes"
PILOT_140TR_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_140TR_sensitivity"
PILOT_SCORING_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_external_scoring"
PILOT_AUDIT_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_audit"
PILOT_POSTMORTEM_DIR = RESULTS_DIR / "oasis_tanda_2026_05_25_external_scoring_postmortem"

NEW_TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_20260530"
NEW_TENSOR_QC_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_qc_20260530"
NEW_SCORING_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_20260530"
NEW_POSTMORTEM_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_postmortem_20260530"
NEW_HANDOFF_QC_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_processed_aal3_handoff_qc_20260530"

OUTPUT_DIR = RESULTS_DIR / "oasis_pilot_vs_new_tensor_scoring_parity_audit_20260531"

ADNI_TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
ADNI_METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

CHANNELS = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]

PILOT_TENSOR_SPECS = {
    "pilot_concatenated_timeseries": (PILOT_CONNECTOME_DIR, "tensor_concatenated_timeseries.npz"),
    "pilot_runwise164_connectome_average": (PILOT_CONNECTOME_DIR, "tensor_runwise_connectome_average.npz"),
    "pilot_runwise_140TR_connectome_average": (PILOT_140TR_DIR, "tensor_runwise_140TR_connectome_average.npz"),
}
NEW_TENSOR_SPECS = {
    "new_concatenated_timeseries": (NEW_TENSOR_DIR, "tensor_concatenated_timeseries.npz"),
    "new_runwise_140TR_connectome_average": (NEW_TENSOR_DIR, "tensor_runwise_140TR_connectome_average.npz"),
    "new_runwise164_connectome_average": (NEW_TENSOR_DIR, "tensor_runwise164_connectome_average.npz"),
}
MAX_KS_SAMPLE = 200_000
RNG_SEED = 20260531


@dataclass(frozen=True)
class TensorBundle:
    cohort: str
    build_label: str
    tensor_path: Path
    manifest_path: Path
    tensor: np.ndarray
    subjects: pd.DataFrame
    channel_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-cross-score", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(map(str, view.columns)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals: list[str] = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.8g}" if np.isfinite(val) else "")
            elif pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def normalize_sex(value: Any) -> str:
    s = str(value).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def load_tensor_bundle(cohort: str, build_label: str, tensor_dir: Path, tensor_file: str) -> TensorBundle:
    tensor_path = tensor_dir / tensor_file
    require(tensor_path, f"{cohort} tensor {build_label}")
    with np.load(tensor_path, allow_pickle=True) as npz:
        tensor = np.asarray(npz["global_tensor_data"], dtype=np.float32)
        channel_names = np.asarray(npz["channel_names"]).astype(str).tolist()
        subjects = pd.DataFrame(
            {
                "SubjectID": np.asarray(npz["subject_ids"]).astype(str),
                "session_id": np.asarray(npz["session_ids"]).astype(str) if "session_ids" in npz.files else "",
                "experiment_id": np.asarray(npz["experiment_ids"]).astype(str) if "experiment_ids" in npz.files else "",
                "diagnosis": np.asarray(npz["diagnosis"]).astype(str) if "diagnosis" in npz.files else "",
            }
        )

    manifest_path = tensor_dir / "subject_manifest.csv"
    require(manifest_path, f"{cohort} subject manifest {build_label}")
    manifest = pd.read_csv(manifest_path).rename(columns={"subject_id": "SubjectID"})
    keep = [
        c
        for c in [
            "SubjectID",
            "session_id",
            "experiment_id",
            "Manufacturer",
            "ScannerModel",
            "age_at_MR",
            "sex",
            "selected_qc_runs",
            "selected_run_ids",
            "selected_total_timepoints",
            "temporal_selection_rule",
        ]
        if c in manifest.columns
    ]
    merge_keys = [c for c in ["SubjectID", "session_id"] if c in keep and c in subjects.columns]
    subjects = subjects.merge(
        manifest[keep].drop_duplicates(merge_keys),
        on=merge_keys,
        how="left",
        suffixes=("", "_manifest"),
    )
    if "experiment_id_manifest" in subjects.columns:
        subjects["experiment_id"] = subjects["experiment_id"].where(
            subjects["experiment_id"].astype(str).ne(""),
            subjects["experiment_id_manifest"],
        )
    subjects["ResearchGroup_Mapped"] = subjects["diagnosis"].map({"CN": "CN", "AD_DEMENTIA": "AD", "AD": "AD"})
    subjects["y_true"] = subjects["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    subjects["Age"] = pd.to_numeric(subjects.get("age_at_MR"), errors="coerce")
    subjects["Sex"] = subjects.get("sex", pd.Series(["UNKNOWN"] * len(subjects))).map(normalize_sex)
    return TensorBundle(cohort, build_label, tensor_path, manifest_path, tensor, subjects, channel_names)


def selected_channels(tensor: np.ndarray, channel_names: Sequence[str], target: Sequence[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(channel_names)}
    missing = [name for name in target if name not in idx]
    if missing:
        raise ValueError(f"Missing channels {missing}; available={list(channel_names)}")
    return tensor[:, [idx[name] for name in target], :, :]


def safe_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def safe_pr_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, score))


def binary_metrics(y_true: Sequence[int], score: Sequence[float], pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    p = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "auc": safe_auc(y, s),
        "pr_auc": safe_pr_auc(y, s),
        "reversed_auc": 1.0 - safe_auc(y, s) if np.isfinite(safe_auc(y, s)) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
        "accuracy": float((tp + tn) / len(y)) if len(y) else np.nan,
    }


def metrics_from_prediction_table(pred: pd.DataFrame, model_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in pred.groupby(["build_candidate", model_col, "prediction_level"], dropna=False):
        build, model, level = keys
        row = {
            "build_candidate": build,
            "model_label": model,
            "prediction_level": level,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def score_pilot_with_next_scorer(
    bundles: Sequence[TensorBundle],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metric_rows = []
    for bundle in bundles:
        x = selected_channels(bundle.tensor, bundle.channel_names, next_scorer.SELECTED_CHANNEL_NAMES)
        for spec in next_scorer.MODEL_SPECS:
            pred = next_scorer.score_subjects_all_folds(spec, x, bundle.subjects, batch_size, device)
            pred["build_candidate"] = bundle.build_label
            pred["scoring_logic"] = "next_60cn60ad_scorer_saved_raw_pipelines"
            ens = pred[pred["fold"].astype(str).eq("ensemble")].copy()
            adni_thr = next_scorer.ensemble_adni_threshold(spec)
            ens["y_pred"] = (ens["y_score"].astype(float) >= adni_thr).astype(int)
            row = {
                "cohort": bundle.cohort,
                "build_candidate": bundle.build_label,
                "model_label": spec.label,
                "prediction_level": "ensemble",
                "scoring_logic": "next_60cn60ad_scorer_saved_raw_pipelines",
                "threshold_source": "mean_hardcoded_adni_thresholds_in_next_scorer",
                "threshold": adni_thr,
            }
            row.update(binary_metrics(ens["y_true"], ens["y_score"], ens["y_pred"]))
            metric_rows.append(row)
            pred["adni_threshold_ensemble"] = adni_thr
            rows.append(pred)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metric_rows)


def score_new_with_pilot_scorer(
    bundles: Sequence[TensorBundle],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = pilot_scorer.AdniModelSpec(
        "primary_v5_1b_ch1_0_2_horizon4480",
        pilot_scorer.DEFAULT_PRIMARY_ADNI_RUN,
        True,
    )
    pred_rows = []
    metric_rows = []
    for bundle in bundles:
        fold_pred = pilot_scorer.score_one_model_on_one_tensor(
            spec,
            bundle.tensor,
            bundle.subjects,
            bundle.channel_names,
            batch_size,
            device,
        )
        fold_pred["build_candidate"] = bundle.build_label
        ens = pilot_scorer.make_ensemble_predictions(fold_pred)
        all_pred = pd.concat([fold_pred, ens], ignore_index=True, sort=False)
        all_pred["scoring_logic"] = "pilot_scorer_stageb_reconstructed_from_adni_trainDev"
        metrics, _, _, _ = pilot_scorer.metrics_tables(all_pred)
        metrics["cohort"] = bundle.cohort
        metrics["scoring_logic"] = "pilot_scorer_stageb_reconstructed_from_adni_trainDev"
        metrics = metrics.rename(columns={"adni_model": "model_label"})
        metric_rows.append(metrics)
        pred_rows.append(all_pred)
    return pd.concat(pred_rows, ignore_index=True), pd.concat(metric_rows, ignore_index=True)


def cohort_balance_and_overlap(pilot_subjects: pd.DataFrame, new_subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in [("pilot_30CN_30AD", pilot_subjects), ("new_60CN_60AD", new_subjects)]:
        rows.append(
            {
                "cohort": label,
                "n_rows": int(len(df)),
                "n_subjects": int(df["SubjectID"].nunique()),
                "n_experiment_ids": int(df["experiment_id"].nunique()),
                "n_cn": int(df["diagnosis"].eq("CN").sum()),
                "n_ad_dementia": int(df["diagnosis"].isin(["AD_DEMENTIA", "AD"]).sum()),
                "age_mean": float(pd.to_numeric(df["Age"], errors="coerce").mean()),
                "age_sd": float(pd.to_numeric(df["Age"], errors="coerce").std(ddof=1)),
                "sex_counts": ";".join(f"{k}:{v}" for k, v in df["Sex"].value_counts(dropna=False).sort_index().items()),
                "selected_qc_runs_mean": float(pd.to_numeric(df.get("selected_qc_runs"), errors="coerce").mean()),
            }
        )
    pilot_ids = set(pilot_subjects["SubjectID"].astype(str))
    new_ids = set(new_subjects["SubjectID"].astype(str))
    pilot_exp = set(pilot_subjects["experiment_id"].astype(str))
    new_exp = set(new_subjects["experiment_id"].astype(str))
    rows.append(
        {
            "cohort": "overlap",
            "n_rows": np.nan,
            "n_subjects": len(pilot_ids & new_ids),
            "n_experiment_ids": len(pilot_exp & new_exp),
            "n_cn": np.nan,
            "n_ad_dementia": np.nan,
            "age_mean": np.nan,
            "age_sd": np.nan,
            "sex_counts": "subject_overlap=" + ";".join(sorted(pilot_ids & new_ids)),
            "selected_qc_runs_mean": np.nan,
        }
    )
    return pd.DataFrame(rows)


def script_parity_table() -> pd.DataFrame:
    paths = [
        SCRIPT_DIR / "build_oasis_tanda_20260525_connectomes.py",
        SCRIPT_DIR / "oasis_tanda_20260525_140tr_sensitivity.py",
        SCRIPT_DIR / "build_oasis_next_60cn_60ad_tensor_20260530.py",
        SCRIPT_DIR / "score_oasis_tanda_20260525_external_adni.py",
        SCRIPT_DIR / "score_oasis_next_60cn_60ad_external_20260530.py",
    ]
    rows = []
    for p in paths:
        rows.append(
            {
                "script": p.name,
                "path": str(p),
                "sha256": sha256(p),
                "exists": p.exists(),
            }
        )
    return pd.DataFrame(rows)


def tensor_construction_parity() -> pd.DataFrame:
    rows = [
        {
            "component": "ROI mapping/order",
            "pilot": "131-row audited AAL3-to-ADNI mapping; roi_order_confirmed=True",
            "new": "Reuses Tanda/preflight ROI mapping; 131-row ADNI final order",
            "parity": "matched_by_design",
            "risk": "low",
        },
        {
            "component": "channel names/order",
            "pilot": ";".join(CHANNELS),
            "new": ";".join(CHANNELS),
            "parity": "matched",
            "risk": "low",
        },
        {
            "component": "Pearson Full",
            "pilot": "np.corrcoef(rowvar=False) -> Fisher r-to-z -> diagonal zero",
            "new": "np.corrcoef(rowvar=False) -> Fisher r-to-z -> diagonal zero",
            "parity": "matched",
            "risk": "low",
        },
        {
            "component": "OMST",
            "pilot": "abs(FisherZ) threshold_omst_global_cost_efficiency, signed mask reapplied",
            "new": "abs(FisherZ) threshold_omst_global_cost_efficiency, signed mask reapplied",
            "parity": "matched",
            "risk": "low",
        },
        {
            "component": "MI-KNN",
            "pilot": "sklearn mutual_info_regression, n_neighbors=5, symmetric average",
            "new": "sklearn mutual_info_regression, n_neighbors=5, symmetric average",
            "parity": "matched",
            "risk": "low",
        },
        {
            "component": "Per-channel scaling",
            "pilot": "RobustScaler on off-diagonal values per subject/channel",
            "new": "RobustScaler on off-diagonal values per subject/channel",
            "parity": "matched_for_concatenated",
            "risk": "medium",
        },
        {
            "component": "Runwise averaging order",
            "pilot": "compute_channels per run, normalize each run, then average normalized connectomes",
            "new": "compute raw channels per run, average raw connectomes, then normalize averaged connectome",
            "parity": "mismatch",
            "risk": "high_for_runwise_comparisons",
        },
        {
            "component": "140TR rule",
            "pilot": "first 140TR per run, normalize per run, then average",
            "new": "first 140TR per run, average raw, then normalize",
            "parity": "mismatch",
            "risk": "high_for_140TR_comparisons",
        },
        {
            "component": "Concatenated run handling",
            "pilot": "concatenate QC-ok runs in time then compute/normalize once",
            "new": "concatenate QC-ok runs in time then compute/normalize once",
            "parity": "matched_algorithmically",
            "risk": "cohort_or_QC_dependent",
        },
    ]
    return pd.DataFrame(rows)


def scoring_script_parity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "Primary ADNI model path",
                "pilot_scorer": str(pilot_scorer.DEFAULT_PRIMARY_ADNI_RUN),
                "new_scorer": str(next_scorer.ADNI_V5_1B_RUN),
                "parity": "mismatch",
                "risk": "high",
            },
            {
                "component": "Classifier/readout",
                "pilot_scorer": "Reconstructs Stage B classifier-only logreg_l2 from ADNI trainDev latent cache and saved C",
                "new_scorer": "Applies saved fold classifier_logreg_raw_pipeline from run folder",
                "parity": "mismatch",
                "risk": "high",
            },
            {
                "component": "Thresholds",
                "pilot_scorer": "Per-fold classifier_only_readout thresholds; ensemble majority vote over fold predictions",
                "new_scorer": "Hard-coded per-fold thresholds averaged into one ensemble threshold",
                "parity": "mismatch",
                "risk": "high",
            },
            {
                "component": "Age/Sex",
                "pilot_scorer": "Age from subject_manifest age_at_MR; Sex normalized 1/2 or M/F",
                "new_scorer": "Age/Sex from locked calibration/test split after merge patch",
                "parity": "compatible_after_patch",
                "risk": "low",
            },
            {
                "component": "Fold ensemble score",
                "pilot_scorer": "Mean fold score; y_pred by majority vote of fold threshold decisions",
                "new_scorer": "Mean calibrated/raw fold score; y_pred by thresholding ensemble mean score",
                "parity": "mismatch",
                "risk": "medium_high",
            },
            {
                "component": "OOF-logitz",
                "pilot_scorer": "Not part of original primary pilot scorer",
                "new_scorer": "Implemented for recover035_oof_logitz only",
                "parity": "not_comparable",
                "risk": "medium",
            },
        ]
    )


def finite_quantiles(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: np.nan for k in ["mean", "std", "min", "q25", "median", "q75", "max"]}
    q25, med, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(q25),
        "median": float(med),
        "q75": float(q75),
        "max": float(np.max(arr)),
    }


def offdiag_flat(tensor: np.ndarray, channel_idx: int) -> np.ndarray:
    n = tensor.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return tensor[:, channel_idx, :, :][:, mask].reshape(-1)


def tensor_distribution_table(bundles: Sequence[TensorBundle]) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    with np.load(ADNI_TENSOR_PATH, allow_pickle=True) as npz:
        adni_tensor = np.asarray(npz["global_tensor_data"], dtype=np.float32)
        adni_channels = np.asarray(npz["channel_names"]).astype(str).tolist()
    adni_refs: Dict[str, np.ndarray] = {}
    for channel in CHANNELS:
        idx = adni_channels.index(channel)
        vals = offdiag_flat(adni_tensor, idx)
        adni_refs[channel] = vals[np.isfinite(vals)]
        rows.append(
            {
                "cohort": "ADNI_v5_1b_140TR_all_training_ready",
                "build_candidate": "ADNI_reference",
                "channel_name": channel,
                "n_subjects": int(adni_tensor.shape[0]),
                **finite_quantiles(vals),
                "ks_stat_vs_adni": 0.0,
                "ks_p_vs_adni": 1.0,
            }
        )

    for bundle in bundles:
        for channel in CHANNELS:
            idx = bundle.channel_names.index(channel)
            vals = offdiag_flat(bundle.tensor, idx)
            row = {
                "cohort": bundle.cohort,
                "build_candidate": bundle.build_label,
                "channel_name": channel,
                "n_subjects": int(bundle.tensor.shape[0]),
                **finite_quantiles(vals),
            }
            ref = adni_refs[channel]
            finite = vals[np.isfinite(vals)]
            row["mean_minus_adni"] = float(row["mean"] - np.mean(ref))
            row["std_ratio_vs_adni"] = float(row["std"] / np.std(ref)) if np.std(ref) else np.nan
            if stats is not None and finite.size:
                a = ref
                b = finite
                if a.size > MAX_KS_SAMPLE:
                    a = rng.choice(a, size=MAX_KS_SAMPLE, replace=False)
                if b.size > MAX_KS_SAMPLE:
                    b = rng.choice(b, size=MAX_KS_SAMPLE, replace=False)
                ks = stats.ks_2samp(a, b)
                row["ks_stat_vs_adni"] = float(ks.statistic)
                row["ks_p_vs_adni"] = float(ks.pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def existing_metric_comparison() -> pd.DataFrame:
    rows = []
    pilot_metrics = pd.read_csv(PILOT_SCORING_DIR / "primary_metrics.csv")
    pilot_primary = pilot_metrics[pilot_metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    for _, row in pilot_primary.iterrows():
        rows.append(
            {
                "source": "pilot_original_scoring",
                "cohort": "pilot_30CN_30AD",
                "build_candidate": row["build_candidate"],
                "model_label": row["adni_model"],
                "prediction_level": row["prediction_level"],
                "threshold_strategy": row["threshold_strategy"],
                "auc": row["auc"],
                "pr_auc": row["pr_auc"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "balanced_accuracy": row["balanced_accuracy"],
                "f1": row["f1"],
            }
        )

    sens140 = pd.read_csv(PILOT_140TR_DIR / "external_scoring_metrics.csv")
    sens140 = sens140[sens140["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    for _, row in sens140.iterrows():
        rows.append(
            {
                "source": "pilot_140TR_sensitivity_scoring",
                "cohort": "pilot_30CN_30AD",
                "build_candidate": row["build_candidate"],
                "model_label": row["adni_model"],
                "prediction_level": row["prediction_level"],
                "threshold_strategy": row["threshold_strategy"],
                "auc": row["auc"],
                "pr_auc": row["pr_auc"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "balanced_accuracy": row["balanced_accuracy"],
                "f1": row["f1"],
            }
        )

    new_metrics = pd.read_csv(NEW_SCORING_DIR / "metrics_locked_test.csv")
    new_primary = new_metrics[new_metrics["threshold_strategy"].eq("adni_fixed")].copy()
    for _, row in new_primary.iterrows():
        rows.append(
            {
                "source": "new_original_scoring_locked_test",
                "cohort": "new_60CN_60AD_locked_test",
                "build_candidate": row["build_candidate"],
                "model_label": row["model_label"],
                "prediction_level": "ensemble",
                "threshold_strategy": row["threshold_strategy"],
                "auc": row["auc"],
                "pr_auc": row["pr_auc"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "balanced_accuracy": row["balanced_accuracy"],
                "f1": row["f1"],
            }
        )
    return pd.DataFrame(rows)


def score_distribution_from_predictions(pred: pd.DataFrame, model_col: str, source: str, cohort: str) -> pd.DataFrame:
    if "prediction_level" in pred.columns:
        sub = pred[pred["prediction_level"].astype(str).isin(["ensemble_mean_score_majority_vote", "ensemble"])]
    else:
        sub = pred.copy()
    rows = []
    for (build, model, diagnosis), g in sub.groupby(["build_candidate", model_col, "diagnosis"], dropna=False):
        q = finite_quantiles(g["y_score"].to_numpy(float))
        rows.append(
            {
                "source": source,
                "cohort": cohort,
                "build_candidate": build,
                "model_label": model,
                "diagnosis": diagnosis,
                "n": int(len(g)),
                **{f"score_{k}": v for k, v in q.items()},
            }
        )
    return pd.DataFrame(rows)


def load_pilot_motion_summary() -> pd.DataFrame:
    file_inv = pd.read_csv(PILOT_AUDIT_DIR / "file_inventory.csv")
    fd = file_inv[file_inv["file_path"].astype(str).str.contains("FD_Jenkinson", na=False)].copy()
    rows = []
    for _, r in fd.iterrows():
        path = Path(r["absolute_path"])
        if not path.exists():
            continue
        vals = np.loadtxt(path)
        rows.append(
            {
                "SubjectID": r["subject_id"],
                "session_id": r["session_id"],
                "run_id": r["run_id"],
                "run_key": path.parent.name,
                "mean_fd_jenkinson": float(np.nanmean(vals)),
                "max_fd_jenkinson": float(np.nanmax(vals)),
                "n_frames": int(np.size(vals)),
            }
        )
    run_fd = pd.DataFrame(rows)
    selected = pd.read_csv(PILOT_CONNECTOME_DIR / "run_selection_used.csv")
    selected = selected[selected["selected_for_preflight"].astype(bool)].copy()
    selected = selected.rename(columns={"subject_id": "SubjectID", "run_id": "run_id"})
    merged = selected.merge(run_fd, on=["SubjectID", "session_id", "run_id"], how="left")
    return merged.groupby("SubjectID", dropna=False).agg(
        cohort=("SubjectID", lambda _: "pilot_30CN_30AD"),
        selected_qc_runs=("run_id", "count"),
        mean_fd_subject=("mean_fd_jenkinson", "mean"),
        max_fd_subject=("max_fd_jenkinson", "max"),
        selected_total_timepoints=("n_timepoints", "sum"),
        diagnosis=("diagnosis", "first"),
    ).reset_index()


def load_new_motion_summary() -> pd.DataFrame:
    manifest = pd.read_csv(NEW_TENSOR_DIR / "subject_manifest.csv").rename(columns={"subject_id": "SubjectID"})
    motion = pd.read_csv(NEW_HANDOFF_QC_DIR / "motion_qc_by_file.csv").rename(columns={"subject_id": "SubjectID"})
    manifest = manifest[manifest["SubjectID"].astype(str) != "subject_id"].copy()
    motion = motion[motion["SubjectID"].astype(str) != "subject_id"].copy()

    base_cols = ["SubjectID", "diagnosis", "selected_qc_runs", "selected_total_timepoints"]
    missing = [c for c in base_cols if c not in manifest.columns]
    if missing:
        raise ValueError(f"New subject_manifest is missing required columns: {missing}")
    base = manifest[base_cols].copy()
    base["selected_qc_runs"] = pd.to_numeric(base["selected_qc_runs"], errors="coerce")
    base["selected_total_timepoints"] = pd.to_numeric(base["selected_total_timepoints"], errors="coerce")

    for col in ("mean_fd_jenkinson", "max_fd_jenkinson"):
        if col in motion.columns:
            motion[col] = pd.to_numeric(motion[col], errors="coerce")
    agg = motion.groupby("SubjectID", dropna=False).agg(
        motion_run_count=("run", "count"),
        mean_fd_subject=("mean_fd_jenkinson", "mean"),
        max_fd_subject=("max_fd_jenkinson", "max"),
    ).reset_index()
    out = base.merge(agg, on="SubjectID", how="left")
    if out["selected_qc_runs"].isna().any() and "motion_run_count" in out.columns:
        out["selected_qc_runs"] = out["selected_qc_runs"].fillna(out["motion_run_count"])
    out["cohort"] = "new_60CN_60AD"
    return out[["SubjectID", "cohort", "selected_qc_runs", "mean_fd_subject", "max_fd_subject", "selected_total_timepoints", "diagnosis"]]


def motion_run_count_audit() -> pd.DataFrame:
    frames = []
    try:
        frames.append(load_pilot_motion_summary())
    except Exception as exc:
        frames.append(pd.DataFrame([{"cohort": "pilot_30CN_30AD", "motion_parse_error": str(exc)}]))
    try:
        frames.append(load_new_motion_summary())
    except Exception as exc:
        frames.append(pd.DataFrame([{"cohort": "new_60CN_60AD", "motion_parse_error": str(exc)}]))
    df = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    if "SubjectID" not in df.columns:
        return df
    for (cohort, diagnosis), sub in df.groupby(["cohort", "diagnosis"], dropna=False):
        rows.append(
            {
                "cohort": cohort,
                "diagnosis": diagnosis,
                "n": int(len(sub)),
                "selected_qc_runs_mean": float(pd.to_numeric(sub["selected_qc_runs"], errors="coerce").mean()),
                "selected_qc_runs_median": float(pd.to_numeric(sub["selected_qc_runs"], errors="coerce").median()),
                "mean_fd_mean": float(pd.to_numeric(sub["mean_fd_subject"], errors="coerce").mean()),
                "mean_fd_median": float(pd.to_numeric(sub["mean_fd_subject"], errors="coerce").median()),
                "max_fd_mean": float(pd.to_numeric(sub["max_fd_subject"], errors="coerce").mean()),
                "total_timepoints_mean": float(pd.to_numeric(sub["selected_total_timepoints"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def final_decision(
    existing: pd.DataFrame,
    pilot_next_metrics: Optional[pd.DataFrame],
    new_pilot_metrics: Optional[pd.DataFrame],
) -> tuple[str, str]:
    reasons = []
    old_best = existing[existing["source"].str.startswith("pilot")]["auc"].max()
    new_best_existing = existing[existing["source"].eq("new_original_scoring_locked_test")]["auc"].max()
    reasons.append(f"Original pilot best AUC={old_best:.4f}; original new locked-test best AUC={new_best_existing:.4f}.")

    if new_pilot_metrics is not None and not new_pilot_metrics.empty:
        ens = new_pilot_metrics[new_pilot_metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")]
        if not ens.empty:
            best_old_logic_new = float(ens["auc"].max())
            reasons.append(f"New tensors rescored with pilot Stage-B logic best AUC={best_old_logic_new:.4f}.")
        else:
            best_old_logic_new = np.nan
    else:
        best_old_logic_new = np.nan

    if pilot_next_metrics is not None and not pilot_next_metrics.empty:
        best_next_logic_pilot = float(pilot_next_metrics["auc"].max())
        reasons.append(f"Pilot tensors rescored with new saved-raw-pipeline logic best AUC={best_next_logic_pilot:.4f}.")
    else:
        best_next_logic_pilot = np.nan

    if np.isfinite(best_next_logic_pilot) and best_next_logic_pilot < old_best - 0.08:
        decision = "scoring_pipeline_mismatch_plus_possible_domain_instability"
        reasons.append("Pilot performance drops when scored with the new saved-raw-pipeline scorer, implicating scoring/readout mismatch.")
    elif np.isfinite(best_old_logic_new) and best_old_logic_new < 0.60:
        decision = "true_cohort_split_domain_instability_with_runwise_tensor_mismatch_risk"
        reasons.append("New data remain weak even under old pilot Stage-B scoring logic; cohort/split/domain instability is likely.")
    else:
        decision = "pipeline_mismatch_unresolved"
        reasons.append("Cross-scoring did not give a single clean explanation.")

    reasons.append(
        "Runwise tensor construction differs between pilot and new builders: pilot normalizes per-run before averaging, new averages raw connectomes before normalization."
    )
    reasons.append(
        "The new scorer does not match the pilot scorer's primary ADNI readout: different model path/readout serialization/threshold aggregation."
    )
    return decision, " ".join(reasons)


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (PILOT_CONNECTOME_DIR, "pilot connectome dir"),
        (PILOT_SCORING_DIR, "pilot scoring dir"),
        (NEW_TENSOR_DIR, "new tensor dir"),
        (NEW_SCORING_DIR, "new scoring dir"),
        (NEW_POSTMORTEM_DIR, "new postmortem dir"),
        (ADNI_TENSOR_PATH, "ADNI tensor"),
        (ADNI_METADATA_PATH, "ADNI metadata"),
    ]:
        require(path, label)

    device = resolve_device(args.device)
    located = []
    for label, path in [
        ("pilot_connectomes", PILOT_CONNECTOME_DIR),
        ("pilot_140TR", PILOT_140TR_DIR),
        ("pilot_external_scoring", PILOT_SCORING_DIR),
        ("pilot_postmortem", PILOT_POSTMORTEM_DIR),
        ("new_tensors", NEW_TENSOR_DIR),
        ("new_tensor_qc", NEW_TENSOR_QC_DIR),
        ("new_external_scoring", NEW_SCORING_DIR),
        ("new_postmortem", NEW_POSTMORTEM_DIR),
    ]:
        located.append({"artifact_group": label, "path": str(path), "exists": path.exists()})
    write_table(outdir, "located_artifacts", pd.DataFrame(located))

    pilot_bundles = [
        load_tensor_bundle("pilot_30CN_30AD", label, tdir, fname)
        for label, (tdir, fname) in PILOT_TENSOR_SPECS.items()
    ]
    new_bundles = [
        load_tensor_bundle("new_60CN_60AD", label, tdir, fname)
        for label, (tdir, fname) in NEW_TENSOR_SPECS.items()
    ]
    pilot_subjects = pilot_bundles[0].subjects.copy()
    new_subjects = new_bundles[0].subjects.copy()
    write_table(outdir, "cohort_balance_overlap", cohort_balance_and_overlap(pilot_subjects, new_subjects))
    write_table(outdir, "tensor_construction_parity", tensor_construction_parity())
    write_table(outdir, "scoring_script_parity", scoring_script_parity())
    write_table(outdir, "script_hashes", script_parity_table())

    existing = existing_metric_comparison()
    write_table(outdir, "existing_metric_comparison", existing)

    all_bundles = pilot_bundles + new_bundles
    tensor_dist = tensor_distribution_table(all_bundles)
    write_table(outdir, "tensor_distribution_adni_pilot_new", tensor_dist)
    write_table(outdir, "motion_run_count_audit", motion_run_count_audit())

    pilot_next_pred: Optional[pd.DataFrame] = None
    pilot_next_metrics: Optional[pd.DataFrame] = None
    new_pilot_pred: Optional[pd.DataFrame] = None
    new_pilot_metrics: Optional[pd.DataFrame] = None

    if not args.skip_cross_score:
        pilot_next_pred, pilot_next_metrics = score_pilot_with_next_scorer(pilot_bundles, device, args.batch_size)
        pilot_next_pred.to_csv(outdir / "pilot_rescored_with_next_scorer_predictions.csv", index=False)
        write_table(outdir, "pilot_rescored_with_next_scorer_metrics", pilot_next_metrics)

        new_pilot_pred, new_pilot_metrics = score_new_with_pilot_scorer(new_bundles, device, args.batch_size)
        new_pilot_pred.to_csv(outdir / "new_rescored_with_pilot_scorer_predictions.csv", index=False)
        write_table(outdir, "new_rescored_with_pilot_scorer_metrics", new_pilot_metrics)

        score_dists = [
            score_distribution_from_predictions(
                pd.read_csv(PILOT_SCORING_DIR / "predictions.csv"),
                "adni_model",
                "pilot_original_scoring",
                "pilot_30CN_30AD",
            ),
            score_distribution_from_predictions(
                pd.read_csv(NEW_SCORING_DIR / "predictions_locked_test.csv"),
                "model_label",
                "new_original_scoring_locked_test",
                "new_60CN_60AD_locked_test",
            ),
            score_distribution_from_predictions(
                pilot_next_pred,
                "model_label",
                "pilot_rescored_with_next_scorer",
                "pilot_30CN_30AD",
            ),
            score_distribution_from_predictions(
                new_pilot_pred,
                "adni_model",
                "new_rescored_with_pilot_scorer",
                "new_60CN_60AD",
            ),
        ]
    else:
        score_dists = [
            score_distribution_from_predictions(
                pd.read_csv(PILOT_SCORING_DIR / "predictions.csv"),
                "adni_model",
                "pilot_original_scoring",
                "pilot_30CN_30AD",
            ),
            score_distribution_from_predictions(
                pd.read_csv(NEW_SCORING_DIR / "predictions_locked_test.csv"),
                "model_label",
                "new_original_scoring_locked_test",
                "new_60CN_60AD_locked_test",
            ),
        ]
    write_table(outdir, "score_distribution_comparison", pd.concat(score_dists, ignore_index=True, sort=False))

    decision, rationale = final_decision(existing, pilot_next_metrics, new_pilot_metrics)
    final_text = [
        "# Final Recommendation",
        "",
        f"Decision: `{decision}`",
        "",
        rationale,
        "",
        "## Guardrails",
        "",
        "- No model training was performed.",
        "- No tensors or model outputs were modified.",
        "- Cross-scoring wrote predictions only inside this audit output directory.",
        "- No threshold fitting on locked test and no OASIS-based promotion were performed.",
        "",
        "## Operational Interpretation",
        "",
        "The weak new OASIS results should not be interpreted as a clean biological failure until the scoring path is harmonized with the pilot path. "
        "The most actionable finding is that the new external scorer is not equivalent to the pilot scorer and the runwise tensor averaging order changed. "
        "A locked parity scorer should use the final horizon4480 Stage-B classifier-only readout for both pilot and new OASIS tensors, with one explicitly chosen tensor build rule.",
        "",
    ]
    (outdir / "final_recommendation.md").write_text("\n".join(final_text), encoding="utf-8")
    readme = [
        "# OASIS Pilot vs New Tensor/Scoring Parity Audit",
        "",
        "Read-only audit comparing the Tanda 2026-05-25 pilot with the new 60CN/60AD OASIS batch.",
        "",
        "Primary outputs include artifact location, cohort overlap, tensor construction parity, scoring script parity, cross-scoring metrics, tensor distributions, motion/run-count summaries, and final recommendation.",
        "",
    ]
    (outdir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    write_json(
        outdir / "command_log.json",
        {
            "script": str(Path(__file__).resolve()),
            "timestamp_utc": now_utc(),
            "output_dir": str(outdir),
            "device": str(device),
            "skip_cross_score": bool(args.skip_cross_score),
            "decision": decision,
            "no_training": True,
            "no_tensor_modification": True,
            "no_model_selection": True,
            "no_locked_test_threshold_fitting": True,
        },
    )
    print(f"Done. Outputs written to: {outdir}")
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
