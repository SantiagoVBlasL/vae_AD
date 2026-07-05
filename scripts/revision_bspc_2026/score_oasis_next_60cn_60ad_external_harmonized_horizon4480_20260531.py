#!/usr/bin/env python
"""Score OASIS 60CN/60AD with the exact final horizon4480 Stage B readout.

This harmonizes the new OASIS scoring path with the pilot scorer:

* final ADNI run:
  adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5
* Stage B classifier-only logreg_l2 readout reconstructed from ADNI train/dev
  latent caches and saved fold C values
* ADNI fixed threshold rule:
  per-fold ADNI thresholds followed by fold-decision majority vote, matching
  score_oasis_tanda_20260525_external_adni.py

The OASIS calibration threshold is fit only on the predeclared calibration
subset and then transferred to locked_test. No VAE/classifier training on
OASIS, no locked-test threshold fitting, and no model selection are performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    binary_metrics as binary_metrics_with_pred,
    load_best_c,
    load_thresholds,
    make_ensemble_predictions,
    score_one_model_on_one_tensor,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"
DEFAULT_TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_20260530"
DEFAULT_SPLIT_CSV = RESULTS_DIR / "oasis_60cn_60ad_calibration_test_protocol" / "split_calibration_test.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_harmonized_horizon4480_20260531"

TENSOR_CANDIDATES = {
    "concatenated_timeseries": "tensor_concatenated_timeseries.npz",
    "runwise_140TR_connectome_average": "tensor_runwise_140TR_connectome_average.npz",
    "runwise164_connectome_average": "tensor_runwise164_connectome_average.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-adni-run-dir", type=Path, default=DEFAULT_PRIMARY_ADNI_RUN)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--confirm-score", action="store_true")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=sorted(TENSOR_CANDIDATES),
        default=sorted(TENSOR_CANDIDATES),
    )
    return parser.parse_args()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def normalize_sex(value: Any) -> str:
    s = str(value).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def load_split(split_csv: Path) -> pd.DataFrame:
    require(split_csv, "OASIS calibration/test split")
    split = pd.read_csv(split_csv)
    required = {"protocol_subset", "subject_id", "diagnosis", "age_at_MR", "sex"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Split CSV missing required columns: {missing}")
    allowed = {"calibration", "locked_test"}
    observed = set(split["protocol_subset"].astype(str))
    if not observed.issubset(allowed):
        raise ValueError(f"Unexpected protocol_subset values: {sorted(observed - allowed)}")
    counts = split.groupby(["protocol_subset", "diagnosis"], dropna=False).size().unstack(fill_value=0)
    if int(counts.loc["calibration", "CN"]) != 30 or int(counts.loc["calibration", "AD_DEMENTIA"]) != 30:
        raise ValueError(f"Calibration split is not 30 CN / 30 AD_DEMENTIA:\n{counts}")
    if int(counts.loc["locked_test", "CN"]) != 30 or int(counts.loc["locked_test", "AD_DEMENTIA"]) != 30:
        raise ValueError(f"Locked-test split is not 30 CN / 30 AD_DEMENTIA:\n{counts}")
    overlap = set(split.loc[split["protocol_subset"].eq("calibration"), "subject_id"].astype(str)) & set(
        split.loc[split["protocol_subset"].eq("locked_test"), "subject_id"].astype(str)
    )
    if overlap:
        raise ValueError(f"Calibration/test subject overlap detected: {sorted(overlap)[:10]}")
    return split


def load_oasis_tensor(tensor_dir: Path, candidate: str) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    npz_path = tensor_dir / TENSOR_CANDIDATES[candidate]
    require(npz_path, f"OASIS tensor {candidate}")
    data = np.load(npz_path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    channel_names = data["channel_names"].astype(str).tolist()
    subjects = pd.DataFrame(
        {
            "SubjectID": data["subject_ids"].astype(str),
            "session_id": data["session_ids"].astype(str),
            "experiment_id": data["experiment_ids"].astype(str),
            "diagnosis_tensor": data["diagnosis"].astype(str),
        }
    )
    if tensor.shape[0] != len(subjects):
        raise ValueError(f"Tensor row count {tensor.shape[0]} != subject row count {len(subjects)}")
    if tensor.shape[1] != 3 or tensor.shape[-2:] != (131, 131):
        raise ValueError(f"Unexpected tensor shape for ADNI [1,0,2] scoring: {tensor.shape}")
    manifest_path = tensor_dir / "subject_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path).rename(columns={"subject_id": "SubjectID"})
        keep = [
            c
            for c in [
                "SubjectID",
                "session_id",
                "diagnosis",
                "age_at_MR",
                "sex",
                "Manufacturer",
                "ScannerModel",
                "selected_qc_runs",
                "selected_run_ids",
                "selected_total_timepoints",
            ]
            if c in manifest.columns
        ]
        subjects = subjects.merge(
            manifest[keep].drop_duplicates(["SubjectID", "session_id"]),
            on=["SubjectID", "session_id"],
            how="left",
        )
    return tensor, subjects, channel_names


def align_to_split(subjects: pd.DataFrame, split: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    split_ids = split["subject_id"].astype(str).tolist()
    split_id_set = set(split_ids)
    idx = np.where(subjects["SubjectID"].astype(str).isin(split_id_set).to_numpy())[0]
    found = set(subjects.iloc[idx]["SubjectID"].astype(str))
    missing = sorted(split_id_set - found)
    if missing:
        raise RuntimeError(f"{len(missing)} split subjects missing from tensor: {missing[:20]}")
    extra = sorted(set(subjects["SubjectID"].astype(str)) - split_id_set)
    if extra:
        raise RuntimeError(f"{len(extra)} tensor subjects absent from fixed split: {extra[:20]}")

    sub = subjects.iloc[idx].reset_index(drop=True)
    rename_for_audit = {
        c: f"tensor_{c}"
        for c in ["session_id", "experiment_id", "diagnosis_tensor"]
        if c in sub.columns
    }
    sub = sub.rename(columns=rename_for_audit)
    sub = sub.drop(
        columns=[
            c
            for c in ["age_at_MR", "sex", "Age", "Sex", "diagnosis", "Manufacturer", "ScannerModel"]
            if c in sub.columns
        ],
        errors="ignore",
    )
    split_meta = split[
        [
            c
            for c in [
                "subject_id",
                "protocol_subset",
                "session_id",
                "experiment_id",
                "diagnosis",
                "age_at_MR",
                "sex",
                "Manufacturer",
                "ScannerModel",
            ]
            if c in split.columns
        ]
    ].rename(columns={"subject_id": "SubjectID"})
    sub = sub.merge(split_meta.drop_duplicates("SubjectID"), on="SubjectID", how="left", validate="one_to_one")
    sub["Age"] = pd.to_numeric(sub["age_at_MR"], errors="coerce")
    sub["Sex"] = sub["sex"].map(normalize_sex)
    bad_age = sub.loc[sub["Age"].isna(), "SubjectID"].astype(str).tolist()
    bad_sex = sub.loc[sub["Sex"].eq("UNKNOWN") | sub["Sex"].isna(), "SubjectID"].astype(str).tolist()
    if bad_age or bad_sex:
        raise RuntimeError(f"Bad OASIS Age/Sex after split merge: bad_age={bad_age[:20]}, bad_sex={bad_sex[:20]}")
    sub["ResearchGroup_Mapped"] = sub["diagnosis"].map({"CN": "CN", "AD_DEMENTIA": "AD", "AD": "AD"})
    if sub["ResearchGroup_Mapped"].isna().any():
        raise RuntimeError("Unexpected diagnosis values after split merge.")
    sub["y_true"] = sub["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    # Score rows in tensor order, but keep protocol_subset from the locked split.
    return idx, sub


def artifact_validation(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = [
        ("tensor_dir", "oasis_next_tensor_dir", args.tensor_dir),
        ("split", "fixed_calibration_test_split", args.split_csv),
        ("adni_run", "final_horizon4480_run_dir", args.primary_adni_run_dir),
        ("adni_readout", "final_horizon4480_classifier_only_readout", args.primary_adni_run_dir / "classifier_only_readout"),
    ]
    for artifact_type, label, path in paths:
        rows.append({"artifact_type": artifact_type, "label": label, "path": str(path), "exists": path.exists()})
    for candidate in args.candidates:
        path = args.tensor_dir / TENSOR_CANDIDATES[candidate]
        rows.append({"artifact_type": "oasis_tensor", "label": candidate, "path": str(path), "exists": path.exists()})
    readout = args.primary_adni_run_dir / "classifier_only_readout"
    for rel in [
        "classifier_sweep_model_status.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "latent_cache",
    ]:
        path = readout / rel
        rows.append({"artifact_type": "adni_readout_component", "label": rel, "path": str(path), "exists": path.exists()})
    for fold in range(1, 6):
        for path in [
            args.primary_adni_run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt",
            args.primary_adni_run_dir / f"fold_{fold}" / "vae_norm_params.joblib",
            readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
        ]:
            rows.append({"artifact_type": "adni_fold_component", "label": f"fold_{fold}", "path": str(path), "exists": path.exists()})
        try:
            rows.append(
                {
                    "artifact_type": "adni_fold_readout_value",
                    "label": f"fold_{fold}_logreg_l2_C",
                    "path": str(readout / "classifier_sweep_model_status.csv"),
                    "exists": True,
                    "value": load_best_c(readout, fold),
                }
            )
            rows.append(
                {
                    "artifact_type": "adni_fold_readout_value",
                    "label": f"fold_{fold}_threshold",
                    "path": str(readout / "classifier_sweep_thresholds_by_fold.csv"),
                    "exists": True,
                    "value": load_thresholds(readout, fold),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "artifact_type": "adni_fold_readout_value",
                    "label": f"fold_{fold}_readout_parse_error",
                    "path": str(readout),
                    "exists": False,
                    "value": str(exc),
                }
            )
    return pd.DataFrame(rows)


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


def metrics_from_pred(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> dict[str, Any]:
    return binary_metrics_with_pred(y_true, y_score, y_pred)


def metrics_from_threshold(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> dict[str, Any]:
    score = np.asarray(y_score, dtype=float)
    return metrics_from_pred(y_true, score, (score >= threshold).astype(int))


def summarize_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for (build_candidate, adni_model), sub_model in ens.groupby(["build_candidate", "adni_model"], dropna=False):
        cal = sub_model[sub_model["protocol_subset"].eq("calibration")].copy()
        test = sub_model[sub_model["protocol_subset"].eq("locked_test")].copy()
        if cal.empty or test.empty:
            raise RuntimeError(f"Missing calibration or locked_test rows for {build_candidate}/{adni_model}")
        oasis_thr = threshold_sens_ge_070_max_spec(cal["y_true"], cal["y_score"])
        adni_mean_thr = float(cal["adni_threshold"].mean())
        threshold_rows.append(
            {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "adni_fixed_prediction_rule": "fold_specific_thresholds_then_majority_vote",
                "adni_threshold_mean_context": adni_mean_thr,
                "oasis_calibration_threshold": oasis_thr,
                "oasis_calibration_threshold_rule": "calibration_subset_sens_ge_0p70_max_spec_on_ensemble_mean_score",
                "locked_test_used_for_threshold": False,
            }
        )
        for subset_name, subset in [("calibration", cal), ("locked_test", test)]:
            fixed = {
                "split_subset": subset_name,
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "prediction_level": "ensemble_mean_score_majority_vote",
                "threshold_strategy": "adni_fixed_fold_majority_vote",
                "prediction_rule": "fold_specific_adni_thresholds_then_majority_vote",
                "threshold": adni_mean_thr,
                "threshold_fit_subset": "ADNI_inner_OOF",
                "locked_test_used_for_threshold": False,
            }
            fixed.update(metrics_from_pred(subset["y_true"], subset["y_score"], subset["y_pred"]))
            metric_rows.append(fixed)

            cal_metric = {
                "split_subset": subset_name,
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "prediction_level": "ensemble_mean_score",
                "threshold_strategy": "oasis_calibration_sens_ge_0p70_max_spec",
                "prediction_rule": "ensemble_mean_score_thresholded_by_calibration_subset",
                "threshold": oasis_thr,
                "threshold_fit_subset": "OASIS_calibration",
                "locked_test_used_for_threshold": False,
            }
            cal_metric.update(metrics_from_threshold(subset["y_true"], subset["y_score"], oasis_thr))
            metric_rows.append(cal_metric)
    return pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows)


def score_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    return (
        ens.groupby(["split_subset", "build_candidate", "adni_model", "diagnosis"], dropna=False)["y_score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "score_mean", "std": "score_std", "min": "score_min", "median": "score_median", "max": "score_max"})
    )


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subset, sub in split.groupby("protocol_subset", dropna=False):
        rows.append(
            {
                "protocol_subset": subset,
                "n": len(sub),
                "n_subjects": sub["subject_id"].nunique(),
                "n_cn": int(sub["diagnosis"].eq("CN").sum()),
                "n_ad_dementia": int(sub["diagnosis"].eq("AD_DEMENTIA").sum()),
                "age_mean": float(pd.to_numeric(sub["age_at_MR"], errors="coerce").mean()),
                "age_sd": float(pd.to_numeric(sub["age_at_MR"], errors="coerce").std(ddof=1)),
                "sex_counts": ";".join(f"{k}:{v}" for k, v in sub["sex"].value_counts(dropna=False).sort_index().items()),
            }
        )
    return pd.DataFrame(rows)


def make_readme(
    output_dir: Path,
    scored: bool,
    metrics: pd.DataFrame | None = None,
    thresholds: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# OASIS 60CN/60AD Harmonized Horizon4480 External Scoring",
        "",
        f"Status: `{'scored' if scored else 'dry_run_only'}`",
        "",
        "This run harmonizes new OASIS scoring with the pilot scorer by using the final",
        f"`{DEFAULT_PRIMARY_ADNI_RUN.name}` Stage B classifier-only readout.",
        "",
        "## Guardrails",
        "",
        "- No VAE or classifier was trained on OASIS.",
        "- No tensors were modified.",
        "- ADNI fixed-threshold predictions use the pilot scorer rule: fold-specific ADNI thresholds, then majority vote.",
        "- OASIS calibration thresholds are fit only on the predeclared calibration subset.",
        "- Locked-test labels are not used for threshold selection.",
    ]
    if scored and metrics is not None:
        locked = metrics[
            metrics["split_subset"].eq("locked_test")
            & metrics["prediction_level"].isin(["ensemble_mean_score_majority_vote", "ensemble_mean_score"])
        ].copy()
        lines += ["", "## Locked-Test Metrics", "", locked.to_markdown(index=False)]
    if scored and thresholds is not None:
        lines += ["", "## Thresholds", "", thresholds.to_markdown(index=False)]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run_report(output_dir: Path, artifact_df: pd.DataFrame, args: argparse.Namespace, split: pd.DataFrame | None) -> None:
    missing = artifact_df[~artifact_df["exists"].astype(bool)].copy()
    lines = [
        "# Dry Run Report",
        "",
        "Mode: `dry_run_only`",
        "",
        f"Output directory: `{output_dir}`",
        f"Tensor directory: `{args.tensor_dir}`",
        f"Split CSV: `{args.split_csv}`",
        f"ADNI run: `{args.primary_adni_run_dir}`",
        "",
        f"Checked artifacts: {len(artifact_df)}",
        f"Missing artifacts: {len(missing)}",
    ]
    if split is not None:
        lines += ["", "## Split Summary", "", split_summary(split).to_markdown(index=False)]
    if not missing.empty:
        lines += ["", "## Missing Artifacts", "", missing.to_markdown(index=False)]
    output_dir.joinpath("dry_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_df = artifact_validation(args)
    write_csv_md(artifact_df, output_dir / "artifact_validation.csv", output_dir / "artifact_validation.md", "Artifact Validation")

    split: pd.DataFrame | None = None
    try:
        split = load_split(args.split_csv)
        write_csv_md(split_summary(split), output_dir / "split_summary.csv", output_dir / "split_summary.md", "Split Summary")
    except Exception as exc:
        if args.confirm_score:
            raise
        output_dir.joinpath("split_summary.md").write_text(f"# Split Summary\n\nDry-run split validation failed: `{exc}`\n", encoding="utf-8")

    if not args.confirm_score:
        dry_run_report(output_dir, artifact_df, args, split)
        make_readme(output_dir, scored=False)
        write_json(
            output_dir / "command_log.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__).resolve()),
                "mode": "dry_run_only",
                "confirm_score": False,
                "oasis_scored": False,
                "oasis_training": False,
                "locked_test_threshold_fitting": False,
                "artifact_missing_count": int((~artifact_df["exists"].astype(bool)).sum()),
            },
        )
        print(json.dumps({"output_dir": str(output_dir), "mode": "dry_run_only", "oasis_scored": False}, indent=2))
        return 0

    missing = artifact_df[~artifact_df["exists"].astype(bool)].copy()
    if not missing.empty:
        raise FileNotFoundError("Cannot score; missing artifacts:\n" + missing.to_string(index=False))
    if split is None:
        split = load_split(args.split_csv)

    spec = AdniModelSpec("primary_v5_1b_ch1_0_2_horizon4480", args.primary_adni_run_dir, True)
    device = resolve_device(args.device)
    split_ids = set(split["subject_id"].astype(str))

    prediction_frames: list[pd.DataFrame] = []
    metadata_audit_frames: list[pd.DataFrame] = []
    for candidate in args.candidates:
        tensor, subjects, channel_names = load_oasis_tensor(args.tensor_dir, candidate)
        idx, score_meta = align_to_split(subjects, split)
        if set(score_meta["SubjectID"].astype(str)) != split_ids:
            raise RuntimeError(f"Split alignment mismatch for {candidate}")
        x = tensor[idx]
        pred = score_one_model_on_one_tensor(
            spec=spec,
            tensor=x,
            oasis_subjects=score_meta,
            oasis_channel_names=channel_names,
            batch_size=int(args.batch_size),
            device=device,
        )
        pred["build_candidate"] = candidate
        prediction_frames.append(pred)
        audit_cols = [
            "SubjectID",
            "protocol_subset",
            "diagnosis",
            "tensor_diagnosis_tensor",
            "session_id",
            "tensor_session_id",
            "experiment_id",
            "tensor_experiment_id",
            "age_at_MR",
            "Age",
            "sex",
            "Sex",
            "Manufacturer",
            "ScannerModel",
            "selected_qc_runs",
            "selected_total_timepoints",
        ]
        audit = score_meta[[c for c in audit_cols if c in score_meta.columns]].copy()
        audit.insert(0, "build_candidate", candidate)
        metadata_audit_frames.append(audit)

    fold_predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble_predictions = make_ensemble_predictions(fold_predictions)
    predictions = pd.concat([fold_predictions, ensemble_predictions], ignore_index=True, sort=False)
    if "protocol_subset" in predictions.columns and "split_subset" not in predictions.columns:
        predictions["split_subset"] = predictions["protocol_subset"]

    metrics, thresholds = summarize_metrics(predictions)
    dist = score_distribution(predictions)
    metadata_audit = pd.concat(metadata_audit_frames, ignore_index=True).drop_duplicates(["build_candidate", "SubjectID"])

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_predictions.to_csv(output_dir / "predictions_fold_models.csv", index=False)
    ensemble_predictions.to_csv(output_dir / "predictions_ensemble_adni_fixed.csv", index=False)
    write_csv_md(metadata_audit, output_dir / "metadata_alignment_audit.csv", output_dir / "metadata_alignment_audit.md", "Metadata Alignment Audit")
    write_csv_md(metrics, output_dir / "primary_metrics.csv", output_dir / "primary_metrics.md", "Primary Metrics")
    write_csv_md(metrics[metrics["split_subset"].eq("calibration")], output_dir / "metrics_calibration.csv", output_dir / "metrics_calibration.md", "Calibration Metrics")
    write_csv_md(metrics[metrics["split_subset"].eq("locked_test")], output_dir / "metrics_locked_test.csv", output_dir / "metrics_locked_test.md", "Locked Test Metrics")
    write_csv_md(thresholds, output_dir / "thresholds.csv", output_dir / "thresholds.md", "Thresholds")
    write_csv_md(dist, output_dir / "score_distribution.csv", output_dir / "score_distribution.md", "Score Distribution")
    make_readme(output_dir, scored=True, metrics=metrics, thresholds=thresholds)

    write_json(
        output_dir / "command_log.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "mode": "harmonized_external_scoring",
            "confirm_score": True,
            "oasis_scored": True,
            "oasis_training": False,
            "adni_stageb_readout_reconstructed_in_memory": True,
            "adni_stageb_readout_path": str(args.primary_adni_run_dir / "classifier_only_readout"),
            "adni_threshold_rule": "fold_specific_thresholds_then_majority_vote",
            "oasis_calibration_threshold_rule": "fit_on_calibration_subset_only_sens_ge_0p70_max_spec",
            "locked_test_threshold_fitting": False,
            "oasis_model_selection": False,
            "tensor_dir": str(args.tensor_dir),
            "split_csv": str(args.split_csv),
            "output_dir": str(output_dir),
            "device": str(device),
            "candidates": list(args.candidates),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "mode": "harmonized_external_scoring", "oasis_scored": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
