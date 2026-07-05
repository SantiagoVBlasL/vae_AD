#!/usr/bin/env python
"""External OASIS scoring with locked ADNI fold models.

Default mode is dry-run/read-only validation. Real scoring requires
--confirm-score. The Stage B classifier-only readout did not persist fitted
estimators, so confirmed scoring reconstructs the ADNI-only logreg_l2 readout
in memory from saved ADNI train/dev latent caches, saved best C values, and
saved ADNI inner-OOF thresholds. No OASIS labels are used for fitting,
thresholding, or model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (  # noqa: E402
    apply_normalization_params,
    encode_mu,
    load_config,
    make_model,
    make_preprocessor,
    score_1d,
)


DEFAULT_CONNECTOME_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectomes"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring"
)
DEFAULT_PRIMARY_ADNI_RUN = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
DEFAULT_SECONDARY_CH1_RUN = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1b_ch1_only_offdiag_channelmean_horizon4480_cycles56_full_5x5"
)
PRIMARY_THRESHOLD_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"
TENSOR_CANDIDATES = {
    "concatenated_timeseries": "tensor_concatenated_timeseries.npz",
    "runwise_connectome_average": "tensor_runwise_connectome_average.npz",
}


@dataclass(frozen=True)
class AdniModelSpec:
    model_label: str
    run_dir: Path
    include: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-adni-run-dir", type=Path, default=DEFAULT_PRIMARY_ADNI_RUN)
    parser.add_argument("--secondary-ch1-run-dir", type=Path, default=DEFAULT_SECONDARY_CH1_RUN)
    parser.add_argument("--include-secondary-ch1", action="store_true")
    parser.add_argument("--confirm-score", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def model_specs(args: argparse.Namespace) -> list[AdniModelSpec]:
    return [
        AdniModelSpec("primary_v5_1b_ch1_0_2_horizon4480", args.primary_adni_run_dir, True),
        AdniModelSpec("secondary_ch1_offdiag_channelmean", args.secondary_ch1_run_dir, bool(args.include_secondary_ch1)),
    ]


def validate_artifacts(args: argparse.Namespace, specs: Sequence[AdniModelSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    require(args.connectome_dir, "OASIS connectome dir")
    require(args.connectome_dir / "subject_manifest.csv", "OASIS subject_manifest.csv")
    for candidate, filename in TENSOR_CANDIDATES.items():
        p = args.connectome_dir / filename
        rows.append({"artifact_type": "oasis_tensor", "label": candidate, "path": str(p), "exists": p.exists()})
    for spec in specs:
        if not spec.include:
            continue
        readout = spec.run_dir / "classifier_only_readout"
        expected = [
            spec.run_dir / "run_config.json",
            readout / "classifier_sweep_model_status.csv",
            readout / "classifier_sweep_thresholds_by_fold.csv",
            readout / "latent_cache",
        ]
        for p in expected:
            rows.append({"artifact_type": "adni_model", "label": spec.model_label, "path": str(p), "exists": p.exists()})
        for fold in range(1, 6):
            fold_files = [
                spec.run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt",
                spec.run_dir / f"fold_{fold}" / "vae_norm_params.joblib",
                readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
            ]
            for p in fold_files:
                rows.append({"artifact_type": "adni_fold", "label": f"{spec.model_label}_fold{fold}", "path": str(p), "exists": p.exists()})
    return pd.DataFrame(rows)


def normalize_oasis_sex(value: Any) -> str:
    s = str(value).strip().upper()
    if s in {"1", "M", "MALE"}:
        return "M"
    if s in {"2", "F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def load_oasis_tensor(connectome_dir: Path, candidate: str) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    path = connectome_dir / TENSOR_CANDIDATES[candidate]
    require(path, f"OASIS tensor {candidate}")
    data = np.load(path, allow_pickle=True)
    tensor = np.asarray(data["global_tensor_data"], dtype=np.float32)
    channel_names = data["channel_names"].astype(str).tolist()
    subjects = pd.DataFrame(
        {
            "SubjectID": data["subject_ids"].astype(str),
            "session_id": data["session_ids"].astype(str),
            "experiment_id": data["experiment_ids"].astype(str),
            "diagnosis": data["diagnosis"].astype(str),
        }
    )
    manifest = pd.read_csv(connectome_dir / "subject_manifest.csv")
    manifest = manifest.rename(columns={"subject_id": "SubjectID"})
    cols = [
        c
        for c in [
            "SubjectID",
            "session_id",
            "Manufacturer",
            "ScannerModel",
            "age_at_MR",
            "sex",
            "selected_qc_runs",
            "selected_run_ids",
            "selected_total_timepoints",
        ]
        if c in manifest.columns
    ]
    subjects = subjects.merge(manifest[cols].drop_duplicates(["SubjectID", "session_id"]), on=["SubjectID", "session_id"], how="left")
    subjects["ResearchGroup_Mapped"] = subjects["diagnosis"].map({"CN": "CN", "AD_DEMENTIA": "AD"})
    subjects["y_true"] = subjects["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    subjects["Age"] = pd.to_numeric(subjects.get("age_at_MR"), errors="coerce")
    subjects["Sex"] = subjects.get("sex", "UNKNOWN").map(normalize_oasis_sex)
    return tensor, subjects, channel_names


def select_channels(tensor: np.ndarray, oasis_channel_names: Sequence[str], selected_channel_names: Sequence[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(oasis_channel_names)}
    missing = [name for name in selected_channel_names if name not in index]
    if missing:
        raise ValueError(f"OASIS tensor is missing selected ADNI channels: {missing}; available={list(oasis_channel_names)}")
    return tensor[:, [index[name] for name in selected_channel_names], :, :].astype(np.float32)


def load_thresholds(readout_dir: Path, fold: int) -> float:
    thr = pd.read_csv(readout_dir / "classifier_sweep_thresholds_by_fold.csv")
    row = thr[
        thr["fold"].astype(int).eq(fold)
        & thr["model_name"].astype(str).eq("logreg_l2")
        & thr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD_STRATEGY)
    ]
    if row.empty:
        raise ValueError(f"Missing ADNI threshold for fold={fold}, strategy={PRIMARY_THRESHOLD_STRATEGY}")
    return float(row.iloc[0]["threshold"])


def load_best_c(readout_dir: Path, fold: int) -> float:
    status = pd.read_csv(readout_dir / "classifier_sweep_model_status.csv")
    row = status[status["fold"].astype(int).eq(fold) & status["model_name"].astype(str).eq("logreg_l2")]
    if row.empty:
        raise ValueError(f"Missing logreg_l2 model status for fold={fold}")
    params = json.loads(row.iloc[0]["best_params"])
    return float(params["model__C"])


def fit_adni_stageb_logreg(train_df: pd.DataFrame, c_value: float, seed: int) -> Pipeline:
    mu_cols = [c for c in train_df.columns if c.startswith("mu_")]
    feature_cols = mu_cols + ["Age", "Sex"]
    pre = make_preprocessor(mu_cols, include_age_sex=True)
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(train_df[feature_cols], train_df["y"].astype(int).to_numpy())
    return pipe


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
        "accuracy": float((tp + tn) / len(y)) if len(y) else np.nan,
    }


def score_one_model_on_one_tensor(
    spec: AdniModelSpec,
    tensor: np.ndarray,
    oasis_subjects: pd.DataFrame,
    oasis_channel_names: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    cfg = load_config(spec.run_dir)
    readout_dir = spec.run_dir / "classifier_only_readout"
    selected_names = cfg["selected_channel_names"]
    x_oasis = select_channels(tensor, oasis_channel_names, selected_names)
    rows: list[pd.DataFrame] = []
    for fold in range(1, int(cfg.get("outer_folds", 5)) + 1):
        fold_dir = spec.run_dir / f"fold_{fold}"
        checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
        norm_path = fold_dir / "vae_norm_params.joblib"
        require(checkpoint, f"{spec.model_label} fold {fold} VAE checkpoint")
        require(norm_path, f"{spec.model_label} fold {fold} normalization params")
        norm_params = joblib.load(norm_path)
        x_norm = apply_normalization_params(x_oasis, norm_params)
        model = make_model(
            cfg,
            image_size=x_oasis.shape[-1],
            n_channels=x_oasis.shape[1],
            device=device,
        )
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        mu = encode_mu(model, x_norm, batch_size=batch_size, device=device)

        train_df = pd.read_csv(readout_dir / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv")
        c_value = load_best_c(readout_dir, fold)
        clf = fit_adni_stageb_logreg(train_df, c_value, int(cfg.get("seed", 42)) + fold)
        mu_df = pd.DataFrame(mu, columns=[f"mu_{i}" for i in range(mu.shape[1])])
        feature_df = pd.concat(
            [mu_df, oasis_subjects[["Age", "Sex"]].reset_index(drop=True)],
            axis=1,
        )
        y_score = score_1d(clf, feature_df)
        threshold = load_thresholds(readout_dir, fold)
        pred = oasis_subjects.copy()
        pred["adni_model"] = spec.model_label
        pred["fold"] = fold
        pred["selected_channel_names"] = "|".join(selected_names)
        pred["readout"] = "stageb_logreg_l2_reconstructed_from_adni_trainDev"
        pred["threshold_strategy"] = PRIMARY_THRESHOLD_STRATEGY
        pred["adni_threshold"] = threshold
        pred["adni_logreg_l2_C"] = c_value
        pred["y_score"] = y_score
        pred["y_pred"] = (y_score >= threshold).astype(int)
        pred["prediction_level"] = "fold_model"
        rows.append(pred)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True)


def make_ensemble_predictions(fold_predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["build_candidate", "adni_model", "SubjectID", "session_id", "experiment_id"]
    rows = []
    for key, sub in fold_predictions.groupby(group_cols, dropna=False):
        base = sub.iloc[0].copy()
        base["fold"] = "ensemble_mean"
        base["prediction_level"] = "ensemble_mean_score_majority_vote"
        base["y_score"] = float(sub["y_score"].mean())
        base["adni_threshold"] = float(sub["adni_threshold"].mean())
        base["y_pred"] = int(sub["y_pred"].mean() >= 0.5)
        base["fold_score_min"] = float(sub["y_score"].min())
        base["fold_score_max"] = float(sub["y_score"].max())
        base["fold_score_std"] = float(sub["y_score"].std(ddof=0))
        rows.append(base)
    return pd.DataFrame(rows)


def metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    conf_rows = []
    for keys, sub in predictions.groupby(["build_candidate", "adni_model", "prediction_level"], dropna=False):
        build_candidate, adni_model, level = keys
        m = {
            "build_candidate": build_candidate,
            "adni_model": adni_model,
            "prediction_level": level,
            "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
        }
        m.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        metric_rows.append(m)
        conf_rows.append(
            {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "prediction_level": level,
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "balanced_accuracy": m["balanced_accuracy"],
                "f1": m["f1"],
            }
        )
    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(conf_rows)
    dist = (
        predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")]
        .groupby(["build_candidate", "adni_model", "diagnosis"], dropna=False)["y_score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "score_mean", "std": "score_std", "min": "score_min", "median": "score_median", "max": "score_max"})
    )
    sensitivity_rows = []
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")]
    for model, sub_model in ens.groupby("adni_model", dropna=False):
        piv = sub_model.pivot_table(index=["SubjectID", "session_id"], columns="build_candidate", values="y_score", aggfunc="first")
        if {"concatenated_timeseries", "runwise_connectome_average"}.issubset(piv.columns):
            diff = piv["concatenated_timeseries"] - piv["runwise_connectome_average"]
            sensitivity_rows.append(
                {
                    "adni_model": model,
                    "n_paired_subject_sessions": int(diff.dropna().shape[0]),
                    "concat_minus_runwise_mean": float(diff.mean()),
                    "concat_minus_runwise_std": float(diff.std(ddof=0)),
                    "concat_minus_runwise_min": float(diff.min()),
                    "concat_minus_runwise_max": float(diff.max()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    return metrics, confusion, dist, sensitivity


def dry_run_report(args: argparse.Namespace, artifact_df: pd.DataFrame, output_dir: Path) -> None:
    missing = artifact_df[~artifact_df["exists"]]
    text = [
        "# OASIS External Scoring Dry Run",
        "",
        "Status: `dry_run_only`",
        "",
        "No OASIS scoring was executed. No VAE or classifier was trained.",
        "",
        "## Guardrails",
        "",
        "- OASIS labels must only be used for final external metrics.",
        "- No OASIS threshold fitting or model selection is allowed.",
        "- Stage B estimators are not serialized in the ADNI classifier-only readout.",
        "- Confirmed scoring reconstructs logreg_l2 in memory from ADNI train/dev latents and saved ADNI hyperparameters only.",
        "",
        "## Artifact Check",
        "",
        f"- Checked artifacts: {len(artifact_df)}",
        f"- Missing artifacts: {len(missing)}",
    ]
    if not missing.empty:
        text += ["", "### Missing", "", missing.to_markdown(index=False)]
    text += [
        "",
        "## Scoring Command",
        "",
        "```bash",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/score_oasis_tanda_20260525_external_adni.py --confirm-score",
        "```",
    ]
    (output_dir / "dry_run_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def final_recommendation(output_dir: Path, scored: bool, metrics: pd.DataFrame | None = None) -> None:
    if not scored:
        text = """# Final Recommendation

Status: `pending_external_scoring`

The OASIS external scoring package is prepared, but no predictions were generated in dry-run mode.
Run with `--confirm-score` after confirming that the OASIS connectome tensors are final.

No OASIS-based tuning, threshold selection, or training is part of this package.
"""
    else:
        primary = metrics[metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
        text = """# Final Recommendation

Status: `external_scoring_complete`

OASIS was scored as external validation only. ADNI fold-specific VAE checkpoints, ADNI normalization parameters, ADNI train/dev readout reconstruction, and ADNI-derived thresholds were used. No OASIS labels were used for model selection or threshold fitting.

## Primary Metrics

""" + primary.to_markdown(index=False) + "\n"
    (output_dir / "final_recommendation.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs(args)
    artifact_df = validate_artifacts(args, specs)
    write_csv_md(
        artifact_df,
        output_dir / "artifact_validation.csv",
        output_dir / "artifact_validation.md",
        "Artifact Validation",
    )

    if not bool(args.confirm_score):
        dry_run_report(args, artifact_df, output_dir)
        final_recommendation(output_dir, scored=False)
        write_json(
            output_dir / "command_log.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__).resolve()),
                "mode": "dry_run_only",
                "confirm_score": False,
                "oasis_scored": False,
                "trained_on_oasis": False,
                "oasis_threshold_tuning": False,
                "artifact_missing_count": int((~artifact_df["exists"]).sum()),
            },
        )
        print(json.dumps({"output_dir": str(output_dir), "mode": "dry_run_only", "oasis_scored": False}, indent=2))
        return 0

    missing = artifact_df[~artifact_df["exists"]]
    if not missing.empty:
        raise FileNotFoundError("Cannot score OASIS; missing artifacts:\n" + missing.to_string(index=False))

    device = resolve_device(args.device)
    prediction_frames = []
    for candidate in TENSOR_CANDIDATES:
        tensor, subjects, channel_names = load_oasis_tensor(args.connectome_dir, candidate)
        for spec in specs:
            if not spec.include:
                continue
            pred = score_one_model_on_one_tensor(
                spec=spec,
                tensor=tensor,
                oasis_subjects=subjects,
                oasis_channel_names=channel_names,
                batch_size=int(args.batch_size),
                device=device,
            )
            pred["build_candidate"] = candidate
            prediction_frames.append(pred)
    fold_predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble = make_ensemble_predictions(fold_predictions)
    predictions = pd.concat([fold_predictions, ensemble], ignore_index=True, sort=False)
    metrics, confusion, dist, sensitivity = metrics_tables(predictions)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_csv_md(metrics, output_dir / "primary_metrics.csv", output_dir / "primary_metrics.md", "Primary Metrics")
    write_csv_md(confusion, output_dir / "confusion_matrices.csv", output_dir / "confusion_matrices.md", "Confusion Matrices")
    write_csv_md(dist, output_dir / "score_distribution_summary.csv", output_dir / "score_distribution_summary.md", "Score Distribution Summary")
    write_csv_md(sensitivity, output_dir / "run_combination_sensitivity.csv", output_dir / "run_combination_sensitivity.md", "Run Combination Sensitivity")
    final_recommendation(output_dir, scored=True, metrics=metrics)
    write_json(
        output_dir / "command_log.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "mode": "external_scoring",
            "confirm_score": True,
            "oasis_scored": True,
            "trained_on_oasis": False,
            "oasis_threshold_tuning": False,
            "oasis_model_selection": False,
            "adni_stageb_readout_reconstructed_in_memory": True,
            "readout_reconstruction_note": "Stage B classifier-only estimators were not serialized; logreg_l2 was refit in memory on ADNI train/dev latents using saved ADNI best C values.",
            "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
            "device": str(device),
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "mode": "external_scoring", "oasis_scored": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
