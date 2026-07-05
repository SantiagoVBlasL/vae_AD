#!/usr/bin/env python
"""Read-only OASIS external scoring sensitivity audit for secondary ADNI models.

This script scores already-built OASIS tensors with ADNI-trained fold models.
It does not train on OASIS, does not tune thresholds on OASIS, and does not use
OASIS labels for model selection. Stage B logreg_l2 estimators are reconstructed
in memory from each ADNI run's train/dev latent cache and saved ADNI
hyperparameters; thresholds are the saved ADNI inner-OOF thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import (  # noqa: E402
    apply_conditioning_transformer,
    apply_normalization_params,
    encode_mu,
    load_config,
    make_model,
    score_1d,
)
from score_oasis_tanda_20260525_external_adni import (  # noqa: E402
    DEFAULT_CONNECTOME_DIR,
    DEFAULT_PRIMARY_ADNI_RUN,
    PRIMARY_THRESHOLD_STRATEGY,
    TENSOR_CANDIDATES,
    binary_metrics,
    fit_adni_stageb_logreg,
    load_best_c,
    load_oasis_tensor,
    load_thresholds,
    require,
    resolve_device,
    select_channels,
    write_csv_md,
    write_json,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring_secondary_models"
)
DEFAULT_PRIMARY_SCORING_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring"
)
DEFAULT_CH1_RUN = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1b_ch1_only_offdiag_channelmean_horizon4480_cycles56_full_5x5"
)
DEFAULT_MFR_FULL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"
)
DEFAULT_MFR_RUN = DEFAULT_MFR_FULL_ROOT / "runs" / "ch1_0_2_decoder_only_manufacturer"


@dataclass(frozen=True)
class ModelSpec:
    model_label: str
    run_dir: Path
    readout_dir: Path
    role: str
    optional: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME_DIR)
    parser.add_argument("--primary-scoring-dir", type=Path, default=DEFAULT_PRIMARY_SCORING_DIR)
    parser.add_argument("--primary-adni-run-dir", type=Path, default=DEFAULT_PRIMARY_ADNI_RUN)
    parser.add_argument("--secondary-ch1-run-dir", type=Path, default=DEFAULT_CH1_RUN)
    parser.add_argument("--manufacturer-conditioned-run-dir", type=Path, default=DEFAULT_MFR_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    primary = normalize_path(args.primary_adni_run_dir)
    ch1 = normalize_path(args.secondary_ch1_run_dir)
    mfr = normalize_path(args.manufacturer_conditioned_run_dir)
    return [
        ModelSpec(
            model_label="locked_v5_1b_ch1_0_2_horizon4480",
            run_dir=primary,
            readout_dir=primary / "classifier_only_readout",
            role="reference_primary",
            optional=False,
        ),
        ModelSpec(
            model_label="secondary_ch1_only_offdiag_channelmean",
            run_dir=ch1,
            readout_dir=ch1 / "classifier_only_readout",
            role="secondary_simplified_auroc",
            optional=False,
        ),
        ModelSpec(
            model_label="secondary_manufacturer_conditioned_deconfounding",
            run_dir=mfr,
            readout_dir=mfr / "classifier_only_readout_z_plus_age_sex",
            role="secondary_deconfounding_sensitivity",
            optional=True,
        ),
    ]


def model_required_paths(spec: ModelSpec) -> list[Path]:
    paths = [
        spec.run_dir / "run_config.json",
        spec.readout_dir / "classifier_sweep_model_status.csv",
        spec.readout_dir / "classifier_sweep_thresholds_by_fold.csv",
        spec.readout_dir / "latent_cache",
    ]
    try:
        cfg = load_config(spec.run_dir)
        outer_folds = int(cfg.get("outer_folds", 5))
        conditioned = str(cfg.get("vae_conditioning_mode", "none")) != "none"
    except Exception:
        outer_folds = 5
        conditioned = False
    for fold in range(1, outer_folds + 1):
        paths.extend(
            [
                spec.run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt",
                spec.run_dir / f"fold_{fold}" / "vae_norm_params.joblib",
                spec.readout_dir / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
            ]
        )
        if conditioned:
            paths.append(spec.run_dir / f"fold_{fold}" / "vae_conditioning_transformer.joblib")
    return paths


def validate_artifacts(args: argparse.Namespace, specs: Sequence[ModelSpec]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    connectome_dir = normalize_path(args.connectome_dir)
    primary_scoring_dir = normalize_path(args.primary_scoring_dir)
    base_paths = [
        (connectome_dir, "oasis_connectome_dir", "connectome_dir"),
        (connectome_dir / "subject_manifest.csv", "oasis_subject_manifest", "subject_manifest"),
        (primary_scoring_dir / "primary_metrics.csv", "primary_oasis_scoring_metrics", "primary_scoring"),
        (primary_scoring_dir / "predictions.csv", "primary_oasis_scoring_predictions", "primary_scoring"),
    ]
    for path, artifact_type, label in base_paths:
        rows.append(
            {
                "artifact_type": artifact_type,
                "label": label,
                "model_label": "",
                "path": str(path),
                "exists": path.exists(),
                "optional": False,
            }
        )
    for candidate, filename in TENSOR_CANDIDATES.items():
        p = connectome_dir / filename
        rows.append(
            {
                "artifact_type": "oasis_tensor",
                "label": candidate,
                "model_label": "",
                "path": str(p),
                "exists": p.exists(),
                "optional": False,
            }
        )
    for spec in specs:
        for path in model_required_paths(spec):
            rows.append(
                {
                    "artifact_type": "adni_model_artifact",
                    "label": path.name,
                    "model_label": spec.model_label,
                    "path": str(path),
                    "exists": path.exists(),
                    "optional": bool(spec.optional),
                }
            )
    df = pd.DataFrame(rows)
    availability = (
        df[df["model_label"].astype(str).ne("")]
        .groupby("model_label", dropna=False)["exists"]
        .all()
        .rename("model_available")
        .reset_index()
    )
    return df.merge(availability, on="model_label", how="left")


def available_specs(specs: Sequence[ModelSpec], artifact_df: pd.DataFrame) -> list[ModelSpec]:
    available: list[ModelSpec] = []
    for spec in specs:
        sub = artifact_df[artifact_df["model_label"].eq(spec.model_label)]
        ok = bool(sub["exists"].all()) if not sub.empty else False
        if ok:
            available.append(spec)
        elif not spec.optional:
            missing = sub[~sub["exists"]]["path"].tolist()
            raise FileNotFoundError(f"Required model artifacts missing for {spec.model_label}:\n" + "\n".join(missing))
    return available


def fold_conditioning_for_oasis(spec: ModelSpec, cfg: Dict[str, Any], fold: int, oasis_subjects: pd.DataFrame) -> np.ndarray | None:
    if str(cfg.get("vae_conditioning_mode", "none")) == "none":
        return None
    transformer_path = spec.run_dir / f"fold_{fold}" / "vae_conditioning_transformer.joblib"
    transformer = joblib.load(transformer_path)
    return apply_conditioning_transformer(oasis_subjects, transformer)


def score_one_spec_on_tensor(
    spec: ModelSpec,
    tensor: np.ndarray,
    oasis_subjects: pd.DataFrame,
    oasis_channel_names: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    cfg = load_config(spec.run_dir)
    selected_names = cfg["selected_channel_names"]
    x_oasis = select_channels(tensor, oasis_channel_names, selected_names)
    rows: list[pd.DataFrame] = []
    for fold in range(1, int(cfg.get("outer_folds", 5)) + 1):
        fold_dir = spec.run_dir / f"fold_{fold}"
        checkpoint = fold_dir / f"vae_model_fold_{fold}.pt"
        norm_path = fold_dir / "vae_norm_params.joblib"
        require(checkpoint, f"{spec.model_label} fold {fold} checkpoint")
        require(norm_path, f"{spec.model_label} fold {fold} normalization params")
        norm_params = joblib.load(norm_path)
        x_norm = apply_normalization_params(x_oasis, norm_params)
        condition = fold_conditioning_for_oasis(spec, cfg, fold, oasis_subjects)
        conditioning_dim = 0 if condition is None else int(condition.shape[1])
        model = make_model(
            cfg,
            image_size=x_oasis.shape[-1],
            n_channels=x_oasis.shape[1],
            device=device,
            conditioning_dim_override=conditioning_dim,
        )
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        mu = encode_mu(model, x_norm, batch_size=batch_size, device=device, condition=condition)

        train_df = pd.read_csv(spec.readout_dir / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv")
        c_value = load_best_c(spec.readout_dir, fold)
        clf = fit_adni_stageb_logreg(train_df, c_value, int(cfg.get("seed", 42)) + fold)
        mu_df = pd.DataFrame(mu, columns=[f"mu_{i}" for i in range(mu.shape[1])])
        feature_df = pd.concat([mu_df, oasis_subjects[["Age", "Sex"]].reset_index(drop=True)], axis=1)
        y_score = score_1d(clf, feature_df)
        threshold = load_thresholds(spec.readout_dir, fold)
        pred = oasis_subjects.copy()
        pred["adni_model"] = spec.model_label
        pred["model_role"] = spec.role
        pred["fold"] = fold
        pred["selected_channel_names"] = "|".join(selected_names)
        pred["vae_conditioning_mode"] = str(cfg.get("vae_conditioning_mode", "none"))
        pred["vae_conditioning_vars"] = str(cfg.get("vae_conditioning_vars", "none"))
        pred["readout"] = "stageb_logreg_l2_reconstructed_from_adni_trainDev"
        pred["readout_feature_set"] = "z_plus_age_sex"
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
    rows: list[pd.Series] = []
    for _key, sub in fold_predictions.groupby(group_cols, dropna=False):
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


def metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    conf_rows: list[dict[str, Any]] = []
    for keys, sub in predictions.groupby(["build_candidate", "adni_model", "model_role", "prediction_level"], dropna=False):
        build_candidate, adni_model, role, level = keys
        row: dict[str, Any] = {
            "build_candidate": build_candidate,
            "external_role": "primary" if build_candidate == "concatenated_timeseries" else "sensitivity",
            "adni_model": adni_model,
            "model_role": role,
            "prediction_level": level,
            "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
            "oasis_threshold_fitting": False,
            "sensitivity_analysis_only": role != "reference_primary",
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        metric_rows.append(row)
        conf_rows.append(
            {
                "build_candidate": build_candidate,
                "external_role": row["external_role"],
                "adni_model": adni_model,
                "model_role": role,
                "prediction_level": level,
                "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "balanced_accuracy": row["balanced_accuracy"],
                "f1": row["f1"],
            }
        )
    dist = (
        predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")]
        .groupby(["build_candidate", "adni_model", "model_role", "diagnosis"], dropna=False)["y_score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "score_mean",
                "std": "score_std",
                "min": "score_min",
                "median": "score_median",
                "max": "score_max",
            }
        )
    )
    return pd.DataFrame(metric_rows), pd.DataFrame(conf_rows), dist


def threshold_transfer_table(predictions: pd.DataFrame) -> pd.DataFrame:
    ens = predictions[predictions["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
    rows: list[dict[str, Any]] = []
    for keys, sub in ens.groupby(["build_candidate", "adni_model", "model_role", "diagnosis"], dropna=False):
        build_candidate, model, role, diagnosis = keys
        threshold = float(sub["adni_threshold"].mean())
        scores = sub["y_score"].astype(float)
        y_true = sub["y_true"].astype(int)
        is_ad = int(y_true.iloc[0]) == 1
        rows.append(
            {
                "build_candidate": build_candidate,
                "external_role": "primary" if build_candidate == "concatenated_timeseries" else "sensitivity",
                "adni_model": model,
                "model_role": role,
                "diagnosis": diagnosis,
                "n": int(len(sub)),
                "mean_adni_threshold": threshold,
                "score_q05": float(scores.quantile(0.05)),
                "score_q25": float(scores.quantile(0.25)),
                "score_median": float(scores.median()),
                "score_q75": float(scores.quantile(0.75)),
                "score_q95": float(scores.quantile(0.95)),
                "fraction_below_mean_adni_threshold": float((scores < threshold).mean()),
                "fraction_above_or_equal_mean_adni_threshold": float((scores >= threshold).mean()),
                "ad_below_threshold_fraction": float((scores < threshold).mean()) if is_ad else np.nan,
                "cn_above_threshold_fraction": float((scores >= threshold).mean()) if not is_ad else np.nan,
                "threshold_use": "ADNI-derived fold thresholds; OASIS descriptive only",
            }
        )
    return pd.DataFrame(rows)


def dry_run_report(output_dir: Path, artifact_df: pd.DataFrame, specs: Sequence[ModelSpec]) -> None:
    missing_required = artifact_df[(~artifact_df["exists"]) & (~artifact_df["optional"])]
    missing_optional = artifact_df[(~artifact_df["exists"]) & (artifact_df["optional"])]
    lines = [
        "# OASIS Secondary-Model External Scoring Dry Run",
        "",
        "Status: `dry_run_only`",
        "",
        "No scoring was executed. No model training, OASIS threshold fitting, or OASIS-based model selection is part of this package.",
        "",
        "## Planned Models",
        "",
    ]
    lines.extend([f"- `{s.model_label}` ({s.role})" for s in specs])
    lines += [
        "",
        "## Artifact Status",
        "",
        f"- Missing required artifacts: {len(missing_required)}",
        f"- Missing optional artifacts: {len(missing_optional)}",
    ]
    if not missing_required.empty:
        lines += ["", "### Missing Required", "", missing_required.to_markdown(index=False)]
    if not missing_optional.empty:
        lines += ["", "### Missing Optional", "", missing_optional.to_markdown(index=False)]
    (output_dir / "dry_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_recommendation(output_dir: Path, metrics: pd.DataFrame, artifact_df: pd.DataFrame) -> None:
    ens = metrics[
        metrics["prediction_level"].eq("ensemble_mean_score_majority_vote")
        & metrics["build_candidate"].eq("concatenated_timeseries")
    ].copy()
    ens = ens.sort_values(["auc", "pr_auc"], ascending=[False, False])
    best = ens.iloc[0] if not ens.empty else None
    lines = [
        "# Final Recommendation",
        "",
        "Decision: `secondary_models_are_external_sensitivity_only`",
        "",
        "The OASIS pilot is an external validation sensitivity analysis. It must not be used to promote or select a manuscript model by itself.",
        "All reported operating points use ADNI-derived thresholds; no OASIS threshold fitting or model selection was performed.",
        "",
        "## Primary Concatenated-Timeseries Comparison",
        "",
    ]
    if ens.empty:
        lines.append("_No concatenated-timeseries ensemble metrics were generated._")
    else:
        lines.append(ens.to_markdown(index=False))
    if best is not None:
        lines += [
            "",
            "## Interpretation",
            "",
            f"The highest concatenated-timeseries AUROC in this sensitivity audit was `{best['adni_model']}` "
            f"(AUC={float(best['auc']):.4f}, PR-AUC={float(best['pr_auc']):.4f}). "
            "This is descriptive external evidence only. The locked ADNI model remains the manuscript reference unless a separately pre-specified validation framework supports a change.",
        ]
    missing_mfr = artifact_df[
        artifact_df["model_label"].eq("secondary_manufacturer_conditioned_deconfounding") & (~artifact_df["exists"])
    ]
    if not missing_mfr.empty:
        lines += [
            "",
            "## Manufacturer-Conditioned Model Availability",
            "",
            "The Manufacturer-conditioned model was not scored because one or more optional artifacts were missing.",
        ]
    (output_dir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = normalize_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs(args)
    artifact_df = validate_artifacts(args, specs)
    write_csv_md(
        artifact_df,
        output_dir / "artifact_validation.csv",
        output_dir / "artifact_validation.md",
        "Artifact Validation",
    )
    if args.dry_run:
        dry_run_report(output_dir, artifact_df, specs)
        write_json(
            output_dir / "command_log.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__).resolve()),
                "mode": "dry_run_only",
                "trained_on_oasis": False,
                "oasis_threshold_fitting": False,
                "oasis_model_selection": False,
                "missing_required_artifacts": int(((~artifact_df["exists"]) & (~artifact_df["optional"])).sum()),
                "missing_optional_artifacts": int(((~artifact_df["exists"]) & artifact_df["optional"]).sum()),
            },
        )
        print(json.dumps({"output_dir": str(output_dir), "mode": "dry_run_only"}, indent=2))
        return 0

    available = available_specs(specs, artifact_df)
    device = resolve_device(args.device)
    prediction_frames: list[pd.DataFrame] = []
    for build_candidate in TENSOR_CANDIDATES:
        tensor, subjects, channel_names = load_oasis_tensor(normalize_path(args.connectome_dir), build_candidate)
        for spec in available:
            pred = score_one_spec_on_tensor(
                spec=spec,
                tensor=tensor,
                oasis_subjects=subjects,
                oasis_channel_names=channel_names,
                batch_size=int(args.batch_size),
                device=device,
            )
            pred["build_candidate"] = build_candidate
            prediction_frames.append(pred)

    fold_predictions = pd.concat(prediction_frames, ignore_index=True)
    ensemble = make_ensemble_predictions(fold_predictions)
    predictions = pd.concat([fold_predictions, ensemble], ignore_index=True, sort=False)
    metrics, confusion, dist = metric_tables(predictions)
    threshold_summary = threshold_transfer_table(predictions)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_csv_md(metrics, output_dir / "primary_metrics.csv", output_dir / "primary_metrics.md", "Primary Metrics")
    write_csv_md(confusion, output_dir / "confusion_matrices.csv", output_dir / "confusion_matrices.md", "Confusion Matrices")
    write_csv_md(
        dist,
        output_dir / "score_distribution_summary.csv",
        output_dir / "score_distribution_summary.md",
        "Score Distribution Summary",
    )
    write_csv_md(
        threshold_summary,
        output_dir / "threshold_transfer_summary.csv",
        output_dir / "threshold_transfer_summary.md",
        "Threshold Transfer Summary",
    )
    write_final_recommendation(output_dir, metrics, artifact_df)
    write_json(
        output_dir / "command_log.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "mode": "external_scoring_secondary_models",
            "trained_on_oasis": False,
            "oasis_threshold_fitting": False,
            "oasis_model_selection": False,
            "oasis_labels_used_for": "final_external_metrics_only",
            "threshold_strategy": PRIMARY_THRESHOLD_STRATEGY,
            "prediction_level_primary": "ensemble_mean_score_majority_vote",
            "concatenated_timeseries_role": "primary",
            "runwise_connectome_average_role": "sensitivity",
            "available_models_scored": [s.model_label for s in available],
            "output_dir": str(output_dir),
            "device": str(device),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "mode": "external_scoring_secondary_models",
                "models_scored": [s.model_label for s in available],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
