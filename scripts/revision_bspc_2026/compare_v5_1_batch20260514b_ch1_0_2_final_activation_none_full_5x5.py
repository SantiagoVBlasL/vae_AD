#!/usr/bin/env python3
"""Read-only final_activation=none FULL 5x5 comparison audit for ADNI v5.1.

Compares final_activation=none [1,0,2] against current [1,0,2], objective-v2
[1,0,2], fc0 [1,0,2], [1], and [1,4] using only classifier-only logreg_l2.
The primary operating point is inner_oof_target_sens_ge_0p70_max_spec.
No training or dataset mutation occurs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_CANDIDATE_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_final_activation_none_full_5x5"
DEFAULT_CANDIDATE_READOUT_DIR = DEFAULT_CANDIDATE_RUN_DIR / "classifier_only_readout"
DEFAULT_OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_final_activation_none_full_5x5_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRIC_COLUMNS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]

RUNS = [
    {
        "run_id": "current_ch1_0_2",
        "label": "current [1,0,2]",
        "channels": "[1,0,2]",
        "objective": "mse_sum_batchmean_current",
        "readout_dir": RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep",
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate",
        "notes": "reference multi-model classifier-only sweep, filtered to logreg_l2",
    },
    {
        "run_id": "objective_v2_ch1_0_2",
        "label": "objective-v2 [1,0,2]",
        "channels": "[1,0,2]",
        "objective": "offdiag_channelmean_sum",
        "readout_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5/classifier_only_readout",
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_objective_v2_full_5x5",
        "notes": "new objective-v2 full candidate",
    },
    {
        "run_id": "fc0_ch1_0_2",
        "label": "fc0 [1,0,2]",
        "channels": "[1,0,2]",
        "objective": "mse_sum_batchmean_current",
        "readout_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_fc0_full_5x5/classifier_only_readout",
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_fc0_full_5x5",
        "notes": "intermediate_fc_dim_vae=0 full candidate, not promoted",
    },
    {
        "run_id": "ch1",
        "label": "[1]",
        "channels": "[1]",
        "objective": "mse_sum_batchmean_current",
        "readout_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate/classifier_only_readout",
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate",
        "notes": "parsimonious single-channel candidate",
    },
    {
        "run_id": "ch1_4",
        "label": "[1,4]",
        "channels": "[1,4]",
        "objective": "mse_sum_batchmean_current",
        "readout_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate/classifier_only_readout",
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate",
        "notes": "secondary FAST pair candidate",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN_DIR)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_output_dir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.{digits}f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def required_readout_files(readout_dir: Path) -> List[Path]:
    return [
        readout_dir / "classifier_sweep_pooled_metrics.csv",
        readout_dir / "classifier_sweep_foldwise_metrics.csv",
        readout_dir / "classifier_sweep_predictions.csv",
        readout_dir / "classifier_sweep_thresholds_by_fold.csv",
        readout_dir / "command_log.json",
    ]


def validate_readout(run: Dict[str, Any]) -> Dict[str, Any]:
    readout_dir = resolve(run["readout_dir"])
    missing = [p.name for p in required_readout_files(readout_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run['run_id']} missing readout files in {readout_dir}: {missing}")
    log = read_json_optional(readout_dir / "command_log.json")
    requested = log.get("classifiers_requested")
    if isinstance(requested, list) and PRIMARY_MODEL not in requested:
        raise RuntimeError(f"{run['run_id']} readout does not include {PRIMARY_MODEL}: {requested}")
    threshold_selection = str(log.get("threshold_selection", ""))
    if threshold_selection and "inner_cv_oof" not in threshold_selection:
        raise RuntimeError(f"{run['run_id']} command_log threshold_selection is not inner-CV OOF: {threshold_selection}")
    for key in ["tensor_modified", "metadata_modified", "ledger_modified", "vae_retrained"]:
        value = log.get(key)
        if value not in (False, None):
            raise RuntimeError(f"{run['run_id']} command_log has {key}={value}")
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    primary = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    fixed = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(FIXED_THRESHOLD))]
    if len(primary) != 1:
        raise RuntimeError(f"{run['run_id']} expected one primary pooled row, found {len(primary)}")
    if len(fixed) != 1:
        raise RuntimeError(f"{run['run_id']} expected one fixed_0p5 pooled row, found {len(fixed)}")
    return log


def primary_pooled_row(run: Dict[str, Any]) -> Dict[str, Any]:
    readout_dir = resolve(run["readout_dir"])
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))].iloc[0]
    out: Dict[str, Any] = {
        "run_id": run["run_id"],
        "label": run["label"],
        "channels": run["channels"],
        "objective": run["objective"],
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
        "notes": run["notes"],
    }
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", *METRIC_COLUMNS, "predicted_ad_rate"]:
        out[col] = row.get(col, np.nan)
    return out


def threshold_rows(run: Dict[str, Any]) -> pd.DataFrame:
    readout_dir = resolve(run["readout_dir"])
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    df = pooled[
        pooled["model_name"].eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])
    ].copy()
    df.insert(0, "run_id", run["run_id"])
    df.insert(1, "label", run["label"])
    df.insert(2, "channels", run["channels"])
    keep = [
        "run_id",
        "label",
        "channels",
        "model_name",
        "threshold_strategy",
        "threshold",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        *METRIC_COLUMNS,
        "predicted_ad_rate",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def foldwise_rows(run: Dict[str, Any]) -> pd.DataFrame:
    readout_dir = resolve(run["readout_dir"])
    df = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    df.insert(0, "run_id", run["run_id"])
    df.insert(1, "label", run["label"])
    df.insert(2, "channels", run["channels"])
    keep = [
        "run_id",
        "label",
        "channels",
        "fold",
        "threshold",
        "threshold_selection_context",
        "inner_cv_context",
        "minimum_inner_stratum_count",
        "best_inner_auc",
        "best_params",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        *METRIC_COLUMNS,
        "predicted_ad_rate",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values(["run_id", "fold"]).reset_index(drop=True)


def prediction_rows(run: Dict[str, Any], threshold: str = PRIMARY_THRESHOLD) -> pd.DataFrame:
    readout_dir = resolve(run["readout_dir"])
    df = pd.read_csv(readout_dir / "classifier_sweep_predictions.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(threshold))].copy()
    df.insert(0, "run_id", run["run_id"])
    df.insert(1, "label", run["label"])
    df.insert(2, "channels", run["channels"])
    return df.reset_index(drop=True)


def metric_from_predictions(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        y = sub["y_true"].astype(int).to_numpy()
        score = sub["y_score"].astype(float).to_numpy()
        pred = sub["y_pred"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        row.update(
            {
                "n": int(len(sub)),
                "n_cn": int((y == 0).sum()),
                "n_ad": int((y == 1).sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": (tn + tp) / len(sub) if len(sub) else np.nan,
                "sensitivity": sens,
                "specificity": spec,
                "balanced_accuracy": np.nanmean([sens, spec]),
                "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                "predicted_ad_rate": (fp + tp) / len(sub) if len(sub) else np.nan,
                "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(predictions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in predictions.columns:
        return pd.DataFrame()
    group_cols = ["run_id", "label", "channels", group_col]
    out = metric_from_predictions(predictions, group_cols)
    return out.sort_values([group_col, "auc", "run_id"], ascending=[True, False, True]).reset_index(drop=True)


def fold4_deep_dive(predictions: pd.DataFrame) -> pd.DataFrame:
    fold4 = predictions[predictions["fold"].eq(4)].copy()
    if fold4.empty:
        return fold4
    fold4["diagnosis"] = np.where(fold4["y_true"].eq(1), "AD", "CN")
    fold4["prediction"] = np.where(fold4["y_pred"].eq(1), "AD_like", "CN_like")
    fold4["error_type"] = np.select(
        [
            fold4["y_true"].eq(1) & fold4["y_pred"].eq(0),
            fold4["y_true"].eq(0) & fold4["y_pred"].eq(1),
        ],
        ["false_negative_AD", "false_positive_CN"],
        default="correct",
    )
    fold4["margin_to_threshold"] = fold4["y_score"].astype(float) - fold4["threshold"].astype(float)
    keep = [
        "run_id",
        "label",
        "channels",
        "SubjectID",
        "diagnosis",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "tensor_source",
        "fold",
        "model_name",
        "threshold_strategy",
        "threshold",
        "y_score",
        "margin_to_threshold",
        "prediction",
        "error_type",
        "y_true",
        "y_pred",
    ]
    return fold4[[c for c in keep if c in fold4.columns]].sort_values(
        ["run_id", "error_type", "diagnosis", "Manufacturer", "SubjectID"]
    ).reset_index(drop=True)


def fold4_error_summary(fold4: pd.DataFrame) -> pd.DataFrame:
    if fold4.empty:
        return pd.DataFrame()
    errors = fold4[fold4["error_type"].ne("correct")].copy()
    if errors.empty:
        return pd.DataFrame()
    return (
        errors.groupby(["run_id", "label", "channels", "error_type", "Manufacturer"], dropna=False)
        .size()
        .reset_index(name="n_errors")
        .sort_values(["run_id", "error_type", "Manufacturer"])
        .reset_index(drop=True)
    )


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame({"read_error": [str(exc)]})


def fold_dirs(run: Dict[str, Any]) -> List[Path]:
    run_dir = resolve(run["run_dir"])
    dirs = []
    for fold in range(1, 6):
        d = run_dir / f"fold_{fold}"
        if d.exists():
            dirs.append(d)
    return dirs


def add_context(df: pd.DataFrame, run: Dict[str, Any], fold: int, source_file: str, qc_kind: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    context = {
        "run_id": run["run_id"],
        "label": run["label"],
        "fold": fold,
        "qc_kind": qc_kind,
        "source_file": source_file,
    }
    for key, value in context.items():
        out[key] = value
    ordered = list(context) + [col for col in out.columns if col not in context]
    return out[ordered]


def collect_reconstruction_qc(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run in runs:
        for d in fold_dirs(run):
            fold = int(d.name.split("_")[-1])
            for suffix in ["raw", "norm", "recon"]:
                name = f"fold_{fold}_dist_{suffix}.csv"
                df = read_optional_csv(d / name)
                rows.append(add_context(df, run, fold, name, f"dist_{suffix}"))
    return pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()


def collect_rate_distortion_qc(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run in runs:
        for d in fold_dirs(run):
            fold = int(d.name.split("_")[-1])
            name = f"fold_{fold}_rate_distortion.csv"
            df = read_optional_csv(d / name)
            rows.append(add_context(df, run, fold, name, "rate_distortion"))
    return pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()


def collect_latent_qc(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run in runs:
        for d in fold_dirs(run):
            fold = int(d.name.split("_")[-1])
            for name, kind in [
                ("latent_qc_metrics.csv", "latent_qc_metrics"),
                (f"fold_{fold}_trainDev_latent_info_summary.csv", "trainDev_latent_info_summary"),
                (f"fold_{fold}_test_latent_info_summary.csv", "test_latent_info_summary"),
            ]:
                df = read_optional_csv(d / name)
                rows.append(add_context(df, run, fold, name, kind))
    return pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()


def collect_scanner_leakage_qc(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run in runs:
        for d in fold_dirs(run):
            fold = int(d.name.split("_")[-1])
            for name, kind in [
                (f"fold_{fold}_scanner_leakage_summary.csv", "trainDev_scanner_leakage_summary"),
                (f"fold_{fold}_test_scanner_leakage_summary.csv", "test_scanner_leakage_summary"),
                (f"fold_{fold}_scanner_leakage.csv", "trainDev_scanner_leakage"),
                (f"fold_{fold}_test_scanner_leakage.csv", "test_scanner_leakage"),
            ]:
                df = read_optional_csv(d / name)
                rows.append(add_context(df, run, fold, name, kind))
    return pd.concat([x for x in rows if not x.empty], ignore_index=True) if rows else pd.DataFrame()


def write_table_pair(output_dir: Path, stem: str, df: pd.DataFrame, digits: int = 4) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(md_table(df, digits=digits), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    main: pd.DataFrame,
    foldwise: pd.DataFrame,
    manufacturer: pd.DataFrame,
    sex: pd.DataFrame,
    thresholds: pd.DataFrame,
    fold4: pd.DataFrame,
    fold4_summary: pd.DataFrame,
    reconstruction_qc: pd.DataFrame,
    rate_distortion_qc: pd.DataFrame,
    latent_qc: pd.DataFrame,
    scanner_leakage_qc: pd.DataFrame,
    logs: Dict[str, Dict[str, Any]],
) -> None:
    write_table_pair(output_dir, "final_activation_none_full_main_comparison", main)
    write_table_pair(output_dir, "final_activation_none_full_foldwise_comparison", foldwise)
    write_table_pair(output_dir, "final_activation_none_full_subgroup_by_manufacturer", manufacturer)
    write_table_pair(output_dir, "final_activation_none_full_subgroup_by_sex", sex)
    write_table_pair(output_dir, "final_activation_none_full_threshold_comparison", thresholds)
    write_table_pair(output_dir, "final_activation_none_full_fold4_deep_dive", fold4)
    write_table_pair(output_dir, "final_activation_none_full_reconstruction_qc", reconstruction_qc)
    write_table_pair(output_dir, "final_activation_none_full_rate_distortion_qc", rate_distortion_qc)
    write_table_pair(output_dir, "final_activation_none_full_latent_qc", latent_qc)
    write_table_pair(output_dir, "final_activation_none_full_scanner_leakage_qc", scanner_leakage_qc)
    fold4_summary.to_csv(output_dir / "fold4_error_summary.csv", index=False)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "training_launched": False,
        "readout_logs": logs,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_recommendation(output_dir, main, thresholds, manufacturer, sex, foldwise, fold4_summary)


def write_recommendation(
    output_dir: Path,
    main: pd.DataFrame,
    thresholds: pd.DataFrame,
    manufacturer: pd.DataFrame,
    sex: pd.DataFrame,
    foldwise: pd.DataFrame,
    fold4_summary: pd.DataFrame,
) -> None:
    ranked = main.sort_values("auc", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    by_id = main.set_index("run_id")
    lines = [
        "# Manuscript Recommendation",
        "",
        "This audit is read-only and compares only classifier-only `logreg_l2` readouts.",
        f"Primary operating point: `{PRIMARY_THRESHOLD}` selected from true inner-CV out-of-fold predictions.",
        "",
        "## Primary Result",
        "",
        f"- Best mean/pooled ranking by AUC in this audit: `{best['label']}` with AUC={best['auc']:.4f}, PR-AUC={best['pr_auc']:.4f}, BA={best['balanced_accuracy']:.4f}, sensitivity={best['sensitivity']:.4f}, specificity={best['specificity']:.4f}.",
    ]
    if "final_activation_none_ch1_0_2" in by_id.index and "current_ch1_0_2" in by_id.index:
        cand = by_id.loc["final_activation_none_ch1_0_2"]
        cur = by_id.loc["current_ch1_0_2"]
        auc_ok = bool(cand["auc"] >= 0.794)
        auc_preferred = bool(cand["auc"] >= 0.800)
        pr_ok = bool(cand["pr_auc"] >= cur["pr_auc"])
        ba_ok = bool(cand["balanced_accuracy"] >= cur["balanced_accuracy"] - 0.01)
        f1_ok = bool(cand["f1"] >= cur["f1"] - 0.01)
        subgroup_fragility = False
        fragility_notes: List[str] = []
        if not manufacturer.empty:
            cand_m = manufacturer[manufacturer["run_id"].eq("final_activation_none_ch1_0_2")]
            cur_m = manufacturer[manufacturer["run_id"].eq("current_ch1_0_2")]
            if not cand_m.empty and not cur_m.empty:
                merged_m = cand_m.merge(cur_m, on="Manufacturer", suffixes=("_candidate", "_current"))
                for _, row in merged_m.iterrows():
                    ba_drop = float(row["balanced_accuracy_candidate"]) - float(row["balanced_accuracy_current"])
                    sens_drop = float(row["sensitivity_candidate"]) - float(row["sensitivity_current"])
                    spec_drop = float(row["specificity_candidate"]) - float(row["specificity_current"])
                    if ba_drop < -0.05 or sens_drop < -0.10 or spec_drop < -0.10:
                        subgroup_fragility = True
                        fragility_notes.append(
                            f"Manufacturer {row['Manufacturer']}: BA {ba_drop:+.3f}, sens {sens_drop:+.3f}, spec {spec_drop:+.3f}"
                        )
        if not sex.empty:
            cand_s = sex[sex["run_id"].eq("final_activation_none_ch1_0_2")]
            cur_s = sex[sex["run_id"].eq("current_ch1_0_2")]
            if not cand_s.empty and not cur_s.empty:
                merged_s = cand_s.merge(cur_s, on="Sex", suffixes=("_candidate", "_current"))
                for _, row in merged_s.iterrows():
                    ba_drop = float(row["balanced_accuracy_candidate"]) - float(row["balanced_accuracy_current"])
                    if ba_drop < -0.05:
                        subgroup_fragility = True
                        fragility_notes.append(f"Sex {row['Sex']}: BA {ba_drop:+.3f}")
        foldwise_cand = foldwise[foldwise["run_id"].eq("final_activation_none_ch1_0_2")].copy()
        fold4_driver = False
        if not foldwise_cand.empty and "auc" in foldwise_cand.columns:
            fold4_auc = foldwise_cand.loc[foldwise_cand["fold"].eq(4), "auc"]
            non_fold4_auc = foldwise_cand.loc[~foldwise_cand["fold"].eq(4), "auc"]
            if not fold4_auc.empty and not non_fold4_auc.empty:
                fold4_driver = bool(float(fold4_auc.iloc[0]) - float(non_fold4_auc.mean()) > 0.04)
        decision_ok = auc_ok and pr_ok and ba_ok and f1_ok and not subgroup_fragility and not fold4_driver
        lines.append(
            "- final_activation=none [1,0,2] vs current tanh [1,0,2]: "
            f"delta AUC={cand['auc'] - cur['auc']:+.4f}, "
            f"delta PR-AUC={cand['pr_auc'] - cur['pr_auc']:+.4f}, "
            f"delta BA={cand['balanced_accuracy'] - cur['balanced_accuracy']:+.4f}, "
            f"delta F1={cand['f1'] - cur['f1']:+.4f}, "
            f"delta sensitivity={cand['sensitivity'] - cur['sensitivity']:+.4f}, "
            f"delta specificity={cand['specificity'] - cur['specificity']:+.4f}."
        )
        lines.append(
            "- Revised-main decision rule for final_activation=none: "
            f"AUC>=0.794: {auc_ok}; AUC>=0.800 preferred: {auc_preferred}; "
            f"PR-AUC nondecreasing: {pr_ok}; BA not materially worse: {ba_ok}; "
            f"F1 not materially worse: {f1_ok}; subgroup fragility: {subgroup_fragility}; "
            f"fold-4-only improvement driver: {fold4_driver}. "
            f"Recommendation: {'promote final_activation=none as revised main model' if decision_ok else 'do not promote final_activation=none as revised main model'}."
        )
        if fragility_notes:
            lines.append("- Subgroup fragility notes: " + " | ".join(fragility_notes))
    if "objective_v2_ch1_0_2" in by_id.index and "current_ch1_0_2" in by_id.index:
        obj = by_id.loc["objective_v2_ch1_0_2"]
        cur = by_id.loc["current_ch1_0_2"]
        lines.append(
            "- Objective-v2 [1,0,2] remains a comparator: "
            f"delta AUC={obj['auc'] - cur['auc']:+.4f}, "
            f"delta PR-AUC={obj['pr_auc'] - cur['pr_auc']:+.4f}, "
            f"delta BA={obj['balanced_accuracy'] - cur['balanced_accuracy']:+.4f}."
        )
    if "ch1" in set(main["run_id"]):
        ch1 = main.set_index("run_id").loc["ch1"]
        lines.append(
            f"- Single-channel [1] remains the parsimonious comparator: AUC={ch1['auc']:.4f}, "
            f"PR-AUC={ch1['pr_auc']:.4f}, BA={ch1['balanced_accuracy']:.4f}."
        )
    if "ch1_4" in set(main["run_id"]):
        ch14 = main.set_index("run_id").loc["ch1_4"]
        lines.append(
            f"- Pair [1,4] remains secondary/exploratory: AUC={ch14['auc']:.4f}, "
            f"PR-AUC={ch14['pr_auc']:.4f}, BA={ch14['balanced_accuracy']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Threshold Interpretation",
            "",
            "- The fixed 0.5 threshold is included for calibration/operating-point context.",
            "- The manuscript primary operating point should be the leakage-safe inner-OOF target-sensitivity rule, not a threshold selected on outer test predictions.",
            "",
            "## Subgroup Interpretation",
            "",
        ]
    )
    if not manufacturer.empty:
        weak_mfr = manufacturer.sort_values("balanced_accuracy", ascending=True).head(6)
        lines.append("Lowest manufacturer subgroup rows by balanced accuracy:")
        lines.append("")
        lines.append(md_table(weak_mfr[["run_id", "channels", "Manufacturer", "n", "n_cn", "n_ad", "sensitivity", "specificity", "balanced_accuracy", "auc"]]))
    if not sex.empty:
        lines.append("")
        lines.append("Sex subgroup rows:")
        lines.append("")
        lines.append(md_table(sex[["run_id", "channels", "Sex", "n", "n_cn", "n_ad", "sensitivity", "specificity", "balanced_accuracy", "auc"]]))
    lines.extend(
        [
            "",
            "## Fold 4",
            "",
        ]
    )
    if fold4_summary.empty:
        lines.append("- No Fold 4 errors were found in the primary operating point table.")
    else:
        lines.append("Fold 4 error counts by model/manufacturer:")
        lines.append("")
        lines.append(md_table(fold4_summary))
    lines.extend(
        [
            "",
            "## Manuscript-Safe Claim",
            "",
            "- Report this as an internal ADNI cross-validation comparison among pre-specified/full-candidate models.",
            "- State that all rows use the same manufacturer-aware 5x5 split design and the same leakage-safe classifier-only logreg_l2 threshold rule.",
            "- Do not claim external validation from these tables.",
            "- Do not use fixed 0.5 as the primary clinical operating point unless separately justified by calibration requirements.",
        ]
    )
    (output_dir / "manuscript_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    logs: Dict[str, Dict[str, Any]] = {}
    runs = [
        {
            "run_id": "final_activation_none_ch1_0_2",
            "label": "final_activation=none [1,0,2]",
            "channels": "[1,0,2]",
            "objective": "mse_sum_batchmean_current",
            "readout_dir": resolve(args.candidate_readout_dir),
            "run_dir": resolve(args.candidate_run_dir),
            "notes": "pre-registered FULL 5x5 confirmation candidate with vae_final_activation=none",
        },
        *RUNS,
    ]

    availability_rows = []
    for run in runs:
        readout_dir = resolve(run["readout_dir"])
        available = all(p.exists() for p in required_readout_files(readout_dir))
        availability_rows.append({"run_id": run["run_id"], "label": run["label"], "readout_dir": str(readout_dir), "available": available})
        if available:
            logs[run["run_id"]] = validate_readout(run)
    availability = pd.DataFrame(availability_rows)
    print("=== FULL 5x5 candidate readout availability ===")
    print(availability.to_string(index=False))
    missing = availability[~availability["available"]]
    if not missing.empty:
        raise RuntimeError("Missing required readouts:\n" + missing.to_string(index=False))
    if args.dry_run:
        print("Dry-run complete. No outputs written.")
        return 0

    prepare_output_dir(output_dir, overwrite=args.overwrite, dry_run=False)
    main = pd.DataFrame([primary_pooled_row(run) for run in runs]).sort_values("auc", ascending=False).reset_index(drop=True)
    thresholds = pd.concat([threshold_rows(run) for run in runs], ignore_index=True)
    foldwise = pd.concat([foldwise_rows(run) for run in runs], ignore_index=True)
    predictions = pd.concat([prediction_rows(run) for run in runs], ignore_index=True)
    manufacturer = subgroup_table(predictions, "Manufacturer")
    sex = subgroup_table(predictions, "Sex")
    fold4 = fold4_deep_dive(predictions)
    fold4_summary = fold4_error_summary(fold4)
    reconstruction_qc = collect_reconstruction_qc(runs)
    rate_distortion_qc = collect_rate_distortion_qc(runs)
    latent_qc = collect_latent_qc(runs)
    scanner_leakage_qc = collect_scanner_leakage_qc(runs)
    availability.to_csv(output_dir / "readout_availability.csv", index=False)
    write_outputs(
        output_dir,
        main,
        foldwise,
        manufacturer,
        sex,
        thresholds,
        fold4,
        fold4_summary,
        reconstruction_qc,
        rate_distortion_qc,
        latent_qc,
        scanner_leakage_qc,
        logs,
    )
    print(f"Wrote audit outputs to {output_dir}")
    print(main[["run_id", "channels", "objective", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
