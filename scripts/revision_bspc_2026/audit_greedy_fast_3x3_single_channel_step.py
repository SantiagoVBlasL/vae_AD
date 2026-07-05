#!/usr/bin/env python3
"""Audit completed greedy FAST 3x3 single-channel step.

This script only reads completed candidate outputs and rewrites lightweight
summary tables in the greedy FAST output root. It does not train, read tensors,
or modify metadata/ledger files.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_greedy_fast_channel_selection_3x3"
RUN_KEYS = [f"ch{i}" for i in range(7)]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
EXPECTED_INNER_CONTEXT = "ResearchGroup_Mapped+Manufacturer"
EXPECTED_CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]
METRIC_COLS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
READOUT_REQUIRED = [
    "classifier_sweep_foldwise_metrics.csv",
    "classifier_sweep_pooled_metrics.csv",
    "classifier_sweep_thresholds_by_fold.csv",
    "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
    "classifier_sweep_predictions.csv",
    "classifier_sweep_model_status.csv",
    "command_log.json",
]
FATAL_PATTERNS = [
    "traceback",
    "fatal",
    "failed_stage",
    "training failed",
    "readout failed",
    "runtimeerror",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_mean_se(values: pd.Series) -> Tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan"), float("nan")
    mean = float(numeric.mean())
    se = float(numeric.std(ddof=1) / math.sqrt(len(numeric))) if len(numeric) > 1 else 0.0
    return mean, se


def bool_all(values: Iterable[bool]) -> bool:
    return all(bool(v) for v in values)


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_text_safely(path: Path, max_chars: int = 1_000_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


def scan_fatal_patterns(run_dir: Path) -> Tuple[int, str]:
    hits: List[str] = []
    root = run_dir.resolve(strict=False)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".csv", ".txt", ".md", ".log"}:
            continue
        text = read_text_safely(path).lower()
        for pattern in FATAL_PATTERNS:
            if pattern in text:
                hits.append(f"{safe_rel(path)}::{pattern}")
                break
        if len(hits) >= 20:
            break
    return len(hits), " | ".join(hits[:10])


def assert_expected_config(run_key: str, run_config: Dict[str, Any], readout_log: Dict[str, Any]) -> List[str]:
    args = run_config.get("args", {})
    errors: List[str] = []
    expected_channel = int(run_key.replace("ch", ""))
    checks = {
        "outer_folds": 3,
        "inner_folds": 3,
        "latent_dim": 128,
        "epochs_vae": 960,
        "cyclical_beta_n_cycles": 12,
        "lr_scheduler_T0": 80,
        "n_iter_logreg": 1,
    }
    for key, expected in checks.items():
        if args.get(key) != expected:
            errors.append(f"Stage A {key}={args.get(key)} expected {expected}")
    if readout_log.get("outer_folds") != 3:
        errors.append(f"Stage B outer_folds={readout_log.get('outer_folds')} expected 3")
    if readout_log.get("inner_folds") != 3:
        errors.append(f"Stage B inner_folds={readout_log.get('inner_folds')} expected 3")
    if args.get("channels_to_use") != [expected_channel]:
        errors.append(f"channels_to_use={args.get('channels_to_use')} expected [{expected_channel}]")
    if args.get("classifier_types") != ["logreg"]:
        errors.append(f"Stage A classifier_types={args.get('classifier_types')} expected ['logreg']")
    if args.get("n_iter_svm") is not None:
        errors.append(f"Stage A n_iter_svm should be omitted/null, got {args.get('n_iter_svm')}")
    if "Manufacturer" not in args.get("classifier_stratify_cols", []):
        errors.append("classifier_stratify_cols missing Manufacturer")
    if "Manufacturer" not in args.get("vae_stratify_cols", []):
        errors.append("vae_stratify_cols missing Manufacturer")
    if "Sex" in args.get("classifier_stratify_cols", []):
        errors.append("Sex present in classifier_stratify_cols")
    if "Sex" in args.get("vae_stratify_cols", []):
        errors.append("Sex present in vae_stratify_cols")
    if args.get("metadata_features") != ["Age", "Sex"]:
        errors.append(f"metadata_features={args.get('metadata_features')} expected ['Age', 'Sex']")
    if readout_log.get("vae_retrained") is not False:
        errors.append("Stage B command_log should report vae_retrained=false")
    if readout_log.get("tensor_modified") is not False:
        errors.append("Stage B command_log should report tensor_modified=false")
    if readout_log.get("metadata_modified") is not False:
        errors.append("Stage B command_log should report metadata_modified=false")
    if readout_log.get("ledger_modified") is not False:
        errors.append("Stage B command_log should report ledger_modified=false")
    return errors


def critical_metric_nan_count(df: pd.DataFrame, cols: Sequence[str]) -> int:
    present = [col for col in cols if col in df.columns]
    if not present:
        return 0
    return int(df[present].isna().sum().sum())


def summarize_primary(run_key: str, channel_idx: int, channel_name: str, readout_dir: Path) -> Dict[str, Any]:
    foldwise = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    thresholds = pd.read_csv(readout_dir / "classifier_sweep_thresholds_by_fold.csv")

    primary = foldwise[
        foldwise["model_name"].eq(PRIMARY_MODEL)
        & foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    if len(primary) != 3:
        raise RuntimeError(f"{run_key}: expected 3 primary fold rows, found {len(primary)}")
    if set(primary["fold"].astype(int)) != {1, 2, 3}:
        raise RuntimeError(f"{run_key}: primary rows do not cover folds 1,2,3")
    if not primary["threshold_selection_context"].eq("true_inner_cv_oof").all():
        raise RuntimeError(f"{run_key}: primary threshold is not true inner-CV OOF")
    if not primary["inner_cv_context"].eq(EXPECTED_INNER_CONTEXT).all():
        raise RuntimeError(f"{run_key}: primary inner CV context is not {EXPECTED_INNER_CONTEXT}")
    if critical_metric_nan_count(primary, METRIC_COLS) > 0:
        raise RuntimeError(f"{run_key}: primary readout contains NaNs in critical metrics")

    row: Dict[str, Any] = {
        "run_key": run_key,
        "channel_index": int(channel_idx),
        "channels": json.dumps([int(channel_idx)], separators=(",", ":")),
        "n_channels": 1,
        "channel_name": channel_name,
        "selected_channel_names": channel_name,
        "run_dir": safe_rel(readout_dir.parent),
        "readout_dir": safe_rel(readout_dir),
        "readout_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "threshold_selection_verified": True,
    }
    for metric in METRIC_COLS:
        mean, se = metric_mean_se(primary[metric])
        row[f"mean_outer_{metric}"] = mean
        row[f"se_outer_{metric}"] = se
    pooled_primary = pooled[
        pooled["model_name"].eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    if len(pooled_primary) == 1:
        prow = pooled_primary.iloc[0]
        for metric in [*METRIC_COLS, "tn", "fp", "fn", "tp"]:
            if metric in pooled_primary.columns:
                row[f"pooled_{metric}"] = float(prow[metric])
    target_thresholds = thresholds[
        thresholds["model_name"].eq(PRIMARY_MODEL)
        & thresholds["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    row["mean_selected_threshold"] = float(pd.to_numeric(target_thresholds["threshold"], errors="coerce").mean())
    row["min_inner_stratum_count"] = int(pd.to_numeric(primary["minimum_inner_stratum_count"], errors="coerce").min())
    return row


def summarize_exploratory_best(readout_dir: Path) -> Dict[str, Any]:
    foldwise = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    candidates = foldwise[
        ~(
            foldwise["model_name"].eq(PRIMARY_MODEL)
            & foldwise["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        )
    ].copy()
    rows: List[Dict[str, Any]] = []
    for (model, threshold), group in candidates.groupby(["model_name", "threshold_strategy"], dropna=False):
        if set(group["fold"].astype(int)) != {1, 2, 3}:
            continue
        item: Dict[str, Any] = {
            "exploratory_best_model": str(model),
            "exploratory_best_threshold_strategy": str(threshold),
        }
        for metric in METRIC_COLS:
            mean, _ = metric_mean_se(group[metric])
            item[f"exploratory_best_mean_outer_{metric}"] = mean
        rows.append(item)
    if not rows:
        return {
            "exploratory_best_model": "",
            "exploratory_best_threshold_strategy": "",
            **{f"exploratory_best_mean_outer_{metric}": np.nan for metric in METRIC_COLS},
        }
    ranked = pd.DataFrame(rows).sort_values(
        ["exploratory_best_mean_outer_auc", "exploratory_best_mean_outer_pr_auc", "exploratory_best_mean_outer_balanced_accuracy"],
        ascending=[False, False, False],
    )
    return ranked.iloc[0].to_dict()


def audit_run(run_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    channel_idx = int(run_key.replace("ch", ""))
    channel_name = EXPECTED_CHANNEL_NAMES[channel_idx]
    run_dir = ROOT / "runs" / run_key
    readout_dir = run_dir / "classifier_only_readout"
    run_config_path = run_dir / "run_config.json"
    readout_log_path = readout_dir / "command_log.json"
    status_path = run_dir / ".greedy_candidate_status.json"

    stage_a_required = [
        run_config_path,
        *list(run_dir.glob("summary_metrics_MULTI_*.txt")),
        *list(run_dir.glob("all_folds_metrics_MULTI_*.csv")),
    ]
    for fold in [1, 2, 3]:
        fold_dir = run_dir / f"fold_{fold}"
        stage_a_required.extend(
            [
                fold_dir / f"vae_model_fold_{fold}.pt",
                fold_dir / "test_predictions_logreg.csv",
                fold_dir / "latent_qc_metrics.csv",
                fold_dir / f"fold_{fold}_test_latent_info_summary.csv",
                fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv",
            ]
        )
    stage_b_required = [readout_dir / name for name in READOUT_REQUIRED]

    stage_a_complete = bool_all(path.exists() for path in stage_a_required)
    stage_b_complete = bool_all(path.exists() for path in stage_b_required)
    if not stage_a_complete or not stage_b_complete:
        missing = [safe_rel(path) for path in [*stage_a_required, *stage_b_required] if not path.exists()]
        raise RuntimeError(f"{run_key}: incomplete required artifacts:\n" + "\n".join(missing))

    run_config = load_json(run_config_path)
    readout_log = load_json(readout_log_path)
    status = load_json(status_path) if status_path.exists() else {}
    config_errors = assert_expected_config(run_key, run_config, readout_log)

    model_status = pd.read_csv(readout_dir / "classifier_sweep_model_status.csv")
    non_fit_ok = model_status[~model_status["status"].eq("fit_ok")]
    foldwise = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    subgroup = pd.read_csv(readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv")
    critical_nan_count = (
        critical_metric_nan_count(foldwise, METRIC_COLS)
        + critical_metric_nan_count(pooled, METRIC_COLS)
        + critical_metric_nan_count(subgroup, METRIC_COLS)
    )
    fatal_hit_count, fatal_hit_examples = scan_fatal_patterns(run_dir)

    completion = {
        "run_key": run_key,
        "channel_index": channel_idx,
        "channel_name": channel_name,
        "stage_a_complete": stage_a_complete,
        "stage_b_complete": stage_b_complete,
        "candidate_status": status.get("status", ""),
        "candidate_status_stage": status.get("stage", ""),
        "candidate_status_readout_complete": status.get("classifier_only_readout_complete", ""),
        "readout_required_files_present": stage_b_complete,
        "readout_model_status_all_fit_ok": non_fit_ok.empty,
        "non_fit_ok_model_status_rows": int(len(non_fit_ok)),
        "fatal_pattern_hits": int(fatal_hit_count),
        "fatal_pattern_examples": fatal_hit_examples,
        "critical_metric_nan_count": int(critical_nan_count),
        "config_valid": not config_errors,
        "config_errors": " | ".join(config_errors),
        "outer_folds_stage_a": run_config.get("args", {}).get("outer_folds"),
        "inner_folds_stage_a": run_config.get("args", {}).get("inner_folds"),
        "outer_folds_stage_b": readout_log.get("outer_folds"),
        "inner_folds_stage_b": readout_log.get("inner_folds"),
        "latent_dim": run_config.get("args", {}).get("latent_dim"),
        "epochs_vae": run_config.get("args", {}).get("epochs_vae"),
        "cyclical_beta_n_cycles": run_config.get("args", {}).get("cyclical_beta_n_cycles"),
        "lr_scheduler_T0": run_config.get("args", {}).get("lr_scheduler_T0"),
        "classifier_stratify_cols": ",".join(run_config.get("args", {}).get("classifier_stratify_cols", [])),
        "vae_stratify_cols": ",".join(run_config.get("args", {}).get("vae_stratify_cols", [])),
        "metadata_features": ",".join(run_config.get("args", {}).get("metadata_features", [])),
        "sex_only_covariate": (
            "Sex" in run_config.get("args", {}).get("metadata_features", [])
            and "Sex" not in run_config.get("args", {}).get("classifier_stratify_cols", [])
            and "Sex" not in run_config.get("args", {}).get("vae_stratify_cols", [])
        ),
        "stage_a_classifier": ",".join(run_config.get("args", {}).get("classifier_types", [])),
        "stage_a_n_iter_logreg": run_config.get("args", {}).get("n_iter_logreg"),
        "stage_a_n_iter_svm": run_config.get("args", {}).get("n_iter_svm"),
        "vae_retrained_in_stage_b": readout_log.get("vae_retrained"),
        "tensor_modified_in_stage_b": readout_log.get("tensor_modified"),
        "metadata_modified_in_stage_b": readout_log.get("metadata_modified"),
        "ledger_modified_in_stage_b": readout_log.get("ledger_modified"),
    }
    metrics = summarize_primary(run_key, channel_idx, channel_name, readout_dir)
    metrics.update(summarize_exploratory_best(readout_dir))
    completion_verified = (
        completion["stage_a_complete"]
        and completion["stage_b_complete"]
        and completion["readout_model_status_all_fit_ok"]
        and completion["fatal_pattern_hits"] == 0
        and completion["critical_metric_nan_count"] == 0
        and completion["config_valid"]
    )
    completion["completion_verified"] = completion_verified
    metrics["completion_verified"] = completion_verified
    return completion, metrics


def write_csvs(completion_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    completion_df.to_csv(ROOT / "single_channel_completion_audit.csv", index=False)
    metrics_df = metrics_df.sort_values(
        ["mean_outer_auc", "mean_outer_pr_auc", "mean_outer_balanced_accuracy", "channel_index"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    metrics_df.to_csv(ROOT / "channel_set_metrics.csv", index=False)

    ranking = metrics_df.copy()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    keep_cols = [
        "rank",
        "run_key",
        "channel_index",
        "channel_name",
        "mean_outer_auc",
        "mean_outer_pr_auc",
        "mean_outer_balanced_accuracy",
        "mean_outer_sensitivity",
        "mean_outer_specificity",
        "mean_outer_f1",
        "se_outer_auc",
        "readout_model",
        "primary_threshold_strategy",
        "exploratory_best_model",
        "exploratory_best_threshold_strategy",
        "exploratory_best_mean_outer_auc",
        "exploratory_best_mean_outer_pr_auc",
        "exploratory_best_mean_outer_balanced_accuracy",
        "exploratory_best_mean_outer_sensitivity",
        "exploratory_best_mean_outer_specificity",
        "exploratory_best_mean_outer_f1",
        "completion_verified",
    ]
    ranking[keep_cols].to_csv(ROOT / "final_greedy_ranking.csv", index=False)

    best = ranking.iloc[0]
    trace = pd.DataFrame(
        [
            {
                "step": 1,
                "selected_run_key": best["run_key"],
                "selected_channels": json.dumps([int(best["channel_index"])], separators=(",", ":")),
                "selected_channel_names": best["channel_name"],
                "mean_outer_auc": best["mean_outer_auc"],
                "se_outer_auc": best["se_outer_auc"],
                "mean_outer_pr_auc": best["mean_outer_pr_auc"],
                "mean_outer_balanced_accuracy": best["mean_outer_balanced_accuracy"],
                "mean_outer_sensitivity": best["mean_outer_sensitivity"],
                "mean_outer_specificity": best["mean_outer_specificity"],
                "mean_outer_f1": best["mean_outer_f1"],
                "auc_delta_vs_previous": np.nan,
                "pr_auc_delta_vs_previous": np.nan,
                "balanced_accuracy_delta_vs_previous": np.nan,
                "stop_after_step": False,
                "stop_reason": "",
                "selection_basis": "best mean outer ROC-AUC among seven verified single-channel FAST 3x3 candidates",
            }
        ]
    )
    trace.to_csv(ROOT / "greedy_selection_trace.csv", index=False)

    completed = completion_df.copy()
    completed["status"] = np.where(completed["completion_verified"], "complete", "audit_failed")
    completed.to_csv(ROOT / "completed_runs.csv", index=False)

    state_path = ROOT / "greedy_state.json"
    state = load_json(state_path) if state_path.exists() else {}
    state.update(
        {
            "updated_utc": now_utc(),
            "status": "single_channel_step_complete_waiting_for_step_2",
            "selected_channels": [int(best["channel_index"])],
            "selected_channel_names": [str(best["channel_name"])],
            "last_completed_step": {
                "step": 1,
                "selected_run_key": str(best["run_key"]),
                "selection_basis": "mean_outer_auc",
                "mean_outer_auc": float(best["mean_outer_auc"]),
                "fast_outer_folds": 3,
                "fast_inner_folds": 3,
            },
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: Optional[int] = None) -> str:
    tmp = df.loc[:, list(cols)].copy()
    if n is not None:
        tmp = tmp.head(n)
    for col in tmp.columns:
        if pd.api.types.is_numeric_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return tmp.to_markdown(index=False)


def write_readme(completion_df: pd.DataFrame, ranking: pd.DataFrame) -> None:
    best = ranking.iloc[0]
    all_complete = bool(completion_df["completion_verified"].all())
    fatal_hits = int(completion_df["fatal_pattern_hits"].sum())
    nan_hits = int(completion_df["critical_metric_nan_count"].sum())
    primary_cols = [
        "rank",
        "channel_index",
        "channel_name",
        "mean_outer_auc",
        "mean_outer_pr_auc",
        "mean_outer_balanced_accuracy",
        "mean_outer_sensitivity",
        "mean_outer_specificity",
        "mean_outer_f1",
    ]
    exploratory_cols = [
        "channel_index",
        "channel_name",
        "exploratory_best_model",
        "exploratory_best_threshold_strategy",
        "exploratory_best_mean_outer_auc",
        "exploratory_best_mean_outer_pr_auc",
        "exploratory_best_mean_outer_balanced_accuracy",
    ]
    lines = [
        "# Greedy FAST 3x3 Single-Channel Audit",
        "",
        f"Updated: `{now_utc()}`",
        "",
        "## Scope",
        "",
        "This audit covers the completed single-channel step of the greedy FAST channel-selection workflow.",
        "It reads existing Stage A/Stage B outputs only. No VAE training, classifier training, tensor modification, metadata modification, or ledger modification was performed by this audit.",
        "",
        "## Completion Check",
        "",
        f"- Candidates audited: `{len(completion_df)}` (`ch0` through `ch6`).",
        f"- Stage A complete for all seven: `{bool(completion_df['stage_a_complete'].all())}`.",
        f"- Stage B complete for all seven: `{bool(completion_df['stage_b_complete'].all())}`.",
        f"- Classifier-only readout model status all `fit_ok`: `{bool(completion_df['readout_model_status_all_fit_ok'].all())}`.",
        f"- Fatal pattern hits / Tracebacks: `{fatal_hits}`.",
        f"- Critical metric NaNs: `{nan_hits}`.",
        f"- Overall completion verified: `{all_complete}`.",
        "",
        "## Configuration Confirmed",
        "",
        "- `outer_folds=3`, `inner_folds=3` for Stage A and Stage B.",
        "- `latent_dim=128`, `epochs_vae=960`, `cyclical_beta_n_cycles=12`, `lr_scheduler_T0=80`.",
        "- Split context: `ResearchGroup_Mapped + Manufacturer`.",
        "- `Sex` is included only as a metadata/covariate feature with `Age`; it is not a stratification column.",
        "- Python bandpass: `OFF`.",
        "- Stage A canonical classifier is dummy `logreg` with `n_iter_logreg=1`; Stage A classifier outputs are ignored for ranking.",
        "- Ranking uses Stage B classifier-only readout: `logreg_l2 + inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "## Primary Ranking",
        "",
        markdown_table(ranking, primary_cols),
        "",
        "## Exploratory Secondary Best Readout",
        "",
        "These rows are exploratory only and are not the greedy selection criterion.",
        "",
        markdown_table(ranking, exploratory_cols),
        "",
        "## Greedy Decision",
        "",
        f"- Selected best single channel by mean outer ROC-AUC: `ch{int(best['channel_index'])}` / `{best['channel_name']}`.",
        f"- Mean outer AUC: `{float(best['mean_outer_auc']):.4f}`.",
        "- Next greedy step should evaluate this selected channel plus each remaining channel.",
        "",
        "## Files Updated",
        "",
        "- `single_channel_completion_audit.csv`",
        "- `channel_set_metrics.csv`",
        "- `final_greedy_ranking.csv`",
        "- `greedy_selection_trace.csv`",
        "- `completed_runs.csv`",
        "- `greedy_state.json`",
        "- `README.md`",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not ROOT.exists():
        raise FileNotFoundError(ROOT)
    completions: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []
    for run_key in RUN_KEYS:
        completion, metric = audit_run(run_key)
        completions.append(completion)
        metrics.append(metric)
    completion_df = pd.DataFrame(completions).sort_values("channel_index").reset_index(drop=True)
    metrics_df = pd.DataFrame(metrics)
    if not completion_df["config_valid"].all():
        raise RuntimeError("Config validation failed:\n" + completion_df[~completion_df["config_valid"]].to_string(index=False))
    if not completion_df["stage_a_complete"].all() or not completion_df["stage_b_complete"].all():
        raise RuntimeError("One or more candidates are incomplete.")
    if not completion_df["readout_model_status_all_fit_ok"].all():
        raise RuntimeError("One or more classifier-only model statuses are not fit_ok.")
    if int(completion_df["fatal_pattern_hits"].sum()) > 0:
        raise RuntimeError("Fatal pattern hits found:\n" + completion_df[completion_df["fatal_pattern_hits"].gt(0)].to_string(index=False))
    if int(completion_df["critical_metric_nan_count"].sum()) > 0:
        raise RuntimeError("Critical metric NaNs found.")
    write_csvs(completion_df, metrics_df)
    ranking = pd.read_csv(ROOT / "final_greedy_ranking.csv")
    write_readme(completion_df, ranking)
    print("Greedy FAST 3x3 single-channel audit complete.")
    print(f"Output root: {ROOT}")
    print(f"Candidates verified: {len(completion_df)}")
    print(f"Best single channel: ch{int(ranking.iloc[0]['channel_index'])} {ranking.iloc[0]['channel_name']}")
    print(f"Best primary mean outer AUC: {float(ranking.iloc[0]['mean_outer_auc']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
