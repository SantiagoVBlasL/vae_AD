#!/usr/bin/env python3
"""Compare FULL 5x5 [1,4] against FULL [1] and FULL [1,0,2].

This is read-only. It expects classifier-only readout outputs and never trains
models or modifies tensors, metadata, or ledger files.
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
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"

DEFAULT_CANDIDATE_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate"
DEFAULT_CANDIDATE_READOUT_DIR = DEFAULT_CANDIDATE_RUN_DIR / "classifier_only_readout"
DEFAULT_CH1_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_full_5x5_candidate"
DEFAULT_CH1_READOUT_DIR = DEFAULT_CH1_RUN_DIR / "classifier_only_readout"
DEFAULT_CH102_RUN_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
DEFAULT_CH102_READOUT_DIR = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
DEFAULT_OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_4_full_5x5_candidate_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN_DIR)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT_DIR)
    parser.add_argument("--ch1-run-dir", type=Path, default=DEFAULT_CH1_RUN_DIR)
    parser.add_argument("--ch1-readout-dir", type=Path, default=DEFAULT_CH1_READOUT_DIR)
    parser.add_argument("--ch102-run-dir", type=Path, default=DEFAULT_CH102_RUN_DIR)
    parser.add_argument("--ch102-readout-dir", type=Path, default=DEFAULT_CH102_READOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def available_readout(readout_dir: Path) -> bool:
    required = [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_predictions.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "command_log.json",
    ]
    return all((readout_dir / name).exists() for name in required)


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


def primary_row(readout_dir: Path, run_id: str, label: str, channels: str) -> Dict[str, Any]:
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    row = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if len(row) != 1:
        raise RuntimeError(f"Expected one primary pooled row for {run_id}, found {len(row)}")
    r = row.iloc[0].to_dict()
    out: Dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "channels": channels,
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", *PRIMARY_METRICS]:
        out[col] = r.get(col, np.nan)
    return out


def primary_foldwise(readout_dir: Path, run_id: str, channels: str) -> pd.DataFrame:
    df = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    df = df[(df["model_name"].eq(PRIMARY_MODEL)) & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "channels", channels)
    keep = [
        "run_id",
        "channels",
        "fold",
        "threshold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
        "threshold_selection_context",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values(["fold"]).reset_index(drop=True)


def group_metrics(readout_dir: Path, run_id: str, channels: str, group_col: str) -> pd.DataFrame:
    pred = pd.read_csv(readout_dir / "classifier_sweep_predictions.csv")
    pred = pred[(pred["model_name"].eq(PRIMARY_MODEL)) & (pred["threshold_strategy"].eq(PRIMARY_THRESHOLD))].copy()
    rows: List[Dict[str, Any]] = []
    for group_value, sub in pred.groupby(group_col, dropna=False):
        y = sub["y_true"].astype(int).to_numpy()
        score = sub["y_score"].astype(float).to_numpy()
        y_pred = sub["y_pred"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        rows.append(
            {
                "run_id": run_id,
                "channels": channels,
                "grouping": group_col,
                "group_value": group_value,
                "n": int(len(sub)),
                "n_cn": int((y == 0).sum()),
                "n_ad": int((y == 1).sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": (tn + tp) / len(sub) if len(sub) else np.nan,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": np.nanmean([sensitivity, specificity]),
                "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                "auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
                "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["group_value", "run_id"]).reset_index(drop=True)


def validate_readout_log(
    readout_dir: Path,
    expected_available: bool,
    *,
    run_id: str,
    require_single_model: bool,
) -> Dict[str, Any]:
    if not expected_available:
        return {}
    log = read_json_optional(readout_dir / "command_log.json")
    requested = log.get("classifiers_requested")
    if require_single_model:
        if requested != [PRIMARY_MODEL]:
            raise RuntimeError(f"{readout_dir} was not restricted to {PRIMARY_MODEL}: {requested}")
    elif isinstance(requested, list) and PRIMARY_MODEL not in requested:
        raise RuntimeError(f"{readout_dir} multi-model sweep does not include {PRIMARY_MODEL}: {requested}")
    pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    primary = pooled[(pooled["model_name"].eq(PRIMARY_MODEL)) & (pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD))]
    if len(primary) != 1:
        raise RuntimeError(
            f"Expected exactly one {PRIMARY_MODEL}/{PRIMARY_THRESHOLD} row for {run_id}; found {len(primary)}"
        )
    if not str(log.get("threshold_selection", "")).startswith("true_inner_cv_oof"):
        raise RuntimeError(f"{readout_dir} does not report true inner-CV OOF threshold selection")
    for key in ["tensor_modified", "metadata_modified", "ledger_modified"]:
        if log.get(key) is not False:
            raise RuntimeError(f"{readout_dir} command log has {key}={log.get(key)}")
    return log


def run_config_summary(run_dir: Path, run_id: str) -> Dict[str, Any]:
    cfg = read_json_optional(run_dir / "run_config.json") or read_json_optional(run_dir / "run_manifest.json")
    args = cfg.get("args", {})
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "channels_to_use": args.get("channels_to_use") or cfg.get("selected_channels") or cfg.get("channels_to_use_indices"),
        "outer_folds": args.get("outer_folds") or cfg.get("outer_folds"),
        "inner_folds": args.get("inner_folds") or cfg.get("inner_folds"),
        "latent_dim": args.get("latent_dim"),
        "epochs_vae": args.get("epochs_vae") or cfg.get("epochs_vae"),
        "cyclical_beta_n_cycles": args.get("cyclical_beta_n_cycles") or cfg.get("cyclical_beta_n_cycles"),
        "lr_scheduler_T0": args.get("lr_scheduler_T0") or cfg.get("lr_scheduler_T0"),
        "classifier_stratify_cols": args.get("classifier_stratify_cols"),
        "vae_stratify_cols": args.get("vae_stratify_cols"),
        "metadata_features": args.get("metadata_features"),
        "python_bandpass_applied": cfg.get("python_bandpass_applied", False),
    }


def write_interpretation(output_dir: Path, main: pd.DataFrame, weak: pd.DataFrame, availability: pd.DataFrame) -> None:
    underperformance_note = "- `[1,4]` completed successfully."
    if {"ch1_4_full_5x5", "ch1_full_5x5", "ch1_0_2_full_5x5"}.issubset(set(main["run_id"])):
        by_id = main.set_index("run_id")
        candidate = by_id.loc["ch1_4_full_5x5"]
        ch1 = by_id.loc["ch1_full_5x5"]
        ch102 = by_id.loc["ch1_0_2_full_5x5"]
        lower_than_both = [
            metric
            for metric in PRIMARY_METRICS
            if float(candidate[metric]) < float(ch1[metric]) and float(candidate[metric]) < float(ch102[metric])
        ]
        if set(lower_than_both) == set(PRIMARY_METRICS):
            underperformance_note = (
                "- `[1,4]` completed successfully but underperformed both `[1]` and `[1,0,2]` "
                "across AUC, PR-AUC, balanced accuracy, sensitivity, specificity, and F1."
            )
        else:
            diffs = []
            for ref_id, ref_label in [("ch1_full_5x5", "[1]"), ("ch1_0_2_full_5x5", "[1,0,2]")]:
                ref = by_id.loc[ref_id]
                diffs.append(
                    f"{ref_label}: delta AUC={float(candidate['auc']) - float(ref['auc']):+.4f}, "
                    f"delta PR-AUC={float(candidate['pr_auc']) - float(ref['pr_auc']):+.4f}, "
                    f"delta BA={float(candidate['balanced_accuracy']) - float(ref['balanced_accuracy']):+.4f}"
                )
            underperformance_note = (
                "- `[1,4]` completed successfully and is below both reference runs on the primary ranking metrics "
                f"({'; '.join(diffs)})."
            )
    lines = [
        "# FULL 5x5 [1,4] Secondary Candidate Comparison",
        "",
        "This comparison is read-only and uses classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "For the `[1,0,2]` reference, the classifier-only directory is a multi-model sweep; this report filters it to `model_name=logreg_l2` and the primary threshold strategy before comparing.",
        "",
        "## Availability",
        "",
        md_table(availability),
        "",
        "## Primary Metrics",
        "",
        md_table(main[["run_id", "channels", *PRIMARY_METRICS]] if not main.empty else main),
        "",
        "## Weak Fold",
        "",
        md_table(weak[["run_id", "channels", "fold", *PRIMARY_METRICS]] if not weak.empty else weak),
        "",
        "## Notes",
        "",
        underperformance_note,
        "- `[1,4]` is a secondary confirmation run for the best FAST pair.",
        "- Stage A canonical classifier outputs are dummy/ignored; final ranking must use the classifier-only readout.",
        "- Python bandpass remains OFF.",
        "- Tensor, metadata, and ledger files are not modified by this comparison.",
    ]
    (output_dir / "interpretation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs = [
        {
            "run_id": "ch1_4_full_5x5",
            "label": "[1,4] full 5x5 candidate",
            "channels": "[1,4]",
            "run_dir": resolve(args.candidate_run_dir),
            "readout_dir": resolve(args.candidate_readout_dir),
            "role": "candidate",
        },
        {
            "run_id": "ch1_full_5x5",
            "label": "[1] full 5x5 candidate",
            "channels": "[1]",
            "run_dir": resolve(args.ch1_run_dir),
            "readout_dir": resolve(args.ch1_readout_dir),
            "role": "reference",
        },
        {
            "run_id": "ch1_0_2_full_5x5",
            "label": "[1,0,2] mfrsplit_3840 final candidate",
            "channels": "[1,0,2]",
            "run_dir": resolve(args.ch102_run_dir),
            "readout_dir": resolve(args.ch102_readout_dir),
            "role": "reference",
        },
    ]
    availability = pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "role": r["role"],
                "run_dir": str(r["run_dir"]),
                "readout_dir": str(r["readout_dir"]),
                "readout_available": available_readout(r["readout_dir"]),
            }
            for r in runs
        ]
    )
    print("=== FULL 5x5 [1,4] vs [1] vs [1,0,2] ===")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'WRITE OUTPUTS'}")
    print(availability.to_string(index=False))
    if args.dry_run:
        print("Dry-run complete. No outputs written.")
        return 0

    output_dir = resolve(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite, args.dry_run)
    for r in runs:
        validate_readout_log(
            r["readout_dir"],
            available_readout(r["readout_dir"]),
            run_id=r["run_id"],
            require_single_model=r["run_id"] != "ch1_0_2_full_5x5",
        )

    available_runs = [r for r in runs if available_readout(r["readout_dir"])]
    if not available_runs:
        raise RuntimeError("No readouts are available to compare.")

    main = pd.DataFrame([primary_row(r["readout_dir"], r["run_id"], r["label"], r["channels"]) for r in available_runs])
    foldwise = pd.concat([primary_foldwise(r["readout_dir"], r["run_id"], r["channels"]) for r in available_runs], ignore_index=True)
    manufacturer = pd.concat([group_metrics(r["readout_dir"], r["run_id"], r["channels"], "Manufacturer") for r in available_runs], ignore_index=True)
    sex = pd.concat([group_metrics(r["readout_dir"], r["run_id"], r["channels"], "Sex") for r in available_runs], ignore_index=True)
    weak = foldwise.sort_values("auc").groupby("run_id").head(1).reset_index(drop=True)
    configs = pd.DataFrame([run_config_summary(r["run_dir"], r["run_id"]) for r in runs])

    main.to_csv(output_dir / "main_model_comparison.csv", index=False)
    foldwise.to_csv(output_dir / "foldwise_comparison.csv", index=False)
    manufacturer.to_csv(output_dir / "manufacturer_subgroup_comparison.csv", index=False)
    sex.to_csv(output_dir / "sex_subgroup_comparison.csv", index=False)
    weak.to_csv(output_dir / "weak_fold_comparison.csv", index=False)
    configs.to_csv(output_dir / "run_config_summary.csv", index=False)
    availability.to_csv(output_dir / "availability.csv", index=False)

    (output_dir / "main_model_comparison.md").write_text(md_table(main), encoding="utf-8")
    (output_dir / "foldwise_comparison.md").write_text(md_table(foldwise), encoding="utf-8")
    (output_dir / "manufacturer_subgroup_comparison.md").write_text(md_table(manufacturer), encoding="utf-8")
    (output_dir / "sex_subgroup_comparison.md").write_text(md_table(sex), encoding="utf-8")
    write_interpretation(output_dir, main, weak, availability)

    serializable_runs = [
        {
            **r,
            "run_dir": str(r["run_dir"]),
            "readout_dir": str(r["readout_dir"]),
        }
        for r in runs
    ]
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "reference_ch1_0_2_readout_note": (
            "The [1,0,2] reference is a multi-model classifier-only sweep; comparison filters to "
            f"model_name={PRIMARY_MODEL} and threshold_strategy={PRIMARY_THRESHOLD}."
        ),
        "runs": serializable_runs,
        "output_dir": str(output_dir),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to: {output_dir}")
    print(main[["run_id", "channels", *PRIMARY_METRICS]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
