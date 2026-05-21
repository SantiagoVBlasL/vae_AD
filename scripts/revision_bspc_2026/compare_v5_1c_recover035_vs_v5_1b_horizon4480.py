#!/usr/bin/env python3
"""Compare v5.1c recover035 FULL run against v5.1b horizon4480.

Read-only comparison. It expects Stage B classifier-only logreg_l2 outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_CANDIDATE_RUN = RESULTS / "adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_REFERENCE_READOUT = DEFAULT_REFERENCE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "adni_v5_1c_recover035_vs_v5_1b_horizon4480_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT)
    parser.add_argument("--reference-run-dir", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--reference-readout-dir", type=Path, default=DEFAULT_REFERENCE_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def required_readout_files(readout: Path) -> List[Path]:
    return [
        readout / "classifier_sweep_pooled_metrics.csv",
        readout / "classifier_sweep_foldwise_metrics.csv",
        readout / "classifier_sweep_predictions.csv",
        readout / "classifier_sweep_thresholds_by_fold.csv",
        readout / "classifier_sweep_subgroup_metrics_by_manufacturer.csv",
        readout / "command_log.json",
    ]


def validate_readout(label: str, readout: Path) -> None:
    missing = [str(p) for p in required_readout_files(readout) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing readout files:\n" + "\n".join(missing))
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if not mask.any():
        raise RuntimeError(f"{label} lacks {PRIMARY_MODEL}/{PRIMARY_THRESHOLD}")


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    (outdir / f"{stem}.md").write_text(view.to_markdown(index=False) + "\n")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def primary_pooled(run_id: str, label: str, readout: Path) -> Dict[str, Any]:
    df = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out = {"run_id": run_id, "label": label}
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]:
        out[col] = row.get(col, np.nan)
    return out


def read_predictions(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    if "SiteCode" not in df.columns and "SubjectID" in df.columns:
        df["SiteCode"] = df["SubjectID"].astype(str).str.extract(r"^(\d{3})_", expand=False).fillna("")
    return df


def foldwise(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def threshold_comparison(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def subgroup(pred: pd.DataFrame, group_col: str, min_n: int = 1) -> pd.DataFrame:
    rows = []
    if group_col not in pred.columns:
        return pd.DataFrame()
    for keys, sub in pred.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group = keys
        if len(sub) < min_n:
            continue
        row = {"run_id": run_id, "label": label, group_col: group}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def scanner_leakage(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("fold_*/fold_*_scanner_leakage_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["label"] = label
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    return raw


def latent_info(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("fold_*/latent_qc_metrics.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["label"] = label
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def add_deltas(main: pd.DataFrame) -> pd.DataFrame:
    ref = main[main["run_id"].eq("v5_1b_horizon4480")]
    if ref.empty:
        return main
    ref_row = ref.iloc[0]
    out = main.copy()
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]:
        out[f"delta_vs_v5_1b_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
    return out


def write_recommendation(outdir: Path, main: pd.DataFrame) -> None:
    cand = main[main["run_id"].eq("v5_1c_recover035")].iloc[0]
    ref = main[main["run_id"].eq("v5_1b_horizon4480")].iloc[0]
    auc_delta = float(cand["auc"]) - float(ref["auc"])
    pr_delta = float(cand["pr_auc"]) - float(ref["pr_auc"])
    if auc_delta > 0 and pr_delta >= 0:
        decision = "promote"
    elif abs(auc_delta) <= 0.005 and abs(pr_delta) <= 0.01:
        decision = "sensitivity-only"
    else:
        decision = "reject"
    text = f"""# v5.1c Recover035 vs v5.1b Horizon4480 Recommendation

Primary readout: `{PRIMARY_MODEL}` with `{PRIMARY_THRESHOLD}`.

- v5.1b AUC={float(ref['auc']):.6f}, PR-AUC={float(ref['pr_auc']):.6f}, BA={float(ref['balanced_accuracy']):.6f}, F1={float(ref['f1']):.6f}
- v5.1c AUC={float(cand['auc']):.6f}, PR-AUC={float(cand['pr_auc']):.6f}, BA={float(cand['balanced_accuracy']):.6f}, F1={float(cand['f1']):.6f}
- Delta AUC={auc_delta:+.6f}; Delta PR-AUC={pr_delta:+.6f}

Final recommendation: **{decision}**.

Interpretation should consider that v5.1c changes the cohort by adding one recoverable AD subject (`035_S_6927`) and removing one unresolved tensor-only subject (`128_S_2002`) from the clean tensor branch.
"""
    (outdir / "final_recommendation.md").write_text(text)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry-run comparison preflight.")
        print(f"Candidate run    : {args.candidate_run_dir}")
        print(f"Candidate readout: {args.candidate_readout_dir}")
        print(f"Reference run    : {args.reference_run_dir}")
        print(f"Reference readout: {args.reference_readout_dir}")
        print(f"Output dir       : {args.output_dir}")
        print("No files were written.")
        return

    validate_readout("candidate", args.candidate_readout_dir)
    validate_readout("reference", args.reference_readout_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main = pd.DataFrame(
        [
            primary_pooled("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", args.reference_readout_dir),
            primary_pooled("v5_1c_recover035", "v5.1c recover035 horizon4480/cycles56", args.candidate_readout_dir),
        ]
    )
    main = add_deltas(main)
    write_table(args.output_dir, "main_model_comparison", main)

    fold = pd.concat(
        [
            foldwise("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", args.reference_readout_dir),
            foldwise("v5_1c_recover035", "v5.1c recover035 horizon4480/cycles56", args.candidate_readout_dir),
        ],
        ignore_index=True,
    )
    write_table(args.output_dir, "foldwise_metrics", fold)

    thresholds = pd.concat(
        [
            threshold_comparison("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", args.reference_readout_dir),
            threshold_comparison("v5_1c_recover035", "v5.1c recover035 horizon4480/cycles56", args.candidate_readout_dir),
        ],
        ignore_index=True,
    )
    write_table(args.output_dir, "threshold_comparison", thresholds)

    preds = pd.concat(
        [
            read_predictions("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", args.reference_readout_dir),
            read_predictions("v5_1c_recover035", "v5.1c recover035 horizon4480/cycles56", args.candidate_readout_dir),
        ],
        ignore_index=True,
    )
    write_table(args.output_dir, "confusion_matrix", main[["run_id", "label", "tn", "fp", "fn", "tp"]])
    write_table(args.output_dir, "manufacturer_subgroup_metrics", subgroup(preds, "Manufacturer"))
    write_table(args.output_dir, "sitecode_subgroup_metrics", subgroup(preds, "SiteCode", min_n=1))
    site035 = preds[preds["SiteCode"].astype(str).eq("035")].copy()
    write_table(args.output_dir, "sitecode035_subjects_predictions", site035)

    leakage = pd.concat(
        [
            scanner_leakage("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", args.reference_run_dir),
            scanner_leakage("v5_1c_recover035", "v5.1c recover035 horizon4480/cycles56", args.candidate_run_dir),
        ],
        ignore_index=True,
    )
    write_table(args.output_dir, "scanner_manufacturer_leakage", leakage)

    latent = pd.concat(
        [
            latent_info("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", args.reference_run_dir),
            latent_info("v5_1c_recover035", "v5.1c recover035 horizon4480/cycles56", args.candidate_run_dir),
        ],
        ignore_index=True,
    )
    write_table(args.output_dir, "latent_information_qc", latent)

    write_recommendation(args.output_dir, main)
    command_log = {
        "script": rel(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": rel(args.candidate_run_dir),
        "candidate_readout": rel(args.candidate_readout_dir),
        "reference_run": rel(args.reference_run_dir),
        "reference_readout": rel(args.reference_readout_dir),
        "output_dir": rel(args.output_dir),
        "read_only": True,
        "modified_tensor_metadata_ledger_config_or_model_outputs": False,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2))
    print(json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
