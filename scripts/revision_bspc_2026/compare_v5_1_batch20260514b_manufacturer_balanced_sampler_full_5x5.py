#!/usr/bin/env python3
"""Compare manufacturer-balanced sampler FULL 5x5 against locked current FULL.

Read-only comparison: uses classifier-only logreg_l2 at
inner_oof_target_sens_ge_0p70_max_spec as the primary operating point.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_CANDIDATE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5_comparison"
REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
REFERENCE_READOUT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT)
    parser.add_argument("--reference-run-dir", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--reference-readout-dir", type=Path, default=REFERENCE_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}" if np.isfinite(val) else "")
            elif pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def required_readout_files(readout: Path) -> List[Path]:
    return [
        readout / "classifier_sweep_pooled_metrics.csv",
        readout / "classifier_sweep_foldwise_metrics.csv",
        readout / "classifier_sweep_predictions.csv",
        readout / "classifier_sweep_thresholds_by_fold.csv",
        readout / "command_log.json",
    ]


def validate_readout(label: str, readout: Path, required: bool = True) -> bool:
    missing = [str(p) for p in required_readout_files(readout) if not p.exists()]
    if missing:
        if required:
            raise FileNotFoundError(f"{label} missing readout files:\n" + "\n".join(missing))
        return False
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if not mask.any():
        if required:
            raise RuntimeError(f"{label} readout lacks {PRIMARY_MODEL}/{PRIMARY_THRESHOLD}")
        return False
    return True


def primary_row(run_id: str, label: str, readout: Path) -> Dict[str, Any]:
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out: Dict[str, Any] = {"run_id": run_id, "label": label, "model_name": PRIMARY_MODEL, "threshold_strategy": PRIMARY_THRESHOLD}
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", *METRICS, "predicted_ad_rate"]:
        out[col] = row.get(col, np.nan)
    return out


def threshold_rows(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    df = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def foldwise_rows(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def predictions(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out = {
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
        "cn_fp_rate": safe_div(fp, fp + tn),
        "ad_fn_rate": safe_div(fn, fn + tp),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def subgroup(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for keys, sub in pred.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group = keys
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
        row["split"] = "test" if "_test_" in path.name else "train_dev"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["run_id", "label", "split"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            acc_site_raw_mean=("acc_site_raw", "mean"),
            acc_site_latent_mean=("acc_site_latent", "mean"),
            acc_site_latent_std=("acc_site_latent", "std"),
            chance_level_mean=("chance_level", "mean"),
        )
        .reset_index()
    )


def add_deltas(main: pd.DataFrame) -> pd.DataFrame:
    ref = main[main["run_id"].eq("current_locked_ch1_0_2")]
    if ref.empty:
        return main
    ref_row = ref.iloc[0]
    out = main.copy()
    for metric in METRICS:
        out[f"delta_vs_current_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
    return out


def write_recommendation(outdir: Path, main: pd.DataFrame, manufacturer: pd.DataFrame, scanner: pd.DataFrame) -> None:
    cand = main[main["run_id"].eq("manufacturer_balanced_sampler_full_5x5")].iloc[0]
    ref = main[main["run_id"].eq("current_locked_ch1_0_2")].iloc[0]
    promote = (
        cand["auc"] >= 0.794
        and cand["pr_auc"] >= ref["pr_auc"]
        and cand["balanced_accuracy"] >= ref["balanced_accuracy"] - 0.01
        and cand["f1"] >= ref["f1"] - 0.01
    )
    lines = [
        "# Manufacturer-Balanced Sampler FULL 5x5 Recommendation",
        "",
        "Primary readout: classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        f"Candidate AUC={cand['auc']:.4f}, PR-AUC={cand['pr_auc']:.4f}, BA={cand['balanced_accuracy']:.4f}, F1={cand['f1']:.4f}.",
        f"Current locked AUC={ref['auc']:.4f}, PR-AUC={ref['pr_auc']:.4f}, BA={ref['balanced_accuracy']:.4f}, F1={ref['f1']:.4f}.",
        f"Delta AUC={cand['delta_vs_current_auc']:+.4f}, Delta PR-AUC={cand['delta_vs_current_pr_auc']:+.4f}, Delta BA={cand['delta_vs_current_balanced_accuracy']:+.4f}, Delta F1={cand['delta_vs_current_f1']:+.4f}.",
        "",
        f"Promotion rule result: {'PROMOTE' if promote else 'DO NOT PROMOTE'} as revised main model.",
        "",
        "Promotion requires AUC >= 0.794 or clear improvement over current FULL, PR-AUC nondecreasing, BA/F1 not materially worse, Philips CN FP and GE AD FN not worse, and QC/scanner leakage not worse.",
        "",
        "Manufacturer subgroup and scanner leakage tables should be checked before any manuscript update.",
    ]
    (outdir / "manufacturer_balanced_sampler_full_5x5_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_outdir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    cand_run = resolve(args.candidate_run_dir)
    cand_readout = resolve(args.candidate_readout_dir)
    ref_run = resolve(args.reference_run_dir)
    ref_readout = resolve(args.reference_readout_dir)
    outdir = resolve(args.output_dir)

    ref_ok = validate_readout("current_locked_ch1_0_2", ref_readout, required=True)
    cand_ok = validate_readout("manufacturer_balanced_sampler_full_5x5", cand_readout, required=not args.dry_run)
    print(f"Reference readout OK: {ref_ok} -> {ref_readout}")
    print(f"Candidate readout OK: {cand_ok} -> {cand_readout}")
    if args.dry_run:
        print("Dry-run complete. No outputs written.")
        return 0

    prepare_outdir(outdir, overwrite=args.overwrite, dry_run=False)
    runs = [
        ("current_locked_ch1_0_2", "current locked FULL [1,0,2]", ref_run, ref_readout),
        ("manufacturer_balanced_sampler_full_5x5", "manufacturer-balanced sampler FULL [1,0,2]", cand_run, cand_readout),
    ]
    main_df = pd.DataFrame([primary_row(run_id, label, readout) for run_id, label, _run, readout in runs])
    main_df = add_deltas(main_df)
    thresholds = pd.concat([threshold_rows(run_id, label, readout) for run_id, label, _run, readout in runs], ignore_index=True)
    foldwise = pd.concat([foldwise_rows(run_id, label, readout) for run_id, label, _run, readout in runs], ignore_index=True)
    pred = pd.concat([predictions(run_id, label, readout) for run_id, label, _run, readout in runs], ignore_index=True)
    manufacturer = subgroup(pred, "Manufacturer")
    sex = subgroup(pred, "Sex")
    scanner = pd.concat([scanner_leakage(run_id, label, run) for run_id, label, run, _readout in runs], ignore_index=True)

    write_table(outdir, "main_model_comparison", main_df)
    write_table(outdir, "threshold_comparison", thresholds)
    write_table(outdir, "foldwise_comparison", foldwise)
    write_table(outdir, "manufacturer_subgroup_comparison", manufacturer)
    write_table(outdir, "sex_subgroup_comparison", sex)
    write_table(outdir, "scanner_leakage_comparison", scanner)
    write_recommendation(outdir, main_df, manufacturer, scanner)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "candidate_run_dir": str(cand_run),
        "candidate_readout_dir": str(cand_readout),
        "reference_run_dir": str(ref_run),
        "reference_readout_dir": str(ref_readout),
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {outdir}")
    print(main_df[["run_id", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
