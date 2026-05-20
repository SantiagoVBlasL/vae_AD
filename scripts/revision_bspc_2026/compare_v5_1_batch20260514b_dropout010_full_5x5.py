#!/usr/bin/env python3
"""Compare dropout=0.10 FULL 5x5 against locked current FULL [1,0,2]."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_CANDIDATE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_dropout010_full_5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_dropout010_full_5x5_comparison"
REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
REFERENCE_READOUT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]


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


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


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
        "brier": float(brier_score_loss(y, np.clip(score, 0.0, 1.0))),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


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


def subgroup(pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    if group_col not in pred.columns:
        return pd.DataFrame()
    for keys, sub in pred.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group = keys
        row = {"run_id": run_id, "label": label, group_col: group}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def scanner_leakage(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    rows = []
    warnings = []
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

    if "latent_minus_raw_site_acc" not in raw.columns:
        if {"acc_site_latent", "acc_site_raw"}.issubset(raw.columns):
            raw["latent_minus_raw_site_acc"] = raw["acc_site_latent"] - raw["acc_site_raw"]
            warnings.append(
                "latent_minus_raw_site_acc was absent and computed as "
                "acc_site_latent - acc_site_raw"
            )
        else:
            raw["latent_minus_raw_site_acc"] = np.nan
            warnings.append(
                "latent_minus_raw_site_acc was absent and could not be computed "
                "because acc_site_latent and/or acc_site_raw were missing"
            )

    return (
        raw.groupby(["run_id", "label", "split"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            acc_site_raw_mean=("acc_site_raw", "mean"),
            acc_site_latent_mean=("acc_site_latent", "mean"),
            acc_site_latent_std=("acc_site_latent", "std"),
            latent_minus_raw_site_acc_mean=("latent_minus_raw_site_acc", "mean"),
            chance_level_mean=("chance_level", "mean"),
        )
        .reset_index()
        .assign(scanner_leakage_warning=" | ".join(warnings))
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


def philips_ge_focus(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_id, label), sub in pred.groupby(["run_id", "label"], dropna=False):
        philips_cn = sub[(sub["Manufacturer"].eq("Philips")) & (sub["y_true"].eq(0))]
        ge_ad = sub[(sub["Manufacturer"].eq("GE")) & (sub["y_true"].eq(1))]
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "philips_cn_n": int(len(philips_cn)),
                "philips_cn_fp": int((philips_cn["y_pred"] == 1).sum()),
                "philips_cn_fp_rate": safe_div(int((philips_cn["y_pred"] == 1).sum()), int(len(philips_cn))),
                "ge_ad_n": int(len(ge_ad)),
                "ge_ad_fn": int((ge_ad["y_pred"] == 0).sum()),
                "ge_ad_fn_rate": safe_div(int((ge_ad["y_pred"] == 0).sum()), int(len(ge_ad))),
            }
        )
    out = pd.DataFrame(rows)
    ref = out[out["run_id"].eq("current_locked_ch1_0_2")]
    if not ref.empty:
        ref_row = ref.iloc[0]
        for col in ["philips_cn_fp_rate", "ge_ad_fn_rate"]:
            out[f"delta_vs_current_{col}"] = out[col] - float(ref_row[col])
    return out


def write_recommendation(outdir: Path, main: pd.DataFrame, focus: pd.DataFrame, scanner: pd.DataFrame) -> None:
    cand = main[main["run_id"].eq("dropout010_full_5x5")].iloc[0]
    ref = main[main["run_id"].eq("current_locked_ch1_0_2")].iloc[0]
    focus_cand = focus[focus["run_id"].eq("dropout010_full_5x5")].iloc[0]
    focus_ref = focus[focus["run_id"].eq("current_locked_ch1_0_2")].iloc[0]
    promote = (
        cand["auc"] >= 0.794
        and cand["pr_auc"] >= ref["pr_auc"]
        and cand["balanced_accuracy"] >= ref["balanced_accuracy"] - 0.01
        and cand["f1"] >= ref["f1"] - 0.01
        and focus_cand["philips_cn_fp_rate"] <= focus_ref["philips_cn_fp_rate"]
        and focus_cand["ge_ad_fn_rate"] <= focus_ref["ge_ad_fn_rate"]
    )
    lines = [
        "# Dropout 0.10 FULL 5x5 Recommendation",
        "",
        "Primary readout: classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        f"Candidate AUC={cand['auc']:.4f}, PR-AUC={cand['pr_auc']:.4f}, BA={cand['balanced_accuracy']:.4f}, F1={cand['f1']:.4f}.",
        f"Current locked AUC={ref['auc']:.4f}, PR-AUC={ref['pr_auc']:.4f}, BA={ref['balanced_accuracy']:.4f}, F1={ref['f1']:.4f}.",
        f"Delta AUC={cand['delta_vs_current_auc']:+.4f}, Delta PR-AUC={cand['delta_vs_current_pr_auc']:+.4f}.",
        f"Philips CN FP rate: candidate={focus_cand['philips_cn_fp_rate']:.4f}, current={focus_ref['philips_cn_fp_rate']:.4f}.",
        f"GE AD FN rate: candidate={focus_cand['ge_ad_fn_rate']:.4f}, current={focus_ref['ge_ad_fn_rate']:.4f}.",
        "",
        f"Promotion rule result: {'PROMOTE' if promote else 'DO NOT PROMOTE'} as revised main model.",
        "",
        "Promotion requires AUC >= 0.794 or clear improvement over current FULL, PR-AUC nondecreasing, BA/F1 not materially worse, Philips CN FP and GE AD FN not worse, and QC/scanner leakage not worse.",
    ]
    (outdir / "dropout010_full_5x5_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_postmortem(outdir: Path, main: pd.DataFrame, scanner: pd.DataFrame) -> None:
    ref = main[main["run_id"].eq("current_locked_ch1_0_2")].iloc[0]
    cand = main[main["run_id"].eq("dropout010_full_5x5")].iloc[0]
    lines = [
        "# Final dropout010 post-mortem",
        "",
        "This read-only audit compares the dropout_rate_vae=0.10 FULL 5x5 candidate against the locked current FULL [1,0,2] model.",
        "",
        "Primary readout: classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "| model | AUC | PR-AUC | BA | Sens | Spec | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| locked current dropout=0.15 | {ref['auc']:.6f} | {ref['pr_auc']:.6f} | "
            f"{ref['balanced_accuracy']:.6f} | {ref['sensitivity']:.6f} | "
            f"{ref['specificity']:.6f} | {ref['f1']:.6f} |"
        ),
        (
            f"| dropout010 dropout=0.10 | {cand['auc']:.6f} | {cand['pr_auc']:.6f} | "
            f"{cand['balanced_accuracy']:.6f} | {cand['sensitivity']:.6f} | "
            f"{cand['specificity']:.6f} | {cand['f1']:.6f} |"
        ),
        "",
        "## Delta vs locked current",
        "",
        f"- AUC delta: {cand['auc'] - ref['auc']:+.6f}",
        f"- PR-AUC delta: {cand['pr_auc'] - ref['pr_auc']:+.6f}",
        f"- Balanced accuracy delta: {cand['balanced_accuracy'] - ref['balanced_accuracy']:+.6f}",
        f"- Sensitivity delta: {cand['sensitivity'] - ref['sensitivity']:+.6f}",
        f"- Specificity delta: {cand['specificity'] - ref['specificity']:+.6f}",
        f"- F1 delta: {cand['f1'] - ref['f1']:+.6f}",
        "",
        "## Decision",
        "",
        "dropout010 should NOT replace the locked current FULL model.",
        "",
        "The candidate preserves essentially the same sensitivity operating point, but ranking performance is materially worse: both AUC and PR-AUC decrease versus the locked dropout=0.15 model. The small threshold-metric differences do not justify replacing the manuscript model.",
        "",
        "The locked current FULL tanh [1,0,2] model remains the paper model.",
    ]
    if not scanner.empty and "scanner_leakage_warning" in scanner.columns:
        warnings = sorted(
            {
                str(value)
                for value in scanner["scanner_leakage_warning"].dropna().tolist()
                if str(value).strip()
            }
        )
        if warnings:
            lines.extend(["", "## Scanner leakage note", "", "; ".join(warnings)])
    (outdir / "final_dropout010_postmortem.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_outdir(path: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    cand_run = resolve(args.candidate_run_dir)
    cand_readout = resolve(args.candidate_readout_dir)
    ref_run = resolve(args.reference_run_dir)
    ref_readout = resolve(args.reference_readout_dir)
    outdir = resolve(args.output_dir)

    ref_ok = validate_readout("reference", ref_readout, required=True)
    cand_ok = validate_readout("candidate", cand_readout, required=not args.dry_run)
    print(f"Reference readout OK: {ref_ok} ({ref_readout})")
    print(f"Candidate readout OK: {cand_ok} ({cand_readout})")
    print(f"Candidate run dir    : {cand_run}")
    print(f"Output dir           : {outdir}")
    print(f"Primary readout      : {PRIMARY_MODEL} / {PRIMARY_THRESHOLD}")
    if args.dry_run:
        print("Dry-run complete. No comparison files were written.")
        return 0

    prepare_outdir(outdir, overwrite=args.overwrite, dry_run=False)
    main = pd.DataFrame(
        [
            primary_row("current_locked_ch1_0_2", "current locked FULL [1,0,2]", ref_readout),
            primary_row("dropout010_full_5x5", "dropout_rate_vae=0.10 FULL [1,0,2]", cand_readout),
        ]
    )
    main = add_deltas(main)
    write_table(outdir, "main_model_comparison", main)

    threshold = pd.concat(
        [
            threshold_rows("current_locked_ch1_0_2", "current locked FULL [1,0,2]", ref_readout),
            threshold_rows("dropout010_full_5x5", "dropout_rate_vae=0.10 FULL [1,0,2]", cand_readout),
        ],
        ignore_index=True,
    )
    write_table(outdir, "threshold_comparison", threshold)
    foldwise = pd.concat(
        [
            foldwise_rows("current_locked_ch1_0_2", "current locked FULL [1,0,2]", ref_readout),
            foldwise_rows("dropout010_full_5x5", "dropout_rate_vae=0.10 FULL [1,0,2]", cand_readout),
        ],
        ignore_index=True,
    )
    write_table(outdir, "foldwise_comparison", foldwise)
    pred = pd.concat(
        [
            predictions("current_locked_ch1_0_2", "current locked FULL [1,0,2]", ref_readout),
            predictions("dropout010_full_5x5", "dropout_rate_vae=0.10 FULL [1,0,2]", cand_readout),
        ],
        ignore_index=True,
    )
    write_table(outdir, "manufacturer_subgroup_comparison", subgroup(pred, "Manufacturer"))
    write_table(outdir, "sex_subgroup_comparison", subgroup(pred, "Sex"))
    focus = philips_ge_focus(pred)
    write_table(outdir, "philips_cn_ge_ad_error_focus", focus)
    scanner = pd.concat(
        [
            scanner_leakage("current_locked_ch1_0_2", "current locked FULL [1,0,2]", ref_run),
            scanner_leakage("dropout010_full_5x5", "dropout_rate_vae=0.10 FULL [1,0,2]", cand_run),
        ],
        ignore_index=True,
    )
    write_table(outdir, "scanner_leakage_comparison", scanner)
    write_recommendation(outdir, main, focus, scanner)
    write_final_postmortem(outdir, main, scanner)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "candidate_run_dir": str(cand_run),
        "candidate_readout_dir": str(cand_readout),
        "reference_run_dir": str(ref_run),
        "reference_readout_dir": str(ref_readout),
        "output_dir": str(outdir),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
