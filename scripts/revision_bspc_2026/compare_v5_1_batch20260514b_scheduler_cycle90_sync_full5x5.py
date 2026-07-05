#!/usr/bin/env python3
"""Compare scheduler_cycle90_sync FULL 5x5 against locked v5.1b horizon4480/cycles56 FULL [1,0,2].

Candidate change: epochs_vae 4480->4500, cyclical_beta_n_cycles 56->50, lr_scheduler_T0 80->90.
Result: synchronized beta/LR cycle length = 90 epochs each.

Promotion rule:
  - AUC > 0.782951 AND PR-AUC >= 0.559873 (must beat BOTH simultaneously)
  - BA/F1 do not materially worsen vs locked current
  - Philips CN FP rate does not worsen
  - GE AD FN rate does not worsen
  - Scanner leakage/QC does not worsen
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
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_CANDIDATE_RUN = RESULTS / "scheduler_cycle90_sync_full5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "scheduler_cycle90_sync_full5x5_comparison"
REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
REFERENCE_READOUT = REFERENCE_RUN / "classifier_only_readout"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

FOCUS_FOLDS = [1, 4]


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
            warnings.append("latent_minus_raw_site_acc was absent and computed as acc_site_latent - acc_site_raw")
        else:
            raw["latent_minus_raw_site_acc"] = np.nan
            warnings.append("latent_minus_raw_site_acc was absent and could not be computed")
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
    ref = main[main["run_id"].eq("current_locked_horizon4480")]
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
    ref = out[out["run_id"].eq("current_locked_horizon4480")]
    if not ref.empty:
        ref_row = ref.iloc[0]
        for col in ["philips_cn_fp_rate", "ge_ad_fn_rate"]:
            out[f"delta_vs_current_{col}"] = out[col] - float(ref_row[col])
    return out


def fold_focus_comparison(foldwise: pd.DataFrame) -> pd.DataFrame:
    """Isolate fold 1 and fold 4 rows for targeted comparison."""
    df = foldwise[foldwise["fold"].isin(FOCUS_FOLDS)].copy()
    ref = df[df["run_id"].eq("current_locked_horizon4480")].set_index("fold")
    cand = df[df["run_id"].eq("scheduler_cycle90_sync_full5x5")].set_index("fold")
    rows = []
    for fold in FOCUS_FOLDS:
        if fold not in ref.index or fold not in cand.index:
            continue
        r = ref.loc[fold]
        c = cand.loc[fold]
        row: Dict[str, Any] = {"fold": fold}
        for m in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            row[f"locked_{m}"] = float(r.get(m, np.nan))
            row[f"candidate_{m}"] = float(c.get(m, np.nan))
            row[f"delta_{m}"] = row[f"candidate_{m}"] - row[f"locked_{m}"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_recommendation(outdir: Path, main: pd.DataFrame, focus: pd.DataFrame, scanner: pd.DataFrame) -> None:
    cand = main[main["run_id"].eq("scheduler_cycle90_sync_full5x5")].iloc[0]
    ref = main[main["run_id"].eq("current_locked_horizon4480")].iloc[0]
    focus_cand = focus[focus["run_id"].eq("scheduler_cycle90_sync_full5x5")].iloc[0]
    focus_ref = focus[focus["run_id"].eq("current_locked_horizon4480")].iloc[0]
    auc_ok = float(cand["auc"]) > LOCKED_AUC
    pr_auc_ok = float(cand["pr_auc"]) >= LOCKED_PR_AUC
    ba_ok = float(cand["balanced_accuracy"]) >= float(ref["balanced_accuracy"]) - 0.01
    f1_ok = float(cand["f1"]) >= float(ref["f1"]) - 0.01
    philips_ok = float(focus_cand["philips_cn_fp_rate"]) <= float(focus_ref["philips_cn_fp_rate"])
    ge_ok = float(focus_cand["ge_ad_fn_rate"]) <= float(focus_ref["ge_ad_fn_rate"])
    promote = auc_ok and pr_auc_ok and ba_ok and f1_ok and philips_ok and ge_ok
    lines = [
        "# Scheduler-Cycle90-Sync FULL 5x5 Recommendation",
        "",
        "Primary readout: classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "Controlled change: epochs_vae 4480->4500, cyclical_beta_n_cycles 56->50, lr_scheduler_T0 80->90.",
        "Result: synchronized beta/LR cycle length = 90 epochs (4500/50=90, T0=90).",
        "Locked reference used 80-epoch cycles (4480/56=80, T0=80).",
        "",
        f"Candidate AUC={cand['auc']:.4f}, PR-AUC={cand['pr_auc']:.4f}, BA={cand['balanced_accuracy']:.4f}, F1={cand['f1']:.4f}.",
        f"Current locked AUC={ref['auc']:.4f}, PR-AUC={ref['pr_auc']:.4f}, BA={ref['balanced_accuracy']:.4f}, F1={ref['f1']:.4f}.",
        f"Delta AUC={cand['delta_vs_current_auc']:+.4f}, Delta PR-AUC={cand['delta_vs_current_pr_auc']:+.4f}.",
        f"Philips CN FP rate: candidate={focus_cand['philips_cn_fp_rate']:.4f}, current={focus_ref['philips_cn_fp_rate']:.4f}.",
        f"GE AD FN rate: candidate={focus_cand['ge_ad_fn_rate']:.4f}, current={focus_ref['ge_ad_fn_rate']:.4f}.",
        "",
        "## Promotion gate checklist",
        "",
        f"- AUC > {LOCKED_AUC}: {'PASS' if auc_ok else 'FAIL'} (candidate={cand['auc']:.4f})",
        f"- PR-AUC >= {LOCKED_PR_AUC}: {'PASS' if pr_auc_ok else 'FAIL'} (candidate={cand['pr_auc']:.4f})",
        f"- BA not materially worse (>= {float(ref['balanced_accuracy']):.4f} - 0.01): {'PASS' if ba_ok else 'FAIL'}",
        f"- F1 not materially worse (>= {float(ref['f1']):.4f} - 0.01): {'PASS' if f1_ok else 'FAIL'}",
        f"- Philips CN FP rate not worse: {'PASS' if philips_ok else 'FAIL'}",
        f"- GE AD FN rate not worse: {'PASS' if ge_ok else 'FAIL'}",
        "",
        f"Promotion rule result: **{'PROMOTE' if promote else 'DO NOT PROMOTE'}** as revised main model.",
        "",
        "Promotion requires ALL gate conditions met simultaneously (dual AUC+PR-AUC gate is strict).",
        "If not promoted, the locked current FULL horizon4480/cycles56 model remains the paper model.",
        "",
        "## Schedule-phase context",
        "",
        "The candidate tests whether slightly longer 90-epoch beta/LR cycles (vs 80 in locked) improve",
        "generalization. The +20 epochs per cycle gives the LR cosine restart more room to explore the loss",
        "surface before the next restart. Fold 4 is a particular focus since it was the last fold to converge",
        "in the locked model (best epoch 3836, phase 0.95 in the 80-epoch cycle).",
        "See fold_focus_comparison.csv/.md for fold 1 and fold 4 specific deltas.",
    ]
    (outdir / "scheduler_cycle90_sync_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_postmortem(outdir: Path, main: pd.DataFrame, scanner: pd.DataFrame) -> None:
    ref = main[main["run_id"].eq("current_locked_horizon4480")].iloc[0]
    cand = main[main["run_id"].eq("scheduler_cycle90_sync_full5x5")].iloc[0]
    lines = [
        "# Final scheduler_cycle90_sync post-mortem",
        "",
        "This read-only comparison contrasts the synchronized 90-epoch cycle candidate against the",
        "locked v5.1b horizon4480/cycles56 FULL [1,0,2] model.",
        "",
        "Primary readout: classifier-only `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec`.",
        "",
        "| model | AUC | PR-AUC | BA | Sens | Spec | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| locked horizon4480 cycles56 (80-epoch cycles) | {ref['auc']:.6f} | {ref['pr_auc']:.6f} | "
            f"{ref['balanced_accuracy']:.6f} | {ref['sensitivity']:.6f} | "
            f"{ref['specificity']:.6f} | {ref['f1']:.6f} |"
        ),
        (
            f"| scheduler_cycle90_sync (90-epoch cycles) | {cand['auc']:.6f} | {cand['pr_auc']:.6f} | "
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
        "## Schedule context",
        "",
        "The candidate changes only the scheduler cycle length: 80-epoch -> 90-epoch synchronized beta/LR cycles",
        "(4500/50=90 for beta, T0=90 for LR cosine restarts). Total epochs change by only +0.4%",
        "(4500 vs 4480). Both cycles start phase-aligned at epoch 0 and restart simultaneously every 90 epochs.",
        "Early stopping patience remains 320 epochs (~3.6 cycles at 90 epochs/cycle).",
        "",
        "## Fold 1 and Fold 4 focus",
        "",
        "Fold 4 was the slowest-converging fold in the locked model (best epoch 3836/4480,",
        "phase 0.95 in the 80-epoch cycle — near the end of a stable-beta phase).",
        "See fold_focus_comparison.csv/.md for fold 1 and fold 4 specific results.",
        "",
        "## Decision",
        "",
        "See scheduler_cycle90_sync_recommendation.md for promotion gate outcome.",
        f"Promotion requires AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} both met.",
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
    (outdir / "final_scheduler_cycle90_sync_postmortem.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    ref_ok = validate_readout("reference", ref_readout, required=not args.dry_run)
    cand_ok = validate_readout("candidate", cand_readout, required=not args.dry_run)
    print(f"Reference readout OK: {ref_ok} ({ref_readout})")
    print(f"Candidate readout OK: {cand_ok} ({cand_readout})")
    print(f"Candidate run dir    : {cand_run}")
    print(f"Output dir           : {outdir}")
    print(f"Primary readout      : {PRIMARY_MODEL} / {PRIMARY_THRESHOLD}")
    print(f"Promotion gate       : AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} (both required simultaneously)")
    print(f"Focus folds          : {FOCUS_FOLDS}")
    if args.dry_run:
        print("Dry-run complete. No comparison files were written.")
        return 0

    prepare_outdir(outdir, overwrite=args.overwrite, dry_run=False)
    main_df = pd.DataFrame(
        [
            primary_row("current_locked_horizon4480", "locked horizon4480/cycles56 FULL [1,0,2]", ref_readout),
            primary_row("scheduler_cycle90_sync_full5x5", "scheduler_cycle90_sync FULL [1,0,2]", cand_readout),
        ]
    )
    main_df = add_deltas(main_df)
    write_table(outdir, "main_model_comparison", main_df)

    threshold = pd.concat(
        [
            threshold_rows("current_locked_horizon4480", "locked horizon4480/cycles56 FULL [1,0,2]", ref_readout),
            threshold_rows("scheduler_cycle90_sync_full5x5", "scheduler_cycle90_sync FULL [1,0,2]", cand_readout),
        ],
        ignore_index=True,
    )
    write_table(outdir, "threshold_comparison", threshold)
    foldwise = pd.concat(
        [
            foldwise_rows("current_locked_horizon4480", "locked horizon4480/cycles56 FULL [1,0,2]", ref_readout),
            foldwise_rows("scheduler_cycle90_sync_full5x5", "scheduler_cycle90_sync FULL [1,0,2]", cand_readout),
        ],
        ignore_index=True,
    )
    write_table(outdir, "foldwise_comparison", foldwise)
    fold_focus = fold_focus_comparison(foldwise)
    write_table(outdir, "fold_focus_comparison", fold_focus)
    pred = pd.concat(
        [
            predictions("current_locked_horizon4480", "locked horizon4480/cycles56 FULL [1,0,2]", ref_readout),
            predictions("scheduler_cycle90_sync_full5x5", "scheduler_cycle90_sync FULL [1,0,2]", cand_readout),
        ],
        ignore_index=True,
    )
    write_table(outdir, "manufacturer_subgroup_comparison", subgroup(pred, "Manufacturer"))
    write_table(outdir, "sex_subgroup_comparison", subgroup(pred, "Sex"))
    focus = philips_ge_focus(pred)
    write_table(outdir, "philips_cn_ge_ad_error_focus", focus)
    scanner = pd.concat(
        [
            scanner_leakage("current_locked_horizon4480", "locked horizon4480/cycles56 FULL [1,0,2]", ref_run),
            scanner_leakage("scheduler_cycle90_sync_full5x5", "scheduler_cycle90_sync FULL [1,0,2]", cand_run),
        ],
        ignore_index=True,
    )
    write_table(outdir, "scanner_leakage_comparison", scanner)
    write_recommendation(outdir, main_df, focus, scanner)
    write_final_postmortem(outdir, main_df, scanner)
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
        "promotion_gate": f"AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} (both simultaneously)",
        "focus_folds": FOCUS_FOLDS,
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
