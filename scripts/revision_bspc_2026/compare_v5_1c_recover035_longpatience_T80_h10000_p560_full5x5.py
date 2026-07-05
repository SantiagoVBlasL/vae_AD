#!/usr/bin/env python3
"""Compare recover035_longpatience_T80_h10000_p560 FULL 5x5 against three references.

References:
  1. locked v5.1b horizon4480/cycles56 [1,0,2]   — primary promotion gate
  2. recover035_full5x5 [1,0,2]                  — isolates the long-patience effect
  3. recover035_scheduler90_sync_full5x5 [1,0,2] — sibling exploratory run

Promotion rule:
  AUC > 0.782951 AND PR-AUC >= 0.559873 (must beat BOTH locked reference values simultaneously)
  Note: exploratory because outer-fold splits shifted due to 035_S_6927 rescue.
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

DEFAULT_CANDIDATE_RUN = RESULTS / "recover035_longpatience_T80_h10000_p560_full5x5"
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "recover035_longpatience_T80_h10000_p560_full5x5_comparison"

REFERENCE_LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
REFERENCE_LOCKED_READOUT = REFERENCE_LOCKED_RUN / "classifier_only_readout"

REFERENCE_RECOVER035_RUN = RESULTS / "recover035_full5x5"
REFERENCE_RECOVER035_READOUT = REFERENCE_RECOVER035_RUN / "classifier_only_readout"

REFERENCE_SCHEDULER90_RUN = RESULTS / "recover035_scheduler90_sync_full5x5"
REFERENCE_SCHEDULER90_READOUT = REFERENCE_SCHEDULER90_RUN / "classifier_only_readout"

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

RUN_ID_LOCKED = "locked_horizon4480_cycles56"
RUN_ID_RECOVER035 = "recover035_full5x5"
RUN_ID_SCHEDULER90 = "recover035_scheduler90_sync_full5x5"
RUN_ID_CAND = "recover035_longpatience_T80_h10000_p560_full5x5"

LABEL_LOCKED = "locked v5.1b horizon4480/cycles56 [1,0,2]"
LABEL_RECOVER035 = "recover035 FULL [1,0,2] (035_S_6927, 4480/56, patience=320)"
LABEL_SCHEDULER90 = "recover035_scheduler90_sync FULL [1,0,2] (035_S_6927, 4500/50, T0=90)"
LABEL_CAND = "recover035_longpatience FULL [1,0,2] (035_S_6927, 10000/125, T0=80, patience=560)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_CANDIDATE_READOUT)
    parser.add_argument("--reference-locked-run-dir", type=Path, default=REFERENCE_LOCKED_RUN)
    parser.add_argument("--reference-locked-readout-dir", type=Path, default=REFERENCE_LOCKED_READOUT)
    parser.add_argument("--reference-recover035-run-dir", type=Path, default=REFERENCE_RECOVER035_RUN)
    parser.add_argument("--reference-recover035-readout-dir", type=Path, default=REFERENCE_RECOVER035_READOUT)
    parser.add_argument("--reference-scheduler90-run-dir", type=Path, default=REFERENCE_SCHEDULER90_RUN)
    parser.add_argument("--reference-scheduler90-readout-dir", type=Path, default=REFERENCE_SCHEDULER90_READOUT)
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
    mask = (
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
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
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
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
    out: Dict[str, Any] = {
        "run_id": run_id, "label": label,
        "model_name": PRIMARY_MODEL, "threshold_strategy": PRIMARY_THRESHOLD,
    }
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
    rows: List[Dict[str, Any]] = []
    if group_col not in pred.columns:
        return pd.DataFrame()
    for keys, sub in pred.groupby(["run_id", "label", group_col], dropna=False):
        run_id, label, group = keys
        row: Dict[str, Any] = {"run_id": run_id, "label": label, group_col: group}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def scanner_leakage(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
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
            warnings.append("latent_minus_raw_site_acc computed from acc_site_latent - acc_site_raw")
        else:
            raw["latent_minus_raw_site_acc"] = np.nan
            warnings.append("latent_minus_raw_site_acc absent and could not be computed")
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


def add_deltas_vs_locked(main: pd.DataFrame) -> pd.DataFrame:
    ref = main[main["run_id"].eq(RUN_ID_LOCKED)]
    if ref.empty:
        return main
    ref_row = ref.iloc[0]
    out = main.copy()
    for metric in METRICS:
        out[f"delta_vs_locked_{metric}"] = (
            pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
        )
    return out


def add_deltas_vs_recover035(main: pd.DataFrame) -> pd.DataFrame:
    ref = main[main["run_id"].eq(RUN_ID_RECOVER035)]
    if ref.empty:
        return main
    ref_row = ref.iloc[0]
    out = main.copy()
    for metric in METRICS:
        out[f"delta_vs_recover035_{metric}"] = (
            pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
        )
    return out


def philips_ge_focus(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (run_id, label), sub in pred.groupby(["run_id", "label"], dropna=False):
        philips_cn = sub[(sub["Manufacturer"].eq("Philips")) & (sub["y_true"].eq(0))]
        ge_ad = sub[(sub["Manufacturer"].eq("GE")) & (sub["y_true"].eq(1))]
        rows.append({
            "run_id": run_id, "label": label,
            "philips_cn_n": int(len(philips_cn)),
            "philips_cn_fp": int((philips_cn["y_pred"] == 1).sum()),
            "philips_cn_fp_rate": safe_div(int((philips_cn["y_pred"] == 1).sum()), int(len(philips_cn))),
            "ge_ad_n": int(len(ge_ad)),
            "ge_ad_fn": int((ge_ad["y_pred"] == 0).sum()),
            "ge_ad_fn_rate": safe_div(int((ge_ad["y_pred"] == 0).sum()), int(len(ge_ad))),
        })
    out = pd.DataFrame(rows)
    ref = out[out["run_id"].eq(RUN_ID_LOCKED)]
    if not ref.empty:
        ref_row = ref.iloc[0]
        for col in ["philips_cn_fp_rate", "ge_ad_fn_rate"]:
            out[f"delta_vs_locked_{col}"] = out[col] - float(ref_row[col])
    return out


def recover035_fold_report(pred_cand: pd.DataFrame) -> pd.DataFrame:
    if "SubjectID" not in pred_cand.columns:
        return pd.DataFrame()
    rows_035 = pred_cand[pred_cand["SubjectID"].astype(str) == RECOVER_SUBJECT].copy()
    if rows_035.empty:
        return pd.DataFrame([{
            "SubjectID": RECOVER_SUBJECT,
            "note": "035_S_6927 not found in candidate predictions — was it in the test fold?",
        }])
    return rows_035[
        [c for c in ["SubjectID", "fold", "y_true", "y_score", "y_pred", "Manufacturer", "ResearchGroup_Mapped"]
         if c in rows_035.columns]
    ].reset_index(drop=True)


def write_recommendation(
    outdir: Path,
    main: pd.DataFrame,
    focus: pd.DataFrame,
    scanner: pd.DataFrame,
    fold035: pd.DataFrame,
    recover035_available: bool,
    scheduler90_available: bool,
) -> None:
    cand = main[main["run_id"].eq(RUN_ID_CAND)].iloc[0]
    ref = main[main["run_id"].eq(RUN_ID_LOCKED)].iloc[0]
    focus_cand = focus[focus["run_id"].eq(RUN_ID_CAND)].iloc[0] if not focus[focus["run_id"].eq(RUN_ID_CAND)].empty else None
    focus_ref = focus[focus["run_id"].eq(RUN_ID_LOCKED)].iloc[0] if not focus[focus["run_id"].eq(RUN_ID_LOCKED)].empty else None

    auc_ok = float(cand["auc"]) > LOCKED_AUC
    pr_auc_ok = float(cand["pr_auc"]) >= LOCKED_PR_AUC
    ba_ok = float(cand["balanced_accuracy"]) >= float(ref["balanced_accuracy"]) - 0.01
    f1_ok = float(cand["f1"]) >= float(ref["f1"]) - 0.01
    philips_ok = (
        float(focus_cand["philips_cn_fp_rate"]) <= float(focus_ref["philips_cn_fp_rate"])
        if (focus_cand is not None and focus_ref is not None)
        else False
    )
    ge_ok = (
        float(focus_cand["ge_ad_fn_rate"]) <= float(focus_ref["ge_ad_fn_rate"])
        if (focus_cand is not None and focus_ref is not None)
        else False
    )
    promote = auc_ok and pr_auc_ok and ba_ok and f1_ok and philips_ok and ge_ok

    fold035_note = ""
    if not fold035.empty and "fold" in fold035.columns:
        fold_val = int(fold035["fold"].iloc[0])
        y_pred = int(fold035["y_pred"].iloc[0]) if "y_pred" in fold035.columns else "?"
        y_score = float(fold035["y_score"].iloc[0]) if "y_score" in fold035.columns else float("nan")
        fold035_note = (
            f"\n\n## 035_S_6927 prediction\n\n"
            f"035_S_6927 (AD, SIEMENS) appeared in outer-test fold {fold_val}. "
            f"Predicted label: {y_pred} (y_score={y_score:.4f}). y_true=1 (AD). "
            f"{'Correctly classified as AD.' if y_pred == 1 else 'Misclassified as CN.'}"
        )

    longpatience_vs_recover035_note = ""
    if recover035_available and RUN_ID_RECOVER035 in main["run_id"].values:
        r035 = main[main["run_id"].eq(RUN_ID_RECOVER035)].iloc[0]
        delta_auc = float(cand["auc"]) - float(r035["auc"])
        delta_pr = float(cand["pr_auc"]) - float(r035["pr_auc"])
        delta_ba = float(cand["balanced_accuracy"]) - float(r035["balanced_accuracy"])
        longpatience_vs_recover035_note = (
            f"\n\n## Long-patience effect (candidate vs recover035_full5x5)\n\n"
            f"recover035_full5x5 uses identical metadata and scheduler (epochs=4480, cycles=56, T0=80, patience=320).\n"
            f"longpatience changes: epochs=10000, cycles=125, patience=560. Cycle length=80 unchanged.\n"
            f"Delta AUC (longpatience - recover035): {delta_auc:+.4f}\n"
            f"Delta PR-AUC: {delta_pr:+.4f}\n"
            f"Delta BA: {delta_ba:+.4f}\n"
        )

    scheduler90_vs_cand_note = ""
    if scheduler90_available and RUN_ID_SCHEDULER90 in main["run_id"].values:
        sched = main[main["run_id"].eq(RUN_ID_SCHEDULER90)].iloc[0]
        delta_auc_s = float(cand["auc"]) - float(sched["auc"])
        delta_pr_s = float(cand["pr_auc"]) - float(sched["pr_auc"])
        scheduler90_vs_cand_note = (
            f"\n\n## Longpatience vs scheduler90_sync comparison\n\n"
            f"recover035_scheduler90_sync uses epochs=4500, cycles=50, T0=90 (90-epoch cycles), patience=320.\n"
            f"longpatience uses epochs=10000, cycles=125, T0=80 (80-epoch cycles), patience=560.\n"
            f"Delta AUC (longpatience - scheduler90): {delta_auc_s:+.4f}\n"
            f"Delta PR-AUC: {delta_pr_s:+.4f}\n"
        )

    lines = [
        "# recover035_longpatience_T80_h10000_p560 FULL 5x5 Recommendation",
        "",
        f"Primary readout: classifier-only `{PRIMARY_MODEL}` with `{PRIMARY_THRESHOLD}`.",
        "",
        "**Exploratory run**: outer-fold splits differ from locked because 035_S_6927 was added to the classifier pool.",
        "Base model: recover035_full5x5. Controlled changes: epochs_vae 4480→10000, cyclical_beta_n_cycles 56→125,",
        "early_stopping_patience 320→560 (7 cycles), n_iter_logreg/svm 300→500.",
        "Cycle length = 80 epochs (unchanged). lr_scheduler_T0 = 80 (unchanged).",
        "",
        f"Candidate AUC={cand['auc']:.4f}, PR-AUC={cand['pr_auc']:.4f}, BA={cand['balanced_accuracy']:.4f}, F1={cand['f1']:.4f}.",
        f"Locked v5.1b AUC={ref['auc']:.4f}, PR-AUC={ref['pr_auc']:.4f}, BA={ref['balanced_accuracy']:.4f}, F1={ref['f1']:.4f}.",
        f"Delta AUC={cand.get('delta_vs_locked_auc', float('nan')):+.4f}, Delta PR-AUC={cand.get('delta_vs_locked_pr_auc', float('nan')):+.4f}.",
    ]
    if focus_cand is not None and focus_ref is not None:
        lines += [
            f"Philips CN FP rate: candidate={focus_cand['philips_cn_fp_rate']:.4f}, locked={focus_ref['philips_cn_fp_rate']:.4f}.",
            f"GE AD FN rate: candidate={focus_cand['ge_ad_fn_rate']:.4f}, locked={focus_ref['ge_ad_fn_rate']:.4f}.",
        ]
    lines += [
        "",
        "## Promotion gate checklist (vs locked v5.1b)",
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
        fold035_note,
        longpatience_vs_recover035_note,
        scheduler90_vs_cand_note,
    ]
    (outdir / "longpatience_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_postmortem(
    outdir: Path,
    main: pd.DataFrame,
    scanner: pd.DataFrame,
    recover035_available: bool,
    scheduler90_available: bool,
) -> None:
    ref = main[main["run_id"].eq(RUN_ID_LOCKED)].iloc[0]
    cand = main[main["run_id"].eq(RUN_ID_CAND)].iloc[0]

    table_rows = [
        f"| locked v5.1b horizon4480/cycles56 | {ref['auc']:.6f} | {ref['pr_auc']:.6f} | "
        f"{ref['balanced_accuracy']:.6f} | {ref['sensitivity']:.6f} | "
        f"{ref['specificity']:.6f} | {ref['f1']:.6f} |",
    ]
    if recover035_available and RUN_ID_RECOVER035 in main["run_id"].values:
        r035 = main[main["run_id"].eq(RUN_ID_RECOVER035)].iloc[0]
        table_rows.append(
            f"| recover035 FULL (epochs=4480, patience=320) | {r035['auc']:.6f} | {r035['pr_auc']:.6f} | "
            f"{r035['balanced_accuracy']:.6f} | {r035['sensitivity']:.6f} | "
            f"{r035['specificity']:.6f} | {r035['f1']:.6f} |"
        )
    if scheduler90_available and RUN_ID_SCHEDULER90 in main["run_id"].values:
        sched = main[main["run_id"].eq(RUN_ID_SCHEDULER90)].iloc[0]
        table_rows.append(
            f"| recover035_scheduler90_sync (epochs=4500, T0=90) | {sched['auc']:.6f} | {sched['pr_auc']:.6f} | "
            f"{sched['balanced_accuracy']:.6f} | {sched['sensitivity']:.6f} | "
            f"{sched['specificity']:.6f} | {sched['f1']:.6f} |"
        )
    table_rows.append(
        f"| recover035_longpatience (epochs=10000, T0=80, patience=560) | {cand['auc']:.6f} | {cand['pr_auc']:.6f} | "
        f"{cand['balanced_accuracy']:.6f} | {cand['sensitivity']:.6f} | "
        f"{cand['specificity']:.6f} | {cand['f1']:.6f} |"
    )

    lines = [
        "# recover035_longpatience_T80_h10000_p560 FULL 5x5 post-mortem",
        "",
        "Read-only comparison of the longpatience candidate vs three references.",
        "",
        f"Primary readout: classifier-only `{PRIMARY_MODEL}` with `{PRIMARY_THRESHOLD}`.",
        "",
        "| model | AUC | PR-AUC | BA | Sens | Spec | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Delta vs locked v5.1b",
        "",
        f"- AUC delta: {cand['auc'] - ref['auc']:+.6f}",
        f"- PR-AUC delta: {cand['pr_auc'] - ref['pr_auc']:+.6f}",
        f"- Balanced accuracy delta: {cand['balanced_accuracy'] - ref['balanced_accuracy']:+.6f}",
        f"- Sensitivity delta: {cand['sensitivity'] - ref['sensitivity']:+.6f}",
        f"- Specificity delta: {cand['specificity'] - ref['specificity']:+.6f}",
        f"- F1 delta: {cand['f1'] - ref['f1']:+.6f}",
        "",
        "## Context",
        "",
        "recover035_longpatience changes vs recover035_full5x5 (which is the base model):",
        "  - epochs_vae: 4480 → 10000",
        "  - cyclical_beta_n_cycles: 56 → 125  (cycle length = 80 epochs, unchanged)",
        "  - early_stopping_patience_vae: 320 → 560  (= 7 complete 80-epoch cycles)",
        "  - n_iter_logreg/svm: 300 → 500",
        "Unchanged: lr_scheduler_T0=80, beta_vae=2.5, latent_dim=256, channels=[1,0,2],",
        "  metadata_path=patched_metadata_candidate.csv, all other VAE/classifier parameters.",
        "128_S_2002 remains excluded from both VAE and classifier pools.",
        "Stage B C-grid unchanged: original [1e-3, 1e-2, 1e-1, 1.0].",
        "",
        "## Decision",
        "",
        "See longpatience_recommendation.md for promotion gate outcome.",
        f"If AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} both met: consider promoting as revised manuscript model.",
        "Otherwise: locked v5.1b horizon4480/cycles56 remains the paper model.",
    ]
    if not scanner.empty and "scanner_leakage_warning" in scanner.columns:
        warnings = sorted({
            str(v) for v in scanner["scanner_leakage_warning"].dropna().tolist() if str(v).strip()
        })
        if warnings:
            lines.extend(["", "## Scanner leakage note", "", "; ".join(warnings)])
    (outdir / "final_postmortem.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    locked_run = resolve(args.reference_locked_run_dir)
    locked_readout = resolve(args.reference_locked_readout_dir)
    recover035_run = resolve(args.reference_recover035_run_dir)
    recover035_readout = resolve(args.reference_recover035_readout_dir)
    scheduler90_run = resolve(args.reference_scheduler90_run_dir)
    scheduler90_readout = resolve(args.reference_scheduler90_readout_dir)
    outdir = resolve(args.output_dir)

    locked_ok = validate_readout("locked reference", locked_readout, required=True)
    recover035_ok = validate_readout("recover035 reference", recover035_readout, required=False)
    scheduler90_ok = validate_readout("scheduler90 reference", scheduler90_readout, required=False)
    cand_ok = validate_readout("candidate", cand_readout, required=not args.dry_run)

    print(f"Locked reference readout OK   : {locked_ok} ({locked_readout})")
    print(f"recover035 reference readout  : {recover035_ok} ({recover035_readout})")
    print(f"scheduler90 reference readout : {scheduler90_ok} ({scheduler90_readout})")
    print(f"Candidate readout OK          : {cand_ok} ({cand_readout})")
    print(f"Output dir                    : {outdir}")
    print(f"Primary readout               : {PRIMARY_MODEL} / {PRIMARY_THRESHOLD}")
    print(f"Promotion gate                : AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} (both required)")
    if not recover035_ok:
        print("  WARNING: recover035_full5x5 readout not available — long-patience effect table will be omitted.")
    if not scheduler90_ok:
        print("  WARNING: scheduler90 readout not available — scheduler comparison will be omitted.")
    if args.dry_run:
        print("Dry-run complete. No comparison files were written.")
        return 0

    prepare_outdir(outdir, overwrite=args.overwrite, dry_run=False)

    active_runs = [(RUN_ID_LOCKED, LABEL_LOCKED, locked_readout)]
    if recover035_ok:
        active_runs.append((RUN_ID_RECOVER035, LABEL_RECOVER035, recover035_readout))
    if scheduler90_ok:
        active_runs.append((RUN_ID_SCHEDULER90, LABEL_SCHEDULER90, scheduler90_readout))
    active_runs.append((RUN_ID_CAND, LABEL_CAND, cand_readout))

    main_rows = [primary_row(rid, lbl, rd) for rid, lbl, rd in active_runs]
    main_df = pd.DataFrame(main_rows)
    main_df = add_deltas_vs_locked(main_df)
    main_df = add_deltas_vs_recover035(main_df)
    write_table(outdir, "main_model_comparison", main_df)

    threshold_parts = [threshold_rows(rid, lbl, rd) for rid, lbl, rd in active_runs]
    write_table(outdir, "threshold_comparison", pd.concat(threshold_parts, ignore_index=True))

    foldwise_parts = [foldwise_rows(rid, lbl, rd) for rid, lbl, rd in active_runs]
    write_table(outdir, "foldwise_comparison", pd.concat(foldwise_parts, ignore_index=True))

    pred_parts = [predictions(rid, lbl, rd) for rid, lbl, rd in active_runs]
    pred = pd.concat(pred_parts, ignore_index=True)
    write_table(outdir, "manufacturer_subgroup_comparison", subgroup(pred, "Manufacturer"))
    write_table(outdir, "sex_subgroup_comparison", subgroup(pred, "Sex"))

    focus = philips_ge_focus(pred)
    write_table(outdir, "philips_cn_ge_ad_error_focus", focus)

    pred_cand_only = pred[pred["run_id"].eq(RUN_ID_CAND)].copy()
    fold035 = recover035_fold_report(pred_cand_only)
    write_table(outdir, "recover035_fold_prediction", fold035)

    scanner_parts = []
    for rid, lbl, run_dir in [
        (RUN_ID_LOCKED, LABEL_LOCKED, locked_run),
        (RUN_ID_CAND, LABEL_CAND, cand_run),
    ]:
        s = scanner_leakage(rid, lbl, run_dir)
        if not s.empty:
            scanner_parts.append(s)
    if recover035_ok:
        s = scanner_leakage(RUN_ID_RECOVER035, LABEL_RECOVER035, recover035_run)
        if not s.empty:
            scanner_parts.append(s)
    if scheduler90_ok:
        s = scanner_leakage(RUN_ID_SCHEDULER90, LABEL_SCHEDULER90, scheduler90_run)
        if not s.empty:
            scanner_parts.append(s)
    scanner = pd.concat(scanner_parts, ignore_index=True) if scanner_parts else pd.DataFrame()
    write_table(outdir, "scanner_leakage_comparison", scanner)

    write_recommendation(outdir, main_df, focus, scanner, fold035, recover035_available=recover035_ok, scheduler90_available=scheduler90_ok)
    write_final_postmortem(outdir, main_df, scanner, recover035_available=recover035_ok, scheduler90_available=scheduler90_ok)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "candidate_run_dir": str(cand_run),
        "candidate_readout_dir": str(cand_readout),
        "reference_locked_run_dir": str(locked_run),
        "reference_locked_readout_dir": str(locked_readout),
        "reference_recover035_run_dir": str(recover035_run),
        "reference_recover035_readout_dir": str(recover035_readout),
        "reference_scheduler90_run_dir": str(scheduler90_run),
        "reference_scheduler90_readout_dir": str(scheduler90_readout),
        "recover035_reference_available": bool(recover035_ok),
        "scheduler90_reference_available": bool(scheduler90_ok),
        "output_dir": str(outdir),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "promotion_gate": f"AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} (both simultaneously; exploratory)",
        "recover_subject": RECOVER_SUBJECT,
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
        "recover035_outputs_modified": False,
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote comparison to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
