#!/usr/bin/env python3
"""Read-only comparison for v5.1c horizon10000/cycles125 FULL run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

CANDIDATE_RUN = RESULTS / "adni_v5_1c_recover035_ch1_0_2_horizon10000_cycles125_full_5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
V51B_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
V51B_READOUT = V51B_RUN / "classifier_only_readout"
V51C4480_RUN = RESULTS / "adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5"
V51C4480_READOUT = V51C4480_RUN / "classifier_only_readout"
OUT_DIR = RESULTS / "adni_v5_1c_horizon10000_cycles125_vs_v5_1b_and_v5_1c4480_comparison"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
PROMOTION_REF = "v5_1b_horizon4480"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--v51b-run-dir", type=Path, default=V51B_RUN)
    parser.add_argument("--v51b-readout-dir", type=Path, default=V51B_READOUT)
    parser.add_argument("--v51c4480-run-dir", type=Path, default=V51C4480_RUN)
    parser.add_argument("--v51c4480-readout-dir", type=Path, default=V51C4480_READOUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def required_readout_files(readout: Path) -> list[Path]:
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
    (outdir / f"{stem}.md").write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: pd.Series, y_score: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    out: dict[str, Any] = {
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


def primary_pooled(run_id: str, label: str, readout: Path) -> dict[str, Any]:
    df = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out: dict[str, Any] = {"run_id": run_id, "label": label}
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


def read_foldwise(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def read_thresholds(run_id: str, label: str, readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def subgroup(preds: pd.DataFrame, group_col: str, min_n: int = 1) -> pd.DataFrame:
    if group_col not in preds.columns:
        return pd.DataFrame()
    rows = []
    for keys, sub in preds.groupby(["run_id", "label", group_col], dropna=False):
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
        if pd.notna(row.get("acc_site_latent")) and pd.notna(row.get("acc_site_raw")):
            row["latent_minus_raw_site_acc"] = row["acc_site_latent"] - row["acc_site_raw"]
        rows.append(row)
    return pd.DataFrame(rows)


def latent_qc(run_id: str, label: str, run_dir: Path) -> pd.DataFrame:
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
    ref = main[main["run_id"].eq(PROMOTION_REF)]
    if ref.empty:
        return main
    ref_row = ref.iloc[0]
    out = main.copy()
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"]:
        out[f"delta_vs_v5_1b_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(ref_row[metric])
    return out


def write_recommendation(outdir: Path, main: pd.DataFrame, leakage: pd.DataFrame, mfr: pd.DataFrame, site: pd.DataFrame) -> None:
    cand = main[main["run_id"].eq("v5_1c_horizon10000")].iloc[0]
    ref = main[main["run_id"].eq(PROMOTION_REF)].iloc[0]
    auc_delta = float(cand["auc"]) - float(ref["auc"])
    pr_delta = float(cand["pr_auc"]) - float(ref["pr_auc"])
    ba_delta = float(cand["balanced_accuracy"]) - float(ref["balanced_accuracy"])
    f1_delta = float(cand["f1"]) - float(ref["f1"])
    primary_pass = (
        float(cand["auc"]) > float(ref["auc"])
        and float(cand["pr_auc"]) >= float(ref["pr_auc"])
        and ba_delta >= -0.01
        and f1_delta >= -0.01
    )
    decision = "promote_pending_integrity_and_subgroup_review" if primary_pass else "do_not_promote"
    text = f"""# v5.1c Horizon10000/Cycles125 Recommendation

Primary readout: `{PRIMARY_MODEL}` with `{PRIMARY_THRESHOLD}`.

Promotion reference is v5.1b horizon4480/cycles56.

- Reference AUC={float(ref['auc']):.6f}, PR-AUC={float(ref['pr_auc']):.6f}, BA={float(ref['balanced_accuracy']):.6f}, F1={float(ref['f1']):.6f}
- Candidate AUC={float(cand['auc']):.6f}, PR-AUC={float(cand['pr_auc']):.6f}, BA={float(cand['balanced_accuracy']):.6f}, F1={float(cand['f1']):.6f}
- Delta AUC={auc_delta:+.6f}
- Delta PR-AUC={pr_delta:+.6f}
- Delta BA={ba_delta:+.6f}
- Delta F1={f1_delta:+.6f}

Decision: **{decision}**.

Final promotion still requires:

- integrity audit PASS;
- no worse manufacturer/site subgroup behavior;
- no worse scanner/manufacturer leakage.

Tables available for that review:

- `manufacturer_subgroup_metrics.csv`
- `sitecode_subgroup_metrics.csv`
- `scanner_manufacturer_leakage.csv`
- `latent_information_qc.csv`
"""
    (outdir / "final_recommendation.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run_dir)
    candidate_readout = resolve(args.candidate_readout_dir)
    v51b_run = resolve(args.v51b_run_dir)
    v51b_readout = resolve(args.v51b_readout_dir)
    v51c4480_run = resolve(args.v51c4480_run_dir)
    v51c4480_readout = resolve(args.v51c4480_readout_dir)
    outdir = resolve(args.output_dir)

    validate_readout("v5.1b reference", v51b_readout)
    validate_readout("v5.1c 4480 reference", v51c4480_readout)
    candidate_available = all(p.exists() for p in required_readout_files(candidate_readout))
    if args.dry_run:
        print("Dry-run comparison preflight.")
        print(f"Candidate run    : {candidate_run}")
        print(f"Candidate readout: {candidate_readout}")
        print(f"Candidate readout available: {candidate_available}")
        print(f"v5.1b reference  : {v51b_run}")
        print(f"v5.1c 4480 ref   : {v51c4480_run}")
        print(f"Output dir       : {outdir}")
        print("No files were written.")
        return 0

    validate_readout("candidate", candidate_readout)
    outdir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("v5_1b_horizon4480", "v5.1b horizon4480/cycles56", v51b_run, v51b_readout),
        ("v5_1c_horizon4480", "v5.1c recover035 horizon4480/cycles56", v51c4480_run, v51c4480_readout),
        ("v5_1c_horizon10000", "v5.1c recover035 horizon10000/cycles125", candidate_run, candidate_readout),
    ]
    main = pd.DataFrame([primary_pooled(run_id, label, readout) for run_id, label, _, readout in specs])
    main = add_deltas(main)
    write_table(outdir, "main_model_comparison", main)

    foldwise = pd.concat([read_foldwise(run_id, label, readout) for run_id, label, _, readout in specs], ignore_index=True)
    write_table(outdir, "foldwise_metrics", foldwise)

    thresholds = pd.concat([read_thresholds(run_id, label, readout) for run_id, label, _, readout in specs], ignore_index=True)
    write_table(outdir, "threshold_comparison", thresholds)

    preds = pd.concat([read_predictions(run_id, label, readout) for run_id, label, _, readout in specs], ignore_index=True)
    write_table(outdir, "confusion_matrix", main[["run_id", "label", "tn", "fp", "fn", "tp"]])
    manufacturer = subgroup(preds, "Manufacturer")
    sitecode = subgroup(preds, "SiteCode")
    write_table(outdir, "manufacturer_subgroup_metrics", manufacturer)
    write_table(outdir, "sitecode_subgroup_metrics", sitecode)
    write_table(outdir, "sitecode035_subjects_predictions", preds[preds["SiteCode"].astype(str).eq("035")].copy())

    leakage = pd.concat([scanner_leakage(run_id, label, run_dir) for run_id, label, run_dir, _ in specs], ignore_index=True)
    latent = pd.concat([latent_qc(run_id, label, run_dir) for run_id, label, run_dir, _ in specs], ignore_index=True)
    write_table(outdir, "scanner_manufacturer_leakage", leakage)
    write_table(outdir, "latent_information_qc", latent)
    write_recommendation(outdir, main, leakage, manufacturer, sitecode)

    command_log = {
        "script": rel(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": rel(candidate_run),
        "candidate_readout": rel(candidate_readout),
        "v51b_reference_run": rel(v51b_run),
        "v51b_reference_readout": rel(v51b_readout),
        "v51c4480_reference_run": rel(v51c4480_run),
        "v51c4480_reference_readout": rel(v51c4480_readout),
        "output_dir": rel(outdir),
        "read_only": True,
        "modified_tensor_metadata_ledger_config_or_model_outputs": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
