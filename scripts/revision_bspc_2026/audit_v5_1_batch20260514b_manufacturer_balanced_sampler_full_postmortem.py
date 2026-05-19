#!/usr/bin/env python3
"""Read-only post-mortem for manufacturer-balanced sampler FULL 5x5.

Compares the AUC Sprint v2 FAST 3x3 signal against the completed FULL 5x5
confirmation and writes a final manuscript decision.  Ranking uses only Stage B
classifier-only logreg_l2 at inner_oof_target_sens_ge_0p70_max_spec.
"""

from __future__ import annotations

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
OUTDIR = RESULTS / "adni_v5_1_batch20260514b_manufacturer_balanced_sampler_full_5x5_postmortem"

FAST_DIR = RESULTS / "adni_v5_1_batch20260514b_auc_sprint_v2_representation_fast3x3"
FAST_DECISION = FAST_DIR / "final_fast_decision_table.csv"
FAST_SCANNER = FAST_DIR / "scanner_leakage_comparison.csv"

CURRENT_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
CURRENT_READOUT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
SAMPLER_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_manufacturer_balanced_sampler_full_5x5"
SAMPLER_READOUT = SAMPLER_RUN / "classifier_only_readout"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"
METRICS = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]


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


def write_table(stem: str, df: pd.DataFrame) -> None:
    df.to_csv(OUTDIR / f"{stem}.csv", index=False)
    (OUTDIR / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate_readout(readout: Path, label: str) -> None:
    required = [
        readout / "classifier_sweep_pooled_metrics.csv",
        readout / "classifier_sweep_foldwise_metrics.csv",
        readout / "classifier_sweep_predictions.csv",
        readout / "classifier_sweep_thresholds_by_fold.csv",
        readout / "command_log.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing readout files:\n" + "\n".join(missing))
    log = json.loads((readout / "command_log.json").read_text(encoding="utf-8"))
    for key in ["tensor_modified", "metadata_modified", "ledger_modified"]:
        if log.get(key) not in (False, None):
            raise RuntimeError(f"{label}: command_log has {key}={log.get(key)}")
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if not mask.any():
        raise RuntimeError(f"{label}: missing primary {PRIMARY_MODEL}/{PRIMARY_THRESHOLD}")


def primary_pooled(readout: Path, run_id: str, label: str) -> Dict[str, Any]:
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out: Dict[str, Any] = {
        "run_id": run_id,
        "label": label,
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "accuracy", *METRICS, "predicted_ad_rate"]:
        out[col] = row.get(col, np.nan)
    return out


def fixed_pooled(readout: Path, run_id: str) -> Dict[str, Any]:
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    row = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(FIXED_THRESHOLD)
    ].iloc[0]
    out = {"run_id": run_id, "threshold_strategy": FIXED_THRESHOLD}
    for col in ["tn", "fp", "fn", "tp", *METRICS, "predicted_ad_rate"]:
        out[f"fixed_0p5_{col}"] = row.get(col, np.nan)
    return out


def foldwise(readout: Path, run_id: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    df.insert(1, "label", label)
    return df


def predictions(readout: Path, run_id: str, label: str) -> pd.DataFrame:
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


def subgroup_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (run_id, label, manufacturer), sub in pred.groupby(["run_id", "label", "Manufacturer"], dropna=False):
        row = {"run_id": run_id, "label": label, "Manufacturer": manufacturer}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["Manufacturer", "run_id"]).reset_index(drop=True)


def scanner_leakage(run_dir: Path, run_id: str, label: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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
    out = (
        raw.groupby(["run_id", "label", "split"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            acc_site_raw_mean=("acc_site_raw", "mean"),
            acc_site_raw_std=("acc_site_raw", "std"),
            acc_site_latent_mean=("acc_site_latent", "mean"),
            acc_site_latent_std=("acc_site_latent", "std"),
            chance_level_mean=("chance_level", "mean"),
            n_sites_mean=("n_sites", "mean"),
            n_samples_mean=("n_samples", "mean"),
        )
        .reset_index()
    )
    out["latent_minus_raw_mean"] = out["acc_site_latent_mean"] - out["acc_site_raw_mean"]
    return out


def add_current_deltas(df: pd.DataFrame, id_col: str = "run_id") -> pd.DataFrame:
    out = df.copy()
    ref = out[out[id_col].eq("current_locked_full")]
    if ref.empty:
        return out
    ref_row = ref.iloc[0]
    for col in METRICS + ["tn", "fp", "fn", "tp", "predicted_ad_rate"]:
        if col in out.columns:
            out[f"delta_vs_current_{col}"] = pd.to_numeric(out[col], errors="coerce") - float(ref_row[col])
    return out


def make_final_confirmation(current: Dict[str, Any], sampler: Dict[str, Any], subgroups: pd.DataFrame) -> pd.DataFrame:
    rows = [current, sampler]
    df = add_current_deltas(pd.DataFrame(rows))
    philips = subgroups[subgroups["Manufacturer"].astype(str).str.lower().eq("philips")]
    ge = subgroups[subgroups["Manufacturer"].astype(str).str.lower().eq("ge")]
    extras: Dict[str, Dict[str, float]] = {}
    for run_id, sub in philips.groupby("run_id"):
        extras.setdefault(run_id, {})["philips_cn_fp_rate"] = float(sub["cn_fp_rate"].iloc[0])
    for run_id, sub in ge.groupby("run_id"):
        extras.setdefault(run_id, {})["ge_ad_fn_rate"] = float(sub["ad_fn_rate"].iloc[0])
    df["philips_cn_fp_rate"] = df["run_id"].map(lambda x: extras.get(x, {}).get("philips_cn_fp_rate", np.nan))
    df["ge_ad_fn_rate"] = df["run_id"].map(lambda x: extras.get(x, {}).get("ge_ad_fn_rate", np.nan))
    ref = df[df["run_id"].eq("current_locked_full")].iloc[0]
    df["delta_vs_current_philips_cn_fp_rate"] = df["philips_cn_fp_rate"] - float(ref["philips_cn_fp_rate"])
    df["delta_vs_current_ge_ad_fn_rate"] = df["ge_ad_fn_rate"] - float(ref["ge_ad_fn_rate"])
    return df


def make_fast_to_full(full: pd.DataFrame) -> pd.DataFrame:
    fast = pd.read_csv(FAST_DECISION)
    mapping = [
        ("control_current_fast3x3", "current_locked_full", "FAST control -> locked current FULL"),
        ("manufacturer_balanced_sampler_fast3x3", "manufacturer_balanced_full", "FAST manufacturer-balanced -> FULL manufacturer-balanced"),
    ]
    rows: List[Dict[str, Any]] = []
    for fast_id, full_id, label in mapping:
        f = fast[fast["candidate_id"].eq(fast_id)].iloc[0]
        q = full[full["run_id"].eq(full_id)].iloc[0]
        row: Dict[str, Any] = {
            "comparison": label,
            "fast_candidate_id": fast_id,
            "full_run_id": full_id,
            "fast_auc": f["auc"],
            "full_auc": q["auc"],
            "full_minus_fast_auc": q["auc"] - f["auc"],
            "fast_pr_auc": f["pr_auc"],
            "full_pr_auc": q["pr_auc"],
            "full_minus_fast_pr_auc": q["pr_auc"] - f["pr_auc"],
            "fast_ba": f["balanced_accuracy"],
            "full_ba": q["balanced_accuracy"],
            "full_minus_fast_ba": q["balanced_accuracy"] - f["balanced_accuracy"],
            "fast_f1": f["f1"],
            "full_f1": q["f1"],
            "full_minus_fast_f1": q["f1"] - f["f1"],
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    control = out[out["full_run_id"].eq("current_locked_full")].iloc[0]
    sampler = out[out["full_run_id"].eq("manufacturer_balanced_full")].iloc[0]
    out["fast_delta_sampler_minus_control_auc"] = float(
        fast[fast["candidate_id"].eq("manufacturer_balanced_sampler_fast3x3")]["auc"].iloc[0]
        - fast[fast["candidate_id"].eq("control_current_fast3x3")]["auc"].iloc[0]
    )
    out["full_delta_sampler_minus_current_auc"] = sampler["full_auc"] - control["full_auc"]
    out["fast_delta_sampler_minus_control_pr_auc"] = float(
        fast[fast["candidate_id"].eq("manufacturer_balanced_sampler_fast3x3")]["pr_auc"].iloc[0]
        - fast[fast["candidate_id"].eq("control_current_fast3x3")]["pr_auc"].iloc[0]
    )
    out["full_delta_sampler_minus_current_pr_auc"] = sampler["full_pr_auc"] - control["full_pr_auc"]
    out["transfer_result"] = np.where(
        out["full_run_id"].eq("manufacturer_balanced_full"),
        "FAST signal failed to transfer to FULL ranking metrics",
        "reference",
    )
    return out


def foldwise_comparison(current: pd.DataFrame, sampler: pd.DataFrame) -> pd.DataFrame:
    cur = current.copy()
    sam = sampler.copy()
    keep = ["fold", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]
    merged = cur[keep].merge(sam[keep], on="fold", suffixes=("_current", "_sampler"))
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        merged[f"delta_sampler_minus_current_{metric}"] = merged[f"{metric}_sampler"] - merged[f"{metric}_current"]
    return merged


def scanner_comparison() -> pd.DataFrame:
    current = scanner_leakage(CURRENT_RUN, "current_locked_full", "current locked FULL [1,0,2]")
    sampler = scanner_leakage(SAMPLER_RUN, "manufacturer_balanced_full", "manufacturer-balanced sampler FULL [1,0,2]")
    full = pd.concat([current, sampler], ignore_index=True, sort=False)
    if full.empty:
        return full
    ref = full[full["run_id"].eq("current_locked_full")].set_index("split")
    deltas: List[float] = []
    for _, row in full.iterrows():
        if row["split"] in ref.index:
            deltas.append(row["acc_site_latent_mean"] - float(ref.loc[row["split"], "acc_site_latent_mean"]))
        else:
            deltas.append(np.nan)
    full["delta_vs_current_acc_site_latent_mean"] = deltas
    return full


def write_decision(full: pd.DataFrame, fast_full: pd.DataFrame, foldwise_df: pd.DataFrame, subgroup: pd.DataFrame, scanner: pd.DataFrame) -> None:
    cur = full[full["run_id"].eq("current_locked_full")].iloc[0]
    sam = full[full["run_id"].eq("manufacturer_balanced_full")].iloc[0]
    auc_delta = sam["auc"] - cur["auc"]
    pr_delta = sam["pr_auc"] - cur["pr_auc"]
    ba_delta = sam["balanced_accuracy"] - cur["balanced_accuracy"]
    f1_delta = sam["f1"] - cur["f1"]
    threshold_comment = (
        "The selected operating threshold moved the sampler to a more favorable confusion profile "
        "for sensitivity/specificity balance, but ROC-AUC and PR-AUC are threshold-independent ranking metrics. "
        "The sampler therefore improved the chosen operating point while worsening global AD-vs-CN score ordering."
    )
    lines = [
        "# Final Post-Mortem: Manufacturer-Balanced Sampler FULL 5x5",
        "",
        "## Decision",
        "",
        "`manufacturer_balanced_sampler_full_5x5` should NOT replace the locked manuscript model.",
        "",
        "The locked current FULL tanh `[1,0,2]` model remains the paper model.",
        "",
        "## Primary FULL Comparison",
        "",
        f"- Current locked FULL: AUC={cur['auc']:.4f}, PR-AUC={cur['pr_auc']:.4f}, BA={cur['balanced_accuracy']:.4f}, F1={cur['f1']:.4f}.",
        f"- Manufacturer-balanced FULL: AUC={sam['auc']:.4f}, PR-AUC={sam['pr_auc']:.4f}, BA={sam['balanced_accuracy']:.4f}, F1={sam['f1']:.4f}.",
        f"- Delta sampler minus current: AUC={auc_delta:+.4f}, PR-AUC={pr_delta:+.4f}, BA={ba_delta:+.4f}, F1={f1_delta:+.4f}.",
        "",
        "The sampler misses the promotion rule: AUC is below 0.794, PR-AUC decreases materially, and the ranking loss is not offset enough by the threshold operating point.",
        "",
        "## FAST-To-FULL Transfer",
        "",
        "FAST 3x3 promoted the sampler because it improved AUC and subgroup error rates over the FAST control. FULL 5x5 did not confirm this: the sampler falls from the locked FULL by AUC and PR-AUC, despite a higher BA/F1 at the selected threshold.",
        "",
        "## Why BA/F1 Improved While AUC/PR-AUC Worsened",
        "",
        threshold_comment,
        "",
        "Practically: the inner-OOF threshold selected for the sampler produced fewer threshold-level mistakes, but the probability distributions became less well ranked overall. For manuscript model selection, ROC-AUC/PR-AUC dominate because they are threshold-independent and less sensitive to one selected operating point.",
        "",
        "## Subgroup Notes",
        "",
        f"- Philips CN FP rate: current={cur['philips_cn_fp_rate']:.4f}, sampler={sam['philips_cn_fp_rate']:.4f}, delta={sam['delta_vs_current_philips_cn_fp_rate']:+.4f}.",
        f"- GE AD FN rate: current={cur['ge_ad_fn_rate']:.4f}, sampler={sam['ge_ad_fn_rate']:.4f}, delta={sam['delta_vs_current_ge_ad_fn_rate']:+.4f}.",
        "",
        "The sampler improves Philips CN false positives but worsens GE AD false negatives in FULL. This mixed subgroup behavior is not enough to override the AUC/PR-AUC degradation.",
        "",
        "## Scanner Leakage/QC",
        "",
    ]
    if scanner.empty:
        lines.append("Scanner leakage summaries were not available.")
    else:
        lines.append("Scanner leakage summaries are in `scanner_leakage_full_comparison.csv`. No promotion decision should be based on sampler BA/F1 alone without checking that latent manufacturer predictability did not worsen.")
    lines.extend(
        [
            "",
            "## Manuscript Decision",
            "",
            "Keep the current locked FULL tanh `[1,0,2]` model as the paper model. Document the manufacturer-balanced sampler as a negative confirmation: a plausible FAST 3x3 hypothesis did not survive FULL 5x5 confirmation. This supports the stop-micro-optimization decision and reduces cherry-picking risk.",
        ]
    )
    (OUTDIR / "final_manuscript_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [FAST_DECISION, CURRENT_READOUT, SAMPLER_READOUT]:
        require_file(path if path.is_file() else path / "classifier_sweep_pooled_metrics.csv")
    validate_readout(CURRENT_READOUT, "current locked FULL")
    validate_readout(SAMPLER_READOUT, "manufacturer-balanced FULL")
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    current_full = primary_pooled(CURRENT_READOUT, "current_locked_full", "current locked FULL tanh [1,0,2]")
    sampler_full = primary_pooled(SAMPLER_READOUT, "manufacturer_balanced_full", "manufacturer-balanced sampler FULL [1,0,2]")
    current_fixed = fixed_pooled(CURRENT_READOUT, "current_locked_full")
    sampler_fixed = fixed_pooled(SAMPLER_READOUT, "manufacturer_balanced_full")

    current_pred = predictions(CURRENT_READOUT, "current_locked_full", "current locked FULL tanh [1,0,2]")
    sampler_pred = predictions(SAMPLER_READOUT, "manufacturer_balanced_full", "manufacturer-balanced sampler FULL [1,0,2]")
    pred = pd.concat([current_pred, sampler_pred], ignore_index=True, sort=False)
    subgroups = subgroup_table(pred)
    final = make_final_confirmation(current_full, sampler_full, subgroups)
    fixed = pd.DataFrame([current_fixed, sampler_fixed])
    final = final.merge(fixed, on="run_id", how="left")
    final = add_current_deltas(final)

    fast_to_full = make_fast_to_full(final)
    current_fold = foldwise(CURRENT_READOUT, "current_locked_full", "current locked FULL tanh [1,0,2]")
    sampler_fold = foldwise(SAMPLER_READOUT, "manufacturer_balanced_full", "manufacturer-balanced sampler FULL [1,0,2]")
    fold_comp = foldwise_comparison(current_fold, sampler_fold)
    scanner = scanner_comparison()

    write_table("final_full_confirmation_table", final)
    write_table("fast_to_full_transfer_audit", fast_to_full)
    write_table("foldwise_full_comparison", fold_comp)
    write_table("subgroup_full_comparison", subgroups)
    write_table("scanner_leakage_full_comparison", scanner)
    write_decision(final, fast_to_full, fold_comp, subgroups, scanner)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_dir": str(OUTDIR),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "fast_decision_table": str(FAST_DECISION),
        "current_run": str(CURRENT_RUN),
        "current_readout": str(CURRENT_READOUT),
        "sampler_run": str(SAMPLER_RUN),
        "sampler_readout": str(SAMPLER_READOUT),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_results_modified": False,
        "decision": "manufacturer_balanced_sampler_should_not_replace_locked_model",
    }
    (OUTDIR / "command_log_final_full_postmortem.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote post-mortem to {OUTDIR}")
    print(final[["run_id", "auc", "pr_auc", "balanced_accuracy", "f1", "delta_vs_current_auc", "delta_vs_current_pr_auc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
