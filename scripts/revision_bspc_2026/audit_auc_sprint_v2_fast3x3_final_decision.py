#!/usr/bin/env python3
"""Final decision audit for AUC Sprint v2 FAST 3x3 candidates.

Read-only with respect to training artifacts: this script only reads completed
Stage A/Stage B outputs and writes audit tables in the sprint root.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPRINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_auc_sprint_v2_representation_fast3x3"
)
RUNS_ROOT = SPRINT_ROOT / "runs"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
COMPLETED_CANDIDATES = [
    "control_current_fast3x3",
    "manufacturer_balanced_sampler_fast3x3",
    "channel_dropout_fast3x3",
    "batch32_fast3x3",
]
ALL_PLANNED = [
    "control_current_fast3x3",
    "manufacturer_balanced_sampler_fast3x3",
    "diagnosis_manufacturer_balanced_sampler_fast3x3",
    "encoder_norm_fast3x3",
    "channel_dropout_fast3x3",
    "batch32_fast3x3",
]


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
    df.to_csv(SPRINT_ROOT / f"{stem}.csv", index=False)
    (SPRINT_ROOT / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def candidate_run_dir(candidate: str) -> Path:
    return RUNS_ROOT / candidate


def candidate_readout_dir(candidate: str) -> Path:
    return candidate_run_dir(candidate) / "classifier_only_readout"


def primary_pooled_row(candidate: str) -> Dict[str, Any]:
    readout = candidate_readout_dir(candidate)
    pooled = read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if not mask.any():
        raise ValueError(f"Missing primary pooled row for {candidate}")
    row = pooled.loc[mask].iloc[0].to_dict()
    out = {"candidate_id": candidate}
    out.update(row)
    return out


def primary_foldwise(candidate: str) -> pd.DataFrame:
    df = read_csv(candidate_readout_dir(candidate) / "classifier_sweep_foldwise_metrics.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    out = df.loc[mask].copy()
    out.insert(0, "candidate_id", candidate)
    return out


def primary_predictions(candidate: str) -> pd.DataFrame:
    df = read_csv(candidate_readout_dir(candidate) / "classifier_sweep_predictions.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    out = df.loc[mask].copy()
    out.insert(0, "candidate_id", candidate)
    return out


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([safe_div(tp, tp + fn), safe_div(tn, tn + fp)])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "cn_fp_rate": safe_div(fp, fp + tn),
        "ad_fn_rate": safe_div(fn, fn + tp),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def manufacturer_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (candidate, manufacturer), sub in predictions.groupby(["candidate_id", "Manufacturer"], dropna=False):
        row = {"candidate_id": candidate, "Manufacturer": manufacturer}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["candidate_id", "Manufacturer"]).reset_index(drop=True)
    return out


def manufacturer_robustness(manufacturer: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate, sub in manufacturer.groupby("candidate_id", dropna=False):
        philips = sub[sub["Manufacturer"].astype(str).str.lower().eq("philips")]
        ge = sub[sub["Manufacturer"].astype(str).str.lower().eq("ge")]
        row: Dict[str, Any] = {
            "candidate_id": candidate,
            "min_manufacturer_balanced_accuracy": pd.to_numeric(sub["balanced_accuracy"], errors="coerce").min(),
            "max_manufacturer_cn_fp_rate": pd.to_numeric(sub["cn_fp_rate"], errors="coerce").max(),
            "max_manufacturer_ad_fn_rate": pd.to_numeric(sub["ad_fn_rate"], errors="coerce").max(),
        }
        row["philips_cn_fp_rate"] = float(philips["cn_fp_rate"].iloc[0]) if not philips.empty else np.nan
        row["ge_ad_fn_rate"] = float(ge["ad_fn_rate"].iloc[0]) if not ge.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_scanner_leakage(candidate: str) -> pd.DataFrame:
    run_dir = candidate_run_dir(candidate)
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("fold_*/fold_*_scanner_leakage_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["candidate_id"] = candidate
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        row["split"] = "test" if "_test_" in path.name else "train_dev"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["candidate_id", "split"], dropna=False)
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
    grouped["latent_minus_raw_mean"] = grouped["acc_site_latent_mean"] - grouped["acc_site_raw_mean"]
    return grouped.sort_values(["candidate_id", "split"]).reset_index(drop=True)


def file_tree_size(path: Path) -> Dict[str, Any]:
    file_count = 0
    total_bytes = 0
    newest_mtime = np.nan
    if path.exists():
        for root, _dirs, files in os.walk(path):
            for name in files:
                p = Path(root) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                file_count += 1
                total_bytes += int(st.st_size)
                newest_mtime = max(float(newest_mtime) if np.isfinite(newest_mtime) else 0.0, st.st_mtime)
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "newest_mtime_utc": datetime.fromtimestamp(newest_mtime, timezone.utc).isoformat() if np.isfinite(newest_mtime) else "",
    }


def stage_a_complete(run_dir: Path) -> bool:
    if not (run_dir / "run_config.json").exists():
        return False
    if not list(run_dir.glob("all_folds_metrics_MULTI*.csv")):
        return False
    for fold in [1, 2, 3]:
        fold_dir = run_dir / f"fold_{fold}"
        required = [
            fold_dir / f"vae_model_fold_{fold}.pt",
            fold_dir / "train_dev_subjects_fold.csv",
            fold_dir / "test_subjects_fold.csv",
            fold_dir / "vae_norm_params.joblib",
        ]
        if not all(p.exists() for p in required):
            return False
    return True


def stage_b_complete(readout_dir: Path) -> bool:
    required = [
        readout_dir / "classifier_sweep_pooled_metrics.csv",
        readout_dir / "classifier_sweep_foldwise_metrics.csv",
        readout_dir / "classifier_sweep_predictions.csv",
        readout_dir / "command_log.json",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    except Exception:
        return False
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    return bool(mask.any())


def storage_manifest() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate in ALL_PLANNED:
        run_dir = candidate_run_dir(candidate)
        readout = candidate_readout_dir(candidate)
        size = file_tree_size(run_dir)
        stage_a = stage_a_complete(run_dir)
        stage_b = stage_b_complete(readout)
        rows.append(
            {
                "candidate_id": candidate,
                "run_dir": str(run_dir),
                "readout_dir": str(readout),
                "stage_a_complete": stage_a,
                "stage_b_complete": stage_b,
                "completed_for_final_decision": candidate in COMPLETED_CANDIDATES and stage_a and stage_b,
                **size,
            }
        )
    return pd.DataFrame(rows)


def final_decision_table(pooled: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    metrics = pooled.merge(robustness, on="candidate_id", how="left", suffixes=("", "_robustness"))
    control = metrics[metrics["candidate_id"].eq("control_current_fast3x3")]
    if control.empty:
        raise ValueError("Control candidate is missing.")
    ref = control.iloc[0]
    for metric in ["auc", "pr_auc", "balanced_accuracy", "f1", "philips_cn_fp_rate", "ge_ad_fn_rate", "min_manufacturer_balanced_accuracy"]:
        metrics[f"delta_vs_control_{metric}"] = pd.to_numeric(metrics[metric], errors="coerce") - float(ref[metric])
    metrics["manufacturer_robustness_penalty"] = (
        (metrics["philips_cn_fp_rate"] > float(ref["philips_cn_fp_rate"]) + 1e-12)
        | (metrics["ge_ad_fn_rate"] > float(ref["ge_ad_fn_rate"]) + 1e-12)
        | (metrics["min_manufacturer_balanced_accuracy"] < float(ref["min_manufacturer_balanced_accuracy"]) - 1e-12)
    )
    metrics = metrics.sort_values(["auc", "pr_auc", "balanced_accuracy", "f1"], ascending=False).reset_index(drop=True)
    metrics.insert(0, "auc_rank", np.arange(1, len(metrics) + 1))
    decision = []
    for _, row in metrics.iterrows():
        cid = row["candidate_id"]
        if cid == "manufacturer_balanced_sampler_fast3x3":
            if row["auc"] > ref["auc"] and not row["manufacturer_robustness_penalty"]:
                decision.append("eligible_for_full_5x5_check")
            elif row["auc"] > ref["auc"]:
                decision.append("not_promoted_due_to_subgroup_penalty")
            else:
                decision.append("not_promoted_auc_below_control")
        elif cid == "control_current_fast3x3":
            decision.append("fast_control_reference")
        elif cid in {"channel_dropout_fast3x3", "batch32_fast3x3"}:
            if row["auc"] > ref["auc"] and row["pr_auc"] >= ref["pr_auc"] and not row["manufacturer_robustness_penalty"]:
                decision.append("possible_only_if_subgroup_review_supports")
            else:
                decision.append("not_promoted")
        else:
            decision.append("not_promoted")
    metrics["decision"] = decision
    keep = [
        "auc_rank",
        "candidate_id",
        "decision",
        "auc",
        "delta_vs_control_auc",
        "pr_auc",
        "delta_vs_control_pr_auc",
        "balanced_accuracy",
        "delta_vs_control_balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "delta_vs_control_f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "philips_cn_fp_rate",
        "delta_vs_control_philips_cn_fp_rate",
        "ge_ad_fn_rate",
        "delta_vs_control_ge_ad_fn_rate",
        "min_manufacturer_balanced_accuracy",
        "delta_vs_control_min_manufacturer_balanced_accuracy",
        "manufacturer_robustness_penalty",
    ]
    return metrics[keep]


def write_recommendation(decision: pd.DataFrame, scanner: pd.DataFrame, storage: pd.DataFrame) -> None:
    control = decision[decision["candidate_id"].eq("control_current_fast3x3")].iloc[0]
    mfr = decision[decision["candidate_id"].eq("manufacturer_balanced_sampler_fast3x3")]
    channel = decision[decision["candidate_id"].eq("channel_dropout_fast3x3")]
    batch32 = decision[decision["candidate_id"].eq("batch32_fast3x3")]
    best = decision.sort_values(["auc", "pr_auc", "balanced_accuracy", "f1"], ascending=False).iloc[0]

    promote_mfr = False
    if not mfr.empty:
        m = mfr.iloc[0]
        promote_mfr = bool(m["decision"] == "eligible_for_full_5x5_check")

    lines = [
        "# AUC Sprint v2 FAST 3x3 Final Decision",
        "",
        "## Scope",
        "",
        "- Ranking uses only Stage B classifier-only `logreg_l2`.",
        f"- Primary threshold strategy: `{PRIMARY_THRESHOLD}`.",
        "- Stage A dummy canonical LogReg metrics are not used for ranking.",
        "- No training was launched by this audit.",
        "- Tensor, metadata, ledger, and training artifact directories were not modified.",
        "",
        "## Result",
        "",
        f"Best FAST candidate by AUC is `{best['candidate_id']}` with AUC={best['auc']:.4f}, PR-AUC={best['pr_auc']:.4f}, BA={best['balanced_accuracy']:.4f}, F1={best['f1']:.4f}.",
        f"The FAST control has AUC={control['auc']:.4f}, PR-AUC={control['pr_auc']:.4f}, BA={control['balanced_accuracy']:.4f}, F1={control['f1']:.4f}.",
        "",
        "## Manufacturer-Balanced Sampler Promotion Decision",
        "",
    ]
    if mfr.empty:
        lines.append("`manufacturer_balanced_sampler_fast3x3` is missing from completed results, so it cannot be promoted.")
    elif promote_mfr:
        m = mfr.iloc[0]
        lines.append(
            "`manufacturer_balanced_sampler_fast3x3` should be promoted to one FULL 5x5 confirmation because it improves AUC over control and does not trigger the manufacturer robustness penalty used here."
        )
        lines.append(
            f"Delta vs control: AUC {m['delta_vs_control_auc']:+.4f}, PR-AUC {m['delta_vs_control_pr_auc']:+.4f}, BA {m['delta_vs_control_balanced_accuracy']:+.4f}, F1 {m['delta_vs_control_f1']:+.4f}."
        )
    else:
        m = mfr.iloc[0]
        lines.append(
            "`manufacturer_balanced_sampler_fast3x3` should not be promoted to FULL 5x5 under the pre-specified decision logic."
        )
        lines.append(
            f"Delta vs control: AUC {m['delta_vs_control_auc']:+.4f}, PR-AUC {m['delta_vs_control_pr_auc']:+.4f}, BA {m['delta_vs_control_balanced_accuracy']:+.4f}, F1 {m['delta_vs_control_f1']:+.4f}; manufacturer robustness penalty={m['manufacturer_robustness_penalty']}."
        )
    lines += [
        "",
        "## Channel Dropout And Batch Size Decisions",
        "",
    ]
    if not channel.empty:
        c = channel.iloc[0]
        lines.append(
            f"`channel_dropout_fast3x3` should not be promoted unless subgroup tables contradict the main metrics. Here its decision is `{c['decision']}` with AUC delta {c['delta_vs_control_auc']:+.4f} and PR-AUC delta {c['delta_vs_control_pr_auc']:+.4f}."
        )
    if not batch32.empty:
        b = batch32.iloc[0]
        lines.append(
            f"`batch32_fast3x3` should not be promoted unless subgroup tables contradict the main metrics. Here its decision is `{b['decision']}` with AUC delta {b['delta_vs_control_auc']:+.4f} and PR-AUC delta {b['delta_vs_control_pr_auc']:+.4f}."
        )
    lines += [
        "",
        "## Scanner Leakage",
        "",
        "Scanner leakage summaries are reported in `scanner_leakage_comparison.csv`. They should be interpreted as QC context, not as ranking metrics. A candidate with higher AUC but clearly worsened latent manufacturer predictability should be treated conservatively.",
        "",
        "## Storage",
        "",
        f"Completed final-decision candidates: {int(storage['completed_for_final_decision'].sum())}/{len(COMPLETED_CANDIDATES)}.",
        "See `storage_manifest.csv` for run/readout directories, file counts, and sizes.",
    ]
    (SPRINT_ROOT / "full_promotion_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    pooled_rows: List[Dict[str, Any]] = []
    foldwise_frames: List[pd.DataFrame] = []
    prediction_frames: List[pd.DataFrame] = []
    scanner_frames: List[pd.DataFrame] = []
    for candidate in COMPLETED_CANDIDATES:
        pooled_rows.append(primary_pooled_row(candidate))
        foldwise_frames.append(primary_foldwise(candidate))
        prediction_frames.append(primary_predictions(candidate))
        scanner_frames.append(load_scanner_leakage(candidate))

    pooled = pd.DataFrame(pooled_rows)
    foldwise = pd.concat(foldwise_frames, ignore_index=True, sort=False)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    manufacturer = manufacturer_subgroups(predictions)
    robustness = manufacturer_robustness(manufacturer)
    decision = final_decision_table(pooled, robustness)
    scanner = pd.concat(scanner_frames, ignore_index=True, sort=False) if scanner_frames else pd.DataFrame()
    storage = storage_manifest()

    write_table("final_fast_decision_table", decision)
    write_table("foldwise_candidate_comparison", foldwise)
    write_table("manufacturer_subgroup_comparison", manufacturer)
    write_table("scanner_leakage_comparison", scanner)
    write_table("storage_manifest", storage)
    write_recommendation(decision, scanner, storage)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "sprint_root": str(SPRINT_ROOT),
        "completed_candidates_requested": COMPLETED_CANDIDATES,
        "primary_model": PRIMARY_MODEL,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "stage_a_dummy_metrics_used_for_ranking": False,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "deleted_files": False,
        "outputs": [
            "final_fast_decision_table.csv",
            "final_fast_decision_table.md",
            "foldwise_candidate_comparison.csv",
            "foldwise_candidate_comparison.md",
            "manufacturer_subgroup_comparison.csv",
            "manufacturer_subgroup_comparison.md",
            "scanner_leakage_comparison.csv",
            "scanner_leakage_comparison.md",
            "storage_manifest.csv",
            "storage_manifest.md",
            "full_promotion_recommendation.md",
            "command_log_final_decision.json",
        ],
    }
    (SPRINT_ROOT / "command_log_final_decision.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Final decision audit complete.")
    print(decision[["candidate_id", "auc", "pr_auc", "balanced_accuracy", "f1", "decision"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
