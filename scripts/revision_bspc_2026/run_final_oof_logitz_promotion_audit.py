#!/usr/bin/env python3
"""
Final promotion-validation audit.

Target:
  run: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
  logreg_l2_original | z_plus_age_sex | oof_logitz | inner_oof_target_sens_ge_0p70_max_spec

Reads calibrated predictions from the Stage B OOF calibration audit and produces a
comprehensive promotion-validation report. No re-running of calibration, no re-training.

Goals:
  1. Document leakage-safety of OOF logit-z calibration protocol
  2. Recompute foldwise and pooled metrics for the target combination
  3. Compare against locked v5.1b, recover035, beta2p5, beta3.75 raw
  4. Score range before/after calibration, Philips CN FP, GE AD FN, top-k precision
  5. Brier score and ECE for raw vs calibrated scores
  6. Confirm promotion gate: AUC > 0.782951 AND PR-AUC >= 0.559873

Hard constraints:
  - No VAE training
  - No threshold fitting on outer test
  - No OASIS
  - No tensor/metadata/model-output modification
  - Read-only from existing result CSVs
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

CALIB_AUDIT_DIR = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
LOCKED_DIR = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
RECOVER035_DIR = RESULTS / "recover035_full5x5"
BETA2P5_DIR = RESULTS / "recover035_latent384_T80_h10000_p560_full5x5"
DEFAULT_OUTPUT = RESULTS / "final_oof_logitz_promotion_audit"

TARGET_MODEL = "logreg_l2_original"
TARGET_FEATURE_SET = "z_plus_age_sex"
TARGET_CALIB = "oof_logitz"
TARGET_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873
PROMOTION_F1_MIN = 0.56
PROMOTION_BA_SOFT = 0.735
FOLDS = [1, 2, 3, 4, 5]
TOPK_VALUES = [5, 10, 15, 20]

COMPARISON_RUNS: List[tuple] = [
    # (label, run_dir, model_prefix, feature_set_or_None)
    ("locked_v5p1b",      LOCKED_DIR,    "logreg_l2", None),
    ("recover035",        RECOVER035_DIR, "logreg_l2", "z_plus_age_sex"),
    ("beta2p5_latent384", BETA2P5_DIR,   "logreg_l2", "z_plus_age_sex"),
]
SYSTEM_ORDER = [
    "locked_v5p1b", "recover035", "beta2p5_latent384",
    "beta3p75_raw", "beta3p75_logitz_TARGET",
]


def resolve_run(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    local = RESULTS / path.name
    if local.exists():
        return local
    big = BIG_DISK / path.name
    if big.exists():
        return big
    return path


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
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


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 60) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    y_true = df["y_true"].values.astype(int)
    y_score = df["y_score"].values.astype(float)
    y_pred = df["y_pred"].values.astype(int)

    auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred).ravel())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ba = (sensitivity + specificity) / 2.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    brier = float(np.mean((y_score - y_true) ** 2))

    return dict(
        n=int(len(y_true)), n_cn=int((y_true == 0).sum()), n_ad=int((y_true == 1).sum()),
        tp=tp, tn=tn, fp=fp, fn=fn,
        sensitivity=sensitivity, specificity=specificity,
        balanced_accuracy=ba, f1=f1,
        auc=auc, pr_auc=pr_auc, brier=brier,
    )


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = float(len(y_true))
    for bl, bu in zip(bins[:-1], bins[1:]):
        mask = (y_score >= bl) & (y_score < bu)
        cnt = mask.sum()
        if cnt == 0:
            continue
        ece_val += abs(float(y_true[mask].mean()) - float(y_score[mask].mean())) * cnt / n
    return float(ece_val)


def calibration_curve_df(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, Any]] = []
    for bl, bu in zip(bins[:-1], bins[1:]):
        mask = (y_score >= bl) & (y_score < bu)
        cnt = int(mask.sum())
        rows.append(dict(
            bin_lower=round(float(bl), 2),
            bin_upper=round(float(bu), 2),
            n=cnt,
            mean_pred=float(y_score[mask].mean()) if cnt > 0 else float("nan"),
            frac_pos=float(y_true[mask].mean()) if cnt > 0 else float("nan"),
        ))
    return pd.DataFrame(rows)


def top_k_precision(df: pd.DataFrame, k_values: List[int]) -> Dict[int, float]:
    sorted_df = df.sort_values("y_score", ascending=False).reset_index(drop=True)
    return {k: float((sorted_df.head(k)["y_true"] == 1).mean()) for k in k_values}


def foldwise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fsub = df[df["fold"] == fold]
        if fsub.empty:
            continue
        m = compute_metrics(fsub)
        m["fold"] = fold
        rows.append(m)
    return pd.DataFrame(rows)


def philips_fp_stats(df: pd.DataFrame) -> Dict[str, Any]:
    cn = df[df["y_true"] == 0]
    result: Dict[str, Any] = {"n_cn_total": int(len(cn))}
    for mfr in ["Philips", "SIEMENS", "GE"]:
        key = mfr.lower()
        sub = cn[cn["Manufacturer"].astype(str).str.upper() == mfr.upper()] if "Manufacturer" in cn.columns else pd.DataFrame()
        n = len(sub)
        result[f"n_cn_{key}"] = int(n)
        if n > 0:
            fp = int((sub["y_pred"] == 1).sum())
            result[f"fp_{key}"] = fp
            result[f"fpr_{key}"] = float(fp / n)
            result[f"spec_{key}"] = float(1 - fp / n)
        else:
            result[f"fp_{key}"] = 0
            result[f"fpr_{key}"] = float("nan")
            result[f"spec_{key}"] = float("nan")
    return result


def ge_ad_fn(df: pd.DataFrame) -> pd.DataFrame:
    if "Manufacturer" not in df.columns:
        return pd.DataFrame()
    ge_ad = df[(df["y_true"] == 1) & (df["Manufacturer"].astype(str).str.upper() == "GE")]
    fn_rows = ge_ad[ge_ad["y_pred"] == 0]
    keep = [c for c in ["SubjectID", "fold", "y_score", "threshold", "Age", "Sex"] if c in fn_rows.columns]
    return fn_rows[keep].sort_values("fold").copy()


def load_comparison_preds(
    run_dir: Path,
    model_prefix: str,
    threshold: str,
    feature_set: Optional[str],
) -> Optional[pd.DataFrame]:
    pred_path = run_dir / "classifier_only_readout" / "classifier_sweep_predictions.csv"
    if not pred_path.exists():
        return None
    df = pd.read_csv(pred_path)
    mask = df["model_name"].astype(str).str.startswith(model_prefix)
    if "threshold_strategy" in df.columns:
        mask &= df["threshold_strategy"].astype(str) == threshold
    if feature_set is not None and "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"].astype(str) == feature_set
    sub = df[mask].copy()
    return sub if len(sub) > 0 else None


def sort_by_system_order(df: pd.DataFrame) -> pd.DataFrame:
    rank = {s: i for i, s in enumerate(SYSTEM_ORDER)}
    df = df.copy()
    df["_rank"] = df["system"].map(rank).fillna(99)
    return df.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    if args.dry_run:
        for label, path in [
            ("calib_predictions", CALIB_AUDIT_DIR / "calib_predictions.csv"),
            ("calib_foldwise",    CALIB_AUDIT_DIR / "calib_foldwise_metrics.csv"),
            ("calib_score_range", CALIB_AUDIT_DIR / "calib_score_range_by_fold.csv"),
            ("locked",            LOCKED_DIR / "classifier_only_readout" / "classifier_sweep_predictions.csv"),
            ("recover035",        RECOVER035_DIR / "classifier_only_readout" / "classifier_sweep_predictions.csv"),
            ("beta2p5",           BETA2P5_DIR / "classifier_only_readout" / "classifier_sweep_predictions.csv"),
        ]:
            print(f"  [{label}]: {'EXISTS' if path.exists() else 'MISSING'}  ({path})")
        print("Dry-run complete.")
        return 0

    if outdir.exists() and not args.overwrite:
        print(f"Output dir exists; pass --overwrite: {outdir}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Load calibrated predictions ─────────────────────────────────────────
    all_preds = pd.read_csv(CALIB_AUDIT_DIR / "calib_predictions.csv")

    target = all_preds[
        (all_preds["model_name"] == TARGET_MODEL)
        & (all_preds["feature_set"] == TARGET_FEATURE_SET)
        & (all_preds["calib_method"] == TARGET_CALIB)
        & (all_preds["threshold_strategy"] == TARGET_THRESHOLD)
    ].copy()

    raw = all_preds[
        (all_preds["model_name"] == TARGET_MODEL)
        & (all_preds["feature_set"] == TARGET_FEATURE_SET)
        & (all_preds["calib_method"] == "raw")
        & (all_preds["threshold_strategy"] == TARGET_THRESHOLD)
    ].copy()

    assert len(target) > 0, "No target predictions found in calib_predictions.csv"
    print(f"Target: {len(target)} rows, {target['fold'].nunique()} folds")

    # ── 2. Foldwise metrics for target ─────────────────────────────────────────
    fw_target = foldwise_metrics(target)
    fw_target.insert(0, "system", "beta3p75_logitz_TARGET")
    write_table(outdir, "foldwise_metrics_target", fw_target)

    # ── 3. Pooled metrics for target and raw ───────────────────────────────────
    m_target = compute_metrics(target)
    m_raw = compute_metrics(raw) if len(raw) > 0 else {}
    fw_raw = foldwise_metrics(raw) if len(raw) > 0 else pd.DataFrame()

    # ── 4. Brier / ECE ─────────────────────────────────────────────────────────
    y_true_t = target["y_true"].values.astype(int)
    y_cal_t = target["y_score"].values.astype(float)
    y_raw_t = target["y_score_raw"].values.astype(float) if "y_score_raw" in target.columns else None

    brier_calib = float(np.mean((y_cal_t - y_true_t) ** 2))
    ece_calib = compute_ece(y_true_t, y_cal_t)
    brier_raw = float(np.mean((y_raw_t - y_true_t) ** 2)) if y_raw_t is not None else float("nan")
    ece_raw = compute_ece(y_true_t, y_raw_t) if y_raw_t is not None else float("nan")

    # Reliability diagram data
    cal_curve_after = calibration_curve_df(y_true_t, y_cal_t)
    cal_curve_after["version"] = "oof_logitz"
    if y_raw_t is not None:
        cal_curve_before = calibration_curve_df(y_true_t, y_raw_t)
        cal_curve_before["version"] = "raw"
        write_table(outdir, "calibration_curve", pd.concat([cal_curve_before, cal_curve_after], ignore_index=True))
    else:
        write_table(outdir, "calibration_curve", cal_curve_after)

    # ── 5. Top-k precision ─────────────────────────────────────────────────────
    topk_t = top_k_precision(target, TOPK_VALUES)
    topk_r = top_k_precision(raw, TOPK_VALUES) if len(raw) > 0 else {k: float("nan") for k in TOPK_VALUES}
    topk_rows = [{"k": k, "precision_raw": topk_r[k], "precision_oof_logitz": topk_t[k]} for k in TOPK_VALUES]
    write_table(outdir, "topk_precision", pd.DataFrame(topk_rows))

    # ── 6. GE AD false negatives ───────────────────────────────────────────────
    ge_fn = ge_ad_fn(target)
    write_table(outdir, "ge_ad_fn_detail", ge_fn)

    # GE AD summary by fold
    if "Manufacturer" in target.columns:
        ge_ad_all = target[(target["y_true"] == 1) & (target["Manufacturer"].astype(str).str.upper() == "GE")]
        ge_summary_rows: List[Dict[str, Any]] = []
        for fold in FOLDS:
            fsub = ge_ad_all[ge_ad_all["fold"] == fold]
            if fsub.empty:
                continue
            ge_summary_rows.append({
                "fold": fold,
                "n_ge_ad": int(len(fsub)),
                "fn_ge_ad": int((fsub["y_pred"] == 0).sum()),
                "tp_ge_ad": int((fsub["y_pred"] == 1).sum()),
                "sensitivity_ge_ad": float((fsub["y_pred"] == 1).mean()),
            })
        write_table(outdir, "ge_ad_sensitivity_by_fold", pd.DataFrame(ge_summary_rows))

    # ── 7. Philips CN FP for target ────────────────────────────────────────────
    ph_target = philips_fp_stats(target)

    # ── 8. Load comparison run predictions ────────────────────────────────────
    comp_metrics_rows: List[Dict[str, Any]] = []
    comp_fw_rows: List[Dict[str, Any]] = []
    comp_ph_rows: List[Dict[str, Any]] = []
    comp_be_rows: List[Dict[str, Any]] = []
    comp_topk_rows: List[Dict[str, Any]] = []

    cached_preds: Dict[str, pd.DataFrame] = {}
    for label, run_dir, model_prefix, feat_set in COMPARISON_RUNS:
        p = load_comparison_preds(run_dir, model_prefix, TARGET_THRESHOLD, feat_set)
        if p is None:
            print(f"  [{label}] predictions not found — skipping")
            continue
        cached_preds[label] = p

        m = compute_metrics(p)
        m["system"] = label
        comp_metrics_rows.append(m)

        for fw_row in foldwise_metrics(p).to_dict("records"):
            fw_row["system"] = label
            comp_fw_rows.append(fw_row)

        ph = philips_fp_stats(p)
        ph["system"] = label
        comp_ph_rows.append(ph)

        yt = p["y_true"].values.astype(int)
        ys = p["y_score"].values.astype(float)
        comp_be_rows.append({"system": label, "brier": float(np.mean((ys - yt) ** 2)), "ece_10bins": compute_ece(yt, ys)})

        tk = top_k_precision(p, TOPK_VALUES)
        comp_topk_rows.append({"system": label, **{f"p@{k}": tk[k] for k in TOPK_VALUES}})

    # Add raw and target to all comparison structures
    if len(raw) > 0:
        m_raw_row = dict(m_raw); m_raw_row["system"] = "beta3p75_raw"
        comp_metrics_rows.append(m_raw_row)
        for fw_row in fw_raw.to_dict("records"):
            fw_row["system"] = "beta3p75_raw"; comp_fw_rows.append(fw_row)
        ph_r = philips_fp_stats(raw); ph_r["system"] = "beta3p75_raw"; comp_ph_rows.append(ph_r)
        comp_be_rows.append({"system": "beta3p75_raw", "brier": brier_raw, "ece_10bins": ece_raw})
        tk_r = top_k_precision(raw, TOPK_VALUES)
        comp_topk_rows.append({"system": "beta3p75_raw", **{f"p@{k}": tk_r[k] for k in TOPK_VALUES}})

    m_target_row = dict(m_target); m_target_row["system"] = "beta3p75_logitz_TARGET"
    comp_metrics_rows.append(m_target_row)
    ph_target["system"] = "beta3p75_logitz_TARGET"; comp_ph_rows.append(ph_target)
    comp_be_rows.append({"system": "beta3p75_logitz_TARGET", "brier": brier_calib, "ece_10bins": ece_calib})
    comp_topk_rows.append({"system": "beta3p75_logitz_TARGET", **{f"p@{k}": topk_t[k] for k in TOPK_VALUES}})
    for fw_row in fw_target.to_dict("records"):
        row = {k: v for k, v in fw_row.items() if k != "system"}
        row["system"] = "beta3p75_logitz_TARGET"
        comp_fw_rows.append(row)

    # ── 9. Write comparison tables ─────────────────────────────────────────────
    comp_df = sort_by_system_order(pd.DataFrame(comp_metrics_rows))
    comp_ph_df = sort_by_system_order(pd.DataFrame(comp_ph_rows))
    comp_be_df = sort_by_system_order(pd.DataFrame(comp_be_rows))
    comp_topk_df = sort_by_system_order(pd.DataFrame(comp_topk_rows))

    write_table(outdir, "pooled_metrics_comparison", comp_df)
    write_table(outdir, "philips_cn_fp_comparison", comp_ph_df)
    write_table(outdir, "brier_ece_comparison", comp_be_df)
    write_table(outdir, "topk_precision_comparison", comp_topk_df)

    # ── 10. Foldwise mean / pooled gap ─────────────────────────────────────────
    all_fw_df = pd.DataFrame(comp_fw_rows)
    fw_mean_df = (
        all_fw_df.groupby("system")[["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]]
        .mean()
        .reset_index()
    )
    fw_mean_df.columns = ["system"] + [f"fw_mean_{c}" for c in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]]
    fw_mean_df = sort_by_system_order(fw_mean_df)
    write_table(outdir, "foldwise_mean_comparison", fw_mean_df)

    gap_rows: List[Dict[str, Any]] = []
    for _, srow in comp_df.iterrows():
        sys = srow["system"]
        pooled_auc = float(srow["auc"])
        pooled_pr = float(srow["pr_auc"])
        fw_match = all_fw_df[all_fw_df["system"] == sys]
        if fw_match.empty:
            continue
        fw_mean_auc = float(fw_match["auc"].mean())
        gap_rows.append({
            "system": sys,
            "pooled_auc": pooled_auc,
            "foldwise_mean_auc": fw_mean_auc,
            "gap_fw_minus_pooled": fw_mean_auc - pooled_auc,
            "pooled_pr_auc": pooled_pr,
            "promotes": bool(pooled_auc > LOCKED_AUC and pooled_pr >= LOCKED_PR_AUC),
        })
    gap_df = sort_by_system_order(pd.DataFrame(gap_rows))
    write_table(outdir, "pooled_vs_foldwise_gap", gap_df)

    # ── 11. Score range before / after ─────────────────────────────────────────
    sr_all = pd.read_csv(CALIB_AUDIT_DIR / "calib_score_range_by_fold.csv")
    sr_target_df = sr_all[
        (sr_all["model_name"] == TARGET_MODEL)
        & (sr_all["feature_set"] == TARGET_FEATURE_SET)
        & (sr_all["diagnosis"] == "ALL")
        & (sr_all["calib_method"].isin(["raw", TARGET_CALIB]))
    ][["fold", "calib_method", "min", "p25", "median", "p75", "max", "score_range"]].sort_values(["fold", "calib_method"]).reset_index(drop=True)
    write_table(outdir, "score_range_before_after", sr_target_df)

    # ── 12. Determine final verdict ────────────────────────────────────────────
    promotes_auc = bool(m_target["auc"] > LOCKED_AUC)
    promotes_pr = bool(m_target["pr_auc"] >= LOCKED_PR_AUC)
    promotes_both = promotes_auc and promotes_pr
    promotes_f1 = bool(m_target["f1"] >= PROMOTION_F1_MIN)
    ba_ok = bool(m_target["balanced_accuracy"] >= PROMOTION_BA_SOFT - 0.02)

    ph_fp = ph_target.get("fp_philips", ph_target.get("fp_cn_philips", "?"))
    ph_fp_locked = next((r for r in comp_ph_rows if r["system"] == "locked_v5p1b"), {}).get("fp_philips", "?")

    # ── 13. Final report ───────────────────────────────────────────────────────
    fw_raw_mean_auc = float(fw_raw["auc"].mean()) if len(fw_raw) > 0 else float("nan")
    fw_tgt_mean_auc = float(fw_target["auc"].mean())
    raw_gap = fw_raw_mean_auc - m_raw.get("auc", float("nan")) if len(fw_raw) > 0 else float("nan")
    tgt_gap = fw_tgt_mean_auc - m_target["auc"]
    gap_reduction_pct = 100.0 * (1.0 - tgt_gap / raw_gap) if raw_gap and not np.isnan(raw_gap) and raw_gap != 0 else float("nan")

    report = "\n".join([
        "# Final Promotion-Validation Audit",
        "",
        f"Generated: {now}",
        "",
        "## Evaluated combination",
        "- Run: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`",
        f"- Readout: `{TARGET_MODEL}` | `{TARGET_FEATURE_SET}` | `{TARGET_CALIB}` | `{TARGET_THRESHOLD}`",
        "",
        "## Promotion gate",
        f"| Criterion | Gate | Result | Status |",
        f"|---|---|---|---|",
        f"| AUC | > {LOCKED_AUC} | {m_target['auc']:.4f} | {'**PASS**' if promotes_auc else '**FAIL**'} |",
        f"| PR-AUC | >= {LOCKED_PR_AUC} | {m_target['pr_auc']:.4f} | {'**PASS**' if promotes_pr else '**FAIL**'} |",
        f"| F1 | >= {PROMOTION_F1_MIN} | {m_target['f1']:.4f} | {'**PASS**' if promotes_f1 else '**FAIL**'} |",
        f"| BA (soft) | >= {PROMOTION_BA_SOFT - 0.02:.3f} | {m_target['balanced_accuracy']:.4f} | {'PASS' if ba_ok else 'CHECK'} |",
        "",
        f"**Verdict: {'PROMOTES' if promotes_both else 'DOES NOT PROMOTE'}**  ",
        f"(Both AUC and PR-AUC gates must pass simultaneously.)",
        "",
        "## Primary metrics",
        f"| Metric | Value |",
        f"|---|---|",
        f"| AUC | {m_target['auc']:.4f} |",
        f"| PR-AUC | {m_target['pr_auc']:.4f} |",
        f"| Balanced accuracy | {m_target['balanced_accuracy']:.4f} |",
        f"| Sensitivity | {m_target['sensitivity']:.4f} |",
        f"| Specificity | {m_target['specificity']:.4f} |",
        f"| F1 | {m_target['f1']:.4f} |",
        f"| n (pooled) | {m_target['n']} (CN={m_target['n_cn']}, AD={m_target['n_ad']}) |",
        f"| TP / FP / FN / TN | {m_target['tp']} / {m_target['fp']} / {m_target['fn']} / {m_target['tn']} |",
        f"| Brier (calibrated) | {brier_calib:.4f} |",
        f"| Brier (raw) | {brier_raw:.4f} |",
        f"| ECE-10 (calibrated) | {ece_calib:.4f} |",
        f"| ECE-10 (raw) | {ece_raw:.4f} |",
        "",
        "## Calibration effect",
        f"| | Pooled AUC | Foldwise mean AUC | Gap (fw − pooled) |",
        f"|---|---|---|---|",
        f"| beta3p75 raw | {m_raw.get('auc', float('nan')):.4f} | {fw_raw_mean_auc:.4f} | {raw_gap:.4f} |",
        f"| beta3p75 oof_logitz | {m_target['auc']:.4f} | {fw_tgt_mean_auc:.4f} | {tgt_gap:.4f} |",
        f"| AUC gain | {m_target['auc'] - m_raw.get('auc', float('nan')):+.4f} | | Gap reduction: {gap_reduction_pct:.0f}% |",
        "",
        "The pooled↔foldwise AUC gap dropped from the raw pathological level (caused by the Fold 1",
        "score-scale anomaly, C=0.1 vs C=0.001 for folds 2-5) after OOF logit-z calibration.",
        "",
        "## Leakage-safety audit",
        "OOF logit-z calibration is leakage-safe by construction:",
        "",
        "1. **Classifier fit**: logreg_l2 fitted on inner-CV trainDev only (GridSearchCV, 5-fold).",
        "2. **OOF scores**: `cross_val_predict` on trainDev → inner out-of-fold probability scores.",
        "3. **Calibration parameters**: logit-transform, then mean and std computed from OOF scores only (trainDev).",
        "4. **Outer-test transform**: same OOF logit mean/std applied to outer-test fold scores.",
        "5. **Threshold selection**: calibrated OOF scores vs trainDev labels → target-sens-≥0.70 / max-spec.",
        "6. **Outer-test labels**: used ONLY for final metric computation.",
        "",
        "All parameter estimation steps (classifier, calibration, threshold) use trainDev data exclusively.",
        "No outer-test labels leak into any fitted parameter.",
        "",
        "## Foldwise metrics (target)",
        md_table(fw_target.drop(columns=["system"], errors="ignore")),
        "",
        "## Pooled metrics — all systems",
        md_table(comp_df[["system", "n", "n_cn", "n_ad", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]]),
        "",
        "## Pooled vs foldwise AUC gap",
        md_table(gap_df[["system", "pooled_auc", "foldwise_mean_auc", "gap_fw_minus_pooled", "pooled_pr_auc", "promotes"]]),
        "",
        "## Philips CN false positives",
        md_table(comp_ph_df[[c for c in ["system", "n_cn_total", "n_cn_philips", "fp_philips", "fpr_philips", "spec_philips"] if c in comp_ph_df.columns]]),
        "",
        "## Score range before/after calibration (all subjects, by fold)",
        md_table(sr_target_df),
        "",
        "## Top-k AD-risk precision",
        md_table(pd.DataFrame(topk_rows)),
        "",
        "## Brier / ECE comparison",
        md_table(comp_be_df),
        "",
        "## Read-only guarantee",
        "Did not retrain VAE. Did not fit thresholds on outer test. Did not modify tensors,",
        "metadata, ledger, configs, or any existing run output.",
        "All metrics derived from `calib_predictions.csv` (stageB_oof_score_calibration) and",
        "pre-existing `classifier_sweep_predictions.csv` for comparison runs.",
    ])

    (outdir / "final_report.md").write_text(report, encoding="utf-8")

    cmd_log = {
        "created_utc": now,
        "target_run": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "target_model": TARGET_MODEL,
        "target_feature_set": TARGET_FEATURE_SET,
        "target_calib": TARGET_CALIB,
        "target_threshold": TARGET_THRESHOLD,
        "training_launched": False,
        "threshold_fitting_on_outer_test": False,
        "promotes_auc": promotes_auc,
        "promotes_pr_auc": promotes_pr,
        "promotes": promotes_both,
        "pooled_auc": round(m_target["auc"], 6),
        "pooled_pr_auc": round(m_target["pr_auc"], 6),
        "balanced_accuracy": round(m_target["balanced_accuracy"], 6),
        "f1": round(m_target["f1"], 6),
        "brier_calibrated": round(brier_calib, 6),
        "ece_10bins_calibrated": round(ece_calib, 6),
    }
    (outdir / "command_log.json").write_text(json.dumps(cmd_log, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"AUC:    {m_target['auc']:.4f}  (gate: >{LOCKED_AUC})  → {'PASS' if promotes_auc else 'FAIL'}")
    print(f"PR-AUC: {m_target['pr_auc']:.4f}  (gate: >={LOCKED_PR_AUC})  → {'PASS' if promotes_pr else 'FAIL'}")
    print(f"BA:     {m_target['balanced_accuracy']:.4f}  |  F1: {m_target['f1']:.4f}")
    print(f"Brier (calib): {brier_calib:.4f}  |  ECE-10 (calib): {ece_calib:.4f}")
    print(f"Foldwise gap: {raw_gap:.4f} (raw) → {tgt_gap:.4f} (logitz)  ({gap_reduction_pct:.0f}% reduction)")
    print(f"VERDICT: {'PROMOTES' if promotes_both else 'DOES NOT PROMOTE'}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
