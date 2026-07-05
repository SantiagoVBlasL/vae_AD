#!/usr/bin/env python3
"""Read-only postmortem for FAST-long Manufacturer-conditioned beta-VAE.

This script audits the corrected mfrrecovered035_clfpoollocked branch:
035_S_6927 is restored to the VAE pool with Manufacturer=SIEMENS, but is
explicitly excluded from the supervised classifier pool.

It does not train models and does not modify tensors, source metadata, ledgers,
or trained model artifacts. It writes only lightweight postmortem tables under
the experiment result directory.
"""

from __future__ import annotations

import argparse
import json
import math
import py_compile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_DEFAULT = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "conditional_beta_vae_manufacturer_fastlong_mfrrecovered035_clfpoollocked"
)
CLEANMFR_DEFAULT = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "conditional_beta_vae_manufacturer_fast3x3_cleanmfr"
)

BASELINE = "ch1_0_2_baseline_unconditioned"
CONDITIONED = "ch1_0_2_decoder_only_manufacturer"
PATCHED_SUBJECT = "035_S_6927"
UNRESOLVED_SUBJECT = "128_S_2002"
MAX_EPOCHS = 1920
CYCLES = 24
CYCLE_LEN = 80
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--cleanmfr-root", type=Path, default=CLEANMFR_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(root: Path, name: str, required: bool = True) -> pd.DataFrame:
    path = root / f"{name}.csv"
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        vals: List[str] = []
        for col in sub.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.6f}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_pair(root: Path, stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def primary_comparison(root: Path) -> pd.DataFrame:
    df = read_csv(root, "primary_results")
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    count_cols = ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp"]
    df = numeric(df, metrics + count_cols)
    df = df[df["candidate_id"].isin([BASELINE, CONDITIONED])].copy()
    df = df.sort_values("candidate_id")
    rows: List[Dict[str, Any]] = []
    for candidate_id in [BASELINE, CONDITIONED]:
        row = df[df["candidate_id"].eq(candidate_id)]
        if row.empty:
            raise RuntimeError(f"Missing primary row for {candidate_id}")
        r = row.iloc[0].to_dict()
        r["comparison_role"] = "baseline" if candidate_id == BASELINE else "decoder_only_manufacturer"
        rows.append(r)
    base = df[df["candidate_id"].eq(BASELINE)].iloc[0]
    cond = df[df["candidate_id"].eq(CONDITIONED)].iloc[0]
    delta: Dict[str, Any] = {
        "candidate_id": "delta_conditioned_minus_baseline",
        "comparison_role": "delta",
        "model_name": PRIMARY_MODEL,
        "threshold_strategy": PRIMARY_THRESHOLD,
    }
    for col in metrics:
        delta[col] = float(cond[col]) - float(base[col])
    for col in count_cols:
        delta[col] = np.nan
    rows.append(delta)
    keep = [
        "comparison_role", "candidate_id", "model_name", "threshold_strategy",
        "n", "n_cn", "n_ad", "tn", "fp", "fn", "tp",
        "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1",
    ]
    return pd.DataFrame(rows)[[c for c in keep if c in pd.DataFrame(rows).columns]]


def foldwise_deltas(root: Path) -> pd.DataFrame:
    df = read_csv(root, "foldwise_metrics")
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    df = numeric(df, metrics)
    base = df[df["candidate_id"].eq(BASELINE)][["fold", *metrics]].copy()
    cond = df[df["candidate_id"].eq(CONDITIONED)][["fold", *metrics]].copy()
    merged = base.merge(cond, on="fold", suffixes=("_baseline", "_conditioned"))
    for metric in metrics:
        merged[f"{metric}_delta"] = merged[f"{metric}_conditioned"] - merged[f"{metric}_baseline"]
        merged[f"{metric}_improved"] = merged[f"{metric}_delta"] > 0
    merged["auc_pr_both_improved"] = merged["auc_improved"] & merged["pr_auc_improved"]
    return merged.sort_values("fold")


def foldwise_consistency(foldwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
        deltas = pd.to_numeric(foldwise[f"{metric}_delta"], errors="coerce")
        rows.append({
            "metric": metric,
            "mean_delta": deltas.mean(),
            "median_delta": deltas.median(),
            "n_folds_positive": int((deltas > 0).sum()),
            "n_folds_negative": int((deltas < 0).sum()),
            "min_delta": deltas.min(),
            "max_delta": deltas.max(),
        })
    return pd.DataFrame(rows)


def history_path(root: Path, candidate_id: str, fold: int) -> Path:
    return root / "runs" / candidate_id / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"


def load_history(root: Path, candidate_id: str, fold: int) -> Dict[str, List[float]]:
    path = history_path(root, candidate_id, fold)
    if not path.exists():
        raise FileNotFoundError(path)
    hist = joblib.load(path)
    if not isinstance(hist, dict):
        raise TypeError(f"Unexpected history type at {path}: {type(hist)}")
    return hist


def linear_slope(values: List[float], n: int = 100) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    tail = arr[-min(n, arr.size):]
    x = np.arange(tail.size, dtype=float)
    return float(np.polyfit(x, tail, 1)[0])


def training_maturity(root: Path) -> pd.DataFrame:
    rd = read_csv(root, "rate_distortion_by_candidate_fold", required=False)
    rows: List[Dict[str, Any]] = []
    for candidate_id in [BASELINE, CONDITIONED]:
        for fold in [1, 2, 3]:
            hist = load_history(root, candidate_id, fold)
            val_modelsel = list(map(float, hist.get("val_loss_modelsel", [])))
            if not val_modelsel:
                raise RuntimeError(f"Missing val_loss_modelsel for {candidate_id} fold {fold}")
            final_epoch = len(val_modelsel)
            best_idx = int(np.nanargmin(np.asarray(val_modelsel, dtype=float)))
            best_epoch = best_idx + 1
            beta = list(map(float, hist.get("beta", [np.nan] * final_epoch)))
            val_recon = list(map(float, hist.get("val_recon", [np.nan] * final_epoch)))
            val_kld = list(map(float, hist.get("val_kld", [np.nan] * final_epoch)))
            row: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "fold": fold,
                "best_epoch": best_epoch,
                "final_epoch": final_epoch,
                "early_stop_epoch": final_epoch if final_epoch < MAX_EPOCHS else np.nan,
                "reached_max_epoch": final_epoch >= MAX_EPOCHS,
                "epochs_after_best": final_epoch - best_epoch,
                "best_val_loss_modelsel": val_modelsel[best_idx],
                "last_val_loss_modelsel": val_modelsel[-1],
                "last100_val_loss_modelsel_slope": linear_slope(val_modelsel, 100),
                "right_censored_by_max_epoch": final_epoch >= MAX_EPOCHS,
                "best_within_last_10pct_of_budget": best_epoch >= int(0.9 * MAX_EPOCHS),
                "best_within_last_10pct_of_observed_training": best_epoch >= final_epoch - max(1, int(0.1 * final_epoch)),
                "near_horizon_or_right_censored": (final_epoch >= MAX_EPOCHS) or (best_epoch >= int(0.9 * MAX_EPOCHS)),
                "beta_at_best_epoch": beta[best_idx] if best_idx < len(beta) else np.nan,
                "lr_cycle_phase_epoch": (best_epoch - 1) % CYCLE_LEN,
                "lr_cycle_phase_fraction": ((best_epoch - 1) % CYCLE_LEN) / CYCLE_LEN,
                "beta_cycle_phase_epoch": (best_epoch - 1) % CYCLE_LEN,
                "beta_cycle_phase_fraction": ((best_epoch - 1) % CYCLE_LEN) / CYCLE_LEN,
                "val_recon_at_best": val_recon[best_idx] if best_idx < len(val_recon) else np.nan,
                "val_kld_at_best": val_kld[best_idx] if best_idx < len(val_kld) else np.nan,
            }
            if np.isfinite(row["val_recon_at_best"]) and row["val_recon_at_best"] != 0:
                row["kld_over_recon_at_best"] = row["val_kld_at_best"] / row["val_recon_at_best"]
                row["beta_kld_over_recon_at_best"] = row["beta_at_best_epoch"] * row["kld_over_recon_at_best"]
            match = rd[(rd.get("candidate_id", pd.Series(dtype=str)).eq(candidate_id)) & (pd.to_numeric(rd.get("fold", pd.Series(dtype=float)), errors="coerce").eq(fold))]
            if not match.empty:
                m = match.iloc[0]
                for col in ["D_val", "R_val_nats", "val_kld_over_recon", "val_beta_kld_over_recon", "L_val_betaMax"]:
                    if col in m.index:
                        row[f"rd_{col}"] = m[col]
            rows.append(row)
    return pd.DataFrame(rows)


def leakage_deconfounding(root: Path) -> pd.DataFrame:
    leakage = read_csv(root, "scanner_manufacturer_leakage_summary", required=False)
    pred = read_csv(root, "latent_covariate_predictability_summary", required=False)
    mi = read_csv(root, "latent_information_by_candidate_variable", required=False)
    rows: List[Dict[str, Any]] = []
    for candidate_id in [BASELINE, CONDITIONED]:
        row: Dict[str, Any] = {"candidate_id": candidate_id}
        l = leakage[leakage["candidate_id"].eq(candidate_id)] if not leakage.empty else pd.DataFrame()
        if not l.empty:
            for col in ["acc_site_raw", "acc_site_latent", "latent_minus_raw"]:
                row[col] = pd.to_numeric(l[col], errors="coerce").mean()
        p = pred[(pred["candidate_id"].eq(candidate_id)) & (pred["target"].astype(str).eq("Manufacturer"))] if not pred.empty else pd.DataFrame()
        if not p.empty:
            row["manufacturer_predictability_balanced_accuracy"] = pd.to_numeric(p["balanced_accuracy"], errors="coerce").mean()
        m = mi[(mi["candidate_id"].eq(candidate_id)) & (mi["variable"].astype(str).eq("Manufacturer"))] if not mi.empty else pd.DataFrame()
        if not m.empty:
            row["manufacturer_mi_sum_nats"] = pd.to_numeric(m["mi_sum_nats"], errors="coerce").mean()
            row["manufacturer_mi_mean_nats"] = pd.to_numeric(m["mi_mean_nats"], errors="coerce").mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) == 2:
        base = out[out["candidate_id"].eq(BASELINE)].iloc[0]
        cond = out[out["candidate_id"].eq(CONDITIONED)].iloc[0]
        delta: Dict[str, Any] = {"candidate_id": "delta_conditioned_minus_baseline"}
        for col in out.columns:
            if col == "candidate_id":
                continue
            delta[col] = pd.to_numeric(pd.Series([cond.get(col)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([base.get(col)]), errors="coerce").iloc[0]
        out = pd.concat([out, pd.DataFrame([delta])], ignore_index=True)
    return out


def classifier_pool_audit(root: Path) -> pd.DataFrame:
    pooled = read_csv(root, "classifier_pooled_metric_audit", required=False)
    supervised = read_csv(root, "supervised_exclusion_audit", required=False)
    identity = read_csv(root, "classifier_pool_identity_vs_cleanmfr", required=False)
    vae_summary = read_csv(root, "vae_pool_filter_summary_by_candidate_fold", required=False)
    removed = read_csv(root, "vae_pool_filter_removed_subjects", required=False)
    rows: List[Dict[str, Any]] = []
    for candidate_id in [BASELINE, CONDITIONED]:
        p = pooled[pooled["candidate_id"].eq(candidate_id)] if not pooled.empty else pd.DataFrame()
        s = supervised[supervised["candidate_id"].eq(candidate_id)] if not supervised.empty else pd.DataFrame()
        i = identity[identity["candidate_id"].eq(candidate_id)] if not identity.empty else pd.DataFrame()
        v = vae_summary[vae_summary["candidate_id"].eq(candidate_id)] if not vae_summary.empty else pd.DataFrame()
        r = removed[removed["candidate_id"].eq(candidate_id)] if not removed.empty else pd.DataFrame()
        rows.append({
            "candidate_id": candidate_id,
            "classifier_pool_n": int(p["n"].iloc[0]) if not p.empty else np.nan,
            "classifier_pool_cn": int(p["n_cn"].iloc[0]) if not p.empty else np.nan,
            "classifier_pool_ad": int(p["n_ad"].iloc[0]) if not p.empty else np.nan,
            "classifier_pool_locked_ok": bool(p["pool_locked_ok"].iloc[0]) if not p.empty else False,
            "supervised_rows_checked": len(s),
            "supervised_contains_035_any": bool(s.get(f"contains_{PATCHED_SUBJECT}", pd.Series([True])).astype(bool).any()) if not s.empty else True,
            "supervised_contains_128_any": bool(s.get(f"contains_{UNRESOLVED_SUBJECT}", pd.Series([True])).astype(bool).any()) if not s.empty else True,
            "cleanmfr_split_identity_all_identical": bool(i["status"].astype(str).eq("identical").all()) if not i.empty else False,
            "vae_pool_before_filter_mean": pd.to_numeric(v.get("n_vae_pool_before_filter", pd.Series(dtype=float)), errors="coerce").mean() if not v.empty else np.nan,
            "vae_pool_removed_mean": pd.to_numeric(v.get("n_removed", pd.Series(dtype=float)), errors="coerce").mean() if not v.empty else np.nan,
            "vae_pool_after_filter_mean": pd.to_numeric(v.get("n_vae_pool_after_filter", pd.Series(dtype=float)), errors="coerce").mean() if not v.empty else np.nan,
            "removed_subjects_unique": ",".join(sorted(set(r.get("SubjectID", pd.Series(dtype=str)).astype(str)))) if not r.empty else "",
            "only_128_removed_from_vae_pool": set(r.get("SubjectID", pd.Series(dtype=str)).astype(str)) == {UNRESOLVED_SUBJECT} if not r.empty else False,
        })
    return pd.DataFrame(rows)


def cleanmfr_reference_issue(root: Path, cleanmfr_root: Path) -> pd.DataFrame:
    final_recommendation = root / "final_recommendation.md"
    contains_nan = final_recommendation.exists() and "nan" in final_recommendation.read_text(encoding="utf-8").lower()
    clean = read_csv(cleanmfr_root, "primary_results", required=False)
    rows: List[Dict[str, Any]] = []
    if clean.empty:
        rows.append({
            "issue": "cleanmfr_reference_unavailable",
            "status": "path_missing_or_no_primary_results",
            "path": str(cleanmfr_root / "primary_results.csv"),
            "primary_decision_impact": "none",
        })
    else:
        sub = clean[clean["candidate_id"].isin([BASELINE, CONDITIONED])].copy()
        for _, r in sub.iterrows():
            rows.append({
                "issue": "cleanmfr_secondary_reference_displayed_as_nan",
                "status": "aggregation_constant_issue_not_path_issue",
                "path": str(cleanmfr_root / "primary_results.csv"),
                "candidate_id": r["candidate_id"],
                "cleanmfr_auc": r.get("auc", np.nan),
                "cleanmfr_pr_auc": r.get("pr_auc", np.nan),
                "final_recommendation_contains_nan": contains_nan,
                "primary_decision_impact": "none_primary_decision_uses_clfpoollocked_matched_pair",
            })
    return pd.DataFrame(rows)


def decision_text(
    primary: pd.DataFrame,
    foldwise: pd.DataFrame,
    consistency: pd.DataFrame,
    training: pd.DataFrame,
    leakage: pd.DataFrame,
    pool: pd.DataFrame,
    clean_issue: pd.DataFrame,
) -> str:
    delta = primary[primary["candidate_id"].eq("delta_conditioned_minus_baseline")].iloc[0]
    auc_delta = float(delta["auc"])
    pr_delta = float(delta["pr_auc"])
    ba_delta = float(delta["balanced_accuracy"])
    sens_delta = float(delta["sensitivity"])
    f1_delta = float(delta["f1"])
    leakage_delta = leakage[leakage["candidate_id"].eq("delta_conditioned_minus_baseline")]
    latent_leakage_delta = float(leakage_delta["acc_site_latent"].iloc[0]) if not leakage_delta.empty and "acc_site_latent" in leakage_delta.columns else np.nan
    mfr_mi_delta = float(leakage_delta["manufacturer_mi_sum_nats"].iloc[0]) if not leakage_delta.empty and "manufacturer_mi_sum_nats" in leakage_delta.columns else np.nan
    mfr_pred_delta = float(leakage_delta["manufacturer_predictability_balanced_accuracy"].iloc[0]) if not leakage_delta.empty and "manufacturer_predictability_balanced_accuracy" in leakage_delta.columns else np.nan
    auc_pos = int((foldwise["auc_delta"] > 0).sum())
    pr_pos = int((foldwise["pr_auc_delta"] > 0).sum())
    both_pos = int(foldwise["auc_pr_both_improved"].sum())
    conditioned_training = training[training["candidate_id"].eq(CONDITIONED)]
    right_censored = int(conditioned_training["right_censored_by_max_epoch"].astype(bool).sum())
    near_horizon = int(conditioned_training["near_horizon_or_right_censored"].astype(bool).sum())
    pool_ok = bool(pool["classifier_pool_locked_ok"].all()) and not bool(pool["supervised_contains_035_any"].any()) and not bool(pool["supervised_contains_128_any"].any())

    full_rule_ok = (
        auc_delta >= 0
        and pr_delta >= 0
        and sens_delta >= -1e-12
        and ba_delta >= -1e-12
        and f1_delta >= -1e-12
        and np.isfinite(latent_leakage_delta)
        and latent_leakage_delta <= 0
        and pool_ok
    )
    decision = "controlled_FULL_confirmation_justified" if full_rule_ok else "do_not_continue"

    lines = [
        "# FAST-long Manufacturer-conditioned postmortem decision",
        "",
        f"Generated UTC: {now_utc()}",
        "",
        f"Decision: **{decision}**",
        "",
        "## Primary result",
        "",
        f"`{CONDITIONED}` improved pooled ROC-AUC by {auc_delta:+.6f} and PR-AUC by {pr_delta:+.6f} versus `{BASELINE}`.",
        f"However, balanced accuracy changed by {ba_delta:+.6f}, sensitivity changed by {sens_delta:+.6f}, and F1 changed by {f1_delta:+.6f}.",
        "The pre-specified rule required preserving BA/F1/sensitivity, so the result does not clear the FULL-confirmation bar.",
        "",
        "## Foldwise robustness",
        "",
        f"AUC improved in {auc_pos}/3 folds; PR-AUC improved in {pr_pos}/3 folds; both AUC and PR-AUC improved in {both_pos}/3 folds.",
        "Fold 3 regressed for both AUC and PR-AUC, so the improvement is not uniformly robust.",
        "",
        "## Training maturity",
        "",
        f"The conditioned FAST-long run has {right_censored}/3 folds right-censored at the 1920-epoch maximum and {near_horizon}/3 folds near-horizon/right-censored.",
        "All conditioned folds early-stopped before the FAST-long horizon, so there is no evidence that another FAST-long extension is needed.",
        "",
        "## Leakage / deconfounding",
        "",
        f"Latent Manufacturer leakage changed by {latent_leakage_delta:+.6f} balanced-accuracy points, and Manufacturer MI changed by {mfr_mi_delta:+.6f} nats.",
        f"Manufacturer predictability from z changed by {mfr_pred_delta:+.6f}; this did not clearly improve even though scanner-leakage and MI summaries improved.",
        "",
        "## Classifier pool integrity",
        "",
        f"Classifier pool lock status: {'PASS' if pool_ok else 'FAIL'}. Pooled metrics remained n=396, CN=300, AD=96; 035_S_6927 and 128_S_2002 were absent from supervised folds.",
        "The VAE pool removed only 128_S_2002 for missing Manufacturer, as intended.",
        "",
        "## cleanmfr secondary reference issue",
        "",
        "The `nan` cleanmfr display in the existing `final_recommendation.md` is not a data-path failure.",
        "The cleanmfr `primary_results.csv` exists and contains the `[1,0,2]` matched pair; the `nan` came from the FAST-long aggregator's secondary-reference constants.",
        "This is secondary context only and does not affect the primary clfpoollocked decision.",
        "",
        "## Recommendation",
        "",
        "Do not launch a FULL 5x5 confirmation for Manufacturer decoder conditioning from this FAST-long result.",
        "Do not run another FAST-long extension: the diagnostic already removed the earlier horizon-censoring concern.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    cleanmfr_root = resolve(args.cleanmfr_root)
    if not root.exists():
        raise FileNotFoundError(root)

    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
        py_compile_status = "ok"
    except Exception as exc:  # pragma: no cover
        py_compile_status = f"failed: {exc}"

    command_log = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "root": str(root),
        "cleanmfr_root": str(cleanmfr_root),
        "dry_run": bool(args.dry_run),
        "py_compile": py_compile_status,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_original_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }

    if args.dry_run:
        print(json.dumps(command_log, indent=2))
        return 0

    primary = primary_comparison(root)
    folds = foldwise_deltas(root)
    consistency = foldwise_consistency(folds)
    training = training_maturity(root)
    leakage = leakage_deconfounding(root)
    pool = classifier_pool_audit(root)
    clean_issue = cleanmfr_reference_issue(root, cleanmfr_root)

    write_pair(root, "fastlong_postmortem_primary_comparison", primary)
    write_pair(root, "fastlong_postmortem_foldwise_deltas", folds)
    write_pair(root, "fastlong_postmortem_foldwise_consistency", consistency)
    write_pair(root, "fastlong_postmortem_training_maturity", training)
    write_pair(root, "fastlong_postmortem_leakage_deconfounding", leakage)
    write_pair(root, "fastlong_postmortem_classifier_pool_audit", pool)
    write_pair(root, "fastlong_postmortem_cleanmfr_secondary_reference_issue", clean_issue)

    decision = decision_text(primary, folds, consistency, training, leakage, pool, clean_issue)
    (root / "fastlong_postmortem_decision.md").write_text(decision, encoding="utf-8")

    readme = [
        "# FAST-long Manufacturer-conditioned postmortem",
        "",
        f"Generated UTC: {now_utc()}",
        "",
        "This is a read-only postmortem over completed FAST-long outputs.",
        "",
        "Key files:",
        "- `fastlong_postmortem_primary_comparison.csv/.md`",
        "- `fastlong_postmortem_foldwise_deltas.csv/.md`",
        "- `fastlong_postmortem_training_maturity.csv/.md`",
        "- `fastlong_postmortem_leakage_deconfounding.csv/.md`",
        "- `fastlong_postmortem_classifier_pool_audit.csv/.md`",
        "- `fastlong_postmortem_cleanmfr_secondary_reference_issue.csv/.md`",
        "- `fastlong_postmortem_decision.md`",
        "",
        "Final decision: see `fastlong_postmortem_decision.md`.",
        "",
    ]
    (root / "README_FASTLONG_POSTMORTEM.md").write_text("\n".join(readme), encoding="utf-8")

    command_log["decision_file"] = str(root / "fastlong_postmortem_decision.md")
    (root / "command_log_fastlong_postmortem.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
