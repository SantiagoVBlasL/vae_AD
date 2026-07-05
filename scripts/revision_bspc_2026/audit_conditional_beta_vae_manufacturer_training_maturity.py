#!/usr/bin/env python3
"""Read-only training-maturity audit for Manufacturer-conditioned beta-VAE FAST runs.

This audit inspects completed FAST 3x3 Manufacturer-conditioning branches and
asks whether any conditional candidate was plausibly penalized by the 960-epoch
FAST horizon. It does not train and does not modify tensors, metadata, ledgers,
or existing model outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import py_compile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/revision_bspc_2026"
OUTPUT_ROOT = RESULTS_ROOT / "conditional_beta_vae_manufacturer_training_maturity_audit"

BRANCHES = {
    "cleanmfr": RESULTS_ROOT / "conditional_beta_vae_manufacturer_fast3x3_cleanmfr",
    "mfrrecovered035": RESULTS_ROOT / "conditional_beta_vae_manufacturer_fast3x3_mfrrecovered035",
    "mfrrecovered035_clfpoollocked": RESULTS_ROOT / "conditional_beta_vae_manufacturer_fast3x3_mfrrecovered035_clfpoollocked",
}

EXPECTED_CANDIDATES = [
    "ch1_baseline_unconditioned",
    "ch1_decoder_only_manufacturer",
    "ch1_0_2_baseline_unconditioned",
    "ch1_0_2_decoder_only_manufacturer",
]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FAST_MAX_EPOCHS = 960
FAST_CYCLE_LEN = 80
FAST_BETA_RAMP_FRACTION = 0.4
FAST_BETA_MAX = 2.5


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
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


def write_pair(root: Path, stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if a branch directory is absent.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def branch_manifest(branch_root: Path) -> pd.DataFrame:
    path = branch_root / "run_manifest.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "candidate_id" in df.columns:
            return df
    rows = []
    for candidate_id in EXPECTED_CANDIDATES:
        run_dir = branch_root / "runs" / candidate_id
        rows.append({
            "candidate_id": candidate_id,
            "channels": "[1, 0, 2]" if "ch1_0_2" in candidate_id else "[1]",
            "vae_conditioning_mode": "decoder_only" if "decoder_only" in candidate_id else "none",
            "vae_conditioning_vars": "manufacturer" if "decoder_only" in candidate_id else "none",
            "readout_feature_set": "z_plus_age_sex",
            "run_dir": str(run_dir),
            "readout_dir": str(run_dir / "classifier_only_readout_z_plus_age_sex"),
        })
    return pd.DataFrame(rows)


def baseline_for(candidate_id: str) -> Optional[str]:
    if "decoder_only" not in candidate_id:
        return None
    return "ch1_0_2_baseline_unconditioned" if candidate_id.startswith("ch1_0_2") else "ch1_baseline_unconditioned"


def phase_for_epoch(epoch: float) -> Dict[str, Any]:
    if not np.isfinite(epoch) or epoch <= 0:
        return {
            "cycle_position": np.nan,
            "lr_phase": "",
            "beta_phase": "",
            "beta_cycle_fraction": np.nan,
        }
    pos = int((int(epoch) - 1) % FAST_CYCLE_LEN) + 1
    frac = pos / FAST_CYCLE_LEN
    if pos <= 5:
        lr_phase = "near_restart"
    elif pos >= FAST_CYCLE_LEN - 5:
        lr_phase = "near_trough"
    elif frac <= 0.5:
        lr_phase = "cosine_descending_early_mid"
    else:
        lr_phase = "cosine_descending_late"
    beta_ramp_epochs = int(round(FAST_CYCLE_LEN * FAST_BETA_RAMP_FRACTION))
    if pos <= beta_ramp_epochs:
        beta_phase = "beta_ramp"
    else:
        beta_phase = "beta_plateau"
    return {
        "cycle_position": pos,
        "lr_phase": lr_phase,
        "beta_phase": beta_phase,
        "beta_cycle_fraction": frac,
    }


def last100_slope(rd: pd.DataFrame) -> float:
    if rd.empty or "L_val_betaMax" not in rd.columns or "epoch" not in rd.columns:
        return np.nan
    tail = rd[["epoch", "L_val_betaMax"]].dropna().tail(100)
    if len(tail) < 10:
        return np.nan
    x = tail["epoch"].to_numpy(dtype=float)
    y = tail["L_val_betaMax"].to_numpy(dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def read_stageb_foldwise(readout_dir: Path, candidate_id: str) -> pd.DataFrame:
    path = readout_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "readout_feature_set" not in df.columns:
        df["readout_feature_set"] = "z_plus_age_sex"
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        & df["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    )
    out = df.loc[mask].copy()
    out["candidate_id"] = candidate_id
    return out


def read_latent_info(branch_root: Path) -> pd.DataFrame:
    path = branch_root / "latent_information_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def latent_info_for(latent: pd.DataFrame, candidate_id: str, fold: int) -> Dict[str, Any]:
    if latent.empty:
        return {"active_units": np.nan, "frac_active": np.nan, "total_correlation_nats": np.nan}
    sub = latent[
        latent.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(candidate_id)
        & pd.to_numeric(latent.get("fold", pd.Series(dtype=float)), errors="coerce").eq(fold)
        & latent.get("latent_split", pd.Series(dtype=str)).astype(str).eq("test")
    ].copy()
    if sub.empty:
        sub = latent[
            latent.get("candidate_id", pd.Series(dtype=str)).astype(str).eq(candidate_id)
            & pd.to_numeric(latent.get("fold", pd.Series(dtype=float)), errors="coerce").eq(fold)
        ].copy()
    if sub.empty:
        return {"active_units": np.nan, "frac_active": np.nan, "total_correlation_nats": np.nan}
    if "variable" in sub.columns and sub["variable"].astype(str).eq("Y_target").any():
        sub = sub[sub["variable"].astype(str).eq("Y_target")]
    row = sub.iloc[0]
    return {
        "active_units": row.get("n_active", np.nan),
        "frac_active": row.get("frac_active", np.nan),
        "total_correlation_nats": row.get("total_correlation_nats", np.nan),
    }


def read_leakage(branch_root: Path) -> pd.DataFrame:
    path = branch_root / "scanner_manufacturer_leakage.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if {"acc_site_latent", "acc_site_raw"}.issubset(df.columns) and "latent_minus_raw" not in df.columns:
        df["latent_minus_raw"] = df["acc_site_latent"] - df["acc_site_raw"]
    return df


def leakage_lookup(leakage: pd.DataFrame) -> Dict[tuple, Dict[str, float]]:
    out: Dict[tuple, Dict[str, float]] = {}
    if leakage.empty:
        return out
    sub = leakage.copy()
    if "leakage_split" in sub.columns:
        sub = sub[sub["leakage_split"].astype(str).eq("test")]
    if "site_col" in sub.columns:
        sub = sub[sub["site_col"].astype(str).eq("Manufacturer")]
    for _, row in sub.iterrows():
        key = (str(row.get("candidate_id", "")), int(row.get("fold", -1)))
        out[key] = {
            "acc_site_raw": float(row.get("acc_site_raw", np.nan)),
            "acc_site_latent": float(row.get("acc_site_latent", np.nan)),
            "latent_minus_raw": float(row.get("latent_minus_raw", np.nan)),
        }
    return out


def audit_branch(branch_name: str, branch_root: Path) -> Dict[str, pd.DataFrame]:
    manifest = branch_manifest(branch_root)
    latent = read_latent_info(branch_root)
    leakage = read_leakage(branch_root)
    leak_by = leakage_lookup(leakage)

    stageb_frames = []
    for _, row in manifest.iterrows():
        stageb = read_stageb_foldwise(Path(str(row["readout_dir"])), str(row["candidate_id"]))
        if not stageb.empty:
            stageb["branch"] = branch_name
            stageb_frames.append(stageb)
    stageb_all = pd.concat(stageb_frames, ignore_index=True, sort=False) if stageb_frames else pd.DataFrame()
    stageb_by = {}
    if not stageb_all.empty:
        for _, row in stageb_all.iterrows():
            stageb_by[(str(row["candidate_id"]), int(row["fold"]))] = row.to_dict()

    rows: List[Dict[str, Any]] = []
    for _, mrow in manifest.iterrows():
        candidate_id = str(mrow["candidate_id"])
        run_dir = Path(str(mrow["run_dir"]))
        baseline_id = baseline_for(candidate_id)
        for fold in range(1, 4):
            rd_path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
            base_record: Dict[str, Any] = {
                "branch": branch_name,
                "candidate_id": candidate_id,
                "matched_baseline_candidate_id": baseline_id or "",
                "fold": fold,
                "channels": mrow.get("channels", ""),
                "vae_conditioning_mode": mrow.get("vae_conditioning_mode", ""),
                "vae_conditioning_vars": mrow.get("vae_conditioning_vars", ""),
                "rate_distortion_file": str(rd_path),
                "rate_distortion_exists": rd_path.exists(),
            }
            if not rd_path.exists():
                rows.append(base_record)
                continue
            rd = pd.read_csv(rd_path)
            if rd.empty or "L_val_betaMax" not in rd.columns:
                rows.append(base_record)
                continue
            best_idx = pd.to_numeric(rd["L_val_betaMax"], errors="coerce").idxmin()
            best = rd.loc[best_idx]
            best_epoch = int(best.get("epoch", best_idx + 1))
            final_epoch = int(pd.to_numeric(rd["epoch"], errors="coerce").max()) if "epoch" in rd.columns else len(rd)
            d_val = float(best.get("D_val", np.nan))
            r_val = float(best.get("R_val_nats", np.nan))
            kld_r = r_val / d_val if np.isfinite(r_val) and np.isfinite(d_val) and d_val else np.nan
            beta_kld_r = FAST_BETA_MAX * kld_r if np.isfinite(kld_r) else np.nan
            phase = phase_for_epoch(best_epoch)
            li = latent_info_for(latent, candidate_id, fold)
            stageb = stageb_by.get((candidate_id, fold), {})
            leak = leak_by.get((candidate_id, fold), {})
            base_leak = leak_by.get((baseline_id, fold), {}) if baseline_id else {}
            leakage_reduction = (
                base_leak.get("acc_site_latent", np.nan) - leak.get("acc_site_latent", np.nan)
                if baseline_id else np.nan
            )
            rows.append({
                **base_record,
                "best_epoch_by_val_beta_max": best_epoch,
                "final_epoch": final_epoch,
                "epochs_after_best": final_epoch - best_epoch,
                "best_epoch_within_last_10pct_of_960": bool(best_epoch >= int(0.9 * FAST_MAX_EPOCHS)),
                "reached_960": bool(final_epoch >= FAST_MAX_EPOCHS),
                "right_censored_or_near_horizon": bool(final_epoch >= FAST_MAX_EPOCHS and best_epoch >= int(0.9 * FAST_MAX_EPOCHS)),
                "last100_valL_betaMax_slope": last100_slope(rd),
                "still_improving_last100": bool(last100_slope(rd) < 0) if np.isfinite(last100_slope(rd)) else np.nan,
                "reached_960_and_still_improving_last100": bool(final_epoch >= FAST_MAX_EPOCHS and last100_slope(rd) < 0) if np.isfinite(last100_slope(rd)) else np.nan,
                "cycle_position_at_best": phase["cycle_position"],
                "lr_phase_at_best": phase["lr_phase"],
                "beta_phase_at_best": phase["beta_phase"],
                "beta_cycle_fraction_at_best": phase["beta_cycle_fraction"],
                "beta_at_best": float(best.get("beta", np.nan)),
                "D_val": d_val,
                "R_val_nats": r_val,
                "KLD_over_R": kld_r,
                "beta_KLD_over_R": beta_kld_r,
                "active_units": li["active_units"],
                "frac_active": li["frac_active"],
                "total_correlation_nats": li["total_correlation_nats"],
                "stageb_auc": stageb.get("auc", np.nan),
                "stageb_pr_auc": stageb.get("pr_auc", np.nan),
                "stageb_balanced_accuracy": stageb.get("balanced_accuracy", np.nan),
                "stageb_f1": stageb.get("f1", np.nan),
                "stageb_sensitivity": stageb.get("sensitivity", np.nan),
                "stageb_specificity": stageb.get("specificity", np.nan),
                "acc_site_raw": leak.get("acc_site_raw", np.nan),
                "acc_site_latent": leak.get("acc_site_latent", np.nan),
                "latent_minus_raw": leak.get("latent_minus_raw", np.nan),
                "matched_baseline_acc_site_latent": base_leak.get("acc_site_latent", np.nan),
                "leakage_reduction_vs_matched_baseline": leakage_reduction,
            })
    maturity = pd.DataFrame(rows)
    return {"maturity": maturity, "stageb": stageb_all, "leakage": leakage, "latent": latent}


def summarize_maturity(maturity: pd.DataFrame) -> pd.DataFrame:
    if maturity.empty:
        return pd.DataFrame()
    numeric_cols = [
        "best_epoch_by_val_beta_max",
        "final_epoch",
        "epochs_after_best",
        "last100_valL_betaMax_slope",
        "D_val",
        "R_val_nats",
        "KLD_over_R",
        "beta_KLD_over_R",
        "active_units",
        "total_correlation_nats",
        "stageb_auc",
        "stageb_pr_auc",
        "stageb_balanced_accuracy",
        "stageb_f1",
        "acc_site_latent",
        "leakage_reduction_vs_matched_baseline",
    ]
    for col in numeric_cols:
        if col in maturity.columns:
            maturity[col] = pd.to_numeric(maturity[col], errors="coerce")
    group_cols = ["branch", "candidate_id", "matched_baseline_candidate_id", "channels", "vae_conditioning_mode", "vae_conditioning_vars"]
    summary = maturity.groupby(group_cols, dropna=False).agg(
        n_folds=("fold", "count"),
        mean_best_epoch=("best_epoch_by_val_beta_max", "mean"),
        max_best_epoch=("best_epoch_by_val_beta_max", "max"),
        mean_final_epoch=("final_epoch", "mean"),
        n_reached_960=("reached_960", "sum"),
        n_best_in_last_10pct=("best_epoch_within_last_10pct_of_960", "sum"),
        n_right_censored_or_near_horizon=("right_censored_or_near_horizon", "sum"),
        mean_last100_slope=("last100_valL_betaMax_slope", "mean"),
        n_still_improving_last100=("still_improving_last100", "sum"),
        n_reached_960_and_still_improving_last100=("reached_960_and_still_improving_last100", "sum"),
        mean_D_val=("D_val", "mean"),
        mean_R_val_nats=("R_val_nats", "mean"),
        mean_KLD_over_R=("KLD_over_R", "mean"),
        mean_beta_KLD_over_R=("beta_KLD_over_R", "mean"),
        mean_active_units=("active_units", "mean"),
        mean_total_correlation_nats=("total_correlation_nats", "mean"),
        mean_stageb_auc=("stageb_auc", "mean"),
        mean_stageb_pr_auc=("stageb_pr_auc", "mean"),
        mean_stageb_balanced_accuracy=("stageb_balanced_accuracy", "mean"),
        mean_stageb_f1=("stageb_f1", "mean"),
        mean_acc_site_latent=("acc_site_latent", "mean"),
        mean_leakage_reduction_vs_matched_baseline=("leakage_reduction_vs_matched_baseline", "mean"),
    ).reset_index()
    return summary


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    for metric in ["mean_stageb_auc", "mean_stageb_pr_auc", "mean_stageb_balanced_accuracy", "mean_stageb_f1"]:
        out[f"delta_{metric}_vs_matched_baseline"] = np.nan
    for idx, row in out.iterrows():
        baseline_id = str(row.get("matched_baseline_candidate_id", ""))
        if not baseline_id:
            continue
        mask = out["branch"].astype(str).eq(str(row["branch"])) & out["candidate_id"].astype(str).eq(baseline_id)
        if not mask.any():
            continue
        base = out.loc[mask].iloc[0]
        for metric in ["mean_stageb_auc", "mean_stageb_pr_auc", "mean_stageb_balanced_accuracy", "mean_stageb_f1"]:
            out.at[idx, f"delta_{metric}_vs_matched_baseline"] = float(row.get(metric, np.nan)) - float(base.get(metric, np.nan))
    return out


def recommendation(summary: pd.DataFrame) -> str:
    lines = [
        "# Manufacturer-Conditioned FAST Training-Maturity Recommendation",
        "",
        f"Generated UTC: {now_utc()}",
        "",
        "Rule: recommend a FAST-long diagnostic only if the conditional candidate is close to the matched baseline in AUC/PR-AUC, reduces Manufacturer leakage, and shows horizon censoring or continuing validation improvement near epoch 960. Do not recommend FULL directly from this audit.",
        "",
    ]
    if summary.empty:
        lines.append("No maturity rows were available. No FAST-long recommendation can be made.")
        return "\n".join(lines) + "\n"
    cond = summary[summary["vae_conditioning_mode"].astype(str).eq("decoder_only")].copy()
    if cond.empty:
        lines.append("No Manufacturer-conditioned candidates found.")
        return "\n".join(lines) + "\n"
    decisions: List[Dict[str, Any]] = []
    for _, row in cond.iterrows():
        auc_delta = float(row.get("delta_mean_stageb_auc_vs_matched_baseline", np.nan))
        pr_delta = float(row.get("delta_mean_stageb_pr_auc_vs_matched_baseline", np.nan))
        close_perf = (
            np.isfinite(auc_delta)
            and np.isfinite(pr_delta)
            and auc_delta >= -0.02
            and pr_delta >= -0.02
        )
        leakage_reduced = float(row.get("mean_leakage_reduction_vs_matched_baseline", np.nan)) > 0.01
        near_horizon_improving = (
            int(row.get("n_reached_960_and_still_improving_last100", 0)) >= 1
        )
        maturity_signal = (
            int(row.get("n_right_censored_or_near_horizon", 0)) >= 1
            or near_horizon_improving
        )
        diagnostic_only = str(row.get("branch", "")) == "mfrrecovered035"
        recommend_fast_long = bool(close_perf and leakage_reduced and maturity_signal and not diagnostic_only)
        decisions.append({
            "branch": row["branch"],
            "candidate_id": row["candidate_id"],
            "diagnostic_only": diagnostic_only,
            "auc_delta": auc_delta,
            "pr_auc_delta": pr_delta,
            "mean_leakage_reduction": row.get("mean_leakage_reduction_vs_matched_baseline", np.nan),
            "n_reached_960": row.get("n_reached_960", np.nan),
            "n_right_censored_or_near_horizon": row.get("n_right_censored_or_near_horizon", np.nan),
            "n_still_improving_last100": row.get("n_still_improving_last100", np.nan),
            "n_reached_960_and_still_improving_last100": row.get("n_reached_960_and_still_improving_last100", np.nan),
            "recommend_fast_long": recommend_fast_long,
        })
    dec = pd.DataFrame(decisions)
    recommended = dec[dec["recommend_fast_long"]]
    if recommended.empty:
        lines.append("Decision: no Manufacturer-conditioned candidate satisfies all FAST-long criteria. Do not run FULL directly.")
    else:
        lines.append("Decision: the following candidates justify at most a FAST-long diagnostic, not FULL:")
        for _, row in recommended.iterrows():
            lines.append(
                f"- `{row['branch']}::{row['candidate_id']}`: "
                f"AUC delta {row['auc_delta']:.4f}, PR-AUC delta {row['pr_auc_delta']:.4f}, "
                f"leakage reduction {row['mean_leakage_reduction']:.4f}, "
                f"right-censored folds {row['n_right_censored_or_near_horizon']}."
            )
    lines.append("")
    lines.append("Screening decisions:")
    lines.append(md_table(dec))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
        py_compile_status = "ok"
    except Exception as exc:
        py_compile_status = f"failed: {exc}"

    missing = [name for name, path in BRANCHES.items() if not path.exists()]
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing branch directories: {missing}. Use --allow-missing to continue.")

    all_maturity: List[pd.DataFrame] = []
    all_stageb: List[pd.DataFrame] = []
    all_leakage: List[pd.DataFrame] = []
    for branch_name, branch_root in BRANCHES.items():
        if not branch_root.exists():
            continue
        result = audit_branch(branch_name, branch_root)
        if not result["maturity"].empty:
            all_maturity.append(result["maturity"])
        if not result["stageb"].empty:
            all_stageb.append(result["stageb"])
        if not result["leakage"].empty:
            leak = result["leakage"].copy()
            leak["branch"] = branch_name
            all_leakage.append(leak)

    maturity = pd.concat(all_maturity, ignore_index=True, sort=False) if all_maturity else pd.DataFrame()
    stageb = pd.concat(all_stageb, ignore_index=True, sort=False) if all_stageb else pd.DataFrame()
    leakage = pd.concat(all_leakage, ignore_index=True, sort=False) if all_leakage else pd.DataFrame()
    summary = add_baseline_deltas(summarize_maturity(maturity))

    write_pair(output_root, "candidate_fold_training_maturity", maturity)
    write_pair(output_root, "candidate_training_maturity_summary", summary)
    write_pair(output_root, "stageb_foldwise_primary_metrics", stageb)
    write_pair(output_root, "scanner_leakage_foldwise", leakage)

    (output_root / "final_recommendation.md").write_text(recommendation(summary), encoding="utf-8")
    readme = [
        "# Manufacturer-Conditioned beta-VAE Training-Maturity Audit",
        "",
        f"Generated UTC: {now_utc()}",
        "",
        "Branches audited:",
        *[f"- `{name}`: `{path}` ({'present' if path.exists() else 'missing'})" for name, path in BRANCHES.items()],
        "",
        "This is read-only. It inspects FAST 960-epoch training dynamics, Stage B primary metrics, and scanner/manufacturer leakage.",
        "",
        "Key outputs:",
        "- `candidate_fold_training_maturity.csv/.md`",
        "- `candidate_training_maturity_summary.csv/.md`",
        "- `stageb_foldwise_primary_metrics.csv/.md`",
        "- `scanner_leakage_foldwise.csv/.md`",
        "- `final_recommendation.md`",
    ]
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "output_root": str(output_root),
        "branches": {name: str(path) for name, path in BRANCHES.items()},
        "missing_branches": missing,
        "py_compile": py_compile_status,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    (output_root / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
