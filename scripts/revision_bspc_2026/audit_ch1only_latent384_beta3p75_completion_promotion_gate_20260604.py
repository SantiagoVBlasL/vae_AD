#!/usr/bin/env python3
"""Read-only completion and promotion-gate audit for ch1-only latent384 beta3.75.

This script reads completed ADNI run artifacts and writes an audit package. It
does not train, score, modify tensors, or modify model artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
OUT = ROOT / "ch1only_latent384_beta3p75_completion_promotion_gate_audit_20260604"

CAND_RUN = ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5"
CAND_OOF = ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration"
CAND_CFG = Path("configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.json")

REF_RUN = ROOT / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REF_OOF = ROOT / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
REF_CFG = Path("configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json")

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, child_prefix))
    else:
        out[prefix] = value
    return out


def stable_repr(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    md_path = path.with_suffix(".md")
    try:
        md_path.write_text(df.to_markdown(index=False) + "\n")
    except Exception:
        md_path.write_text(df.to_string(index=False) + "\n")


def first_existing(patterns: Iterable[Path]) -> Path | None:
    for path in patterns:
        if path.exists():
            return path
    return None


def stagea_metrics(run_dir: Path, label: str) -> pd.DataFrame:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    if not files:
        return pd.DataFrame([{"run": label, "status": "missing"}])
    df = pd.read_csv(files[0])
    df.insert(0, "run", label)
    df.insert(1, "source_file", str(files[0]))
    return df


def raw_pooled(run_dir: Path, label: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame([{"run": label, "status": "missing", "source": "classifier_sweep_pooled_metrics"}])
    df = pd.read_csv(path)
    df = df[(df["model_name"] == "logreg_l2") & (df["readout_feature_set"] == PRIMARY_FEATURES)].copy()
    df.insert(0, "run", label)
    df.insert(1, "stageb_source", "classifier_sweep_raw")
    df["calib_method"] = "raw_classifier_sweep"
    df["feature_set"] = df["readout_feature_set"]
    return df


def oof_pooled(oof_dir: Path, label: str) -> pd.DataFrame:
    path = oof_dir / "calib_pooled_metrics.csv"
    if not path.exists():
        return pd.DataFrame([{"run": label, "status": "missing", "source": "calib_pooled_metrics"}])
    df = pd.read_csv(path)
    df = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURES)
        & (df["calib_method"].isin(["raw", "oof_logitz", "oof_ecdf"]))
    ].copy()
    df.insert(0, "run", label)
    df.insert(1, "stageb_source", "oof_score_calibration")
    return df


def primary_row(oof_dir: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(oof_dir / "calib_pooled_metrics.csv")
    row = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURES)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    row.insert(0, "run", label)
    return row


def oof_foldwise(oof_dir: Path, label: str) -> pd.DataFrame:
    path = oof_dir / "calib_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame([{"run": label, "status": "missing", "source": "calib_foldwise_metrics"}])
    df = pd.read_csv(path)
    df = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURES)
        & (df["calib_method"].isin(["raw", "oof_logitz", "oof_ecdf"]))
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run", label)
    return df


def philips_fpr(oof_dir: Path, label: str) -> pd.DataFrame:
    path = oof_dir / "calib_philips_fpr_pooled.csv"
    if not path.exists():
        return pd.DataFrame([{"run": label, "status": "missing"}])
    df = pd.read_csv(path)
    df = df[
        (df["model_name"] == PRIMARY_MODEL)
        & (df["feature_set"] == PRIMARY_FEATURES)
        & (df["calib_method"] == PRIMARY_CALIB)
        & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run", label)
    return df


def completion_status(run_dir: Path, oof_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        checks = {
            "fold_dir": fold_dir,
            "vae_model": fold_dir / f"vae_model_fold_{fold}.pt",
            "vae_history": fold_dir / f"vae_train_history_fold_{fold}.joblib",
            "rate_distortion": fold_dir / f"fold_{fold}_rate_distortion.csv",
            "trainDev_latent_info": fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv",
            "test_latent_info": fold_dir / f"fold_{fold}_test_latent_info_summary.csv",
            "trainDev_scanner_leakage": fold_dir / f"fold_{fold}_scanner_leakage_summary.csv",
            "test_scanner_leakage": fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv",
            "stageA_logreg_predictions": fold_dir / "test_predictions_logreg.csv",
            "stageA_svm_predictions": fold_dir / "test_predictions_svm.csv",
            "latent_cache_trainDev": run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
            "latent_cache_test": run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_test_latent_mu.csv",
        }
        row = {"run": label, "fold": fold}
        for key, path in checks.items():
            row[key] = path.exists()
        row["dropout_manifest_optional"] = (fold_dir / "dropout_manifest.csv").exists()
        row["fold_complete"] = all(bool(row[key]) for key in checks)
        rows.append(row)
    global_checks = {
        "stageA_metrics": bool(sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))),
        "stageB_raw_pooled": (run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv").exists(),
        "stageB_raw_foldwise": (run_dir / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv").exists(),
        "oof_pooled": (oof_dir / "calib_pooled_metrics.csv").exists(),
        "oof_foldwise": (oof_dir / "calib_foldwise_metrics.csv").exists(),
        "oof_philips_fpr": (oof_dir / "calib_philips_fpr_pooled.csv").exists(),
    }
    rows.append({"run": label, "fold": "all", **global_checks, "fold_complete": all(global_checks.values())})
    return pd.DataFrame(rows)


def config_diff() -> pd.DataFrame:
    cand = flatten(read_json(CAND_CFG))
    ref = flatten(read_json(REF_CFG))
    rows: list[dict[str, Any]] = []
    for key in sorted(set(cand) | set(ref)):
        cand_v = cand.get(key, "<MISSING>")
        ref_v = ref.get(key, "<MISSING>")
        if stable_repr(cand_v) == stable_repr(ref_v):
            continue
        if key in {"parameters.channels_to_use", "selected_channel_names"}:
            cls = "allowed_scientific_channel_change"
        elif key.startswith("paths.") or key in {"run_name", "description"}:
            cls = "allowed_provenance_path_or_name"
        else:
            cls = "unexpected_scientific_or_runtime_difference"
        rows.append({"config_key": key, "reference_value": stable_repr(ref_v), "candidate_value": stable_repr(cand_v), "classification": cls})
    return pd.DataFrame(rows)


def rate_distortion(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not path.exists():
            rows.append({"run": label, "fold": fold, "status": "missing_rate_distortion"})
            continue
        df = pd.read_csv(path)
        best = df.loc[df["L_val_betaMax"].idxmin()]
        final = df.iloc[-1]
        beta_max = 3.75
        rows.append(
            {
                "run": label,
                "fold": fold,
                "best_epoch": int(best["epoch"]),
                "final_epoch": int(final["epoch"]),
                "best_D_val": float(best["D_val"]),
                "best_R_val_nats": float(best["R_val_nats"]),
                "best_R_val_bits": float(best["R_val_bits"]),
                "best_L_val_betaMax": float(best["L_val_betaMax"]),
                "best_beta_kld_over_D": float(beta_max * best["R_val_nats"] / best["D_val"]) if best["D_val"] else np.nan,
                "final_D_val": float(final["D_val"]),
                "final_R_val_bits": float(final["R_val_bits"]),
                "final_L_val_betaMax": float(final["L_val_betaMax"]),
            }
        )
    return pd.DataFrame(rows)


def latent_info(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split, file_stub in [("trainDev", "trainDev"), ("test", "test")]:
            path = run_dir / f"fold_{fold}" / f"fold_{fold}_{file_stub}_latent_info_summary.csv"
            if not path.exists():
                rows.append({"run": label, "fold": fold, "split": split, "status": "missing_latent_info"})
                continue
            df = pd.read_csv(path)
            by_var = {str(r["variable"]): r for _, r in df.iterrows()}
            y = by_var.get("Y_target")
            m = by_var.get("Manufacturer")
            rows.append(
                {
                    "run": label,
                    "fold": fold,
                    "split": split,
                    "mi_z_y_sum_nats": float(y["mi_sum_nats"]) if y is not None else np.nan,
                    "mi_z_manufacturer_sum_nats": float(m["mi_sum_nats"]) if m is not None else np.nan,
                    "mi_manufacturer_over_mi_y": float(m["mi_sum_nats"] / y["mi_sum_nats"]) if y is not None and m is not None and y["mi_sum_nats"] else np.nan,
                    "active_units": int(y["n_active"]) if y is not None else np.nan,
                    "frac_active": float(y["frac_active"]) if y is not None else np.nan,
                    "total_correlation_nats": float(y["total_correlation_nats"]) if y is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def scanner_leakage(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        for split, filename in [("trainDev", f"fold_{fold}_scanner_leakage_summary.csv"), ("test", f"fold_{fold}_test_scanner_leakage_summary.csv")]:
            path = run_dir / f"fold_{fold}" / filename
            if not path.exists():
                rows.append({"run": label, "fold": fold, "split": split, "status": "missing_scanner_leakage"})
                continue
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict()
            row.update({"run": label, "fold": fold, "split": split})
            row["latent_minus_raw"] = row.get("acc_site_latent", np.nan) - row.get("acc_site_raw", np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "acc_site_latent" in out.columns:
        summary = (
            out.groupby(["run", "split"], dropna=False)[["acc_site_raw", "acc_site_latent", "latent_minus_raw"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        summary["fold"] = "mean"
        out = pd.concat([out, summary], ignore_index=True, sort=False)
    return out


def make_primary_gate(cand_primary: pd.DataFrame, ref_primary: pd.DataFrame, fpr: pd.DataFrame, leakage: pd.DataFrame) -> pd.DataFrame:
    cand = cand_primary.iloc[0]
    ref = ref_primary.iloc[0]
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "reference": ref.get(metric, np.nan),
                "candidate": cand.get(metric, np.nan),
                "delta_candidate_minus_reference": cand.get(metric, np.nan) - ref.get(metric, np.nan),
            }
        )
    phil = fpr[fpr["manufacturer"].astype(str).str.lower() == "philips"]
    if len(phil) == 2:
        ref_fpr = phil[phil["run"] == "promoted_reference"]["fpr_cn_pooled"].iloc[0]
        cand_fpr = phil[phil["run"] == "ch1only_candidate"]["fpr_cn_pooled"].iloc[0]
        rows.append({"metric": "philips_cn_fpr", "reference": ref_fpr, "candidate": cand_fpr, "delta_candidate_minus_reference": cand_fpr - ref_fpr})
    test_mean = leakage[(leakage["fold"].astype(str) == "mean") & (leakage["split"] == "test")]
    if len(test_mean) == 2:
        ref_leak = test_mean[test_mean["run"] == "promoted_reference"]["acc_site_latent"].iloc[0]
        cand_leak = test_mean[test_mean["run"] == "ch1only_candidate"]["acc_site_latent"].iloc[0]
        rows.append({"metric": "mean_test_latent_scanner_leakage_acc", "reference": ref_leak, "candidate": cand_leak, "delta_candidate_minus_reference": cand_leak - ref_leak})
    out = pd.DataFrame(rows)
    gate_auc = float(cand["auc"]) >= PROMOTED_AUC
    gate_pr = float(cand["pr_auc"]) >= PROMOTED_PR_AUC
    phil_row = out[out["metric"] == "philips_cn_fpr"]
    gate_phil = bool(len(phil_row) and float(phil_row["candidate"].iloc[0]) <= float(phil_row["reference"].iloc[0]))
    leak_row = out[out["metric"] == "mean_test_latent_scanner_leakage_acc"]
    gate_leak = bool(len(leak_row) and float(leak_row["candidate"].iloc[0]) <= float(leak_row["reference"].iloc[0]))
    out["gate_auc_not_worse"] = gate_auc
    out["gate_pr_auc_not_worse"] = gate_pr
    out["gate_philips_fpr_not_worse"] = gate_phil
    out["gate_scanner_leakage_not_worse"] = gate_leak
    out["promotion_gate_pass"] = gate_auc and gate_pr and gate_phil and gate_leak
    return out


def oasis_status() -> pd.DataFrame:
    patterns = [
        "*ch1only*oasis*",
        "*oasis*ch1only*",
        "*recover035_ch1only*external*",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(ROOT.glob(pattern))
    rows = [{"artifact_query": p, "path": str(path), "exists": path.exists()} for p in patterns for path in []]
    if found:
        rows.extend({"artifact_query": "candidate_oasis_like_artifact", "path": str(path), "exists": True} for path in sorted(set(found)))
    else:
        rows.append(
            {
                "artifact_query": "candidate_oasis_like_artifact",
                "path": "",
                "exists": False,
                "status": "No OASIS scoring artifacts were found for recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.",
            }
        )
    return pd.DataFrame(rows)


def write_text_outputs(primary_gate: pd.DataFrame, cfgdiff: pd.DataFrame, oasis: pd.DataFrame) -> None:
    passed = bool(primary_gate["promotion_gate_pass"].iloc[0])
    unexpected = cfgdiff[cfgdiff["classification"] == "unexpected_scientific_or_runtime_difference"]
    decision = "promote" if passed else "parsimonious_sensitivity_only"
    cand_auc = primary_gate.loc[primary_gate["metric"] == "auc", "candidate"].iloc[0]
    cand_pr = primary_gate.loc[primary_gate["metric"] == "pr_auc", "candidate"].iloc[0]
    cand_ba = primary_gate.loc[primary_gate["metric"] == "balanced_accuracy", "candidate"].iloc[0]
    cand_f1 = primary_gate.loc[primary_gate["metric"] == "f1", "candidate"].iloc[0]
    cand_sens = primary_gate.loc[primary_gate["metric"] == "sensitivity", "candidate"].iloc[0]
    cand_spec = primary_gate.loc[primary_gate["metric"] == "specificity", "candidate"].iloc[0]
    fpr_row = primary_gate[primary_gate["metric"] == "philips_cn_fpr"]
    leak_row = primary_gate[primary_gate["metric"] == "mean_test_latent_scanner_leakage_acc"]
    fpr_txt = "not available"
    if not fpr_row.empty:
        fpr_txt = f"{float(fpr_row['candidate'].iloc[0]):.4f} vs reference {float(fpr_row['reference'].iloc[0]):.4f}"
    leak_txt = "not available"
    if not leak_row.empty:
        leak_txt = f"{float(leak_row['candidate'].iloc[0]):.4f} vs reference {float(leak_row['reference'].iloc[0]):.4f}"
    oasis_txt = "No candidate OASIS scoring artifacts were found." if not bool(oasis["exists"].any()) else "Candidate OASIS scoring artifacts were found; see oasis_artifact_status.csv."

    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# ch1-only latent384 beta3.75 completion and promotion-gate audit",
                "",
                f"Generated: {now_iso()}",
                "",
                "This read-only audit compares the ch1-only parsimonious branch against the promoted [1,0,2] latent384 beta3.75 reference.",
                "",
                "Primary convention: `logreg_l2_original / z_plus_age_sex / oof_ecdf / inner_oof_target_sens_ge_0p70_max_spec`.",
                "",
                f"Decision: **{decision}**.",
                "",
                f"Candidate primary metrics: AUC={cand_auc:.6f}, PR-AUC={cand_pr:.6f}, BA={cand_ba:.6f}, Sens={cand_sens:.6f}, Spec={cand_spec:.6f}, F1={cand_f1:.6f}.",
                f"Philips CN FPR: {fpr_txt}.",
                f"Mean test latent scanner leakage accuracy: {leak_txt}.",
                oasis_txt,
            ]
        )
        + "\n"
    )
    (OUT / "final_decision.md").write_text(
        "\n".join(
            [
                "# Final Decision",
                "",
                f"Decision: **{decision}**.",
                "",
                "Promotion gate required AUC and PR-AUC not worse than the promoted reference, plus Philips CN FPR and scanner/manufacturer leakage not worse.",
                "",
                "- AUC/PR-AUC gate: see `primary_promotion_gate_table.csv`.",
                "- Philips CN FPR gate: see `philips_cn_fpr_comparison.csv`.",
                "- Scanner/manufacturer leakage gate: see `scanner_leakage_comparison.csv`.",
                "",
                "The candidate is retained only as a parsimonious/channel sensitivity if any gate fails.",
                "",
                "Config guardrail:",
                "- Expected scientific difference is `parameters.channels_to_use [1,0,2] -> [1]` with matching selected channel-name provenance.",
                f"- Unexpected config differences found: {len(unexpected)}.",
            ]
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only; do not write audit outputs.")
    args = parser.parse_args()

    required = [CAND_RUN, CAND_OOF, CAND_CFG, REF_RUN, REF_OOF, REF_CFG]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    if args.dry_run:
        print("dry_run_ok")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = [{"timestamp": now_iso(), "argv": sys.argv, "cwd": str(Path.cwd())}]

    completion = pd.concat(
        [
            completion_status(CAND_RUN, CAND_OOF, "ch1only_candidate"),
            completion_status(REF_RUN, REF_OOF, "promoted_reference"),
        ],
        ignore_index=True,
        sort=False,
    )
    cfgdiff = config_diff()
    stagea = pd.concat([stagea_metrics(CAND_RUN, "ch1only_candidate"), stagea_metrics(REF_RUN, "promoted_reference")], ignore_index=True, sort=False)
    raw_oof = pd.concat(
        [
            raw_pooled(CAND_RUN, "ch1only_candidate"),
            raw_pooled(REF_RUN, "promoted_reference"),
            oof_pooled(CAND_OOF, "ch1only_candidate"),
            oof_pooled(REF_OOF, "promoted_reference"),
        ],
        ignore_index=True,
        sort=False,
    )
    foldwise = pd.concat([oof_foldwise(CAND_OOF, "ch1only_candidate"), oof_foldwise(REF_OOF, "promoted_reference")], ignore_index=True, sort=False)
    fpr = pd.concat([philips_fpr(CAND_OOF, "ch1only_candidate"), philips_fpr(REF_OOF, "promoted_reference")], ignore_index=True, sort=False)
    rd = pd.concat([rate_distortion(CAND_RUN, "ch1only_candidate"), rate_distortion(REF_RUN, "promoted_reference")], ignore_index=True, sort=False)
    li = pd.concat([latent_info(CAND_RUN, "ch1only_candidate"), latent_info(REF_RUN, "promoted_reference")], ignore_index=True, sort=False)
    vae_qc = rd.merge(li[li["split"] == "test"], on=["run", "fold"], how="left")
    leakage = pd.concat([scanner_leakage(CAND_RUN, "ch1only_candidate"), scanner_leakage(REF_RUN, "promoted_reference")], ignore_index=True, sort=False)
    cand_primary = primary_row(CAND_OOF, "ch1only_candidate")
    ref_primary = primary_row(REF_OOF, "promoted_reference")
    primary_gate = make_primary_gate(cand_primary, ref_primary, fpr, leakage)
    oasis = oasis_status()

    outputs = {
        "completion_status.csv": completion,
        "config_diff.csv": cfgdiff,
        "stagea_metrics.csv": stagea,
        "stageb_raw_oof_comparison.csv": raw_oof,
        "foldwise_metrics.csv": foldwise,
        "philips_cn_fpr_comparison.csv": fpr,
        "scanner_leakage_comparison.csv": leakage,
        "vae_qc_comparison.csv": vae_qc,
        "primary_promotion_gate_table.csv": primary_gate,
        "oasis_artifact_status.csv": oasis,
    }
    for name, df in outputs.items():
        write_df(df, OUT / name)

    write_text_outputs(primary_gate, cfgdiff, oasis)
    commands.append({"timestamp": now_iso(), "event": "audit_outputs_written", "output_dir": str(OUT)})
    (OUT / "command_log.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(f"audit_written={OUT}")
    print(f"decision={('promote' if bool(primary_gate['promotion_gate_pass'].iloc[0]) else 'parsimonious_sensitivity_only')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
