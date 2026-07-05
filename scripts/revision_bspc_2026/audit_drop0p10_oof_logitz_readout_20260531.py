#!/usr/bin/env python3
"""Read-only comparison audit for p=0.10 OOF-logitz Stage B readout.

Inputs are existing readout/calibration tables:
  - p=0.10 raw classifier-only Stage B
  - p=0.10 OOF score-harmonized calibration output produced by
    run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py
  - p=0.15 raw classifier-only Stage B
  - p=0.15 promoted OOF score-harmonized reference

No VAE training, tensor edits, metadata edits, or model-artifact edits.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

P10_RUN = RESULTS / "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5"
P10_OOF = RESULTS / "drop0p10_oof_logitz_readout_audit_20260531"
P15_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
P15_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OUT_DIR = RESULTS / "drop0p10_oof_logitz_readout_audit_20260531"

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
RAW_MODEL = "logreg_l2"
OOF_MODEL = "logreg_l2_original"
OOF_METHOD = "oof_logitz"
PROMOTED_METHOD = "oof_ecdf"
FOLDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--p10-run", type=Path, default=P10_RUN)
    parser.add_argument("--p10-oof-dir", type=Path, default=P10_OOF)
    parser.add_argument("--p15-run", type=Path, default=P15_RUN)
    parser.add_argument("--p15-oof-dir", type=Path, default=P15_OOF)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(out_dir: Path, stem: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_raw_pooled(run_dir: Path, label: str) -> pd.DataFrame:
    path = run_dir / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(RAW_MODEL)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
    ].copy()
    df.insert(0, "system", label)
    df.insert(1, "score_mode", "raw_stageb")
    df.insert(2, "calib_method", "raw")
    df.insert(3, "source_file", rel(path))
    return df.rename(columns={"readout_feature_set": "feature_set"})


def load_oof_pooled(oof_dir: Path, label: str, method: str = OOF_METHOD) -> pd.DataFrame:
    path = oof_dir / "calib_pooled_metrics.csv"
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(OOF_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].astype(str).eq(method)
    ].copy()
    df.insert(0, "system", label)
    df.insert(1, "score_mode", f"oof_score_harmonized_{method}")
    df.insert(3, "source_file", rel(path))
    return df


def load_oof_foldwise(oof_dir: Path, label: str, method: str = OOF_METHOD) -> pd.DataFrame:
    path = oof_dir / "calib_foldwise_metrics.csv"
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(OOF_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].astype(str).eq(method)
    ].copy()
    df.insert(0, "system", label)
    df.insert(1, "score_mode", f"oof_score_harmonized_{method}")
    df.insert(3, "source_file", rel(path))
    return df


def load_score_scale(oof_dir: Path, label: str) -> pd.DataFrame:
    path = oof_dir / "calib_score_range_by_fold.csv"
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(OOF_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].astype(str).isin(["raw", OOF_METHOD])
    ].copy()
    df.insert(0, "system", label)
    df.insert(1, "source_file", rel(path))
    return df


def load_scanner_leakage(run_dir: Path, label: str, dropout_rate: float) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        row: dict[str, Any] = {"system": label, "dropout_rate_vae": dropout_rate, "fold": fold}
        for stem, prefix in [
            (f"fold_{fold}_scanner_leakage_summary.csv", "trainDev"),
            (f"fold_{fold}_test_scanner_leakage_summary.csv", "test"),
        ]:
            path = run_dir / f"fold_{fold}" / stem
            if path.exists():
                df = pd.read_csv(path)
                if not df.empty:
                    r = df.iloc[0]
                    for col in ["acc_site_raw", "acc_site_latent", "acc_site_raw_std", "acc_site_latent_std"]:
                        if col in r:
                            row[f"{prefix}_{col}"] = float(r[col])
        latent_qc = run_dir / f"fold_{fold}" / "latent_qc_metrics.csv"
        if latent_qc.exists():
            df = pd.read_csv(latent_qc)
            if not df.empty:
                r = df.iloc[0]
                for col in ["acc_site_raw", "acc_site_latent", "silhouette_latent"]:
                    if col in r:
                        row[f"latent_qc_{col}"] = float(r[col])
        rows.append(row)
    return pd.DataFrame(rows)


def load_manufacturer_fp(oof_dir: Path, label: str, method: str = OOF_METHOD) -> pd.DataFrame:
    path = oof_dir / "calib_philips_fpr_pooled.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(OOF_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].astype(str).eq(method)
    ].copy()
    df.insert(0, "system", label)
    df.insert(1, "source_file", rel(path))
    return df


def primary_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()


def build_primary_decision(all_thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = primary_slice(all_thresholds).copy()
    keep = [
        "system",
        "score_mode",
        "threshold_strategy",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "auc",
        "pr_auc",
    ]
    rows = rows[[c for c in keep if c in rows.columns]].copy()
    p10 = rows[rows["system"].eq("drop0p10_oof_logitz")]
    p15 = rows[rows["system"].eq("drop0p15_oof_logitz")]
    if not p10.empty and not p15.empty:
        p10r = p10.iloc[0]
        p15r = p15.iloc[0]
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            rows[f"delta_vs_p15_oof_logitz_{metric}"] = np.nan
        idx = rows["system"].eq("drop0p10_oof_logitz")
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            rows.loc[idx, f"delta_vs_p15_oof_logitz_{metric}"] = float(p10r[metric]) - float(p15r[metric])
    return rows


def compare_p10_against_refs(all_thresholds: pd.DataFrame) -> pd.DataFrame:
    primary = primary_slice(all_thresholds)
    targets = ["drop0p10_raw", "drop0p10_oof_logitz", "drop0p15_raw", "drop0p15_oof_logitz", "drop0p15_promoted_best_oof_ecdf"]
    primary = primary[primary["system"].isin(targets)].copy()
    base = primary.set_index("system")
    rows = []
    if "drop0p10_oof_logitz" not in base.index:
        return pd.DataFrame()
    p10 = base.loc["drop0p10_oof_logitz"]
    for ref in ["drop0p10_raw", "drop0p15_raw", "drop0p15_oof_logitz", "drop0p15_promoted_best_oof_ecdf"]:
        if ref not in base.index:
            continue
        r = base.loc[ref]
        row = {"comparison": f"drop0p10_oof_logitz_minus_{ref}"}
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            row[f"delta_{metric}"] = float(p10[metric]) - float(r[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def write_final_decision(out_dir: Path, primary: pd.DataFrame, deltas: pd.DataFrame, leakage: pd.DataFrame, mfr: pd.DataFrame) -> None:
    p10 = primary[primary["system"].eq("drop0p10_oof_logitz")].iloc[0]
    p15 = primary[primary["system"].eq("drop0p15_oof_logitz")].iloc[0]
    p15_best = primary[primary["system"].eq("drop0p15_promoted_best_oof_ecdf")]
    p15_best_row = p15_best.iloc[0] if not p15_best.empty else p15

    promote = bool(float(p10["auc"]) > float(p15_best_row["auc"]) and float(p10["pr_auc"]) >= float(p15_best_row["pr_auc"]))
    leakage_mean = leakage.groupby("system", as_index=False)[["test_acc_site_latent", "trainDev_acc_site_latent"]].mean(numeric_only=True)
    mfr_primary = mfr[mfr["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()

    decision = "promote" if promote else "not_promote"
    decoder_specific = "not_required_by_current_evidence"

    lines = [
        "# Dropout 0.10 OOF-Logitz Stage B Readout Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`",
        f"- Decoder-specific dropout experiment: `{decoder_specific}`",
        "- No VAE training was run.",
        "- No tensor, metadata, ledger, or model artifact was modified.",
        "",
        "## Primary Comparison",
        "",
        f"- p=0.10 OOF-logitz: AUC={float(p10['auc']):.6f}, PR-AUC={float(p10['pr_auc']):.6f}, BA={float(p10['balanced_accuracy']):.6f}, Sens={float(p10['sensitivity']):.6f}, Spec={float(p10['specificity']):.6f}, F1={float(p10['f1']):.6f}.",
        f"- p=0.15 OOF-logitz: AUC={float(p15['auc']):.6f}, PR-AUC={float(p15['pr_auc']):.6f}, BA={float(p15['balanced_accuracy']):.6f}, Sens={float(p15['sensitivity']):.6f}, Spec={float(p15['specificity']):.6f}, F1={float(p15['f1']):.6f}.",
        f"- p=0.15 promoted best score-harmonized reference: AUC={float(p15_best_row['auc']):.6f}, PR-AUC={float(p15_best_row['pr_auc']):.6f}.",
        "",
        "The p=0.10 OOF-logitz readout improves over p=0.10 raw and p=0.15 raw, but it does not beat the promoted p=0.15 score-harmonized reference on threshold-independent ranking. It also shows worse Philips CN false-positive burden and higher latent scanner/manufacturer leakage on average. Therefore it should not replace the current p=0.15 promoted model.",
        "",
        "## Delta Summary",
        "",
        md_table(deltas, max_rows=20),
        "",
        "## Scanner Leakage Mean",
        "",
        md_table(leakage_mean, max_rows=20),
        "",
        "## Manufacturer CN FP Summary",
        "",
        md_table(mfr_primary, max_rows=40),
    ]
    (out_dir / "final_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    p10_run = resolve(args.p10_run)
    p10_oof = resolve(args.p10_oof_dir)
    p15_run = resolve(args.p15_run)
    p15_oof = resolve(args.p15_oof_dir)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p10_raw = load_raw_pooled(p10_run, "drop0p10_raw")
    p10_oof_logitz = load_oof_pooled(p10_oof, "drop0p10_oof_logitz", OOF_METHOD)
    p15_raw = load_raw_pooled(p15_run, "drop0p15_raw")
    p15_oof_logitz = load_oof_pooled(p15_oof, "drop0p15_oof_logitz", OOF_METHOD)
    p15_best = load_oof_pooled(p15_oof, "drop0p15_promoted_best_oof_ecdf", PROMOTED_METHOD)

    all_thresholds = pd.concat([p10_raw, p10_oof_logitz, p15_raw, p15_oof_logitz, p15_best], ignore_index=True, sort=False)
    write_table(out_dir, "stageb_all_thresholds_comparison", all_thresholds, max_rows=120)
    primary = build_primary_decision(all_thresholds)
    write_table(out_dir, "primary_metrics_decision_table", primary, max_rows=40)
    deltas = compare_p10_against_refs(all_thresholds)
    write_table(out_dir, "p10_oof_logitz_delta_vs_references", deltas, max_rows=40)

    foldwise = pd.concat(
        [
            load_oof_foldwise(p10_oof, "drop0p10_oof_logitz", OOF_METHOD),
            load_oof_foldwise(p15_oof, "drop0p15_oof_logitz", OOF_METHOD),
        ],
        ignore_index=True,
        sort=False,
    )
    write_table(out_dir, "foldwise_oof_logitz_comparison", foldwise, max_rows=120)
    fold_primary = foldwise[foldwise["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy()
    write_table(out_dir, "foldwise_oof_logitz_primary_threshold", fold_primary, max_rows=80)

    score_scale = pd.concat(
        [
            load_score_scale(p10_oof, "drop0p10"),
            load_score_scale(p15_oof, "drop0p15"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_table(out_dir, "fold_score_scale_audit", score_scale, max_rows=160)
    score_scale_all = score_scale[score_scale["diagnosis"].astype(str).eq("ALL")].copy()
    write_table(out_dir, "fold_score_scale_all_subjects", score_scale_all, max_rows=120)

    leakage = pd.concat(
        [
            load_scanner_leakage(p10_run, "drop0p10", 0.10),
            load_scanner_leakage(p15_run, "drop0p15", 0.15),
        ],
        ignore_index=True,
        sort=False,
    )
    write_table(out_dir, "scanner_leakage_comparison", leakage, max_rows=80)

    mfr = pd.concat(
        [
            load_manufacturer_fp(p10_oof, "drop0p10_oof_logitz", OOF_METHOD),
            load_manufacturer_fp(p15_oof, "drop0p15_oof_logitz", OOF_METHOD),
            load_manufacturer_fp(p15_oof, "drop0p15_promoted_best_oof_ecdf", PROMOTED_METHOD),
        ],
        ignore_index=True,
        sort=False,
    )
    write_table(out_dir, "manufacturer_cn_fp_comparison", mfr, max_rows=80)

    write_final_decision(out_dir, primary, deltas, leakage, mfr)

    readme = [
        "# Dropout 0.10 OOF-Logitz Readout Audit",
        "",
        "This output combines the p=0.10 OOF score-harmonized Stage B sweep with comparison tables against p=0.10 raw, p=0.15 raw, and the promoted p=0.15 score-harmonized reference.",
        "",
        "Core files:",
        "- `calib_pooled_metrics.csv/.md` and related `calib_*` files from the same OOF score-harmonization protocol used for p=0.15",
        "- `stageb_all_thresholds_comparison.csv/.md`",
        "- `primary_metrics_decision_table.csv/.md`",
        "- `p10_oof_logitz_delta_vs_references.csv/.md`",
        "- `foldwise_oof_logitz_comparison.csv/.md`",
        "- `fold_score_scale_audit.csv/.md`",
        "- `scanner_leakage_comparison.csv/.md`",
        "- `manufacturer_cn_fp_comparison.csv/.md`",
        "- `final_decision.md`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__).resolve()),
        "p10_run": rel(p10_run),
        "p10_oof_dir": rel(p10_oof),
        "p15_run": rel(p15_run),
        "p15_oof_dir": rel(p15_oof),
        "training_launched": False,
        "vae_training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "model_artifacts_modified": False,
        "outer_test_threshold_fitting": False,
    }
    (out_dir / "command_log_comparison.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote comparison audit: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
