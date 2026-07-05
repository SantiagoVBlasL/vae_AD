#!/usr/bin/env python3
"""Create paper-ready tables from the final threshold audit.

Read-only summarization layer. It does not train models and does not touch
tensors, metadata, or ledgers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_AUDIT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_threshold_final_audit"
SWEEP_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
MFR_AUDIT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_audit"
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_paper_ready_threshold_tables"

PRIMARY_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def round_float(value: object, ndigits: int = 3) -> object:
    if pd.isna(value):
        return value
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def paper_metric_table(df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, label in label_map.items():
        model_name, strategy = key.split("|", 1)
        match = df[
            (df["model_name"].astype(str) == model_name)
            & (df["threshold_strategy"].astype(str) == strategy)
            & (df["row_type"].astype(str) == "absolute")
        ]
        if match.empty:
            raise RuntimeError(f"Missing requested primary-results row: {key}")
        row = match.iloc[0]
        rows.append(
            {
                "Readout": label,
                "Threshold rule": strategy.replace("original_fixed_0p5", "0.5").replace("fixed_0p5", "0.5"),
                "N": int(row["n"]),
                "AD": int(row["n_ad"]),
                "CN": int(row["n_cn"]),
                "AUC": round_float(row["auc"]),
                "PR-AUC": round_float(row["pr_auc"]),
                "Balanced accuracy": round_float(row["balanced_accuracy"]),
                "Sensitivity": round_float(row["sensitivity"]),
                "Specificity": round_float(row["specificity"]),
                "F1": round_float(row["f1"]),
                "TN": int(row["tn"]),
                "FP": int(row["fp"]),
                "FN": int(row["fn"]),
                "TP": int(row["tp"]),
                "Predicted AD rate": round_float(row["predicted_ad_rate"]),
                "Leakage-safe threshold selection": "Yes" if "inner_oof" in strategy else "Not applicable",
            }
        )
    return pd.DataFrame(rows)


def subgroup_table(df: pd.DataFrame, group_col_name: str) -> pd.DataFrame:
    sub = df[df["threshold_strategy"].astype(str) == PRIMARY_STRATEGY].copy()
    if sub.empty:
        raise RuntimeError(f"Missing subgroup rows for {PRIMARY_STRATEGY}")
    rows: List[Dict[str, object]] = []
    for _, row in sub.iterrows():
        rows.append(
            {
                group_col_name: row["group_value"],
                "N": int(row["n"]),
                "AD": int(row["n_ad"]),
                "CN": int(row["n_cn"]),
                "AUC": round_float(row["auc"]),
                "PR-AUC": round_float(row["pr_auc"]),
                "Balanced accuracy": round_float(row["balanced_accuracy"]),
                "Sensitivity": round_float(row["sensitivity"]),
                "Specificity": round_float(row["specificity"]),
                "F1": round_float(row["f1"]),
                "TN": int(row["tn"]),
                "FP": int(row["fp"]),
                "FN": int(row["fn"]),
                "TP": int(row["tp"]),
                "Selected threshold": "fold-specific inner-OOF target sensitivity >=0.70",
            }
        )
    return pd.DataFrame(rows)


def fold4_error_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["threshold_strategy"].astype(str) == PRIMARY_STRATEGY].copy()
    if sub.empty:
        raise RuntimeError(f"Missing Fold 4 rows for {PRIMARY_STRATEGY}")
    out = pd.DataFrame(
        {
            "SubjectID": sub["SubjectID"],
            "Diagnosis": sub["y_true"].map({1: "AD", 0: "CN"}),
            "Manufacturer": sub["Manufacturer"],
            "Age": sub["Age"].map(lambda x: round_float(x, 1)),
            "Sex": sub["Sex"],
            "Score": sub["y_score"].map(round_float),
            "Selected threshold": sub["threshold"].map(round_float),
            "Margin to threshold": sub["score_minus_threshold"].map(round_float),
            "Prediction": sub["y_pred"].map({1: "AD-like", 0: "CN-like"}),
            "Error type": sub["error_type"],
            "Error interpretation": sub["error_interpretation"],
            "Threshold-near <=0.05": sub["threshold_near_0p05"],
            "Threshold-near <=0.10": sub["threshold_near_0p10"],
            "Source batch": sub.get("source_batch", pd.Series([""] * len(sub))),
            "Source label": sub.get("source_label", pd.Series([""] * len(sub))),
            "Tensor source": sub.get("tensor_source", pd.Series([""] * len(sub))),
        }
    )
    order = {"false_negative": 0, "false_positive": 1}
    out["_error_order"] = out["Error type"].map(order).fillna(99)
    out["_abs_margin"] = out["Margin to threshold"].abs()
    out = out.sort_values(["_error_order", "_abs_margin", "SubjectID"]).drop(columns=["_error_order", "_abs_margin"])
    return out


def make_readme(
    primary: pd.DataFrame,
    manufacturer: pd.DataFrame,
    sex: pd.DataFrame,
    fold4: pd.DataFrame,
    verification: pd.DataFrame,
) -> str:
    target = primary[primary["Readout"] == "LogReg L2 + inner-OOF target sensitivity >=0.70"].iloc[0]
    fixed = primary[primary["Readout"] == "LogReg L2 + threshold 0.5"].iloc[0]
    original = primary[primary["Readout"] == "Original LogReg + threshold 0.5"].iloc[0]
    philips = manufacturer[manufacturer["Manufacturer"].astype(str).str.lower() == "philips"].iloc[0]
    ge = manufacturer[manufacturer["Manufacturer"].astype(str).str.lower() == "ge"].iloc[0]
    verification_pass = bool((verification["verification_status"] == "PASS").all())
    n_fn = int((fold4["Error type"] == "false_negative").sum())
    n_fp = int((fold4["Error type"] == "false_positive").sum())
    n_near = int(fold4["Threshold-near <=0.05"].sum())

    return f"""# Paper-Ready Threshold Tables: ADNI v5.1 batch20260514b

## Recommended readout

Primary readout: `logreg_l2` with fold-specific inner-OOF threshold selection targeting sensitivity >=0.70 while maximizing specificity.

This operating point gives AUC={target['AUC']:.3f}, PR-AUC={target['PR-AUC']:.3f}, sensitivity={target['Sensitivity']:.3f}, specificity={target['Specificity']:.3f}, balanced accuracy={target['Balanced accuracy']:.3f}, and F1={target['F1']:.3f}.

## Why threshold 0.5 is not clinically appropriate here

The original calibrated LogReg at 0.5 is very conservative: sensitivity={original['Sensitivity']:.3f}, specificity={original['Specificity']:.3f}. That misses most AD cases. The classifier-only LogReg L2 at 0.5 is less conservative, but still leaves sensitivity at {fixed['Sensitivity']:.3f}. For an AD/CN clinical screening readout, a fixed 0.5 threshold is an arbitrary probability-scale choice and does not reflect the desired sensitivity/specificity tradeoff.

## Why LogReg L2 is the primary readout

LogReg L2 keeps the best ranking performance among the audited classifier-only options while remaining interpretable and stable. The target-sensitivity threshold improves sensitivity to {target['Sensitivity']:.3f} with specificity {target['Specificity']:.3f}; random forest Youden has lower AUC/PR-AUC despite competitive balanced accuracy.

## Leakage safety

Threshold verification pass: `{verification_pass}`.

Non-0.5 thresholds were selected from train/dev inner-CV out-of-fold predictions only. Outer-test labels were used only for final evaluation, not threshold selection.

## Remaining weaknesses

Fold 4 remains the main stress point for the primary operating point: {n_fn} false negatives and {n_fp} false positives, with {n_near} errors within 0.05 of the selected threshold.

Manufacturer residuals are not uniform. Philips CN specificity remains weak at {philips['Specificity']:.3f}, while GE AD sensitivity remains weak at {ge['Sensitivity']:.3f}. These should be reported as subgroup limitations rather than hidden in pooled metrics.

## Manuscript-safe claim

Safe: In internal 5-fold evaluation on fixed VAE folds, classifier-only LogReg L2 with leakage-safe inner-OOF threshold selection improved the sensitivity/specificity operating point relative to the original 0.5 readouts while preserving similar AUC/PR-AUC.

Not safe: claiming deployment readiness, external validation, or a clinically final threshold. The threshold remains an internal cross-validated operating point and should be validated externally.

## Files

- `primary_results_table.csv` and `primary_results_table.md`
- `manufacturer_subgroup_primary_model.csv` and `manufacturer_subgroup_primary_model.md`
- `sex_subgroup_primary_model.csv` and `sex_subgroup_primary_model.md`
- `fold4_subject_level_error_table.csv` and `fold4_subject_level_error_table.md`
- `command_log.json`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(require(SWEEP_DIR / "comparison_vs_original_logreg_svm.csv"))
    manufacturer = pd.read_csv(require(THRESHOLD_AUDIT_DIR / "logreg_l2_subgroup_manufacturer.csv"))
    sex = pd.read_csv(require(THRESHOLD_AUDIT_DIR / "logreg_l2_subgroup_sex.csv"))
    fold4 = pd.read_csv(require(THRESHOLD_AUDIT_DIR / "fold4_logreg_l2_error_audit.csv"))
    verification = pd.read_csv(require(THRESHOLD_AUDIT_DIR / "threshold_selection_verification.csv"))
    require(MFR_AUDIT_DIR / "README.md")

    label_map = {
        "original_logreg|original_fixed_0p5": "Original LogReg + threshold 0.5",
        "original_svm|original_fixed_0p5": "Original SVM + threshold 0.5",
        "logreg_l2|fixed_0p5": "LogReg L2 + threshold 0.5",
        "logreg_l2|inner_oof_youden_j": "LogReg L2 + inner-OOF Youden",
        "logreg_l2|inner_oof_target_sens_ge_0p70_max_spec": "LogReg L2 + inner-OOF target sensitivity >=0.70",
        "random_forest|inner_oof_youden_j": "Random forest + inner-OOF Youden",
    }
    primary = paper_metric_table(comparison, label_map)
    save_table(primary, OUT_DIR / "primary_results_table.csv", OUT_DIR / "primary_results_table.md")

    manufacturer_primary = subgroup_table(manufacturer, "Manufacturer")
    save_table(
        manufacturer_primary,
        OUT_DIR / "manufacturer_subgroup_primary_model.csv",
        OUT_DIR / "manufacturer_subgroup_primary_model.md",
    )

    sex_primary = subgroup_table(sex, "Sex")
    save_table(sex_primary, OUT_DIR / "sex_subgroup_primary_model.csv", OUT_DIR / "sex_subgroup_primary_model.md")

    fold4_table = fold4_error_table(fold4)
    save_table(fold4_table, OUT_DIR / "fold4_subject_level_error_table.csv", OUT_DIR / "fold4_subject_level_error_table.md")

    readme = make_readme(primary, manufacturer_primary, sex_primary, fold4_table, verification)
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    log = {
        "script": str(Path(__file__).resolve()),
        "threshold_audit_dir": str(THRESHOLD_AUDIT_DIR),
        "classifier_sweep_dir": str(SWEEP_DIR),
        "mfrsplit_audit_dir": str(MFR_AUDIT_DIR),
        "output_dir": str(OUT_DIR),
        "threshold_selection_verified": bool((verification["verification_status"] == "PASS").all()),
        "vae_retrained": False,
        "training_run": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
