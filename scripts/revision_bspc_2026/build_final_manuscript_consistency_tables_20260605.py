#!/usr/bin/env python3
"""Final manuscript-consistency audit and compact ADNI/OASIS tables.

Read-only with respect to model/tensor/metadata artifacts. The script only
reads the already-generated final model-decision package and writes manuscript
support tables/paragraphs.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT = Path("results/revision_bspc_2026/final_model_decision_adni_oasis_manuscript_support_20260605")
OUT = Path("results/revision_bspc_2026/final_manuscript_consistency_tables_20260605")

MAIN_MODELS = [
    "promoted_latent384_beta3p75_ch1_0_2",
    "ch1only_latent384_beta3p75",
    "latent448_beta4p0",
    "latent512_beta3p75",
    "mfrBalancedVAE_latent384_beta3p75",
    "residualized_mfr_stageB",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def md_write(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    view = df if max_rows is None else df.head(max_rows)
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    path.write_text(text + "\n", encoding="utf-8")


def fmt6(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def fmt4(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def value_variants(x: Any) -> set[str]:
    vals: set[str] = set()
    try:
        if pd.isna(x):
            return vals
        f = float(x)
        vals.add(str(x))
        vals.add(f"{f:.6f}")
        vals.add(f"{f:.4f}")
        vals.add(f"{f:.3f}")
        vals.add(f"{f:.6g}")
        vals.add(f"{f:.5g}")
        vals.add(f"{f:.4g}")
        # pandas markdown may trim trailing zeroes.
        vals.add((f"{f:.6f}").rstrip("0").rstrip("."))
        vals.add((f"{f:.4f}").rstrip("0").rstrip("."))
    except Exception:
        vals.add(str(x))
    return {v for v in vals if v and v.lower() != "nan"}


def csv_md_consistency(csv_path: Path, md_path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not csv_path.exists() or not md_path.exists():
        rows.append(
            {
                "check_group": "csv_md_consistency",
                "source": label,
                "check_name": "file_exists",
                "expected": f"{csv_path.name} and {md_path.name}",
                "observed": "missing",
                "status": "FAIL",
                "details": "",
            }
        )
        return rows
    df = pd.read_csv(csv_path)
    text = read_text(md_path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    checked = 0
    missing: list[str] = []
    for col in numeric_cols:
        for i, val in df[col].items():
            if pd.isna(val):
                continue
            checked += 1
            if not any(v in text for v in value_variants(val)):
                missing.append(f"row={i}; col={col}; value={val}")
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    rows.append(
        {
            "check_group": "csv_md_consistency",
            "source": label,
            "check_name": "numeric_values_present_in_md",
            "expected": f"{checked} non-null numeric CSV cells represented in markdown",
            "observed": "first_missing=" + "; ".join(missing) if missing else "all checked values found",
            "status": "PASS" if not missing else "FAIL",
            "details": f"numeric_columns={len(numeric_cols)}",
        }
    )
    return rows


def text_checks(decision: pd.DataFrame, input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = decision.set_index("model_id")
    promoted = by_id.loc["promoted_latent384_beta3p75_ch1_0_2"]
    ch1 = by_id.loc["ch1only_latent384_beta3p75"]
    beta35 = by_id.loc["latent384_beta3p5"]
    files = {
        "manuscript_results_paragraph.md": read_text(input_dir / "manuscript_results_paragraph.md"),
        "manuscript_limitations_paragraph.md": read_text(input_dir / "manuscript_limitations_paragraph.md"),
        "reviewer_response_scanner_oasis_paragraph.md": read_text(input_dir / "reviewer_response_scanner_oasis_paragraph.md"),
        "final_interpretation.md": read_text(input_dir / "final_interpretation.md"),
    }
    expected_by_file: dict[str, list[tuple[str, Any]]] = {
        "manuscript_results_paragraph.md": [
            ("promoted_adni_auc", promoted["adni_oof_ecdf_auc"]),
            ("promoted_adni_pr_auc", promoted["adni_oof_ecdf_pr_auc"]),
            ("promoted_adni_ba", promoted["adni_oof_ecdf_balanced_accuracy"]),
            ("promoted_adni_sens", promoted["adni_oof_ecdf_sensitivity"]),
            ("promoted_adni_spec", promoted["adni_oof_ecdf_specificity"]),
            ("promoted_adni_f1", promoted["adni_oof_ecdf_f1"]),
            ("ch1_adni_auc", ch1["adni_oof_ecdf_auc"]),
            ("ch1_adni_pr_auc", ch1["adni_oof_ecdf_pr_auc"]),
            ("promoted_oasis_runwise164_auc", promoted["oasis_runwise164_auc"]),
            ("promoted_oasis_runwise164_pr_auc", promoted["oasis_runwise164_pr_auc"]),
        ],
        "reviewer_response_scanner_oasis_paragraph.md": [
            ("promoted_adni_auc", promoted["adni_oof_ecdf_auc"]),
            ("promoted_adni_pr_auc", promoted["adni_oof_ecdf_pr_auc"]),
            ("promoted_adni_ba", promoted["adni_oof_ecdf_balanced_accuracy"]),
            ("promoted_adni_f1", promoted["adni_oof_ecdf_f1"]),
            ("promoted_philips_cn_fpr", promoted["philips_cn_fpr"]),
            ("promoted_scanner_leakage", promoted["scanner_leakage_latent_acc"]),
        ],
        "final_interpretation.md": [
            ("promoted_adni_auc", promoted["adni_oof_ecdf_auc"]),
            ("promoted_adni_pr_auc", promoted["adni_oof_ecdf_pr_auc"]),
            ("promoted_adni_ba", promoted["adni_oof_ecdf_balanced_accuracy"]),
            ("promoted_adni_f1", promoted["adni_oof_ecdf_f1"]),
            ("promoted_philips_cn_fpr", promoted["philips_cn_fpr"]),
            ("promoted_scanner_leakage", promoted["scanner_leakage_latent_acc"]),
            ("beta35_adni_auc", beta35["adni_oof_ecdf_auc"]),
            ("beta35_adni_pr_auc", beta35["adni_oof_ecdf_pr_auc"]),
        ],
        "manuscript_limitations_paragraph.md": [],
    }
    for file_name, checks in expected_by_file.items():
        text = files[file_name]
        if not checks:
            numeric_tokens = re.findall(r"\d+\.\d+", text)
            rows.append(
                {
                    "check_group": "text_numeric_consistency",
                    "source": file_name,
                    "check_name": "no_specific_numeric_claims_to_crosscheck",
                    "expected": "no decimal metric claims",
                    "observed": f"decimal_tokens={numeric_tokens}",
                    "status": "PASS" if not numeric_tokens else "REVIEW",
                    "details": "",
                }
            )
            continue
        for name, expected in checks:
            variants = value_variants(expected)
            present = any(v in text for v in variants)
            rows.append(
                {
                    "check_group": "text_numeric_consistency",
                    "source": file_name,
                    "check_name": name,
                    "expected": fmt6(expected),
                    "observed": "present" if present else "not_found",
                    "status": "PASS" if present else "FAIL",
                    "details": "accepted_variants=" + ",".join(sorted(variants)),
                }
            )
    return rows


def build_main_table(decision: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "display_name": "model",
        "adni_oof_ecdf_auc": "ADNI AUC",
        "adni_oof_ecdf_pr_auc": "ADNI PR-AUC",
        "adni_oof_ecdf_balanced_accuracy": "ADNI BA",
        "adni_oof_ecdf_f1": "ADNI F1",
        "philips_cn_fpr": "Philips CN FPR",
        "oasis_runwise164_auc": "OASIS runwise164 AUC",
        "oasis_runwise164_pr_auc": "OASIS runwise164 PR-AUC",
        "oasis_runwise140_auc": "OASIS runwise140 AUC",
        "oasis_runwise140_pr_auc": "OASIS runwise140 PR-AUC",
        "decision_class": "decision class",
    }
    table = decision[decision["model_id"].isin(MAIN_MODELS)].copy()
    table["_order"] = table["model_id"].map({m: i for i, m in enumerate(MAIN_MODELS)})
    table = table.sort_values("_order")
    out = table[list(cols)].rename(columns=cols)
    for c in out.columns:
        if c != "model" and c != "decision class" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(6)
    return out


def build_supplement_table(decision: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "model_id",
        "display_name",
        "decision_class",
        "final_decision",
        "channel_set_order",
        "beta_vae",
        "latent_dim",
        "adni_metric_source",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_ecdf_balanced_accuracy",
        "adni_oof_ecdf_sensitivity",
        "adni_oof_ecdf_specificity",
        "adni_oof_ecdf_f1",
        "philips_cn_fpr",
        "scanner_leakage_latent_acc",
        "adni_stageb_raw_auc",
        "adni_stageb_raw_pr_auc",
        "adni_stageb_raw_balanced_accuracy",
        "adni_stageb_raw_sensitivity",
        "adni_stageb_raw_specificity",
        "adni_stageb_raw_f1",
        "D_val_best_mean",
        "R_val_bits_best_mean",
        "bits_per_latent_dim_best_mean",
        "beta_KLD_over_D_best_mean",
        "active_units_mean",
        "total_correlation_nats_mean",
        "MI_Z_Y_nats_mean",
        "MI_Z_Manufacturer_nats_mean",
        "MI_Manufacturer_over_MI_Y_mean",
        "oasis_concatenated_auc",
        "oasis_concatenated_pr_auc",
        "oasis_concatenated_balanced_accuracy",
        "oasis_concatenated_sensitivity",
        "oasis_concatenated_specificity",
        "oasis_concatenated_f1",
        "oasis_runwise164_auc",
        "oasis_runwise164_pr_auc",
        "oasis_runwise164_balanced_accuracy",
        "oasis_runwise164_sensitivity",
        "oasis_runwise164_specificity",
        "oasis_runwise164_f1",
        "oasis_runwise140_auc",
        "oasis_runwise140_pr_auc",
        "oasis_runwise140_balanced_accuracy",
        "oasis_runwise140_sensitivity",
        "oasis_runwise140_specificity",
        "oasis_runwise140_f1",
        "oasis_artifact_status",
        "rationale_short",
    ]
    cols = [c for c in ordered_cols if c in decision.columns]
    return decision[cols].copy()


def write_polished_paragraphs(out: Path, decision: pd.DataFrame) -> None:
    by_id = decision.set_index("model_id")
    p = by_id.loc["promoted_latent384_beta3p75_ch1_0_2"]
    ch1 = by_id.loc["ch1only_latent384_beta3p75"]
    l448 = by_id.loc["latent448_beta4p0"]
    l512 = by_id.loc["latent512_beta3p75"]
    mfr = by_id.loc["mfrBalancedVAE_latent384_beta3p75"]

    results = (
        "The final prespecified ADNI model was the recover035 multichannel [1,0,2] "
        "latent384 beta3.75 beta-VAE with the OOF-ECDF logreg_l2 readout. In nested "
        f"ADNI evaluation, this model achieved AUC {fmt6(p['adni_oof_ecdf_auc'])}, "
        f"PR-AUC {fmt6(p['adni_oof_ecdf_pr_auc'])}, balanced accuracy {fmt6(p['adni_oof_ecdf_balanced_accuracy'])}, "
        f"sensitivity {fmt6(p['adni_oof_ecdf_sensitivity'])}, specificity {fmt6(p['adni_oof_ecdf_specificity'])}, "
        f"and F1 {fmt6(p['adni_oof_ecdf_f1'])}. The ch1-only model was more parsimonious "
        f"and had higher ADNI AUC/PR-AUC ({fmt6(ch1['adni_oof_ecdf_auc'])}/{fmt6(ch1['adni_oof_ecdf_pr_auc'])}), "
        "but it did not improve the operating-point profile and showed weaker OASIS transfer, "
        "so it was retained as a sensitivity model. Capacity, beta, and deconfounding variants "
        f"did not provide a clean replacement: latent448 beta4.0 reached AUC {fmt6(l448['adni_oof_ecdf_auc'])}, "
        f"latent512 beta3.75 reached AUC {fmt6(l512['adni_oof_ecdf_auc'])}, and mfrBalancedVAE reached "
        f"AUC {fmt6(mfr['adni_oof_ecdf_auc'])}. In OASIS external stress testing, the promoted model "
        f"showed its strongest ranking signal on the runwise164 build (AUC {fmt6(p['oasis_runwise164_auc'])}, "
        f"PR-AUC {fmt6(p['oasis_runwise164_pr_auc'])}) and a similar but weaker signal on the runwise140 build "
        f"(AUC {fmt6(p['oasis_runwise140_auc'])}, PR-AUC {fmt6(p['oasis_runwise140_pr_auc'])})."
    )
    (out / "final_results_section_paragraph.md").write_text(results + "\n", encoding="utf-8")

    limitations = (
        "The OASIS analyses were external stress tests, not a tuning set. All external predictions used "
        "frozen ADNI-derived preprocessing, fold VAEs, classifier readouts, score transformations, and thresholds; "
        "OASIS labels were used only after prediction for metric reporting. The transferred ADNI thresholds were "
        "conservative on OASIS, producing low predicted-AD rates and low sensitivity in several builds, which "
        "indicates score-distribution and threshold-transfer shift rather than an internally valid reason to alter "
        "the selected ADNI model. Some historical beta/capacity runs lack matched OOF-ECDF or OASIS artifacts, so "
        "those missing entries are treated as absent evidence rather than external failures. A future external "
        "calibration/test study should estimate any threshold recalibration only in a prespecified calibration subset "
        "and evaluate it on an untouched locked test subset."
    )
    (out / "final_limitations_paragraph.md").write_text(limitations + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    decision = pd.read_csv(input_dir / "final_model_decision_table.csv")
    comparison = pd.read_csv(input_dir / "adni_oasis_final_comparison.csv")
    main_table = build_main_table(decision)
    supp_table = build_supplement_table(decision)

    audit_rows: list[dict[str, Any]] = []
    audit_rows.extend(csv_md_consistency(input_dir / "final_model_decision_table.csv", input_dir / "final_model_decision_table.md", "final_model_decision_table"))
    audit_rows.extend(csv_md_consistency(input_dir / "adni_oasis_final_comparison.csv", input_dir / "adni_oasis_final_comparison.md", "adni_oasis_final_comparison"))
    audit_rows.extend(text_checks(decision, input_dir))
    # Cross-check matching values between the two CSVs for shared columns.
    shared = [c for c in comparison.columns if c in decision.columns and c not in {"rationale_short"}]
    merged = comparison[shared].merge(decision[shared], on=["model_id", "display_name"], suffixes=("_comparison", "_decision"))
    mismatches: list[str] = []
    for col in shared:
        if col in {"model_id", "display_name"}:
            continue
        a = merged.get(f"{col}_comparison")
        b = merged.get(f"{col}_decision")
        if a is None or b is None:
            continue
        if pd.api.types.is_numeric_dtype(a):
            mask = ~np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            mask = a.fillna("").astype(str) != b.fillna("").astype(str)
        if mask.any():
            mismatches.extend([f"{col}: rows={list(np.where(mask)[0])[:5]}"])
    audit_rows.append(
        {
            "check_group": "csv_cross_consistency",
            "source": "final_model_decision_table.csv vs adni_oasis_final_comparison.csv",
            "check_name": "shared_values_match",
            "expected": "all shared values match",
            "observed": "; ".join(mismatches) if mismatches else "all shared values match",
            "status": "PASS" if not mismatches else "FAIL",
            "details": f"shared_columns={len(shared)}",
        }
    )
    audit = pd.DataFrame(audit_rows)

    if args.dry_run:
        print(f"output_dir={out}")
        print(f"main_table_rows={len(main_table)}")
        print(f"supplement_rows={len(supp_table)}")
        print(f"audit_rows={len(audit)}")
        print(f"audit_failures={(audit['status'] == 'FAIL').sum()}")
        return 0

    (out / "README.md").write_text(
        "# Final Manuscript Consistency Tables\n\n"
        "Read-only consistency audit and manuscript-ready tables derived from the selected ADNI/OASIS model package.\n",
        encoding="utf-8",
    )
    audit.to_csv(out / "consistency_audit.csv", index=False)
    md_write(audit, out / "consistency_audit.md")
    main_table.to_csv(out / "main_paper_model_table.csv", index=False)
    md_write(main_table, out / "main_paper_model_table.md")
    supp_table.to_csv(out / "supplement_extended_model_table.csv", index=False)
    md_write(supp_table, out / "supplement_extended_model_table.md")
    write_polished_paragraphs(out, decision)

    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "guardrails": {
            "no_training": True,
            "no_scoring": True,
            "no_threshold_fitting": True,
            "no_calibration_fitting": True,
            "no_model_artifact_modification": True,
        },
        "rows": {
            "main_paper_model_table": int(len(main_table)),
            "supplement_extended_model_table": int(len(supp_table)),
            "consistency_audit": int(len(audit)),
            "consistency_failures": int((audit["status"] == "FAIL").sum()),
            "consistency_review_flags": int((audit["status"] == "REVIEW").sum()),
        },
        "outputs": [
            "consistency_audit.csv",
            "consistency_audit.md",
            "main_paper_model_table.csv",
            "main_paper_model_table.md",
            "supplement_extended_model_table.csv",
            "supplement_extended_model_table.md",
            "final_results_section_paragraph.md",
            "final_limitations_paragraph.md",
        ],
    }
    (out / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote manuscript consistency package to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
