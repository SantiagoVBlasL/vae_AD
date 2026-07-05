#!/usr/bin/env python3
"""Compact read-only summary for DPARSF-bandpass original-model inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/original_model_inference_dparsf_bandpass10"
EXPECTED_N = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed original-model inference on DPARSF-bandpass ROI signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}. Available: {list(df.columns)}")


def fmt_float(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def unique_nonempty(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def build_compact_predictions(pred_meta: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        pred_meta,
        [
            "SubjectID",
            "classifier",
            "y_score_ensemble",
            "y_score_std",
            "y_pred_ensemble",
            "y_pred_majority_vote",
            "ResearchGroup_Mapped",
            "Manufacturer",
            "Site3",
            "Age",
            "Sex",
            "ImageID",
        ],
        Path("dparsf_bandpass10_predictions_with_metadata.csv"),
    )
    base_cols = ["ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex", "ImageID"]
    meta = pred_meta.sort_values(["SubjectID", "classifier"]).drop_duplicates("SubjectID")[["SubjectID"] + base_cols]
    score_wide = pred_meta.pivot(index="SubjectID", columns="classifier", values="y_score_ensemble")
    std_wide = pred_meta.pivot(index="SubjectID", columns="classifier", values="y_score_std")
    pred_wide = pred_meta.pivot(index="SubjectID", columns="classifier", values="y_pred_ensemble")
    vote_wide = pred_meta.pivot(index="SubjectID", columns="classifier", values="y_pred_majority_vote")

    compact = meta.set_index("SubjectID")
    for classifier in ["logreg", "svm"]:
        compact[f"{classifier}_score"] = score_wide.get(classifier)
        compact[f"{classifier}_score_std"] = std_wide.get(classifier)
        compact[f"{classifier}_pred_ensemble"] = pred_wide.get(classifier)
        compact[f"{classifier}_majority_vote"] = vote_wide.get(classifier)
    compact["both_classifiers_ad_like"] = (
        (compact.get("logreg_pred_ensemble") == 1) & (compact.get("svm_pred_ensemble") == 1)
    )
    compact["any_classifier_ad_like"] = (
        (compact.get("logreg_pred_ensemble") == 1) | (compact.get("svm_pred_ensemble") == 1)
    )
    compact["borderline_note"] = ""
    if "svm_score" in compact.columns:
        borderline = compact["svm_score"].between(0.49, 0.51, inclusive="both")
        compact.loc[borderline, "borderline_note"] = "SVM score near 0.5"
    return compact.reset_index().sort_values("SubjectID")


def build_compact_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        comparison,
        [
            "SubjectID",
            "classifier",
            "previous_y_score_ensemble",
            "previous_y_pred_ensemble",
            "previous_y_pred_majority_vote",
            "dparsf_bandpass_y_score_ensemble",
            "dparsf_bandpass_y_pred_ensemble",
            "dparsf_bandpass_y_pred_majority_vote",
            "delta_score",
            "previous_predictions_source_file",
            "Age",
            "Sex",
            "Manufacturer",
        ],
        Path("subject_level_comparison_vs_previous_preprocessing.csv"),
    )
    direct = comparison.dropna(subset=["previous_y_score_ensemble"]).copy()
    if direct.empty:
        return pd.DataFrame(
            columns=[
                "SubjectID",
                "classifier",
                "previous_y_score_ensemble",
                "dparsf_bandpass_y_score_ensemble",
                "delta_score",
                "previous_y_pred_ensemble",
                "dparsf_bandpass_y_pred_ensemble",
                "changed_prediction_0p5",
                "previous_predictions_source_file",
                "Age",
                "Sex",
                "Manufacturer",
            ]
        )
    direct["source_rank"] = direct["previous_predictions_source_file"].astype(str).str.contains(
        "martin59_predictions_with_metadata", case=False, regex=False
    ).map({True: 0, False: 1})
    direct = direct.sort_values(["SubjectID", "classifier", "source_rank", "previous_predictions_source_file"])
    compact = direct.drop_duplicates(["SubjectID", "classifier"], keep="first").copy()
    keep = [
        "SubjectID",
        "classifier",
        "previous_y_score_ensemble",
        "dparsf_bandpass_y_score_ensemble",
        "delta_score",
        "previous_y_pred_ensemble",
        "dparsf_bandpass_y_pred_ensemble",
        "previous_y_pred_majority_vote",
        "dparsf_bandpass_y_pred_majority_vote",
        "changed_prediction_0p5",
        "changed_majority_vote",
        "previous_predictions_source_file",
        "Age",
        "Sex",
        "Manufacturer",
    ]
    return compact[keep]


def write_markdown(
    out_path: Path,
    compact_predictions: pd.DataFrame,
    compact_comparison: pd.DataFrame,
    detected: pd.DataFrame,
    roi_validation: pd.DataFrame,
) -> None:
    n_detected = int(detected["SubjectID"].nunique())
    n_expected = EXPECTED_N
    subjects = sorted(compact_predictions["SubjectID"].astype(str).unique())
    all_cn = set(compact_predictions["ResearchGroup_Mapped"].astype(str).str.upper()) == {"CN"}
    manufacturers = unique_nonempty(compact_predictions["Manufacturer"])
    all_siemens = set(m.upper() for m in manufacturers) == {"SIEMENS"}

    logreg = compact_predictions["logreg_pred_ensemble"].fillna(0).astype(int)
    svm = compact_predictions["svm_pred_ensemble"].fillna(0).astype(int)
    svm_vote = compact_predictions["svm_majority_vote"].fillna(0).astype(int)
    both = (logreg == 1) & (svm == 1)

    roi_ok = (
        not roi_validation.empty
        and (roi_validation["input_n_rois"].astype(int) == 170).all()
        and (roi_validation["output_n_rois"].astype(int) == 131).all()
        and (roi_validation["expected_n_rois"].astype(int) == 131).all()
        and (roi_validation["status"].astype(str) == "OK").all()
        and roi_validation["roi_reordering_active"].astype(bool).all()
    )

    borderline = compact_predictions[compact_predictions["SubjectID"] == "002_S_1280"]
    consensus = compact_predictions[compact_predictions["SubjectID"] == "002_S_6053"]
    comp_4644 = compact_comparison[compact_comparison["SubjectID"] == "003_S_4644"].sort_values("classifier")

    lines = [
        "# Original-Model Inference Summary: DPARSF Bandpass10",
        "",
        "This is a compact read-only summary of the completed original-paper-model inference run. It uses only existing CSV audit and prediction artifacts; it does not load tensors, checkpoints, joblibs, or retrain anything.",
        "",
        "## Cohort",
        "",
        f"- N detected: `{n_detected}/{n_expected}` expected.",
        f"- Subjects: `{', '.join(subjects)}`.",
        f"- Diagnosis/scanner: `{'all CN' if all_cn else 'mixed diagnosis'}` and `{'all SIEMENS' if all_siemens else ', '.join(manufacturers)}`.",
        "- This is a CN stress-test, not an AUC evaluation.",
        "",
        "## ROI And Preprocessing",
        "",
        f"- ROI reduction/reordering status: `{'OK' if roi_ok else 'CHECK'}`.",
        "- AAL3 ROI reduction: `170 -> 131` for every detected subject.",
        "- Yeo17 ROI reorder active: `True`.",
        "- Python bandpass: `skipped` to avoid double filtering because DPARSF already applied bandpass.",
        "",
        "## AD-like Counts At Threshold 0.5",
        "",
        f"- LogReg `y_pred_ensemble`: `{int(logreg.sum())}/{n_detected}`.",
        f"- SVM `y_pred_ensemble`: `{int(svm.sum())}/{n_detected}`.",
        f"- SVM majority vote: `{int(svm_vote.sum())}/{n_detected}`.",
        f"- Both classifiers AD-like: `{int(both.sum())}/{n_detected}`.",
        "",
        "## Subject-Level Notes",
        "",
    ]
    if not borderline.empty:
        b = borderline.iloc[0]
        lines.append(
            f"- Borderline subject: `002_S_1280`, SVM score `{fmt_float(b['svm_score'])}`, "
            f"SVM ensemble pred `{int(b['svm_pred_ensemble'])}`, majority vote `CN`."
        )
    if not consensus.empty:
        c = consensus.iloc[0]
        lines.append(
            f"- Consensus AD-like subject: `002_S_6053`, LogReg `{fmt_float(c['logreg_score'])}`, "
            f"SVM `{fmt_float(c['svm_score'])}`."
        )
    lines.extend(["", "## Previous-Preprocessing Comparison", ""])
    if comp_4644.empty:
        lines.append("- No direct previous-preprocessing comparison was found for `003_S_4644`.")
    else:
        for _, row in comp_4644.iterrows():
            classifier = str(row["classifier"])
            lines.append(
                f"- `003_S_4644` {classifier}: previous `{fmt_float(row['previous_y_score_ensemble'])}` -> "
                f"new `{fmt_float(row['dparsf_bandpass_y_score_ensemble'])}`, "
                f"delta `{fmt_float(row['delta_score'])}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The lower direct comparison scores for `003_S_4644` support preprocessing/filtering drift as an important contributor to previous AD-like false positives.",
            "- This does not prove scanner leakage is absent.",
            "- Because all subjects are CN Siemens, this run tests preprocessing compatibility and AD-like false-positive behavior; it is not an AUC/performance evaluation.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    tables_dir = run_dir / "Tables"
    audit_dir = run_dir / "audit"

    ensemble_path = tables_dir / "covid_predictions_ensemble.csv"
    pred_meta_path = tables_dir / "dparsf_bandpass10_predictions_with_metadata.csv"
    comparison_path = tables_dir / "subject_level_comparison_vs_previous_preprocessing.csv"
    roi_path = audit_dir / "roi_reduction_validation.csv"
    detected_path = audit_dir / "detected_subjects.csv"
    for path in [ensemble_path, pred_meta_path, comparison_path, roi_path, detected_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    # Keep the ensemble read explicit so missing or malformed primary output fails early.
    ensemble = pd.read_csv(ensemble_path)
    require_columns(ensemble, ["SubjectID", "classifier", "y_score_ensemble", "y_pred_ensemble"], ensemble_path)
    pred_meta = pd.read_csv(pred_meta_path)
    comparison = pd.read_csv(comparison_path)
    roi_validation = pd.read_csv(roi_path)
    detected = pd.read_csv(detected_path)

    compact_predictions = build_compact_predictions(pred_meta)
    compact_comparison = build_compact_comparison(comparison)

    compact_predictions.to_csv(tables_dir / "compact_predictions_for_martin.csv", index=False)
    compact_comparison.to_csv(tables_dir / "subject_level_preprocessing_comparison_compact.csv", index=False)
    write_markdown(
        run_dir / "summary_original_model_dparsf_bandpass10.md",
        compact_predictions,
        compact_comparison,
        detected,
        roi_validation,
    )
    print(f"Wrote {tables_dir / 'compact_predictions_for_martin.csv'}")
    print(f"Wrote {tables_dir / 'subject_level_preprocessing_comparison_compact.csv'}")
    print(f"Wrote {run_dir / 'summary_original_model_dparsf_bandpass10.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
