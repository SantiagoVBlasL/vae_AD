#!/usr/bin/env python3
"""Compare DPARSF-only vs DPARSF+Python-bandpass original-model inference.

Read-only with respect to the inference runs. It reads only small CSV outputs
and writes compact comparison tables under results/revision_bspc_2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DPARSE_ONLY = PROJECT_ROOT / "results/revision_bspc_2026/original_model_inference_dparsf_bandpass10"
DEFAULT_DPARSE_PYTHON = PROJECT_ROOT / "results/revision_bspc_2026/original_model_inference_dparsf_bandpass10_pythonbandpass"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/dparsf_filtering_control_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original-model DPARSF-only vs DPARSF+Python-bandpass inference scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dparsf-only-run", type=Path, default=DEFAULT_DPARSE_ONLY)
    parser.add_argument("--dparsf-python-run", type=Path, default=DEFAULT_DPARSE_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ensemble(run_dir: Path, prefix: str) -> pd.DataFrame:
    path = run_dir / "Tables/covid_predictions_ensemble.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ensemble predictions: {path}")
    df = pd.read_csv(path)
    required = {"SubjectID", "classifier", "y_score_ensemble", "y_pred_ensemble", "y_pred_majority_vote"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    keep = [
        "SubjectID",
        "classifier",
        "y_score_ensemble",
        "y_pred_ensemble",
        "y_pred_majority_vote",
    ]
    if "y_score_std" in df.columns:
        keep.append("y_score_std")
    out = df[keep].copy()
    out = out.rename(
        columns={
            "y_score_ensemble": f"score_{prefix}",
            "y_pred_ensemble": f"pred_{prefix}",
            "y_pred_majority_vote": f"majority_{prefix}",
            "y_score_std": f"score_std_{prefix}",
        }
    )
    return out


def read_previous_reference(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "Tables/subject_level_comparison_vs_previous_preprocessing.csv"
    cols = [
        "SubjectID",
        "classifier",
        "previous_y_score_ensemble",
        "previous_preproc_source",
        "previous_predictions_source_file",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    missing = [col for col in cols[:2] if col not in df.columns]
    if missing:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    ref = df[cols].copy()
    ref["previous_score_if_available"] = pd.to_numeric(ref["previous_y_score_ensemble"], errors="coerce")
    ref["previous_source_if_available"] = ref["previous_preproc_source"].fillna(ref["previous_predictions_source_file"])
    ref = ref.drop(columns=["previous_y_score_ensemble", "previous_preproc_source", "previous_predictions_source_file"])
    ref = ref.dropna(subset=["previous_score_if_available"], how="all")
    if ref.empty:
        return ref
    return ref.sort_values(["SubjectID", "classifier", "previous_source_if_available"]).drop_duplicates(
        ["SubjectID", "classifier"], keep="first"
    )


def ad_like_counts(df: pd.DataFrame, pred_col: str, majority_col: str) -> list[str]:
    lines = []
    for classifier, sub in df.groupby("classifier"):
        pred_n = int(pd.to_numeric(sub[pred_col], errors="coerce").fillna(0).eq(1).sum())
        maj_n = int(pd.to_numeric(sub[majority_col], errors="coerce").fillna(0).eq(1).sum())
        lines.append(f"- {classifier}: ensemble AD-like={pred_n}/{len(sub)}, majority-vote AD-like={maj_n}/{len(sub)}")
    return lines


def maybe_fmt(value: object) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def write_summary(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "# DPARSF Filtering Control Comparison",
        "",
        "This compares the same original paper model and the same 9 CN Siemens subjects under two tensor-generation controls:",
        "",
        "- DPARSF-bandpass input with Python bandpass OFF.",
        "- DPARSF-bandpass input with Python bandpass ON (0.01-0.08 Hz, TR=3.0, target_len_ts=140).",
        "",
        "No retraining was performed. This is a filtering compatibility test, not an AUC evaluation.",
        "",
        "## AD-like Counts: DPARSF-only",
    ]
    lines.extend(ad_like_counts(table, "pred_dparsf_only", "majority_dparsf_only"))
    lines.extend(["", "## AD-like Counts: DPARSF+Python Bandpass"])
    lines.extend(ad_like_counts(table, "pred_dparsf_pythonbandpass", "majority_dparsf_pythonbandpass"))
    lines.extend(["", "## Mean Score Delta: Python minus DPARSF-only"])
    for classifier, sub in table.groupby("classifier"):
        delta = pd.to_numeric(sub["delta_python_minus_only"], errors="coerce")
        lines.append(f"- {classifier}: mean={delta.mean():.4f}, median={delta.median():.4f}")
    lines.extend(["", "## Subject 003_S_4644"])
    sub4644 = table[table["SubjectID"] == "003_S_4644"].copy()
    if sub4644.empty:
        lines.append("- Subject 003_S_4644 was not present in the comparison table.")
    else:
        for _, row in sub4644.sort_values("classifier").iterrows():
            lines.append(
                f"- {row['classifier']}: previous={maybe_fmt(row.get('previous_score_if_available'))}, "
                f"DPARSF-only={maybe_fmt(row.get('score_dparsf_only'))}, "
                f"DPARSF+Python={maybe_fmt(row.get('score_dparsf_pythonbandpass'))}, "
                f"delta_python_minus_only={maybe_fmt(row.get('delta_python_minus_only'))}"
            )
    lines.extend(
        [
            "",
            "## Interpretation Caveats",
            "- This tests filtering compatibility, not diagnostic AUC.",
            "- N=9/10 expected subjects were available.",
            "- All subjects are CN Siemens in the recovered metadata.",
            "- The original DPARSF status in paper training remains not fully proven, but Python bandpass in feature extraction is high-confidence.",
            "- Similar DPARSF-only and DPARSF+Python scores would argue against Python filtering as the dominant explanation for prior AD-like calls.",
            "- A shift toward previous preprocessing scores would make filtering compatibility a more likely contributor.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "dparsf_only_vs_pythonbandpass_subject_scores.csv"
    summary_path = args.output_dir / "dparsf_only_vs_pythonbandpass_summary.md"
    if (table_path.exists() or summary_path.exists()) and not args.overwrite:
        raise RuntimeError(f"Output exists; pass --overwrite to replace files in {args.output_dir}")

    only = read_ensemble(args.dparsf_only_run, "dparsf_only")
    py = read_ensemble(args.dparsf_python_run, "dparsf_pythonbandpass")
    table = only.merge(py, on=["SubjectID", "classifier"], how="outer")
    previous = read_previous_reference(args.dparsf_only_run)
    if not previous.empty:
        table = table.merge(previous, on=["SubjectID", "classifier"], how="left")
    else:
        table["previous_score_if_available"] = np.nan
        table["previous_source_if_available"] = np.nan

    table["delta_python_minus_only"] = pd.to_numeric(table["score_dparsf_pythonbandpass"], errors="coerce") - pd.to_numeric(
        table["score_dparsf_only"], errors="coerce"
    )
    table["changed_prediction"] = pd.to_numeric(table["pred_dparsf_only"], errors="coerce") != pd.to_numeric(
        table["pred_dparsf_pythonbandpass"], errors="coerce"
    )
    table = table.sort_values(["SubjectID", "classifier"])
    table.to_csv(table_path, index=False)
    write_summary(summary_path, table)
    print(f"Wrote {table_path}")
    print(f"Wrote {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
