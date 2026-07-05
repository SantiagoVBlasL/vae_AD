#!/usr/bin/env python3
"""Repair the FDR reporting for the local Martin Philips source audit.

This script is intentionally narrow: it reads the existing FP-vs-TN test table,
recomputes Benjamini-Hochberg q-values correctly, and rewrites only the
statistical/reporting files requested for the 2026-06-12 patch. It does not
touch tensors, metadata, predictions, thresholds, or model artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "local_martin_philips_source_audit_20260611"
TEST_CSV = OUT_DIR / "acquisition_qc_fp_vs_tn_tests.csv"

EXPECTED_ROBUST_Q10 = [
    "y_score_final",
    "Age",
    "philips_problem_site_flag",
    "raw_tp_group",
    "tensor_ch0_offdiag_mean",
    "COLPROT",
    "ORIGPROT",
    "Site3",
    "tensor_ch2_offdiag_mean",
]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_md_table(path: Path, title: str, df: pd.DataFrame, preamble: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    if preamble:
        lines.extend(preamble)
        lines.append("")
    if df.empty:
        lines.append("_No rows._")
    else:
        lines.append(df.to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Correct BH FDR with original-row mapping and NaN preservation."""
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(len(pvals), np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return qvals
    finite_idx = np.where(finite)[0]
    sorted_idx = finite_idx[np.argsort(pvals[finite])]
    m = len(sorted_idx)
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = pvals[sorted_idx] * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    qvals[sorted_idx] = np.minimum(q_sorted, 1.0)
    return qvals


def fmt_var_list(items: list[str]) -> str:
    return "\n".join(f"- `{x}`" for x in items)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)

    df = pd.read_csv(TEST_CSV)
    if "p_raw" not in df.columns:
        raise ValueError(f"{TEST_CSV} has no p_raw column")

    pvals = pd.to_numeric(df["p_raw"], errors="coerce").to_numpy(dtype=float)
    df["p_fdr"] = fdr_bh(pvals)
    df["sig_fdr_0p10"] = df["p_fdr"] < 0.10
    df["sig_fdr_0p05"] = df["p_fdr"] < 0.05
    df = df.sort_values("p_raw", na_position="last").reset_index(drop=True)

    sig = df[df["sig_fdr_0p10"]].copy()
    sig_vars = sig["variable"].astype(str).tolist()
    expected_set = set(EXPECTED_ROBUST_Q10)
    observed_set = set(sig_vars)

    df.to_csv(TEST_CSV, index=False)
    md_cols = [
        "variable",
        "type",
        "test",
        "FP_n",
        "FP_median",
        "TN_n",
        "TN_median",
        "p_raw",
        "p_fdr",
        "sig_fdr_0p10",
        "CLES",
        "n_missing",
    ]
    preamble = [
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Benjamini-Hochberg FDR was recomputed with the standard sorted-rank procedure:",
        "`q_sorted = p_sorted * m / rank`, reverse cumulative-min monotonicity, then mapped back to original rows.",
        "Rows with undefined `p_raw` retain `NaN` `p_fdr`.",
        "",
        f"Corrected significant variables at q < 0.10: N={len(sig_vars)}.",
        "",
        "Robust q<0.10 variables expected after correction:",
        fmt_var_list(EXPECTED_ROBUST_Q10),
        "",
        "Observed q<0.10 variables:",
        fmt_var_list(sig_vars) if sig_vars else "- None",
    ]
    write_md_table(
        OUT_DIR / "acquisition_qc_fp_vs_tn_tests.md",
        "Acquisition/QC Tests: Philips CN FP vs TN",
        df[[c for c in md_cols if c in df.columns]],
        preamble,
    )

    # Compact executive summary focused on the corrected statistical conclusion.
    top = df[[c for c in ["variable", "p_raw", "p_fdr", "sig_fdr_0p10"] if c in df.columns]].head(12)
    exec_lines = [
        "# 00 Executive Summary - Local Martin Philips Source Audit",
        "",
        f"**Patched**: {datetime.now(timezone.utc).isoformat()}",
        "**Scope**: FDR/reporting-only repair. No models were trained; no tensors, metadata, predictions, or thresholds were modified.",
        "",
        "## FDR Patch",
        "",
        "The previous `acquisition_qc_fp_vs_tn_tests.csv/.md` contained impossible FDR values: many high raw p-values had q-values near 0.018.",
        "The corrected implementation applies Benjamini-Hochberg on finite p-values sorted ascending, enforces reverse cumulative-min monotonicity, maps q-values back to original rows, and preserves NaN where `p_raw` is undefined.",
        "",
        f"Corrected q<0.10 variables: N={len(sig_vars)}.",
        "",
        fmt_var_list(sig_vars) if sig_vars else "- None",
        "",
        "This removes the prior unsupported claim that 25 variables were FDR-significant.",
        "",
        "## Expected Robust Variables",
        "",
        fmt_var_list(EXPECTED_ROBUST_Q10),
        "",
        "## Top Corrected Rows",
        "",
        top.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The robust descriptive associations remain model score, Age, Philips problem-site/protocol fields, ADNI protocol labels, Site3, and tensor channel 0/2 off-diagonal means.",
        "Findings remain descriptive/exploratory and do not define any subject exclusion, threshold change, or model-selection rule.",
    ]
    (OUT_DIR / "00_EXECUTIVE_SUMMARY.md").write_text("\n".join(exec_lines) + "\n", encoding="utf-8")

    interp_lines = [
        "# Final Local Source Audit Interpretation - Philips CN",
        "",
        f"**Patched**: {datetime.now(timezone.utc).isoformat()}",
        "**Patch scope**: Statistical/reporting repair only.",
        "",
        "## Corrected FP vs TN Statistical Tests",
        "",
        f"Correct Benjamini-Hochberg FDR yields `N={len(sig_vars)}` variables at `q < 0.10`, not 25.",
        "",
        "Robust q<0.10 variables after correction:",
        "",
        fmt_var_list(sig_vars) if sig_vars else "- None",
        "",
        "Variables expected to remain robust are approximately:",
        "",
        fmt_var_list(EXPECTED_ROBUST_Q10),
        "",
        "The observed and expected robust sets "
        + ("match exactly." if observed_set == expected_set else f"differ: missing={sorted(expected_set - observed_set)}, extra={sorted(observed_set - expected_set)}."),
        "",
        "## Variables No Longer FDR-Significant",
        "",
        "The previous report incorrectly marked many descriptive variables as FDR-significant because q-values were not mapped correctly after BH monotonicity enforcement.",
        "High-p variables such as `CDRSB`, `APOE4`, `tensor_ch1_offdiag_mean`, `droi_rms_corrected`, `tsnr_proxy_median_corrected`, `source_batch`, `TR`, and `TE` are not FDR-significant under the corrected calculation.",
        "",
        "## Scientific Interpretation",
        "",
        "The corrected result strengthens the narrower interpretation: Philips CN false positives remain associated descriptively with age, protocol/site structure, and tensor channel shifts, especially lower Pearson/OMST-related channel 0 mean and higher MI-KNN channel 2 mean.",
        "It does not support a broad claim that 25 local acquisition/QC/clinical variables are FDR-significant.",
        "",
        "All findings remain descriptive and exploratory. This patch does not justify subject exclusion, model retraining, threshold changes, OASIS calibration, or promotion/demotion of any model.",
        "",
        "## Guardrails",
        "",
        "- No model training.",
        "- No tensor modification.",
        "- No metadata modification.",
        "- No prediction or threshold modification.",
        "- No subject exclusion rule derived.",
    ]
    (OUT_DIR / "final_local_source_audit_interpretation.md").write_text("\n".join(interp_lines) + "\n", encoding="utf-8")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started,
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
        "rewritten_files": [
            "acquisition_qc_fp_vs_tn_tests.csv",
            "acquisition_qc_fp_vs_tn_tests.md",
            "00_EXECUTIVE_SUMMARY.md",
            "final_local_source_audit_interpretation.md",
            "command_log.json",
        ],
        "fdr_method": "Benjamini-Hochberg finite p-values sorted ascending, q=p*m/rank, reverse cumulative minimum, mapped back to original rows; NaN p_raw preserved.",
        "n_finite_p": int(np.isfinite(pvals).sum()),
        "n_sig_fdr_0p10": int(len(sig_vars)),
        "sig_fdr_0p10_variables": sig_vars,
        "expected_robust_q10_variables": EXPECTED_ROBUST_Q10,
        "guardrails": {
            "model_training": False,
            "tensor_modification": False,
            "metadata_modification": False,
            "prediction_modification": False,
            "threshold_modification": False,
        },
        "argv": sys.argv,
    }
    write_json(OUT_DIR / "command_log.json", command_log)


if __name__ == "__main__":
    main()
