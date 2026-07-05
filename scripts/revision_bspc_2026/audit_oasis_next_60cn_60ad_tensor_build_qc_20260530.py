#!/usr/bin/env python3
"""Read-only QC audit for the OASIS next 60CN/60AD tensor build."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis_next_60cn_60ad_tensor_build_20260530"
OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis_next_60cn_60ad_tensor_build_qc_20260530"
EXPECTED_TENSORS = [
    "tensor_concatenated_timeseries.npz",
    "tensor_runwise_140TR_connectome_average.npz",
    "tensor_runwise164_connectome_average.npz",
]
EXPECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_ROIS = 131
EXPECTED_N = 120
EXPECTED_DIAGNOSIS = {"CN": 60, "AD_DEMENTIA": 60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 160) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals: List[str] = []
        for col in view.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.8g}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 160) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def finite_stats(values: np.ndarray) -> Dict[str, Any]:
    flat = np.asarray(values).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def offdiag_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return mat[..., mask]


def audit_tensor(path: Path, tensor_name: str) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary: Dict[str, Any] = {
        "tensor_name": tensor_name,
        "path": str(path),
        "exists": path.exists(),
    }
    channel_rows: List[Dict[str, Any]] = []
    diagnosis_rows: List[Dict[str, Any]] = []
    compatibility_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    if not path.exists():
        summary["recommendation_blocker"] = "missing_tensor"
        return summary, channel_rows, diagnosis_rows, compatibility_rows, subject_rows

    with np.load(path, allow_pickle=True) as npz:
        keys = list(npz.files)
        tensor = np.asarray(npz["global_tensor_data"])
        subject_ids = np.asarray(npz["subject_ids"]).astype(str)
        diagnosis = np.asarray(npz["diagnosis"]).astype(str)
        session_ids = np.asarray(npz["session_ids"]).astype(str) if "session_ids" in npz.files else np.array([""] * len(subject_ids))
        experiment_ids = np.asarray(npz["experiment_ids"]).astype(str) if "experiment_ids" in npz.files else np.array([""] * len(subject_ids))
        channel_names = np.asarray(npz["channel_names"]).astype(str).tolist()
        roi_names = np.asarray(npz["roi_names_in_order"]).astype(str) if "roi_names_in_order" in npz.files else np.array([])
        rois_count = int(scalar(npz["rois_count"])) if "rois_count" in npz.files else int(tensor.shape[-1])
        python_bandpass = bool(scalar(npz["python_bandpass_applied"])) if "python_bandpass_applied" in npz.files else None
        external_only = bool(scalar(npz["external_validation_only"])) if "external_validation_only" in npz.files else None
        build_candidate = str(scalar(npz["build_candidate"])) if "build_candidate" in npz.files else ""
        roi_order_name = str(scalar(npz["roi_order_name"])) if "roi_order_name" in npz.files else ""

    n_subjects, n_channels, n_roi_a, n_roi_b = tensor.shape
    nan_count = int(np.isnan(tensor).sum())
    inf_count = int(np.isinf(tensor).sum())
    finite_count = int(np.isfinite(tensor).sum())
    diag = np.diagonal(tensor, axis1=-2, axis2=-1)
    sym_abs = np.abs(tensor - np.swapaxes(tensor, -1, -2))
    diag_stats = finite_stats(diag)
    sym_stats = finite_stats(sym_abs)
    unique_subjects = int(pd.Series(subject_ids).nunique())
    unique_sessions = int(pd.Series(session_ids).nunique())
    unique_experiments = int(pd.Series(experiment_ids).nunique())
    diagnosis_counts = pd.Series(diagnosis).value_counts(dropna=False).to_dict()

    compatible_channels = channel_names == EXPECTED_CHANNEL_NAMES
    compatible_roi = n_roi_a == EXPECTED_ROIS and n_roi_b == EXPECTED_ROIS and rois_count == EXPECTED_ROIS
    aligned_lengths = len(subject_ids) == n_subjects and len(diagnosis) == n_subjects
    no_duplicate_subjects = unique_subjects == n_subjects
    no_nan_inf = nan_count == 0 and inf_count == 0
    symmetric_ok = float(sym_stats["max"]) <= 1e-5 if np.isfinite(sym_stats["max"]) else False
    diag_ok = float(np.nanmax(np.abs(diag))) <= 1e-5 if diag.size else False
    diagnosis_ok = all(int(diagnosis_counts.get(k, 0)) == v for k, v in EXPECTED_DIAGNOSIS.items())

    summary.update(
        {
            "shape": str(tuple(tensor.shape)),
            "subject_count": int(n_subjects),
            "channel_count": int(n_channels),
            "roi_count_a": int(n_roi_a),
            "roi_count_b": int(n_roi_b),
            "rois_count_field": int(rois_count),
            "channel_names": ";".join(channel_names),
            "build_candidate": build_candidate,
            "roi_order_name": roi_order_name,
            "python_bandpass_applied": python_bandpass,
            "external_validation_only": external_only,
            "subject_ids_count": int(len(subject_ids)),
            "diagnosis_labels_count": int(len(diagnosis)),
            "unique_subjects": unique_subjects,
            "unique_sessions": unique_sessions,
            "unique_experiments": unique_experiments,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "finite_count": finite_count,
            "diagonal_mean": diag_stats["mean"],
            "diagonal_std": diag_stats["std"],
            "diagonal_min": diag_stats["min"],
            "diagonal_max": diag_stats["max"],
            "diagonal_abs_max": float(np.nanmax(np.abs(diag))) if diag.size else np.nan,
            "symmetry_abs_mean": sym_stats["mean"],
            "symmetry_abs_std": sym_stats["std"],
            "symmetry_abs_max": sym_stats["max"],
            "compatible_3_channels_for_adni_1_0_2": bool(compatible_channels and n_channels == 3),
            "compatible_131_roi_space": bool(compatible_roi),
            "subject_ids_and_diagnosis_aligned": bool(aligned_lengths),
            "no_duplicate_subject_ids": bool(no_duplicate_subjects),
            "diagnosis_counts_match_expected_60_60": bool(diagnosis_ok),
            "no_nan_or_inf": bool(no_nan_inf),
            "diagonal_zero_ok": bool(diag_ok),
            "symmetric_ok": bool(symmetric_ok),
            "tensor_ready": bool(compatible_channels and compatible_roi and aligned_lengths and no_duplicate_subjects and diagnosis_ok and no_nan_inf and diag_ok and symmetric_ok),
        }
    )

    for dx, count in sorted(diagnosis_counts.items()):
        diagnosis_rows.append({"tensor_name": tensor_name, "diagnosis": dx, "n": int(count)})

    for idx, name in enumerate(channel_names):
        vals = tensor[:, idx, :, :]
        off = offdiag_values(vals)
        all_stats = finite_stats(vals)
        off_stats = finite_stats(off)
        diag_ch = np.diagonal(vals, axis1=-2, axis2=-1)
        sym_ch = np.abs(vals - np.swapaxes(vals, -1, -2))
        channel_rows.append(
            {
                "tensor_name": tensor_name,
                "channel_index": idx,
                "channel_name": name,
                "nan_count": int(np.isnan(vals).sum()),
                "inf_count": int(np.isinf(vals).sum()),
                "all_mean": all_stats["mean"],
                "all_std": all_stats["std"],
                "all_min": all_stats["min"],
                "all_max": all_stats["max"],
                "offdiag_mean": off_stats["mean"],
                "offdiag_std": off_stats["std"],
                "offdiag_min": off_stats["min"],
                "offdiag_max": off_stats["max"],
                "diagonal_abs_max": float(np.nanmax(np.abs(diag_ch))),
                "symmetry_abs_max": float(np.nanmax(sym_ch)),
            }
        )

    checks = {
        "tensor_exists": path.exists(),
        "shape_is_120x3x131x131": tuple(tensor.shape) == (EXPECTED_N, 3, EXPECTED_ROIS, EXPECTED_ROIS),
        "channel_names_match_adni_selected_1_0_2": compatible_channels,
        "roi_space_131x131": compatible_roi,
        "subject_id_count_matches_tensor": len(subject_ids) == n_subjects,
        "diagnosis_count_matches_tensor": len(diagnosis) == n_subjects,
        "diagnosis_counts_60cn_60ad": diagnosis_ok,
        "no_duplicate_subject_ids": no_duplicate_subjects,
        "no_nan_or_inf": no_nan_inf,
        "diagonal_zero": diag_ok,
        "symmetric": symmetric_ok,
        "python_bandpass_off": python_bandpass is False,
        "external_validation_only_true": external_only is True,
    }
    for check, ok in checks.items():
        compatibility_rows.append({"tensor_name": tensor_name, "check": check, "pass": bool(ok)})

    for i in range(n_subjects):
        subject_rows.append(
            {
                "tensor_name": tensor_name,
                "row_index": i,
                "SubjectID": subject_ids[i],
                "session_id": session_ids[i],
                "experiment_id": experiment_ids[i],
                "diagnosis": diagnosis[i],
            }
        )
    return summary, channel_rows, diagnosis_rows, compatibility_rows, subject_rows


def recommendation(existence: pd.DataFrame, summary: pd.DataFrame) -> str:
    existing = set(existence.loc[existence["exists"], "tensor_name"].astype(str))
    all_expected = set(EXPECTED_TENSORS)
    all_present = all_expected.issubset(existing)
    required_primary = {
        "tensor_concatenated_timeseries.npz",
        "tensor_runwise_140TR_connectome_average.npz",
    }.issubset(existing)
    ready_rows = summary[summary["exists"].eq(True)] if "exists" in summary.columns else pd.DataFrame()
    all_ready = bool((not ready_rows.empty) and ready_rows["tensor_ready"].all() and len(ready_rows) == len(EXPECTED_TENSORS))
    primary_ready = bool(required_primary and summary[summary["tensor_name"].isin(["tensor_concatenated_timeseries.npz", "tensor_runwise_140TR_connectome_average.npz"])]["tensor_ready"].all())
    if all_present and all_ready:
        decision = "ready_for_external_scoring"
    elif primary_ready and "tensor_runwise164_connectome_average.npz" not in existing:
        decision = "partial_ready_if_runwise164_missing"
    else:
        decision = "not_ready"
    lines = [
        "# Final Recommendation",
        "",
        f"Decision: `{decision}`.",
        "",
        "This was a read-only QC audit. No model inference, model training, threshold fitting, or tensor modification was performed.",
        "",
    ]
    if decision == "ready_for_external_scoring":
        lines.append("All three expected tensors are present, finite, symmetric, zero-diagonal, aligned to 120 subjects with 60 CN and 60 AD_DEMENTIA labels, and compatible with the ADNI 3-channel `[1,0,2]` 131-ROI model input.")
    elif decision == "partial_ready_if_runwise164_missing":
        lines.append("The primary concatenated and ADNI-like 140TR tensors are ready, but the runwise164 supplementary tensor is missing.")
    else:
        lines.append("At least one required tensor or compatibility/QC check failed. Review `compatibility_audit.csv` and `tensor_qc_summary.csv` before scoring.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existence_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    diagnosis_rows: List[Dict[str, Any]] = []
    compatibility_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []

    for name in EXPECTED_TENSORS:
        path = input_dir / name
        existence_rows.append(
            {
                "tensor_name": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
            }
        )
        summary, channels, diagnoses, compatibility, subjects = audit_tensor(path, name)
        summary_rows.append(summary)
        channel_rows.extend(channels)
        diagnosis_rows.extend(diagnoses)
        compatibility_rows.extend(compatibility)
        subject_rows.extend(subjects)

    existence = pd.DataFrame(existence_rows)
    summary_df = pd.DataFrame(summary_rows)
    channels_df = pd.DataFrame(channel_rows)
    diagnosis_df = pd.DataFrame(diagnosis_rows)
    compat_df = pd.DataFrame(compatibility_rows)
    subjects_df = pd.DataFrame(subject_rows)

    write_table(output_dir, "tensor_existence", existence)
    write_table(output_dir, "tensor_qc_summary", summary_df)
    write_table(output_dir, "diagnosis_counts", diagnosis_df)
    write_table(output_dir, "channel_statistics", channels_df)
    write_table(output_dir, "compatibility_audit", compat_df, max_rows=240)
    write_table(output_dir, "subject_label_alignment", subjects_df, max_rows=80)
    final_text = recommendation(existence, summary_df)
    (output_dir / "final_recommendation.md").write_text(final_text, encoding="utf-8")
    readme = f"""# OASIS Next 60CN/60AD Tensor Build QC

Input: `{input_dir}`

Expected tensors:
- `tensor_concatenated_timeseries.npz`
- `tensor_runwise_140TR_connectome_average.npz`
- `tensor_runwise164_connectome_average.npz`

This audit verifies tensor existence, shape, subject/diagnosis alignment, finite values,
diagonal/symmetry behavior, per-channel distributions, and compatibility with ADNI
models using channels `[1,0,2]` in a 131-ROI space.

No model inference, model training, threshold fitting, or tensor modification was performed.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    write_json(
        output_dir / "command_log.json",
        {
            "timestamp": now(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "expected_tensors": EXPECTED_TENSORS,
            "training_launched": False,
            "model_inference_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "recommendation": final_text.split("`")[1] if "`" in final_text else "",
        },
    )
    print(final_text)
    print(f"Wrote QC outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
