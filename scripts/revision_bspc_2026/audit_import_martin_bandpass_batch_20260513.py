#!/usr/bin/env python3
"""Audit Martin's 2026-05-13 ADNI v5.1 bandpass ROISignals batch.

Read-only with respect to existing datasets:
- no training;
- no tensor construction;
- no Python bandpass in the final path;
- writes audit CSVs and a candidate ledger only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_and_import_local_ge_cn_roisignals_v5_1 import (  # noqa: E402
    classify_scale,
    load_signal,
    normalize_subject,
    orient_time_by_roi,
    stage_guess,
)


BATCH_ROOT = PROJECT_ROOT / "data" / "OneDrive_1_13-5-2026"
RESULTS_AAL3 = BATCH_ROOT / "ResultsAAL3"
CHECK_DIR = BATCH_ROOT / "Check"
LEDGER_CURRENT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "final_cn_ge_inventory_before_martin_request"
    / "ADNI_v5_1_DATA_LEDGER_CURRENT.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "martin_bandpass_batch_20260513"

TR_SECONDS = 3.0
LOW_HZ = 0.01
HIGH_HZ = 0.08
EXPECTED_ROIS = 170
BATCH_LABEL = "20260513_bandpass_batch1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit/import-readiness for Martin bandpass ROISignals batch 20260513.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--ledger", type=Path, default=LEDGER_CURRENT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path.resolve())


def safe_stat(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"file_size_bytes": np.nan, "mtime": ""}
    return {
        "file_size_bytes": int(stat.st_size),
        "mtime": pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
    }


def discover_check_files(check_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not check_dir.exists():
        return pd.DataFrame(columns=["SubjectID", "check_path", "check_suffix", "check_file_size_bytes", "check_mtime"])
    for path in sorted(p for p in check_dir.rglob("*") if p.is_file()):
        sid = normalize_subject(path.name)
        stat = safe_stat(path)
        rows.append(
            {
                "SubjectID": sid,
                "check_path": str(path.resolve()),
                "check_suffix": path.suffix.lower(),
                "check_file_size_bytes": stat["file_size_bytes"],
                "check_mtime": stat["mtime"],
            }
        )
    return pd.DataFrame(rows)


def stage_root(path: Path, results_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(results_root.resolve())
    except Exception:
        return ""
    return rel.parts[0] if rel.parts else ""


def discover_roisignals(batch_root: Path, check_df: pd.DataFrame) -> pd.DataFrame:
    results_root = batch_root / "ResultsAAL3"
    check_counts = (
        check_df[check_df["SubjectID"].astype(bool)]
        .groupby("SubjectID")
        .agg(
            check_file_count=("check_path", "nunique"),
            check_paths=("check_path", lambda s: "|".join(sorted(set(clean(x) for x in s if clean(x))))),
        )
        .reset_index()
        if not check_df.empty
        else pd.DataFrame(columns=["SubjectID", "check_file_count", "check_paths"])
    )
    rows: List[Dict[str, Any]] = []
    if results_root.exists():
        for path in sorted(results_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".mat", ".txt"}:
                continue
            if not path.name.startswith("ROISignals_"):
                continue
            sid = normalize_subject(path.name)
            stat = safe_stat(path)
            stage = stage_guess(path)
            rows.append(
                {
                    "SubjectID": sid,
                    "path": str(path.resolve()),
                    "relative_path": rel_or_abs(path),
                    "suffix": path.suffix.lower(),
                    "stage_folder": stage_root(path, results_root),
                    "stage_guess": stage,
                    "explicit_has_F_stage": "yes" if stage in {"ARWSDCFN", "ARWSDCF"} else "no",
                    "file_size_bytes": stat["file_size_bytes"],
                    "mtime": stat["mtime"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(check_counts, on="SubjectID", how="left")
    out["check_file_count"] = out["check_file_count"].fillna(0).astype(int)
    out["check_paths"] = out["check_paths"].fillna("")
    out["batch_root"] = str(batch_root.resolve())
    return out.sort_values(["SubjectID", "suffix", "path"]).reset_index(drop=True)


def signal_stats(path: Path) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    arr, var_name, raw_shape, load_status = load_signal(path)
    row: Dict[str, Any] = {
        "main_variable": var_name,
        "shape": raw_shape,
        "load_status": load_status,
        "orientation_status": "",
        "n_timepoints": 0,
        "n_rois": 0,
        "finite_fraction": 0.0,
        "nan_count": np.nan,
        "all_nan_columns": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "median": np.nan,
        "min": np.nan,
        "max": np.nan,
        "scale_label": "other",
        "is_aal3_170": "no",
        "qc_signal_pass": "no",
        "qc_signal_reason": "load_failed",
        "qc_signal_warning": "",
    }
    if arr is None or arr.ndim != 2 or arr.size == 0:
        return row, None
    oriented, n_tp, n_rois, orient_status = orient_time_by_roi(arr)
    if oriented is None:
        row["qc_signal_reason"] = "not_2d_or_orientation_failed"
        return row, None
    finite = np.isfinite(oriented)
    vals = oriented[finite]
    row.update(
        {
            "shape": raw_shape,
            "orientation_status": orient_status,
            "n_timepoints": int(n_tp),
            "n_rois": int(n_rois),
            "finite_fraction": float(finite.mean()),
            "nan_count": int(np.isnan(oriented).sum()),
            "all_nan_columns": int(np.isnan(oriented).all(axis=0).sum()),
            "mean": float(np.nanmean(vals)) if vals.size else np.nan,
            "std": float(np.nanstd(vals)) if vals.size else np.nan,
            "median": float(np.nanmedian(vals)) if vals.size else np.nan,
            "min": float(np.nanmin(vals)) if vals.size else np.nan,
            "max": float(np.nanmax(vals)) if vals.size else np.nan,
            "scale_label": classify_scale(vals),
            "is_aal3_170": "yes" if int(n_rois) == EXPECTED_ROIS else "no",
        }
    )
    reasons: List[str] = []
    if int(n_rois) != EXPECTED_ROIS:
        reasons.append(f"n_rois={n_rois}, expected 170")
    if float(row["finite_fraction"]) < 0.95:
        reasons.append(f"finite_fraction={row['finite_fraction']:.4f} < 0.95")
    if row["scale_label"] != "around_10000_global_scaled":
        reasons.append(f"scale_label={row['scale_label']}")
    if not reasons:
        row["qc_signal_pass"] = "yes"
        row["qc_signal_reason"] = "valid_170roi_finite_10000_scale"
        if int(row["all_nan_columns"]) > 0:
            row["qc_signal_warning"] = (
                f"all_nan_columns={row['all_nan_columns']} accepted because finite_fraction remains >=0.95"
            )
    else:
        row["qc_signal_reason"] = "; ".join(reasons)
    return row, oriented


def spectral_metrics(ts: Optional[np.ndarray]) -> Dict[str, Any]:
    out = {
        "spectral_status": "not_computed",
        "spectral_valid_rois": 0,
        "energy_below_0p01": np.nan,
        "energy_0p01_0p08": np.nan,
        "energy_above_0p08": np.nan,
        "spectral_class": "uncertain",
    }
    if ts is None or ts.ndim != 2 or ts.shape[0] < 8:
        out["spectral_status"] = "invalid_signal"
        return out
    fractions: List[Tuple[float, float, float]] = []
    for idx in range(ts.shape[1]):
        col = np.asarray(ts[:, idx], dtype=np.float64)
        finite = np.isfinite(col)
        if finite.mean() < 0.95 or finite.sum() < 8:
            continue
        filled = col.copy()
        filled[~finite] = np.nanmean(filled[finite])
        centered = filled - np.mean(filled)
        if np.nanstd(centered) <= 1e-12:
            continue
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        freqs = np.fft.rfftfreq(centered.size, d=TR_SECONDS)
        non_dc = freqs > 0
        total = float(np.sum(spectrum[non_dc]))
        if not np.isfinite(total) or total <= 0:
            continue
        below = float(np.sum(spectrum[(freqs > 0) & (freqs < LOW_HZ)]) / total)
        band = float(np.sum(spectrum[(freqs >= LOW_HZ) & (freqs <= HIGH_HZ)]) / total)
        above = float(np.sum(spectrum[freqs > HIGH_HZ]) / total)
        fractions.append((below, band, above))
    if not fractions:
        out["spectral_status"] = "no_valid_roi_columns"
        return out
    arr = np.asarray(fractions, dtype=np.float64)
    below, band, above = np.nanmedian(arr, axis=0)
    if band >= 0.75 and below <= 0.05 and above <= 0.25:
        label = "bandpassed_like"
    elif band < 0.62 or below > 0.20 or above > 0.35:
        label = "unfiltered_like"
    else:
        label = "uncertain"
    out.update(
        {
            "spectral_status": "ok",
            "spectral_valid_rois": int(arr.shape[0]),
            "energy_below_0p01": float(below),
            "energy_0p01_0p08": float(band),
            "energy_above_0p08": float(above),
            "spectral_class": label,
        }
    )
    return out


def qc_roisignals(inventory: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qc_rows: List[Dict[str, Any]] = []
    spectral_rows: List[Dict[str, Any]] = []
    for _, row in inventory.iterrows():
        path = Path(clean(row["path"]))
        stats, oriented = signal_stats(path)
        base = row.to_dict()
        qc_row = {**base, **stats}
        spectral = spectral_metrics(oriented)
        qc_rows.append(qc_row)
        spectral_rows.append(
            {
                "SubjectID": row["SubjectID"],
                "path": row["path"],
                "relative_path": row["relative_path"],
                "suffix": row["suffix"],
                "stage_guess": row["stage_guess"],
                "n_timepoints": qc_row["n_timepoints"],
                "n_rois": qc_row["n_rois"],
                **spectral,
            }
        )
    qc = pd.DataFrame(qc_rows)
    spectral = pd.DataFrame(spectral_rows)
    if not qc.empty and not spectral.empty:
        qc = qc.merge(
            spectral[
                [
                    "path",
                    "spectral_status",
                    "spectral_valid_rois",
                    "energy_below_0p01",
                    "energy_0p01_0p08",
                    "energy_above_0p08",
                    "spectral_class",
                ]
            ],
            on="path",
            how="left",
        )
    return qc, spectral


def ledger_lookup(ledger: pd.DataFrame) -> Dict[str, pd.Series]:
    return {clean(row["SubjectID"]): row for _, row in ledger.iterrows()}


def build_vs_ledger(inventory: pd.DataFrame, ledger: pd.DataFrame, check_df: pd.DataFrame) -> pd.DataFrame:
    ledger_by_sid = ledger_lookup(ledger)
    check_counts = (
        check_df[check_df["SubjectID"].astype(bool)]
        .groupby("SubjectID")
        .agg(
            check_file_count=("check_path", "nunique"),
            check_paths=("check_path", lambda s: "|".join(sorted(set(clean(x) for x in s if clean(x))))),
        )
        .reset_index()
        if not check_df.empty
        else pd.DataFrame(columns=["SubjectID", "check_file_count", "check_paths"])
    )
    inv_counts = inventory.groupby("SubjectID").agg(
        roisignals_file_count=("path", "nunique"),
        stage_guesses=("stage_guess", lambda s: "|".join(sorted(set(clean(x) for x in s if clean(x))))),
        suffixes=("suffix", lambda s: "|".join(sorted(set(clean(x) for x in s if clean(x))))),
        roisignals_paths=("path", lambda s: "|".join(sorted(set(clean(x) for x in s if clean(x))))),
    ).reset_index()
    all_subjects = sorted(set(inv_counts["SubjectID"]) | set(check_counts["SubjectID"]))
    rows: List[Dict[str, Any]] = []
    for sid in all_subjects:
        ledger_row = ledger_by_sid.get(sid)
        inv_row = inv_counts[inv_counts["SubjectID"].eq(sid)]
        chk_row = check_counts[check_counts["SubjectID"].eq(sid)]
        rows.append(
            {
                "SubjectID": sid,
                "in_ledger": "yes" if ledger_row is not None else "no",
                "ledger_scope": clean(ledger_row.get("ledger_scope")) if ledger_row is not None else "",
                "priority_batch": clean(ledger_row.get("priority_batch")) if ledger_row is not None else "",
                "dicom_series_ok": clean(ledger_row.get("dicom_series_ok")) if ledger_row is not None else "",
                "processing_status_previous": clean(ledger_row.get("processing_status")) if ledger_row is not None else "",
                "action_needed_previous": clean(ledger_row.get("action_needed")) if ledger_row is not None else "",
                "python_bandpass_requested": clean(ledger_row.get("python_bandpass_requested")) if ledger_row is not None else "",
                "Manufacturer": clean(ledger_row.get("Manufacturer")) if ledger_row is not None else "",
                "Diagnosis": clean(ledger_row.get("Diagnosis")) if ledger_row is not None else "",
                "ImageID": clean(ledger_row.get("ImageID")) if ledger_row is not None else "",
                "Visit": clean(ledger_row.get("Visit")) if ledger_row is not None else "",
                "roisignals_file_count": int(inv_row["roisignals_file_count"].iloc[0]) if not inv_row.empty else 0,
                "stage_guesses": clean(inv_row["stage_guesses"].iloc[0]) if not inv_row.empty else "",
                "suffixes": clean(inv_row["suffixes"].iloc[0]) if not inv_row.empty else "",
                "roisignals_paths": clean(inv_row["roisignals_paths"].iloc[0]) if not inv_row.empty else "",
                "check_file_count": int(chk_row["check_file_count"].iloc[0]) if not chk_row.empty else 0,
                "check_paths": clean(chk_row["check_paths"].iloc[0]) if not chk_row.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)


def candidate_rank(row: pd.Series) -> Tuple[int, int, int, int, str]:
    qc_rank = 0 if clean(row.get("qc_signal_pass")) == "yes" else 1
    spectral_rank = {"bandpassed_like": 0, "uncertain": 1, "unfiltered_like": 2}.get(clean(row.get("spectral_class")), 3)
    suffix_rank = 0 if clean(row.get("suffix")) == ".mat" else 1
    size_rank = -int(float(row.get("file_size_bytes", 0) or 0))
    return qc_rank, spectral_rank, suffix_rank, size_rank, clean(row.get("path"))


def decide_subjects(qc: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    ledger_by_sid = ledger_lookup(ledger)
    rows: List[Dict[str, Any]] = []
    for sid, sub in qc.groupby("SubjectID"):
        sub = sub.copy()
        sub["_rank"] = sub.apply(candidate_rank, axis=1)
        sub = sub.sort_values("_rank", kind="mergesort")
        best = sub.iloc[0]
        ledger_row = ledger_by_sid.get(sid)
        in_ledger = ledger_row is not None
        reasons: List[str] = []
        if not in_ledger:
            reasons.append("subject_not_in_current_ledger")
        elif clean(ledger_row.get("dicom_series_ok")) == "NO":
            reasons.append("dicom_series_ok_NO")
        if clean(best.get("qc_signal_pass")) != "yes":
            reasons.append(f"signal_qc_fail:{clean(best.get('qc_signal_reason'))}")
        if int(best.get("n_rois", 0) or 0) != EXPECTED_ROIS:
            reasons.append(f"n_rois={best.get('n_rois')}, expected 170")
        if clean(best.get("scale_label")) != "around_10000_global_scaled":
            reasons.append(f"scale_label={clean(best.get('scale_label'))}")
        if clean(best.get("stage_guess")) not in {"ARWSDCFN", "ARWSDCF"}:
            reasons.append(f"stage_not_explicit_F:{clean(best.get('stage_guess'))}")
        if clean(best.get("spectral_class")) == "unfiltered_like":
            reasons.append("spectral_unfiltered_like")
        if in_ledger and clean(ledger_row.get("python_bandpass_requested")) != "NO":
            reasons.append(f"python_bandpass_requested={clean(ledger_row.get('python_bandpass_requested'))}")
        qc_pass_files = sub[sub["qc_signal_pass"].eq("yes")]
        stages = sorted(set(clean(x) for x in sub["stage_guess"] if clean(x)))
        same_subject_pair_ok = len(stages) == 1 and len(sub) <= 2 and set(sub["suffix"]).issubset({".mat", ".txt"})
        if len(qc_pass_files) > 2 or len(stages) > 1:
            reasons.append("duplicated_ambiguous_candidates")
        import_ready = "yes" if not reasons else "no"
        if import_ready == "yes":
            decision_reason = "ledger_match_dicom_ok_signal_qc_stage_bandpass_path_python_bandpass_NO"
        else:
            decision_reason = "; ".join(reasons)
        rows.append(
            {
                "SubjectID": sid,
                "import_ready": import_ready,
                "decision_reason": decision_reason,
                "selected_path": clean(best.get("path")),
                "selected_relative_path": clean(best.get("relative_path")),
                "selected_suffix": clean(best.get("suffix")),
                "stage_guess": clean(best.get("stage_guess")),
                "n_timepoints": best.get("n_timepoints"),
                "n_rois": best.get("n_rois"),
                "finite_fraction": best.get("finite_fraction"),
                "scale_label": clean(best.get("scale_label")),
                "qc_signal_pass": clean(best.get("qc_signal_pass")),
                "qc_signal_reason": clean(best.get("qc_signal_reason")),
                "qc_signal_warning": clean(best.get("qc_signal_warning")),
                "spectral_class": clean(best.get("spectral_class")),
                "energy_below_0p01": best.get("energy_below_0p01"),
                "energy_0p01_0p08": best.get("energy_0p01_0p08"),
                "energy_above_0p08": best.get("energy_above_0p08"),
                "in_ledger": "yes" if in_ledger else "no",
                "ledger_scope": clean(ledger_row.get("ledger_scope")) if in_ledger else "",
                "priority_batch": clean(ledger_row.get("priority_batch")) if in_ledger else "",
                "dicom_series_ok": clean(ledger_row.get("dicom_series_ok")) if in_ledger else "",
                "previous_processing_status": clean(ledger_row.get("processing_status")) if in_ledger else "",
                "python_bandpass_requested": clean(ledger_row.get("python_bandpass_requested")) if in_ledger else "",
                "matched_file_count": int(len(sub)),
                "same_subject_mat_txt_pair_only": "yes" if same_subject_pair_ok else "no",
                "all_candidate_paths": "|".join(clean(x) for x in sub["path"]),
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)


def append_note(existing: str, addition: str) -> str:
    existing = clean(existing)
    addition = clean(addition)
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing} | {addition}"


def update_candidate_ledger(ledger: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    if "ledger_scope" not in out.columns:
        out["ledger_scope"] = "cn_ge_request"
    decision_by_sid = {clean(row["SubjectID"]): row for _, row in decisions.iterrows()}
    for idx, row in out.iterrows():
        sid = clean(row["SubjectID"])
        decision = decision_by_sid.get(sid)
        if decision is None:
            continue
        note = (
            f"{BATCH_LABEL}: import_ready={clean(decision.get('import_ready'))}; "
            f"stage={clean(decision.get('stage_guess'))}; "
            f"n_rois={decision.get('n_rois')}; finite={float(decision.get('finite_fraction', np.nan)):.4f}; "
            f"scale={clean(decision.get('scale_label'))}; spectral={clean(decision.get('spectral_class'))}; "
            f"warning={clean(decision.get('qc_signal_warning'))}; "
            f"reason={clean(decision.get('decision_reason'))}"
        )
        out.at[idx, "notes_santiago"] = append_note(row.get("notes_santiago", ""), note)
        if clean(decision.get("import_ready")) == "yes":
            out.at[idx, "roisignals_status"] = "qc_pass"
            out.at[idx, "processing_status"] = "uploaded"
            out.at[idx, "uploaded_batch"] = BATCH_LABEL
            out.at[idx, "uploaded_path"] = clean(decision.get("selected_path"))
        else:
            if clean(decision.get("qc_signal_pass")) == "yes" and clean(row.get("dicom_series_ok")) == "NO":
                out.at[idx, "roisignals_status"] = "qc_pass"
                out.at[idx, "processing_status"] = "excluded"
            else:
                out.at[idx, "roisignals_status"] = "qc_fail"
                out.at[idx, "processing_status"] = "excluded" if clean(row.get("dicom_series_ok")) == "NO" else "pending"
    return out


def write_readme(
    output_dir: Path,
    inventory: pd.DataFrame,
    vs_ledger: pd.DataFrame,
    qc: pd.DataFrame,
    decisions: pd.DataFrame,
    candidate_ledger: pd.DataFrame,
    check_df: pd.DataFrame,
) -> None:
    total_files = int(len(inventory))
    total_subjects = int(inventory["SubjectID"].nunique()) if not inventory.empty else 0
    matched_subjects = int(decisions["in_ledger"].eq("yes").sum()) if not decisions.empty else 0
    priority_first_67 = int(decisions["priority_batch"].eq("first_67").sum()) if not decisions.empty else 0
    qc_pass_subjects = int(decisions["qc_signal_pass"].eq("yes").sum()) if not decisions.empty else 0
    import_ready = int(decisions["import_ready"].eq("yes").sum()) if not decisions.empty else 0
    outside_ledger = sorted(decisions.loc[decisions["in_ledger"].eq("no"), "SubjectID"].tolist()) if not decisions.empty else []
    dicom_no = sorted(vs_ledger.loc[vs_ledger["dicom_series_ok"].eq("NO"), "SubjectID"].tolist()) if not vs_ledger.empty else []
    ledger_dicom_no = sorted(candidate_ledger.loc[candidate_ledger["dicom_series_ok"].eq("NO"), "SubjectID"].tolist())
    safe_incremental = "YES" if import_ready > 0 else "NO"
    lines = [
        "# Martin Bandpass Batch 20260513 Audit",
        "",
        "Read-only audit. No training, no tensor construction, and no final Python bandpass were performed.",
        "",
        "## Answers",
        "",
        f"- ROISignals files in batch: `{total_files}`.",
        f"- Unique ROISignals subjects in batch: `{total_subjects}`.",
        f"- Check files inventoried: `{len(check_df)}`.",
        f"- Subjects matching current ledger: `{matched_subjects}`.",
        f"- Matched subjects in `priority_batch=first_67`: `{priority_first_67}`.",
        f"- Subjects passing signal QC: `{qc_pass_subjects}`.",
        f"- Subjects ready for connectivity/import: `{import_ready}`.",
        f"- Subjects outside current ledger: `{len(outside_ledger)}`" + (f" ({', '.join(outside_ledger[:20])}{'...' if len(outside_ledger) > 20 else ''})" if outside_ledger else "."),
        f"- Batch subjects marked DICOM NO: `{len(dicom_no)}`" + (f" ({', '.join(dicom_no)})" if dicom_no else "."),
        f"- DICOM NO rows retained in candidate ledger: `{len(ledger_dicom_no)}`" + (f" ({', '.join(ledger_dicom_no)})" if ledger_dicom_no else "."),
        f"- Safe to build an incremental tensor from import-ready rows? `{safe_incremental}`. Use only `batch_import_decision.csv` rows with `import_ready=yes`; do not use quarantined `v5.1_gecn9` as final.",
        "- Python bandpass final path: `OFF`. This audit only computes spectral diagnostics.",
        "",
        "## Recommended Next Command",
        "",
        "Do not build the full tensor yet. First review `batch_import_decision.csv` and use only rows with `import_ready=yes` for the next dry-run incremental connectivity build, still without Python bandpass and without overwriting v5/v5.1 tensors.",
        "",
        "## Outputs",
        "",
        "- `batch_roisignals_inventory.csv`",
        "- `batch_vs_ledger_match.csv`",
        "- `batch_roisignals_qc.csv`",
        "- `batch_spectral_qc.csv`",
        "- `batch_import_decision.csv`",
        "- `ADNI_v5_1_DATA_LEDGER_20260513_after_batch1_candidate.csv`",
        "- `batch_check_inventory.csv`",
        "- `command_log.json`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.output_dir)
    ledger = read_csv(args.ledger)
    check_df = discover_check_files(args.batch_root / "Check")
    inventory = discover_roisignals(args.batch_root, check_df)
    if inventory.empty:
        raise RuntimeError(f"No ROISignals files found under {args.batch_root / 'ResultsAAL3'}")
    vs_ledger = build_vs_ledger(inventory, ledger, check_df)
    qc, spectral = qc_roisignals(inventory)
    decisions = decide_subjects(qc, ledger)
    candidate_ledger = update_candidate_ledger(ledger, decisions)

    inventory.to_csv(args.output_dir / "batch_roisignals_inventory.csv", index=False)
    vs_ledger.to_csv(args.output_dir / "batch_vs_ledger_match.csv", index=False)
    qc.to_csv(args.output_dir / "batch_roisignals_qc.csv", index=False)
    spectral.to_csv(args.output_dir / "batch_spectral_qc.csv", index=False)
    decisions.to_csv(args.output_dir / "batch_import_decision.csv", index=False)
    candidate_ledger.to_csv(args.output_dir / "ADNI_v5_1_DATA_LEDGER_20260513_after_batch1_candidate.csv", index=False)
    check_df.to_csv(args.output_dir / "batch_check_inventory.csv", index=False)
    command = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "batch_root": str(args.batch_root.resolve()),
        "ledger": str(args.ledger.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "batch_label": BATCH_LABEL,
        "tr_seconds": TR_SECONDS,
        "python_bandpass_final_pipeline": False,
        "training_run": False,
        "tensor_constructed": False,
        "ledger_original_modified": False,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(args.output_dir, inventory, vs_ledger, qc, decisions, candidate_ledger, check_df)

    print(f"Wrote audit outputs to {args.output_dir}")
    print(f"roisignals_files={len(inventory)}")
    print(f"unique_subjects={inventory['SubjectID'].nunique()}")
    print(f"ledger_matches={decisions['in_ledger'].eq('yes').sum()}")
    print(f"priority_first_67={decisions['priority_batch'].eq('first_67').sum()}")
    print(f"qc_signal_pass={decisions['qc_signal_pass'].eq('yes').sum()}")
    print(f"import_ready={decisions['import_ready'].eq('yes').sum()}")
    print(f"outside_ledger={decisions['in_ledger'].eq('no').sum()}")
    print(f"dicom_no={decisions['dicom_series_ok'].eq('NO').sum()}")
    print("No training. No tensor construction. No Python bandpass final. Original ledger not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
