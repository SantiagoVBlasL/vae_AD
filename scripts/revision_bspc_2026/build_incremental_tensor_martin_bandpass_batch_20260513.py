#!/usr/bin/env python3
"""Build an incremental no-Python-bandpass tensor for Martin batch 20260513.

This script uses only rows marked import_ready=yes by the batch audit. It does
not assemble with v5/v5.1 and does not train.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_v5_dparsf10000_no_pybandpass_manifest_and_extract import (  # noqa: E402
    CHANNEL_NAMES,
    OUTPUT_ROIS,
    PREPROCESSING_SOURCE,
    TARGET_LEN,
    TR_SECONDS,
    build_roi_reduction_and_order,
    compute_channels,
    load_signal,
    preprocess_timeseries_no_pybandpass,
)


DEFAULT_DECISION_CSV = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260513"
    / "batch_import_decision.csv"
)
DEFAULT_LEDGER_CSV = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "martin_bandpass_batch_20260513"
    / "ADNI_v5_1_DATA_LEDGER_20260513_after_batch1_candidate.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_v5_1_batch20260513_incremental_no_pybandpass"
)
DEFAULT_LOCAL_SYMLINK = (
    PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_v5_1_batch20260513_incremental_no_pybandpass"
)

DATASET_NAME = "adni_v5_1_batch20260513_incremental_no_pybandpass"
EXPECTED_IMPORT_READY_N = 42
FINAL_TENSOR_NAME = "INCREMENTAL_TENSOR_MARTIN_BANDPASS_BATCH20260513.npz"
PYTHON_BANDPASS_APPLIED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build incremental tensor for Martin bandpass batch 20260513.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--decision-csv", type=Path, default=DEFAULT_DECISION_CSV)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-n-subjects", type=int, default=0)
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-symlink", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--pairwise-n-jobs", type=int, default=1)
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


def import_ready_rows(decision_csv: Path) -> pd.DataFrame:
    df = read_csv(decision_csv)
    ready = df[df["import_ready"].astype(str).str.lower().eq("yes")].copy()
    ready = ready.sort_values("SubjectID").reset_index(drop=True)
    if len(ready) != EXPECTED_IMPORT_READY_N:
        raise RuntimeError(f"Expected {EXPECTED_IMPORT_READY_N} import_ready=yes rows, found {len(ready)}")
    if ready["SubjectID"].duplicated().any():
        dup = sorted(ready.loc[ready["SubjectID"].duplicated(), "SubjectID"].unique())
        raise RuntimeError(f"Duplicate import-ready SubjectID rows: {dup}")
    if not ready["python_bandpass_requested"].eq("NO").all():
        bad = ready.loc[~ready["python_bandpass_requested"].eq("NO"), "SubjectID"].tolist()
        raise RuntimeError(f"python_bandpass_requested must be NO for every import-ready row: {bad}")
    return ready


def ledger_lookup(path: Path) -> Dict[str, pd.Series]:
    if not path.exists():
        return {}
    ledger = read_csv(path)
    return {clean(row["SubjectID"]): row for _, row in ledger.iterrows()}


def build_metadata(ready: pd.DataFrame, ledger_by_sid: Dict[str, pd.Series]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, row in ready.reset_index(drop=True).iterrows():
        sid = clean(row["SubjectID"])
        ledger = ledger_by_sid.get(sid)
        rows.append(
            {
                "tensor_index": idx,
                "SubjectID": sid,
                "ImageID": clean(ledger.get("ImageID")) if ledger is not None else "",
                "Visit": clean(ledger.get("Visit")) if ledger is not None else "",
                "Diagnosis": clean(ledger.get("Diagnosis")) if ledger is not None else "",
                "Manufacturer": clean(ledger.get("Manufacturer")) if ledger is not None else "",
                "ledger_scope": clean(ledger.get("ledger_scope")) if ledger is not None else "",
                "priority_batch": clean(ledger.get("priority_batch")) if ledger is not None else clean(row.get("priority_batch")),
                "dicom_series_ok": clean(ledger.get("dicom_series_ok")) if ledger is not None else clean(row.get("dicom_series_ok")),
                "source_batch": "20260513_bandpass_batch1",
                "selected_path": clean(row["selected_path"]),
                "selected_relative_path": clean(row["selected_relative_path"]),
                "stage_guess": clean(row["stage_guess"]),
                "n_timepoints_raw": clean(row.get("n_timepoints")),
                "n_rois_raw": clean(row.get("n_rois")),
                "finite_fraction": clean(row.get("finite_fraction")),
                "scale_label": clean(row.get("scale_label")),
                "spectral_class": clean(row.get("spectral_class")),
                "python_bandpass_requested": clean(row.get("python_bandpass_requested")),
                "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
                "included_in_dataset_version": DATASET_NAME,
            }
        )
    return pd.DataFrame(rows)


def process_one_subject(task: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    started = time.time()
    sid = clean(task["SubjectID"])
    path = Path(clean(task["selected_path"]))
    roi_info = task["roi_info"]
    pairwise_n_jobs = int(task.get("pairwise_n_jobs", 1))
    qc: Dict[str, Any] = {
        "SubjectID": sid,
        "selected_path": str(path),
        "status": "pending",
        "raw_shape": "",
        "load_status": "",
        "preprocess_status": "",
        "processed_shape": "",
        "tensor_shape": "",
        "tensor_dtype": "",
        "tensor_nan_count": np.nan,
        "tensor_min": np.nan,
        "tensor_max": np.nan,
        "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
        "elapsed_sec": np.nan,
    }
    channel_rows: List[Dict[str, Any]] = []
    try:
        raw, _var, raw_shape, load_status = load_signal(path)
        qc["raw_shape"] = raw_shape
        qc["load_status"] = load_status
        if raw is None:
            qc["status"] = "load_failed"
            return None, qc, channel_rows
        ts, pre_status, pre_qc = preprocess_timeseries_no_pybandpass(raw, roi_info)
        qc.update(pre_qc)
        qc["preprocess_status"] = pre_status
        if ts is None:
            qc["status"] = "preprocess_failed"
            return None, qc, channel_rows
        if tuple(ts.shape) != (TARGET_LEN, OUTPUT_ROIS):
            qc["status"] = f"bad_processed_shape:{tuple(ts.shape)}"
            return None, qc, channel_rows
        tensor, statuses, raw_channel_rows = compute_channels(ts, pairwise_n_jobs=pairwise_n_jobs)
        qc["channel_statuses"] = json.dumps(statuses, sort_keys=True)
        qc["tensor_shape"] = str(tuple(int(x) for x in tensor.shape))
        qc["tensor_dtype"] = str(tensor.dtype)
        qc["tensor_nan_count"] = int(np.isnan(tensor).sum())
        qc["tensor_min"] = float(np.nanmin(tensor))
        qc["tensor_max"] = float(np.nanmax(tensor))
        if tensor.shape != (len(CHANNEL_NAMES), OUTPUT_ROIS, OUTPUT_ROIS):
            qc["status"] = f"bad_tensor_shape:{tensor.shape}"
            return None, qc, channel_rows
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
            qc["tensor_dtype"] = str(tensor.dtype)
        if np.isnan(tensor).any():
            qc["status"] = f"tensor_has_nans:{int(np.isnan(tensor).sum())}"
            return None, qc, channel_rows
        qc["status"] = "ok"
        channel_rows = [{"SubjectID": sid, **x} for x in raw_channel_rows]
        return tensor, qc, channel_rows
    except Exception as exc:
        qc["status"] = f"failed:{exc}"
        return None, qc, channel_rows
    finally:
        qc["elapsed_sec"] = time.time() - started


def run_subjects(rows: pd.DataFrame, n_jobs: int, pairwise_n_jobs: int) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    roi_info = build_roi_reduction_and_order()
    tasks = []
    for _, row in rows.iterrows():
        payload = row.to_dict()
        payload["roi_info"] = roi_info
        payload["pairwise_n_jobs"] = pairwise_n_jobs
        tasks.append(payload)

    results: Dict[str, np.ndarray] = {}
    qc_rows: List[Dict[str, Any]] = []
    channel_rows: List[Dict[str, Any]] = []
    if n_jobs <= 1 or len(tasks) <= 1:
        for task in tasks:
            tensor, qc, ch_qc = process_one_subject(task)
            qc_rows.append(qc)
            channel_rows.extend(ch_qc)
            if tensor is not None and qc["status"] == "ok":
                results[qc["SubjectID"]] = tensor
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(process_one_subject, task): clean(task["SubjectID"]) for task in tasks}
            for future in as_completed(futures):
                tensor, qc, ch_qc = future.result()
                qc_rows.append(qc)
                channel_rows.extend(ch_qc)
                if tensor is not None and qc["status"] == "ok":
                    results[qc["SubjectID"]] = tensor

    qc_df = pd.DataFrame(qc_rows).sort_values("SubjectID").reset_index(drop=True)
    channel_df = pd.DataFrame(channel_rows).sort_values(["SubjectID", "channel_name"]).reset_index(drop=True)
    failed = qc_df[~qc_df["status"].eq("ok")]
    if not failed.empty:
        raise RuntimeError(f"Subject extraction failures: {failed[['SubjectID', 'status']].to_dict(orient='records')}")
    ordered_tensors = []
    missing = []
    for sid in rows["SubjectID"].astype(str).tolist():
        if sid not in results:
            missing.append(sid)
        else:
            ordered_tensors.append(results[sid])
    if missing:
        raise RuntimeError(f"Missing tensors after extraction: {missing}")
    return np.stack(ordered_tensors, axis=0).astype(np.float32), qc_df, channel_df


def create_symlink(link: Path, target: Path, overwrite: bool) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        current = link.resolve()
        if current == target.resolve():
            return "already_correct"
        if not overwrite:
            return f"existing_symlink_points_elsewhere:{current}"
        link.unlink()
    elif link.exists():
        return "existing_non_symlink_not_modified"
    link.symlink_to(target, target_is_directory=True)
    return "created"


def validate_tensor(tensor: np.ndarray, expected_n: int) -> Dict[str, Any]:
    return {
        "expected_n": expected_n,
        "shape": list(tensor.shape),
        "shape_ok": tensor.shape == (expected_n, len(CHANNEL_NAMES), OUTPUT_ROIS, OUTPUT_ROIS),
        "dtype": str(tensor.dtype),
        "dtype_ok": tensor.dtype == np.float32,
        "nan_count": int(np.isnan(tensor).sum()),
        "nan_count_ok": int(np.isnan(tensor).sum()) == 0,
        "python_bandpass_applied": PYTHON_BANDPASS_APPLIED,
        "python_bandpass_ok": PYTHON_BANDPASS_APPLIED is False,
    }


def write_outputs(
    output_root: Path,
    tensor: np.ndarray,
    metadata: pd.DataFrame,
    qc_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    mode_label: str,
    symlink_status: str,
    validation: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if mode_label == "full":
        tensor_path = output_root / FINAL_TENSOR_NAME
        metadata_path = output_root / "incremental_subject_metadata.csv"
        qc_path = output_root / "incremental_qc_summary.csv"
        channel_path = output_root / "incremental_channel_qc.csv"
    else:
        tensor_path = output_root / f"SMOKE_N{tensor.shape[0]}_{FINAL_TENSOR_NAME}"
        metadata_path = output_root / f"smoke_n{tensor.shape[0]}_subject_metadata.csv"
        qc_path = output_root / f"smoke_n{tensor.shape[0]}_qc_summary.csv"
        channel_path = output_root / f"smoke_n{tensor.shape[0]}_channel_qc.csv"

    np.savez_compressed(
        tensor_path,
        global_tensor_data=tensor.astype(np.float32, copy=False),
        subject_ids=np.asarray(metadata["SubjectID"].astype(str).tolist(), dtype="U32"),
        channel_names=np.asarray(CHANNEL_NAMES, dtype="U64"),
        rois_count=np.asarray(OUTPUT_ROIS, dtype=np.int32),
        target_len_ts=np.asarray(TARGET_LEN, dtype=np.int32),
        tr_seconds=np.asarray(TR_SECONDS, dtype=np.float32),
        python_bandpass_applied=np.asarray(PYTHON_BANDPASS_APPLIED, dtype=np.bool_),
        preprocessing_source=np.asarray(PREPROCESSING_SOURCE, dtype="U96"),
        dataset_name=np.asarray(DATASET_NAME, dtype="U96"),
        source_batch=np.asarray("20260513_bandpass_batch1", dtype="U32"),
        uses_v5_1_gecn9=np.asarray(False, dtype=np.bool_),
        assembled_with_v5=np.asarray(False, dtype=np.bool_),
    )
    metadata.to_csv(metadata_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    channel_df.to_csv(channel_path, index=False)

    readme_lines = [
        "# ADNI v5.1 Martin Bandpass Batch 20260513 Incremental Tensor",
        "",
        f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"Mode: `{mode_label}`",
        "",
        "## Safety",
        "",
        "- Python bandpass final path: `OFF`.",
        "- v5.1_gecn9 was not used; it remains quarantine.",
        "- v5 was not overwritten.",
        "- No training was run.",
        "- This output is not assembled with v5.",
        "",
        "## Tensor",
        "",
        f"- Tensor file: `{tensor_path.name}`",
        f"- Shape: `{tuple(tensor.shape)}`",
        f"- dtype: `{tensor.dtype}`",
        f"- NaNs: `{int(np.isnan(tensor).sum())}`",
        f"- channels: `{'|'.join(CHANNEL_NAMES)}`",
        f"- ROI mapping: `AAL3 170 -> {OUTPUT_ROIS}` using v5 `build_roi_reduction_and_order()`",
        f"- target_len: `{TARGET_LEN}`",
        f"- TR: `{TR_SECONDS}`",
        "",
        "## Validation",
        "",
        f"- expected_n: `{validation['expected_n']}`",
        f"- shape_ok: `{validation['shape_ok']}`",
        f"- dtype_ok: `{validation['dtype_ok']}`",
        f"- nan_count_ok: `{validation['nan_count_ok']}`",
        f"- python_bandpass_applied: `{validation['python_bandpass_applied']}`",
        "",
        "## Symlink",
        "",
        f"- local symlink status: `{symlink_status}`",
        "",
        "## Next Step",
        "",
        "Review this incremental output before any assembly with v5. Do not train until the final v5.1 dataset is explicitly confirmed.",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def dry_run_report(ready: pd.DataFrame, metadata: pd.DataFrame, args: argparse.Namespace) -> None:
    print("Dry run only; no outputs written.")
    print(f"import_ready_rows={len(ready)}")
    print(f"expected_import_ready_rows={EXPECTED_IMPORT_READY_N}")
    print(f"output_root={args.output_root}")
    print(f"local_symlink={args.local_symlink}")
    print(f"first_subjects={metadata['SubjectID'].head(10).tolist()}")
    print("python_bandpass_applied=False")
    print("uses_v5_1_gecn9=False")
    print("assembled_with_v5=False")
    print("training_run=False")


def main() -> int:
    args = parse_args()
    if not args.dry_run and args.smoke_n_subjects <= 0 and not args.run_full:
        raise SystemExit("Pass --dry-run, --smoke-n-subjects N, or --run-full.")

    ready = import_ready_rows(args.decision_csv)
    ledger_by_sid = ledger_lookup(args.ledger_csv)
    metadata = build_metadata(ready, ledger_by_sid)

    if args.dry_run:
        dry_run_report(ready, metadata, args)
        return 0

    mode_label = "full" if args.run_full else f"smoke_n{args.smoke_n_subjects}"
    selected = ready.copy()
    selected_metadata = metadata.copy()
    expected_n = EXPECTED_IMPORT_READY_N
    if not args.run_full:
        n = int(args.smoke_n_subjects)
        if n <= 0:
            raise SystemExit("--smoke-n-subjects must be >0 unless --run-full is set")
        selected = selected.head(n).reset_index(drop=True)
        selected_metadata = selected_metadata.head(n).copy()
        selected_metadata["tensor_index"] = np.arange(len(selected_metadata))
        expected_n = len(selected)

    if args.run_full:
        final_tensor_path = args.output_root / FINAL_TENSOR_NAME
        if final_tensor_path.exists() and not args.overwrite:
            raise RuntimeError(f"Final tensor already exists; pass --overwrite to replace: {final_tensor_path}")

    tensor, qc_df, channel_df = run_subjects(
        selected,
        n_jobs=max(1, int(args.n_jobs)),
        pairwise_n_jobs=max(1, int(args.pairwise_n_jobs)),
    )
    validation = validate_tensor(tensor, expected_n)
    if not all([validation["shape_ok"], validation["dtype_ok"], validation["nan_count_ok"], validation["python_bandpass_ok"]]):
        raise RuntimeError(f"Tensor validation failed: {validation}")

    symlink_status = create_symlink(args.local_symlink, args.output_root, overwrite=args.overwrite_symlink)
    write_outputs(args.output_root, tensor, selected_metadata, qc_df, channel_df, "full" if args.run_full else "smoke", symlink_status, validation)

    print(f"mode={mode_label}")
    print(f"output_root={args.output_root}")
    print(f"local_symlink_status={symlink_status}")
    print(f"tensor_shape={tuple(tensor.shape)}")
    print(f"tensor_dtype={tensor.dtype}")
    print(f"tensor_nan_count={int(np.isnan(tensor).sum())}")
    print(f"python_bandpass_applied={PYTHON_BANDPASS_APPLIED}")
    print("uses_v5_1_gecn9=False")
    print("assembled_with_v5=False")
    print("training_run=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
