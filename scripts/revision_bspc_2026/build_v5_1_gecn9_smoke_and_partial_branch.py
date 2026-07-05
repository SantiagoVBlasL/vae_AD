#!/usr/bin/env python3
"""Build a GECN9-only smoke/partial branch for ADNI v5.1.

This computes tensors only for the 9 CN-GE subjects selected as
direct-compatible-with-warning by the local ROISignals audit. It does not build
the full v5.1 tensor and does not train models.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    extract_one_subject,
)


DATASET_NAME = "adni_expanded_v5_1_gecn9_no_pybandpass_partial"
DEFAULT_SOURCES_TO_ADD = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_augmented_manifest"
    / "adni_v5_1_gecn_sources_to_add.csv"
)
DEFAULT_MEMBERSHIP_README = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "ge_cn_v4_membership_discrepancy"
    / "README.md"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_gecn9_no_pybandpass_partial"
)
DEFAULT_LOCAL_SYMLINK = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_gecn9_no_pybandpass_partial"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a GECN9-only smoke tensor branch with Python bandpass OFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sources-to-add", type=Path, default=DEFAULT_SOURCES_TO_ADD)
    parser.add_argument("--membership-readme", type=Path, default=DEFAULT_MEMBERSHIP_README)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--no-symlink", action="store_true")
    return parser.parse_args()


def prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output root exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def yes_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().isin({"yes", "true", "1"})


def load_sources(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"SubjectID", "recommended_path", "recommended_compatibility", "recommended_to_add"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    selected = df[
        df["recommended_to_add"].eq("yes")
        & df["recommended_compatibility"].eq("direct_compatible_with_warning")
        & df["recommended_path"].map(clean_string).astype(bool)
    ].copy()
    selected = selected.drop_duplicates("SubjectID", keep="first").sort_values("SubjectID")
    if len(selected) != 9:
        raise RuntimeError(f"Expected exactly 9 GECN warning subjects, found {len(selected)}")
    missing_files = [p for p in selected["recommended_path"] if not Path(p).exists()]
    if missing_files:
        raise RuntimeError(f"Missing selected ROISignals files: {missing_files}")
    selected["signal_path"] = selected["recommended_path"]
    selected["source_label"] = "v5_1_gecn9_ARWSDCF_warning"
    selected["v5_1_gecn9_candidate"] = True
    selected["preprocessing_source"] = PREPROCESSING_SOURCE
    selected["python_bandpass_applied"] = False
    selected["dataset_name"] = DATASET_NAME
    selected["target_len"] = TARGET_LEN
    selected["TR"] = TR_SECONDS
    selected["roi_output_count"] = OUTPUT_ROIS
    return selected


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


def run_gecn9_extraction(manifest: pd.DataFrame, output_root: Path, n_jobs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tensor_dir = output_root / "subject_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    roi_info = build_roi_reduction_and_order()
    records = manifest.to_dict(orient="records")
    results: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    if n_jobs <= 1:
        for row in records:
            res = extract_one_subject(row, tensor_dir, roi_info, write_tensor=True, pairwise_n_jobs=1)
            qc_rows.extend({"SubjectID": res["SubjectID"], **qc} for qc in res.pop("_qc_rows", []))
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(extract_one_subject, row, tensor_dir, roi_info, True, 1): row["SubjectID"] for row in records}
            for future in as_completed(futures):
                res = future.result()
                qc_rows.extend({"SubjectID": res["SubjectID"], **qc} for qc in res.pop("_qc_rows", []))
                results.append(res)
    return pd.DataFrame(results).sort_values("SubjectID"), pd.DataFrame(qc_rows)


def assemble_global_if_passed(output_root: Path, extraction_qc: pd.DataFrame) -> Optional[Path]:
    if extraction_qc.empty or not extraction_qc["status"].eq("ok").all():
        return None
    tensor_list = []
    subject_ids = []
    for _, row in extraction_qc.sort_values("SubjectID").iterrows():
        with np.load(row["tensor_path"], allow_pickle=False) as zf:
            tensor = zf["tensor"].astype(np.float32)
        if tensor.shape != (len(CHANNEL_NAMES), OUTPUT_ROIS, OUTPUT_ROIS):
            return None
        if np.isnan(tensor).any():
            return None
        tensor_list.append(tensor)
        subject_ids.append(row["SubjectID"])
    roi_info = build_roi_reduction_and_order()
    global_tensor = np.stack(tensor_list, axis=0).astype(np.float32)
    global_path = output_root / "GLOBAL_TENSOR_ADNI_expanded_v5_1_gecn9_no_pybandpass_partial.npz"
    np.savez_compressed(
        global_path,
        global_tensor_data=global_tensor,
        subject_ids=np.asarray(subject_ids),
        channel_names=np.asarray(CHANNEL_NAMES),
        rois_count=np.asarray(OUTPUT_ROIS),
        target_len_ts=np.asarray(TARGET_LEN),
        tr_seconds=np.asarray(TR_SECONDS),
        python_bandpass_applied=np.asarray(False),
        preprocessing_source=np.asarray(PREPROCESSING_SOURCE),
        dataset_name=np.asarray(DATASET_NAME),
        branch_scope=np.asarray("GECN9_partial_smoke_only"),
        roi_order_name=np.asarray("aal3_manual_yeo17_order"),
        roi_names_in_order=np.asarray(roi_info["roi_names_new_order"]),
        network_labels_in_order=np.asarray(roi_info["network_labels_new_order"]),
    )
    return global_path


def write_readme(
    output_root: Path,
    manifest: pd.DataFrame,
    extraction_qc: pd.DataFrame,
    channel_qc: pd.DataFrame,
    global_path: Optional[Path],
    symlink_status: str,
    elapsed_sec: float,
) -> None:
    ok_n = int(extraction_qc["status"].eq("ok").sum()) if not extraction_qc.empty else 0
    passed = ok_n == len(manifest) and global_path is not None
    subjects = ", ".join(manifest["SubjectID"].tolist())
    lines = [
        "# ADNI v5.1 GECN9 No-Python-Bandpass Partial Branch",
        "",
        "This is a partial branch containing only the 9 selected CN-GE subjects. It is not the full v5.1 tensor and was not used for training.",
        "",
        "## Method",
        "",
        f"- dataset_name: `{DATASET_NAME}`",
        "- scope: `GECN9_partial_smoke_only`",
        "- Python bandpass: `OFF`",
        "- input: local DPARSF/MATLAB `ROISignals_AAL3_FunImgARWSDCF` files selected by the GECN audit",
        f"- TR: `{TR_SECONDS}`",
        f"- target_len: `{TARGET_LEN}`",
        f"- ROI output count: `{OUTPUT_ROIS}`",
        f"- channels: `{'|'.join(CHANNEL_NAMES)}`",
        "",
        "## Smoke Result",
        "",
        f"- requested subjects: `{len(manifest)}`",
        f"- succeeded subjects: `{ok_n}`",
        f"- smoke passed: `{'YES' if passed else 'NO'}`",
        f"- global partial tensor: `{global_path or ''}`",
        f"- local symlink status: `{symlink_status}`",
        f"- elapsed seconds: `{elapsed_sec:.2f}`",
        f"- subjects: `{subjects}`",
        "",
        "## QC",
        "",
        f"- extraction QC rows: `{len(extraction_qc)}`",
        f"- channel QC rows: `{len(channel_qc)}`",
        f"- tensor NaNs total: `{int(extraction_qc['tensor_nan_count'].sum()) if 'tensor_nan_count' in extraction_qc else 'NA'}`",
        "",
        "## Next Step",
        "",
        "Review this branch and the warning-stage provenance before building a full v5.1 tensor. No training has been run.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    prepare_output_root(args.output_root, args.overwrite)
    manifest = load_sources(args.sources_to_add)
    manifest.to_csv(args.output_root / "subject_manifest_v5_1_gecn9_no_pybandpass_partial.csv", index=False)
    metadata_cols = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Age",
        "Sex",
        "Manufacturer",
        "ImageID",
        "Visit",
        "recommended_stage_guess",
        "recommended_compatibility",
        "compatibility_reason",
        "signal_path",
    ]
    manifest[[c for c in metadata_cols if c in manifest.columns]].to_csv(
        args.output_root / "training_ready_metadata_v5_1_gecn9_no_pybandpass_partial.csv",
        index=False,
    )
    extraction_qc, channel_qc = run_gecn9_extraction(manifest, args.output_root, max(1, args.n_jobs))
    extraction_qc.to_csv(args.output_root / "gecn9_smoke_extraction_qc.csv", index=False)
    channel_qc.to_csv(args.output_root / "gecn9_smoke_channel_qc.csv", index=False)
    global_path = assemble_global_if_passed(args.output_root, extraction_qc)
    symlink_status = "skipped_by_user"
    if not args.no_symlink:
        symlink_status = create_symlink(args.local_symlink, args.output_root, args.overwrite)
    command = {
        "script": str(Path(__file__).resolve()),
        "sources_to_add": str(args.sources_to_add),
        "membership_readme": str(args.membership_readme),
        "output_root": str(args.output_root),
        "local_symlink": str(args.local_symlink),
        "overwrite": bool(args.overwrite),
        "n_jobs": int(args.n_jobs),
        "python_bandpass_applied": False,
        "full_v5_1_tensor_computed": False,
        "training_run": False,
    }
    (args.output_root / "command_log.json").write_text(json.dumps(command, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elapsed = time.time() - started
    write_readme(args.output_root, manifest, extraction_qc, channel_qc, global_path, symlink_status, elapsed)
    ok_n = int(extraction_qc["status"].eq("ok").sum()) if not extraction_qc.empty else 0
    print(f"Wrote GECN9 partial branch to {args.output_root}")
    print(f"subjects={len(manifest)} ok={ok_n} global_tensor={global_path or ''}")
    print("No full v5.1 tensor computed. No training run.")


if __name__ == "__main__":
    main()
