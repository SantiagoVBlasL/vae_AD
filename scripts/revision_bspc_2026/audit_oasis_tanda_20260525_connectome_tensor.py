#!/usr/bin/env python
"""Audit OASIS connectome tensors after the guarded build.

This script is read-only. In --dry-run mode it reports whether expected tensor
files exist without failing if the build is still blocked.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONNECTOME_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectomes"
)
EXPECTED_TENSORS = [
    "tensor_concatenated_timeseries.npz",
    "tensor_runwise_connectome_average.npz",
]
EXPECTED_CHANNELS = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def load_expected_roi_names(connectome_dir: Path) -> list[str]:
    mapping_path = connectome_dir / "roi_mapping_used.csv"
    if not mapping_path.exists():
        return []
    mapping = pd.read_csv(mapping_path)
    if "ADNI_final_ROI_name" not in mapping:
        return []
    return mapping["ADNI_final_ROI_name"].astype(str).tolist()


def audit_tensor(path: Path, expected_roi_names: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "tensor_file": path.name,
            "exists": False,
            "status": "missing_not_built",
        }
    data = np.load(path, allow_pickle=True)
    tensor = data["global_tensor_data"] if "global_tensor_data" in data.files else None
    if tensor is None:
        return {
            "tensor_file": path.name,
            "exists": True,
            "status": "invalid_missing_global_tensor_data",
        }
    channel_names = data["channel_names"].astype(str).tolist() if "channel_names" in data.files else []
    roi_names = data["roi_names_in_order"].astype(str).tolist() if "roi_names_in_order" in data.files else []
    subject_ids = data["subject_ids"].astype(str).tolist() if "subject_ids" in data.files else []
    expected_shape = (len(subject_ids), len(EXPECTED_CHANNELS), 131, 131)
    finite_fraction = float(np.isfinite(tensor).mean())
    symmetric = bool(np.allclose(tensor, np.swapaxes(tensor, -1, -2), equal_nan=False, atol=1e-5))
    diagonal_abs_max = float(np.nanmax(np.abs(np.diagonal(tensor, axis1=-2, axis2=-1))))
    return {
        "tensor_file": path.name,
        "exists": True,
        "status": "pass" if tuple(tensor.shape) == expected_shape and finite_fraction == 1.0 else "check",
        "shape": str(tuple(tensor.shape)),
        "expected_shape": str(expected_shape),
        "n_subjects": int(len(subject_ids)),
        "n_channels": int(tensor.shape[1]) if tensor.ndim == 4 else "",
        "n_rois": int(tensor.shape[2]) if tensor.ndim == 4 else "",
        "finite_fraction": finite_fraction,
        "symmetric": symmetric,
        "diagonal_abs_max": diagonal_abs_max,
        "channel_names_match": channel_names == EXPECTED_CHANNELS,
        "roi_names_match_mapping": bool(expected_roi_names) and roi_names == expected_roi_names,
        "build_candidate": str(data["build_candidate"]) if "build_candidate" in data.files else "",
        "external_validation_only": bool(data["external_validation_only"]) if "external_validation_only" in data.files else "",
    }


def main() -> None:
    args = parse_args()
    args.connectome_dir.mkdir(parents=True, exist_ok=True)
    expected_roi_names = load_expected_roi_names(args.connectome_dir)
    rows = [audit_tensor(args.connectome_dir / name, expected_roi_names) for name in EXPECTED_TENSORS]
    audit_df = pd.DataFrame(rows)
    write_csv_md(
        audit_df,
        args.connectome_dir / "connectome_tensor_audit.csv",
        args.connectome_dir / "connectome_tensor_audit.md",
        "OASIS Connectome Tensor Audit",
    )
    any_missing = audit_df["status"].astype(str).eq("missing_not_built").any()
    decision = "pending_build" if any_missing else "audit_complete"
    if any_missing and not args.dry_run:
        decision = "missing_expected_tensors"

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "connectome_dir": str(args.connectome_dir),
        "dry_run": bool(args.dry_run),
        "read_only": True,
        "decision": decision,
        "expected_tensors": EXPECTED_TENSORS,
    }
    (args.connectome_dir / "connectome_tensor_audit_command_log.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"connectome_dir": str(args.connectome_dir), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
