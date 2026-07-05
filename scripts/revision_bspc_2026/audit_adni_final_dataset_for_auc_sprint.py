#!/usr/bin/env python3
"""Read-only ADNI dataset freeze audit for the AUC sprint."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/dataset_freeze_audit"
DATASET_CANDIDATES = [
    {
        "name": "v5",
        "tensor": PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v5_all_available/GLOBAL_TENSOR_ADNI_expanded_v5_all_available.npz",
        "metadata": PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v5_all_available/subject_metadata_adni_expanded_v5_all_available.csv",
    },
    {
        "name": "v4",
        "tensor": PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v4_all_available/GLOBAL_TENSOR_ADNI_expanded_v4_all_available.npz",
        "metadata": PROJECT_ROOT / "data/revision_bspc_2026/adni_expanded_v4_all_available/subject_metadata_adni_expanded_v4_all_available.csv",
    },
]
NEW_AD_SUBJECTS = ["035_S_6927", "094_S_6736", "114_S_6039", "131_S_10801"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def load_npz_metadata(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "keys": []}
    if not path.exists():
        return out
    with np.load(path, allow_pickle=True) as z:
        out["keys"] = list(z.files)
        for key in ["subject_ids", "channel_names", "rois_count", "target_len_ts", "tr_seconds", "filter_low_hz", "filter_high_hz"]:
            if key in z.files:
                arr = z[key]
                if arr.shape == ():
                    out[key] = arr.item()
                elif arr.size <= 1000:
                    out[key] = arr.tolist()
                else:
                    out[f"{key}_shape"] = list(arr.shape)
    return out


def find_current_dataset() -> Dict[str, Path]:
    for cand in DATASET_CANDIDATES:
        if cand["tensor"].exists() and cand["metadata"].exists():
            return cand
    raise FileNotFoundError("No v5 or v4 ADNI expanded dataset found")


def image_col(df: pd.DataFrame) -> str:
    for col in ["ImageID", "Image Data ID", "IMAGEUID", "image_id"]:
        if col in df.columns:
            return col
    return ""


def visit_col(df: pd.DataFrame) -> str:
    for col in ["Visit", "VISCODE", "VisitCode", "visit"]:
        if col in df.columns:
            return col
    return ""


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    dataset = find_current_dataset()
    df = pd.read_csv(dataset["metadata"])
    subj_col = "SubjectID"
    dx_col = "ResearchGroup_Mapped"
    img_col = image_col(df)
    vis_col = visit_col(df)

    counts = df[dx_col].value_counts(dropna=False).rename_axis("ResearchGroup_Mapped").reset_index(name="n")
    manufacturer_dx = pd.crosstab(df["Manufacturer"], df[dx_col], dropna=False).reset_index()
    site_dx = pd.crosstab(df["Site3"], df[dx_col], dropna=False).reset_index()
    duplicate_subjects = df[df.duplicated(subj_col, keep=False)].sort_values(subj_col)
    duplicate_images = pd.DataFrame()
    if img_col:
        duplicate_images = df[df.duplicated(img_col, keep=False)].sort_values(img_col)

    new_rows: List[Dict[str, Any]] = []
    for sid in NEW_AD_SUBJECTS:
        sub = df[df[subj_col].astype(str).eq(sid)].copy()
        visits = "|".join(safe_text(x) for x in sub[vis_col].tolist()) if vis_col and not sub.empty else ""
        images = "|".join(safe_text(x) for x in sub[img_col].tolist()) if img_col and not sub.empty else ""
        new_rows.append(
            {
                "SubjectID": sid,
                "present": not sub.empty,
                "n_rows": len(sub),
                "diagnoses": "|".join(sorted(set(safe_text(x) for x in sub[dx_col].tolist() if safe_text(x)))),
                "visits": visits,
                "image_ids": images,
                "uses_only_one_scan": len(sub) == 1,
                "prefer_sc_status": "cannot_verify_no_visit_column" if not vis_col else ("sc_or_baseline" if any(str(v).lower() in {"sc", "screening", "bl"} for v in sub[vis_col].tolist()) else "not_sc_or_unknown"),
            }
        )
    new_ad = pd.DataFrame(new_rows)

    has_all_new_ad = bool(new_ad["present"].all())
    recommendation = "use_v4" if dataset["name"] == "v4" and not has_all_new_ad else "use_current_dataset"
    if not has_all_new_ad:
        recommendation = "build_v5_with_missing_new_AD_subjects"
    elif dataset["name"] == "v5":
        recommendation = "use_v5"

    counts.to_csv(outdir / "diagnosis_counts.csv", index=False)
    manufacturer_dx.to_csv(outdir / "manufacturer_diagnosis_table.csv", index=False)
    site_dx.to_csv(outdir / "site_diagnosis_table.csv", index=False)
    duplicate_subjects.to_csv(outdir / "duplicate_subjectid_rows.csv", index=False)
    duplicate_images.to_csv(outdir / "duplicate_imageid_rows.csv", index=False)
    new_ad.to_csv(outdir / "new_ad_subject_presence.csv", index=False)

    tensor_meta = load_npz_metadata(dataset["tensor"])
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash(),
        "selected_dataset_name": dataset["name"],
        "tensor_path": str(dataset["tensor"]),
        "metadata_path": str(dataset["metadata"]),
        "metadata_columns": list(df.columns),
        "n_total": int(len(df)),
        "diagnosis_counts": counts.to_dict(orient="records"),
        "n_duplicate_subject_rows": int(len(duplicate_subjects)),
        "n_duplicate_image_rows": int(len(duplicate_images)),
        "image_id_column": img_col,
        "visit_column": vis_col,
        "new_ad_subjects_all_present": has_all_new_ad,
        "recommendation": recommendation,
        "tensor_metadata": tensor_meta,
    }
    (outdir / "dataset_freeze_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    readme = [
        "# ADNI Dataset Freeze Audit",
        "",
        f"- Selected dataset: `{dataset['name']}`",
        f"- Tensor: `{dataset['tensor']}`",
        f"- Metadata: `{dataset['metadata']}`",
        f"- N total: {len(df)}",
        f"- Diagnosis counts: {counts.set_index('ResearchGroup_Mapped')['n'].to_dict()}",
        f"- Duplicate SubjectID rows: {len(duplicate_subjects)}",
        f"- Duplicate ImageID rows: {len(duplicate_images)}" if img_col else "- Duplicate ImageID rows: cannot check; no ImageID column in metadata",
        f"- New AD subjects all present: {has_all_new_ad}",
        f"- Recommendation: `{recommendation}`",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print("\n".join(readme))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
