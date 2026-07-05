#!/usr/bin/env python3
"""Resolve CN-GE v4 membership discrepancy across tensors and metadata.

Read-only audit:
- no tensor extraction;
- no training;
- no source/output overwrites except this script's own output directory when
  --overwrite is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "ge_cn_v4_membership_discrepancy"
)
DEFAULT_RECOMMENDED = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_local_roisignals_audit"
    / "ge_cn_recommended_roisignals.csv"
)
DEFAULT_V5_1_AUGMENTED = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_gecn_augmented_manifest"
    / "adni_v5_1_gecn_augmented_first_visit_manifest.csv"
)
DEFAULT_V5_1_FIRST_VISIT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_master_manifest"
    / "adni_v5_1_master_subject_manifest_first_visit.csv"
)
DEFAULT_V4_TENSOR = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "GLOBAL_TENSOR_ADNI_expanded_v4_all_available.npz"
)
DEFAULT_V4_METADATA = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
DEFAULT_PAPER_METADATA = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
DEFAULT_PAPER_TENSOR = (
    PROJECT_ROOT
    / "data"
    / "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned"
    / "GLOBAL_TENSOR_from_AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced_v6.5.17_AAL3_131ROIs_OMST_GCE_Signed_GrangerLag1_ChNorm_ROIreorderedYeo17_ParallelTuned.npz"
)

PRIORITY19_CN_GE = [
    "005_S_0602",
    "005_S_0610",
    "005_S_6084",
    "005_S_6093",
    "009_S_0751",
    "009_S_6163",
    "009_S_6212",
    "009_S_6286",
    "010_S_6567",
    "135_S_6473",
    "135_S_6509",
    "135_S_6510",
    "135_S_4446",
    "135_S_4598",
    "135_S_5113",
    "135_S_6104",
    "135_S_6359",
    "135_S_6360",
    "135_S_6411",
]

SUBJECT_RE = re.compile(r"(?<!\d)(\d{3}_S_\d{4})(?!\d)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit where the 19 priority CN-GE subjects exist across ADNI v4/v5 tensors and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ge-cn-recommended", type=Path, default=DEFAULT_RECOMMENDED)
    parser.add_argument("--v4-expanded-tensor", type=Path, default=DEFAULT_V4_TENSOR)
    parser.add_argument("--v4-expanded-metadata", type=Path, default=DEFAULT_V4_METADATA)
    parser.add_argument("--historical-paper-tensor", type=Path, default=DEFAULT_PAPER_TENSOR)
    parser.add_argument("--historical-paper-metadata", type=Path, default=DEFAULT_PAPER_METADATA)
    parser.add_argument("--v5-manifest", type=Path, default=DEFAULT_V5_1_FIRST_VISIT)
    parser.add_argument("--v5-1-gecn-augmented-manifest", type=Path, default=DEFAULT_V5_1_AUGMENTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output directory exists and is not empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_subject(value: Any) -> str:
    match = SUBJECT_RE.search(clean_string(value).upper())
    return match.group(1).upper() if match else ""


def first_existing_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        hit = lower.get(candidate.lower())
        if hit:
            return hit
    return None


def load_target_subjects(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if path.exists():
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "recommended_path" in df.columns:
            df = df[df["recommended_path"].map(clean_string).astype(bool)].copy()
        for sid in sorted(set(df["SubjectID"].map(normalize_subject))):
            if sid:
                rows.append(
                    {
                        "SubjectID": sid,
                        "target_source": "ge_cn_recommended_physical",
                        "is_priority19_v4_cn_ge": sid in PRIORITY19_CN_GE,
                    }
                )
    for sid in PRIORITY19_CN_GE:
        if sid not in {row["SubjectID"] for row in rows}:
            rows.append(
                {
                    "SubjectID": sid,
                    "target_source": "priority19_fallback",
                    "is_priority19_v4_cn_ge": True,
                }
            )
    return pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)


def tensor_search_paths(explicit_paths: Sequence[Path]) -> List[Path]:
    roots = [PROJECT_ROOT / "data", Path("/media/diego/Datos/desde_cero")]
    paths: List[Path] = []
    for path in explicit_paths:
        if path.exists():
            paths.append(path)
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*GLOBAL_TENSOR*.npz")))
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def classify_tensor(path: Path, v4_tensor: Path, paper_tensor: Path) -> str:
    if path == v4_tensor:
        return "v4_expanded_tensor"
    if path == paper_tensor:
        return "historical_paper_tensor"
    text = str(path)
    if "adni_expanded_v4_all_available" in text:
        return "v4_expanded_tensor"
    if "AAL3_dynamicROIs_fmri_tensor_NeuroEnhanced" in text:
        return "historical_paper_tensor"
    if "adni_expanded_v" in text:
        return "adni_expanded_other_tensor"
    if "COVID" in text.upper():
        return "non_adni_covid_tensor"
    return "other_tensor"


def load_tensor_subject_ids(path: Path) -> Tuple[Set[str], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "tensor_path": str(path),
        "exists": path.exists(),
        "status": "",
        "subject_id_key": "",
        "n_subject_ids": 0,
        "keys": "",
        "shape_global_tensor_data": "",
    }
    if not path.exists():
        info["status"] = "missing"
        return set(), info
    try:
        with np.load(path, allow_pickle=False) as zf:
            info["keys"] = "|".join(zf.files)
            if "global_tensor_data" in zf.files:
                info["shape_global_tensor_data"] = str(tuple(int(x) for x in zf["global_tensor_data"].shape))
            sid_key = ""
            for key in ["subject_ids", "subjects", "SubjectID", "subject_id"]:
                if key in zf.files:
                    sid_key = key
                    break
            if not sid_key:
                info["status"] = "no_subject_ids_key"
                return set(), info
            ids = {normalize_subject(x) for x in zf[sid_key].astype(str)}
            ids = {sid for sid in ids if sid}
            info["status"] = "ok"
            info["subject_id_key"] = sid_key
            info["n_subject_ids"] = len(ids)
            return ids, info
    except Exception as exc:
        info["status"] = f"failed:{exc}"
        return set(), info


def metadata_search_paths(explicit_paths: Sequence[Path]) -> List[Path]:
    extra = [
        PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v1" / "subject_metadata_adni_expanded_v1.csv",
        PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v2" / "subject_metadata_adni_expanded_v2.csv",
        PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v3_all_available" / "subject_metadata_adni_expanded_v3_all_available.csv",
        PROJECT_ROOT / "data" / "revision_bspc_2026" / "adni_expanded_v4_all_available" / "subject_metadata_adni_expanded_v4_all_available.csv",
        PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv",
        Path("/home/diego/proyectos/betavae-xai-ad/data/SubjectsData_AAL3_procesado2.csv"),
    ]
    paths = list(explicit_paths) + extra
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if path.exists() and key not in seen:
            seen.add(key)
            out.append(path)
    return out


def classify_metadata(path: Path, v4_metadata: Path, paper_metadata: Path) -> str:
    if path == v4_metadata or "adni_expanded_v4_all_available" in str(path):
        return "v4_expanded_metadata"
    if path == paper_metadata or path.name == "SubjectsData_AAL3_procesado2.csv":
        return "historical_paper_metadata"
    if "adni_expanded_v" in str(path):
        return "adni_expanded_other_metadata"
    if "adni_v5_1" in str(path):
        return "v5_1_manifest"
    return "other_metadata"


def load_metadata_subject_ids(path: Path) -> Tuple[Set[str], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "metadata_path": str(path),
        "exists": path.exists(),
        "status": "",
        "subject_id_column": "",
        "n_subject_ids": 0,
        "columns": "",
    }
    if not path.exists():
        info["status"] = "missing"
        return set(), info
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=0)
        info["columns"] = "|".join(df.columns.astype(str))
        sid_col = first_existing_col(df.columns, ["SubjectID", "Subject", "PTID", "subject_id"])
        if sid_col is None:
            info["status"] = "no_subject_column"
            return set(), info
        full = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=[sid_col])
        ids = {normalize_subject(x) for x in full[sid_col]}
        ids = {sid for sid in ids if sid}
        info["status"] = "ok"
        info["subject_id_column"] = sid_col
        info["n_subject_ids"] = len(ids)
        return ids, info
    except Exception as exc:
        info["status"] = f"failed:{exc}"
        return set(), info


def bool_count(df: pd.DataFrame, col: str) -> int:
    return int(df[col].fillna(False).astype(bool).sum()) if col in df.columns else 0


def write_readme(output_dir: Path, matrix: pd.DataFrame, tensor_inv: pd.DataFrame, metadata_inv: pd.DataFrame) -> None:
    n = matrix["SubjectID"].nunique()
    v4_tensor_n = bool_count(matrix, "in_v4_expanded_tensor")
    v4_meta_n = bool_count(matrix, "in_v4_expanded_metadata")
    paper_tensor_n = bool_count(matrix, "in_historical_paper_tensor")
    paper_meta_n = bool_count(matrix, "in_historical_paper_metadata")
    v5_manifest_n = bool_count(matrix, "in_v5_manifest")
    aug_n = bool_count(matrix, "in_v5_1_gecn_augmented_manifest")
    v4_paths = sorted(set(matrix.loc[matrix["in_v4_expanded_tensor"].astype(bool), "v4_expanded_tensor_paths"].dropna().astype(str)))
    paper_paths = sorted(set(matrix.loc[matrix["in_historical_paper_tensor"].astype(bool), "historical_paper_tensor_paths"].dropna().astype(str)))
    resolved = (v4_meta_n == n and paper_tensor_n == 0)
    lines = [
        "# CN-GE v4 Membership Discrepancy Audit",
        "",
        "Read-only audit. No tensor extraction, no full tensor build, and no training were run.",
        "",
        "## Explicit Answers",
        "",
        f"- CN-GE target subjects audited: `{n}`.",
        f"- In v4 expanded tensor: `{v4_tensor_n}/{n}`.",
        f"- In v4 expanded metadata: `{v4_meta_n}/{n}`.",
        f"- In historical paper tensor: `{paper_tensor_n}/{n}`.",
        f"- In historical paper metadata: `{paper_meta_n}/{n}`.",
        f"- In v5.1 master manifest: `{v5_manifest_n}/{n}`.",
        f"- In v5.1 GECN augmented manifest: `{aug_n}/{n}`.",
        f"- v4 expanded tensor paths with hits: `{'; '.join(v4_paths) if v4_paths else 'none'}`.",
        f"- historical paper tensor paths with hits: `{'; '.join(paper_paths) if paper_paths else 'none'}`.",
        "",
        "## Resolution",
        "",
        (
            "The discrepancy is resolved: the previous `19/19` statement referred to v4 metadata / "
            "priority CN-GE tracking, while the later `0/19` came from checking the historical paper "
            "tensor used for the 431-subject paper dataset. These are different tensors/cohorts."
            if resolved
            else "The discrepancy is not fully resolved by this audit; inspect the membership matrix."
        ),
        "",
        "The smoke comparison against historical paper channel 1 correctly produced no comparable rows if `in_historical_paper_tensor` is 0/19.",
        "",
        "## Outputs",
        "",
        "- `ge_cn_19_membership_matrix.csv`",
        "- `tensor_membership_long.csv`",
        "- `tensor_inventory.csv`",
        "- `metadata_membership_long.csv`",
        "- `metadata_inventory.csv`",
        "",
        "## Next Command",
        "",
        "`/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/build_v5_1_gecn9_smoke_and_partial_branch.py --overwrite`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    targets = load_target_subjects(args.ge_cn_recommended)
    subject_ids = targets["SubjectID"].tolist()

    tensor_paths = tensor_search_paths([args.v4_expanded_tensor, args.historical_paper_tensor])
    tensor_rows: List[Dict[str, Any]] = []
    tensor_inventory_rows: List[Dict[str, Any]] = []
    tensor_hits_by_class: Dict[str, Dict[str, List[str]]] = {}
    for path in tensor_paths:
        ids, info = load_tensor_subject_ids(path)
        tensor_class = classify_tensor(path, args.v4_expanded_tensor, args.historical_paper_tensor)
        info["tensor_class"] = tensor_class
        tensor_inventory_rows.append(info)
        for sid in subject_ids:
            hit = sid in ids
            tensor_rows.append(
                {
                    "SubjectID": sid,
                    "tensor_path": str(path),
                    "tensor_class": tensor_class,
                    "in_tensor": hit,
                    "tensor_status": info["status"],
                    "n_subject_ids": info["n_subject_ids"],
                    "shape_global_tensor_data": info["shape_global_tensor_data"],
                }
            )
            if hit:
                tensor_hits_by_class.setdefault(tensor_class, {}).setdefault(sid, []).append(str(path))

    metadata_paths = metadata_search_paths(
        [args.v4_expanded_metadata, args.historical_paper_metadata, args.v5_manifest, args.v5_1_gecn_augmented_manifest]
    )
    metadata_rows: List[Dict[str, Any]] = []
    metadata_inventory_rows: List[Dict[str, Any]] = []
    metadata_hits_by_class: Dict[str, Dict[str, List[str]]] = {}
    for path in metadata_paths:
        ids, info = load_metadata_subject_ids(path)
        metadata_class = classify_metadata(path, args.v4_expanded_metadata, args.historical_paper_metadata)
        if path == args.v5_manifest:
            metadata_class = "v5_manifest"
        if path == args.v5_1_gecn_augmented_manifest:
            metadata_class = "v5_1_gecn_augmented_manifest"
        info["metadata_class"] = metadata_class
        metadata_inventory_rows.append(info)
        for sid in subject_ids:
            hit = sid in ids
            metadata_rows.append(
                {
                    "SubjectID": sid,
                    "metadata_path": str(path),
                    "metadata_class": metadata_class,
                    "in_metadata": hit,
                    "metadata_status": info["status"],
                    "n_subject_ids": info["n_subject_ids"],
                }
            )
            if hit:
                metadata_hits_by_class.setdefault(metadata_class, {}).setdefault(sid, []).append(str(path))

    matrix = targets.copy()
    class_to_col = {
        "v4_expanded_tensor": "in_v4_expanded_tensor",
        "historical_paper_tensor": "in_historical_paper_tensor",
    }
    for tensor_class, col in class_to_col.items():
        matrix[col] = matrix["SubjectID"].map(lambda sid: sid in tensor_hits_by_class.get(tensor_class, {}))
        matrix[f"{tensor_class}_paths"] = matrix["SubjectID"].map(
            lambda sid: "|".join(tensor_hits_by_class.get(tensor_class, {}).get(sid, []))
        )
    metadata_class_to_col = {
        "v4_expanded_metadata": "in_v4_expanded_metadata",
        "historical_paper_metadata": "in_historical_paper_metadata",
        "v5_manifest": "in_v5_manifest",
        "v5_1_gecn_augmented_manifest": "in_v5_1_gecn_augmented_manifest",
    }
    for metadata_class, col in metadata_class_to_col.items():
        matrix[col] = matrix["SubjectID"].map(lambda sid: sid in metadata_hits_by_class.get(metadata_class, {}))
        matrix[f"{metadata_class}_paths"] = matrix["SubjectID"].map(
            lambda sid: "|".join(metadata_hits_by_class.get(metadata_class, {}).get(sid, []))
        )

    tensor_long = pd.DataFrame(tensor_rows)
    tensor_inventory = pd.DataFrame(tensor_inventory_rows)
    metadata_long = pd.DataFrame(metadata_rows)
    metadata_inventory = pd.DataFrame(metadata_inventory_rows)

    matrix.to_csv(args.output_dir / "ge_cn_19_membership_matrix.csv", index=False)
    tensor_long.to_csv(args.output_dir / "tensor_membership_long.csv", index=False)
    tensor_inventory.to_csv(args.output_dir / "tensor_inventory.csv", index=False)
    metadata_long.to_csv(args.output_dir / "metadata_membership_long.csv", index=False)
    metadata_inventory.to_csv(args.output_dir / "metadata_inventory.csv", index=False)
    command = {
        "script": str(Path(__file__).resolve()),
        "ge_cn_recommended": str(args.ge_cn_recommended),
        "v4_expanded_tensor": str(args.v4_expanded_tensor),
        "v4_expanded_metadata": str(args.v4_expanded_metadata),
        "historical_paper_tensor": str(args.historical_paper_tensor),
        "historical_paper_metadata": str(args.historical_paper_metadata),
        "v5_manifest": str(args.v5_manifest),
        "v5_1_gecn_augmented_manifest": str(args.v5_1_gecn_augmented_manifest),
        "output_dir": str(args.output_dir),
        "overwrite": bool(args.overwrite),
        "training_run": False,
        "full_tensor_computed": False,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output_dir, matrix, tensor_inventory, metadata_inventory)
    print(f"Wrote CN-GE v4 membership discrepancy audit to {args.output_dir}")
    print(
        f"subjects={len(matrix)} "
        f"v4_tensor={bool_count(matrix, 'in_v4_expanded_tensor')}/{len(matrix)} "
        f"v4_metadata={bool_count(matrix, 'in_v4_expanded_metadata')}/{len(matrix)} "
        f"paper_tensor={bool_count(matrix, 'in_historical_paper_tensor')}/{len(matrix)} "
        f"paper_metadata={bool_count(matrix, 'in_historical_paper_metadata')}/{len(matrix)}"
    )
    print("No full tensor computed. No training run.")


if __name__ == "__main__":
    main()
