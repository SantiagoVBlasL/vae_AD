#!/usr/bin/env python3
"""
Select Philips-CN candidates for a new external stress test.

This script is intentionally non-destructive: it reads metadata/NPZ files and
only writes small selection tables/text files for the BSPC 2026 revision.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "adni_download_now.csv"
DEFAULT_HISTORICAL = PROJECT_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
DEFAULT_EXPANDED_V2 = (
    PROJECT_ROOT
    / "data"
    / "revision_bspc_2026"
    / "adni_expanded_v2"
    / "subject_metadata_adni_expanded_v2.csv"
)
DEFAULT_BATCH_DIR = PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_pilot" / "batches"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "philips_cn_stress_test_selection"
)

OUTPUT_ROOTS = [
    Path("/media/diego/Datos/adni_expansion/MARTIN59"),
    Path("/media/diego/Datos/adni_expansion/SIEMENS_available"),
    Path("/media/diego/Datos/adni_expansion/GE_smoketest3"),
    Path("/media/diego/Datos/adni_expansion/GE_batch7"),
]

REQUIRED_CANDIDATE_COLUMNS = [
    "SubjectID",
    "ImageID",
    "ResearchGroup",
    "Sex",
    "Age",
    "Manufacturer",
]
REQUIRED_METADATA_COLUMNS = [
    "SubjectID",
    "ResearchGroup_Mapped",
    "Manufacturer",
    "Site3",
]
PHILIPS_MANUFACTURERS = {"Philips Medical Systems", "Philips Healthcare", "Philips"}
SUBJECT_ID_RE = re.compile(r"\d{3}_S_\d{4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Philips-CN candidates for the BSPC 2026 stress test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--historical-metadata", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--expanded-v2-metadata", type=Path, default=DEFAULT_EXPANDED_V2)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=6)
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_subject_id(value: object) -> str:
    return str(value).strip()


def normalize_image_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.startswith("I") and text[1:].isdigit():
        text = text[1:]
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def site3_from_subject_id(subject_id: object) -> str:
    return normalize_site3(normalize_subject_id(subject_id).split("_", 1)[0])


def normalize_site3(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return str(int(text))
    return text


def normalize_manufacturer(value: object) -> str:
    text = str(value).strip()
    upper = text.upper()
    if "PHILIPS" in upper:
        return "Philips"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "GE" in upper:
        return "GE MEDICAL SYSTEMS"
    return text


def validate_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def load_candidates(path: Path) -> pd.DataFrame:
    ensure_exists(path, "candidate CSV")
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_CANDIDATE_COLUMNS, str(path))
    out = df.copy()
    out["SubjectID"] = out["SubjectID"].map(normalize_subject_id)
    out["ImageID"] = out["ImageID"].map(normalize_image_id)
    out["ResearchGroup"] = out["ResearchGroup"].astype(str).str.strip()
    out["Manufacturer"] = out["Manufacturer"].astype(str).str.strip()
    out["Manufacturer_norm"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Site3"] = out["SubjectID"].map(site3_from_subject_id)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Sex"] = out["Sex"].astype(str).str.strip()
    return out


def load_metadata(path: Path, label: str) -> pd.DataFrame:
    ensure_exists(path, label)
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_METADATA_COLUMNS, label)
    out = df.copy()
    out["SubjectID"] = out["SubjectID"].map(normalize_subject_id)
    out["ResearchGroup_Mapped"] = out["ResearchGroup_Mapped"].astype(str).str.strip().str.upper()
    out["Manufacturer_norm"] = out["Manufacturer"].map(normalize_manufacturer)
    out["Site3"] = out["Site3"].map(normalize_site3)
    return out


def read_subject_ids_from_npz(path: Path) -> Set[str]:
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "subject_ids" not in npz:
                return set()
            return {normalize_subject_id(x) for x in np.asarray(npz["subject_ids"]).reshape(-1)}
    except Exception:
        return set()


def collect_processed_npz_ids(roots: Sequence[Path]) -> Dict[str, Set[str]]:
    by_root: Dict[str, Set[str]] = {}
    for root in roots:
        ids: Set[str] = set()
        if root.exists():
            for npz_path in sorted(root.glob("**/GLOBAL_TENSOR*.npz")):
                ids.update(read_subject_ids_from_npz(npz_path))
        by_root[root.name] = ids
    return by_root


def collect_batch_membership(batch_dir: Path) -> Dict[str, List[str]]:
    membership: Dict[str, List[str]] = {}
    if not batch_dir.exists():
        return membership
    for path in sorted(batch_dir.glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        for sid in sorted(set(SUBJECT_ID_RE.findall(text))):
            membership.setdefault(sid, []).append(path.name)
    return membership


def build_philips_site_profile(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ph = df[df["Manufacturer_norm"] == "Philips"].copy()
    counts = (
        ph.groupby(["Site3", "ResearchGroup_Mapped"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    for diag in ["CN", "AD", "MCI"]:
        if diag not in counts.columns:
            counts[diag] = 0
    counts = counts[["CN", "AD", "MCI"]].reset_index()
    counts = counts.rename(
        columns={
            "CN": f"{prefix}_philips_CN",
            "AD": f"{prefix}_philips_AD",
            "MCI": f"{prefix}_philips_MCI",
        }
    )
    return counts


def rank_candidates(
    candidates: pd.DataFrame,
    historical: pd.DataFrame,
    expanded_v2: pd.DataFrame,
    batch_membership: Dict[str, List[str]],
) -> pd.DataFrame:
    hist_profile = build_philips_site_profile(historical, "historical")
    v2_profile = build_philips_site_profile(expanded_v2, "expanded_v2")

    ranked = candidates.merge(hist_profile, on="Site3", how="left")
    ranked = ranked.merge(v2_profile, on="Site3", how="left")
    count_cols = [
        "historical_philips_CN",
        "historical_philips_AD",
        "historical_philips_MCI",
        "expanded_v2_philips_CN",
        "expanded_v2_philips_AD",
        "expanded_v2_philips_MCI",
    ]
    for col in count_cols:
        ranked[col] = ranked[col].fillna(0).astype(int)

    tiers: List[int] = []
    scores: List[int] = []
    reasons: List[str] = []
    memberships: List[str] = []

    for row in ranked.itertuples(index=False):
        cn = int(row.expanded_v2_philips_CN)
        ad = int(row.expanded_v2_philips_AD)
        if ad >= 1 and cn >= 1:
            tier = 1
            reason = "Philips_site_has_AD_and_CN"
            score = 1000 + 10 * min(ad, 5) + min(cn, 5)
        elif ad >= 1:
            tier = 2
            reason = "Philips_site_has_AD_but_few_CN"
            score = 800 + 10 * min(ad, 5) - min(cn, 5)
        else:
            tier = 3
            reason = "Philips_CN_available"
            score = 500 + min(cn, 5)

        tiers.append(tier)
        scores.append(score)
        reasons.append(reason)
        memberships.append(";".join(batch_membership.get(row.SubjectID, [])))

    ranked["priority_tier"] = tiers
    ranked["priority_score"] = scores
    ranked["rank_reason"] = reasons
    ranked["existing_batch_membership"] = memberships

    ranked = ranked.sort_values(
        [
            "priority_tier",
            "priority_score",
            "expanded_v2_philips_AD",
            "expanded_v2_philips_CN",
            "Site3",
            "Age",
            "SubjectID",
        ],
        ascending=[True, False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def write_subject_list(path: Path, subject_ids: Iterable[str]) -> None:
    path.write_text("\n".join(subject_ids) + "\n", encoding="utf-8")


def make_readme(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    excluded_counts: Dict[str, int],
    output_paths: Dict[str, Path],
) -> str:
    top = ranked.head(10)
    top_lines = [
        f"- {r.SubjectID} | Site3={r.Site3} | {r.Manufacturer} | Age={r.Age} | "
        f"Sex={r.Sex} | ImageID={r.ImageID} | score={r.priority_score} | {r.rank_reason}"
        for r in top.itertuples(index=False)
    ]
    selected_lines = [f"- {sid}" for sid in selected["SubjectID"].astype(str).tolist()]
    exclusion_lines = [f"- {k}: {v}" for k, v in sorted(excluded_counts.items())]

    return "\n".join(
        [
            "# Philips-CN Stress Test Selection",
            "",
            f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
            "",
            "Purpose: select new Philips CN subjects to test whether the original model marks any new CN as AD-like, or whether the effect is concentrated in Siemens/GE.",
            "",
            "Ranking logic:",
            "- Tier 1: Philips sites already represented in expanded_v2 with both AD and CN.",
            "- Tier 2: Philips sites with AD but few/no CN.",
            "- Tier 3: any remaining Philips CN candidate.",
            "",
            "Exclusions:",
            *exclusion_lines,
            "",
            "Selected stress-test batch:",
            *(selected_lines if selected_lines else ["- No candidates available."]),
            "",
            "Top 10 ranked candidates:",
            *(top_lines if top_lines else ["- No candidates available."]),
            "",
            "Outputs:",
            *[f"- {name}: {path}" for name, path in output_paths.items()],
            "",
        ]
    )


def main() -> int:
    args = parse_args()

    candidates = load_candidates(args.candidates)
    historical = load_metadata(args.historical_metadata, "historical metadata")
    expanded_v2 = load_metadata(args.expanded_v2_metadata, "expanded_v2 metadata")

    historical_ids = set(historical["SubjectID"])
    expanded_v2_ids = set(expanded_v2["SubjectID"])
    processed_by_root = collect_processed_npz_ids(OUTPUT_ROOTS)
    processed_all = set().union(*processed_by_root.values()) if processed_by_root else set()
    batch_membership = collect_batch_membership(args.batch_dir)

    philips_cn = candidates[
        (candidates["ResearchGroup"].astype(str).str.upper() == "CN")
        & (candidates["Manufacturer"].isin(PHILIPS_MANUFACTURERS))
    ].copy()

    philips_cn["in_historical"] = philips_cn["SubjectID"].isin(historical_ids)
    philips_cn["in_expanded_v2"] = philips_cn["SubjectID"].isin(expanded_v2_ids)
    philips_cn["in_processed_npz_outputs"] = philips_cn["SubjectID"].isin(processed_all)
    for root_name, ids in processed_by_root.items():
        philips_cn[f"in_output_{root_name}"] = philips_cn["SubjectID"].isin(ids)

    available = philips_cn[
        ~philips_cn["in_historical"]
        & ~philips_cn["in_expanded_v2"]
        & ~philips_cn["in_processed_npz_outputs"]
    ].copy()

    ranked = rank_candidates(available, historical, expanded_v2, batch_membership)
    selected = ranked.head(args.top_n if len(ranked) >= args.top_n else len(ranked)).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.batch_dir.mkdir(parents=True, exist_ok=True)

    ranked_path = args.output_dir / "philips_cn_candidates_ranked.csv"
    top_path = args.batch_dir / "philips_cn_stress_test6.txt"
    all_path = args.batch_dir / "philips_cn_stress_test_all.txt"
    readme_path = args.output_dir / "README.md"
    manifest_path = args.output_dir / "selection_manifest.json"

    ranked.to_csv(ranked_path, index=False)
    write_subject_list(top_path, selected["SubjectID"].astype(str).tolist())
    write_subject_list(all_path, ranked["SubjectID"].astype(str).tolist())

    excluded_counts = {
        "raw_philips_cn_candidates": int(len(philips_cn)),
        "excluded_historical": int(philips_cn["in_historical"].sum()),
        "excluded_expanded_v2": int(philips_cn["in_expanded_v2"].sum()),
        "excluded_processed_npz_outputs": int(philips_cn["in_processed_npz_outputs"].sum()),
        "available_after_exclusions": int(len(ranked)),
    }
    output_paths = {
        "ranked_csv": ranked_path,
        "stress_test6": top_path,
        "stress_test_all": all_path,
        "readme": readme_path,
    }
    readme_path.write_text(make_readme(ranked, selected, excluded_counts, output_paths), encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "candidates": str(args.candidates),
            "historical_metadata": str(args.historical_metadata),
            "expanded_v2_metadata": str(args.expanded_v2_metadata),
            "batch_dir": str(args.batch_dir),
            "output_roots": [str(p) for p in OUTPUT_ROOTS],
        },
        "excluded_counts": excluded_counts,
        "processed_npz_output_counts": {k: len(v) for k, v in processed_by_root.items()},
        "selected_subjects": selected["SubjectID"].astype(str).tolist(),
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nPhilips-CN stress-test selection complete")
    print(f"Raw Philips-CN candidates: {len(philips_cn)}")
    print(f"Available after exclusions: {len(ranked)}")
    print(f"Selected for stress-test6: {len(selected)}")
    print(f"Ranked CSV: {ranked_path}")
    print(f"Batch file: {top_path}")
    print("\nTop 10 candidates:")
    display_cols = [
        "rank",
        "SubjectID",
        "Site3",
        "Manufacturer",
        "Age",
        "Sex",
        "ImageID",
        "priority_score",
        "rank_reason",
        "expanded_v2_philips_CN",
        "expanded_v2_philips_AD",
    ]
    if ranked.empty:
        print("No candidates available.")
    else:
        print(ranked.head(10)[display_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
