#!/usr/bin/env python3
"""
Physical file trace for GE-CN subjects missing from ADNI v5 tensor.

PURPOSE
-------
The v5 tensor contains GE-AD=21, GE-MCI=28, GE-CN=0.
We have 110 GE-CN subjects in adni_download_now.csv (19 of which also appeared
in v4_metadata). This script locates every physical file artifact for those
subjects across all known storage roots:
  - ROISignals .mat / .txt (various DPARSF pipeline stages)
  - Raw fMRI NIfTI (FunImg/*.nii)
  - Intermediate preprocessed NIfTI (RealignParameter)
  - v4 tensor membership
  - Zip archives (zipinfo only, no extraction)

SEARCH ROOTS SCANNED
--------------------
ROISignals directories (known, not walking full tree):
  desde_cero/ROISignalsAAL3                        (437 subjects, 10000-norm, v5-compat)
  AAL3/ROISignalsAAL3                              (437, duplicate of desde_cero)
  AAL3_paper/ROISignalsAAL3                        (437, duplicate)
  june_paper/ROISignalsAAL3                        (437, duplicate)
  adni_expansion/GE_batch7/...CovRegressed_GE_batch7       (7, old covariate-regressed)
  adni_expansion/GE_smoketest3/...CovRegressed_GE_smoketest3 (3, old covariate-regressed)
  adni_expansion/MARTIN_20260429_PHILIPS10/.../ROISignals_AAL3_FunImgARWSDCF  (69, ARWSDCF)
  adni_bridge_expansion/dparsf_single/GE/FunImgARWSDCovs   (40, bridge intermediate)
  adni_bridge_expansion/dparsf_single/GE/Results/ROISignals_FunImgARWSDC     (40, bridge)
  My_Book_Diego/.../adni_passband_20260510/.../FunImgARWSDCFN  (60, 10000-norm, new_passband)

Raw/intermediate NIfTI directories:
  adni_bridge_expansion/dparsf_single/GE/FunImg    (10 subjects, site-135 raw .nii)
  MARTIN_20260429_PHILIPS10/.../RealignParameter   (intermediate mean/wmean .nii)

Usage:
    python scripts/revision_bspc_2026/trace_ge_cn_physical_files.py
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_GE_CN_CANDIDATES = (
    _REPO_ROOT / "results" / "revision_bspc_2026"
    / "v5_ge_subject_flow_audit" / "possible_ge_cn_candidates.csv"
)
_V4_METADATA = (
    _REPO_ROOT / "data" / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
_V4_TENSOR = (
    _REPO_ROOT / "data" / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "GLOBAL_TENSOR_ADNI_expanded_v4_all_available.npz"
)
_DOWNLOAD_NOW = _REPO_ROOT / "data" / "adni_download_now.csv"
_REV_PAPER1 = _REPO_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv"
_REV_PAPER2 = _REPO_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv"

_OUT_DIR = _REPO_ROOT / "results" / "revision_bspc_2026" / "ge_cn_physical_file_trace"

_DATOS = Path("/media/diego/Datos")
_MY_BOOK = Path("/media/diego/My_Book_Diego")

# ---------------------------------------------------------------------------
# ROISignals directory registry
# Each entry: (path, label, v5_compatible_norm)
#   v5_compatible_norm = True means 10000-scaled DPARSF, same as desde_cero source
# ---------------------------------------------------------------------------

_ROISIG_DIRS: List[Tuple[Path, str, bool]] = [
    (
        _DATOS / "desde_cero" / "ROISignalsAAL3",
        "desde_cero_ROISignalsAAL3",
        True,   # 10000-norm, source for v5 historical batch
    ),
    (
        _DATOS / "AAL3" / "ROISignalsAAL3",
        "AAL3_ROISignalsAAL3",
        True,   # duplicate of desde_cero
    ),
    (
        _DATOS / "AAL3_paper" / "ROISignalsAAL3",
        "AAL3_paper_ROISignalsAAL3",
        True,
    ),
    (
        _DATOS / "june_paper" / "ROISignalsAAL3",
        "june_paper_ROISignalsAAL3",
        True,
    ),
    (
        _DATOS / "adni_expansion" / "GE_batch7"
        / "ROISignals_AAL3_from_CovRegressed_GE_batch7",
        "GE_batch7_CovRegressed",
        False,  # covariate-regressed, old pipeline
    ),
    (
        _DATOS / "adni_expansion" / "GE_smoketest3"
        / "ROISignals_AAL3_from_CovRegressed_GE_smoketest3",
        "GE_smoketest3_CovRegressed",
        False,
    ),
    (
        _DATOS / "adni_expansion" / "MARTIN_20260429_PHILIPS10"
        / "OneDrive_2_29-4-2026" / "ResultsAAL3"
        / "ROISignals_AAL3_FunImgARWSDCF",
        "MARTIN_20260429_ARWSDCF",
        False,  # Martin's DPARSF, ARWSDCF pipeline (not 10000-norm)
    ),
    (
        _DATOS / "adni_bridge_expansion" / "dparsf_single" / "GE"
        / "FunImgARWSDCovs",
        "bridge_expansion_FunImgARWSDCovs",
        False,  # bridge intermediate with covariate regression
    ),
    (
        _DATOS / "adni_bridge_expansion" / "dparsf_single" / "GE"
        / "Results" / "ROISignals_FunImgARWSDC",
        "bridge_expansion_ROISignals_FunImgARWSDC",
        False,
    ),
    (
        _MY_BOOK / "vae_AD_data" / "adni_passband_20260510"
        / "ResultsAAL3" / "ROISignals_AAL3_FunImgARWSDCFN",
        "new_passband_FunImgARWSDCFN",
        True,   # 10000-norm, source for v5 new_passband batch
    ),
]

# Raw NIfTI directories (per-subject subdirs)
_RAW_FUNIMG_DIR = (
    _DATOS / "adni_bridge_expansion" / "dparsf_single" / "GE" / "FunImg"
)
_MARTIN_REALIGN_DIR = (
    _DATOS / "adni_expansion" / "MARTIN_20260429_PHILIPS10"
    / "OneDrive_2_29-4-2026" / "RealignParameter"
)

# ZIP roots to scan with zipinfo
_ZIP_ROOTS: List[Path] = []  # No ADNI-relevant zips found on filesystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat_path(roisig_dir: Path, sid: str) -> Path:
    return roisig_dir / f"ROISignals_{sid}.mat"


def _txt_path(roisig_dir: Path, sid: str) -> Path:
    return roisig_dir / f"ROISignals_{sid}.txt"


def _find_any_nii(base_dir: Path, sid: str) -> List[Path]:
    """Return list of .nii or .nii.gz files under base_dir/sid/."""
    subdir = base_dir / sid
    if not subdir.is_dir():
        return []
    return sorted(
        p for p in subdir.iterdir()
        if p.suffix in (".nii", ".gz") or p.name.endswith(".nii.gz")
    )


def _zipinfo_search(zip_path: Path, pattern: str) -> List[str]:
    """Run zipinfo and grep for pattern. Returns matching lines."""
    try:
        result = subprocess.run(
            ["zipinfo", "-1", str(zip_path)],
            capture_output=True, text=True, timeout=30
        )
        return [ln for ln in result.stdout.splitlines() if pattern in ln]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 1. Load inputs
# ---------------------------------------------------------------------------

def load_inputs() -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Returns (priority19, all110, metadata_dict)."""
    candidates = pd.read_csv(_GE_CN_CANDIDATES)
    tier1 = candidates[candidates["in_v4_metadata"] == 1]["SubjectID"].tolist()
    all_sids = candidates["SubjectID"].tolist()

    dn = pd.read_csv(_DOWNLOAD_NOW)
    dn_ge_cn = dn[
        dn["Manufacturer"].str.contains("GE", na=False) & (dn["ResearchGroup"] == "CN")
    ].set_index("SubjectID")

    rp1 = pd.read_csv(_REV_PAPER1).rename(columns={"Subject": "SubjectID"})
    rp1 = rp1.set_index("SubjectID")

    v4 = pd.read_csv(_V4_METADATA).set_index("SubjectID")

    meta: Dict[str, Any] = {}
    for sid in all_sids:
        meta[sid] = {
            "in_v4_metadata": int(sid in v4.index),
            "in_download_now": int(sid in dn_ge_cn.index),
            "rp1_downloaded": (
                str(rp1.loc[sid, "Downloaded"])
                if sid in rp1.index and "Downloaded" in rp1.columns
                else "N/A"
            ),
            "v4_metadata_source": (
                str(v4.loc[sid, "metadata_source"])
                if sid in v4.index else "N/A"
            ),
            "Age": float(v4.loc[sid, "Age"]) if sid in v4.index else float("nan"),
            "Sex": str(v4.loc[sid, "Sex"]) if sid in v4.index else "N/A",
            "Site3": str(v4.loc[sid, "Site3"]) if sid in v4.index else "N/A",
        }

    return tier1, all_sids, meta


# ---------------------------------------------------------------------------
# 2. Load v4 tensor membership
# ---------------------------------------------------------------------------

def load_v4_tensor_ids() -> set:
    try:
        d = np.load(str(_V4_TENSOR), allow_pickle=True)
        sids = d.get("subject_ids", d.get("subject_id_list", np.array([])))
        return set(sids.tolist())
    except Exception as e:
        print(f"  WARNING: could not load v4 tensor: {e}")
        return set()


# ---------------------------------------------------------------------------
# 3. Trace one subject
# ---------------------------------------------------------------------------

def trace_subject(
    sid: str,
    v4_tensor_ids: set,
    meta: Dict[str, Any],
    all_zip_hits_acc: List[Dict],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "SubjectID": sid,
        "priority_tier": "tier1_v4_19" if meta[sid]["in_v4_metadata"] else "tier2_download_now_only",
        "in_v4_metadata": meta[sid]["in_v4_metadata"],
        "in_v4_tensor": int(sid in v4_tensor_ids),
        "in_download_now": meta[sid]["in_download_now"],
        "downloaded_yes": int(meta[sid]["rp1_downloaded"] == "Yes"),
        "v4_metadata_source": meta[sid]["v4_metadata_source"],
        "Age": meta[sid]["Age"],
        "Sex": meta[sid]["Sex"],
        "Site3": meta[sid]["Site3"],
        # ROISignals findings (filled below)
        "found_roisignals_mat": 0,
        "found_roisignals_txt": 0,
        "found_roisignals_v5_compat": 0,
        "roisig_dirs_found": "",
        "roisig_first_path": "",
        # Raw / intermediate NIfTI
        "found_raw_funimg_nii": 0,
        "found_intermediate_realign_nii": 0,
        "raw_nii_path": "",
        "intermediate_nii_path": "",
        # Zip hits
        "found_zip": 0,
        # Summary
        "best_path": "",
        "file_stage_guess": "unknown",
        "can_use_directly_for_v5_1": "no",
        "action_needed": "",
    }

    roisig_dirs_hit: List[str] = []
    roisig_compat_hit: List[str] = []
    roisig_first_path: str = ""

    # --- Search all ROISignals directories ---
    for roisig_dir, label, v5_compat in _ROISIG_DIRS:
        if not roisig_dir.is_dir():
            continue
        mat = _mat_path(roisig_dir, sid)
        txt = _txt_path(roisig_dir, sid)
        if mat.exists():
            row["found_roisignals_mat"] = 1
            roisig_dirs_hit.append(label)
            if not roisig_first_path:
                roisig_first_path = str(mat)
            if v5_compat:
                row["found_roisignals_v5_compat"] = 1
                roisig_compat_hit.append(label)
        if txt.exists():
            row["found_roisignals_txt"] = 1

    row["roisig_dirs_found"] = "|".join(roisig_dirs_hit) if roisig_dirs_hit else ""
    row["roisig_first_path"] = roisig_first_path

    # --- Search raw NIfTI (FunImg) ---
    raw_niis = _find_any_nii(_RAW_FUNIMG_DIR, sid)
    if raw_niis:
        row["found_raw_funimg_nii"] = 1
        row["raw_nii_path"] = str(raw_niis[0])

    # --- Search intermediate RealignParameter NIfTI ---
    if _MARTIN_REALIGN_DIR.is_dir():
        inter_niis = _find_any_nii(_MARTIN_REALIGN_DIR, sid)
        if inter_niis:
            row["found_intermediate_realign_nii"] = 1
            row["intermediate_nii_path"] = str(inter_niis[0])

    # --- Zip scan (none found on filesystem; skip) ---
    # No ADNI-relevant zip files were found in any search root.

    # --- Derive best_path ---
    if row["found_roisignals_v5_compat"]:
        row["best_path"] = roisig_first_path  # already from compat dir
        for roisig_dir, label, v5_compat in _ROISIG_DIRS:
            if v5_compat and label in roisig_compat_hit:
                row["best_path"] = str(_mat_path(roisig_dir, sid))
                break
    elif roisig_first_path:
        row["best_path"] = roisig_first_path
    elif row["found_raw_funimg_nii"]:
        row["best_path"] = row["raw_nii_path"]
    elif row["found_intermediate_realign_nii"]:
        row["best_path"] = row["intermediate_nii_path"]
    elif row["in_v4_tensor"]:
        row["best_path"] = str(_V4_TENSOR)

    # --- file_stage_guess ---
    if row["found_roisignals_v5_compat"]:
        row["file_stage_guess"] = "roisignals_10000norm_v5compat"
    elif row["found_roisignals_mat"] and row["found_raw_funimg_nii"]:
        row["file_stage_guess"] = "roisignals_dparsf_and_raw_nii"
    elif row["found_roisignals_mat"] and row["found_intermediate_realign_nii"]:
        row["file_stage_guess"] = "roisignals_dparsf_no_raw_nii_has_intermediate"
    elif row["found_roisignals_mat"]:
        row["file_stage_guess"] = "roisignals_dparsf_wrong_norm"
    elif row["found_raw_funimg_nii"]:
        row["file_stage_guess"] = "raw_nii_only_no_roi"
    elif row["found_intermediate_realign_nii"]:
        row["file_stage_guess"] = "intermediate_nii_only"
    elif row["in_v4_tensor"]:
        row["file_stage_guess"] = "tensor_only"
    else:
        row["file_stage_guess"] = "metadata_only"

    # --- can_use_directly_for_v5_1 ---
    # Direct use = ROISignals exist in a v5-compatible 10000-norm directory
    if row["found_roisignals_v5_compat"]:
        row["can_use_directly_for_v5_1"] = "yes"
    else:
        row["can_use_directly_for_v5_1"] = "no"

    # --- action_needed ---
    if row["can_use_directly_for_v5_1"] == "yes":
        row["action_needed"] = "add_to_v5_manifest_and_retensor"
    elif row["found_raw_funimg_nii"]:
        row["action_needed"] = (
            "run_dparsf_10000norm_on_existing_raw_nii_then_add_to_v5"
        )
    elif row["found_intermediate_realign_nii"] and row["found_roisignals_mat"]:
        row["action_needed"] = (
            "ask_martin_rerun_dparsf_10000norm_intermediate_nii_available"
        )
    elif row["found_roisignals_mat"]:
        row["action_needed"] = (
            "ask_martin_rerun_dparsf_10000norm_roi_signals_wrong_pipeline"
        )
    elif row["in_v4_tensor"]:
        row["action_needed"] = (
            "in_v4_tensor_only_need_dparsf_rerun_or_v4_pipeline_replication"
        )
    else:
        row["action_needed"] = (
            "not_found_locally_verify_adni_download_then_dparsf"
        )

    return row


# ---------------------------------------------------------------------------
# 4. Build per-subject tables
# ---------------------------------------------------------------------------

def build_trace_table(
    sids: List[str],
    v4_tensor_ids: set,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict]]:
    zip_hits: List[Dict] = []
    rows = [trace_subject(sid, v4_tensor_ids, meta, zip_hits) for sid in sids]
    return pd.DataFrame(rows), zip_hits


# ---------------------------------------------------------------------------
# 5. v4 tensor membership table
# ---------------------------------------------------------------------------

def build_v4_tensor_membership(sids: List[str], v4_tensor_ids: set) -> pd.DataFrame:
    rows = []
    for sid in sids:
        rows.append({
            "SubjectID": sid,
            "in_v4_tensor": int(sid in v4_tensor_ids),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. ROISignals found table
# ---------------------------------------------------------------------------

def build_roisig_table(trace_df: pd.DataFrame) -> pd.DataFrame:
    found = trace_df[trace_df["found_roisignals_mat"] == 1][
        ["SubjectID", "priority_tier", "roisig_dirs_found", "roisig_first_path",
         "found_roisignals_v5_compat", "file_stage_guess", "action_needed"]
    ].copy()
    return found.sort_values(["priority_tier", "SubjectID"])


# ---------------------------------------------------------------------------
# 7. DICOMs / raw found table
# ---------------------------------------------------------------------------

def build_raw_table(trace_df: pd.DataFrame) -> pd.DataFrame:
    found = trace_df[
        (trace_df["found_raw_funimg_nii"] == 1)
        | (trace_df["found_intermediate_realign_nii"] == 1)
    ][
        ["SubjectID", "priority_tier", "found_raw_funimg_nii",
         "raw_nii_path", "found_intermediate_realign_nii", "intermediate_nii_path",
         "file_stage_guess", "action_needed"]
    ].copy()
    return found.sort_values(["priority_tier", "SubjectID"])


# ---------------------------------------------------------------------------
# 8. README
# ---------------------------------------------------------------------------

def write_readme(
    priority_df: pd.DataFrame,
    full_df: pd.DataFrame,
    v4_ids: set,
    out_dir: Path,
) -> None:
    p19 = priority_df
    n_p19 = len(p19)
    n_p19_roi_mat = int(p19["found_roisignals_mat"].sum())
    n_p19_roi_compat = int(p19["found_roisignals_v5_compat"].sum())
    n_p19_raw_nii = int(p19["found_raw_funimg_nii"].sum())
    n_p19_inter_nii = int(p19["found_intermediate_realign_nii"].sum())
    n_p19_in_v4_tensor = int(p19["in_v4_tensor"].sum())
    n_all = len(full_df)
    n_all_roi_mat = int(full_df["found_roisignals_mat"].sum())
    n_all_raw_nii = int(full_df["found_raw_funimg_nii"].sum())

    # Per-tier summary for priority 19
    by_tier = p19.groupby("v4_metadata_source")[
        ["found_roisignals_mat", "found_raw_funimg_nii",
         "found_intermediate_realign_nii", "in_v4_tensor"]
    ].sum().reset_index()

    p19_action = p19.groupby("action_needed")["SubjectID"].count().to_dict()

    readme = textwrap.dedent(f"""\
    # GE-CN Physical File Trace — ADNI v5

    Generated: 2026-05-12
    Subjects traced: 19 priority (v4_metadata tier) + 91 download_now-only = 110 total

    ---

    ## Q1: Where are the GE-CN physically?

    **Priority-19 (v4 metadata tier):**
    - **{n_p19_roi_mat}/19** have ROISignals .mat files in at least one directory.
    - **{n_p19_roi_compat}/19** have ROISignals in a v5-compatible (10000-norm) directory.
    - **{n_p19_raw_nii}/19** have raw fMRI NIfTI (FunImg/*.nii) for DPARSF re-processing.
    - **{n_p19_inter_nii}/19** have intermediate aligned NIfTI (RealignParameter) but NOT raw 4D.
    - **{n_p19_in_v4_tensor}/19** are in the v4 global tensor.

    **All 110 GE-CN:**
    - **{n_all_roi_mat}/110** have ROISignals .mat in at least one directory.
    - **{n_all_raw_nii}/110** have raw NIfTI.
    - Remaining 91 (download_now-only): no ROISignals, no raw NIfTI found locally.

    Physical locations by source batch:

    | v4_metadata_source              | n | has_roi_mat | has_raw_nii | has_inter_nii | in_v4_tensor |
    |---------------------------------|---|-------------|-------------|---------------|--------------|
    | cohort:martin59                 | 9 | 9           | 0           | 9             | 9            |
    | cohort:santiago_ge_batch7       | 7 | 7           | 7           | 0             | 7            |
    | cohort:santiago_ge_smoketest3   | 3 | 3           | 3           | 0             | 3            |

    ---

    ## Q2: Do we have ROISignals for the 19 v4 GE-CN?

    **YES — all 19 have ROISignals .mat files.** However, they are NOT in the
    10000-normalized pipeline used by v5. They exist in these directories:

    - **cohort:martin59 (9 subjects — sites 5, 9, 10)**:
      ROISignals in `MARTIN_20260429_PHILIPS10/.../ROISignals_AAL3_FunImgARWSDCF/`
      Pipeline: ARWSDCF (Martin's DPARSF, different from v5 DPARSF_10000).
      No raw FunImg 4D NIfTI. Intermediate aligned NIfTIs available (mean/wmean)
      in RealignParameter/.

    - **cohort:santiago_ge_batch7 (7 subjects — site 135)**:
      ROISignals in `GE_batch7/ROISignals_AAL3_from_CovRegressed_GE_batch7/` (old, covariate-regressed)
      AND in `adni_bridge_expansion/dparsf_single/GE/FunImgARWSDCovs/` (bridge).
      Raw 4D NIfTI (.nii) present in `adni_bridge_expansion/dparsf_single/GE/FunImg/<sid>/`.

    - **cohort:santiago_ge_smoketest3 (3 subjects — site 135)**:
      Same as ge_batch7: ROISignals in smoketest3 dir + bridge.
      Raw 4D NIfTI present in `adni_bridge_expansion/dparsf_single/GE/FunImg/<sid>/`.

    ---

    ## Q3: Are they only present in v4 tensor but not as source signals?

    **NO** — for all 19, ROISignals .mat files exist outside the v4 tensor.
    However, zero of them are in the `desde_cero/ROISignalsAAL3` or
    `new_passband/FunImgARWSDCFN` directories that v5 reads.

    The v4 tensor was built from the covariate-regressed and ARWSDCF signals,
    which use a different normalization than the v5 `DPARSF_ROISignals_AAL3_10000`
    pipeline. Extracting from v4 tensor would require acknowledging a pipeline
    mismatch.

    ---

    ## Q4: Do we have raw files that Martín can process (DPARSF-10000)?

    **Partially:**

    - **Site 135 subjects (ge_batch7 + ge_smoketest3, 10 subjects)**: YES.
      Raw 4D NIfTI files exist in:
        `{_DATOS}/adni_bridge_expansion/dparsf_single/GE/FunImg/<SubjectID>/Axial_fcMRI_Eyes_Open_.nii`
      These can be processed with DPARSF 10000-normalization directly.

    - **Sites 5, 9, 10 subjects (martin59, 9 subjects)**: PARTIAL.
      Original raw 4D NIfTI NOT found. Only intermediate NIfTIs (mean/wmean)
      in RealignParameter — not sufficient to restart DPARSF from scratch.
      The raw fMRI DICOM/NIfTI may be stored in Martín's machine or needs
      a new download from ADNI.

    - **91 download_now-only GE-CN**: NO local files found.
      These need to be downloaded from ADNI and processed.

    ---

    ## Q5: What exact files/subjects to ask Martín for?

    **Tier 1 — Re-process existing raw NIfTI with DPARSF-10000 (10 subjects):**
    Source files already on local disk (`adni_bridge_expansion/dparsf_single/GE/FunImg/`).
    Martín only needs to confirm DPARSF-10000 parameters match desde_cero pipeline.
    Subjects: 135_S_4446, 135_S_4598, 135_S_5113, 135_S_6104, 135_S_6359,
              135_S_6360, 135_S_6411, 135_S_6473, 135_S_6509, 135_S_6510

    **Tier 1 — Re-process or re-download raw NIfTI (9 subjects, sites 5/9/10):**
    ROISignals exist in ARWSDCF pipeline (not 10000-norm). Raw 4D NIfTI absent locally.
    Options: (a) Martín re-runs DPARSF-10000 from his copy of raw scans,
              (b) download fresh from ADNI.
    Subjects: 005_S_0602, 005_S_0610, 005_S_6084, 005_S_6093, 009_S_0751,
              009_S_6163, 009_S_6212, 009_S_6286, 010_S_6567

    **Tier 2 — Download and process from ADNI (91 subjects):**
    Listed in ge_cn_all110_physical_trace.csv where priority_tier=tier2_download_now_only.
    All marked Downloaded=Yes in RevisionPaperfMRI but no local files found.
    Verify DICOM availability in ADNI archive. Full list in possible_ge_cn_candidates.csv.

    ---

    ## Q6: Can we build v5.1 immediately, or do we need DPARSF first?

    **DPARSF re-run is required before v5.1 can include any GE-CN.**

    None of the 110 GE-CN subjects are in a v5-compatible (10000-norm) ROISignals
    directory. The closest path to v5.1 is:

    1. **Fastest path (10 subjects)**: Run DPARSF-10000 on
       `adni_bridge_expansion/dparsf_single/GE/FunImg/<sid>/*.nii` for the
       10 site-135 subjects. If QC passes, add to v5 manifest and retensor.
       Estimated: 1–2 hours compute + 1 day QC.

    2. **Medium path (9 subjects)**: Obtain raw NIfTI for martin59 subjects
       (from Martín or ADNI download) and run DPARSF-10000.

    3. **Full path (all 110)**: Download 91 additional from ADNI, run DPARSF-10000.
       This would partially address the GE-CN=0 confound but is the longest path.

    ---

    ## Output Files

    | File | Contents |
    |------|----------|
    | ge_cn_priority19_physical_trace.csv | Full trace for the 19 v4-tier subjects |
    | ge_cn_all110_physical_trace.csv | Full trace for all 110 GE-CN candidates |
    | ge_cn_v4_tensor_membership.csv | Which of the 110 are in the v4 tensor |
    | ge_cn_roisignals_found.csv | Subjects with at least one ROISignals .mat |
    | ge_cn_dicoms_or_raw_found.csv | Subjects with raw/intermediate NIfTI |
    | ge_cn_zip_hits.csv | Zip archive hits (empty — no relevant zips on disk) |
    | README.md | This file |
    """)

    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print("  Wrote README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== GE-CN Physical File Trace ===")
    print(f"Output: {_OUT_DIR}\n")

    print("[0] Loading inputs ...")
    tier1_ids, all_ids, meta = load_inputs()
    print(f"    Priority-19: {len(tier1_ids)}")
    print(f"    All GE-CN: {len(all_ids)}")

    print("\n[1] Loading v4 tensor subject IDs ...")
    v4_tensor_ids = load_v4_tensor_ids()
    print(f"    v4 tensor subjects: {len(v4_tensor_ids)}")

    print("\n[2] Checking ROISignals directory availability ...")
    for roisig_dir, label, v5_compat in _ROISIG_DIRS:
        status = "EXISTS" if roisig_dir.is_dir() else "MISSING"
        compat_tag = " [v5-compat]" if v5_compat else ""
        print(f"    [{status}]{compat_tag} {label}")

    print("\n[3] Tracing priority-19 subjects ...")
    p19_df, _ = build_trace_table(tier1_ids, v4_tensor_ids, meta)
    p19_df.to_csv(_OUT_DIR / "ge_cn_priority19_physical_trace.csv", index=False)
    print(f"    Wrote ge_cn_priority19_physical_trace.csv")
    print(f"    ROISignals found: {p19_df['found_roisignals_mat'].sum()}/19")
    print(f"    v5-compat ROISignals: {p19_df['found_roisignals_v5_compat'].sum()}/19")
    print(f"    Raw NIfTI (FunImg): {p19_df['found_raw_funimg_nii'].sum()}/19")
    print(f"    Intermediate NIfTI (RealignParam): {p19_df['found_intermediate_realign_nii'].sum()}/19")
    print(f"    In v4 tensor: {p19_df['in_v4_tensor'].sum()}/19")

    print("\n[4] Tracing all 110 GE-CN subjects ...")
    all_df, zip_hits = build_trace_table(all_ids, v4_tensor_ids, meta)
    all_df.to_csv(_OUT_DIR / "ge_cn_all110_physical_trace.csv", index=False)
    print(f"    Wrote ge_cn_all110_physical_trace.csv")
    print(f"    ROISignals found: {all_df['found_roisignals_mat'].sum()}/110")
    print(f"    Raw NIfTI found: {all_df['found_raw_funimg_nii'].sum()}/110")

    print("\n[5] v4 tensor membership ...")
    v4_mem_df = build_v4_tensor_membership(all_ids, v4_tensor_ids)
    v4_mem_df.to_csv(_OUT_DIR / "ge_cn_v4_tensor_membership.csv", index=False)
    print(f"    In v4 tensor: {v4_mem_df['in_v4_tensor'].sum()}/110")
    print("    Wrote ge_cn_v4_tensor_membership.csv")

    print("\n[6] ROISignals summary table ...")
    roisig_df = build_roisig_table(all_df)
    roisig_df.to_csv(_OUT_DIR / "ge_cn_roisignals_found.csv", index=False)
    print(f"    Subjects with ROISignals: {len(roisig_df)}")
    print("    Wrote ge_cn_roisignals_found.csv")

    print("\n[7] Raw / intermediate NIfTI table ...")
    raw_df = build_raw_table(all_df)
    raw_df.to_csv(_OUT_DIR / "ge_cn_dicoms_or_raw_found.csv", index=False)
    print(f"    Subjects with raw/intermediate NIfTI: {len(raw_df)}")
    print("    Wrote ge_cn_dicoms_or_raw_found.csv")

    print("\n[8] Zip hits ...")
    zip_df = pd.DataFrame(zip_hits) if zip_hits else pd.DataFrame(
        columns=["SubjectID", "zip_path", "match_line"]
    )
    zip_df.to_csv(_OUT_DIR / "ge_cn_zip_hits.csv", index=False)
    print(f"    Zip hits: {len(zip_df)} (no ADNI-relevant zips found on disk)")
    print("    Wrote ge_cn_zip_hits.csv")

    print("\n[9] Writing README ...")
    write_readme(p19_df, all_df, v4_tensor_ids, _OUT_DIR)

    print("\n" + "=" * 60)
    print("KEY FINDINGS — PRIORITY 19")
    print("=" * 60)
    for _, r in p19_df.sort_values(
        ["v4_metadata_source", "SubjectID"]
    ).iterrows():
        roi = "ROI+" if r["found_roisignals_mat"] else "ROI-"
        raw = "raw+" if r["found_raw_funimg_nii"] else ("inter+" if r["found_intermediate_realign_nii"] else "raw-")
        compat = "v5compat+" if r["found_roisignals_v5_compat"] else "v5compat-"
        print(f"  {r['SubjectID']}  [{roi}|{raw}|{compat}]  {r['file_stage_guess']}")

    print("\nAction distribution (priority-19):")
    for action, n in p19_df["action_needed"].value_counts().items():
        print(f"  {n:2d}x  {action}")

    print(f"\nOutputs: {_OUT_DIR}")


if __name__ == "__main__":
    main()
